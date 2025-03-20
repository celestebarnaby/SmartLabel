import heapq as hq
import itertools
import random

from synthesis import Synthesizer
from constants import *

from image_search_domain.image_search_utils import *
from image_search_domain.image_search_interpreter import ImageSearchInterpreter

class ImageSearchSynthesizer(Synthesizer):
    def __init__(self, semantics):
        super().__init__(semantics)
        self.worklist = []
        self.interp = ImageSearchInterpreter()
        self.program_counter = itertools.count(0)
        self.output_dict = {} 

    def set_object_list(self, input_space):
        labels_to_occs = {}
        for abs_img in input_space.values():
            labels_in_img = {abs_img['conf'][obj_id]["Label"] for obj_id in abs_img['conf']}
            for label in labels_in_img:
                if label not in labels_to_occs:
                    labels_to_occs[label] = 0 
                labels_to_occs[label] += 1 
        sorted_labels_to_occs = sorted(list(labels_to_occs.items()), key=lambda x: x[1], reverse=True)
        # object must appear in at least a certain percent of images -- i.e. remove rare objects
        self.all_objects = [label for label, occs in sorted_labels_to_occs if occs >= max(len(input_space) * .05, 3)]


    def check_programs(self, program_space, examples):
        progs_matching_examples = []
        check = self.interp.get_check(self.semantics)
        for prog in program_space:
            if check(prog, examples):
                progs_matching_examples.append(prog)
        return progs_matching_examples


    def synthesize(self, examples, learnsy=False):
        '''
        Given a set of I/O examples, perform top-down enumeration up to a set AST size, and return all programs matching 
        the I/O examples
        '''

        self.worklist = []
        self.output_dict = {}

        output_per_example = [output for _, output in examples]
        self.output_dict[str(output_per_example)] = output_per_example

        progs_matching_examples = []
        seen_progs = set()
        
        # The function that we use to check whether an enumerated program matches the I/O examples (varies depending on semantics)
        check = self.interp.get_check(self.semantics)
        
        # The function that we use to perform equivalence reduction on enumerated programs (varies depending on semantics)
        simplify = get_simplify(self.semantics)

        # Create an initial partial program consistent of just a hole
        tree = Tree(next(self.program_counter))
        tree.nodes[0] = Hole(0, "expr", str(output_per_example), str(output_per_example))
        tree.var_nodes.append(0)
        hq.heappush(self.worklist, tree)

        num_progs = 0
        while self.worklist:
            num_progs += 1
            cur_tree = hq.heappop(self.worklist)
            prog = construct_prog_from_tree(cur_tree)
            if get_ast_size(prog) > (LEARNSY_IMAGE_EDIT_AST_SIZE if learnsy else IMAGE_SEARCH_AST_SIZE):
                break

            # A pruning technique for the image editing DSL that utilizes equivalence reduction
            # to prune programs that are equivalent to a previously enumerated program
            if not isinstance(prog, Hole):
                simplified_prog = simplify(prog.duplicate())
                if simplified_prog is None or str(simplified_prog) in seen_progs:
                    continue 
                seen_progs.add(str(simplified_prog))

            # If the program is complete, check whether it matches examples
            if not cur_tree.var_nodes:
                if check(prog, examples):
                    progs_matching_examples.append(prog)
                continue 

            hole_num = cur_tree.var_nodes.pop(0)
            hole = cur_tree.nodes[hole_num]
            node_type = cur_tree.nodes[hole_num].node_type 
            parent_node = None if hole_num == 0 else cur_tree.nodes[cur_tree.to_parent[hole_num]]
            if node_type == "expr":
                new_sub_exprs = get_expressions(
                    parent_node,
                    self.output_dict[hole.output_over],
                    self.output_dict[hole.output_under],
                    examples,
                    self.semantics,
                )
            elif node_type == "attr":
                new_sub_exprs = get_attributes(
                    self.output_dict[hole.output_over], self.output_dict[hole.output_under]
                )
            elif node_type == "obj":
                new_sub_exprs = [
                    (obj, [], [], [], 0)
                    for obj in self.all_objects 
                ]
            else:
                # TODO: error handling
                raise TypeError
            
            if not learnsy:
                new_sub_exprs = random.sample(new_sub_exprs, min(len(new_sub_exprs), IMAGE_SEARCH_NUM_SAMPLED_SUB_EXPRS))

            for (
                sub_expr,
                children,
                child_outputs_over,
                child_outputs_under,
                size,
            ) in new_sub_exprs:
                if isinstance(sub_expr, Node):
                    sub_expr.output_over = str(hole.output_over)
                    sub_expr.output_under = str(hole.output_under)

                prog_output_per_example = self.interp.get_prog_output(sub_expr, examples, parent_node, self.semantics)

                # A pruning technique that checks whether a subtree is able to produce a goal output
                # derived from the example output
                if prog_output_per_example and invalid_output(
                    self.output_dict[hole.output_over],
                    self.output_dict[hole.output_under],
                    prog_output_per_example                        
                ):
                    continue 
                new_tree = cur_tree.duplicate(next(self.program_counter))
                new_tree.nodes[hole_num] = sub_expr
                new_tree.size += size
                for child, child_output_over, child_output_under in zip(
                    children, child_outputs_over, child_outputs_under
                ):  
                    over_str = str(child_output_over)
                    under_str = str(child_output_under)
                    if over_str not in self.output_dict:
                        self.output_dict[over_str] = child_output_over
                    if under_str not in self.output_dict:
                        self.output_dict[under_str] = child_output_under
                    new_hole = Hole(hole.depth + 1, child, over_str, under_str)
                    new_tree.depth = max(new_tree.depth, new_hole.depth)
                    new_node_num = len(new_tree.nodes)
                    new_tree.nodes[new_node_num] = new_hole
                    new_tree.var_nodes.append(new_node_num)
                    new_tree.to_parent[new_node_num] = hole_num
                    if hole_num in new_tree.to_children:
                        new_tree.to_children[hole_num].append(new_node_num)
                    else:
                        new_tree.to_children[hole_num] = [new_node_num]
                hq.heappush(self.worklist, new_tree)
        return progs_matching_examples


def simplify_abs(prog):
    if isinstance(prog, Union) or isinstance(prog, Intersection):
        for sub_expr in prog.expressions:
            prog.expressions = [
                simplify_abs(sub_expr)
                for sub_expr in prog.expressions
            ]
            if None in prog.expressions:
                return None
    elif isinstance(prog, Complement) or isinstance(prog, Map):
        prog.expression = simplify_abs(prog.expression)
        if prog.expression is None:
            return None
    pos_to_inverse = {
        "GetLeft": GetRight(),
        "GetRight": GetLeft(),
        "GetAbove": GetBelow(),
        "GetBelow": GetAbove(),
        "GetContains": GetIsContained(),
        "GetIsContained": GetContains(),
    }

    new_prog = None
    while True:
        changed = True
        if isinstance(prog, Union):
            if len(prog.expressions) == 1:
                new_prog = prog.expressions[0]
            else:
                new_sub_exprs = []
                for i, sub_expr in enumerate(prog.expressions):
                    if (
                        isinstance(sub_expr, Union)
                        or isinstance(sub_expr, Intersection)
                        and not sub_expr.expressions
                    ):
                        continue
                    # Domination
                    should_keep = True
                    for j, other_sub_expr in enumerate(prog.expressions):
                        # Idempotency
                        if sub_expr == other_sub_expr and i < j:
                            should_keep = False
                            break
                        # Absorption
                        if (
                            isinstance(sub_expr, Intersection)
                            and other_sub_expr in sub_expr.expressions
                        ):
                            should_keep = False
                            break
                    if should_keep:
                        new_sub_exprs.append(sub_expr)
                new_sub_exprs.sort()
                if new_sub_exprs == prog.expressions:
                    changed = False
                if len(new_sub_exprs) < 2:
                    return None
                new_prog = Union(new_sub_exprs)

        elif isinstance(prog, Intersection):
            if len(prog.expressions) == 1:
                new_prog = prog.expressions[0]
            else:
                new_sub_exprs = []
                for i, sub_expr in enumerate(prog.expressions):
                    should_keep = True
                    # Identity
                    if (
                        isinstance(sub_expr, Union)
                        or isinstance(sub_expr, Intersection)
                        and not sub_expr.expressions
                    ):
                        continue
                    # Domination
                    for j, other_sub_expr in enumerate(prog.expressions):
                        # Idempotency
                        if sub_expr == other_sub_expr and i < j:
                            should_keep = False
                            break
                        # Absorption
                        if (
                            isinstance(sub_expr, Union)
                            and other_sub_expr in sub_expr.expressions
                        ):
                            should_keep = False
                            break
                    if should_keep:
                        new_sub_exprs.append(sub_expr)
                new_sub_exprs.sort()
                if new_sub_exprs == prog.expressions:
                    changed = False
                if len(new_sub_exprs) < 2:
                    return None
                new_prog = Intersection(new_sub_exprs)

        # Double negation
        elif isinstance(prog, Complement) and isinstance(prog.expression, Complement):
            return None

        # Map inverse
        elif (
            isinstance(prog, Map)
            and isinstance(prog.expression, Map)
            and str(prog.position) in pos_to_inverse
            and pos_to_inverse[str(prog.position)] == prog.expression.position
        ):
            new_prog = prog.expression.expression

        elif isinstance(prog, Complement) and isinstance(prog.expression, Intersection):
            new_sub_exprs = [
                Complement(sub_expr) for sub_expr in prog.expression.expressions
            ]
            new_prog = Union(new_sub_exprs)
        elif isinstance(prog, Complement) and isinstance(prog.expression, Union):
            new_sub_exprs = [
                Complement(sub_expr) for sub_expr in prog.expression.expressions
            ]
            new_prog = Intersection(new_sub_exprs)
        else:
            new_prog = prog
            changed = False

        if not changed:
            break
        prog = new_prog
    return prog


def get_simplify(semantics):
    semantics_to_simplify_fn = {
        "CCE-NoAbs": simplify_abs,
        "CCE": simplify_abs,
        "standard": simplify_standard
    }
    return semantics_to_simplify_fn[semantics]


def simplify_standard(prog):
    if isinstance(prog, Union) or isinstance(prog, Intersection):
        for sub_expr in prog.expressions:
            prog.expressions = [
                simplify_standard(sub_expr)
                for sub_expr in prog.expressions
            ]
            if None in prog.expressions:
                return None
    elif isinstance(prog, Complement) or isinstance(prog, Map):
        prog.expression = simplify_standard(prog.expression)
        if prog.expression is None:
            return None
    pos_to_inverse = {
        "GetLeft": GetRight(),
        "GetRight": GetLeft(),
        "GetAbove": GetBelow(),
        "GetBelow": GetAbove(),
        "GetContains": GetIsContained(),
        "GetIsContained": GetContains(),
    }

    new_prog = None
    while True:
        changed = True
        if isinstance(prog, Union):
            if len(prog.expressions) == 1:
                new_prog = prog.expressions[0]
            else:
                new_sub_exprs = []
                for i, sub_expr in enumerate(prog.expressions):
                    if (
                        isinstance(sub_expr, Union)
                        or isinstance(sub_expr, Intersection)
                        and not sub_expr.expressions
                    ):
                        continue
                    should_keep = True
                    for j, other_sub_expr in enumerate(prog.expressions):
                        # Idempotency
                        if sub_expr == other_sub_expr and i < j:
                            should_keep = False
                            break
                        # Absorption
                        if (
                            isinstance(sub_expr, Intersection)
                            and other_sub_expr in sub_expr.expressions
                        ):
                            should_keep = False
                            break
                    if should_keep:
                        new_sub_exprs.append(sub_expr)
                new_sub_exprs.sort()
                if new_sub_exprs == prog.expressions:
                    changed = False
                if len(new_sub_exprs) < 2:
                    return None
                new_prog = Union(new_sub_exprs)

        elif isinstance(prog, Intersection):
            if len(prog.expressions) == 1:
                new_prog = prog.expressions[0]
            else:
                new_sub_exprs = []
                for i, sub_expr in enumerate(prog.expressions):
                    should_keep = True
                    # Identity
                    if (
                        isinstance(sub_expr, Union)
                        or isinstance(sub_expr, Intersection)
                        and not sub_expr.expressions
                    ):
                        continue
                    # Domination
                    for j, other_sub_expr in enumerate(prog.expressions):
                        # Idempotency
                        if sub_expr == other_sub_expr and i < j:
                            should_keep = False
                            break
                        # Absorption
                        if (
                            isinstance(sub_expr, Union)
                            and other_sub_expr in sub_expr.expressions
                        ):
                            should_keep = False
                            break
                    if should_keep:
                        new_sub_exprs.append(sub_expr)
                new_sub_exprs.sort()
                if new_sub_exprs == prog.expressions:
                    changed = False
                if len(new_sub_exprs) < 2:
                    return None
                new_prog = Intersection(new_sub_exprs)

        # Double negation
        elif isinstance(prog, Complement) and isinstance(prog.expression, Complement):
            return None

        # Map inverse
        elif (
            isinstance(prog, Map)
            and isinstance(prog.expression, Map)
            and str(prog.position) in pos_to_inverse
            and pos_to_inverse[str(prog.position)] == prog.expression.position
        ):
            new_prog = prog.expression.expression

        elif isinstance(prog, Complement) and isinstance(prog.expression, Intersection):
            new_sub_exprs = [
                Complement(sub_expr) for sub_expr in prog.expression.expressions
            ]
            new_prog = Union(new_sub_exprs)
        elif isinstance(prog, Complement) and isinstance(prog.expression, Union):
            new_sub_exprs = [
                Complement(sub_expr) for sub_expr in prog.expression.expressions
            ]
            new_prog = Intersection(new_sub_exprs)
        else:
            new_prog = prog
            changed = False

        if not changed:
            break
        prog = new_prog
    return prog



