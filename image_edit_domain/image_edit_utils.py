from image_edit_domain.image_edit_dsl import *
import copy
from constants import *

def get_attributes(output_over_per_example, output_under_per_example):
    attrs = [
        (IsSmiling(), [], [], [], 0),
        (EyesOpen(), [], [], [], 0),
        (MouthOpen(), [], [], [], 0),
        (IsObject(None), ["obj"], [output_over_per_example], [output_under_per_example], 0),
    ]
    # if dataset == "receipts":
    #     attrs += [
    #         (IsPrice(), [], [], [], 0),
    #         (IsPhoneNumber(), [], [], [], 0),
    #         (MatchesWord(None), ["word"], [output_over_per_example], [output_under_per_example], 0),
    #     ]
    return attrs

def get_expressions(
    parent_expr: Expression, output_over_per_example, output_under_per_example, examples, semantics
) -> List[Expression]:
    exprs = get_attributes(output_over_per_example, output_under_per_example)
    exprs += (
        (
            Complement(None),
            ["expr"],
            [[get_abs_img_overapprox(examples[i][0], semantics) - output_under for i, output_under in enumerate(output_under_per_example)]],
            [[get_abs_img_underapprox(examples[i][0], semantics) - output_over for i, output_over in enumerate(output_over_per_example)]],
            1,
        ),
    )
    if not isinstance(parent_expr, Union):
        for i in range(2, 4):
            exprs.append(
                (
                    Union([None] * i), 
                    ["expr"] * i,
                    [output_over_per_example] * i, 
                    [[set()] * len(examples)] * i, 
                    i
                 )
            ),
    if not isinstance(parent_expr, Intersection):
        for i in range(2, 4):
            exprs.append(
                (
                    Intersection([None] * i),
                    ["expr"] * i,
                    [[get_abs_img_overapprox(abs_img, semantics) for abs_img, _ in examples]] * i,
                    [output_under_per_example] * i,
                    i,
                )
            ),
    map_weight = 3 if isinstance(parent_expr, Map) else 2
    exprs += [
        (
            Map(None, None, pos),
            ["expr", "attr"],
            [[get_abs_img_overapprox(abs_img, semantics) for abs_img, _ in examples], [get_abs_img_overapprox(abs_img, semantics) for abs_img, _ in examples]],
            [[set()] * len(examples), output_under_per_example],
            map_weight,
        )
        for pos in get_positions()
    ]
    return exprs

def get_positions() -> List[Position]:
    return [
        GetLeft(),
        GetRight(),
        GetAbove(),
        GetBelow(),
        GetContains(),
        GetIsContained(),
    ]

def get_abs_img_overapprox(abs_img, semantics):
    '''
    Returns all objects that are DEFINITELY in the image (i.e. there is no uncertainty in the conformal prediction)
    '''
    if semantics == "standard":
        return set(abs_img[semantics].keys())
    return set(abs_img['conf'].keys())


def get_abs_img_underapprox(abs_img, semantics):
    '''
    Returns all objects that MIGHT be in the image (i.e. there IS uncertainty in the conformal prediction)
    '''
    if semantics == "standard":
        return set(abs_img[semantics].keys())
    return set([obj_id for obj_id, obj in abs_img["conf"].items() if obj["Flag"]])


def get_ast_depth(prog):
    if isinstance(prog, Union) or isinstance(prog, Intersection):
        return max([get_ast_depth(extr) for extr in prog.expressions]) + 1
    if isinstance(prog, Complement) or isinstance(prog, Map):
        return get_ast_depth(prog.expression) + 1
    else:
        return 1


def get_ast_size(prog, map_parent=False):
    if isinstance(prog, Union) or isinstance(prog, Intersection):
        return sum([get_ast_size(extr) for extr in prog.expressions]) + 1
    elif isinstance(prog, Complement):
        return get_ast_size(prog.expression) + 1
    elif isinstance(prog, Map):
        return (
            get_ast_size(prog.expression, True)
            + get_ast_size(prog.restriction, True)
            + 1
            + (1 if map_parent else 0)
        )
    else:
        return 1
    

def invalid_output(output_over_per_example, output_under_per_example, prog_output_per_example):
    '''
    Determines whether a partial program is invalid (because no completion could match the over-/under-approximated goal output).
    Used for pruning during synthesis.
    '''
    for goal_over, goal_under, (prog_under, prog_over) in zip(output_over_per_example, output_under_per_example, prog_output_per_example):
        if (
            goal_under.difference(prog_over) or prog_under.difference(goal_over)
        ):
            return True 
    return False


def construct_prog_from_tree(tree, node_num=0, should_copy=False):
    if should_copy:
        prog = copy.copy(tree.nodes[node_num])
    else:
        prog = tree.nodes[node_num]
    if not isinstance(prog, Node):
        return prog
    prog_dict = vars(prog)
    if node_num in tree.to_children:
        child_nums = tree.to_children[node_num]
    else:
        child_nums = []
    # Ignore values that are not relevant to the AST
    child_types = [
        item for item in list(prog_dict) if item not in {"position", "val", "output_over", "output_under", "abs_value", "formula"}
    ]
    if child_types and child_types[0] == "expressions":
        for child_num in child_nums:
            prog_dict["expressions"].pop(0)
            child_prog = construct_prog_from_tree(tree, child_num)
            prog_dict["expressions"].append(child_prog)
        return prog
    assert len(child_nums) == len(child_types)
    for child_type, child_num in zip(child_types, child_nums):
        child_prog = construct_prog_from_tree(tree, child_num)
        prog_dict[child_type] = child_prog
    return prog


def update_abs_output(expression, new_under, new_over):
    expression.abs_value = (expression.abs_value[0].union(new_under), expression.abs_value[1].intersection(new_over))
    if not expression.abs_value[0].issubset(expression.abs_value[1]):
        return False 
    return True


def is_contained(bbox1, bbox2, include_edges=False):
    '''
    Determines whether one bounding box is contained inside another
    '''
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2
    if include_edges:
        return left1 >= left2 and top1 >= top2 and bottom1 <= bottom2 and right1 <= right2
    else:
        return left1 > left2 and top1 > top2 and bottom1 < bottom2 and right1 < right2
    

def check_output_abs(output, abs_value):
    return abs_value[0].issubset(output) and output.issubset(abs_value[1])


def get_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # determine the coordinates of the intersection rectangle
    int_left = max(left1, left2)
    int_top = max(top1, top2)
    int_right = min(right1, right2)
    int_bottom = min(bottom1, bottom2)

    if int_right < int_left or int_bottom < int_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (int_right - int_left) * (int_bottom - int_top)

    # compute the area of both AABBs
    bb1_area = (right1 - left1) * (bottom1 - top1)
    bb2_area = (right2 - left2) * (bottom2 - top2)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
