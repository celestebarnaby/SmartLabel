import math


from interpreter import Interpreter
from constants import *

from mnist_domain.mnist_utils import *
from mnist_domain.mnist_dsl import Expr

class MNISTInterpreter(Interpreter):

    def forward_ai(
        self,
        expr,
        inp
    ):
        output, _ = execute_helper(expr, get_eval_fns(False), inp, False)
        expr.abs_value = output

    def backward_ai(
            self,
            expr,
            inp,
            goal_under,
            goal_over,
            constraints
    ):
        return execute_backwards_helper(expr, get_backwards_eval_fns(), inp, (goal_under, goal_over))

    def eval_standard(
        self,
        expr,
        inp,    
    ):
        return execute_helper(expr, get_eval_fns(True), inp, True)

    def no_children(self, rule):
        if rule in {'map', 'filter', 'fold', 'length', 'map_imgs', 'apply', 'curry'}:
            return False 
        return True
    
    def subprogs_not_equal(self, prog1, prog2, inp, semantics):
            return any([self.eval_standard(child1, inp[semantics]) != self.eval_standard(child2, inp[semantics]) for child1, child2 in zip(prog1.children, prog2.children)])
    
    def get_all_universes(self, inp):
        l = inp["img-list"] + [inp["img"]]
        options = itertools.product(*l)
        universes = []
        for option in options:
            universes.append({"img-list": option[:-1], "img": option[-1]})
        return universes
    
    def apply_model(self, self_per_rule, cross_per_rule_pair, prog1, prog2):
        if prog1.name != prog2.name:
            if str(sorted([prog1.name, prog2.name])) not in cross_per_rule_pair:
                raise TypeError
            return cross_per_rule_pair[str(sorted([prog1.name, prog2.name]))]
        if len(prog1.children) == 0:
            return 1 
        
        w = math.prod([self.apply_model(self_per_rule, cross_per_rule_pair, child1, child2) for child1, child2 in zip(prog1.children, prog2.children)])
        res = w + (1-w)*self_per_rule[prog1.name]
        return res
    
    def gt_matches_abs_output(self, gt_output, abs_value):
        return gt_output >= abs_value[0] and gt_output <= abs_value[1]
    
    def parse(self, s):
        toks = list(reversed(s.replace("(", " ( ").replace(")", " ) ").split()))
        if len(toks) == 1:
            if "(" in toks[0] or ")" in toks[0]:
                raise Exception()
            return Expr(toks[0])
        else:
            if toks.pop() != "(":
                raise Exception()
            return parse_helper(toks)
        

    def get_labelling_q_answers(self, inp, obj_id, key):
        if obj_id == "img-list":
            return inp["conf"][obj_id][key]
        else:
            return inp["conf"][obj_id]

    def set_labelling_q_answer(self, inp, obj_id, key, answer):
        if obj_id == "img-list":
            original_obj = inp[obj_id][key]
            inp[obj_id][key] = [answer]
        else:
            original_obj = inp[obj_id]
            inp[obj_id] = [answer]
        return original_obj

    def reset_labelling_q(self, inp, obj_id, key, original_obj):
        if obj_id == "img-list":
            inp[obj_id][key] = original_obj
        else:
            inp[obj_id] = original_obj


    def get_num_partial_conf_samples(self, num_universes):
        return MNIST_NUM_PARTIAL_SAMPLES
    
    def ask_labelling_question(self, inp, key, obj_id, inp_id):
        if obj_id == "img-list":
            inp["conf"][obj_id][key] = [inp['gt'][obj_id][key]]
        else:
            inp["conf"][obj_id] = [inp["gt"][obj_id]]
        return True
        


def parse_helper(toks):
    name = toks.pop()
    if "(" in name or ")" in name:
        raise Exception()
    children = []
    while True:
        tok = toks.pop()
        if tok == ")":
            return Expr(name, children)
        elif tok == "(":
            children.append(parse_helper(toks))
        else:
            children.append(Expr(tok))



def execute_helper(expr, fns, inp, is_standard):
    args = []
    for child in expr.children:
        args.append(execute_helper(child, fns, inp, is_standard))
    res = fns[expr.name](args, inp, is_standard)
    expr.abs_value = res
    return res


def execute_backwards_helper(expr, fns, inp, goal):
    res = fns[expr.name](goal, expr.children, inp)
    for child in expr.children:
        if len(child.children) == 0 and not (child.name == "pred_int" and not expr.name == "map_imgs"):
            continue
        # TODO: why??
        if expr.name == "filter" and child.name == "curry":
            continue
        res = res and execute_backwards_helper(child, fns, inp, child.abs_value)
        if not res:
            break
    return res


def get_eval_fns(use_standard): 
    fns = {
        # higher order combinators
        "map": map_fn,
        "map_imgs": map_imgs_fn,
        "filter": filter_fn,
        "fold": fold_fn,
        "length": length_fn,
        # function application
        "apply": apply_fn,
        "curry": curry_fn,
        # functions
        "plus": lambda args, inp, is_standard: lambda x, y: plus_fn(x, y, is_standard),
        "max": lambda args, inp, is_standard: lambda x, y: max_fn(x, y, is_standard),
        "mult": lambda args, inp, is_standard: lambda x, y: mult_fn(x, y, is_standard),
        "ge": lambda args, inp, is_standard: lambda x, y: ge_fn(x, y, is_standard),
        "le": lambda args, inp, is_standard: lambda x, y: le_fn(x, y, is_standard),
        "input-list": lambda args, inp, is_standard: inp["img-list"],
        "input-img": lambda args, inp, is_standard: inp["img"],
        # list functions
        "id": id_fn,
        "pred_int": pred_int_fn,
    }
    # constants
    for i in range(MAX_DIGIT):
        fns[str(i)] = (lambda args, inp, is_standard, i=i: i) if use_standard else (lambda args, inp, is_standard, i=i: ((i, i), False))
    return fns


def get_backwards_eval_fns():
    return {
        "map": map_fn_backwards,
        "map_imgs": map_imgs_fn_backwards, 
        "filter": filter_fn_backwards,
        "fold": fold_fn_backwards,
        "length": length_fn_backwards,
        "apply": apply_fn_backwards,
        "curry": curry_fn_backwards,
        "pred_int": pred_int_fn_backwards
    }


def map_fn(args, inp, is_standard):
    if len(args) != 2:
        raise Exception()
    f = args[0]
    xs = args[1]

    if is_standard:
        return [None if x is None else f(x) for x in xs]
    else:
        # assuming f is monotonic
        return [(f(pred), flag) if pred is not None else (pred, flag) for (pred, flag) in xs]


def map_imgs_fn(args, inp, is_standard):
    if len(args) != 2:
        raise Exception()
    f = args[0]
    xs = args[1]

    if is_standard:
        return [None if x is None else f(x) for x in xs]
    else:
        pred_sets = []
        for img in xs:
            pred_sets.append(None if img is None else f(img))
        return pred_sets


# TODO: WTF is going on here
def map_fn_backwards(goal, children, inp):
    f = children[0]
    # TODO: WHY do we need both of these :'((((
    f_inverse = get_inverse(f)
    f_inverse2 = get_inverse2(f)
    xs = children[1]
    new_abs_value = []
    new_f_value = None
    if len(xs.abs_value) != len(goal):
        # TODO: error handling
        raise TypeError 
    for goal_item, item in zip(goal, xs.abs_value):
        goal_interval, _ = goal_item 
        interval, val = item
        if goal_interval is None:
            new_abs_value.append((interval, val))
            continue
        if interval is None:
            raise TypeError
        new_interval = intersect_intervals(interval, f_inverse(goal_interval))
        if new_interval is None or new_interval[0] < 0:
            return False
        if new_f_value is None:
            new_f_value = f_inverse2(goal_interval, new_interval)
            if new_f_value is None:
                return False
        else:
            new_f_value = intersect_intervals(new_f_value, f_inverse2(goal_interval, new_interval))
            if new_f_value is None:
                return False
        new_abs_value.append((new_interval, val))
    f.abs_value = (new_f_value, False)
    xs.abs_value = new_abs_value
    return True


def get_inverse2(f):
    oper = f.children[0]
    if oper.name == "plus":
        return lambda tup1, tup2: (max(tup1[0] - tup2[1], 0), max(tup1[1] - tup2[0], 0))
    elif oper.name == "mult":
        return lambda tup1, tup2: (math.floor(tup1[0] / tup2[1]) if tup2[1] > 0 else 0, math.ceil(tup1[1]/tup2[0]) if tup2[0] > 0 else 1000000000000)
    elif oper.name == "max":
        def max_inverse(goal, inter):
            if goal[0] < inter[0]:
                return None 
            return (goal[0] if goal[0] > inter[0] else 0, goal[1])
        return max_inverse
    else:
        raise TypeError


def get_inverse(f):
    oper = f.children[0]
    arg_interval, _ = f.children[1].abs_value
    if oper.name == "plus":
        return lambda tup: (max(0, tup[0] - arg_interval[1]), tup[1] - arg_interval[0])
    elif oper.name == "mult":
        return lambda tup: (math.floor(tup[0]/arg_interval[1]) if arg_interval[1] > 0 else 0, math.ceil(tup[1]/arg_interval[0]) if arg_interval[0] > 0 else 10000)
    elif oper.name == "max":
        def max_inverse(tup):
            if tup[0] < arg_interval[0]:
                return None 
            return (tup[0] if tup[0] > arg_interval[0] else 0, tup[1])
        return max_inverse
    else:
        raise TypeError


def map_imgs_fn_backwards(goal, children, inp):
    if len(goal) != len(inp['img-list']):
        raise TypeError 
    new_img_list = []
    for goal_item, item in zip(goal, inp['img-list']):
        goal_interval, _ = goal_item 
        new_item = [val for val in item if val >= goal_interval[0] and val <= goal_interval[1]]
        if len(new_item) == 0:
            return False 
        new_img_list.append(new_item)
    inp['img-list'] = new_img_list
    return True


def fold_fn(args, inp, is_standard):
    if len(args) != 3:
        raise Exception()
    f = args[0]
    b = args[1]
    xs = args[2]

    if is_standard:
        for x in xs:
            if not x is None:
                b = f(x, b)
        return b
    else:
        b = b[0]
        for (pred, flag) in xs:
            if pred is None:
                continue
            elif flag:
                b = join(b, f(b, pred))
            else:
                b = f(b, pred)
        return (b, False) 
    

# TODO: IMPROVE
def fold_fn_backwards(goal, children, inp):
    '''
    Performs backward AI on a fold expression.
    '''
    f = children[0]
    b = children[1]
    xs = children[2]
    new_abs_value = []
    
    # Enumerate all items in the child list of fold
    for i, (interval, val) in enumerate(xs.abs_value + [b.abs_value]):
        is_b_value = (i == len(xs.abs_value))
        if f.name == "plus":

            # Compute the smallest and largest possible values that this item could have, w.r.t. the abstract values of all OTHER items in the list,
            # (derived during forward AI), and the backward semantics of the fold function.
            smallest_val = sum([other_interval[0] for j, (other_interval, other_val) in enumerate(xs.abs_value + [b.abs_value]) if other_interval is not None and i != j and not other_val])
            largest_val = sum([other_interval[1] for j, (other_interval, _) in enumerate(xs.abs_value + [b.abs_value]) if other_interval is not None and i != j]) 
            if interval is None:
                if is_b_value:
                    raise TypeError
                new_abs_value.append((interval, val))
                continue

            # Intersect the newly derived range with the previously derived abstract output. 
            new_interval = intersect_intervals(interval, (max(0, goal[0] - largest_val), goal[1] - smallest_val if not val else interval[1]))
            if new_interval is None:
                if is_b_value:
                    return False
                if val and interval[0] > goal[1] - smallest_val:
                    new_abs_value.append((None, False))
                else:
                    return False 
            else:
                if is_b_value:
                    b.abs_value = (new_interval, val)
                else:
                    new_abs_value.append((new_interval, val))
        elif f.name == "mult":
            smallest_val = math.prod(([other_interval[0] for j, (other_interval, other_val) in enumerate(xs.abs_value + [b.abs_value]) if other_interval is not None and i != j and not other_val])) 
            # Special case just for multiplication where other_val == True but other_interval[0] == 0
            if any([other_interval[0] == 0 for j, (other_interval, _) in enumerate(xs.abs_value + [b.abs_value]) if other_interval is not None and i != j]):
                smallest_val = 0
            largest_val = math.prod([other_interval[1] for j, (other_interval, _) in enumerate(xs.abs_value + [b.abs_value]) if other_interval is not None and i != j]) #if not val else interval[1]
            if interval is None:
                if is_b_value:
                    raise TypeError
                new_abs_value.append((interval, val))
                continue
            new_interval = intersect_intervals(interval, (math.floor(goal[0]/largest_val) if largest_val > 0 else 0, math.ceil(goal[1]/smallest_val) if (smallest_val > 0 and not val) else interval[1]))
            if new_interval is None:
                if is_b_value:
                    return False
                if val and interval[0] > (math.ceil(goal[1]/smallest_val) if smallest_val > 0 else goal[1]):
                    new_abs_value.append((None, False))
                else:
                    return False 
            else: 
                if is_b_value:
                    b.abs_value = (new_interval, val)
                else:
                    new_abs_value.append((new_interval, val))
        elif f.name == "max":
            if interval is None:
                if is_b_value:
                    raise TypeError
                new_abs_value.append((interval, val))
                continue
            new_interval = intersect_intervals(interval, (0, goal[1]))
            if new_interval is None:
                if is_b_value:
                    return False
                if val:
                    new_abs_value.append((None, False))
                else:
                    return False 
            else:
                if is_b_value:
                    b.abs_value = (new_interval, val)
                else:
                    new_abs_value.append((new_interval, val))
    xs.abs_value = new_abs_value 
    return True


def filter_fn(args, inp, is_standard):
    if len(args) != 2:
        raise Exception()
    f = args[0]
    xs = args[1]

    if is_standard:
        return [None if x is None or not f(x) else x for x in xs]
    else:
        output = []
        for (pred, flag) in xs:
            if pred is None:
                output.append((None, False))
                continue
            res = f(pred)
            if res == 1:
                # The item DEFINITELY passes through the filter is, so we fall back to its existing flag
                output.append((pred, flag))
            elif res == 0:
                # True means there is UNCERTAINTY about the item's existence
                output.append((pred, True))
            else:
                # The item DEFINITELY gets filtered out, so, we don't keep the val and set the flag to False
                output.append((None, False))
        return output


def filter_fn_backwards(goal, children, inp):
    xs = children[1]
    new_abs_value = []
    if len(xs.abs_value) != len(goal):
        raise TypeError
    for goal_item, item in zip(goal, xs.abs_value):
        goal_interval, _ = goal_item 
        interval, val = item
        if goal_interval is None:
            new_abs_value.append((interval, val))
            continue
        new_interval = intersect_intervals(goal_interval, interval)
        if new_interval is None:
            return False
        new_abs_value.append((new_interval, val))
    xs.abs_value = new_abs_value
    return True


def length_fn(args, inp, is_standard):
    if len(args) != 1:
        raise Exception()
    xs = args[0]
    if is_standard:
        return len([x for x in xs if x is not None])
    else:
        min_len =  len([x for (x, flag) in xs if not flag and x is not None])
        max_len = len([x for x in xs if x is not None])
        return ((min_len, max_len), False)


def length_fn_backwards(goal, children, inp):
    xs = children[0]
    true_vals = [item for item in xs.abs_value if item[0] is not None and not item[1]]
    if len(true_vals) == goal[1]:
        new_abs_value = []
        for interval, val in xs.abs_value:
            if val:
                new_abs_value.append((None, False))
            else:
                new_abs_value.append((interval, val))
        xs.abs_value = new_abs_value
    false_vals = [item for item in xs.abs_value if item[0] is None]
    if len(xs.abs_value) - len(false_vals) == goal[0]:
        new_abs_value = []
        for interval, val in xs.abs_value:
            if val:
                new_abs_value.append((interval, False))
            else:
                new_abs_value.append((interval, val))
        xs.abs_value = new_abs_value
    return True


def apply_fn(args, inp, is_standard):
    if len(args) != 2:
        raise Exception()
    f = args[0]
    x = args[1]
    return f(x)

def apply_fn_backwards(goal, children, inp):
    f = children[0]
    f.abs_value = goal 
    return True

def curry_fn(args, inp, is_standard):
    if len(args) != 2:
        raise Exception()
    f = args[0]
    x = args[1]

    if is_standard:
        return lambda y: f(y, x)
    else:
        return lambda y: f(y, x[0]) 

def curry_fn_backwards(goal, children, inp):
    x = children[1]
    x.abs_value = goal
    return True


def id_fn(args, inp):
    if len(args) != 0:
        raise Exception()
    return lambda img: img


def plus_fn(x, y, is_standard):
    if is_standard:
        if x is None and y is None:
            raise TypeError
            return None
        elif x is None:
            raise TypeError
            return y
        elif y is None:
            raise TypeError
            return x
        else:
            return x + y
    else:
        return (x[0] + y[0], x[1] + y[1])
    

def max_fn(x, y, is_standard):
    if is_standard:
        if x is None and y is None:
            return None
        elif x is None:
            return y
        elif y is None:
            return x
        else:
            return max(x, y)
    else:
        return (max(x[0], y[0]), max(x[1], y[1]))


def mult_fn(x, y, is_standard):
    if is_standard:
        if x is None and y is None:
            return None
        elif x is None:
            return y
        elif y is None:
            return x
        else:
            return x * y
    else:
        return (x[0] * y[0], x[1] * y[1])


def ge_fn(x, y, is_standard):
    if is_standard:
        if x is None or y is None:
            return None
        else:
            return x >= y
    else:
        # DEFINITELY NOT filtered
        if x[0] >= y[1]:
            return 1 
        # DEFINITELY filtered
        elif x[1] < y[0]:
            return -1 
        # MAYBE filtered
        else:
            return 0


def le_fn(x, y, is_standard):
    if is_standard:
        if x is None or y is None:
            return None
        else:
            return x <= y
    else:
        # The item DEFINITELY doesn't get filtered
        if y[0] >= x[1]:
            return 1 
        # The item DEFINITELY gets filtered
        elif y[1] < x[0]:
            return -1 
        # There is UNCERTAINTY about whether it gets filtered
        else:
            return 0

def pred_int_fn(args, inp, is_standard):
    if len(args) != 0:
        # TODO: error handling
        raise Exception()

    def get_standard_list(img_list):
        return img_list
    
    def get_abs_list(img_list):
        return ((min(img_list), max(img_list)), False)

    return get_standard_list if is_standard else get_abs_list 


def pred_int_fn_backwards(goal, children, inp):
    goal_interval, _ = goal
    if goal_interval is None:
        return True
    inp['img'] = [val for val in inp['img'] if val >= goal_interval[0] and val <= goal_interval[1]]
    if len(inp['img']) == 0:
        return False 
    return True




