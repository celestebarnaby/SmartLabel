from image_edit_dsl import *
from image_edit_utils import *
from typing import Any, Set
from interpreter import Interpreter
import math

class ImageEditInterpreter(Interpreter):

    def forward_ai(
            self, 
            expr: Expression,
            abs_img,
    ):
        if expr.abs_value is None:
            expr.abs_value = (set(), set(abs_img.keys()))
        if isinstance(expr, IsObject):
            objs_under = set()
            objs_over = set()
            for (obj_id, obj_abs_img) in abs_img.items():
                if obj_abs_img["Label"] == expr.obj:
                    objs_over.add(obj_id)
                    if obj_abs_img["Flag"]:
                        objs_under.add(obj_id)
        elif (
            isinstance(expr, IsSmiling)
            or isinstance(expr, EyesOpen)
            or isinstance(expr, MouthOpen)
        ):
            objs_under = {obj for obj in abs_img if str(expr) in abs_img[obj] and abs_img[obj][str(expr)] == [True] and abs_img[obj]["Flag"]}
            objs_over = {obj for obj in abs_img if str(expr) in abs_img[obj] and (abs_img[obj][str(expr)] == True or (isinstance(abs_img[obj][str(expr)], list) and True in abs_img[obj][str(expr)]))}
        elif isinstance(expr, Union):
            objs_under = set()
            objs_over = set()
            for sub_expr in expr.expressions:
                new_under, new_over = self.forward_ai(sub_expr, abs_img)
                objs_under = objs_under.union(new_under)
                objs_over = objs_over.union(new_over)
        elif isinstance(expr, Intersection):
            objs_under = set(abs_img.keys())
            objs_over = set(abs_img.keys())
            for sub_expr in expr.expressions:
                new_under, new_over = self.forward_ai(sub_expr, abs_img)
                objs_under = objs_under.intersection(new_under)
                objs_over = objs_over.intersection(new_over)
        elif isinstance(expr, Complement):
            extracted_objs_under, extracted_objs_over = self.forward_ai(expr.expression, abs_img)
            objs_over = abs_img.keys() - extracted_objs_under
            objs_under = {obj for obj in abs_img.keys() - extracted_objs_over if abs_img[obj]["Flag"]}
        elif isinstance(expr, Map):
            objs_under, objs_over = self.eval_map_abs(expr, abs_img)
        else:
            # TODO: error handling
            print("Invalid expr")
            print(expr)
            raise TypeError 
        expr.abs_value = (expr.abs_value[0].union(objs_under), expr.abs_value[1].intersection(objs_over))
        return (expr.abs_value[0], expr.abs_value[1])


    def backward_ai(
            self,
            expr, 
            abs_img, 
            goal_under, 
            goal_over, 
            constraints):
        if isinstance(expr, IsObject):
            for obj_id in goal_under:
                if obj_id not in abs_img or abs_img[obj_id]["Label"] != expr.obj:
                    return False
                if obj_id not in constraints:
                    constraints[obj_id] = {}

                if "Exists" in constraints[obj_id] and not constraints[obj_id]["Exists"]:
                    return False
                
                constraints[obj_id]["Exists"] = True
            for obj_id in set(abs_img.keys()) - goal_over:
                if abs_img[obj_id]["Flag"] == True and abs_img[obj_id]['Label'] == expr.obj:
                    return False
                if abs_img[obj_id]['Label'] == expr.obj:
                    if obj_id not in constraints:
                        constraints[obj_id] = {}

                    if "Exists" in constraints[obj_id] and constraints[obj_id]["Exists"]:
                        return False

                    constraints[obj_id]["Exists"] = False 
            return True
        elif (
            isinstance(expr, IsSmiling)
            or isinstance(expr, EyesOpen)
            or isinstance(expr, MouthOpen)
        ):
            for obj_id in goal_under:
                if obj_id not in abs_img or str(expr) not in abs_img[obj_id] or abs_img[obj_id][str(expr)] == [False]:
                    return False 
                if obj_id not in constraints:
                    constraints[obj_id] = {}

                if "Exists" in constraints[obj_id] and not constraints[obj_id]["Exists"]:
                    return False
                if str(expr) in constraints[obj_id] and not constraints[obj_id][str(expr)]:
                    return False 
                
                constraints[obj_id]["Exists"] = True 
                constraints[obj_id][str(expr)] = True

            for obj_id in set(abs_img.keys()) - goal_over:
                if abs_img[obj_id]["Flag"] == True and (str(expr) in abs_img[obj_id] and abs_img[obj_id][str(expr)] == [True]):
                    return False 
                if abs_img[obj_id]["Flag"] and str(expr) in abs_img[obj_id]:
                    if obj_id not in constraints:
                        constraints[obj_id] = {}

                    if str(expr) in constraints[obj_id] and constraints[obj_id][str(expr)]:
                        return False

                    constraints[obj_id][str(expr)] = False
                if str(expr) in abs_img[obj_id] and abs_img[obj_id][str(expr)] == [True]:
                    if obj_id not in constraints:
                        constraints[obj_id] = {}

                    if "Exists" in constraints[obj_id] and constraints[obj_id]["Exists"]:
                        return False

                    constraints[obj_id]["Exists"] = False
            return True
        elif isinstance(expr, Complement):
            sub_expr_under = {obj for obj in set(abs_img.keys()) - goal_over if abs_img[obj]["Flag"]}
            sub_expr_over = set(abs_img.keys()) - goal_under
            if not update_abs_output(expr.expression, sub_expr_under, sub_expr_over):
                return False 
            return self.backward_ai(expr.expression, abs_img, expr.expression.abs_value[0], expr.expression.abs_value[1], constraints)
        elif isinstance(expr, Union):
            for i, sub_expr in enumerate(expr.expressions):
                sub_expr_over = goal_over
                sub_expr_under = goal_under
                for j, other_sub_expr in enumerate(expr.expressions):
                    if i == j:
                        continue 
                    sub_expr_under = sub_expr_under - other_sub_expr.abs_value[1] 
                if not update_abs_output(sub_expr, sub_expr_under, sub_expr_over):
                    return False
                if not self.backward_ai(sub_expr, abs_img, sub_expr.abs_value[0], sub_expr.abs_value[1], constraints):
                    return False
            return True 
        elif isinstance(expr, Intersection):
            for i, sub_expr in enumerate(expr.expressions):
                sub_expr_under = goal_under 
                # these are the objects that the expr CAN'T output
                impossible_objs = set(abs_img.keys()) - goal_over
                # if EVERY OTHER sub expr MUST output this object, then the ith sub expr CAN'T
                for j, other_sub_expr in enumerate(expr.expressions):
                    if i == j:
                        continue
                    impossible_objs = impossible_objs.intersection(other_sub_expr.abs_value[0])
                sub_expr_over = set(abs_img.keys()) - impossible_objs
                if not update_abs_output(sub_expr, sub_expr_under, sub_expr_over):
                    return False
                if not self.backward_ai(sub_expr, abs_img, sub_expr.abs_value[0], sub_expr.abs_value[1], constraints):
                    return False
            return True
        elif isinstance(expr, Map):
            sub_expr_under = set()
            sub_expr_over = set(abs_img.keys())
            rest_under = goal_under 
            rest_over = set(abs_img.keys())
            if not update_abs_output(expr.expression, sub_expr_under, sub_expr_over):
                return False
            if not update_abs_output(expr.restriction, rest_under, rest_over):
                return False
            if not self.backward_ai(expr.expression, abs_img, expr.expression.abs_value[0], expr.expression.abs_value[1], constraints) or not self.backward_ai(expr.restriction, abs_img, expr.restriction.abs_value[0], expr.restriction.abs_value[1], constraints):
                return False 
            return True
        else: 
            # TODO: error handling
            print("INVALID EXTRACTOR")
            print(str(expr))
            raise TypeError


    def eval_map_abs(self, prog, abs_img):
        (objs_under, objs_over) = self.forward_ai(prog.expression, abs_img)
        (rest_under, rest_over) = self.forward_ai(prog.restriction, abs_img)
        mapped_objs_under = set()
        mapped_objs_over = set()
        if isinstance(prog.position, GetLeft):
            for target_obj_id in objs_over:
                target_left, target_top, target_right, target_bottom = abs_img[
                    target_obj_id
                ]["bbox"]
                target_x = abs_img[target_obj_id]["center"][0]
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    x = abs_img_map["center"][0]
                    if x > target_x:
                        continue
                    if top > target_bottom or bottom < target_top:
                        continue
                    mapped_objs_over.add(obj_id)
            for target_obj_id in objs_under:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_x = abs_img[target_obj_id]["center"][0]
                cur_obj_id = None
                cur_x = None
                for obj_id, abs_img_map in abs_img.items():
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    x = abs_img_map["center"][0]
                    if x > target_x:
                        continue
                    if top > target_bottom or bottom < target_top:
                        continue
                    if cur_x is None or x > cur_x:
                        cur_x = x
                        cur_obj_id = obj_id
                if cur_obj_id is not None and cur_obj_id in rest_under:
                    mapped_objs_under.add(cur_obj_id)
        elif isinstance(prog.position, GetRight):
            for target_obj_id in objs_over:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_x = abs_img[target_obj_id]["center"][0]
                for obj_id, abs_img_map in abs_img.items():
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    x = abs_img_map["center"][0]
                    if x < target_x:
                        continue
                    if top > target_bottom or bottom < target_top:
                        continue
                    mapped_objs_over.add(obj_id)
            for target_obj_id in objs_under:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_x = abs_img[target_obj_id]["center"][0]
                cur_obj_id = None
                cur_x = None
                for obj_id, abs_img_map in abs_img.items():
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    x = abs_img_map["center"][0]
                    if x < target_x:
                        continue
                    if top > target_bottom or bottom < target_top:
                        continue
                    if cur_x is None or x < cur_x:
                        cur_x = x
                        cur_obj_id = obj_id
                if cur_obj_id is not None and cur_obj_id in rest_under:
                    mapped_objs_under.add(cur_obj_id)
        elif isinstance(prog.position, GetAbove):
            for target_obj_id in objs_over:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_y = abs_img[target_obj_id]["center"][1]
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    y = abs_img_map["center"][1]
                    if y > target_y:
                        continue
                    if right < target_left or left > target_right:
                        continue
                    mapped_objs_over.add(obj_id)
            for target_obj_id in objs_under:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_y = abs_img[target_obj_id]["center"][1]
                cur_obj_id = None
                cur_y = None
                for obj_id, abs_img_map in abs_img.items():
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    y = abs_img_map["center"][1]
                    if y > target_y:
                        continue
                    if right < target_left or left > target_right:
                        continue
                    if cur_y is None or y > cur_y:
                        cur_y = y
                        cur_obj_id = obj_id
                if cur_obj_id is not None and cur_obj_id in rest_under:
                    mapped_objs_under.add(cur_obj_id)
        elif isinstance(prog.position, GetBelow):
            for target_obj_id in objs_over:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_y = abs_img[target_obj_id]["center"][1]
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    y = abs_img_map["center"][1]
                    if y < target_y:
                        continue
                    if right < target_left or left > target_right:
                        continue
                    mapped_objs_over.add(obj_id)
            for target_obj_id in objs_under:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_y = abs_img[target_obj_id]["center"][1]
                cur_obj_id = None
                cur_y = None
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    y = abs_img_map["center"][1]
                    if y < target_y:
                        continue
                    if right < target_left or left > target_right:
                        continue
                    if cur_y is None or y < cur_y:
                        cur_y = y
                        cur_obj_id = obj_id
                if cur_obj_id is not None and cur_obj_id in rest_under:
                    mapped_objs_under.add(cur_obj_id)
        elif isinstance(prog.position, GetContains):
            for target_obj_id in objs_over:
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    if not is_contained(abs_img_map["bbox"], abs_img[target_obj_id]["bbox"]):
                        continue
                    mapped_objs_over.add(obj_id)
                    if target_obj_id not in objs_under:
                        continue
                    if obj_id not in rest_under:
                        continue
                    mapped_objs_under.add(obj_id)
        elif isinstance(prog.position, GetIsContained):
            for target_obj_id in objs_over:
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest_over:
                        continue
                    if not is_contained(abs_img[target_obj_id]["bbox"], abs_img_map["bbox"]):
                        continue
                    mapped_objs_over.add(obj_id)
                    if target_obj_id not in objs_under:
                        continue
                    if obj_id not in rest_under:
                        continue
                    mapped_objs_under.add(obj_id)
        else:
            # TODO: error handling
            print("Invalid Map")
            raise TypeError
        return mapped_objs_under, mapped_objs_over


    def eval_standard(
        self,
        expr: Expression,
        abs_img: Dict[str, Dict[str, Any]],
    ):  # -> Set[dict[str, str]]:
        if isinstance(expr, Map):
            res = self.eval_map_standard(expr, abs_img)
        elif isinstance(expr, IsObject):
            objs = set()
            for (obj_id, obj_abs_img) in abs_img.items():
                if obj_id == "prob":
                    continue
                if obj_abs_img["Label"] == expr.obj:
                    objs.add(obj_id)
            res = objs
        elif isinstance(expr, Union):
            res = set()
            for sub_expr in expr.expressions:
                res = res.union(
                    self.eval_standard(sub_expr, abs_img)
                )
        elif isinstance(expr, Intersection):
            res = set(abs_img.keys())
            for sub_expr in expr.expressions:
                res = res.intersection(
                    self.eval_standard(sub_expr, abs_img)
                )
        elif isinstance(expr, Complement):
            # All objs in target image except those extracted
            extracted_objs = self.eval_standard(expr.expression, abs_img)
            res = abs_img.keys() - extracted_objs
        elif (
            isinstance(expr, IsSmiling)
            or isinstance(expr, EyesOpen)
            or isinstance(expr, MouthOpen)
        ):
            res = {obj for obj in abs_img if obj != "prob" and str(expr) in abs_img[obj] and abs_img[obj][str(expr)]}
        else:
            # TODO: error handling
            print(expr)
            raise Exception
        res = {item for item in res if item != "prob"}
        return res


    def eval_map_standard(
        self,
        map_expr: Map,
        abs_img: Dict[str, Dict[str, Any]],
    ) -> Set[str]:
        objs = self.eval_standard(
            map_expr.expression, abs_img
        )
        rest = self.eval_standard(
            map_expr.restriction,
            abs_img,
        )
        mapped_objs = set()
        if isinstance(map_expr.position, GetBelow):
            for target_obj_id in objs:
                target_left, target_top, target_right, _ = abs_img[target_obj_id]["bbox"]
                target_y = abs_img[target_obj_id]["center"][1]
                cur_obj_id = None
                cur_y = None
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == "prob":
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    y = abs_img_map["center"][1]
                    if y < target_y:
                        continue
                    if right < target_left or left > target_right:
                        continue
                    if cur_y is None or y < cur_y:
                        cur_y = y
                        cur_obj_id = obj_id
                if cur_obj_id is not None:
                    mapped_objs.add(cur_obj_id)
        elif isinstance(map_expr.position, GetAbove):
            for target_obj_id in objs:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_y = abs_img[target_obj_id]["center"][1]
                cur_obj_id = None
                cur_y = None
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == "prob":
                        continue
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    y = abs_img_map["center"][1]
                    if y > target_y:
                        continue
                    if right < target_left or left > target_right:
                        continue
                    if cur_y is None or y > cur_y:
                        cur_y = y
                        cur_obj_id = obj_id
                if cur_obj_id is not None:
                    mapped_objs.add(cur_obj_id)
        elif isinstance(map_expr.position, GetLeft):
            for target_obj_id in objs:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_x = abs_img[target_obj_id]["center"][0]
                cur_obj_id = None
                cur_x = None
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == "prob":
                        continue
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    x = abs_img_map["center"][0]
                    if x > target_x:
                        continue
                    if top > target_bottom or bottom < target_top:
                        continue
                    if cur_x is None or x > cur_x:
                        cur_x = x
                        cur_obj_id = obj_id
                if cur_obj_id is not None:
                    mapped_objs.add(cur_obj_id)
        elif isinstance(map_expr.position, GetRight):
            for target_obj_id in objs:
                target_left, target_top, target_right, target_bottom = abs_img[target_obj_id]["bbox"]
                target_x = abs_img[target_obj_id]["center"][0]
                cur_obj_id = None
                cur_x = None
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == "prob":
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest:
                        continue
                    left, top, right, bottom = abs_img_map["bbox"]
                    x = abs_img_map["center"][0]
                    if x < target_x:
                        continue
                    if top > target_bottom or bottom < target_top:
                        continue
                    if cur_x is None or x < cur_x:
                        cur_x = x
                        cur_obj_id = obj_id
                if cur_obj_id is not None:
                    mapped_objs.add(cur_obj_id)
        elif isinstance(map_expr.position, GetContains):
            for target_obj_id in objs:
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == "prob":
                        continue
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest:
                        continue
                    if is_contained(abs_img_map["bbox"], abs_img[target_obj_id]["bbox"]):
                        mapped_objs.add(obj_id)
        elif isinstance(map_expr.position, GetIsContained):
            for target_obj_id in objs:
                for obj_id, abs_img_map in abs_img.items():
                    if obj_id == "prob":
                        continue
                    if abs_img_map["ImgIndex"] != abs_img[target_obj_id]["ImgIndex"]:
                        continue
                    if obj_id == target_obj_id:
                        continue
                    if obj_id not in rest:
                        continue
                    if is_contained(abs_img[target_obj_id]["bbox"], abs_img_map["bbox"]):
                        mapped_objs.add(obj_id)
        else: 
            # TODO: erro handling
            print("Invalid Map")
            raise TypeError
        return mapped_objs
    

    def get_prog_output(self, prog, examples, parent, semantics):
        if (
            isinstance(prog, Union)
            or isinstance(prog, Intersection)
            or isinstance(prog, Complement)
            or isinstance(prog, Map)
            or isinstance(prog, IsObject)
        ):
            return None
        elif isinstance(parent, IsObject):
            prog = IsObject(prog)
        if semantics == "standard":
            output_per_example = []
            for abs_img, _ in examples: 
                output = self.eval_standard(prog.duplicate(), abs_img[semantics])
                output_per_example.append((output, output))
            return output_per_example
        output_per_example = []
        for abs_img, _ in examples:
            prog_output_under, prog_output_over = self.forward_ai(prog.duplicate(), abs_img["conf"])
            output_per_example.append((prog_output_under, prog_output_over))
        return output_per_example



    def subprogs_not_equal(self, prog1, prog2, abs_img, semantics):
        if isinstance(prog1, Map):
            return self.eval_standard(prog1.expression, abs_img[semantics]) != self.eval_standard(prog2.expression, abs_img[semantics]) or self.eval_standard(prog1.restriction, abs_img[semantics]) != self.eval_standard(prog2.restriction, abs_img[semantics])
        elif isinstance(prog1, Union) or isinstance(prog1, Intersection):
            return any([self.eval_standard(sub_extr1, abs_img[semantics]) != self.eval_standard(sub_extr2, abs_img[semantics]) for (sub_extr1, sub_extr2) in zip(prog1.expressions, prog2.expressions)])
        elif isinstance(prog1, Complement):
            return self.eval_standard(prog1.expression, abs_img[semantics]) != self.eval_standard(prog2.expression, abs_img[semantics])
        else:
            # TODO: error handling
            raise TypeError
        

    def no_children(self, rule):
        if rule.startswith("Union") or rule.startswith("Intersection") or rule in {"Map", "Complement"}:
            return False 
        return True
    

    # TODO: I should be able to make this language agnostic...?
    def apply_model(self, self_per_rule, cross_per_rule_pair, prog1, prog2):
        if prog1.get_grammar_rule() != prog2.get_grammar_rule():
            return cross_per_rule_pair[str(sorted([prog1.get_grammar_rule(), prog2.get_grammar_rule()]))]
        if self.no_children(prog1.get_grammar_rule()):
            return 1 
        if isinstance(prog1, Map):
            w = self.apply_model(self_per_rule, cross_per_rule_pair, prog1.expression, prog2.expression) * self.apply_model(self_per_rule, cross_per_rule_pair, prog1.restriction, prog2.restriction)
        elif isinstance(prog1, Union) or isinstance(prog1, Intersection):
            w =  math.prod([self.apply_model(self_per_rule, cross_per_rule_pair, sub_extr1, sub_extr2) for (sub_extr1, sub_extr2) in zip(prog1.expressions, prog2.expressions)])
        elif isinstance(prog1, Complement):
            w = self.apply_model(self_per_rule, cross_per_rule_pair, prog1.expression, prog2.expression)
        res = w + (1-w)*self_per_rule[prog1.get_grammar_rule()]
        return res
        
    def matches_constraints(self, abs_img, constraints):
        for obj_id in constraints:
            if "Exists" in constraints[obj_id] and constraints[obj_id]["Exists"] and obj_id not in abs_img:
                return False 
            if "Exists" in constraints[obj_id] and not constraints[obj_id]["Exists"] and obj_id in abs_img:
                return False 
            for attr in ATTRIBUTES:
                if attr not in constraints[obj_id]:
                    continue
                if constraints[obj_id][attr] and obj_id not in abs_img:
                    return False 
                if constraints[obj_id][attr] and (attr not in abs_img[obj_id] or not abs_img[obj_id][attr]):
                    return False 
                if not constraints[obj_id][attr] and obj_id in abs_img and attr in abs_img[obj_id] and abs_img[obj_id][attr]:
                    return False 
        return True


