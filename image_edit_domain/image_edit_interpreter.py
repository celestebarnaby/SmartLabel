from typing import Any, Set
import math
import itertools

from interpreter import Interpreter

from image_edit_domain.image_edit_dsl import *
from image_edit_domain.image_edit_utils import *

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
                if obj_id == "prob":
                    continue
                if obj_abs_img["Label"] == expr.obj:
                    objs_over.add(obj_id)
                    if obj_abs_img["Flag"]:
                        objs_under.add(obj_id)
        # elif isinstance(expr, MatchesWord):
        #     objs_under = set()
        #     objs_over = set()
        #     for (obj_id, obj_abs_img) in abs_img.items():
        #         if obj_abs_img["Label"] == "Text" and obj_abs_img["Text"] == expr.word:
        #             objs_over.add(obj_id)
        #             if obj_abs_img["Flag"]:
        #                 objs_under.add(obj_id)
        elif (
            isinstance(expr, IsSmiling)
            or isinstance(expr, EyesOpen)
            or isinstance(expr, MouthOpen)
            # or isinstance(expr, IsPrice)
            # or isinstance(expr, IsPhoneNumber)
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
            constraints
    ):
        # When we reach a leaf node in the AST, we check whether the leaf node's output on the image is consistent
        # with the goal output (derived from forward_ai) and with the constraints (derived from other leaf nodes in the program).
        # If it is not, we return False. Otherwise, we update the constraints and continue backward_ai        
        if isinstance(expr, IsObject):
            # These are the objects that the leaf node MUST output
            for obj_id in goal_under:
                # If the image doesn't contain this object, or the object has a different label than the expression, return False
                if obj_id not in abs_img or abs_img[obj_id]["Label"] != expr.obj:
                    return False
                if obj_id not in constraints:
                    constraints[obj_id] = {}

                # If our constraints specify that this object CANNOT be output, return False
                if "Exists" in constraints[obj_id] and not constraints[obj_id]["Exists"]:
                    return False
                
                # Update constraints to specify that the object MUST exists
                constraints[obj_id]["Exists"] = True

            # These are the objects that MUST NOT be output by the leaf node
            for obj_id in set(abs_img.keys()) - goal_over:
                # If the lead node WILL output the object, return False
                if abs_img[obj_id]["Flag"] == True and abs_img[obj_id]['Label'] == expr.obj:
                    return False
                if abs_img[obj_id]['Label'] == expr.obj:
                    if obj_id not in constraints:
                        constraints[obj_id] = {}

                    # If our constraints specify that the object MUST be output, return False
                    if "Exists" in constraints[obj_id] and constraints[obj_id]["Exists"]:
                        return False
                    
                    # Update our constraints to specify that the object MUST NOT exist
                    constraints[obj_id]["Exists"] = False 
            return True
        # elif isinstance(expr, MatchesWord):
        #     # These are the objects that the leaf node MUST output
        #     for obj_id in goal_under:
        #         # If the image doesn't contain this object, or the object has different text than the expression, return False
        #         if obj_id not in abs_img or abs_img[obj_id]["Label"] != "Text" or abs_img[obj_id]["Text"] != expr.word:
        #             return False
        #         if obj_id not in constraints:
        #             constraints[obj_id] = {}

        #         # If our constraints specify that this object CANNOT be output, return False
        #         if "Exists" in constraints[obj_id] and not constraints[obj_id]["Exists"]:
        #             return False
                
        #         # Update constraints to specify that the object MUST exists
        #         constraints[obj_id]["Exists"] = True

        #     # These are the objects that MUST NOT be output by the leaf node
        #     for obj_id in set(abs_img.keys()) - goal_over:
        #         # If the lead node WILL output the object, return False
        #         if abs_img[obj_id]["Flag"] == True and abs_img[obj_id]['Label'] == "Text" and abs_img[obj_id]["Text"] == expr.word:
        #             return False
        #         if abs_img[obj_id]["Label"] == "Text" and abs_img[obj_id]["Text"] == expr.word:
        #             if obj_id not in constraints:
        #                 constraints[obj_id] = {}

        #             # If our constraints specify that the object MUST be output, return False
        #             if "Exists" in constraints[obj_id] and constraints[obj_id]["Exists"]:
        #                 return False
                    
        #             # Update our constraints to specify that the object MUST NOT exist
        #             constraints[obj_id]["Exists"] = False 
        #     return True
        # Leaf nodes for specific attributes of human faces. This case is similar to the IsObject case.
        elif (
            isinstance(expr, IsSmiling)
            or isinstance(expr, EyesOpen)
            or isinstance(expr, MouthOpen)
            # or isinstance(expr, IsPrice)
            # or isinstance(expr, IsPhoneNumber)
        ):
            # Objects that MUST be output
            for obj_id in goal_under:
                # If the object does not have the attribute, return False
                if obj_id not in abs_img or str(expr) not in abs_img[obj_id] or abs_img[obj_id][str(expr)] == [False]:
                    return False 
                if obj_id not in constraints:
                    constraints[obj_id] = {}

                # If the constraints specify that the object must not exist, return False
                if "Exists" in constraints[obj_id] and not constraints[obj_id]["Exists"]:
                    return False
                # If the constraints specify that the object does not have the attribute, return False
                if str(expr) in constraints[obj_id] and not constraints[obj_id][str(expr)]:
                    return False 
                
                # Update the constraints to specify that the object must exist AND must have the attribute
                constraints[obj_id]["Exists"] = True 
                constraints[obj_id][str(expr)] = True

            # Object that MUST NOT be output
            for obj_id in set(abs_img.keys()) - goal_over:
                # If the object exists and definitely has the attribute, return False
                if abs_img[obj_id]["Flag"] == True and (str(expr) in abs_img[obj_id] and abs_img[obj_id][str(expr)] == [True]):
                    return False 
                if abs_img[obj_id]["Flag"] and str(expr) in abs_img[obj_id]:
                    if obj_id not in constraints:
                        constraints[obj_id] = {}

                    # If the constraints specify that the object has the attribute, return False
                    if str(expr) in constraints[obj_id] and constraints[obj_id][str(expr)]:
                        return False

                    # Update the constraints to specify that the object MUST NOT have the attribute
                    constraints[obj_id][str(expr)] = False
                if str(expr) in abs_img[obj_id] and abs_img[obj_id][str(expr)] == [True]:
                    if obj_id not in constraints:
                        constraints[obj_id] = {}

                    # If the object has the attribute and the constraints specify that the object MUST be output, return False
                    if "Exists" in constraints[obj_id] and constraints[obj_id]["Exists"]:
                        return False
                    
                    # If the object has the attribute, update the constraints to specify that the object MUST NOT be output
                    constraints[obj_id]["Exists"] = False
            return True
        elif isinstance(expr, Complement):
            sub_expr_under = {obj for obj in set(abs_img.keys()) - goal_over if abs_img[obj]["Flag"]}
            sub_expr_over = set(abs_img.keys()) - goal_under
            if not update_abs_output(expr.expression, sub_expr_under, sub_expr_over):
                return False 
            # Recursively perform backward AI on the subexpression of compement
            return self.backward_ai(expr.expression, abs_img, expr.expression.abs_value[0], expr.expression.abs_value[1], constraints)
        elif isinstance(expr, Union):
            for i, sub_expr in enumerate(expr.expressions):
                # Overapproximated output of subexpression consists of all objects in the overapproximated goal output
                sub_expr_over = goal_over
                sub_expr_under = goal_under
                # Underapproximated output of union's subexpression consist of all objects that in the goal output
                # that MUST NOT be output by ANY OTHER subexpression
                for j, other_sub_expr in enumerate(expr.expressions):
                    if i == j:
                        continue 
                    sub_expr_under = sub_expr_under - other_sub_expr.abs_value[1] 
                if not update_abs_output(sub_expr, sub_expr_under, sub_expr_over):
                    return False
                # Recursively perform backward AI on the subexpressions of union
                if not self.backward_ai(sub_expr, abs_img, sub_expr.abs_value[0], sub_expr.abs_value[1], constraints):
                    return False
            return True 
        elif isinstance(expr, Intersection):
            for i, sub_expr in enumerate(expr.expressions):
                # Underapproximated output of subexpression consists of all objects in the underapproximated goal output
                sub_expr_under = goal_under 
                # These are the objects that must not be output by ALL subexpressions of intersection
                impossible_objs = set(abs_img.keys()) - goal_over
                # If EVERY OTHER sub expr MUST output this object, then the ith sub expr CAN'T
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
        '''
        Performs forward AI on a Map expression
        '''
        # Recursively perform forward AI on both subexpressions of the Map expression
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
                    if obj_id == "prob":
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
                    mapped_objs_over.add(obj_id)
            for target_obj_id in objs_under:
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
                    if obj_id == "prob":
                        continue
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
                    if obj_id == "prob":
                        continue
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
                    if obj_id == "prob":
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
                    mapped_objs_over.add(obj_id)
            for target_obj_id in objs_under:
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
                    if obj_id == "prob":
                        continue
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
                    if obj_id == "prob":
                        continue
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
                    if obj_id == "prob":
                        continue
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
                    if obj_id == "prob":
                        continue
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
    ):  
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
        # elif isinstance(expr, MatchesWord):
        #     objs = set()
        #     for (obj_id, obj_abs_img) in abs_img.items():
        #         if obj_abs_img["Label"] == "Text" and obj_abs_img["Text"] == expr.word:
        #             objs.add(obj_id)
        #     res = objs
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
            # or isinstance(expr, IsPrice)
            # or isinstance(expr, IsPhoneNumber)
        ):
            res = {obj for obj in abs_img if obj != "prob" and  str(expr) in abs_img[obj] and abs_img[obj][str(expr)]}
        else:
            # TODO: error handling
            print(expr)
            raise Exception
        return {obj for obj in res if obj != "prob"}


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
    

    def apply_model(self, self_per_rule, cross_per_rule_pair, prog1, prog2):
        '''
        For LearnSy only. 
        '''
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
        '''
        Checks whether an image matches the constraints generated during backward AI.
        '''
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
    

    def get_all_universes(self, full_inp):
        inp_conf = full_inp["conf"]
        inp_conf_copy = {}
        for key, val in inp_conf.items():
            inp_conf_copy[key] = [item for item in self.get_all_versions_of_object(val)]
        keys = list(inp_conf_copy.keys())
        vals = list(inp_conf_copy.values())
        all_lists = list(itertools.product(*vals))
        all_universes = [{keys[i]: l[i] for i in range(len(l)) if l[i] is not None} for l in all_lists]
        for universe in all_universes:
            prob = 1
            for obj_id, obj in inp_conf.items():
                if obj_id in universe and obj["Flag"] == False:
                    prob *= obj["Flag_prob"]
                elif obj["Flag"] == False:
                    prob *= (1 - obj["Flag_prob"])
                    # not sure about this
                    continue
                for attr in ATTRIBUTES:
                    if attr in universe[obj_id] and len(obj[attr]) == 2:
                        prob *= inp_conf[obj_id][f"{attr}_prob"][universe[obj_id][attr]]
                    # else:
                        # prob *= abs_img_conf[obj_id][key + "_prob"][False]
            universe["prob"] = prob
        return all_universes
    
    def get_all_universes2(self, full_inp):
        inp_conf = full_inp["conf"]
        inp_conf_copy = {}
        for key, val in inp_conf.items():
            inp_conf_copy[key] = [item for item in self.get_all_versions_of_object(val)]
        keys = list(inp_conf_copy.keys())
        vals = list(inp_conf_copy.values())
        all_lists = list(itertools.product(*vals))
        all_universes = [{keys[i]: l[i] for i in range(len(l)) if l[i] is not None} for l in all_lists]
        for universe in all_universes:
            prob = 1
            for obj_id, obj in inp_conf.items():
                if obj_id in universe and obj["Flag"] == False:
                    prob *= obj["Flag_prob"]
                elif obj["Flag"] == False:
                    prob *= (1 - obj["Flag_prob"])
                    # not sure about this
                    continue
                for attr in ATTRIBUTES:
                    if attr in universe[obj_id] and len(obj[attr]) == 2:
                        prob *= inp_conf[obj_id][f"{attr}_prob"][universe[obj_id][attr]]
                    # else:
                        # prob *= abs_img_conf[obj_id][key + "_prob"][False]
            universe["prob"] = prob
        return all_universes
    

    def get_all_versions_of_object(self, obj):
        versions = []
        if not obj["Flag"]:
            versions.append(None)
        if obj["Label"] != "Face":
            versions.append(obj)
            return versions 
        options = [obj[key] for key in ATTRIBUTES]
        all_lists = list(itertools.product(*options))
        all_dicts = [
            {key: l[i] for i, key in enumerate(ATTRIBUTES)}
            for l in all_lists
        ]
        obj_without_keys = {key : val for key, val in obj.items() if key not in ATTRIBUTES}
        for d in all_dicts:
            versions.append(d | obj_without_keys)
        return versions
    
    def gt_matches_abs_output(self, output, abs_value):
        return abs_value[0].issubset(output) and output.issubset(abs_value[1])
    

    def represent_output(self, output):
        return str(sorted(list(output)))


    def get_labelling_q_answers(self, inp, obj_id, key):
        return [True, False]
    

    def get_labelling_q_probs(self, inp, obj_id, key):
        val = inp["conf"][obj_id][f"{key}_prob"]
        if key == "Flag":
            return [val, 1 - val]
        else:
            return list(val.values())
    

    def set_labelling_q_answer(self, inp, obj_id, key, answer):
        if key == "Flag":
            original_obj = inp[obj_id]
            if answer:
                original_obj["Flag"] = True 
            else:
                del inp[obj_id]
        else:
            original_obj = inp[obj_id][key]
            inp[obj_id][key] = [answer]
        return original_obj
    

    def set_labelling_q_probs(self, inp, obj_id, key, answer):
        if key == "Flag":
            original_obj = inp["conf"][obj_id][f"{key}_prob"]
            if answer:
                inp["conf"][obj_id][f"{key}_prob"] = 1
            # TODO: I think not needed? 
            # else:
                # del inp[obj_id]
        else:
            original_obj = inp["conf"][obj_id][f"{key}_prob"]
            inp["conf"][obj_id][f"{key}_prob"] = {answer: 1, not answer: 0}
        return original_obj
        

    def reset_labelling_q(self, inp, obj_id, key, original_obj):
        if key == "Flag":
            inp[obj_id] = original_obj
            original_obj["Flag"] = False 
        else:
            inp[obj_id][key] = [True, False]


    def reset_labelling_q_probs(self, inp, obj_id, key, probs):
        inp["conf"][obj_id][f"{key}_prob"] = probs


    def get_num_partial_conf_samples(self, num_universes):
        return min(IMG_EDIT_MAX_PARTIAL_SAMPLES, int(num_universes * PARTIAL_AMT)) if int(num_universes * PARTIAL_AMT) > 0 else min(IMG_EDIT_MIN_PARTIAL_SAMPLES, num_universes) 
    

    def ask_labelling_question(self, abs_img, key, obj_id, img):

        # Find the ground truth label that corresponds to the predicted label
        gt_ids = {gt_id for gt_id in abs_img["gt"] if abs_img["gt"][gt_id]["Label"] == abs_img["conf"][obj_id]["Label"]}

        # If the object does not exist in the ground truth, then the user cannot label this image. 
        # We delete the object from the conformal prediction, and return the image 
        if len(gt_ids) == 0:
            del abs_img["conf"][obj_id]
            return img
        # Select the ground truth object with maximal IOU with predicted object
        gt_id = max(gt_ids, key=lambda x: get_iou(abs_img["gt"][x]["bbox"], abs_img["conf"][obj_id]["bbox"]))
        if get_iou(abs_img["gt"][gt_id]["bbox"], abs_img["conf"][obj_id]["bbox"]) < MIN_IOU:
            # If there is NO matching ground truth label -- i.e., the predicted object does not exist in the image,
            # we again delete the object from the conformal prediction and return
            del abs_img["conf"][obj_id]
            return img

        if key == "Flag":
            # If the object DOES exist in ground truth, we set flag to True
            abs_img["conf"][obj_id]["Flag"] = True 
            abs_img["conf"][obj_id]["Flag_prob"] = 1
        else:
            # If the key is a specific attribute, we set the conformal prediction to the correct value
            gt_val = abs_img["gt"][gt_id][key] if key in abs_img["gt"][gt_id] else False
            abs_img["conf"][obj_id][key] = [gt_val]
            abs_img["conf"][obj_id][f"{key}_prob"] = {gt_val : 1, not gt_val :0 }