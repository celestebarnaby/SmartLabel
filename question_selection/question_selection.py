from abc import ABC, abstractmethod
from constants import *
from image_edit_utils import *
import itertools

class QuestionSelector(ABC):
    def __init__(self, interpreter):
        self.interp = interpreter 

    @abstractmethod
    def select_question(
            self, 
            program_space, 
            input_space, 
            labelling_qs, 
            examples, 
            skipped_inputs, 
            semantics
            ):
        pass

    # TODO: I don't think this can be domain agnostic... what do we do about that...
    def ask_labelling_question(self, abs_img, key, obj_id, img):

        # we must find the ground truth label that corresponds to the predicted label
        gt_ids = {gt_id for gt_id in abs_img["gt"] if abs_img["gt"][gt_id]["Label"] == abs_img["conf"][obj_id]["Label"]}

        # The object does not exist in the ground truth
        if len(gt_ids) == 0:
            del abs_img["conf"][obj_id]
            abs_img["conf_list"] = self.get_all_universes(abs_img["conf"])
            return img
        gt_id = max(gt_ids, key=lambda x: get_iou(abs_img["gt"][x]["bbox"], abs_img["conf"][obj_id]["bbox"]))
        if get_iou(abs_img["gt"][gt_id]["bbox"], abs_img["conf"][obj_id]["bbox"]) < MIN_IOU:
            # there is NO matching ground truth label -- i.e., the predicted object does not exist
            del abs_img["conf"][obj_id]
            abs_img["conf_list"] = self.get_all_universes(abs_img["conf"])
            return img

        if key == "Flag":
            # Object DOES exist in ground truth
            abs_img["conf"][obj_id]["Flag"] = True 
        else:
            gt_val = abs_img["gt"][gt_id][key] if key in abs_img["gt"][gt_id] else False
            abs_img["conf"][obj_id][key] = [gt_val]
        abs_img["conf_list"] = self.get_all_universes(abs_img["conf"])


    def learn_models(self, input_space, semantics, synthesizer):
        return {}


    def distinguish(self, program_space, input_qs, examples, skipped_inputs):
        # TODO: REMOVE LATER!!
        input_qs = sorted(list(input_qs.items()))
        for inp_id, inp in input_qs:
            if inp_id in INDIST_INPS:
                continue
            for universe in inp["conf_list"]:
                base_prog_output = self.interp.eval_standard(program_space[0], universe)
                for prog in program_space[1:]:
                    if self.interp.eval_standard(prog, universe) != base_prog_output:
                        self.backup_question_index = inp_id
                        return False 
            INDIST_INPS.append(inp_id)
        return True
    

    def prune_program_space(self, program_space, examples, semantics):
        new_program_space = []
        check = self.interp.get_check(semantics)
        for prog in program_space:
            if check(prog, examples):
                new_program_space.append(prog)
        return new_program_space
    

    # TODO: this should be domain agnostic
    def get_all_universes(self, abs_img_conf):
        abs_img_conf_copy = {}
        for key, val in abs_img_conf.items():
            abs_img_conf_copy[key] = [item for item in self.get_all_versions_of_object(val)]
        keys = list(abs_img_conf_copy.keys())
        vals = list(abs_img_conf_copy.values())
        all_lists = list(itertools.product(*vals))
        all_universes = [{keys[i]: l[i] for i in range(len(l)) if l[i] is not None} for l in all_lists]
        return all_universes


    # TODO: this should be domain agnostic
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




