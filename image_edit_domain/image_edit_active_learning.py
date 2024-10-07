import os
import json
import random

from constants import * 
from active_learning import ActiveLearning, LabelQuestion

from image_edit_domain.image_edit_utils import * 
from image_edit_domain.image_edit_interpreter import ImageEditInterpreter
from image_edit_domain.image_edit_synthesis import ImageEditSynthesizer
from image_edit_domain.image_edit_benchmarks import image_edit_benchmarks


class ImageEditActiveLearning(ActiveLearning):
    def __init__(self, semantics, question_selection):
        super().__init__(semantics, question_selection)

    def set_benchmarks(self):
        self.benchmarks = image_edit_benchmarks 
    
    def set_synthesizer(self):
        self.synth = ImageEditSynthesizer(self.semantics)
    
    def set_interpreter(self):
        self.interp = ImageEditInterpreter()

    def set_question_space(self, benchmark, i):
        print("Loading images...")

        dataset_dir = IMG_EDIT_DIR.format(benchmark.dataset_name)
        if os.path.exists(dataset_dir):
            with open(dataset_dir, "r") as fp:
                preprocessed_images = json.load(fp)
                all_images = preprocessed_images
        else:
            # TODO: error handling
            raise TypeError 
        input_space = {img : abs_img for img, abs_img in all_images.items() if len(abs_img["conf_list"]) <= MAX_PRED_SET_SIZE}
        examples = self.get_examples(benchmark.gt_prog, all_images)
        for inp, _ in examples:
            input_space[inp] = all_images[inp]
        labelling_qs = self.get_labelling_qs(input_space)
        self.input_space = input_space 
        self.examples = examples 
        self.labelling_qs = labelling_qs
        self.synth.set_object_list(self.input_space)
        self.gt_prog = benchmark.gt_prog


    def get_examples(self, gt_prog, all_images):
        examples = []
        used_imgs = set()
        while len(examples) < NUM_INITIAL_EXAMPLES:
            if len(set(all_images.keys()) - used_imgs) == 0:
                return examples
            random.seed(123)
            inp = random.choice(sorted(list(set(all_images.keys()) - used_imgs)))
            used_imgs.add(inp)
            if len(all_images[inp]["conf_list"]) > MAX_PRED_SET_SIZE:
                continue
            gt_output = self.interp.eval_standard(gt_prog, all_images[inp]["gt"])
            key = "conf" if self.semantics in {"CCE", "CCE-NoAbs"} else self.semantics
            output = self.get_pred_output(gt_output, all_images[inp]["gt"], all_images[inp][key])
            if output is None or len(output) == 0:
                continue
            examples.append((inp, output)) 
        return examples
    

    def get_labelling_qs(self, input_qs):
        label_qs = []
        for img, abs_img in input_qs.items():
            for obj_id, obj in abs_img["conf"].items():
                if not obj["Flag"]:
                    label_qs.append(LabelQuestion(img, obj_id, "Flag"))
                for attr in ATTRIBUTES: 
                    if attr in obj and len(obj[attr]) == 2:
                        label_qs.append(LabelQuestion(img, obj_id, attr))
        return label_qs


    # During active learning, the user is presented with an image and asked to annotate the objects
    # in the image that they wish to apply an action to. The objects they annotate correspond to ground truth labels.
    # We must identify the predicted labels that these ground truth labels correspond to. 
    def get_pred_output(self, gt_output, gt_abs_img, abs_img):
        pred_output = set()
        for gt_id in gt_output: 
            l = {pred_id for pred_id in abs_img if (pred_id not in pred_output) and (abs_img[pred_id]["Label"] == gt_abs_img[gt_id]["Label"])}
            if len(l) == 0:
                return None
            pred_id = max({pred_id for pred_id in abs_img if (pred_id not in pred_output) and (abs_img[pred_id]["Label"] == gt_abs_img[gt_id]["Label"])}, key=lambda x: get_iou(abs_img[x]["bbox"], gt_abs_img[gt_id]["bbox"]))
            if get_iou(gt_abs_img[gt_id]["bbox"], abs_img[pred_id]["bbox"]) < MIN_IOU:
                return None 
            pred_output.add(pred_id)
        return pred_output
    

    def add_example(self, new_question, new_answer, skipped_inputs):
        semantics_key = "conf" if self.semantics in {"CCE", "CCE-NoAbs"} else self.semantics
        new_pred_output = self.get_pred_output(new_answer, self.input_space[new_question]["gt"], self.input_space[new_question][semantics_key])
        if new_pred_output is None:
            print("Skipping question")
            labelling_qs = [(img, obj_id, key) for (img, obj_id, key) in labelling_qs if img != new_question]
            # TODO: make sure this works
            skipped_inputs.add(new_question)
        else:
            print("New answer: {}".format(new_pred_output))
            self.examples.append((new_question, new_pred_output))