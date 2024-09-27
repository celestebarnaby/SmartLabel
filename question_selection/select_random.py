from question_selection.question_selection import QuestionSelector
from question_selection.samplesy import SampleSy
import random


class SelectRandom(QuestionSelector):
    def __init__(self, interpreter):
        super().__init__(interpreter)


    def select_question(
            self, 
            program_space, 
            input_space, 
            labelling_qs, 
            examples, 
            skipped_inputs, 
            semantics
            ):
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        random.seed(123)
        i = random.choice(list(range(len(labelling_qs))))
        img, obj_id, key = labelling_qs.pop(i)
        abs_img = input_space[img]
        skip = self.ask_labelling_question(abs_img, key, obj_id, img)
        if skip is not None:
            labelling_qs[:] = [(other_img, other_obj_id, other_key) for (other_img, other_obj_id, other_key) in labelling_qs if img != other_img or obj_id != other_obj_id]
        if img not in current_qs:
            return img 
        return None
    

    def distinguish(self, program_space, input_qs, examples, skipped_inputs):
        return SampleSy(self.interp).distinguish(program_space, input_qs, examples, skipped_inputs)