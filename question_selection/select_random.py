from question_selection.question_selection import QuestionSelector
from question_selection.smart_label import SmartLabel
import random
import constants


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
        """
        In the SelectRandom question, we simply select a random, unanswered labelling question.
        If the input question corresponding to the labelling question has also not been answer,
        we return that input question as well.
        """
        random.seed(constants.SEED)
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        if len(labelling_qs) == 0:
            remaining_input_qs = [inp for inp in input_space if inp not in current_qs]
            return random.choice(remaining_input_qs)
        i = random.choice(list(range(len(labelling_qs))))
        label_q = labelling_qs.pop(i)
        inp_id = label_q.input_id 
        obj_id = label_q.obj_id 
        key = label_q.attr_id 
        inp = input_space[inp_id]
        skip = self.interp.ask_labelling_question(inp, key, obj_id, inp_id)
        # Update conf_list 
        inp["conf_list"] = self.interp.get_all_universes(inp["conf"])
        if skip is not None:
            labelling_qs[:] = [other_label_q for other_label_q in labelling_qs if inp_id != other_label_q.input_id or obj_id != other_label_q.obj_id]
        if inp_id not in current_qs:
            return inp_id 
        return None
    

    def distinguish(self, program_space, input_qs, examples, skipped_inputs):
        return SmartLabel(self.interp).distinguish(program_space, input_qs, examples, skipped_inputs)