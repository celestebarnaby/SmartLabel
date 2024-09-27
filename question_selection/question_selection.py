from abc import ABC, abstractmethod
from constants import *
from image_edit_domain.image_edit_utils import *
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
    






