import random
from question_selection.smart_label import SmartLabel

class SmartLabelNoUB(SmartLabel):
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

        pruning_power_per_question = self.get_all_qs_pruning_power(program_space, input_space, labelling_qs, examples, skipped_inputs)
        pruning_power_per_question.sort()
        while True:
            best_q = pruning_power_per_question.pop(0)
            if best_q.answer_list[-1] == len(program_space) and len(pruning_power_per_question) > 0:
                continue
            q_index = best_q.q_index
            q_type = best_q.q_type
            optimal_answer_list = best_q.answer_list
            break

        if optimal_answer_list[-1] == len(program_space):
            # Just arbitrarily pick some label question where there is a weird universe
            options = sorted([i for i, label_q in enumerate(labelling_qs) if label_q.input_id == self.backup_question_index])
            if len(options) == 0:
                print(self.backup_question_index)
                raise TypeError 
            q_type = "label"
            q_index = options[0]
            optimal_answer_list = None

        if q_type == "label":
            label_q = labelling_qs.pop(q_index)
            inp_id = label_q.input_id
            obj_id = label_q.obj_id 
            attr_id = label_q.attr_id

            inp = input_space[inp_id]
            skip = self.ask_labelling_question(inp, attr_id, obj_id, inp)
            # Update conf_list 
            inp["conf_list"] = self.interp.get_all_universes(inp["conf"])
            if skip is not None:
                labelling_qs[:] = [other_labelling_q for other_labelling_q in labelling_qs if inp_id != other_labelling_q.input_id or obj_id != other_labelling_q.obj_id]
            if inp_id not in current_qs:
                return inp_id 
            return None 
        return q_index