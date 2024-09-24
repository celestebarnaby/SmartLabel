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
            options = sorted([i for i, (j, _, _) in enumerate(labelling_qs) if j == self.backup_question_index])
            if len(options) == 0:
                print(self.backup_question_index)
                raise TypeError 
            q_type = "label"
            q_index = options[0]
            optimal_answer_list = None

        if q_type == "label":
            img, obj_id, key = labelling_qs.pop(q_index)
            abs_img = input_space[img]
            skip = self.ask_labelling_question(abs_img, key, obj_id, img)
            if skip is not None:
                labelling_qs[:] = [(other_img, other_obj_id, other_key) for (other_img, other_obj_id, other_key) in labelling_qs if img != other_img or obj_id != other_obj_id]
            if img not in current_qs:
                return img 
            return None 
        return q_index