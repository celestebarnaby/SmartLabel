from question_selection.smart_label import SmartLabel


class SelectAbstract(SmartLabel):
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

        pruning_power_per_question = self.get_all_qs_pruning_power(program_space, input_space, labelling_qs, examples, skipped_inputs, partial_conf=True)
        pruning_power_per_question.sort()
        best_q = pruning_power_per_question[0]
        q_index = best_q.q_index
        q_type = best_q.q_type

        print("Best question num: {}".format(best_q.answer_list))
        print("Optimal question index: {}".format(q_index))
        print(f"Question type: {q_type}")
        print(f"# questions: {len(pruning_power_per_question)}")
        print(f"# labelling questions: {len(labelling_qs)}")
        print(f"# input q's: {len(input_space)}")
        print([item.answer_list for item in pruning_power_per_question[:5]])
        if q_type == "label":
            img, obj_id, key = labelling_qs.pop(q_index)
            print(f"Label question img: {img}")
            print(f"Label question obj id: {obj_id}")
            print(f"Label question key: {key}")
            abs_img = input_space[img]
            skip = self.ask_labelling_question(abs_img, key, obj_id, img)
            if skip is not None:
                labelling_qs[:] = [(other_img, other_obj_id, other_key) for (other_img, other_obj_id, other_key) in labelling_qs if img != other_img or obj_id != other_obj_id]
            if img not in current_qs:
                return img 
            return None 
        return q_index