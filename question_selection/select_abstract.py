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
        '''
        Selects the program with maximal ABSTRACT pruning power derived using BCE.
        An ablation in our evaluation.
        '''
        current_qs = [item[0] for item in examples] + list(skipped_inputs)

        # Computes only the partial pruning power of each question.
        pruning_power_per_question = self.get_all_qs_pruning_power(program_space, input_space, labelling_qs, examples, skipped_inputs, partial_conf=True)
        pruning_power_per_question.sort()
        if len(pruning_power_per_question) == 0:
            # TODO: error handling
            raise TypeError
        else:
            best_q = pruning_power_per_question[0]
            q_index = best_q.q_index
            q_type = best_q.q_type

        if q_type == "label":
            label_q = labelling_qs.pop(q_index)
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
        return q_index