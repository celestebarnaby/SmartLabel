from question_selection.question_selection import QuestionSelector
from question_selection.smart_label import SmartLabel
import random
import constants


class PLDIReviewerDRequest(QuestionSelector):
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
        print("selecting question!")
        print(f"# labelling questions: {len(labelling_qs)}")
        print(f"# asked input qs: {len(examples)}")
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        for prog in program_space:
            for inp_id, inp in input_space.items():
                if inp_id in skipped_inputs:
                    continue
                answers = self.interp.eval_consistent(prog, inp["conf_list"])
                for answer in answers:
                    # this means something was pruned from a prediction set during backward AI
                    if self.interp.eval_cce2(prog, inp, answer):
                        relevant_qs = [label_q for label_q in labelling_qs if inp_id == label_q.input_id]
                        if len(relevant_qs) == 0:
                            if inp_id in current_qs:
                                continue
                            else:
                                # i don't think this should happen either but whatever...
                                return inp_id, "input"
                        else:
                            selected_label_q = random.choice(labelling_qs)
                            # pop this q from list
                            labelling_qs[:] = [label_q for label_q in labelling_qs if label_q.input_id != selected_label_q.input_id or label_q.obj_id != selected_label_q.obj_id or label_q.attr_id != selected_label_q.attr_id]
                            selected_inp_id = selected_label_q.input_id 
                            obj_id = selected_label_q.obj_id 
                            key = selected_label_q.attr_id 
                            selected_inp = input_space[selected_inp_id] 
                            skip = self.interp.ask_labelling_question(selected_inp, key, obj_id, selected_inp_id)
                            selected_inp["conf_list"] = self.interp.get_all_universes2(selected_inp)
                            if skip is not None:
                                labelling_qs = [other_label_q for other_label_q in labelling_qs if selected_inp_id != other_label_q.input_id or obj_id != other_label_q.obj_id]
                            if selected_inp_id not in current_qs:
                                return selected_inp_id, "label_and_input" 
                            return None, "label"


        # if this doesn't work, default to picking a random question
        random.seed(constants.SEED)
        
        if len(labelling_qs) == 0:
            remaining_input_qs = [inp for inp in input_space if inp not in current_qs]
            return random.choice(remaining_input_qs), "input"
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
            return inp_id, "label_and_input" 
        return None, "label"
    

    def distinguish(self, program_space, input_qs, examples, skipped_inputs):
        return SmartLabel(self.interp).distinguish(program_space, input_qs, examples, skipped_inputs)