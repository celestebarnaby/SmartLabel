from question_selection.question_selection import QuestionSelector

class SampleSy(QuestionSelector):
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
        best_question = None 
        best_question_nums = None 
        for id_, q in input_space.items():
            if id_ in current_qs:
                continue
            answer_to_num_progs = {}
            for prog in program_space:
                answer = str(sorted(list(self.interp.eval_standard(prog, q[semantics]))))
                if answer not in answer_to_num_progs:
                    answer_to_num_progs[answer] = 1
                else:
                    answer_to_num_progs[answer] += 1
            worst_answer_nums = sorted(answer_to_num_progs.values(), reverse=True)
            if best_question_nums is None or worst_answer_nums[0] < best_question_nums[0]:
                best_question_nums = worst_answer_nums
                best_question = id_
        return best_question 
    

    def distinguish(self, program_space, input_qs, examples, skipped_inputs):
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        questions_and_outputs = []
        for prog in program_space[1:]:
            for img, abs_img in input_qs.items():
                if img in current_qs:
                    continue 
                if self.interp.eval_standard(prog, abs_img["standard"]) != self.interp.eval_standard(program_space[0], abs_img["standard"]):
                    return False
                questions_and_outputs.append((abs_img, self.interp.eval_standard(program_space[0], abs_img["standard"])))
        return True  