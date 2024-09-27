from question_selection.question_selection import QuestionSelector
from constants import *
import random 

class SmartLabel(QuestionSelector):
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

        found_optimal_q = False
        concrete_pruning_powers = []
        while not found_optimal_q:
            possible_best_q = pruning_power_per_question.pop(0)
            q_index = possible_best_q.q_index
            q_type = possible_best_q.q_type 
            if q_type == "input":
                pruning_power = self.get_input_q_pruning_power(input_space[q_index], False, program_space, q_index)
            else:
                labelling_q = labelling_qs[q_index]
                inp_id = labelling_q.input_id 
                obj_id = labelling_q.obj_id
                attr_id = labelling_q.attr_id

                inp = input_space[inp_id]
                if inp_id in current_qs and inp_id not in skipped_inputs:
                    output = self.interp.represent_output([output for asked_q, output in examples if asked_q == inp_id][0])
                else:
                    output = None
                pruning_power = self.get_labelling_q_pruning_power(inp, obj_id, attr_id, program_space, q_index, output, False)
            concrete_pruning_powers.append(pruning_power)
            if pruning_power.answer_list[-1] == len(program_space) and len(pruning_power_per_question) > 0:
                continue 
            if len(pruning_power_per_question) == 0 or pruning_power < pruning_power_per_question[0]:
                found_optimal_q = True 
                concrete_pruning_powers.sort()

                while True:
                    best_q = concrete_pruning_powers.pop(0)
                    if best_q.answer_list[-1] == len(program_space) and len(concrete_pruning_powers) > 0:
                        continue
                    optimal_q = best_q.q_index
                    optimal_q_type = best_q.q_type
                    optimal_answer_list = best_q.answer_list
                    break
                
        if optimal_answer_list[-1] == len(program_space):
            # Just arbitrarily pick some label question where there is a weird universe
            options = sorted([i for i, label_q in enumerate(labelling_qs) if label_q.input_id == self.backup_question_index])
            # TODO: does this ever happen ?
            if len(options) == 0:
                print(f"Back up question index: {self.backup_question_index}")
                print(f"Current examples: {examples}")
                raise TypeError 
            optimal_q_type = "label"
            optimal_q = options[0]
            optimal_answer_list = None

        if optimal_q_type == "label":
            labelling_q = labelling_qs.pop(optimal_q)
            inp_id = labelling_q.input_id 
            obj_id = labelling_q.obj_id
            attr_id = labelling_q.attr_id

            inp = input_space[inp_id]
            skip = self.interp.ask_labelling_question(inp, attr_id, obj_id, inp_id)
            # Update conf_list 
            inp["conf_list"] = self.interp.get_all_universes(inp["conf"])
            if skip is not None:
                labelling_qs[:] = [other_labelling_q for other_labelling_q in labelling_qs if inp_id != other_labelling_q.input_id or obj_id != other_labelling_q.obj_id]
            if inp_id not in current_qs:
                return inp_id
            return None
        return optimal_q
    

    def get_input_q_pruning_power(self, q, partial_conf, progs, img):
        answer_to_freq = {}
        if partial_conf:
            num_samples = int(len(q["conf_list"]) * PARTIAL_AMT) if int(len(q["conf_list"]) * PARTIAL_AMT) > 0 else min(MIN_SAMPLES, len(q["conf_list"]))
            random.seed(123)
            conf_list = random.sample(q["conf_list"], num_samples)
        else:
            conf_list = q["conf_list"]
        for prog in progs:
            answers = self.interp.eval_consistent(prog, conf_list)  
            for answer in answers:
                if answer not in answer_to_freq:
                    answer_to_freq[answer] = 0
                answer_to_freq[answer] += 1

        answer_nums = sorted(list(answer_to_freq.values()), reverse=True)
        return PruningPowerInfo(answer_nums, img, "input", partial_conf)
    

    def get_labelling_q_pruning_power(self, inp, obj_id, key, progs, q_index, output, partial_conf):
        answers_to_labelling_q = self.interp.get_labelling_q_answers(inp, obj_id, key)
        answer_to_freq = {}
        for label_q_answer in answers_to_labelling_q:
            original_obj = self.interp.set_labelling_q_answer(inp["conf"], obj_id, key, label_q_answer)
            universes = self.interp.get_all_universes(inp["conf"])
            self.interp.reset_labelling_q(inp["conf"], obj_id, key, original_obj)
            if partial_conf:
                num_samples = self.interp.get_num_partial_conf_samples(len(universes))
                random.seed(123)
                universes = random.sample(universes, num_samples)
            for prog in progs:
                partial_conf_answers = self.interp.eval_consistent(prog, universes)
                for partial_conf_answer in partial_conf_answers:
                    if output is not None and output != partial_conf_answer:
                        continue
                    if (partial_conf_answer, label_q_answer) not in answer_to_freq:
                        answer_to_freq[(partial_conf_answer, label_q_answer)] = 0
                    answer_to_freq[(partial_conf_answer, label_q_answer)] += 1
        answer_nums = sorted(list(answer_to_freq.values()), reverse=True)
        return PruningPowerInfo(answer_nums, q_index, "label", partial_conf)


    def get_all_qs_pruning_power(self, progs, input_qs, labelling_qs, examples, skipped_inputs, partial_conf=False):
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        pruning_power_per_question = []

        # input questions
        for inp in input_qs:
            if inp in INDIST_INPS:
                continue
            if inp in current_qs:
                continue 
            pruning_power = self.get_input_q_pruning_power(input_qs[inp], partial_conf, progs, inp)
            pruning_power_per_question.append(pruning_power)

        # labelling questions
        for q_index, label_q in enumerate(labelling_qs):
            if label_q.input_id in INDIST_INPS:
                continue
            if label_q.input_id in skipped_inputs:
                continue
            inp = input_qs[label_q.input_id]
            output = None if label_q.input_id not in current_qs else self.interp.represent_output([output for asked_q, output in examples if asked_q == label_q.input_id][0])
            pruning_power = self.get_labelling_q_pruning_power(inp, label_q.obj_id, label_q.attr_id, progs, q_index, output, partial_conf)
            if len(pruning_power.answer_list) == 0:
                pruning_power = self.get_labelling_q_pruning_power(inp, label_q.obj_id, label_q.attr_id, progs, q_index, output, False)
            if len(pruning_power.answer_list) == 0:
                continue
            pruning_power_per_question.append(pruning_power)

        return pruning_power_per_question
    

class PruningPowerInfo:
    def __init__(self, answer_list, q_index, q_type, partial_conf):
        self.answer_list = answer_list 
        self.q_index = q_index 
        self.q_type = q_type 
        self.partial_conf = partial_conf

    def __lt__(x, y):
        # check if the WORST ANSWERS are equal
        if x.answer_list[0] == y.answer_list[0]:
            # if ONLY ONE is partial_conf 
            if x.partial_conf and not y.partial_conf:
                return False
            elif y.partial_conf and not x.partial_conf:
                return True
            # if the answer list AND the question type are the same, fall back to question index
            if x.q_type == y.q_type:
                return x.q_index < y.q_index 
            # if the answer lists are the same, prioritize the input question
            return x.q_type == "label"
        # if the answer lists are different, prioritize whichever one has more pruning power
        return more_pruning_power(x.answer_list, y.answer_list)



# does l1 have more pruning power than l2?
def more_pruning_power(l1, l2):
    i = 0
    while i < min(len(l1), len(l2)):
        # TODO is this ok
        if l1[i] == 0:
            return False
        elif l2[i] == 0:
            return True
        elif l1[i] < l2[i]:
            return True 
        elif l1[i] > l2[i]:
            return False 
        i += 1
    if i < len(l1):
        return True 
    else:
        return False
    

