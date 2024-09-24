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
                img, obj_id, key = labelling_qs[q_index]
                abs_img = input_space[img]
                if img in current_qs and img not in skipped_inputs:
                    output = str(sorted(list([output for asked_q, output in examples if asked_q == img][0])))
                else:
                    output = None
                pruning_power = self.get_labelling_q_pruning_power(abs_img, obj_id, key, program_space, q_index, output, False)
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
            options = sorted([i for i, (j, _, _) in enumerate(labelling_qs) if j == self.backup_question_index])
            # TODO: does this ever happen ?
            if len(options) == 0:
                print(self.backup_question_index)
                raise TypeError 
            optimal_q_type = "label"
            optimal_q = options[0]
            optimal_answer_list = None

        if optimal_q_type == "label":
            img, obj_id, key = labelling_qs.pop(optimal_q)
            abs_img = input_space[img]
            skip = self.ask_labelling_question(abs_img, key, obj_id, img)
            if skip is not None:
                labelling_qs[:] = [(other_img, other_obj_id, other_key) for (other_img, other_obj_id, other_key) in labelling_qs if img != other_img or obj_id != other_obj_id]
            if img not in current_qs:
                return img
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


    # TODO: this should be domain agnostic
    def get_labelling_q_pruning_power(self, abs_img, obj_id, key, progs, q_index, output, partial_conf):
        answer_to_freq = {}
        for val in [True, False]:
            if key == "Flag":
                temp_obj = abs_img["conf"][obj_id]
                if val:
                    temp_obj["Flag"] = True 
                else:
                    del abs_img["conf"][obj_id]
            else:
                abs_img["conf"][obj_id][key] = [val]
            conf_list = self.get_all_universes(abs_img["conf"])
            # put it back
            if key == "Flag":
                abs_img["conf"][obj_id] = temp_obj
                temp_obj["Flag"] = False 
            else:
                abs_img["conf"][obj_id][key] = [True, False]

            if partial_conf:
                num_samples = min(MAX_PARTIAL_SAMPLES, int(len(conf_list) * PARTIAL_AMT)) if int(len(conf_list) * PARTIAL_AMT) > 0 else min(3, len(conf_list))  
                random.seed(123)
                conf_list = random.sample(conf_list, num_samples)
            for prog in progs:
                answers = self.interp.eval_consistent(prog, conf_list)
                for answer in answers:
                    if output is not None and output != answer:
                        continue
                    if (answer, val) not in answer_to_freq:
                        answer_to_freq[(answer, val)] = 0
                    answer_to_freq[(answer, val)] += 1
        answer_nums = sorted(list(answer_to_freq.values()), reverse=True)
        return PruningPowerInfo(answer_nums, q_index, "label", partial_conf)


    def get_all_qs_pruning_power(self, progs, input_qs, labelling_qs, examples, skipped_inputs, partial_conf=False):
        print(f"# INDIST INPS: {len(INDIST_INPS)}")
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        pruning_power_per_question = []

        # input questions
        for img in input_qs:
            if img in INDIST_INPS:
                continue
            if img in current_qs:
                continue 
            pruning_power = self.get_input_q_pruning_power(input_qs[img], partial_conf, progs, img)
            pruning_power_per_question.append(pruning_power)

        # labelling questions
        for q_index, (img, obj_id, key) in enumerate(labelling_qs):
            if img in INDIST_INPS:
                continue
            if img in skipped_inputs:
                continue
            abs_img = input_qs[img]
            output = None if img not in current_qs else str(sorted(list([output for asked_q, output in examples if asked_q == img][0])))
            pruning_power = self.get_labelling_q_pruning_power(abs_img, obj_id, key, progs, q_index, output, partial_conf)
            if len(pruning_power.answer_list) == 0:
                pruning_power = self.get_labelling_q_pruning_power(abs_img, obj_id, key, progs, q_index, output, False)
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
    

