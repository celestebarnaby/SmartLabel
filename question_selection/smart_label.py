from question_selection.question_selection import QuestionSelector
from constants import *
import constants
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
        '''
        The question selection algorithm described in the paper. Selects the question with optimal pruning power. First, computes the partial
        pruning power of each question using BCE, and then computes the complete pruning power of of questions enumeratively until the 
        optimal question is found.
        '''
        current_qs = [item[0] for item in examples] + list(skipped_inputs)

        # Get partial pruning power of each question.
        pruning_power_per_question = self.get_all_qs_pruning_power(program_space, input_space, labelling_qs, examples, skipped_inputs, partial_conf=True)
        pruning_power_per_question.sort()

        found_optimal_q = False
        concrete_pruning_powers = []

        while not found_optimal_q:

            # In each iteration, pop the question with optimal upper bound on pruning power.
            possible_best_q = pruning_power_per_question.pop(0)
            q_index = possible_best_q.q_index
            q_type = possible_best_q.q_type 

            # Compute the complete pruning power of just that question.
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

            # If the best concrete pruning power we have seen so far is greater than other upper bounds and concrete pruning powers,
            # This MUST be the optimal question.
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
            # It is possible that all questions have 0 pruning power, since we are sampling a subset of the program space.
            # If this happens, select a "backup" question that will prune at least 1 program.
            options = sorted([i for i, label_q in enumerate(labelling_qs) if label_q.input_id == self.backup_question_index])
            print("Using backup question!")
            print(f"Num options: {len(options)}")
            if len(options) == 0:
                optimal_q_type = "input"
                q_index = self.backup_question_index
            else:
                optimal_q_type = "label"
                optimal_q = options[0]
                optimal_answer_list = None

        # Ask the optimal question
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
    

    def get_input_q_pruning_power(self, q, partial_conf, progs, q_index):
        '''
        Compute the pruning power of an input question. 
        '''
        answer_to_freq = {}
        # Under BCE, sample just a subset of the universes of the input
        if partial_conf:
            num_samples = self.interp.get_num_partial_conf_samples(len(q["conf_list"]))
            random.seed(constants.SEED)
            conf_list = random.sample(q["conf_list"], num_samples)
        # Otherwise, consider all universes
        else:
            conf_list = q["conf_list"]
        for prog in progs:
            answers = self.interp.eval_consistent(prog, conf_list)  
            for answer in answers:
                if answer not in answer_to_freq:
                    answer_to_freq[answer] = 0
                answer_to_freq[answer] += 1

        # Sort answer frequencies. A question's pruning power is proportional to how many programs its WORST answer will prune. 
        answer_nums = sorted(list(answer_to_freq.values()), reverse=True)
        return PruningPowerInfo(answer_nums, q_index, "input", partial_conf)
    
    
    def get_labelling_q_pruning_power(self, inp, obj_id, key, progs, q_index, output, partial_conf):
        '''
        Compute the pruning power of a labelling question
        '''

        # Get all potential answers to the question
        answers_to_labelling_q = self.interp.get_labelling_q_answers(inp, obj_id, key)
        answer_to_freq = {}

        # Consider the answers we would get to the input question for each potential answer to the labelling question.
        for label_q_answer in answers_to_labelling_q:
            original_obj = self.interp.set_labelling_q_answer(inp["conf"], obj_id, key, label_q_answer)
            universes = self.interp.get_all_universes(inp["conf"])
            self.interp.reset_labelling_q(inp["conf"], obj_id, key, original_obj)
            if partial_conf:
                num_samples = self.interp.get_num_partial_conf_samples(len(universes))
                random.seed(constants.SEED)
                universes = random.sample(universes, num_samples)
            for prog in progs:
                input_q_answers = self.interp.eval_consistent(prog, universes)
                for input_q_answer in input_q_answers:
                    if output is not None and output != input_q_answer:
                        continue
                    if (input_q_answer, label_q_answer) not in answer_to_freq:
                        answer_to_freq[(input_q_answer, label_q_answer)] = 0
                    answer_to_freq[(input_q_answer, label_q_answer)] += 1
        answer_nums = sorted(list(answer_to_freq.values()), reverse=True)
        return PruningPowerInfo(answer_nums, q_index, "label", partial_conf)


    def get_all_qs_pruning_power(self, progs, input_qs, labelling_qs, examples, skipped_inputs, partial_conf=False):
        '''
        Compute the pruning powers of all questions. If partial_conf is True, use BCE.
        '''
        current_qs = [item[0] for item in examples] + list(skipped_inputs)
        pruning_power_per_question = []

        # First, get pruning powers of all input questions
        for inp in input_qs:
            if inp in INDIST_INPS:
                continue
            if inp in current_qs:
                continue 
            pruning_power = self.get_input_q_pruning_power(input_qs[inp], partial_conf, progs, inp)
            pruning_power_per_question.append(pruning_power)

        # Then get pruning powers of all labelling questions
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
        """
        Comparing the pruning powers of 2 questions.
        """

        # The case that both questions have EQUIVALENT answer lists
        if x.answer_list[0] == y.answer_list[0]:
            # If only ONE pruning power if partial, the complete pruning power is greater
            if x.partial_conf and not y.partial_conf:
                return False
            # If both questions also have the same type, defer to index
            if x.q_type == y.q_type:
                return x.q_index < y.q_index 
            # If the questions have different types, the label question is greater
            return x.q_type == "label"
        # If the answer lists are different, compare and return the question with the better answer list
        return x.answer_list[0] < y.answer_list[0]
