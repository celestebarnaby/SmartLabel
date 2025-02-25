import time 
import random
from abc import ABC, abstractmethod

from constants import *

from image_edit_domain.image_edit_dsl import *
from utils import *

class ActiveLearning(ABC):
    def __init__(self, semantics, question_selection):
        '''
        This is the base class for active learning. Each domain requires its own subclass with its own interpreter, 
        synthesizer, benchmarks, and question space.
        '''
        self.semantics = semantics 

        self.question_selection = question_selection(None)

        self.set_benchmarks()
        self.set_interpreter()
        self.set_synthesizer()
        
        self.question_selection.interp = self.interp

    @abstractmethod 
    def set_interpreter(self):
        pass 

    @abstractmethod 
    def set_synthesizer(self):
        pass 

    @abstractmethod
    def set_question_space(self, benchmark):
        pass 

    def run(self, benchmark, program_space):
        '''
        Runs the active learning procedure, and outputs a single program.
        '''
        time_per_round = []
        num_label_qs = 0
        num_input_qs = 0
        INDIST_INPS[:] = []
        try:
            rounds = 1
            skipped_inputs = set()
            while True:
                print(f"Starting Round {rounds}!")
                round_start_time = time.perf_counter()

                # The program space may be empty only if the ground truth program's prediction set(s) did not contain the ground truth output(s)/
                # Under our conformal guarantee, this happens with low probability.
                if len(program_space) == 0:
                    print("Active learning failed.")
                    return "FAIL", time_per_round, num_label_qs, num_input_qs, skipped_inputs
                print(f"Num programs in program space: {len(program_space)}")
                print("Checking distinguishability!")
                finished = self.question_selection.distinguish(program_space, self.input_space, self.examples, skipped_inputs)
                if finished:
                    print("All programs indistinguishable! Active learning finished!")
                    print("Synthesized prog: {}".format(program_space[0]))
                    return program_space, time_per_round, num_label_qs, num_input_qs, skipped_inputs
                # We seed here so that we sample the same programs for every benchmark/ablation
                random.seed(rounds)
                # Sample a subset of the program space
                samples = random.sample(program_space, min(self.num_samples, len(program_space)))
                new_input_question, question_type = self.question_selection.select_question(samples, self.input_space, self.labelling_qs, self.examples, skipped_inputs, self.semantics)
                if question_type == "label":
                    num_label_qs += 1
                elif question_type == "input":
                    num_input_qs += 1 
                else:
                    num_label_qs += 1
                    num_input_qs += 1
                # This will happen if the question is a labelling question
                if new_input_question is None:
                    pass 
                else: 
                    print(f"Asking input question with id: {new_input_question}")
                    # Get the ground truth answer to the question, and add to set of examples
                    new_answer = self.interp.eval_standard(self.gt_prog, self.input_space[new_input_question]["gt"]) 
                    self.add_example(new_input_question, new_answer, skipped_inputs)
                # Update the program space with new I/O examples
                program_space = self.question_selection.prune_program_space(program_space, [(self.input_space[q], a) for q, a in self.examples], self.semantics)
                rounds += 1
                time_per_round.append(time.perf_counter() - round_start_time)
        except TimeOutException:
            return "TIMEOUT", time_per_round, num_label_qs, num_input_qs, skipped_inputs
        

    def add_example(self, new_question, new_answer, skipped_inputs):
        self.examples.append((new_question, new_answer))


class LabelQuestion:
    def __init__(self, input_id, obj_id, attr_id):
        self.input_id = input_id
        self.obj_id = obj_id
        self.attr_id = attr_id