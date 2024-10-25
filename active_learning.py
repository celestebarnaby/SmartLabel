import time 
import random
from abc import ABC, abstractmethod

from constants import *

# from image_search_domain.image_search_dsl import *
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
        INDIST_INPS[:] = []

        select_question_time = 0
        refine_hs_time = 0
        distinguish_time = 0
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
                    return "FAIL", time_per_round, skipped_inputs, refine_hs_time, select_question_time, distinguish_time
                print(f"Num programs in program space: {len(program_space)}")
                print("Checking distinguishability!")
                distinguish_start_time = time.perf_counter()
                finished = self.question_selection.distinguish(program_space, self.input_space, self.examples, skipped_inputs)
                distinguish_time += time.perf_counter() - distinguish_start_time
                if finished:
                    print("All programs indistinguishable! Active learning finished!")
                    print("Synthesized prog: {}".format(program_space[0]))
                    return program_space, time_per_round, skipped_inputs, refine_hs_time, select_question_time, distinguish_time
                # We seed here so that we sample the same programs for every benchmark/ablation
                random.seed(rounds)
                # Sample a subset of the program space
                samples = random.sample(program_space, min(self.num_samples, len(program_space)))
                select_question_start_time = time.perf_counter()
                new_input_question = self.question_selection.select_question(samples, self.input_space, self.labelling_qs, self.examples, skipped_inputs, self.semantics)
                select_question_time += time.perf_counter() - select_question_start_time
                # This will happen if the question is a labelling question
                if new_input_question is None:
                    pass 
                else: 
                    print(f"Asking input question with id: {new_input_question}")
                    # Get the ground truth answer to the question, and add to set of examples
                    new_answer = self.interp.eval_standard(self.gt_prog, self.input_space[new_input_question]["gt"]) 
                    self.add_example(new_input_question, new_answer, skipped_inputs)
                # Update the program space with new I/O examples
                refine_hs_start_time = time.perf_counter()
                program_space = self.question_selection.prune_program_space(program_space, [(self.input_space[q], a) for q, a in self.examples], self.semantics)
                refine_hs_time += time.perf_counter() - refine_hs_start_time
                rounds += 1
                time_per_round.append(time.perf_counter() - round_start_time)
        except TimeOutException:
            return "TIMEOUT", time_per_round, skipped_inputs, refine_hs_time, select_question_time, distinguish_time
        

    def add_example(self, new_question, new_answer, skipped_inputs):
        self.examples.append((new_question, new_answer))


class LabelQuestion:
    def __init__(self, input_id, obj_id, attr_id):
        self.input_id = input_id
        self.obj_id = obj_id
        self.attr_id = attr_id