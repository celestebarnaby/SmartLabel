import time 
import random
from image_edit_dsl import *
from image_edit_utils import *
from constants import *
from abc import ABC, abstractmethod


class ActiveLearning(ABC):
    def __init__(self, semantics, question_selection):
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
        self.set_question_space(benchmark)
        time_per_round = []
        INDIST_INPS[:] = []
        try:
            rounds = 1
            skipped_inputs = set()
            while True:
                print(f"Starting Round {rounds}!")
                round_start_time = time.perf_counter()
                if len(program_space) == 0:
                    print("Active learning failed. ")
                    return "FAIL", time_per_round, skipped_inputs
                print(f"Num programs in program space: {len(program_space)}")
                print("Checking distinguishability!")
                finished = self.question_selection.distinguish(program_space, self.input_space, self.examples, skipped_inputs)
                if finished:
                    print("All programs indistinguishable! Active learning finished!")
                    print("Synthesized prog: {}".format(program_space[0]))
                    return program_space, time_per_round, skipped_inputs
                random.seed(123 + rounds)
                samples = random.sample(program_space, min(NUM_SAMPLES, len(program_space)))
                print("Starting question selector!")
                new_input_question = self.question_selection.select_question(samples, self.input_space, self.labelling_qs, self.examples, skipped_inputs, self.semantics)
                # this will happen if the question is a labelling question
                if new_input_question is None:
                    pass 
                else: 
                    print(f"New input question: {new_input_question}")
                    # get the ground truth answer to the question
                    new_answer = self.interp.eval_standard(self.gt_prog, self.input_space[new_input_question]["gt"]) 
                    self.add_example(new_input_question, new_answer, skipped_inputs)
                program_space = self.question_selection.prune_program_space(program_space, [(self.input_space[q], a) for q, a in self.examples], self.semantics)
                rounds += 1
                time_per_round.append(time.perf_counter() - round_start_time)
                print()
        except TimeOutException:
            print("TIMEOUT")
            return "TIMEOUT", time_per_round, skipped_inputs
        

    def add_example(self, new_question, new_answer, skipped_inputs):
        self.examples.append((new_question, new_answer))