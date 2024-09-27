import itertools
from constants import *
import random
from image_edit_domain.image_edit_dsl import *
from question_selection.question_selection import QuestionSelector

class LearnSy(QuestionSelector):
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
        best_obj = None 
        best_q = None

        # get objectives for input questions
        for inp, model in self.models.items():
            if inp in current_qs:
                continue
            obj = self.approximate_obj(model, program_space)
            if best_obj is None or obj < best_obj:
                best_obj = obj 
                best_q = inp
        return best_q      
    

    def approximate_obj(self, model, program_space):
        total = 0
        self_per_rule, cross_per_rule_pair = model
        for prog1, prog2 in itertools.combinations(program_space, 2):
            total += self.interp.apply_model(self_per_rule, cross_per_rule_pair, prog1, prog2) 
        return total 
    

    def learn_models(self, input_space, semantics, synthesizer):
        print("Learning models...")
        program_space = synthesizer.synthesize_for_learnsy()
        grammar_rule_to_progs = {}
        for prog in program_space:
            if prog.get_grammar_rule() not in grammar_rule_to_progs:
                grammar_rule_to_progs[prog.get_grammar_rule()] = []
            grammar_rule_to_progs[prog.get_grammar_rule()].append(prog)
        models = {}
        for inp_id, inp in input_space.items():
            cross_per_rule_pair = {}
            self_per_rule = {}
            for rule, progs_per_rule in grammar_rule_to_progs.items():
                if self.interp.no_children(rule):
                    self_per_rule[rule] = 1 
                    continue 
                n1 = 0
                n2 = 0
                random.seed(123)
                samples1 = random.sample(progs_per_rule, min(NUM_LEARNSY_SAMPLES, len(progs_per_rule)))
                samples2 = random.sample(progs_per_rule, min(NUM_LEARNSY_SAMPLES, len(progs_per_rule))) 
                for prog1, prog2 in zip(samples1, samples2):
                    if self.interp.subprogs_not_equal(prog1, prog2, inp, semantics):
                        n1 += 1
                        if self.interp.eval_standard(prog1, inp[semantics]) == self.interp.eval_standard(prog2, inp[semantics]):
                            n2 += 1 
                self_per_rule[rule] = n2/n1 if n1 > 0 else W_DEFAULT
            for rule1, rule2 in list(itertools.combinations(list(grammar_rule_to_progs.keys()), 2)):
                progs_per_rule1 = grammar_rule_to_progs[rule1]
                progs_per_rule2 = grammar_rule_to_progs[rule2]
                random.seed(123)
                samples1 = random.choices(progs_per_rule1, k=NUM_LEARNSY_SAMPLES)
                samples2 = random.choices(progs_per_rule2, k=NUM_LEARNSY_SAMPLES)
                n = 0
                for prog1, prog2 in zip(samples1, samples2):
                    prog1_output = self.interp.eval_standard(prog1, inp[semantics])
                    prog2_output = self.interp.eval_standard(prog2, inp[semantics])
                    if type(prog1_output) == type(prog2_output) and prog1_output == prog2_output:
                        n += 1
                cross_per_rule_pair[str(sorted([rule1, rule2]))] = n/NUM_LEARNSY_SAMPLES 
            models[inp_id] = (self_per_rule, cross_per_rule_pair)
        print("Done learning!")
        self.models = models
    


     