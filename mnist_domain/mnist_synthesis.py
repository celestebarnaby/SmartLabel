import itertools

from synthesis import Synthesizer 
from constants import *

from mnist_domain.mnist_interpreter import MNISTInterpreter
from mnist_domain.mnist_dsl import Expr, get_grammar

class MNISTSynthesizer(Synthesizer):
    def __init__(self, semantics):
        super().__init__(semantics)
        self.worklist = []
        self.interp = MNISTInterpreter()
        self.program_counter = itertools.count(0)

    def synthesize(self, examples):
        grammar = get_grammar(START_SYMBOL)
        check = self.interp.get_check(self.semantics)
        _, program_space = self.synth_helper(MNIST_AST_DEPTH, grammar, grammar.start, examples, check)
        return program_space
    
    def synthesize_for_learnsy(self):
        '''
        Returns ALL possible programs and subprograms, not just ones that return integers. Used to learn models for LearnSy.
        '''
        grammar = get_grammar(START_SYMBOL)
        return sum([self.synth_helper(MNIST_AST_DEPTH, grammar, symb, [], lambda x, y: True)[0] for symb in {'list-int', 'int', 'const-int', 'int->int', 'int->bool', 'int->int->int', 'int->int->bool', 'pimg-int->int', 'list-pimg-int', 'pimg-int', 'const-int'}], [])
    
    # TODO: can we make this faster?
    def synth_helper(
            self,
            depth, 
            grammar, 
            symb, 
            examples, 
            check):
        if depth < 0:
            return [], [], 0
        exprs = []
        exprs_matching_examples = []
        for rule in grammar.rules[symb]:
            if len(rule) == 1:
                expr = Expr(rule[0])
                exprs.append(expr)
                if symb == "int":
                    if check(expr, examples):
                        exprs_matching_examples.append(expr)
            else:
                for children in itertools.product(
                    *[
                        self.synth_helper(depth - 1, grammar, child, examples, check)[0]
                        for child in rule[1:]
                    ]
                ):
                    expr = Expr(rule[0], children)
                    exprs.append(expr)
                    if symb == "int":
                        # print('asdfasdf')
                        # print(examples)
                        if check(expr, examples):
                            exprs_matching_examples.append(expr)
        return exprs, exprs_matching_examples
