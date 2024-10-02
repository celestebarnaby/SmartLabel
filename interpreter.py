from abc import ABC, abstractmethod


class Interpreter(ABC):
    @abstractmethod
    def forward_ai(self, expr, inp):
        '''
        Perform forward AI on an expression and an input. 
        Returns: the abstract output of the expression (type varies by domain)
        '''
        pass

    @abstractmethod
    def backward_ai(
        self,
        expr,
        input, 
        goal_under,
        goal_over, 
        constraints
    ):
        '''
        Perform backward AI on an expression and input
        Args:
            expr: an expression in the target DSL
            input: the input
            goal_under: the under-approximated goal output of the expression. For the base program, 
                        this will be the output in the user-provided I/O example. For subexpressions,
                        the approximation will be derived from the backwards semantics of the subexpression.
            goal_output: same as above, except it's an over-approximation
            constraints: a mapping from components of the input to constraints derived during backwards AI.
                         We currently only use the in the Image Editing domain - see that implementation for
                        details.

        Returns: False if it is impossible for the expression to output the goal, True otherwise
        '''
        pass

    @abstractmethod
    def eval_standard(
        self,
        expr,
        inp
    ):
        '''
        Evaluates expr on inp under the standard semantics of the DSL
        '''
        pass 


    @abstractmethod
    def no_children(self, rule):
        '''
        Checks whether a grammar rule has children. Used for LearnSy only.
        '''
        pass

    @abstractmethod 
    def gt_matches_abs_output(self, gt_output, abs_value):
        '''
        Checks whether a goal output matches the abstract output derived during forward AI.
        '''
        pass

    def matches_constraints(inp, constraints):
        '''
        Checks whether the constraints derived during backward AI are consistent with the input.
        Returns True by default, but the image editing domain has its own implementation.
        '''
        return True

    def eval_cce(
        self,
        expr,
        inp,
        gt_output  
    ):
        """
        Evaluates an expr on an input w.r.t. a goal output using constrained conformal evluation (CCE)

        args:
            expr: an expression in the target DSL
            inp: an input in the input space
            gt_output: the goal output of expr on inp

        Returns:
            True if gt_output is contained in the prediction set output by expr on inp, and False otherwise
        """
        # Perform forward abstract interpretation
        self.forward_ai(expr, inp["conf"])
        # If the goal output is not contained in the abstract value of the expression, return False
        if not self.gt_matches_abs_output(gt_output, expr.abs_value):
            return False
        constraints = {}
        # Perform backward abstract interpretation. If expr could not possibly output the goal, return False
        inp_copy = inp.copy()
        if not self.backward_ai(expr, inp_copy["conf"], gt_output, gt_output, constraints):
            return False
        # TODO: this is not really what we should be doing in MNIST...
        inp_conf_list = inp_copy["conf_list"]
        # Compute the complete prediction set of the expression on the input.
        complete_pred_set = self.eval_consistent(expr, inp_conf_list, gt_output=self.represent_output(gt_output), constraints=constraints)
        # If the goal is contained in the prediction set, return True. Otherwise return False
        if self.represent_output(gt_output) in complete_pred_set:
            return True
        return False
    

    def represent_output(self, output):
        '''
        Outputs a representation of an output. Returns the output itself by default, but the image editing domain
        has its own implementation as inputs in this domain can be large dictionaries.
        '''
        return output


    def eval_consistent(
            self,
            expr, 
            inp_conf_list, 
            gt_output=None, 
            constraints={}):
        """
        Computes the set of outputs consistent with the constraints for a given expression and input

        Args:
            expr: an expression in the target DSL
            inp_conf_list: the conformal list of in input
            gt_output (optional): the goal output
            constraints (optional): a set of constraints on the values of the neural components of the input

        Returns: a set of outputs
        """
        res_set = set()
        for inp in inp_conf_list:
            if constraints and not self.matches_constraints(inp, constraints):
                continue
            res = self.eval_standard(expr, inp)
            res_rep = self.represent_output(res)
            res_set.add(res_rep)
            if gt_output == res_rep:
                break
        return res_set


    @abstractmethod 
    def subprogs_not_equal(self, prog1, prog2, inp):
        '''
        Checks whether two programs have the same output on a given input. For LearnSy only.
        '''
        pass 

    @abstractmethod
    def get_all_universes(self, inp):
        '''
        Outputs all "universes" of an input under conformal semantics. A universe is a possible
        ground truth labelling of an input, under the assumption that a prediction set contains the ground truth label.
        For instance, suppose an input a list of 3 MNIST digits:
        [img1, img2, img3]
        And suppose the prediction sets computed under conformal semantics of each image are is follows:
        [{1, 2}, {3}, {4}]
        Then this input has the following TWO universes:
        [1, 3, 4]
        [2, 3, 4]
        '''
        pass

    @abstractmethod 
    def get_labelling_q_answers(self, inp, obj_id, key):
        '''
        Given an input, a specific object in the input, and an optional "key" of that object, 
        return all possible answers to the corresponding labelling question.

        Examples:"
        1. In the MNIST domain, suppose the obj_id is "img-list" and the key is 1 (i.e. a specific index of the list).
           Then the set of possible answers is the set of integers in the i'th prediction set of the img-list.
        2. In the image editing domain, all labelling questions are binary (e.g. an object DOES or DOES NOT have a specific attribute).
        '''
        pass 

    @abstractmethod
    def set_labelling_q_answer(self, inp, obj_id, key, answer):
        '''
        Given an answer to a labelling question, update the input to replace the corresponding prediction set with the answer,
        and return the updated input.
        '''
        pass 

    @abstractmethod
    def reset_labelling_q(self, inp, obj_id, key, original_obj):
        '''
        Given an input where a prediction set has been replaced by a potential answer, reset to the original prediction set.
        '''
        pass 

    @abstractmethod 
    def get_num_partial_conf_samples(self, num_universes):
        '''
        Return the number of samples taken during partial conformal evaluation (domain specific)
        '''
        pass

    def get_check(self, semantics):
        semantics_to_check = {
            "CCE-NoAbs" : self.check_prog_cce_no_abs,
            "CCE" : self.check_prog_cce,
            "standard" : self.check_prog_standard
        }
        return semantics_to_check[semantics]
    

    def check_prog_cce_no_abs(self, prog, examples):
        '''
        Checks whether a given program matches a set of IO examples under conformal semantics, WITHOUT abstract interpretation.
        This corresponding to the CCE-NoAbs ablation in our evaluation.
        '''
        for inp, output in examples:
            output_rep = self.represent_output(output)
            prog_output = self.eval_consistent(prog, inp["conf_list"], gt_output=output_rep)
            if output_rep not in prog_output:
                return False 
        return True 


    def check_prog_cce(self, prog, examples):
        '''
        Checks whether a program matches a set of IO examples under conformal semantics,
        using our CCE technique with bidirectional abstract reasoning.
        '''
        for inp, output in examples:
            result = self.eval_cce(prog.duplicate(), inp, output)
            if not result:
                return False 
        return True


    def check_prog_standard(self, prog, examples):
        '''
        Checks whether a program matches a set of IO examples under standard semantics
        '''
        for inp, output in examples:
            result = self.eval_standard(prog.duplicate(), inp["standard"])
            if result != output:
                return False
        return True


    def check_gt_equivalence(self, gt_prog, prog, input_qs, skipped_inputs):
        '''
        Checks whether the program output by active learning is observationally equivalent
        to the benchmark's ground truth program w.r.t. the input space.
        '''
        for id_, inp in input_qs.items():
            if id_ in skipped_inputs:
                continue
            if self.eval_standard(prog, inp["gt"]) != self.eval_standard(gt_prog, inp["gt"]):
                return False
        return True
