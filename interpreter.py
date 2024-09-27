from abc import ABC, abstractmethod


class Interpreter(ABC):
    @abstractmethod
    def forward_ai(self, expr, inp):
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
        pass

    @abstractmethod
    def eval_standard(
        self,
        expr,
        inp
    ):
        pass 


    @abstractmethod
    def no_children(self, rule):
        pass

    @abstractmethod 
    def gt_matches_abs_output(self, gt_output, abs_value):
        pass

    def matches_constraints(inp, constraints):
        return True

    def eval_cce(
        self,
        expr,
        inp,
        gt_output  
    ):
        inp_conf = inp["conf"]
        self.forward_ai(expr, inp_conf)
        if not self.gt_matches_abs_output(gt_output, expr.abs_value):
            return False
        constraints = {}
        # if not self.backward_ai(expr, inp_conf, gt_output, gt_output, constraints):
            # return False
        inp_conf_list = inp["conf_list"]
        output_concrete = self.eval_consistent(expr, inp_conf_list, gt_output=self.represent_output(gt_output), constraints=constraints)
        if self.represent_output(gt_output) in output_concrete:
            return True
        return False
    

    def represent_output(self, output):
        return output


    def eval_consistent(
            self,
            expr, 
            inp_conf_list, 
            gt_output=None, 
            constraints={}):
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
    def subprogs_not_equal(self, prog1, prog2, inp, semantics):
        pass 

    @abstractmethod
    def get_all_universes(self, inp):
        pass

    @abstractmethod 
    def get_labelling_q_answers(self, inp, obj_id, key):
        pass 

    @abstractmethod
    def set_labelling_q_answer(self, inp, obj_id, key, answer):
        pass 

    @abstractmethod
    def reset_labelling_q(self, inp, obj_id, key, original_obj):
        pass 

    @abstractmethod 
    def get_num_partial_conf_samples(self, num_universes):
        pass

    def get_check(self, semantics):
        semantics_to_check = {
            "CCE-NoAbs" : self.check_prog_cce_no_abs,
            "CCE" : self.check_prog_cce,
            "standard" : self.check_prog_standard
        }
        return semantics_to_check[semantics]
    

    def check_prog_cce_no_abs(self, prog, examples):
        for inp, output in examples:
            output_rep = self.represent_output(output)
            prog_output = self.eval_consistent(prog, inp["conf_list"], gt_output=output_rep)
            if output_rep not in prog_output:
                return False 
        return True 


    def check_prog_cce(self, prog, examples):
        for inp, output in examples:
            result = self.eval_cce(prog, inp, output)
            if not result:
                return False 
        return True


    def check_prog_standard(self, prog, examples):
        for inp, output in examples:
            result = self.eval_standard(prog.duplicate(), inp["standard"])
            if result != output:
                return False
        return True


    def check_gt_equivalence(self, gt_prog, prog, input_qs, skipped_inputs):
        for id_, inp in input_qs.items():
            if id_ in skipped_inputs:
                continue
            if self.eval_standard(prog, inp["gt"]) != self.eval_standard(gt_prog, inp["gt"]):
                return False
        return True
