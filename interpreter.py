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
    def eval_consistent(
        self,
        expr, 
        input_list,
        gt_output=None,
        constraints={}
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
    def matches_constraints(inp, constraints):
        pass


    def eval_cce(
        self,
        expr,
        inp,
        gt_output  
    ):
        inp_conf = inp["conf"]
        self.forward_ai(expr, inp_conf)
        if not check_output_abs(gt_output, expr.abs_value):
            return False
        constraints = {}
        if not self.backward_ai(expr, inp_conf, gt_output, gt_output, constraints):
            return False
        abs_img_conf_list = inp["conf_list"]
        output_concrete = self.eval_consistent(expr, abs_img_conf_list, gt_output=str(sorted(list(gt_output))), constraints=constraints)
        if str(sorted(list(gt_output))) in output_concrete:
            return True
        return False
    

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
            res_str = str(sorted(list(res)))
            res_set.add(res_str)
            if gt_output == res_str:
                break
        return res_set


    @abstractmethod 
    def subprogs_not_equal(self, prog1, prog2, inp, semantics):
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
            output_str = str(sorted(list(output)))
            prog_output = self.eval_consistent(prog, inp["conf_list"], gt_output=output_str)
            if output_str not in prog_output:
                return False 
        return True 


    def check_prog_cce(self, prog, examples):
        for inp, output in examples:
            result = self.eval_cce(prog.duplicate(), inp, output)
            if not result:
                return False 
        return True


    def check_prog_standard(self, prog, examples):
        global eval_times
        for inp, output in examples:
            prog_output = self.eval_standard(
                prog, inp["standard"])
            if prog_output != output:
                return False
        return True


    def check_gt_equivalence(self, gt_prog, prog, input_qs, skipped_inputs):
        for id_, inp in input_qs.items():
            if id_ in skipped_inputs:
                continue
            if self.eval_standard(prog, inp["gt"]) != self.eval_standard(gt_prog, inp["gt"]):
                return False
        return True


def check_output_abs(output, abs_value):
    return abs_value[0].issubset(output) and output.issubset(abs_value[1])