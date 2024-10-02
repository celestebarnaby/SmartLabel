import random

from constants import *

from active_learning import ActiveLearning, LabelQuestion

from mnist_domain.mnist_interpreter import MNISTInterpreter 
from mnist_domain.mnist_synthesis import MNISTSynthesizer
from mnist_domain.mnist_benchmarks import mnist_benchmarks
from mnist_domain.mnist_utils import *


class MNISTActiveLearning(ActiveLearning):
    def __init__(self, semantics, question_selection):
        super().__init__(semantics, question_selection)

    def set_benchmarks(self):
        self.benchmarks = mnist_benchmarks 
    
    def set_synthesizer(self):
        self.synth = MNISTSynthesizer(self.semantics)
    
    def set_interpreter(self):
        self.interp = MNISTInterpreter()

    def set_question_space(self, benchmark, i):
        imgs = load_mnist()

        input_space = {}
        labelling_qs = []

        wrongly_predicted_imgs = [img for img in imgs if img.get_pred() != img.gt]
        correctly_predicted_imgs = [img for img in imgs if img.get_pred() == img.gt]

        random.seed(123 + i)
        while len(input_space) < NUM_INPUTS:
            cur_int_list = []
            for _ in range(LIST_LENGTH):
                cur_int_list.append(get_int(correctly_predicted_imgs, wrongly_predicted_imgs))
            additional_int = get_int(correctly_predicted_imgs, wrongly_predicted_imgs)
            inp = {
                "gt" : {"img-list": [get_gt(cur_int) for cur_int in cur_int_list], "img" : get_gt(additional_int)},
                "standard" : {"img-list": [get_standard(cur_int) for cur_int in cur_int_list], "img" : get_standard(additional_int)},
                "conf" : {"img-list": [get_conf(cur_int) for cur_int in cur_int_list], "img" : get_conf(additional_int)},
                }
            
            inp["conf_list"] = self.interp.get_all_universes(inp["conf"])
            inp_id = len(input_space)
            input_space[inp_id] = inp

            
            labelling_qs += [LabelQuestion(inp_id, "img-list", i) for i in range(len(inp["conf"]["img-list"])) if len(inp["conf"]["img-list"][i]) > 1]
            if len(inp["conf"]["img"]) > 1:
                labelling_qs.append(LabelQuestion(inp_id, "img", None))

        gt_prog = self.interp.parse(benchmark.gt_prog)
        self.input_space = input_space 

        random.seed(123 + i)
        self.examples = [(inp_id, self.interp.eval_standard(gt_prog, inp["gt"])) for inp_id, inp in random.sample(sorted(input_space.items()), NUM_INITIAL_EXAMPLES)] 
        self.labelling_qs = labelling_qs
        self.gt_prog = gt_prog 

