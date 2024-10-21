import random
import time

from constants import *

from active_learning import ActiveLearning, LabelQuestion

from mnist_domain.mnist_interpreter import MNISTInterpreter 
from mnist_domain.mnist_synthesis import MNISTSynthesizer
from mnist_domain.mnist_benchmarks import mnist_benchmarks
from mnist_domain.mnist_utils import *


import json

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

        # random.seed(123 + i)


        # with open(MNIST_QUESTIONS_DIR, 'r') as f:
        #     qs = json.load(f)

        # self.transform_qs(qs)
    
        # with open(MNIST_NEW_QUESTIONS_DIR, 'r') as f:
        #     qs = json.load(f)
    
        # input_space = qs['input_space']
        # input_space = {int(key) : val for key, val in input_space.items()}
        # labelling_qs = qs['label_questions']
        # labelling_qs = [LabelQuestion(input_id, "img-list" if name == "list-img-int-conf" else "img", i) for (input_id, name, i) in labelling_qs]

        random.seed(124)  
        per_digit_correct = []
        while len(input_space) < NUM_INPUTS:
            cur_int_list = []
            for _ in range(LIST_LENGTH):
                cur_int_list.append(get_int(correctly_predicted_imgs, wrongly_predicted_imgs))

                # TODO: remove this stuff later
                for item in cur_int_list:
                    for digit in item:
                        per_digit_correct.append(digit.gt == digit.get_pred())

            additional_int = get_int(correctly_predicted_imgs, wrongly_predicted_imgs)

            for digit in additional_int:
                per_digit_correct.append(digit.gt == digit.get_pred())

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


        self.examples = [(inp_id, self.interp.eval_standard(gt_prog, inp["gt"])) for inp_id, inp in random.sample(sorted(input_space.items()), NUM_INITIAL_EXAMPLES)] 

        print(f"Per digit accuracy: {len([item for item in per_digit_correct if item])/len(per_digit_correct)}")
        print(len(per_digit_correct))
        self.get_accuracy()

        self.labelling_qs = labelling_qs
        self.gt_prog = gt_prog 
        self.num_samples = MNIST_NUM_SAMPLES


    def get_accuracy(self):
        # digits_correct = []
        entries_correct_standard = []
        entries_correct_conf = []
        for input_list in self.input_space.values():
            for gt_val, standard_val, conf_val in zip(input_list['gt']['img-list'] + [input_list['gt']['img']], input_list['standard']['img-list'] + [input_list['standard']['img']], input_list['conf']['img-list'] + [input_list['conf']['img']]):
                entries_correct_standard.append(gt_val == standard_val)
                entries_correct_conf.append(gt_val in conf_val)


        print(f"Standard Semantics Accuracy: {len([entry for entry in entries_correct_standard if entry])/len(entries_correct_standard)}")
        print(f"Conformal Semantics Accuracy: {len([entry for entry in entries_correct_conf if entry])/len(entries_correct_conf)}")

    def set_program_space(self, benchmark, i):
        initial_synth_start_time = time.perf_counter()
        self.program_space = self.synth.synthesize([(self.input_space[q], a) for q, a in self.examples])
        initial_synthesis_time = time.perf_counter() - initial_synth_start_time
        return initial_synthesis_time

    def transform_qs(self, qs):
        new_input_questions = {}


        input_questions = qs['inputs']
        labelling_questions = qs['labels']

        for q in input_questions:
            new_inp = {
                "gt" : {"img-list" : q["list-img-int-star"], "img" : q["img-int-star"]},
                "standard" : {"img-list" : q["list-img-int-hat"], "img" : q["img-int-hat"]},
                "conf" : {"img-list" : q["list-img-int-conf"], "img" : q["img-int-conf"]}
            }

            new_inp["conf_list"] = self.interp.get_all_universes(new_inp["conf"])
            inp_id = len(new_input_questions)
            new_input_questions[inp_id] = new_inp 

        # new_labelling_questions = [LabelQuestion(input_id, "img-list" if name == "list-img-int-conf" else "img", i) for (input_id, name, i) in labelling_questions]

        new_questions = {'input_space' : new_input_questions, 'label_questions' : labelling_questions}
        with open("./mnist_domain/mnist_dataset/questions_NEW.json", 'w') as f:
            json.dump(new_questions, f)