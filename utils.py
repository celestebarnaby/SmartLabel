class TimeOutException(Exception):
    def __init__(self, message, errors=None):
        super(TimeOutException, self).__init__(message)
        self.errors = errors


def handler(signum, frame):
    raise TimeOutException("Timeout")

# TODO: decide whether we need this.
        # Current setup: same input space every time
        # imgs = load_mnist()
        # input_space = {}
        # random.seed(124)

        # per_digit_correct = []
        # while len(input_space) < NUM_INPUTS:
        #     cur_int_list = []
        #     for _ in range(LIST_LENGTH):
        #         cur_int_list.append(get_int(imgs))

        #         # TODO: remove this stuff later
        #         for item in cur_int_list:
        #             for digit in item:
        #                 per_digit_correct.append(digit.gt == digit.get_pred())

        #     additional_int = get_int(imgs)

        #     for digit in additional_int:
        #         per_digit_correct.append(digit.gt == digit.get_pred())

        #     inp = {
        #         "gt" : {"img-list": [int(get_gt(cur_int)) for cur_int in cur_int_list], "img" : int(get_gt(additional_int))},
        #         "standard" : {"img-list": [int(get_standard(cur_int)) for cur_int in cur_int_list], "img" : int(get_standard(additional_int))},
        #         "conf" : {"img-list": [get_conf(cur_int) for cur_int in cur_int_list], "img" : get_conf(additional_int)},
        #         }
            
        #     inp["conf_list"] = self.interp.get_all_universes(inp["conf"])
        #     inp_id = len(input_space)
        #     input_space[inp_id] = inp

            
        #     labelling_qs += [LabelQuestion(inp_id, "img-list", i) for i in range(len(inp["conf"]["img-list"])) if len(inp["conf"]["img-list"][i]) > 1]
        #     if len(inp["conf"]["img"]) > 1:
        #         labelling_qs.append(LabelQuestion(inp_id, "img", None))

        # gt_prog = self.interp.parse(benchmark.gt_prog)
        # self.input_space = input_space 
        # with open('./input_space.json', 'w') as f:
        #     json.dump(input_space, f)


        # Different examples for every benchmark

        # print(f"Per digit accuracy: {len([item for item in per_digit_correct if item])/len(per_digit_correct)}")
        # print(len(per_digit_correct))
        # self.get_accuracy()
        # print(sorted([len(inp["conf_list"]) for inp in input_space.values()]))
        # print(np.mean(sorted([len(inp["conf_list"]) for inp in input_space.values()])))
        # raise TypeError