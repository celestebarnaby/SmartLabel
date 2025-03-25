import copy
import csv 
import time 
import random
import signal
import numpy as np 
import ast
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import json
from scipy.optimize import curve_fit

# Question selection
from question_selection.learnsy import LearnSy
from question_selection.samplesy import SampleSy
from question_selection.select_random import SelectRandom
from question_selection.smart_label import SmartLabel
from question_selection.smart_label_no_upper_bound import SmartLabelNoUB

# Active learning
from image_edit_domain.image_edit_active_learning import ImageEditActiveLearning
from image_search_domain.image_search_active_learning import ImageSearchActiveLearning
from mnist_domain.mnist_active_learning import MNISTActiveLearning

# For scalability experiment
from mnist_domain.mnist_utils import get_int, load_mnist, get_gt, get_conf, get_standard, Image, get_w_alg
from mnist_domain.mnist_interpreter import MNISTInterpreter

# Handle timeouts
from utils import handler

# Profiling
import io
import pstats
import cProfile
import statistics

# Constants
from constants import *

def run_experiments(domain, input_space, delta):

    active_learning_data = [(
             "GT Program",
             "Benchmark Info",
             "Semantics", 
             "Delta",
             "Avg. Per Component Pred. Set Size",
             "Avg. Per Input Pred. Set Size",
             "Question Selector",  
             "Correct?",
             "Initial Synthesis Time",
             "Active Learning Time", 
             "Initial Program Space Size", 
             "Final Program Space Size", 
             "Input Space Size",
             "Question Space Size",
             "Avg. Answer Space Size per Question",
             "Avg. Prediction Set Size",
             "# Rounds",  
             "Time Per Round",
            )]


    # Our technique, baselines, and ablations
    test_settings = [
        # # LearnSy (baseline)
        ("standard", LearnSy),
        # # SampleSy (baseline)
        ("standard", SampleSy),
        # # SmartLabel (our technique)
        ("CCE", SmartLabel),
        # # CCE-NoAbs (ablation)
        ("CCE-NoAbs", SmartLabelNoUB),
        # # QS-noUB (ablation)
        # ("CCE", SmartLabelNoUB),
        # Select random question (baseline)
        ("CCE", SelectRandom),
    ] 

    for semantics, question_selection in test_settings:

        pr = cProfile.Profile()
        pr.enable()
        active_learning = domain(semantics, question_selection)
        avg_pred_set_size, avg_total_pred_set_size = get_pred_set_sizes(input_space)
        print(f"Average prediction set size: {avg_pred_set_size}")
        print(f"Average total prediction set size: {avg_total_pred_set_size}")
        for i, benchmark in enumerate(active_learning.benchmarks):
            random.seed(SEED + seed_inc + i)

            print(f"Benchmark: {benchmark.gt_prog}")
            print(f"Domain: {question_selection.__name__}")

            # Generate the input space, question space, and initial examples specific to the domain
            active_learning.set_question_space(benchmark, i, copy.deepcopy(input_space))

            print("Performing initial synthesis...")
            initial_synthesis_time = active_learning.set_program_space(benchmark, i)

            print("Initial synthesis complete.")
            initial_program_space_size = len(active_learning.program_space)
            active_learning_start_time = time.perf_counter()

            # Learn models for inputs (this is specific to the LearnSy baseline)
            active_learning.question_selection.learn_models(active_learning.input_space, semantics, active_learning.synth)

            # Timeout after 600 seconds
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(600)
            output_progs, time_per_round, skipped_inputs = active_learning.run(benchmark, active_learning.program_space)
            signal.alarm(0)
            active_learning_time = time.perf_counter() - active_learning_start_time
            correct = active_learning.synth.interp.check_gt_equivalence(active_learning.gt_prog, output_progs[0], active_learning.input_space, skipped_inputs)  if not isinstance(output_progs, str) else output_progs
            active_learning_data.append(
                (
                    benchmark.gt_prog,
                    benchmark.dataset_name,
                    semantics,
                    delta,
                    avg_pred_set_size,
                    avg_total_pred_set_size,
                    question_selection.__name__,
                    correct, 
                    initial_synthesis_time,
                    active_learning_time,
                    initial_program_space_size,
                    len(output_progs) if output_progs is not None else "FAILED",
                    len(active_learning.input_space),
                    len(active_learning.input_space) + len(active_learning.labelling_qs),
                    avg_answer_space_per_question,
                    avg_pred_set_size,
                    len(time_per_round),
                    time_per_round if len(time_per_round) < 5000 else [], # so csv is readable
                )
            )


        with open(f"./output/{domain.__name__}_active_learning_results_SCALABILITY_{delta}.csv", "w") as f:
            writer = csv.writer(f)
            for row in active_learning_data:
                writer.writerow(row)

        s = io.StringIO()
        pr.disable()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        with open(f"./profiling/{domain.__name__}_{semantics}_{question_selection.__name__}_{domain.__name__}_SCALABILITY_{delta}.txt", "w") as f:
            f.write(s.getvalue())


def get_pred_set_sizes(input_space):
    per_input_pred_set_sizes = []
    per_component_pred_set_sizes = []
    for inp_id, inp in input_space.items():
        for comp in inp['conf']['img-list']:
            if len(comp) > 1:
                per_component_pred_set_sizes.append(len(comp))
        if len(inp['conf']['img']) > 1:
            per_component_pred_set_sizes.append(len(inp['conf']['img']))
        per_input_pred_set_sizes.append(len(inp['conf_list']))
    return sum(per_component_pred_set_sizes)/len(per_component_pred_set_sizes), sum(per_input_pred_set_sizes)/len(per_input_pred_set_sizes)



def csv_to_dict(filename):
    data_dict = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        keys = next(reader)  # Read the first row as keys
        print(filename)
        for row in reader:
            row = [item.strip() for item in row]
            # if "Type" in keys and task_type not in row:
                # continue
            for key, value in zip(keys, row):
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(value)
    return data_dict


def get_experiment_results(domains, seed_inc):
    setting_to_data_overall = {}
    rows = [[
        "Domain",
        "Test Setting",
        "Avg. # Rounds of Interaction",
        "Avg. # Initial Programs",
        "Avg. # Final Programs",
        "Avg. Input Space Size",
        "Avg. Question Space Size",
        "Avg. Answer Space Size per Question",
        "Avg. Prediction Set Size",
        "# Benchmarks Solved",
        "Avg. Time per Round of Interaction"
    ]]

    # Create a table that has the results presented in tables 1, 2, 3 in the paper.
    for domain in domains:
        for task_type in [""] if domain.__name__ == "ImageEditActiveLearning" else [""]:
            setting_to_data_per_domain = {}
            data_dict = csv_to_dict(f"./output/{domain.__name__}_active_learning_results_{seed_inc}.csv", task_type)

            for (
                semantics, 
                question_selector, 
                time_per_round, 
                init_time, 
                correct, 
                num_initial_programs, 
                num_final_programs, 
                input_space_size,
                question_space_size,
                answer_space_size,
                avg_pred_set_size,
                ) in zip(
                    data_dict["Semantics"], 
                    data_dict["Question Selector"], 
                    data_dict["Time Per Round"], 
                    data_dict["Initial Synthesis Time"], 
                    data_dict["Correct?"], 
                    data_dict["Initial Program Space Size"], 
                    data_dict["Final Program Space Size"],
                    data_dict["Input Space Size"],
                    data_dict["Question Space Size"],
                    data_dict["Avg. Answer Space Size per Question"],
                    data_dict["Avg. Prediction Set Size"],
                    ):
                key = f"{semantics}_{question_selector}"
                if key not in setting_to_data_per_domain:
                    setting_to_data_per_domain[key] = {
                        "runtimes" : [],
                        "num_rounds" : [],
                        "num_init_progs" : [],
                        "num_final_progs" : [],
                        "input_space_sizes" : [],
                        "question_space_sizes" : [],
                        "avg_answer_space_sizes" : [],
                        "avg_pred_set_sizes" : [],
                        "correct" : 0 
                    } 
                if key not in setting_to_data_overall:
                    setting_to_data_overall[key] = {
                        "runtimes" : [],
                        "num_rounds" : [],
                        "num_init_progs" : [],
                        "num_final_progs" : [],
                        "input_space_sizes" : [],
                        "question_space_sizes" : [],
                        "avg_answer_space_sizes" : [],
                        "avg_pred_set_sizes" : [],
                        "correct" : 0          
                    } 
                time_per_round = ast.literal_eval(time_per_round)
                setting_to_data_per_domain[key]["num_rounds"].append(len(time_per_round))
                setting_to_data_per_domain[key]["num_init_progs"].append(int(num_initial_programs))
                setting_to_data_per_domain[key]["num_final_progs"].append(int(num_final_programs))
                setting_to_data_per_domain[key]["input_space_sizes"].append(int(input_space_size))
                setting_to_data_per_domain[key]["question_space_sizes"].append(int(question_space_size))
                setting_to_data_per_domain[key]["avg_answer_space_sizes"].append(float(answer_space_size))
                setting_to_data_per_domain[key]["avg_pred_set_sizes"].append(float(avg_pred_set_size))
                setting_to_data_per_domain[key]["correct"] += 1 if correct in {"TRUE", "True"} else 0 

                setting_to_data_overall[key]["num_rounds"].append(len(time_per_round))
                setting_to_data_overall[key]["num_init_progs"].append(int(num_initial_programs))
                setting_to_data_overall[key]["num_final_progs"].append(int(num_final_programs))
                setting_to_data_overall[key]["input_space_sizes"].append(int(input_space_size))
                setting_to_data_overall[key]["question_space_sizes"].append(int(question_space_size))
                setting_to_data_overall[key]["avg_answer_space_sizes"].append(float(answer_space_size))
                setting_to_data_overall[key]["avg_pred_set_sizes"].append(float(avg_pred_set_size))
                setting_to_data_overall[key]["correct"] += 1 if correct in {"TRUE", "True"} else 0

                for i, round_time in enumerate(time_per_round):
                    if i == 0:
                        round_time += float(init_time)
                    setting_to_data_per_domain[key]["runtimes"].append(round_time)
                    setting_to_data_overall[key]["runtimes"].append(round_time)

            for key, val in setting_to_data_per_domain.items():
                rows.append([
                    f"{domain.__name__}{task_type}",
                    key,
                    np.mean(val["num_rounds"]),
                    np.mean(val["num_init_progs"]),
                    np.mean(val["num_final_progs"]),
                    np.mean(val["input_space_sizes"]),
                    np.mean(val["question_space_sizes"]),
                    np.mean(val["avg_answer_space_sizes"]),
                    np.mean(val["avg_pred_set_sizes"]),
                    val["correct"],
                    np.mean(val["runtimes"])
                ])

    for key, val in setting_to_data_overall.items():
        rows.append([
            "Overall",
            key,
            np.mean(val["num_rounds"]),
            np.mean(val["num_init_progs"]),
            np.mean(val["num_final_progs"]),
            np.mean(val["input_space_sizes"]),
            np.mean(val["question_space_sizes"]),
            np.mean(val["avg_answer_space_sizes"]),
            np.mean(val["avg_pred_set_sizes"]),
            val["correct"],
            np.mean(val["runtimes"])
        ])

    with open(f"./output/table_data_{seed_inc}.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def get_combined_table():
    overall_data_dict = {}
    keys = [
        "Domain",
        "Test Setting",
        "Avg. # Rounds of Interaction",
        "Avg. # Initial Programs",
        "Avg. # Final Programs",
        "Avg. Input Space Size",
        "Avg. Question Space Size",
        "Avg. Answer Space Size per Question",
        "Avg. Prediction Set Size",
        "# Benchmarks Solved",
        "Avg. Time per Round of Interaction"
    ]
    for i in range(NUM_SEEDS):
        data_dict = csv_to_dict(f"./output/table_data_{i}.csv", "")
        for key in keys:
            if key in {"Domain", "Test Setting"}:
                overall_data_dict[key] = data_dict[key] 
                continue
            if key not in overall_data_dict:
                overall_data_dict[key] = [[float(item)] for item in data_dict[key]]
            else:
                overall_data_dict[key] = [item1 + [float(item2)] for item1, item2 in zip(overall_data_dict[key], data_dict[key])]
    rows = [keys]
    rows += [[overall_data_dict[key][i] if key in {"Domain", "Test Setting"} else (float(np.mean(overall_data_dict[key][i])), statistics.stdev(overall_data_dict[key][i])) for key in keys] for i in range(len(overall_data_dict["Domain"]))]

    with open(f"./output/overall_table_data.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


    # Create the bar plot presented in Figure 22 in the paper.
    # if constants.TIME_EVALS:
    #     for domain in domains:
    #         data_dict = csv_to_dict(f"./output/{domain.__name__}_eval_results.csv")

    #         domain_to_buckets = {
    #             "MNISTActiveLearning" : [((1, 10), "(1, 10]"), ((11, 50), "(10, 50]"), ((51, 300), "(51, 300]")],
    #             "ImageEditingActiveLearning" : [((1, 20), "(1, 20]"), ((21, 200), "(20, 200]"), ((201, 600), "(200, 600]")]
    #         }

    #         buckets = domain_to_buckets[domain.__name__]
    #         plot_data = []
    #         for semantics, question_selector, eval_times, num_evals in zip(data_dict["Semantics"], data_dict["Question Selector"], data_dict["Eval Times"], data_dict["Num Evals"]):

    #             if question_selector != "SmartLabel":
    #                 continue

    #             eval_times = ast.literal_eval(eval_times)
    #             num_evals = ast.literal_eval(num_evals)
    #             for bucket, bucket_name in buckets:
    #                 plot_data.append([semantics, bucket_name, sum([val * 1000 for key, val in eval_times.items() if key >= bucket[0] and key <= bucket[1]])/sum([val for key, val in num_evals.items() if key >= bucket[0] and key <= bucket[1]])])

    #         # the sample dataframe from the OP
    #         df = pd.DataFrame(plot_data, columns=['group', 'column', 'val'])

    #         plt.figure(figsize=(5, 3))
    #         plt.xlabel("Prediction Set Size")
    #         plt.ylabel("Average Evaluation Time (ms)")
    #         plt.title(domain.__name__)
    #         plt.tight_layout()

    #         # plot with seaborn barplot
    #         sns.barplot(data=df, x='column', y='val', hue='group', edgecolor="black", palette='BuPu')

    #         plt.legend(loc='upper left', fontsize=12)
    #         plt.savefig(f"./output/{domain.__name__}_CCE_ablation.pdf")


def json_to_img(json_dict):
    return Image(json_dict["preds"], json_dict["gt"])


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def get_img_lists(save=False, load=False):
    if load:
        with open('./mnist_domain/img_lists.json', 'r') as f:
            img_lists = json.load(f)
        return [[[json_to_img(img) for img in imgs] for imgs in img_list] for img_list in img_lists]

    imgs = load_mnist()

    img_lists = []

    while len(img_lists) < constants.NUM_INPUTS:
        img_list = []
        # Add one more for img_int
        for _ in range(constants.LIST_LENGTH + 1):
            img_list.append(get_int(imgs))

        img_lists.append(img_list)

    if save:
        with open('./mnist_domain/img_lists.json', 'w') as f:
            json.dump([[[img.to_json() for img in imgs] for imgs in img_list] for img_list in img_lists], f, cls=NpEncoder)

    return img_lists


def get_questions_from_img_lists(img_lists, interp, delta):

    input_space = {}

    threshold = get_w_alg(delta)

    for inp_id, img_list in enumerate(img_lists):

        input_imgs = img_list[:constants.LIST_LENGTH]
        img_int = img_list[-1]
        
        inp = {
            "gt" : {"img-list": [int(get_gt(img)) for img in input_imgs], "img" : int(get_gt(img_int))},
            "standard" : {"img-list": [int(get_standard(img)) for img in input_imgs], "img" : int(get_standard(img_int))},
            "conf" : {"img-list": [get_conf(cur_int, threshold) for cur_int in input_imgs], "img" : get_conf(img_int, threshold)},
            }
        
        inp["conf_list"] = interp.get_all_universes(inp["conf"])

        input_space[inp_id] = inp

    return input_space

def make_scalability_experiment_plot(domain, deltas):

    pred_set_size_to_avg_runtime = {}
    
    first_threshold = True
    benchmark_to_num_rounds = {}

    for delta in deltas:

        data_dict = csv_to_dict(f"./output/{domain.__name__}_active_learning_results_SCALABILITY_{delta}.csv")

        semantics_to_rtimes_per_round = {
            "CCE_SmartLabel" : [],
            "CCE-NoAbs_SmartLabelNoUB" : []
        }

        for (benchmark, semantics, avg_pred_set_size, qs, time_per_round, init_time, num_rounds) in zip(
            data_dict["GT Program"], 
            data_dict["Semantics"], 
            data_dict["Avg. Per Component Pred. Set Size"], 
            data_dict["Question Selector"], 
            data_dict["Time Per Round"], 
            data_dict["Initial Synthesis Time"],
            data_dict["# Rounds"]
            ):
            key = "{}_{}".format(semantics, qs)
            if key not in semantics_to_rtimes_per_round:
                continue 
            time_per_round = ast.literal_eval(time_per_round)
            for i, round_time in enumerate(time_per_round):
                if i == 0:
                    round_time += float(init_time)
                if i not in semantics_to_rtimes_per_round[key]:
                    semantics_to_rtimes_per_round[key].append([])
                if first_threshold or i < benchmark_to_num_rounds[benchmark]:
                    # if True:
                    semantics_to_rtimes_per_round[key][i].append(round_time)
            if first_threshold:
                benchmark_to_num_rounds[benchmark] = int(num_rounds)
                
        pred_set_size_to_avg_runtime[float(avg_pred_set_size)] = {}

        for key, val in semantics_to_rtimes_per_round.items():
            all_runtimes = []
            for l in val:
                all_runtimes += l

            pred_set_size_to_avg_runtime[float(avg_pred_set_size)][key] = np.mean(all_runtimes)

        first_threshold = False


    x_axis = list(pred_set_size_to_avg_runtime.keys())



    # Fit linear model
    linear_coeffs = np.polyfit(x_axis, [pred_set_size_to_avg_runtime[item]["CCE_SmartLabel"] for item in x_axis], 1)
    linear_fit = np.poly1d(linear_coeffs)

    # fit exponential model
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    

    print(len(x_axis))
    print(len([pred_set_size_to_avg_runtime[item]["CCE-NoAbs_SmartLabelNoUB"] for item in x_axis]))
    params, params_covariance = curve_fit(exp_func, x_axis, [pred_set_size_to_avg_runtime[item]["CCE-NoAbs_SmartLabelNoUB"] for item in x_axis])
    # Plot the fitted exponential curve
    x_fit = np.linspace(min(x_axis), max(x_axis), 500)
    y_fit = exp_func(x_fit, params[0], params[1])
    # plt.plot(x_fit, y_fit, color='mediumpurple', alpha=.5)

    # Create scatter plot
    x_vals = np.linspace(min(x_axis), max(x_axis), 500)
    plt.plot(x_vals, linear_fit(x_vals), color='cornflowerblue', alpha=.5)
    plt.scatter(x_axis, [pred_set_size_to_avg_runtime[item]["CCE_SmartLabel"] for item in x_axis], color='cornflowerblue', label='SmartLabel')
    # plt.scatter(x_axis, [pred_set_size_to_avg_runtime[item]["conf_minimax_partial_conf"] for item in x_axis], color='orange', label='CCE-NoAbs')
    # plt.scatter(x_axis, [pred_set_size_to_avg_runtime[item]["bidirect_minimax_conf"] for item in x_axis], color='mediumseagreen', label='QS-NoUB')
    plt.scatter(x_axis, [pred_set_size_to_avg_runtime[item]["CCE-NoAbs_SmartLabelNoUB"] for item in x_axis], color='mediumpurple', label='QS-NoUB + CCE-NoAbs')



    # Add labels and title
    plt.xlabel('Avg. Prediction Set Size')
    plt.ylabel('Avg. User Interaction Time per Round (s)')
    # plt.title('')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.savefig('./output/plt.png', dpi=300)



if __name__ == "__main__":
    img_lists = get_img_lists(load=True)
    deltas = [
        .005,
        .00475,
        .0045,
        .00425,
        .004
    ]
    for delta in deltas:
        interp = MNISTInterpreter()
        input_questions = get_questions_from_img_lists(img_lists, interp, delta)

        domains = [ MNISTActiveLearning ]
        # for domain in domains:
            # run_experiments(domain, input_questions, delta)
        # get_experiment_results(domains)
    make_scalability_experiment_plot(MNISTActiveLearning, deltas)
