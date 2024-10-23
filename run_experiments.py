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

# Question selection
from question_selection.learnsy import LearnSy
from question_selection.samplesy import SampleSy
from question_selection.select_abstract import SelectAbstract
from question_selection.select_random import SelectRandom
from question_selection.smart_label import SmartLabel
from question_selection.smart_label_no_upper_bound import SmartLabelNoUB

# Active learning
from image_edit_domain.image_edit_active_learning import ImageEditActiveLearning
from mnist_domain.mnist_active_learning import MNISTActiveLearning

# Handle timeouts
from utils import handler

# Profiling
import io
import pstats
import cProfile

# Constants
import constants

def run_experiments(domain):

    active_learning_data = [(
             "GT Program",
             "Benchmark Info",
             "Semantics", 
             "Question Selector",  
             "Correct?",
             "Initial Synthesis Time",
             "Active Learning Time", 
             "Initial Program Space Size", 
             "Final Program Space Size", 
             "# Rounds",  
             "Time Per Round",
            )]
    eval_data = [(
        "Semantics",
        "Question Selector",
        "Eval Times",
        "Num Evals"
    )]


    # Our technique, baselines, and ablations
    test_settings = [
        # # LearnSy (baseline)
        # ("standard", LearnSy),
        # # SampleSy (baseline)
        ("standard", SampleSy),
        # # SmartLabel Abstract (ablation)
        # ("CCE", SelectAbstract),
        # # SmartLabel (our technique)
        ("CCE", SmartLabel),
        # # CCE-NoAbs (ablation)
        ("CCE-NoAbs", SmartLabel),
        # # QS-noUB (ablation)
        ("CCE", SmartLabelNoUB),
        # Select random question (baseline)
        # ("CCE", SelectRandom),
    ] 

    random.seed(123)
    for semantics, question_selection in test_settings:

        constants.TIME_PER_EVAL = {}
        constants.NUM_EVALS = {}

        pr = cProfile.Profile()
        pr.enable()
        active_learning = domain(semantics, question_selection)
        for i, benchmark in enumerate(active_learning.benchmarks):

            print(f"Benchmark: {benchmark.gt_prog}")
            print(f"Domain: {question_selection.__name__}")

            # Generate the input space, question space, and initial examples specific to the domain
            active_learning.set_question_space(benchmark, i)

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
                    question_selection.__name__,
                    correct, 
                    initial_synthesis_time,
                    active_learning_time,
                    initial_program_space_size,
                    len(output_progs) if output_progs is not None else "FAILED",
                    len(time_per_round),
                    time_per_round
                )
            )

        eval_data.append((
            semantics,
            question_selection.__name__,
            copy.deepcopy(constants.TIME_PER_EVAL),
            copy.deepcopy(constants.NUM_EVALS),
        ))

        with open(f"./output/{domain.__name__}_active_learning_results.csv", "w") as f:
            writer = csv.writer(f)
            for row in active_learning_data:
                writer.writerow(row)

        with open(f"./output/{domain.__name__}_eval_results.csv", "w") as f:
            writer = csv.writer(f)
            for row in eval_data:
                writer.writerow(row)

        s = io.StringIO()
        pr.disable()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        with open(f"./profiling/{domain.__name__}_{semantics}_{question_selection.__name__}_{domain.__name__}.txt", "w") as f:
            f.write(s.getvalue())


def csv_to_dict(filename):
    data_dict = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        keys = next(reader)  # Read the first row as keys
        for row in reader:
            for key, value in zip(keys, row):
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(value)
    return data_dict


def get_experiment_results(domains):
    setting_to_data_overall = {}
    rows = [[
        "Domain",
        "Test Setting",
        "Avg. # Rounds of Interaction",
        "Avg. # Initial Programs",
        "Avg. # Final Programs",
        "# Benchmarks Solved",
        "Avg. Time per Round of Interaction"
    ]]

    # Create a table that has the results presented in tables 1, 2, 3 in the paper.
    for domain in domains:
        setting_to_data_per_domain = {}
        data_dict = csv_to_dict(f"./output/{domain.__name__}_active_learning_results.csv")

        for (semantics, question_selector, time_per_round, init_time, correct, num_initial_programs, num_final_programs) in zip(data_dict["Semantics"], data_dict["Question Selector"], data_dict["Time Per Round"], data_dict["Initial Synthesis Time"], data_dict["Correct?"], data_dict["Initial Program Space Size"], data_dict["Final Program Space Size"]):
            key = "{}_{}".format(semantics, question_selector)
            if key not in setting_to_data_per_domain:
                setting_to_data_per_domain[key] = {
                    "runtimes" : [],
                    "num_rounds" : [],
                    "num_init_progs" : [],
                    "num_final_progs" : [],
                    "correct" : 0 
                } 
            if key not in setting_to_data_overall:
                setting_to_data_overall[key] = {
                    "runtimes" : [],
                    "num_rounds" : [],
                    "num_init_progs" : [],
                    "num_final_progs" : [],
                    "correct" : 0          
                } 
            time_per_round = ast.literal_eval(time_per_round)
            setting_to_data_per_domain[key]["num_rounds"].append(len(time_per_round))
            setting_to_data_per_domain[key]["num_init_progs"].append(int(num_initial_programs))
            setting_to_data_per_domain[key]["num_final_progs"].append(int(num_final_programs))
            setting_to_data_per_domain[key]["correct"] += 1 if correct in {"TRUE", "True"} else 0 

            setting_to_data_overall[key]["num_rounds"].append(len(time_per_round))
            setting_to_data_overall[key]["num_init_progs"].append(int(num_initial_programs))
            setting_to_data_overall[key]["num_final_progs"].append(int(num_final_programs))
            setting_to_data_overall[key]["correct"] += 1 if correct in {"TRUE", "True"} else 0

            for i, round_time in enumerate(time_per_round):
                if i == 0:
                    round_time += float(init_time)
                setting_to_data_per_domain[key]["runtimes"].append(round_time)
                setting_to_data_overall[key]["runtimes"].append(round_time)

        for key, val in setting_to_data_per_domain.items():
            rows.append([
                domain.__name__,
                key,
                np.mean(val["num_rounds"]),
                np.mean(val["num_init_progs"]),
                np.mean(val["num_final_progs"]),
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
            val["correct"],
            np.mean(val["runtimes"])
        ])

    with open(f"./output/table_data.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


    # Create the bar plot presented in Figure 22 in the paper.
    if constants.TIME_EVALS:
        for domain in domains:
            data_dict = csv_to_dict(f"./output/{domain.__name__}_eval_results.csv")

            domain_to_buckets = {
                "MNISTActiveLearning" : [((1, 10), "(1, 10]"), ((11, 50), "(10, 50]"), ((51, 300), "(51, 300]")],
                "ImageEditingActiveLearning" : [((1, 20), "(1, 20]"), ((21, 200), "(20, 200]"), ((201, 600), "(200, 600]")]
            }

            buckets = domain_to_buckets[domain.__name__]
            plot_data = []
            for semantics, question_selector, eval_times, num_evals in zip(data_dict["Semantics"], data_dict["Question Selector"], data_dict["Eval Times"], data_dict["Num Evals"]):

                if question_selector != "SmartLabel":
                    continue

                eval_times = ast.literal_eval(eval_times)
                num_evals = ast.literal_eval(num_evals)
                for bucket, bucket_name in buckets:
                    plot_data.append([semantics, bucket_name, sum([val * 1000 for key, val in eval_times.items() if key >= bucket[0] and key <= bucket[1]])/sum([val for key, val in num_evals.items() if key >= bucket[0] and key <= bucket[1]])])

            # the sample dataframe from the OP
            df = pd.DataFrame(plot_data, columns=['group', 'column', 'val'])

            plt.figure(figsize=(5, 3))
            plt.xlabel("Prediction Set Size")
            plt.ylabel("Average Evaluation Time (ms)")
            plt.title(domain.__name__)
            plt.tight_layout()

            # plot with seaborn barplot
            sns.barplot(data=df, x='column', y='val', hue='group', edgecolor="black", palette='BuPu')

            plt.legend(loc='upper left', fontsize=12)
            plt.savefig(f"./output/{domain.__name__}_CCE_ablation.pdf")



if __name__ == "__main__":
    domains = [ MNISTActiveLearning]
    for domain in domains:
        run_experiments(domain)
    get_experiment_results(domains)