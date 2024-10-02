import csv 
import time 
import random
import signal
import numpy as np 
import ast

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

def run_experiments(domain):

    rows = [(
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

    # Our technique, baselines, and ablations
    test_settings = [
        # # LearnSy (baseline)
        ("standard", LearnSy),
        # # SampleSy (baseline)
        ("standard", SampleSy),
        # # SmartLabel Abstract (ablation)
        ("CCE", SelectAbstract),
        # # SmartLabel (our technique)
        ("CCE", SmartLabel),
        # # CCE-NoAbs (ablation)
        ("CCE-NoAbs", SmartLabel),
        # # QS-noUB (ablation)
        ("CCE", SmartLabelNoUB),
        # Select random question (baseline)
        ("CCE", SelectRandom),
    ] 

    random.seed(123)
    for semantics, question_selection in test_settings:
        pr = cProfile.Profile()
        pr.enable()
        active_learning = domain(semantics, question_selection)
        for i, benchmark in enumerate(active_learning.benchmarks):

            # Generate the input space, question space, and initial examples specific to the domain
            active_learning.set_question_space(benchmark, i)

            # Learn models for inputs (this is specific to the LearnSy baseline)
            active_learning.question_selection.learn_models(active_learning.input_space, semantics, active_learning.synth)
            print(f"Benchmark: {benchmark.gt_prog}")
            print(f"Domain: {question_selection.__name__}")

            print("Performing initial synthesis...")
            initial_synthesis_start_time = time.perf_counter()
            program_space = active_learning.synth.synthesize([(active_learning.input_space[q], a) for q, a in active_learning.examples])

            initial_synthesis_time = time.perf_counter() - initial_synthesis_start_time
            print("Initial synthesis complete.")
            initial_program_space_size = len(program_space)
            active_learning_start_time = time.perf_counter()

            # Timeout after 600 seconds
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(600)
            output_progs, time_per_round, skipped_inputs = active_learning.run(benchmark, program_space)
            signal.alarm(0)
            active_learning_time = time.perf_counter() - active_learning_start_time
            correct = active_learning.synth.interp.check_gt_equivalence(active_learning.gt_prog, output_progs[0], active_learning.input_space, skipped_inputs)  if not isinstance(output_progs, str) else output_progs
            rows.append(
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

        with open(f"./output/{domain.__name__}_results.csv", "w") as f:
            writer = csv.writer(f)
            for row in rows:
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


def get_table_data(domains):
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
    for domain in domains:
        setting_to_data_per_domain = {}
        data_dict = csv_to_dict(f"./output/{domain.__name__}_results.csv")

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
            setting_to_data_per_domain[key]["correct"] += 1 if correct == "True" else 0 

            setting_to_data_overall[key]["num_rounds"].append(len(time_per_round))
            setting_to_data_overall[key]["num_init_progs"].append(int(num_initial_programs))
            setting_to_data_overall[key]["num_final_progs"].append(int(num_initial_programs))
            setting_to_data_overall[key]["correct"] += 1 if correct == "True" else 0

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





if __name__ == "__main__":
    domains = [ImageEditActiveLearning, MNISTActiveLearning]
    for domain in domains:
        run_experiments(domain)
    get_table_data(domains)