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
from image_edit_active_learning import ImageEditActiveLearning

# Handle timeouts
from image_edit_utils import handler

# Profiling
import io
import pstats
import cProfile

def run_experiments():

    rows = [(
             "GT Program",
             "Benchmark Info",
             "Semantics", 
             "Question Selector",  
             "Correct?",
             "Active Learning Time", 
             "Initial Program Space Size", 
             "Final Program Space Size", 
             "# Rounds",  
             "Time Per Round",
            )]

    test_settings = [
        # # LearnSy
        # ("standard", LearnSy),
        # # SampleSy
        # ("standard", SampleSy),
        # # SmartLabel Abstract Ablation
        ("CCE", SelectAbstract),
        # # SmartLabel
        ("CCE", SmartLabel),
        # # CCE-NoAbs
        ("CCE-NoAbs", SmartLabel),
        # # QS-noUB
        ("CCE", SmartLabelNoUB),
        # Select random question
        ("CCE", SelectRandom),
    ] 

    domains = [
        ImageEditActiveLearning
    ]

    random.seed(123)
    for domain in domains:
        for semantics, question_selection in test_settings:
            pr = cProfile.Profile()
            pr.enable()
            active_learning = domain(semantics, question_selection)
            for benchmark in active_learning.benchmarks:
                active_learning.set_question_space(benchmark)
                active_learning.question_selection.learn_models(active_learning.input_space, semantics, active_learning.synth)
                print(f"Benchmark: {benchmark.gt_prog}")
                print(f"Domain: {question_selection.__name__}")

                print("Performing initial synthesis...")
                program_space = active_learning.synth.synthesize([(active_learning.input_space[q], a) for q, a in active_learning.examples])
                print("Initial synthesis complete.")
                initial_program_space_size = len(program_space)
                active_learning_start_time = time.perf_counter()

                # Timeout after 600 seconds
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(600)
                output_progs, time_per_round, skipped_inputs = active_learning.run(benchmark, program_space)
                signal.alarm(0)
                active_learning_time = time.perf_counter() - active_learning_start_time
                correct = active_learning.synth.interp.check_gt_equivalence(benchmark.gt_prog, output_progs[0], active_learning.input_space, skipped_inputs)  if not isinstance(output_progs, str) else output_progs
                rows.append(
                    (
                        benchmark.gt_prog,
                        benchmark.dataset_name,
                        semantics,
                        question_selection.__name__,
                        correct, 
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
            with open(f"./profiling/{semantics}_{question_selection.__name__}_{domain.__name__}.txt", "w") as f:
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
    setting_to_runtimes_overall = {}
    for domain in domains:
        setting_to_runtimes_per_domain = {}
        data_dict = csv_to_dict(f"./output/{domain.__name__}_results.csv")

        for (semantics, question_selector, time_per_round, init_time) in zip(data_dict["Semantics"], data_dict["Question Selector"], data_dict["Time Per Round"], data_dict["Initial Synthesis Time"]):
            key = "{}_{}".format(semantics, question_selector)
            if key not in setting_to_runtimes_per_domain:
                setting_to_runtimes_per_domain[key] = [] 
            if key not in setting_to_runtimes_overall:
                setting_to_runtimes_overall[key] = [] 
            time_per_round = ast.literal_eval(time_per_round)
            for i, round_time in enumerate(time_per_round):
                if i == 0:
                    round_time += float(init_time)
                setting_to_runtimes_per_domain[key][i].append(round_time)
                setting_to_runtimes_overall[key][i].append(round_time)

        for key, val in setting_to_runtimes_per_domain.items():
            all_runtimes = [runtime for runtimes in val for runtime in runtimes]
            # TODO: delete
            # all_runtimes = []
            # for l in val:
            #     all_runtimes += l

            print(f"Mean runtime for {key} in the {domain.__name__} domain: {np.mean}")
            print(f"Median for {key} in the {domain.__name__} domain: {np.median(all_runtimes)}")
            print()
        num_rounds = [len(runtimes) for runtimes in setting_to_runtimes_per_domain["CCE_SmartLabel"]]
        print(f"Total # of rounds in domain {domain.__name__}: {sum(num_rounds)}")
        print(f"Average # of rounds in domain {domain.__name__}: {np.mean(num_rounds)}")
        print(f"Median # of rounds in domain {domain.__name__}: {np.median(num_rounds)}")



    for key, val in setting_to_runtimes_overall.items():
        all_runtimes = [runtime for runtimes in val for runtime in runtimes]

        print(f"Overall runtime for {key}: {np.mean}")
        print(f"Median for {key}: {np.median(all_runtimes)}")
        print()
    num_rounds = [len(runtimes) for runtimes in setting_to_runtimes_overall["CCE_SmartLabel"]]
    print(f"Total # of rounds overall: {sum(num_rounds)}")
    print(f"Average # of rounds overall: {np.mean(num_rounds)}")
    print(f"Median # of rounds overall: {np.median(num_rounds)}")





if __name__ == "__main__":
    domains = [ImageEditActiveLearning]
    for domain in domains:
        run_experiments()
    # get_table_data(domains)