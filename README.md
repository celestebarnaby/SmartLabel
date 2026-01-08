# 🧪 Artifact README for "Active Learning for Neurosymbolic Program Synthesis"

This artifact supports the experiments presented in our OOPSLA 2025 paper. 

Our paper makes the following key contributions:
- We define the neurosymbolic active learning problem and propose the first algorithm for solving it.
- We introduce constrained conformal evaluation (CCE) as a new type of program semantics that takes into account both user feedback and prediction sets obtained through conformal prediction. 
- We implement our approach in a new tool called SmartLabel and instantiate it on 3 neurosymbolic domains. 

This artifact contains our tool, SmartLabel, applied to the 3 application domains described in the paper, along with all baselines and ablations 
studied in our evaluation. 

---

## 🖥️ Hardware Dependencies

This artifact has no particular hardware dependencies. It was developed and tested on 2022 MacBook pro.

---

## 📦 Contents

This artifact includes:

- Four branches corresponding to various evaluations:
  - `main`
  - `scalability_experiment`
  - `bottleneck_experiment`
  - `expectation_experiment`
- `requirements.txt` for Python dependencies
- An `output` directory to which all results will be written
- This README

---

### 🧪 What Each Branch Does

The experiments for each branch may be run with the command `python run_experiments.py`

1. `main`

Runs the experiments described in Sections 6.3, 6.4, and 6.5 (except for the question select strategy called "Expected" in Section 6.4).
Results will be written to the `output` directory. For a summary of these results, see `overall_table_data.csv`. This CSV will have 6 rows for each 
of our 3 application domains (as well as 6 overall rows). Each of these 6 rows represents a test setting described in the paper:

- `standard_LearnSy` - LearnSy baseline (Section 6.3, Table 4).
- `standard_SampleSy` - SampleSy baseline (Section 6.3, Table 4).
- `CCE_SmartLabel` - Our technique.
- `CCE-NoAbs_SmartLabel` - NoAbs ablation (Section 6.5, Table 5).
- `CCE_SmartLabelNoUB` - NoBCE ablation (Section 6.5, Table 5).
- `CCE_SelectRandom` - Random question selection strategy (Section 6.4, Table 3).

As described in Section 6.2 of our paper, we repeat this experiments 5 times (each with a different seed) to account for potential variability from 
the initial random IO examples. In `overall_table_data.csv`, we report the mean and standard deviations of the outcomes. Each seed takes ~10 hours to run, so
the full experiment takes ~48 hours.  

2. `scalability_experiment`

Runs the scalability experiment described in Section 6.5.2. Results will be written to the `output` directory. The tables in Figure 20 will be
written to files called `scalability_plot_<DOMAIN_NAME>.pdf`

Each of the 3 tables has 6 datapoints. It takes 2-4 hours to run the benchmarks for each datapoint. Generating the full experimental results takes ~48 hours.

3. `bottleneck_experiment`

Runs the runtime experiment described in Section 6.6 Results will be written to the `output` directory. The table in Figure 21 will be written 
to a file called `bottleneck.pdf`.

This experiment takes <1 hour to run. 

4. `expectation_experiment`

Runs the experiments pertaining to the "Expected" question selection strategy described in Section 6.4 and Table 3. Results will be written 
to the `output` directory. For a summary of these results, see `overall_table_data_expected.csv`. This CSV will have 1 row for each of 
our 3 application domains. Each row represents the results of our technique with the Expected question selection strategy. As with the `main`
experiment, we run this experiment 5 times with different seeds. Each seed takes 1-2 hours, so the full experiment takes ~8 hours. 

## 📝 Reusability Guide

SmartLabel may be applied to (and evaluated on) any neurosymbolic domain for which the user can provide the following:

1. A standard interpreter, and a forward/backward abstract interpreter. The required interface is given in `./main/interpreter.py`.
2. A synthesis engine that can produce an initial hypothesis space. The required interface is given in `./main/synthesis.py`.
3. An input space and corresponding question space. The required interface is given in `./main/active_learning.py`
4. (Optional) A benchmark set to evaluate on, where each benchmark has a ground truth program in the target DSL. The required benchmark format is given in `./main/benchmark.py`.

For an example application, see the `mnist_domain` directory. 

Our active learning learning shown in Figure 9 is implemented by the function `run` in `./main/active_learning.py`.

## 🙏 Thanks for checking out our artifact!