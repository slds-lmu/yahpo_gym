Frequently Asked Questions
************************

In the following, we maintain a list of frequently asked questions.

Citation
=======================

If you use YAHPO Gym, please cite the following paper:

* Pfisterer, F., Schneider, L., Moosbauer, J., Binder, M., & Bischl, B. (2022). YAHPO Gym - An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization. In International Conference on Automated Machine Learning.

Moreover, certain `scenarios` built upon previous work, e.g., the `lcbench` scenario uses data from:

* Zimmer, L., Lindauer, M., & Hutter, F. (2021). Auto-Pytorch: Multi-Fidelity Metalearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(9), 3079-3090.

* Zimmer, L. (2020). data_2k_lw.zip. figshare. Dataset. https://doi.org/10.6084/m9.figshare.11662422.v1, Apache License, Version 2.0.

OpenML task_id and dataset_id
=======================

Currently, the `rbv2_*`, `lcbench`, and `iaml_*` scenarios contain instances based on OpenML datasets.
For `rbv2_*` and `iaml_*` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the `lcbench` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>.

Reproducibility
=======================

`YAHPO Gym` relies on static neural networks compiled via `ONNX`. 
This should result in reproducible results given equal hardware and software versions.
Unfortunately, `ONNX` models do not always yield reproducible results across different hardware.
This is, e.g. discussed in https://github.com/microsoft/onnxruntime/issues/12086.

In practice, we have not observed relevant differences between different hardware versions, but this might help to explain observations
regarding a lack of exact reproducibility.

lcbench and epochs
==================

The original LCBench data includes 52 epochs.
The surrogates of `YAHPO Gym` v1.0 were trained on this data.
Note, however, that the first epoch in the LCBench data refers to the models only being initialized (i.e., not trained).
Usually, it is therefore best to exclude this first epoch for learning curve purposes or when doing multi-fidelity HPO.
Moreover, the last epoch in the original LCBench data mostly contains exactly the same performance metrics as the penultimate epoch.
Often, a sensible epoch range for the `lcbench` scenario in `YAHPO Gym` is therefore given by 2 to 51.

Monotonicity in Runtime
=======================

Currently, `YAHPO Gym` surrogates do **not** enforce runtime predictions to be monotone increasing with respect to the fidelity parameter.
This is mainly due to most of our scenarios not involving the training of neural networks (except for `nb301` and `lcbench`) and in the case of, e.g., the fidelity parameter being `trainsize`, it is not necessary meaningful to assume a monotone increasing relationship between runtime and fidelity.
As we are using the same surrogate architecture for all scenarios, monotonicity is therefore also not enforced for the `lcbench` and `nb301` scenarios.
We plan to incorporate surrogates that enforce a monotone increasing relationship between runtime and and the fidelity parameter for the `lcbench` and `nb301` scenarios in upcoming versions of `YAHPO Gym`.

Using F1 scores for rbv2_*
=======================

`F1` scores in the `rbv2_*` scenarios are only available for binary classification datasets. 
On multiclass datasets, the corresponding `F1` score is imputed with `0` and returned by the surrogate model.
The information on which `id` corresponds to a multiclass dataset can be obtained from the entry `is_multicrit` in `BenchmarkSet.config.config`.

Memory Estimation for rbv2_*
=======================

For the `rbv2_*` settings, memory consumption was estimated by observing the memory consumption during training via `/usr/bin/time`. 
This estimates the `maximum resident size`.
In general, we assume that this provides a coarse estimation of the processes memory consumption.
However, it does not seem to work if the goal, is to, e.g., measure memory consumption across *learning curves*. 
In this setting, we often observe constant memory consumption across a full learning curve and also very low memory estimates close to `0`. 
We therefore discourage using memory metrics in this setting.
In addition, memory estimation was not always logged properly resulting in memory consumption imputed with `0`, which might lead to problems on some instances.

Multi-Objective Benchmarking
=======================

We observed that one some multi-objective benchmark problems, Pareto fronts can collapse, i.e., although we initially
assume that objectives are in competition we can find a single best point that optimizes all objectives simultaneously
and optimizers can then proceed to only further optimize a subset of all objectives because the other ones have
become irrelevant.

While we believe that this is still a well defined multi-objective optimization problem and multi-objective quality
indicators can still be computed (even if the resulting Pareto set contains only a single point) we want to note that
such problems can introduce some biases, i.e., favouring optimizers that explore the extreme regions of the Pareto front.

This mostly affects `rbv2_*` scenarios (mostly `rbv2_xgboost` and `rbv2_super`) and hardware metrics like `memory` but
can sometimes also be observed for `iaml_*` scenarios (e.g., if `nf` is included as an objective).

For `rbv2_*` problems, this is a result of the memory estimation (see above), but in general, this effect is intensified
by the extrapolation behavior of the surrogate.

We will try to address this issue in upcoming versions of `YAHPO Gym`.

Performance Metrics for rbv2_xgboost
=======================

We observed that our surrogate for the `rbv2_xgboost` scenarios tends to predict very good performance (e.g., `acc`, `auc`) for most `instances` for a large amount of hyperparameter configurations.
While XGBoost can be considered state-of-the art on tabular data and very good performance can be expected, this might also be a result of an unaccounted ceiling effect within the surrogate.

We are looking into this issue and will try to address it in upcoming versions of `YAHPO Gym`.

Noisy Surrogates
=======================

`YAHPO Gym` allows using *noisy* surrogates, this means that surrogates will predict targets from a distribution conditional on hyperparameters.
This internally works as follows: 
1. Given 3 neural networks `f_1` - `f_3` that predict targets from hyperparameters, run the prediction step 
2. Sample a vector alpha of length 3, such that each `alpha_i` is in `[0, 1]` and they sum to 1
3. The noisy prediction is given by the sum of neural network predictions weighted by the respective alpha

While this works well in theory, this was not tested thoroughly and the use of noisy surrogates is therefore discouraged at the moment.
Furthermore, we have not extensively tested whether all noisy surrogates indeed correctly return noisy predictions.
We will improve this in upcoming versions of `YAHPO Gym`.
