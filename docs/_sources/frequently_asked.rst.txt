Frequently asked questions
************************

In the following, we will try to keep a list of frequently asked questions.


OpenML task_id and dataset_id
=======================

Currently, the ``rbv2_*``, ``lcbench``, and ``iaml_*`` scenarios contain instances based on OpenML datasets.
For ``rbv2_*`` and ``iaml_*`` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the ``lcbench`` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>.

Reproducbility
=======================

`YAHPO Gym` relies on static neural networks compiled via `ONNX`. 
This should result in reproducible results given equal hardware and software versions.
Unfortunately, `ONNX`models do not always yield reproducible results across different hardware.
This is, e.g. discussed in https://github.com/microsoft/onnxruntime/issues/12086.

In practice, we have not observed relevant differences between different hardware versions, but this might help to explain observations
regarding a lack of exact reproducbility.

rbv2_* data source
=======================

The rbv2_* data and data collection method is described in Binder et al., 2020 Collecting Empirical Data about Hyperparameters <https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_63.pdf>.
Benchmarks with the `iaml_*`prefix are largely based on the same methodology. 
Additional details for all benchmarks can be found in the YAHPO Gym paper's supplementary material.

Using F1 scores for rbv2_* and iaml_*
=======================

F1 scores in the rbv2_* scenarios are only available for binary class datasets. 
On multiclass datasets, the corresponding F1 score is imputed with `0` and returned by the surrogate model.
The information on which `id` corresponds to a multiclass dataset can be obtained from the entry `is_multicrit` in `BenchmarkSet.config.config`.

Memory estimation for rbv2_*
=======================

For the rbv2_* settings, memory consumption was estimated by observing the memory consumption during training via `/usr/bin/time`. 
This estimates the `Maximum resident size`.
In general, we assume that this is provides a coarse estimation of the processes memory consumption.
However, it does not seem to work if the goal, e.g. is to measure memory consumption across *learning curves*. 
In this setting, we often observe constant memory consumption across a full learning curve. 
We therefore discourage using memory metrics in this setting.
In addition, memory estimation was not always logged properly resulting in memory consumption imputed with `0`, which might lead to problems on some instances.
