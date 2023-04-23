.. yahpo_gym documentation master file, created by
   sphinx-quickstart on Wed Sep 29 19:00:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

YAHPO Gym
=========

`YAHPO Gym` (Yet Another Hyperparameter Optimization Gym) is a collection of interesting problem sets for benchmark hyperparameter optimization / black-box optimization methods described in here: https://arxiv.org/abs/2109.03670.

`YAHPO Gym` consists of several `scenarios`, e.g., the collection of all benchmark instances in `lcbench` is a `scenario`.
Here, an `instance` is the concrete task of optimizing hyperparameters of a neural network on a given dataset from OpenML.

**A note on OpenML IDs**

Currently, the ``rbv2_*``, ``lcbench``, and ``iaml_*`` scenarios contain instances based on OpenML datasets.
For ``rbv2_*`` and ``iaml_*`` scenarios, the `task_id` parameter of the `ConfigSpace` corresponds to the OpenML **dataset** identifier (i.e., this is the **dataset** id and **not** the task id).
To query meta information, use https://www.openml.org/d/<dataset_id>.
For the ``lcbench`` scenario, the `OpenML_task_id` parameter of the `ConfigSpace` directly corresponds to OpenML **tasks** identifier (i.e., this is the **task** id and **not** the dataset id).
To query meta information, use https://www.openml.org/t/<task_id>. 

For other questions, see the *frequently asked questions* section.

**Why should I use it?**

`YAHPO Gym` provides blazing fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks across a large variety of problems.
Our library makes use of ConfigSpace(https://automl.github.io/ConfigSpace/) to describe the hyperparameter space to optimize and can thus be seamlessly integrated into many existing projects e.g. HpBandSter(https://github.com/automl/HpBandSter).


.. image:: _static/anytime_average_rank_mf.jpg
    :align: center
    :width: 800px
    :alt: alternate text


Module Documentation
***********************
.. toctree::
   :maxdepth: 1

   yahpo_gym


Getting Started
***********************
.. toctree::
   :maxdepth: 1

   getting_started
   frequently_asked

Scenarios & Instances
***********************
.. toctree::
   :maxdepth: 1

   scenarios


Examples
***********************
.. toctree::
   :maxdepth: 1

   examples

Extending YAHPO Gym
***********************
.. toctree::
   :maxdepth: 1

   extending
   

Data
***********************
The relevant surrogate models and meta-data can be obtained from yahpo_data (https://github.com/slds-lmu/yahpo_data).
We will release new versioned updates of the repository in the future.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
