.. yahpo_gym documentation master file, created by
   sphinx-quickstart on Wed Sep 29 19:00:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

YAHPO GYM
=========

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) is a collection of interesting problem sets for benchmark hyperparameter optimization / black-box optimization methods described in here: <https://arxiv.org/abs/2109.03670>.

YAHPO GYM consists of several `scenarios`, e.g. the collection of all benchmark instances in `lcbench` is a `scenario`.
Here, an ``instance`` is the concrete task of optimizing hyperparameters of a neural network on a given dataset from OpenML.

**Why should I use it?**

**YAHPO GYM** (Yet Another Hyperparameter Optimization GYM) provides blazingly fast and simple access to a variety of interesting benchmark problems for hyperparameter optimization.
Since all our benchmarks are based on surrogate models that approximate the underlying HPO problems with very high fidelity, function evaluations are fast and memory friendly allowing for fast benchmarks
across a large variety of problems.
Our library makes use of ConfigSpace(https://automl.github.io/ConfigSpace/> to describe the hyperparameter space to optimize and can thus be seamlessly integrated into
many existing projects e.g. HpBandSter(https://github.com/automl/HpBandSter).


.. image:: _static/results.png
    :align: center
    :width: 800px
    :alt: alternate text


Module Documentation
***********************
.. toctree::
   :maxdepth: 1

   yahpo_gym

Scenarios & Instances
***********************
.. toctree::
   :maxdepth: 1

   scenarios

Extending YAHO Gym
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
