
Extending YAHPO Gym
************************

This document describes how to extend `YAHPO Gym` with new configurations and scenarios.
PR's with new problems are welcome! See below on how to add new problem sets.

There are two main steps that need to be added for a new `scenario` locally.

1. **Add new configuration**:

The `yahpo_gym.configuration` module contains the relevant meta-data required to fit and predict surrogates.
This includes a `config_id` (an id variable); `y_names`, `cont_names` and `cat_names` describing names of the target,
numeric and categorical variables respectively and so on. 

.. code-block:: python

    from yahpo_gym.configuration import config_dict, cfg
    _new_dict = {
        'scenario' : '<ID>', # Add some Id
        'y_names' : ['valid_loss', ...], # names of target variables
        'y_minimize' : [True, ...], # whether target variables should be minimized
        'cont_names': ['epoch', 'learning_rate'], # numeric variables
        'cat_names': ['task', 'activation_fn_1'], # categorical variables
        'instance_names': 'task', # which column describes the instance.
        'fidelity_params': ['epoch', 'replication'], # which columns are fidelity parameters
        'runtime_name': 'runtime', # which columns predict runtime (if available)s
        'citation': 'CITE' # Reference if available
    }
    config_dict.update({'<ID>' : _new_dict})

Users can now instantiate a `Benchmark` or `config` with this `<ID>`, e.g. using `cfg("<ID>")`.



2. **Add the required metadata**:
This includes fitted surrogates and parameter spaces that we'll add to our ``data_path``.
For an example on how the final data looks like implemented models
and hyperparameters see https://github.com/slds-lmu/yahpo_data.

- Add a folder using the benchmark's `<ID>` to your ``data_path``.
- Add the data required to train the surrogate model as `data.csv`.
- Add the `ConfigSpace.ConfigSpace` in the `.json` format as `config_space.json`.
  This defines all hyperparameters and instances used throughout the optimization.
- The `yahpo_train` module can now be used to train a new surrogate. See the `notebooks` folder
  for training code. This produces the `encoding.json` and `model.onnx` files required for prediction.
- For compatibility with the `R` package, a `paradox::ParamSet` similar to the `ConfigSpace` is required.
  Create a file `param_set.R` containing the `ParamSet`.


Include your benchmark in YAHPO Gym
====================================

Once you have tested your benchmark, you can create a `PR` to the `yahpo_gym` and `yahpo_data` 
repositories. 
Note that the `yahpo_data` PR should not include the data, but instead only the model (`.onnx`) and parameter
spaces. Please include information on the problem, the surrogate's performance and other relevant info in 
the PR.
