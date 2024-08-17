from yahpo_gym.configuration import config_dict

_fcnet_dict = {
    "config_id": "fcnet",
    "y_names": ["valid_mse", "runtime", "runtime_increase"],
    "y_minimize": [True, True, True],
    "cont_names": [
        "epoch",
        "batch_size",
        "dropout_1",
        "dropout_2",
        "init_lr",
        "n_units_1",
        "n_units_2",
    ],
    "cat_names": [
        "task",
        "repl",
        "activation_fn_1",
        "activation_fn_2",
        "lr_schedule",
    ],
    "instance_names": "task",
    "fidelity_params": ["epoch", "repl"],
    "runtime_name": "runtime",
    "citation": ["FIXME:"],
}
config_dict.update({"fcnet": _fcnet_dict})
