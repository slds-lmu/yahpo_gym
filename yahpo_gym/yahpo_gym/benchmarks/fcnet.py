from yahpo_gym.configuration import config_dict, cfg

_fcnet_dict = {
    "config_id": "fcnet",
    "y_names": ["valid_loss", "valid_mse", "runtime", "n_params"],
    "y_minimize": [True, True, True, True],
    "cont_names": [
        "epoch",
        "replication",
        "batch_size",
        "dropout_1",
        "dropout_2",
        "init_lr",
        "n_units_1",
        "n_units_2",
    ],
    "cat_names": ["task", "activation_fn_1", "activation_fn_2", "lr_schedule"],
    "instance_names": "task",
    "fidelity_params": ["epoch"],
    "runtime_name": "runtime",
    "citation": [
        "Falkner, S., Klein, A. & Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. Proceedings of the 35th International Conference on Machine Learning, in Proceedings of Machine Learning Research, 80, 1437-1446."
    ],
}
config_dict.update({"fcnet": _fcnet_dict})
