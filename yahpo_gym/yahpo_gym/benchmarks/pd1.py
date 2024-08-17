from yahpo_gym.configuration import config_dict

_pd1_dict = {
    "config_id": "pd1",
    "y_names": [
        "valid_error",
        "test_error",
        "eval_time",
        "eval_time_increase",
    ],
    "y_minimize": [True, True, True, True],
    "cont_names": [
        "epoch",
        "lr_initial_value",
        "lr_decay_steps_factor",
        "lr_power",
        "one_minus_momentum",
    ],
    "cat_names": ["task"],
    "instance_names": "task",
    "fidelity_params": ["epoch"],
    "runtime_name": "eval_time",
    "citation": [
        "Wang, Z., Dahl, G. E., Swersky, K., Lee, C., Nado, Z., Gilmer, J., Snoek, J. & Ghahramani, Z. (2021). Pre-trained Gaussian Processes for Bayesian Optimization. arXiv preprint arXiv:2109.08215."
    ],
}
config_dict.update({"pd1": _pd1_dict})
