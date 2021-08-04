from yahpo_gym.configuration import config_dict, cfg

_lcbench_dict = {
    'config_id' : "lcbench",
    'y_names' : "",
    "fidelity_params": ""
}
config_dict.update({'lcbench' : _lcbench_dict})

if __name__ == '__main__':
    print(cfg("lcbench"))
