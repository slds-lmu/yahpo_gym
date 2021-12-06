from yahpo_gym.configuration import config_dict, cfg

_lcbench_dict = {
    'config_id' : 'lcbench',
    'y_names' : ['time', 'val_accuracy', 'val_cross_entropy', 'val_balanced_accuracy', 'test_cross_entropy', 'test_balanced_accuracy'],
    'y_minimize' : [True, False, True, False, True, False],
    'cont_names': ['epoch', 'batch_size', 'learning_rate', 'momentum', 'weight_decay', 'num_layers', 'max_units', 'max_dropout'],
    'cat_names': ['OpenML_task_id'],
    'instance_names' : 'OpenML_task_id',
    'fidelity_params': ['epoch'],
    'runtime_name': 'time',
    'citation' : 'L. Zimmer, M. Lindauer, and F. Hutter, “Auto-pytorch tabular: Multi-fidelity metalearning for efficient and robust autodl,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 9, pp. 3079 – 3090, 2021.'
}
config_dict.update({'lcbench' : _lcbench_dict})
