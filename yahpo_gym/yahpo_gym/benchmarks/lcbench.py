from yahpo_gym.configuration import config_dict, cfg

_lcbench_dict = {
    'config_id' : 'lcbench',
    'y_names' : ['time', 'val_accuracy', 'val_cross_entropy', 'val_balanced_accuracy', 'test_cross_entropy', 'test_balanced_accuracy'],
    'y_minimize' : [True, False, True, False, True, False],
    'cont_names': ['epoch', 'batch_size', 'learning_rate', 'momentum', 'weight_decay', 'num_layers', 'max_units', 'max_dropout'],
    'cat_names': ['OpenML_task_id'],
    'instance_names' : 'OpenML_task_id',
    'fidelity_params': ['epoch'],
    'runtime_name': 'time'
}
config_dict.update({'lcbench' : _lcbench_dict})
