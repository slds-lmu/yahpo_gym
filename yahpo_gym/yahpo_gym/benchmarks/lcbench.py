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

if __name__ == '__main__':
    from yahpo_gym.benchmark_set import BenchmarkSet
    b = BenchmarkSet('lcbench')
    b.config_space
    # hposet = {'OpenML_task_id': 3, 'epoch':100, 'batch_size':10, 'learning_rate':.1, 'momentum':.9, 'weight_decay':.01, 'num_layers':3, 'max_units':100, 'max_dropout':.6}
    # res = b.objective_function(hposet)
    print(res)
