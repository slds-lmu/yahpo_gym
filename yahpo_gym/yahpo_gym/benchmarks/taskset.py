from yahpo_gym.configuration import config_dict, cfg

_task_set_dict = {
    'config_id' : 'task_set',
    'y_names' : ['train','valid1','valid2','test'],
    'y_minimize' : [True, False, True, False, True, False],
    'cont_names': ['epoch', 'replication', 'learning_rate', 'beta1', 'beta2', 'epsilon', 'l1', 'l2', 'linear_decay', 'exponential_decay'],
    'cat_names': ['task_name', 'optimizer' ],
    'instance_names': 'task_name',
    'fidelity_params': ['epoch', 'replication'],
    'runtime_name': None,
    'citation' : 'L. Metz, N. Maheswaranathan, R. Sun, C. D. Freeman, B. Poole, and J. Sohl-Dickstein, “TaskSet: A Dataset of Optimization Tasks,” 2020.'
}
config_dict.update({'taskset' : _task_set_dict})
