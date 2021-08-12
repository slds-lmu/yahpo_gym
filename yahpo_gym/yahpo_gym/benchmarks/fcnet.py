from yahpo_gym.configuration import config_dict

_fcnet_dict = {
    'config_id' : 'fcnet',
    'y_names' : ['valid_loss', 'runtime','n_params'],
    'y_minimize' : [True, False, True, False, True, False],
    'cont_names': ['epoch', 'batch_size', 'dropout_1', 'dropout_2', 'init_lr', 'n_units_1', 'n_units_2', 'replication'],
    'cat_names': ['task', 'activation_fn_1' , 'activation_fn_2', 'lr_schedule'],
    'instance_names': 'task',
    'fidelity_params': ['epoch', 'replication'],
    'runtime_name': 'runtime'
}
config_dict.update({'fcnet' : _fcnet_dict})
