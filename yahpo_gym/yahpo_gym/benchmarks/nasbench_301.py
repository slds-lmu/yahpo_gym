from yahpo_gym.configuration import config_dict, cfg

_nasbench_301_dict = {
    'config_id' : 'nb301',
    'y_names' : ['val_accuracy', 'runtime'],
    'y_minimize' : [False, True],
    'cont_names': ['epoch'],
    'cat_names': ['dataset', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_0', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_1', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_2', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_3', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_4', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_5',
                  'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_6','NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_7', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_8', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_9', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_10', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_11',
                  'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_12', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_normal_13', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_0','NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_1', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_2', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_3', 
                  'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_4', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_5', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_6', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_7', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_8', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_9', 
                  'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_10', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_11', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_12', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_edge_reduce_13', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_normal_3', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_normal_4',
                  'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_normal_5', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_reduce_3', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_reduce_4', 'NetworkSelectorDatasetInfo_COLON_darts_COLON_inputs_node_reduce_5'],
    'fidelity_params': ['epoch'],
    'runtime_name': 'runtime',
    'instance_names': 'dataset'
}
config_dict.update({'nb301': _nasbench_301_dict})
