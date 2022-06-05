from yahpo_gym.configuration import config_dict, cfg

# Default dict, holds for all 'fair_' benchmarks
# Note fpp (Calders-Wevers gap) exlcuded for now
_fair_dict = {
    'y_names' : ['mmce', 'f1', 'feo', 'fpredp', 'facc', 'ftpr', 'ffomr', 'ffnr', 'timetrain'],
    'y_minimize' : [True, False, True, True, True, True, True, True, True],
    'fidelity_params': ['trainsize'],
    'instance_names': 'task_id',
    'runtime_name': 'timetrain'
}

## Fgrrm
_fair_fgrrm = _fair_dict.copy()
_fair_fgrrm.update({
    'config_id' : 'fair_fgrrm',
    'cont_names' : ['lambda', 'unfairness', 'trainsize'],
    'cat_names' : ['task_id', 'definition'],
})
config_dict.update({'fair_fgrrm' : _fair_fgrrm})

## Decision Trees
_fair_rpart = _fair_dict.copy()
_fair_rpart.update({
    'config_id' : 'fair_rpart',
    'cont_names' : ['cp', 'maxdepth', 'minbucket', 'minsplit', 'reweighing_os_alpha', 'EoD_alpha', 'trainsize'],
    'cat_names' : ['task_id', "pre_post"],
    'hierarchical': True
})
config_dict.update({'fair_rpart' : _fair_rpart})

## Ranger (Random Forest) 
_fair_ranger = _fair_dict.copy()
_fair_ranger.update({
    'config_id' : 'fair_ranger',
    'cont_names' : ['min.node.size', 'mtry.ratio', 'num.random.splits', 'num.trees', 'sample.fraction', 'reweighing_os_alpha', 'EoD_alpha', 'trainsize'],
    'cat_names' : ['replace', 'respect.unordered.factors', 'splitrule', 'task_id', "pre_post"],
    'hierarchical': True
})
config_dict.update({'fair_ranger' : _fair_ranger})

## XGBOOST
_fair_xgboost = _fair_dict.copy()
_fair_xgboost.update({
    'config_id' : 'fair_xgboost',
    'cont_names' : ['nrounds', 'eta', 'gamma', 'lambda', 'alpha', 'subsample', 'max_depth', 'min_child_weight', 'colsample_bytree', 'colsample_bylevel', 'reweighing_os_alpha', 'EoD_alpha', 'trainsize'],
    'cat_names' : ['booster', 'task_id', "pre_post"],
    'hierarchical': True
})
config_dict.update({'fair_xgboost' : _fair_xgboost})
 
## Superset Model Multiplexer
_fair_super = _fair_dict.copy()
_fair_super.update({
    'config_id' : 'fair_super',
    'cont_names' : ['fgrrm.lambda', 'fgrrm.unfairness',
        'rpart.cp', 'rpart.maxdepth', 'rpart.minbucket', 'rpart.minsplit', 
        'ranger.min.node.size', 'ranger.mtry.ratio', 'ranger.num.random.splits', 'ranger.num.trees', 'ranger.sample.fraction',
        'xgboost.nrounds', 'xgboost.eta', 'xgboost.gamma', 'xgboost.lambda', 'xgboost.alpha', 'xgboost.subsample', 'xgboost.max_depth', 'xgboost.min_child_weight', 'xgboost.colsample_bytree', 'xgboost.colsample_bylevel',
        'reweighing_os_alpha', 'EoD_alpha', 'trainsize'],
    'cat_names' : ['fgrrm.definition', 'ranger.replace', 'ranger.respect.unordered.factors', 'ranger.splitrule', 'xgboost.booster', 'learner', 'task_id', "pre_post"],
    'hierarchical': True
})
config_dict.update({'fair_super' : _fair_super})

