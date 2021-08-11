from yahpo_gym.configuration import config_dict, cfg

# Default dict, holds for all 'rbv2_' benchmarks
_rbv2_dict = {
    'y_names' : ['mmce', 'f1', 'auc', 'logloss', 'timetrain', 'timepredict'],
    'y_minimize' : [True, False, False, True, True, True],
    'fidelity_params': ['trainsize', 'repl'],
    'runtime_name': 'traintime'
}

# SVM (LIBSVM)
_rbv2_svm = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_svm',
    'cont_names': ['cost', 'gamma', 'tolerance', 'degree', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'kernel', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_svm' : _rbv2_svm})

# Ranger (Random Forest) 
_rbv2_ranger = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_ranger',
    'cont_names': ['num.trees', 'sample.fraction', 'mtry.power', 'min.node.size', 'num.random.splits', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'respect.unordered.factors', 'splitrule', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_ranger' : _rbv2_ranger})

# Decisiont Trees
_rbv2_rpart = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_rpart',
    'cont_names': ['cp', 'maxdepth', 'minbucket', 'minsplit', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_svm' : _rbv2_svm})


# ElasticNet
_rbv2_glmnet = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_glmnet',
    'cont_names': ['alpha', 's', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_glmnet' : _rbv2_glmnet})

# XGBOOST
_rbv2_xgboost = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_xgboost',
    'cont_names': ['nrounds', 'eta', 'gamma', 'lambda',  'alpha', 'subsample', 'max_depth', 'min_child_weight','colsample_bytree', 'colsample_bylevel', 'rate_drop', 'skip_drop', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'booster', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_xgboost' : _rbv2_xgboost})

# AKNN
_rbv2_aknn = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_aknn',
    'cont_names': ['k','M', 'aknn.ef', 'ef_construction', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'distance', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_aknn' : _rbv2_aknn})


# Superset Model Multiplexer for SVM, RF, DT, XGB  # UNFINISHED
_rbv2_super = _rbv2_dict.copy().update({
    'config_id' : 'rbv2_super',
    'cont_names': ['svm.cost', 'svm.gamma', 'svm.tolerance', 'svm.degree', 
                   'glmnet.alpha', 'glmnet.s',
                   'rpart.cp', 'rpart.maxdepth', 'rpart.minbucket', 'rpart.minsplit',
                   'ranger.num.trees', 'ranger.sample.fraction','ranger.mtry.power', 'ranger.min.node.size',   'ranger.num.random.splits',
                   'aknn.k','aknn.M', 'aknn.ef', 'aknn.ef_construction'
                   'xgboost.nrounds', 'xgboost.eta', 'xgboost.gamma', 'xgboost.lambda',  'xgboost.alpha', 'xgboost.subsample', 'xgboost.max_depth', 'xgboost.min_child_weight',
                   'xgboost.colsample_bytree', 'xgboost.colsample_bylevel', 'xgboost.rate_drop', 'xgboost.skip_drop', 
                   'trainsize', 'repl'],
    'cat_names': ['task_id', 'learner', 'svm.kernel', 'ranger.respect.unordered.factors', 'ranger.splitrule', 'aknn.distance', 'xgboost.booster', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_super' : _rbv2_super})
