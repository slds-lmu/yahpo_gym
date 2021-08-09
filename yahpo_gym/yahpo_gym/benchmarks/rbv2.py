from yahpo_gym.configuration import config_dict, cfg

# Default dict, holds for all 'rbv2_' benchmarks
_rbv2_dict = {
    'y_names' : ['mmce', 'f1', 'auc', 'logloss', 'timetrain', 'timepredict'],
    'y_minimize' : [True, False, False, True, True, True],
    'fidelity_params': ['trainsize', 'repl'],
    'runtime_name': 'traintime'
}

# SVM (LIBSVM)
_rbv2_svm = copy(_rbv2_dict).update({
    'config_id' : 'rbv2_svm',
    'cont_names': ['cost', 'gamma', 'tolerance', 'degree', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'kernel', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_svm' : _rbv2_svm})

# Ranger (Random Forest) 
_rbv2_ranger = copy(_rbv2_dict).update({
    'config_id' : 'rbv2_ranger',
    'cont_names': ['num.trees', 'sample.fraction', 'mtry.power', 'min.node.size', 'num.random.splits', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'respect.unordered.factors', 'splitrule', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_ranger' : _rbv2_ranger})

# Decision Tree  # UNFINISHED
_rbv2_rpart = copy(_rbv2_dict).update({
    'config_id' : 'rbv2_rpart',
    'cont_names': ['cost', 'gamma', 'tolerance', 'degree', 'trainsize', 'repl', 'max_units', 'max_dropout'],
    'cat_names': ['task_id', 'kernel', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_svm' : _rbv2_svm})


# ElasticNet
_rbv2_glmnet = copy(_rbv2_dict).update({
    'config_id' : 'rbv2_glmnet',
    'cont_names': ['alpha', 's', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_glmnet' : _rbv2_glmnet})

# XGBoost # UNFINISHED
_rbv2_xgboost = copy(_rbv2_dict).update({
    'config_id' : 'rbv2_glmnet',
    'cont_names': ['alpha', 's', 'trainsize', 'repl'],
    'cat_names': ['task_id', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_glmnet' : _rbv2_glmnet})


# Superset Model Multiplexer for SVM, RF, DT, XGB  # UNFINISHED
_rbv2_super = copy(_rbv2_dict).update({
    'config_id' : 'rbv2_super',
    'cont_names': ['cost', 'gamma', 'tolerance', 'degree', 'trainsize', 'repl', 'max_units', 'max_dropout'],
    'cat_names': ['task_id', 'kernel', 'num.impute.selected.cpo']
})
config_dict.update({'rbv2_super' : _rbv2_super})
