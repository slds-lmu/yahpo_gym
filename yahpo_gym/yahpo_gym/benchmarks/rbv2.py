from yahpo_gym.configuration import config_dict, cfg

# Default dict, holds for all 'rbv2_' benchmarks
_rbv2_dict = {
    "y_names": [
        "acc",
        "bac",
        "auc",
        "brier",
        "f1",
        "logloss",
        "timetrain",
        "timepredict",
        "memory",
    ],
    "y_minimize": [False, False, False, True, True, True, True, True, True],
    "fidelity_params": ["trainsize", "repl"],
    "instance_names": "task_id",
    "runtime_name": "timetrain",
    "memory_name": "memory",
    "is_multicrit": {
        "3": False,
        "11": True,
        "12": True,
        "14": True,
        "15": False,
        "16": True,
        "18": True,
        "22": True,
        "23": True,
        "24": False,
        "28": True,
        "29": False,
        "31": False,
        "32": True,
        "37": False,
        "38": False,
        "42": True,
        "44": False,
        "46": True,
        "50": False,
        "54": True,
        "60": True,
        "151": False,
        "181": True,
        "182": True,
        "188": True,
        "300": True,
        "307": True,
        "312": False,
        "334": False,
        "375": True,
        "377": True,
        "458": True,
        "469": True,
        "470": False,
        "1040": False,
        "1049": False,
        "1050": False,
        "1053": False,
        "1056": False,
        "1063": False,
        "1067": False,
        "1068": False,
        "1111": False,
        "1220": False,
        "1457": True,
        "1461": False,
        "1462": False,
        "1464": False,
        "1468": True,
        "1475": True,
        "1476": True,
        "1478": True,
        "1479": False,
        "1480": False,
        "1485": False,
        "1486": False,
        "1487": False,
        "1489": False,
        "1494": False,
        "1497": True,
        "1501": True,
        "1510": False,
        "1515": True,
        "1590": False,
        "4134": False,
        "4135": False,
        "4154": False,
        "4534": False,
        "4538": True,
        "4541": True,
        "6332": False,
        "23381": False,
        "23512": False,
        "40496": True,
        "40498": True,
        "40499": True,
        "40536": False,
        "40668": True,
        "40670": True,
        "40701": False,
        "40900": False,
        "40966": True,
        "40975": True,
        "40978": False,
        "40979": True,
        "40981": False,
        "40982": True,
        "40983": False,
        "40984": True,
        "40994": False,
        "41138": False,
        "41142": False,
        "41143": False,
        "41146": False,
        "41156": False,
        "41157": False,
        "41159": False,
        "41161": False,
        "41162": False,
        "41163": True,
        "41164": True,
        "41212": True,
        "41278": True,
    },
    "citation": [
        "Binder M., Pfisterer F. & Bischl B. (2020). Collecting Empirical Data About Hyperparameters for Data Driven AutoML. 7th ICML Workshop on Automated Machine Learning."
    ],
}

# SVM (LIBSVM)
_rbv2_svm = _rbv2_dict.copy()
_rbv2_svm.update(
    {
        "config_id": "rbv2_svm",
        "cont_names": ["cost", "gamma", "tolerance", "degree", "trainsize", "repl"],
        "cat_names": ["task_id", "kernel", "num.impute.selected.cpo"],
    }
)
config_dict.update({"rbv2_svm": _rbv2_svm})

# Ranger (Random Forest)
_rbv2_ranger = _rbv2_dict.copy()
_rbv2_ranger.update(
    {
        "config_id": "rbv2_ranger",
        "cont_names": [
            "num.trees",
            "sample.fraction",
            "mtry.power",
            "min.node.size",
            "num.random.splits",
            "trainsize",
            "repl",
        ],
        "cat_names": [
            "task_id",
            "respect.unordered.factors",
            "splitrule",
            "num.impute.selected.cpo",
        ],
    }
)
config_dict.update({"rbv2_ranger": _rbv2_ranger})

# Decision Trees
_rbv2_rpart = _rbv2_dict.copy()
_rbv2_rpart.update(
    {
        "config_id": "rbv2_rpart",
        "cont_names": ["cp", "maxdepth", "minbucket", "minsplit", "trainsize", "repl"],
        "cat_names": ["task_id", "num.impute.selected.cpo"],
    }
)
config_dict.update({"rbv2_rpart": _rbv2_rpart})

# ElasticNet
_rbv2_glmnet = _rbv2_dict.copy()
_rbv2_glmnet.update(
    {
        "config_id": "rbv2_glmnet",
        "cont_names": ["alpha", "s", "trainsize", "repl"],
        "cat_names": ["task_id", "num.impute.selected.cpo"],
    }
)
config_dict.update({"rbv2_glmnet": _rbv2_glmnet})

# XGBOOST
_rbv2_xgboost = _rbv2_dict.copy()
_rbv2_xgboost.update(
    {
        "config_id": "rbv2_xgboost",
        "cont_names": [
            "nrounds",
            "eta",
            "gamma",
            "lambda",
            "alpha",
            "subsample",
            "max_depth",
            "min_child_weight",
            "colsample_bytree",
            "colsample_bylevel",
            "rate_drop",
            "skip_drop",
            "trainsize",
            "repl",
        ],
        "cat_names": ["task_id", "booster", "num.impute.selected.cpo"],
    }
)
config_dict.update({"rbv2_xgboost": _rbv2_xgboost})

# AKNN
_rbv2_aknn = _rbv2_dict.copy()
_rbv2_aknn.update(
    {
        "config_id": "rbv2_aknn",
        "cont_names": ["k", "M", "ef", "ef_construction", "trainsize", "repl"],
        "cat_names": ["task_id", "distance", "num.impute.selected.cpo"],
    }
)
config_dict.update({"rbv2_aknn": _rbv2_aknn})


# Superset Model Multiplexer for SVM, RF, DT, XGBOOST
_rbv2_super = _rbv2_dict.copy()
_rbv2_super.update(
    {
        "config_id": "rbv2_super",
        "cont_names": [
            "svm.cost",
            "svm.gamma",
            "svm.tolerance",
            "svm.degree",
            "glmnet.alpha",
            "glmnet.s",
            "rpart.cp",
            "rpart.maxdepth",
            "rpart.minbucket",
            "rpart.minsplit",
            "ranger.num.trees",
            "ranger.sample.fraction",
            "ranger.mtry.power",
            "ranger.min.node.size",
            "ranger.num.random.splits",
            "aknn.k",
            "aknn.M",
            "aknn.ef",
            "aknn.ef_construction",
            "xgboost.nrounds",
            "xgboost.eta",
            "xgboost.gamma",
            "xgboost.lambda",
            "xgboost.alpha",
            "xgboost.subsample",
            "xgboost.max_depth",
            "xgboost.min_child_weight",
            "xgboost.colsample_bytree",
            "xgboost.colsample_bylevel",
            "xgboost.rate_drop",
            "xgboost.skip_drop",
            "trainsize",
            "repl",
        ],
        "cat_names": [
            "task_id",
            "learner_id",
            "svm.kernel",
            "ranger.respect.unordered.factors",
            "ranger.splitrule",
            "aknn.distance",
            "xgboost.booster",
            "num.impute.selected.cpo",
        ],
    }
)
config_dict.update({"rbv2_super": _rbv2_super})
