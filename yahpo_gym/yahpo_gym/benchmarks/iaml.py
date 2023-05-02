from yahpo_gym.configuration import config_dict, cfg

# Default dict, holds for all "iaml_" benchmarks
_iaml_dict = {
    "y_names": [
        "mmce",
        "f1",
        "auc",
        "logloss",
        "ramtrain",
        "rammodel",
        "rampredict",
        "timetrain",
        "timepredict",
        "mec",
        "ias",
        "nf",
    ],
    "y_minimize": [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ],
    "fidelity_params": ["trainsize"],
    "instance_names": "task_id",
    "runtime_name": "timetrain",
    "memory_name": "rammodel",
    "citation": None,
}

# Ranger (Random Forest)
_iaml_ranger = _iaml_dict.copy()
_iaml_ranger.update(
    {
        "config_id": "iaml_ranger",
        "cont_names": [
            "num.trees",
            "sample.fraction",
            "mtry.ratio",
            "min.node.size",
            "num.random.splits",
            "trainsize",
        ],
        "cat_names": ["task_id", "replace", "respect.unordered.factors", "splitrule"],
        "hierarchical": True,
    }
)
config_dict.update({"iaml_ranger": _iaml_ranger})

# Decision Trees
_iaml_rpart = _iaml_dict.copy()
_iaml_rpart.update(
    {
        "config_id": "iaml_rpart",
        "cont_names": ["cp", "maxdepth", "minbucket", "minsplit", "trainsize"],
        "cat_names": ["task_id"],
    }
)
config_dict.update({"iaml_rpart": _iaml_rpart})

# ElasticNet
_iaml_glmnet = _iaml_dict.copy()
_iaml_glmnet.update(
    {
        "config_id": "iaml_glmnet",
        "cont_names": ["alpha", "s", "trainsize"],
        "cat_names": ["task_id"],
    }
)
config_dict.update({"iaml_glmnet": _iaml_glmnet})

# XGBOOST
_iaml_xgboost = _iaml_dict.copy()
_iaml_xgboost.update(
    {
        "config_id": "iaml_xgboost",
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
        ],
        "cat_names": ["task_id", "booster"],
        "hierarchical": True,
    }
)
config_dict.update({"iaml_xgboost": _iaml_xgboost})

# Superset Model Multiplexer
_iaml_super = _iaml_dict.copy()
_iaml_super.update(
    {
        "config_id": "iaml_super",
        "cont_names": [
            "ranger.num.trees",
            "ranger.sample.fraction",
            "ranger.mtry.ratio",
            "ranger.min.node.size",
            "ranger.num.random.splits",
            "rpart.cp",
            "rpart.maxdepth",
            "rpart.minbucket",
            "rpart.minsplit",
            "glmnet.alpha",
            "glmnet.s",
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
        ],
        "cat_names": [
            "task_id",
            "learner",
            "ranger.replace",
            "ranger.respect.unordered.factors",
            "ranger.splitrule",
            "xgboost.booster",
        ],
        "hierarchical": True,
    }
)
config_dict.update({"iaml_super": _iaml_super})
