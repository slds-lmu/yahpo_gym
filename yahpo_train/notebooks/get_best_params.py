import optuna
import json
from yahpo_gym import benchmark_set

keys = [
    "fair_fgrrm",
    "fair_rpart",
    "fair_ranger",
    "fair_xgboost",
    "fair_super",
    "iaml_glmnet",
    "iaml_rpart",
    "iaml_ranger",
    "iaml_xgboost",
    "iaml_super",
]
for key in keys:
    bench = benchmark_set.BenchmarkSet(key)
    storage = "sqlite:///{}.db".format("tune_" + key + "_resnet")
    study = optuna.load_study(study_name="tune_" + key + "_resnet", storage=storage)
    best_params = study.best_params
    with open(bench.config.config_path + "/best_params_resnet_v2.json", "w") as f:
        json.dump(best_params, f)
