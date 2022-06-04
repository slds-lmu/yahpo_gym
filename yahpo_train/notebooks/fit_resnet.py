if __name__ == "__main__":

    import optuna
    import json
    import torch
    import random
    import numpy as np
    from tune_resnet import *
    from yahpo_gym import benchmark_set
    from yahpo_train.cont_scalers import *
    from yahpo_train.helpers import generate_all_test_set_metrics
    from functools import partial

    study_path = "/home/ru84tad2/"  # FIXME: needs adaption

    # tfms_list holds for each benchmark scenario (key) optional transformers that should be fixed and not tuned
    # taken from tune_resnet.py

    tfms_list = {}

    tfms_lcbench = {}
    tfms_list.update({"lcbench":tfms_lcbench})

    tfms_nb301 = {}
    tfms_nb301.update({"epoch":partial(ContTransformerMultScalar, m=1/98)})
    tfms_nb301.update({"val_accuracy":partial(ContTransformerMultScalar, m=1/100)})
    tfms_nb301.update({"runtime":ContTransformerRange})
    tfms_list.update({"nb301":tfms_nb301})

    tfms_rbv2_super = {}  # FIXME:
    tfms_list.update({"rbv2_super":tfms_rbv2_super})

    tfms_rbv2_svm = {}  # FIXME:
    tfms_list.update({"rbv2_svm":tfms_rbv2_svm})

    tfms_rbv2_xgboost = {}  # FIXME:
    tfms_list.update({"rbv2_xgboost":tfms_rbv2_xgboost})

    tfms_rbv2_ranger = {}  # FIXME:
    tfms_list.update({"rbv2_ranger":tfms_rbv2_ranger})

    tfms_rbv2_rpart = {}  # FIXME:
    tfms_list.update({"rbv2_rpart":tfms_rbv2_rpart})

    tfms_rbv2_glmnet = {}  # FIXME:
    tfms_list.update({"rbv2_glmnet":tfms_rbv2_glmnet})

    tfms_rbv2_aknn = {}  # FIXME:
    tfms_list.update({"rbv2_aknn":tfms_rbv2_aknn})

    tfms_iaml_super = {}
    [tfms_iaml_super.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_super":tfms_iaml_super})

    tfms_iaml_xgboost = {}
    [tfms_iaml_xgboost.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_xgboost":tfms_iaml_xgboost})

    tfms_iaml_ranger = {}
    [tfms_iaml_ranger.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_ranger":tfms_iaml_ranger})

    tfms_iaml_rpart = {}
    [tfms_iaml_rpart.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_rpart":tfms_iaml_rpart})

    tfms_iaml_glmnet = {}
    [tfms_iaml_glmnet.update({k:tfms_chain([ContTransformerInt, ContTransformerRange])}) for k in ["nf"]]
    tfms_list.update({"iaml_glmnet":tfms_iaml_glmnet})

    tfms_fcnet = {}  # FIXME:
    tfms_list.update({"fcnet":tfms_fcnet})

    tfms_fair_fgrrm = {}
    tfms_fair_fgrrm.update({"fpp":ContTransformerRange})
    tfms_list.update({"fair_fgrrm":tfms_fair_fgrrm})

    tfms_fair_rpart = {}
    tfms_fair_rpart.update({"fpp":ContTransformerRange})
    tfms_list.update({"fair_rpart":tfms_fair_rpart})

    tfms_fair_ranger = {}
    tfms_fair_ranger.update({"fpp":ContTransformerRange})
    tfms_list.update({"fair_ranger":tfms_fair_ranger})

    tfms_fair_xgboost = {}
    tfms_fair_xgboost.update({"fpp":ContTransformerRange})
    tfms_list.update({"fair_xgboost":tfms_fair_xgboost})

    tfms_fair_super = {}
    tfms_fair_super.update({"fpp":ContTransformerRange})
    tfms_list.update({"fair_super":tfms_fair_super})

    keys = ["fair_fgrrm", "fair_rpart", "fair_ranger", "fair_xgboost", "fair_super"]
    for key in keys:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        bs = 1024
        if key == "iaml_glmnet":
            bs = 128
        if key == "fair_fgrrm":
            bs = 64
        cc = cfg(key)
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise ValueError("No cuda device available. You probably do not want to fit on CPUs.")
        
        storage_name = "sqlite:///{}.db".format(study_path + "tune_" + key + "_resnet")
        study = optuna.load_study("tune_" + key + "_resnet", storage_name)
        best_params = study.best_params
        with open(cc.config_path + "/best_params_resnet.json", "w") as f:
            json.dump(best_params, f)

        dls_train = dl_from_config(cfg(key), bs=bs, frac=1., save_df_test=True, save_encoding=True)
        
        # tfms see tfms_list above
        l = fit_from_best_params_resnet(key, best_params=best_params, tfms_fixed=tfms_list.get(key), export=False, device="cuda:0", dls_train = dls_train)
        l.export_onnx(cfg(key), device="cuda:0")

        l_noisy = fit_from_best_params_resnet(key, best_params=best_params, tfms_fixed=tfms_list.get(key), noisy=True, export=False, device="cuda:0", dls_train = dls_train)
        l_noisy.export_onnx(cfg(key), device="cuda:0", suffix="noisy")

    keys = ["fair_fgrrm", "fair_rpart", "fair_ranger", "fair_xgboost", "fair_super"]
    for key in keys:
        bs = 1024
        if key == "iaml_glmnet":
            bs = 128
        if key == "fair_fgrrm":
            bs = 64
        #dl_from_config(cfg(key), bs=bs, frac=1., save_df_test=True, save_encoding=True)
        generate_all_test_set_metrics(key, model="model.onnx", save_to_csv=True)

