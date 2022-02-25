from yahpo_gym import benchmark_set
from yahpo_gym.benchmarks import lcbench, nb301, rbv2, fcnet, taskset, iaml
from yahpo_gym.configuration import cfg
from yahpo_train.metrics import *
import numpy as np
import pandas as pd
import torch

def chunk(n, size):
    m = n // size
    result = []
    lower = 0
    upper = 0
    for i in range(0, m - 1):
        lower = upper
        upper = lower + size
        result.append([lower, upper])
    lower = upper
    upper = n
    result.append([lower, upper])
    return result

def get_set_metrics(key, set="test", model=None, instance=None, chunk_size=10000):
    # NOTE: this is somewhat slow because we first map the points to the right format for the onnx model and then use the onnx model for prediction
    bench = benchmark_set.BenchmarkSet(key, active_session=True)
    if model is not None:
        bench.config.config.update({"model":model})
    bench.check = False  # see note below
    dtypes = dict(zip(bench.config.cat_names, ["object"] * len(bench.config.cat_names)))
    dtypes.update(dict(zip(bench.config.cont_names+bench.config.y_names, ["float32"] * len(bench.config.cont_names+bench.config.y_names))))
    if set == "test":
        df = pd.read_csv(bench.config.get_path("test_dataset"), dtype=dtypes)
    elif set == "all":
        df = pd.read_csv(bench.config.get_path("dataset"), dtype=dtypes)

    if instance is not None:
        x = df[df[bench.config.instance_names] == instance][bench.config.hp_names]
        truth = df[df[bench.config.instance_names] == instance][bench.config.y_names]
    else:
        x = df[bench.config.hp_names]
        truth = df[bench.config.y_names]

    n = x.shape[0]
    chunks = chunk(n, chunk_size)
    response = np.zeros(shape=(n, len(bench.config.y_names)))  # create response array in memory and fill rows by chunk results
    for i in range(0, len(chunks)):
        indices = chunks[i]
        points = x[indices[0]:indices[1]].apply(lambda point: point[~point.isna()].to_dict(), axis=1, result_type=None).tolist()
        result = pd.DataFrame(bench.objective_function(points)).values
        response[indices[0]:indices[1]] = result

    truth_tensor = torch.tensor(truth.values)
    response_tensor = torch.tensor(response)

    metrics_dict = {}
    metrics = {"mae":mae, "r2":r2, "spearman":spearman}
    for metric_name,metric in zip(metrics.keys(), metrics.values()):
        values = metric(truth_tensor, response_tensor)
        metrics_dict.update({metric_name:dict(zip([y for y in bench.config.y_names], [*values]))})

    return metrics_dict

def generate_all_test_set_metrics(key, model=None, save_to_csv=False):
    bench = benchmark_set.BenchmarkSet(key)
    test_set_metrics = pd.DataFrame.from_dict(get_set_metrics(key, model = model))
    test_set_metrics["instance"] = "all"
    for instance in bench.instances:
        tmp = pd.DataFrame.from_dict(get_set_metrics(key, model = model, instance = instance))
        tmp["instance"] = instance
        test_set_metrics = pd.concat([test_set_metrics, tmp])
    if save_to_csv:
        test_set_metrics.to_csv(bench.config.config_path + "/test_set_metrics_" + model + ".csv", index_label = "target")

