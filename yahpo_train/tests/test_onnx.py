import onnxruntime as rt
from onnxruntime.datasets import get_example

if __name__ == '__main__':
    file = "/home/flo/lrz_synchshare/multifidelity_data/lcbench/model.onnx"
    rt.InferenceSession(file)