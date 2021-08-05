from yahpo_gym.configuration import cfg
from ConfigSpace.read_and_write import json
import numpy as np
import torch
import onnxruntime as rt

class BenchmarkInstance():

    def __init__(self, config_id = None, active_session = True):
        """
        Combination of an objective function and a configuration space
        with additional helpers that allow querying properties and further customization.
        """
        self.config = cfg(config_id)
        self.config_space = self.get_config_space()
        self.active_session = active_session
        if self.active_session:
            self.set_session()

    def objective_function(self, configuration):
        if not self.active_session:
            self.set_session()
        x_cat = torch.ones(1, 1, dtype = torch.int)
        x_cont = torch.randn(1, len(self.config.cont_names))
        # input & output names and dims
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        results = self.sess.run([output_name], (x_cat, {"x_cont":x_cont}))[0]
        return results
    
    def set_constant(self, param, value=None):
        NotImplementedError()

    def check_update_xs(self, xs):
        NotImplementedError()

    def get_config_space(self):
        with open(self.config.get_path("config_space"), 'r') as f:
            json_string = f.read()
            cs = json.read(json_string)
        return cs
    
    def __repr__(self):
        return f"BenchmarkInstance ({self.config.config_id})"
    
    def set_session(self):
        model_path = self.config.get_path("model")
        print(model_path)
        self.sess = rt.InferenceSession(model_path)


if __name__ == '__main__':
    import yahpo_gym.benchmarks.lcbench
    print(BenchmarkInstance("lcbench").config_space)

