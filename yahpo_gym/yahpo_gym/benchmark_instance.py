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
        self.constants = {}
        if self.active_session:
            self.set_session()

    def objective_function(self, configuration):
        if not self.active_session:
            self.set_session()
        x_cont, x_cat = self.config_to_xs(configuration)
        # input & output names and dims
        input_names = [x.name for x in self.sess.get_inputs()]
        output_name = self.sess.get_outputs()[0].name
        results = self.sess.run([output_name], {input_names[0]: x_cat, input_names[1]: x_cont})[0]
        return results
    
    def config_to_xs(self, configuration):
        # Update with constants
        if len(self.constants):
            [configuration.update({k : v}) for k,v in self.constants.items()]
        x_cat = np.array([configuration[x] for x in self.config.cat_names]).reshape(1, -1).astype(np.int32)
        x_cont = np.array([configuration[x] for x in self.config.cont_names]).reshape(1, -1).astype(np.float32)
        # x_cat = np.ones((1,1), dtype = np.int32)
        # x_cont = np.random.randn(1,8).astype(np.float32)
        return x_cont, x_cat
    
    def set_constant(self, param, value=None):
        self.constants[param] = value

    def check_update_xs(self, xs):
        NotImplementedError()

    def set_task(value):
        self.config_space
        self.constants.update({self.config.task_name : value})

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

