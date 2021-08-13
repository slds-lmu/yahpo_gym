from yahpo_gym.configuration import cfg
from yahpo_gym.benchmarks import *
from ConfigSpace.read_and_write import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as rt

class BenchmarkSet():

    def __init__(self, config_id = None, active_session = False):
        """
        Combination of an objective function and a configuration space
        with additional helpers that allow querying properties and further customization.
        """
        self.config = cfg(config_id)
        self.config_space = self._get_config_space()
        self.active_session = active_session
        self.constants = {}
        if self.active_session:
            self.set_session()

    def objective_function(self, configuration):
        """
        Evaluate the surrogate for a given configuration.
        """
        if not self.active_session:
            self.set_session()
        x_cont, x_cat = self._config_to_xs(configuration)
        # input & output names and dims
        input_names = [x.name for x in self.sess.get_inputs()]
        output_name = self.sess.get_outputs()[0].name
        results = self.sess.run([output_name], {input_names[0]: x_cat, input_names[1]: x_cont})[0]
        return results
    
    def set_constant(self, param, value=None):
        hpar = self.config_space.get_hyperparameter(self.config.instance_names)
        # value in hpar.choices
        self.constants[param] = value
    
    def set_instance(self, value):
        self.set_constant(self.config.instance_names, value)

    def _config_to_xs(self, configuration):
        # Update with constants (constants overwrite configuration values)
        if len(self.constants):
            [configuration.update({k : v}) for k,v in self.constants.items()]  
        # FIXME: Here we should check and update the configuration with the ConfigSpace  
        x_cat = np.array([self._integer_encode(configuration[x], x) for x in self.config.cat_names]).reshape(1, -1).astype(np.int32)
        x_cont = np.array([configuration[x] for x in self.config.cont_names]).reshape(1, -1).astype(np.float32)
        return x_cont, x_cat

    def _integer_encode(self, value, name):
        """Integer encode categorical variables"""
        return 1

    def _get_config_space(self):
        with open(self.config.get_path("config_space"), 'r') as f:
            json_string = f.read()
            cs = json.read(json_string)
        return cs
    
    def __repr__(self):
        return f"BenchmarkInstance ({self.config.config_id})"
    
    def set_session(self):
        model_path = self.config.get_path("model")
        if not Path(model_path).is_file():
            raise Exception(f("ONNX file {model_path} not found!"))
        self.sess = rt.InferenceSession(model_path)

    @property
    def instances(self):
        if self.config.instance_names is None:
            return []
        return [*self.config_space.get_hyperparameter(self.config.instance_names).choices]


if __name__ == '__main__':
    import yahpo_gym.benchmarks.lcbench
    import yahpo_gym.benchmarks.nasbench_301
    x = BenchmarkSet("lcbench")
    x.set_instance("3945")
    value = {'epoch':1, 'batch_size':1, 'learning_rate':.1, 'momentum':.1, 'weight_decay':.1, 'num_layers':1, 'max_units':1, 'max_dropout':.1}
    print(x.objective_function(value))
