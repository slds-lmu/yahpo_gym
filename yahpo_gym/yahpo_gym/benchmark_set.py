from yahpo_gym.configuration import cfg
from yahpo_gym.benchmarks import *
import json
from ConfigSpace.read_and_write import json as CS_json
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from pathlib import Path
import numpy as np
import onnxruntime as rt
import time

class BenchmarkSet():

    def __init__(self, config_id = None, active_session = False, quant = 0.01):
        """
        Combination of an objective function and a configuration space
        with additional helpers that allow querying properties and further customization.
        """
        self.config = cfg(config_id)
        self.encoding = self._get_encoding()
        self.config_space = self._get_config_space()
        self.active_session = active_session
        self.quant = quant
        
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
        results = self.sess.run([output_name], {input_names[0]: x_cat, input_names[1]: x_cont})[0][0]
        return {k:v for k,v in zip(self.config.y_names, results)}

    def objective_function_timed(self, configuration):
        """
        Evaluate the surrogate for a given configuration.
        Waits for 'quant * predicted runtime' before returining results.
        """
        start_time = time.time()
        results = self.objective_function(configuration)
        rt = results[self.config.runtime_name]
        offset = time.time() - start_time
        sleepit = (rt - offset) * self.quant
        time.sleep(sleepit)
        return results

    def set_constant(self, param, value=None):
        hpar = self.config_space.get_hyperparameter(self.config.instance_names)
        # value in hpar.choices
        self.constants[param] = value
    
    def set_instance(self, value):
        self.set_constant(self.config.instance_names, value)

    def get_opt_space(self, instance, drop_fidelity_params = True):
        """
        Get the search space to be optimized.
        Sets 'instance# as a constant instance and removes all fidelity parameters if 'drop_fidelity_params = True'.
        """
        # FIXME: assert instance is a valid choice
        hps = self.config_space.get_hyperparameters()
        instance_names_idx = self.config_space.get_hyperparameter_names().index(self.config.instance_names)
        hps[instance_names_idx] = CSH.Constant(self.config.instance_names, instance)
        if drop_fidelity_params:
            fidelity_params_idx = [self.config_space.get_hyperparameter_names().index(fidelity_param) for fidelity_param in self.config.fidelity_params]
            for idx in fidelity_params_idx:
                del hps[idx]
        cnds = self.config_space.get_conditions()
        fbds = self.config_space.get_forbiddens()
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        cs.add_conditions(cnds)
        cs.add_forbidden_clauses(fbds)
        return cs

    def _config_to_xs(self, configuration):
        # Update with constants (constants overwrite configuration values)
        if len(self.constants):
            [configuration.update({k : v}) for k,v in self.constants.items()]

        # FIXME: check NA handling below
        all = self.config_space.get_hyperparameter_names()
        missing = list(set(all).difference(set(configuration.keys())))
        for hp in missing:
            value = '#na#' if hp in self.config.cat_names else 0  # '#na#' for cats, see _integer_encode below
            configuration.update({hp:value})

        # FIXME: Check the configuration with the ConfigSpace
        x_cat = np.array([self._integer_encode(configuration[x], x) for x in self.config.cat_names]).reshape(1, -1).astype(np.int32)
        x_cont = np.array([configuration[x] for x in self.config.cont_names]).reshape(1, -1).astype(np.float32)
        return x_cont, x_cat

    def _integer_encode(self, value, name):
        """
        Integer encode categorical variables.
        """
        # see model.py dl_from_config on how the encoding was generated and stored
        return self.encoding.get(name).get(value)

    def _get_encoding(self):
        with open(self.config.get_path("encoding"), 'r') as f:
            encoding = json.load(f)
        return encoding

    def _get_config_space(self):
        with open(self.config.get_path("config_space"), 'r') as f:
            json_string = f.read()
            cs = CS_json.read(json_string)
        return cs

    def _eval_random(self):
        cfg = self.config_space.sample_configuration().get_dictionary()
        print(cfg)
        return self.objective_function_timed(cfg)
    
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
    x.set_constant("epoch", 50)
