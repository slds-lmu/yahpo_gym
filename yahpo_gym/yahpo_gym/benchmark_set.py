from yahpo_gym.configuration import cfg
from ConfigSpace.read_and_write import json
import onnxruntime

class BenchmarkInstance():

    def __init__(self, config_id = None):
        """
        Combination of an objective function and a configuration space
        with additional helpers that allow querying properties and further customization.
        """
        self.config = cfg(config_id)
        self.config_space = self.get_config_space()

    def objective_function(self, configuration):
        NotImplementedError()
    
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

if __name__ == '__main__':
    import yahpo_gym.benchmarks.lcbench
    print(BenchmarkInstance("lcbench").config_space)

