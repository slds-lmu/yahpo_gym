import pandas as pd
import abc
from yahpo_gym.local_config import local_config

_yahpo_default_dict = {
    'basedir': local_config.data_path,
    'config_id': '',
    'model': 'new_model.onnx',
    'dataset': 'data.csv',
    'config_space': 'config_space.json',
    'encoding': 'encoding.json',
    'y_names' : [],
    'cont_names': [],
    'cat_names': [],
    'fidelity_params': [],
    'runtime_name': '',
    'model_old': 'model.onnx'
}

class Configuration():
    def __init__(self, config_dict):
        """
        Interface for benchmark scenario meta information. 
        Abstract base class used to instantiate configurations that contain all
        relevant meta-information about a specific benchmark scenario.

        Parameters
        ----------
        config_dict: dict
            A dictionary of settings required for a given configuration.
        """
        config = _yahpo_default_dict.copy()
        config.update(config_dict)
        self.config = config
        
        # Set attributes
        self.config_id = self.config['config_id']
        self.y_names = self.config['y_names']
        self.cat_names = self.config['cat_names']
        self.cont_names = self.config['cont_names']
        self.fidelity_params = self.config['fidelity_params']
        self.instance_names = self.config['instance_names']
        self.runtime_name = self.config['runtime_name']
        
    def get_path(self, key):
        return f'{self.config_path}/{self.config[key]}'

    @property
    def config_path(self):
        return f"{self.config['basedir']}/{self.config['config_id']}"

    @property
    def data(self):
        return pd.read_csv(self.get_path("dataset"))
     
    def __repr__(self): 
        return f"Configuration: ({self.config['config_id']})"

    def __str__(self):
        return self.config.__str__()


class ConfigDict():
    def __init__(self):
        """
        Dictionary of available benchmark scenarios (configurations). 
        This provides a thin wrapper allowing for easy updating and retrieving of 
        configurations pertaining to a specific benchmark scenario.
        """
        self.configs = {}

    def update(self, config_dict):
        """
        Add new or update existing benchmark scenario configuration.

        Parameters
        ----------
        config_dict: dict
            A dictionary of settings required for a given configuration.
        """
        self.configs.update(config_dict)
    
    def get_item(self, key):
        """
        Instantiate a given Configuration.

        Parameters
        ----------
        key: str
            The key of the configuration to retrieve
        """
        return Configuration(self.configs[key])

    def __repr__(self):
        return self.configs.__repr__()
    
    def __str__(self):
        out = "{:<15} {:<10} {:<10} {:<10} {:<10}".format("Key", "Instances", "Cat. HP", "Cont. HP", "Targets")
        if len(self.configs) == 0:
            out += "\n< No configs loaded >"
        for k in self.configs.keys():
            v = self.get_item(k)
            out += "\n{:<15} {:<15} {:<10} {:<10} {:<10}".format(k, v.instance_names, len(v.cat_names), len(v.cont_names), len(v.y_names))
        return out


def cfg(key = None):
    """
        Shorthand acces to 'ConfigDict'.
        
        Parameters
        ----------
        key: str
            The key of the configuration to retrieve.
            If none, prints available keys.
    """
    if key is not None:
        return config_dict.get_item(key)
    else:
        return config_dict

config_dict = ConfigDict()
