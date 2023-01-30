from pathlib import Path
from typing import Optional
import os
import yaml

class LocalConfiguration():
    def __init__(self, settings_path: str =None):
        """
        Interface for setting up a local configuration.
        This reads from and writes to a configuration file in the YAML format,
        allowing to store paths to the underlying data and models required for
        inference on the fitted surrogates.
        

        Parameters
        ----------
        settings_path: str
            Path to the local configuration file.
            The default is "~/.config/yahpo_gym".
        """
        if settings_path is None: 
            if 'YAHPO_LOCAL_CONFIG' in os.environ:
                settings_path = os.environ['YAHPO_LOCAL_CONFIG']
            else: 
                settings_path = "~/.config/yahpo_gym"
        self.settings_path = Path(settings_path).expanduser().absolute()
        self._config = None
    
    def init_config(self, data_path: str = ""):
        """
        Initialize a new local configuration.

        This writes a local configuration file to the specified 'settings_path'.
        The 
        It is currently used to globally store the following information
        'data_path': A path to the metadata required for inference.

        Parameters
        ----------
        settings_path: str
            Path to the directory where surrogate models and metadata are saved.
        """
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        config = {'data_path': str(data_path)}
        with self.settings_path.open('w', encoding='utf-8') as fh:
            yaml.dump(config, fh)
        return None

    def set_data_path(self, data_path: str):
        """
        Set path to directory where required models and metadata are stored.

        Parameters
        ----------
        data_path: str
            Path to the directory where surrogate models and metadata are saved.
        """
        config = self.config
        config.update({'data_path': str(data_path)})
        
        with self.settings_path.open('w', encoding='utf-8') as fh:
            yaml.dump(config, fh)

    def _load_config(self):
        config = {"data_path":""}
        try:
            with self.settings_path.open('r') as fh:
                config = yaml.load(fh, Loader=yaml.FullLoader)
        except yaml.parser.ParserError:
            raise yaml.parser.ParserError("Could not load config! (Invalid YAML?)")
        except:
            raise Exception("Could not load local_config! Please run LocalConfiguration.init_config() and restart.")
        return config

    @property
    def config(self):
        """
        The stored settings dictionary (cached).
        """
        if self._config is None:
            self._config = self._load_config()
        return(self._config)

    @property
    def data_path(self):
        """
        Path where metadata and surrogate models for inference are stored.
        """
        return Path(self.config.get('data_path')).expanduser().absolute()

local_config = LocalConfiguration()
__all__ = ['local_config', 'LocalConfiguration']
