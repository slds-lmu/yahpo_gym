from pathlib import Path
import os
import yaml

class LocalConfiguration():

    def __init__(self, settings_path="~/.config/yahpo_gym"):
            self.settings_path = Path(settings_path).expanduser().absolute()

    def set_config_path(self, config_path):
        config = self.config()
        config.update({'data_path': str(config_path)})
        
        with self.settings_path.open('w', encoding='utf-8') as fh:
            yaml.dump(config, fh)
    
    def init_config(self, config_path=""):
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        config = {'data_path': str(config_path)}
        with self.settings_path.open('w', encoding='utf-8') as fh:
            yaml.dump(config, fh)

    def load_config(self):
        try:
            with self.settings_path.open('r') as fh:
                config = yaml.load(fh, Loader=yaml.FullLoader)
        except yaml.parser.ParserError:
            raise yaml.parser.ParserError("Could not load config!")
        return config

    @property
    def config(self):
        return self.load_config()
    @property
    def data_path(self):
        return Path(self.config.get('data_path')).expanduser().absolute()

local_config = LocalConfiguration()
__all__ = [local_config, LocalConfiguration]