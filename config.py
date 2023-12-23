import os

from munch import Munch
import torch
import yaml

from utils.file_util import FileUtil

def get_args():
    config = None
    rootdir = FileUtil.get_root_dir()
    with open(rootdir+os.path.sep+'config.yml', 'r') as file:
        config = yaml.safe_load(file)
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        config = Munch(config)
    return config

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigSingleton(metaclass=Singleton):
    def __init__(self):
        self.config = None
        self.load_config()
        self.flag = 0

    def load_config(self):
        rootdir = FileUtil.get_root_dir()
        with open(rootdir+os.path.sep+'config.yml', 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config = Munch(config)

    def get_config(self):
        self.flag += 1
        return self.config

if __name__ == '__main__':
    args = ConfigSingleton().get_config()
    print(ConfigSingleton().get_config().qasm)
    ConfigSingleton().get_config().qasm = 'test'
    print(ConfigSingleton().get_config().qasm)
