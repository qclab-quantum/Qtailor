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
