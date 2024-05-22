import os
import re
import tempfile
from datetime import datetime

from ray.tune import ResultGrid
from ray.tune.logger import UnifiedLogger

from utils.file.csv_util import CSVUtil
from utils.file.file_util import FileUtil


def analyze_result(results:ResultGrid):
    ## result analysis
    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.path  # Get best trial's result directory
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    best_metrics = best_result.metrics  # Get best trial's last results
    best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

    # Get a dataframe with the last results for each trial
    df_results = results.get_dataframe()

    # Get a dataframe of results for a specific score or mode
    df = results.get_dataframe(filter_metric="score", filter_mode="max")
    ##
    # print(best_config)
    # print('====================')
    # print(best_metrics)

def set_logger():
    import warnings
    warnings.simplefilter('ignore')
    import logging

    # First, get the handle for the logger you want to modify
    ray_data_logger = logging.getLogger("ray.data")
    ray_tune_logger = logging.getLogger("ray.tune")
    ray_rllib_logger = logging.getLogger("ray.rllib")
    ray_train_logger = logging.getLogger("ray.train")
    ray_serve_logger = logging.getLogger("ray.serve")

    # Modify the ray.data logging level
    ray_data_logger.setLevel(logging.ERROR)
    ray_tune_logger.setLevel(logging.ERROR)
    ray_rllib_logger.setLevel(logging.ERROR)
    ray_train_logger.setLevel(logging.ERROR)
    ray_serve_logger.setLevel(logging.ERROR)

def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

from typing import Dict, Union
from pathlib import Path
from ray import cloudpickle
def load_checkpoint_from_path(checkpoint_to_load: Union[str, Path]) -> Dict:

    """Utility function to load a checkpoint Dict from a path."""
    checkpoint_path = Path(checkpoint_to_load).expanduser()
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
    with checkpoint_path.open("rb") as f:
        return cloudpickle.load(f)

def new_csv(time_str):
    sep = '/'
    csv_path = FileUtil.get_root_dir() + sep + 'benchmark' + sep + 'a-result' + sep + time_str + '.csv'
    CSVUtil.write_data(csv_path,[['datetime', 'qasm', 'rl', 'qiskit','mix', 'result', 'iter','checkpoint','remark', ]])
    return  csv_path

def get_qasm():
    qasm = [
       'qft\\qft_indep_qiskit_10.qasm',
    ]
    return qasm

def parse_tensorboard(content):
    # 使用正则表达式搜索匹配的字符串
    pattern = r'tensorboard --logdir\s(.+)'
    result = re.findall(pattern, content)

    if result:
        matched_string = result[0][:-1]
        tensorboard = matched_string[matched_string.find("PPO"):]
        return  tensorboard
    else:
        print("未找到匹配的 tensorboard")
        return  ''