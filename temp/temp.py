import numpy as np
from gymnasium.spaces import Box, flatten_space
from gymnasium.vector.utils import spaces

from utils.file_util import FileUtil

#content = FileUtil.read_all('d:/temp.txt')

import re

#text = "tensorboard --logdir C:/Users/Administrator/ray_results/PPO_2024-01-01_20-58-08"

# 使用正则表达式搜索匹配的字符串
# pattern = r'tensorboard --logdir\s(.+)'
# result = re.findall(pattern, content)
#
# if result:
#     matched_string = result[0][:-1]
#     ppo = matched_string[matched_string.find("PPO"):]
#     print("提取的字符串:", matched_string)
#     print("ppo:", ppo)
# else:
#     print("未找到匹配的字符串")

# import scipy.io as sio
# sio.savemat("d:/xxx.mat", {"graph_sparse":[[1,2,3],[4,5,6],[7,8,9]]})

print(flatten_space(spaces.Box(0, 1, (1, 9), dtype=np.float32, )).sample())