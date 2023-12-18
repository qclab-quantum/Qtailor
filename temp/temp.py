import os

print(os.environ.get("RLLIB_NUM_GPUS", "0"))