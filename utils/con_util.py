import math
import time

import torch
import torch.nn as nn





class ConvolutionUtil():
    @staticmethod
    def conv(matrix):
        # 输入图像数据
        #input_data = torch.randn(1, 1, 32, 32)  # (batch_size, channels, height, width)
        tm = torch.tensor(matrix)
        tm = tm.unsqueeze(0)
        tm = tm.float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 创建Conv2d对象
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        conv.to(device)

        # 创建ReLU对象
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()

        # 创建MaxPool2d对象
        pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # CONV操作
        conv_output = conv(tm)

        # RELU操作
        relu_output = sigmoid(conv_output)

        # POOL操作
        pool_output = pool(relu_output)

        # 查看输出结果的形状
        # print(conv_output.shape)
        # print(relu_output.shape)
        # print(pool_output.shape)
        flatten_output = pool_output.view(pool_output.size(0), -1)
        #print(flatten_output)

        return flatten_output.detach().numpy()[0]

if __name__ == '__main__':
    matrix = [[0, 1, 1, 0, 1, 0, 1, 1],
              [1, 0, 1, 0, 0, 1, 0, 1],
              [1, 1, 0, 1, 1, 0, 1, 0],
              [0, 0, 1, 0, 1, 1, 0, 1],
              [1, 0, 1, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 1, 0, 1, 0],
              [1, 0, 1, 0, 0, 1, 0, 1],
              [1, 1, 0, 1, 1, 0, 0, 1]]
    stime = time.time()
    for i in range(10000):
        ConvolutionUtil.conv(matrix)
    print('time used ', time.time() - stime)