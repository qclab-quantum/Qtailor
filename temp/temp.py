import numpy as np
import matplotlib.pyplot as plt

# 生成从正态分布中采样的数据点
data = np.random.normal(size=10000)

# 组成100x100像素的图像
image = np.reshape(data, (100, 100))

# 显示图像
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()