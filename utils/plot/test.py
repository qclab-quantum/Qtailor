import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制折线图
line1, = plt.plot(x, y1, label='Sine', color='blue')
line2, = plt.plot(x, y2, label='Cosine', color='orange')

# 创建自定义的标记
circle_marker_line1 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
circle_marker_line2 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10)

# 添加图例，并使用自定义的标记
plt.legend([circle_marker_line1, circle_marker_line2], ['Sine', 'Cosine'])

# 显示图形
plt.show()
