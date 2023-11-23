import matplotlib.pyplot as plt
from math import sqrt
'''
输入点的二维坐标，在二维坐标系中画出点的布局
'''

def plot_points(points):
    point_dict = {i: points[i] for i in range(len(points))}

    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))

    for i, point in enumerate(sorted_points):
        plt.scatter(point[0], point[1], color='blue')
        plt.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            if dist <= 1:
                plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], color='red')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Device Topology')
    plt.grid(True)
    plt.show()

# 测试示例
points = [(0,0),(0,1),(0,2),
          (1,0),(1,1),(1,2)
          ,(2,0),(2,1),(2,2)]
plot_points(points)