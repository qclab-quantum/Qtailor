# 二维坐标->邻接表（qiskit 可识别的形式）
# 坐标形式：  points=[(1,0),(1,1),(1,2),(1,3),(1,4)]
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.ticker as ticker
from qiskit import transpile
from qiskit.visualization import plot_circuit_layout

from utils.circuit_util import CircutUtil
from qiskit_aer import AerSimulator

class PointsUtil:
    @staticmethod
    def coordinate2adjacent(points):
        import math
        point_dict = {i: points[i] for i in range(len(points))}
        adjacency_dict = {}

        for i, point in point_dict.items():
            adjacent_points = []
            for j, other_point in point_dict.items():
                if i != j:
                    if math.sqrt((point[0] - other_point[0]) ** 2 + (point[1] - other_point[1]) ** 2) == 1:
                        adjacent_points.append(j)
            adjacency_dict[i] = adjacent_points

        # transform adjacency_dict to qiskit format
        res = []
        for k in adjacency_dict:
            v = adjacency_dict.get(k)
            for node in v:
                res.append([k, node])
        # return adjacency_dict
        return res

    # 邻接表-> 邻接矩阵
    @staticmethod
    def adjacency2matrix(adj_list):
        max_index = max(max(pair) for pair in adj_list)
        matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

        for pair in adj_list:
            matrix[pair[0]][pair[1]] = 1
            matrix[pair[1]][pair[0]] = 1

        return matrix



    # draw couping map from coordinate
    @staticmethod
    def plot_points(points):
        point_dict = {i: points[i] for i in range(len(points))}

        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
        # 设置x轴和y轴的刻度为整数

        for i, point in enumerate(sorted_points):
            plt.scatter(point[0], point[1], color='blue')
            plt.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(0, 10), ha='center')

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if dist <= 1:
                    plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], color='red')

        # 将x轴和y轴的刻度设置为整数
        plt.xticks(range(int(0), int(10) + 1))
        plt.yticks(range(int(0), int(10) + 1))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('coupling_map ')
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    pu = PointsUtil()
    points = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),
              (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),
              (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),
              (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
              (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),
              (5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),
              (6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),
              (7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),
              (8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,8),(8,9),
              (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9)]
    # for i in range(10):
    #     for j in range(10):
    #         points.append((i, j))
    adj_list = pu.coordinate2adjacent(points)
    c = CircutUtil.get_from_qasm('qftentangled_indep_qiskit_10.qasm')
    #c.draw('mpl').show()
    simulator = AerSimulator()
    for i in range(5):
        avr = 0
        for i in range(20):
            ct = transpile(circuits=c, coupling_map=adj_list, optimization_level=3,
                           backend=simulator)
            avr += ct.decompose().depth()

        print(avr/20)
    #print(ct.layout.initial_layout)
    #ct.draw('latex').show()
    #print(pu.adjacency2matrix(pu.coordinate2adjacent(points)))
    #plot_circuit_layout(ct, simulator)