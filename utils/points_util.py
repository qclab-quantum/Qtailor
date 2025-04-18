import matplotlib.pyplot as plt
from math import sqrt

class PointsUtil:

    @staticmethod
    def coordinate2adjacent(points):
        '''
          将若干点的二维坐标转换为邻接表（qiskit 可识别的邻接表形式）
          二维坐标形式:  points=[(1,0),(1,1),(1,2),(1,3),(1,4)]
        '''
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
        return res

    '''
    transform adjacency list to adjacency matrix
    '''
    @staticmethod
    def adjacency2matrix(adj_list):
        max_index = max(max(pair) for pair in adj_list)
        matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

        for pair in adj_list:
            matrix[pair[0]][pair[1]] = 1
            matrix[pair[1]][pair[0]] = 1

        return matrix

    '''
    draw couping map using  2-d points coordinate
    '''
    @staticmethod
    def plot_points(points):
        point_dict = {i: points[i] for i in range(len(points))}

        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
        # 设置x轴和y轴的刻度为整数
        layout = []
        for i, point in enumerate(sorted_points):
            color = 'black'
            text = str(i)
            if i in layout:
                color = 'red'
                text = ''
            plt.scatter(point[0], point[1], color='black',linewidths=0.1)
            plt.annotate(str(i), (point[0], point[1]), textcoords="offset points", fontsize = 7,xytext=(0, 10), ha='center')

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                if dist <= 1:
                    plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], color='#c6c6c6')

        # 将x轴和y轴的刻度设置为整数
        plt.xticks(range(int(0), int(10) + 1))
        plt.yticks(range(int(0), int(10) + 1))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('coupling_map ')
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    pass