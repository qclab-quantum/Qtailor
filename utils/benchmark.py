import traceback

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from utils.circuit_util import CircutUtil as cu
from utils.file.csv_util import CSVUtil
from utils.graph_util import GraphUtil as gu
from utils.points_util import PointsUtil as pu

points = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
          (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
          (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
          (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
          (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
          (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
          (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
          (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
          (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
          (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]
class Benchmark():

    def __init__(self,qasm ):
        self.qasm = qasm


    @staticmethod
    def depth_benchmark(file_path,matrix:np.ndarray,qasm:str,is_draw=False):
        # b_1[
        # [rl * n]
        # [rl_qiskit * n]
        # ]
        b_1 = gu.test_adj_matrix(matrix, qasm)
        rl = b_1[-1][0]
        mix = b_1[-1][1]
        #get Qiskit result
        b_2 = Benchmark.get_qiskit_depth(qasm)
        qiskit = b_2[-1]
        print('rl = %r,qiskit= %r, mix = %r '%(rl,qiskit,mix))

        #write to csv file
        data = []
        for i in range(len(b_1)):
            data.append(['','',b_1[i][0],b_2[i],b_1[i][1]])
        CSVUtil.append_data(file_path, data)
        if is_draw:
            gu.draw_adj_matrix(matrix,is_draw_nt=True)
            pu.plot_points(points)
        return rl,qiskit,mix

    @staticmethod
    def get_qiskit_depth(qasm:str):
        result = []
        repeat = 10
        adj_list = pu.coordinate2adjacent(points)
        c = cu.get_from_qasm(qasm)
        # c.draw('mpl').show()
        simulator = AerSimulator()
        avr = 0
        for i in range(repeat):
            try:
                ct = transpile(circuits=c, coupling_map=adj_list, optimization_level=3,backend=simulator)
                d = ct.decompose().depth()
                avr += d
                result.append(d)
            except Exception as e:
                traceback.print_exc()
        result.append(avr/repeat)
        return  result

#用于测试模型运行后的结果
def test_result(qasm,matrix = None, array = None):
    #test rl and mix
    if matrix is None:
        matrix =gu.restore_from_1d_array(array)
    res = gu.test_adj_matrix(matrix,qasm)
    print(res)
    mean = np.mean(res, axis=0)
    rl = mean[0]
    mix = mean[1]
    qiskit = np.mean(Benchmark.get_qiskit_depth(qasm))
    print('rl = %r,qiskit = %r , mix = %r:'%(rl,qiskit,mix))

if __name__ == '__main__':
    array  = \
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # matrix = [[0, 1, 0, 0, 1, 0, 1, 1],
    #    [1, 0, 1, 0, 0, 1, 0, 1],
    #    [0, 1, 0, 1, 1, 0, 1, 0],
    #    [0, 0, 1, 0, 1, 1, 0, 1],
    #    [1, 0, 1, 1, 0, 1, 0, 0],
    #    [0, 1, 0, 1, 1, 0, 1, 0],
    #    [1, 0, 1, 0, 0, 1, 0, 1],
    #    [1, 1, 0, 1, 0, 0, 1, 0]]
    qasm = 'su2/su2random_indep_qiskit_14.qasm'
    test_result(matrix = None,array=array,qasm=qasm)

    #print(Benchmark.get_qiskit_depth('random/random_indep_qiskit_14.qasm'))