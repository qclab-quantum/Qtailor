import matplotlib.pyplot as plt
import numpy as np
from utils.file.csv_util import CSVUtil
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from utils.file.excel_util import ExcelUtil
from utils.file.file_util import FileUtil
import matplotlib.ticker as ticker


mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] = 12
line_width = 2.0
dpi = 500
np.dtype(float)
'''
bits 数量不变 gates 增加,观察depth变化趋势，折线图
'''
#read data
def read_gates_scale_data():
    rl = []
    qiskit=[]
    # get data
    sheets, dfs = ExcelUtil.read_by_sheet('d:/15bits_custom.xlsx')
    df = dfs[sheets[0]]


    for start in range(0, len(df['rl']), 12):
        end = start + 10  # 定义结束位置
        temp_data = df['rl'][start:end].tolist()  #

        if len(temp_data) == 10:
            rl.append(np.mean(temp_data))

    for start in range(0, len(df['qiskit']), 12):
        end = start + 10  # 定义结束位置
        temp_data = df['qiskit'][start:end].tolist()

        if len(temp_data) == 10:
            qiskit.append(np.mean(temp_data))

    return rl,qiskit

# 创建一个2x1的子图

def plot1():

    rl =[115.7, 234.5, 435.8, 972.7, 1988.1]
    qiskit=[163.7, 335.7, 665.9, 1495.8, 3168.7]
    x_label=['150','300','600','1200','2400']
    x=[0,1,2,3,4]
    # 创建折线图
    plt.plot( x,rl,color='#1565c0', linewidth=line_width, label='QTailor',marker='o')
    plt.plot( x,qiskit,color='#df6172', linewidth=line_width, label='Qiskit',marker = 's')

    plt.xticks(x, x_label)
    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)

    y_min, y_max =  plt.ylim()
    # #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(0, 4000, 500):
        plt.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )


    #plt.title('amplitude_estimation')
    # plt.set_xlabel('Gates')
    # plt.set_ylabel('Depth')
    plt.xlabel('Gates')
    plt.ylabel('Depth')

    plt.legend(loc='upper left',fontsize='medium')



if __name__ == '__main__':
    plot1()
    plt.tight_layout()
    plt.savefig(FileUtil.get_root_dir()+'/data/fig/scale_gates.png',dpi = dpi)
    # 显示图形
    plt.show()