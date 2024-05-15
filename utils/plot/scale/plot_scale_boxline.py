
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

fig, axs = plt.subplots(1, 2, figsize=(12, 8))

'''

bits 数量不变 gates 增加,观察depth变化趋势，箱线图
'''

def read_gates_scale_data():
    rl = []
    qiskit=[]
    # get data
    sheets, dfs = ExcelUtil.read_by_sheet('d:/15bits_custom.xlsx')
    df = dfs[sheets[0]]


    for start in range(0, len(df['rl']), 12):  # 从0开始，每次跳过12个（10个取出，2个跳过）
        end = start + 10  # 定义结束位置
        temp_data = df['rl'][start:end].tolist()  #

        if len(temp_data) == 10:
            rl.append(temp_data)  # 将这10个数据的列表添加到temp_arrays中

    for start in range(0, len(df['qiskit']), 12):
        end = start + 10  # 定义结束位置
        temp_data = df['rl'][start:end].tolist()

        if len(temp_data) == 10:
            qiskit.append(temp_data)

    return rl,qiskit



def plot_gates_scale():
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成一些随机数据
    np.random.seed(10)
    data = [np.random.normal(0, std, 100) for std in range(1, 6)]

    # 创建箱线图
    fig, ax = plt.subplots()
    bplot = ax.boxplot(data, patch_artist=True, notch=False,
                       )  # 使用patch_artist来填充颜色

    # 设置每个箱体的颜色
    colors = ['#ed9a9c', '#3d7fcc','#bfd566','#ffce75','#dbc5a9']
    edge_colors=['#c43737','#1e3b6d','#889a46','#ef9c09','#422511']
    for patch, color,edge_color in zip(bplot['boxes'], colors,edge_colors):
        patch.set_facecolor(color)
        # 加深边框颜色，这里我们将边框颜色设置为填充颜色的深色版本
        patch.set_edgecolor(edge_color)  # 或者使用任何你想要的颜色

    # 设置离群点的颜色
    for flier, color in zip(bplot['fliers'], edge_colors):
        flier.set_markerfacecolor(color)
        flier.set_marker('o')
        flier.set_markersize(5)

    # 调整中位线的颜色和线型
    i=0
    for median in bplot['medians']:
        median.set(color=edge_colors[i], linewidth=2)
        i +=1

    # 添加一些标签（仅作为示例）
    plt.xticks([1, 2, 3,4,5], ['Group 1', 'Group 2', 'Group 3','Group 4','Group'])
    plt.title('Customized Boxplot')

    plt.show()

def test():

    # 生成一些示例数据
    # np.random.seed(10)
    # data1 = [np.random.normal(0, std, 100) for std in range(1, 4)]
    # data2 = [np.random.normal(0.5, std, 100) for std in range(1, 4)]

    data1,data2 = read_gates_scale_data()
    # 准备绘制箱线图的数据
    data = [val for pair in zip(data1, data2) for val in pair]

    # 创建箱线图
    fig, ax = plt.subplots()

    # 设置每个箱体的位置，确保每对箱体靠近且与下一对有一定间隔
    positions = [1, 2, 4, 5, 7, 8,10,11,13,14]

    # 设置每个箱体的宽度，以避免重叠
    width = 0.35

    # 绘制箱线图
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=True)

    # 设置颜色以区分不同的组
    colors = ['lightblue', 'lightgreen'] * 5
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 添加一些自定义的图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Group 1'),
                       Patch(facecolor='lightgreen', label='Group 2')]
    ax.legend(handles=legend_elements)

    ax.set_title('Comparative Boxplot')
    ax.set_xlabel('X-axis Label')
    ax.set_ylabel('Y-axis Label')

    # 设置x轴的刻度标签
    plt.xticks([1.5, 4.5, 7.5], ['Category 1', 'Category 2', 'Category 3'])

    plt.show()


if __name__ == '__main__':
    #read_gates_scale_data()

    #plot_gates_scale()

    test()