import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
from utils.file.excel_util import ExcelUtil
mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] = 12
# 假设我们有9组数据，每组都有两个子组
# data = {
#     'Group1': ([12, 30, 1, 8, 22], [28, 6, 16, 5, 10]),
#     'Group2': ([15, 29, 2, 6, 25], [26, 7, 14, 8, 11]),
#     'Group3': ([10, 20, 10, 7, 27], [30, 5, 18, 3, 12]),
#     'Group4': ([14, 25, 3, 9, 21], [24, 9, 12, 6, 13]),
#     'Group5': ([13, 22, 5, 11, 19], [27, 8, 11, 7, 14]),
#     'Group6': ([11, 28, 4, 10, 23], [29, 4, 13, 4, 15]),
#     'Group7': ([9, 27, 6, 12, 18], [25, 10, 10, 9, 16]),
#     'Group8': ([8, 26, 7, 13, 17], [23, 11, 9, 10, 17]),
#     'Group9': ([7, 24, 8, 14, 16], [22, 12, 8, 11, 18])
# }

data={}
labels_2d = [

]
# get data
sheets, dfs = ExcelUtil.read_by_sheet('d:/temp.xlsx')
# pharse data
for sheet in sheets:
    df = dfs[sheet]
    circuits = df['circuit']
    # 从字符串中提取出该线路的比特数量 qnn/qnn_indep_qiskit_5.qasm-> 5
    labels = list(map(lambda x: ''.join(re.findall(r'\d', x)), circuits))
    print(labels)
    labels_2d.append(labels)
    rl = df['rl']
    qiskit = df['qiskit']
    mix = df['mix']

    data.update({sheet:(rl,qiskit,mix)})


# 设置每个柱子的宽度
bar_width = 0.25

# 设置柱子的位置
index = np.arange(len(labels))

# 创建3x3的子图布局
fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # figsize可以根据需要调整

plt.subplots_adjust(hspace=0.4,wspace = 0.2)
# 遍历数据和子图网格，绘制柱状图
for i, (group_name, (group1, group2,group3)) in enumerate(data.items()):
    # 计算子图的行和列索引
    row = i // 3
    col = i % 3

    # 获取当前子图的axes对象
    ax = axes[row, col]

    # 绘制第一组数据的柱状图
    ax.bar(index, group1,  bar_width,color = '#5370c4',label=f'{group_name} 1',hatch='-', edgecolor='black')

    # 绘制第二组数据的柱状图
    ax.bar(index + bar_width, group2, bar_width,color = '#f16569', label=f'{group_name} 2',hatch='.', edgecolor='black')
    ax.bar(index + bar_width*2, group3, bar_width,color = '#95c978', label=f'{group_name} 3',hatch='//', edgecolor='black')

    # 添加图例
    ax.legend(['Reinforce','Qiskit','Mix'], loc='upper left')

    # 设置横轴的标签
    ax.set_xticks(index + bar_width / 1)
    ax.set_xticklabels(labels_2d[i])
    if i ==0 or i==3:
        ax.set_ylabel('Depth')
    if i in range(6,8):
        ax.set_xlabel('bits')
    # 设置图表的标题
    ax.set_title(f'{group_name}')
    # 显示背景网格
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
# 调整子图之间的间距
#plt.tight_layout()

# 显示图表
plt.show()