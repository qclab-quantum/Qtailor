import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Get the directory of the current script
current_dir = Path(__file__).resolve().parent

# Get the parent directory
parent_dir = current_dir.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

# Now you can import the module from the parent directory
from utils.file.csv_util import CSVUtil

def plot_box_plot():
    #todo read from csv
    g1 = [249, 281, 252, 270, 273, 260, 251, 284, 294, 278]
    g2 = [330, 366, 297, 378, 322, 330, 325, 347, 331, 352]
    g3 = [313, 324, 348, 307, 335, 318, 322, 322, 311, 290]

    # Combine the data sets into a list
    data = [g1, g2, g3]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a box plot for multiple groups
    ax.boxplot(data)

    # Add x-axis labels
    groups = ['RL', 'Qiskit', 'RL_Qiskit']
    ax.set_xticklabels(groups)

    # Add labels and title
   # ax.set_xlabel("Groups")
    ax.set_ylabel("Depth")
    ax.set_title("qft_indep_qiskit_15")

    # Show the plot
    plt.show()

def plot_bar():

    x = np.array([1, 2, 3])
    y = np.array([1113, 1680.75,1510.75])

    plt.bar(x, y)
    plt.xlabel('')
    plt.ylabel('Depth')
    plt.title('amplitude_estimation_40bits')
    plt.xticks(x, ['RL', 'Qiskit','RL_Qiskit'])

    bars = plt.bar(x, y)

    # 定义颜色和条纹
    colors = [r'#f76f69', '#1597a5', '#ffc24b', 'yellow', 'purple']
    hatches = ['//', '\\', 'x', '\\\\', '.']

    for i in range(len(x)):
        plt.bar(x[i], y[i], color=colors[i], hatch=hatches[i],edgecolor='white')
    # 根据每个柱状图的索引设置颜色和条纹
    # for i in range(len(bars)):
    #     bars[i].set_color(colors[i])
    #     bars[i].set_hatch(hatches[i])

    #plt.yticks(np.arange(0, 12, 2))
    plt.grid(True)
    plt.show()

def plot_line():
    import seaborn as sns
    import matplotlib.pyplot as plt
    # data = {'X': [1, 2, 3, 4, 5],
    #         'Z':[[1, 1, 1], [2, 8, 10], [7, 6,2], [9, 2, 1, 1], [12, 9, 7]],
    #         'Y': [[10, 12, 11], [9, 8, 10], [7, 6, 5], [9, 10, 11, 12], [8, 9, 7]]}
    def read_data():
        # data1 = CSVUtil.read_data('E:/benchmark/qnn3-5.csv')[1:]
        # data1=[row[:5] for row in data1]
        # data2 = CSVUtil.read_data('E:/benchmark/qnn6-10.csv')[1:]
        # data3= CSVUtil.read_data('E:/benchmark/qnn11-15_400iter.csv')[1:]
        # data4 = CSVUtil.read_data('E:/benchmark/qnn16-20_400iter.csv')[1:]
        data1 = CSVUtil.read_data('D:/workspace/data/benchmark/portf/portfolio_vqe_5-10.csv')[1:]
        data1 = [row[:5] for row in data1]
        data2 = CSVUtil.read_data('D:/workspace/data/benchmark/portf/portfolio_vqe_11-18_300iter.csv')[1:]

        # data = np.concatenate((data1, data2), axis=0)
        data = []
        for row in data1:
            data.append(row)
        for row in data2:
            data.append(row)
        # for row in data3:
        #     data.append(row)
        # for row in data4:
        #     data.append(row)
        import copy
        d1 = []
        d2 = []
        d3 = []
        row = 1
        temp1 = []
        temp2 = []
        temp3 = []

        while row < len(data):

            if len(str(data[row][0])) > 1:
                # print(str(data[row][0]))
                d1.append(copy.deepcopy(temp1[:-1]))
                d2.append(copy.deepcopy(temp2[:-1]))
                d3.append(copy.deepcopy(temp3[:-1]))
                temp1 = []
                temp2 = []
                temp3 = []
            else:
                print(data[row])
                temp1.append(data[row][2])
                temp2.append(data[row][3])
                temp3.append(data[row][4])

            row = row + 1
        # print(d1)
        # print(d2)
        # print(d3)
        for row in range(len(d1)):
            for col in range(len(d1[row])):
                d1[row][col] = float(d1[row][col]) / float(d3[row][col])
                d2[row][col] = float(d2[row][col]) / float(d3[row][col])
                d3[row][col] = 1.
        data = {

            'x': range(3, len(d1) + 3),
            'rl': d1,
            'mix': d2,
            'qiskit': d3

        }

        return data

    data = read_data()
    new_data = {'x': [], 'rl': [], 'qiskit': [], 'mix': []}
    for i in range(len(data['x'])):
        x_val = data['x'][i]
        y_vals = data['rl'][i]
        z_vals = data['qiskit'][i]
        mix_vals = data['mix'][i]
        for y_val in y_vals:
            new_data['x'].append(x_val)
            new_data['rl'].append(y_val)
        for z_val in z_vals:
            new_data['qiskit'].append(z_val)
        for m_val in mix_vals:
            new_data['mix'].append(m_val)
    print(new_data)
    sns.lineplot(data=new_data, x='x', y='rl', errorbar=('ci', 68), err_style='band', err_kws={'alpha': 0.2},
                 label='rl')
    sns.lineplot(data=new_data, x='x', y='qiskit', errorbar=('ci', 68), err_style='band', err_kws={'alpha': 0.2},
                 label='qiskit')
    sns.lineplot(data=new_data, x='x', y='mix', errorbar=('ci', 68), err_style='band', err_kws={'alpha': 0.2},
                 label='mix')
    plt.xticks(range(3, len(data['x']) + 3))
    plt.title('portfolio_vqe circuit RL vs Qiskit ')
    plt.xlabel('bits number')
    plt.ylabel('circuit depth')
    plt.legend()

    plt.show()
if __name__ == '__main__':
    #plot_box_plot()
    #plot_bar()
    plot_line()