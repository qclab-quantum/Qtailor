#compare reward replay(env7) and traditional RL(env6)
import matplotlib.pyplot as plt
import numpy as np
from utils.file.csv_util import CSVUtil
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from utils.file.file_util import FileUtil
import matplotlib.ticker as ticker


mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] = 18
line_width = 2.2
v6=[]
v7=[]
dpi = 500
# Custom formatter function
def time_formatter(x,pos):
    if x <3600:
        # Format as hours and minutes if more than 60 minutes
        return f'{(x/60).__round__(1)}min'
    else:
        # Format as minutes otherwise
        return f'{(x/3600).__round__(1)}hr'
def get_data(folder,x_index):
    dfv6=CSVUtil.to_dataframe(relative_path=f'data/train_1/ae40/{folder}/env6.csv')
    dfv7=CSVUtil.to_dataframe(relative_path=f'data/train_1/ae40/{folder}/env7.csv')

    xv6=dfv6[x_index].values
    yv6=dfv6['Value'].values

    xv7=dfv7[x_index].values
    yv7=dfv7['Value'].values

    return xv6,yv6,xv7,yv7

# 创建一个3x1的子图
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

#mean kl loss
def plot1():
    ax = axs[0][0]
    xv6,yv6,xv7,yv7 = get_data('mean_kl_loss',x_index = 'relative')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width)
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width)

    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.002):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Training Time')
    ax.set_ylabel('KL Loss')

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))



#total loss
def plot2():
    ax = axs[0][1]
    xv6,yv6,xv7,yv7 = get_data('total_loss',x_index = 'relative')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width)
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width)


    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.5):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Training Time')
    ax.set_ylabel('Total Loss')

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

# sample_steps
##The throughput of sampled environmental steps per second,
def plot3():
    ax = axs[1][0]
    xv6,yv6,xv7,yv7 = get_data('sample_steps',x_index = 'Step')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width)
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 3):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Throughput ( sample/sec )')

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

#Time in seconds this iteration took to run
def plot4():
    ax = axs[1][1]
    xv6, yv6, xv7, yv7 = get_data('iter_secs', x_index='Step')
    # 创建折线图
    ax.plot(xv6, yv6, color='#1565c0', linewidth=line_width)
    ax.plot(xv7, yv7, color='#df6172', linewidth=line_width)

    y_min, y_max = ax.get_ylim()
    # 循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 20):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0)

    # plt.title('amplitude_estimation')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Time Cost ( seconds) ')

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
if __name__ == '__main__':
    plot1()
    plot2()
    plot3()
    plot4()

    plt.tight_layout()
    plt.savefig(FileUtil.get_root_dir()+'/data/fig/tensorboard.png',dpi = dpi)
    # 显示图形
    plt.show()