#compare reward replay(env7) and traditional RL(env6)
import matplotlib.pyplot as plt
import numpy as np
from utils.file.csv_util import CSVUtil
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from utils.file.file_util import FileUtil
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# from matplotlib import font_manager
#
# # 列出可用字体
# available_fonts = sorted([f.name for f in font_manager.fontManager.ttflist])
# print(available_fonts)
mpl.rcParams['font.size'] = 20
label_size = 18
line_width = 2.0
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
# 定义格式化函数，将秒转换为小时格式
def seconds_to_hours(x, pos):
    # 将秒转换为小时
    hours = x / 3600
    return f'{hours:.1f}'
def get_data(folder,x_index):
    dfv6=CSVUtil.to_dataframe(relative_path=f'data\\train_demo\\ae40/{folder}/env6.csv')
    dfv7=CSVUtil.to_dataframe(relative_path=f'data\\train_demo\\ae40/{folder}/env7.csv')

    xv6=dfv6[x_index].values
    yv6=dfv6['Value'].values

    xv7=dfv7[x_index].values
    yv7=dfv7['Value'].values

    return xv6,yv6,xv7,yv7

# 创建一个2x1的子图
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

#mean kl loss
def plot1():
    ax = axs[0]
    xv6,yv6,xv7,yv7 = get_data('mean_kl_loss',x_index = 'relative')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width, label='PPO')
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width, label='RR-PPO')

    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.002):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Training Time(hour)',fontsize = label_size)
    ax.set_ylabel('KL Loss',fontsize = label_size)

    # ustom formatter

    #ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    # 设置 x 轴的主刻度为每 1800 秒（0.5 小时）
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))

    # 使用 FuncFormatter 应用自定义格式
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_hours))

    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    ax.legend(loc='upper right', fontsize='small')



#total loss
def plot2():
    ax = axs[1]
    xv6,yv6,xv7,yv7 = get_data('total_loss',x_index = 'relative')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width, label='PPO')
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width, label='RR-PPO')


    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.5):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Training Time(hour)',fontsize = label_size)
    ax.set_ylabel('Total Loss',fontsize = label_size)

    # ustom formatter
    # 设置 x 轴的主刻度为每 1800 秒（0.5 小时）
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1800))

    # 使用 FuncFormatter 应用自定义格式
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_hours))
    ax.legend(loc='upper right', fontsize='small')

# sample_steps
##The throughput of sampled environmental steps per second,
def plot3():
    ax = axs[0]
    xv6,yv6,xv7,yv7 = get_data('sample_time_s',x_index = 'Step')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width, label='PPO')
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width, label='RR-PPO')
    ax.set_ylim(bottom=85)
    ax.set_ylim(top=190)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 10):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Steps',fontsize = label_size)
    ax.set_ylabel('Sample Time (s)',fontsize = label_size)

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(loc='upper right', fontsize='medium')

#Time in seconds this iteration took to run
def plot4():
    ax = axs[1]
    xv6, yv6, xv7, yv7 = get_data('train_iter_s', x_index='Step')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width, label='PPO')
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width, label='RR-PPO')
    ax.set_ylim(bottom=85)
    ax.set_ylim(top=190)
    y_min, y_max = ax.get_ylim()
    # 循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 10):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0)

    # plt.title('amplitude_estimation')
    ax.set_xlabel('Steps',fontsize = label_size)
    ax.set_ylabel('Iteration Time (s) ',fontsize = label_size)

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.legend(loc='upper right', fontsize='medium')

def plot_t1():
    plot1()
    plot2()
    plt.tight_layout()
    plt.savefig(FileUtil.get_root_dir()+'/data/fig/t1.png',dpi = dpi)
    plt.show()

def plot_t2():
    plot3()
    plot4()
    plt.tight_layout()
    plt.savefig(FileUtil.get_root_dir()+'/data/fig/t2.png',dpi = dpi)
    plt.show()

if __name__ == '__main__':
    plot_t2()