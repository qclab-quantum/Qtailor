import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#立体曲面

def normalize_array_to_range(arr, new_min=0.4, new_max=1.0):
    min_value = np.min(arr)
    max_value = np.max(arr)
    normalized_arr = (arr - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
    return normalized_arr

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate three groups of data
# np.random.seed(5)  # Set random seed for reproducibility
# x1, y1, z1 = np.random.rand(3, 10)
# x2, y2, z2 = np.random.rand(3, 10) + 1  # Offset to distinguish groups
# x3, y3, z3 = np.random.rand(3, 10) + 2

x = [10,15,20,25,30,35,40]
y = [2,3,4,5,6]
z1 = [
[11,	23,	    29,	  37, 40],
[25.9,	46.2,	47.9,	63.8,	81.7],
[33.3,	45.2,	72,	   74.8,	102.7],
[38,	64,	84.2,	109.9,	115.2],
[44,    67,	    90.4,	123.4,	156.6]  ,
[59,	100.7,	116,	142.6,	0],
[62.8,	98.3,	145,	208,	178.3],
]

z2=[
[12.5,	37	,44.8,	62.9,	58],
[36.3,	62.2,	74.9,	97,	119.3],
[46.5	,56.3	,104.5,	125.2,	155.3],
[66.5,	100.2,	154.8,	179,	189.2],
[66.2	,106.9,	135.7,185.2,	259.5],
[81.5,	130.5,	200.3,	224.6,	0],
[86.8,	150.9	,194.3,	255.1,	280.8,],
]

z3=[
[11	,23,	29	,37	,42.2],
[30.2,	55.6,	71.5,	80.6,	117.1],
[43.2,	49.7,	82	,96.4,	112],
[68.6,	109.6,	154.3,	185,	204.8],
[54.2,	91.8	,126.8,	168	,215.9],
[76.7,	134.9,	177.1,	222.2,	0],
[74.1,	138.2,	170.7,	227.9,	239.6]

]

alpha_z1 = normalize_array_to_range(z1)
alpha_z2 = normalize_array_to_range(z2)
alpha_z3 = normalize_array_to_range(z3)

# X, Y = np.meshgrid(x, y)
# z1=np.array(z1)
# z2=np.array(z2)
# 绘制第一个曲面
#surf1 = ax.plot_surface(X, Y, z1.T, cmap='viridis', edgecolor='none', label='Qtailor')
# 绘制第二个曲面
#surf2 = ax.plot_surface(X, Y, z2.T, cmap='inferno', edgecolor='none', label='Qiskit')

# Plot each group with different markers and colors
# Use the normalized z values as alpha values
for i in range(len(x)):
    for j in range(len(y)):
        print(x[i], y[j])
        ax.scatter(x[i], y[j], z1[i][j], c='r', marker='o', label='Qtailor' if (i == 0 and j==0) else "",alpha=alpha_z1[i][j] )
        ax.scatter(x[i], y[j], z2[i][j], c='blue', marker='^', label='Qiskit' if (i == 0 and j==0) else "",alpha=alpha_z2[i][j])
        #ax.scatter(x[i], y[j], z3[i][j], c='b', marker='s', label='MIX' if (i == 0 and j==0) else "", alpha=alpha_z3[i][j])

# Set axis labels
ax.set_xlabel('Bits')
ax.set_ylabel('Gates (bis * N)')
ax.set_zlabel('Depth')
from matplotlib.lines import Line2D
legend1 = Line2D([0], [0], label='Qtailor', color='w', markerfacecolor='w', marker='o')
legend2 = Line2D([0], [0], label='Qiskit', color='w', markerfacecolor='w', marker='^')
ax.legend(handles=[legend1, legend2])
# Add legend
ax.legend()
# 获取当前的Axes对象并对调y轴方向
#plt.gca().invert_yaxis()
# Show the plot
plt.savefig('fig7.png',dpi=300)
plt.show()

