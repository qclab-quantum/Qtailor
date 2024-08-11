import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
#立体曲面
mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] =15
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
z1 = np.array([
[11,	23,	    29,	  37, 40],
[25.9,	46.2,	47.9,	63.8,	81.7],
[33.3,	45.2,	72,	   74.8,	102.7],
[38,	64,	84.2,	109.9,	115.2],
[44,    67,	    90.4,	123.4,	156.6]  ,
[59,	100.7,	116,	142.6,	160],
[62.8,	98.3,	145,	208,	178.3],
])

z2=np.array([
[12.5,	37	,44.8,	62.9,	58],
[36.3,	62.2,	74.9,	97,	119.3],
[46.5	,56.3	,104.5,	125.2,	155.3],
[66.5,	100.2,	154.8,	179,	189.2],
[66.2	,106.9,	135.7,185.2,	259.5],
[81.5,	130.5,	200.3,	224.6,	270],
[86.8,	150.9	,194.3,	255.1,	280.8,],
])

z1=z1.T
z2=z2.T


# z3=[
# [11	,23,	29	,37	,42.2],
# [30.2,	55.6,	71.5,	80.6,	117.1],
# [43.2,	49.7,	82	,96.4,	112],
# [68.6,	109.6,	154.3,	185,	204.8],
# [54.2,	91.8	,126.8,	168	,215.9],
# [76.7,	134.9,	177.1,	222.2,	0],
# [74.1,	138.2,	170.7,	227.9,	239.6]
#
# ]

# alpha_z1 = normalize_array_to_range(z1)
# alpha_z2 = normalize_array_to_range(z2)
colors=['#c43737','#1e3b6d','#889a46','#ef9c09','#422511']

for i in range(len(y)):
        if i ==0 :
            ax.plot([y[i]]*7,x,z2[i],label='Qiskit', color=colors[i])
            ax.plot([y[i]]*7,x,z1[i],label='Qtailor', color=colors[i],linestyle='--')
        else:
            ax.plot([y[i]] * 7, x, z1[i],  color=colors[i], linestyle='--')
            ax.plot([y[i]] * 7, x, z2[i],  color=colors[i])



# Set axis labels
ax.set_xlabel('Gates Factor')
ax.set_ylabel('Qubits')
ax.set_zlabel('Circuits Depth')
ax.legend()
# 添加图例
plt.savefig('trend.pdf',dpi=600)
plt.show()

