import matplotlib.pyplot as plt

def plot_box_plot():
    #todo read from csv
    g1 = [370,350,342,360,320,341,326,326,324,339,336]
    g2 = [462,454,364,421,340,388,405,347,462,439,408.2]
    g3 = [346,333,352,335,326,322,345,337,337,327,339.8]

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
    ax.set_title("qnn_indep_qiskit_15")

    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_box_plot()