import os

import qiskit
from qiskit import QuantumCircuit


class FileUtil:

    @staticmethod
    def read_all(path:str)-> str:
        root_dir = FileUtil.get_root_dir()
        path = os.path.join(root_dir,path)
        # Open the file in read mode
        file = open(path, 'r',encoding='utf-8')
        # Read the content of the file as a string
        content = file.read()
        # Close the file
        file.close()
        return content

    @staticmethod
    def get_root_dir():
        #方法一 从其他目录调用该方法不一定返回正确的目录

        # current_dir = os.getcwd() # Get the current directory
        # root_dir = os.path.dirname(current_dir)
        #方法二
        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = root_dir[:-6]
        return root_dir
    @staticmethod
    def write(file,content):

        # Open the file in write mode
        file = open(file, "w")
        # Close the file
        file.close()
if __name__ == '__main__':
    circuit = QuantumCircuit(5)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.cx(0, 4)
    print(circuit.qasm())