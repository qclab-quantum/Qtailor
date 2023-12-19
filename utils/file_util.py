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

        try:
            directory = os.path.dirname(file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Open file in write mode; if file doesn't exist, it will be created
            file = open(file, "w")
            file.write(content)
            file.close()
            print(f"Successfully written to '{file}'.")

        except Exception as e:
            print(f"Error occurred: {e}")
if __name__ == '__main__':
    file = 'D:\\project\\QCC\\qccrl\\benchmark\\a-result\\ghz\\ghz_indep_qiskit_25.qasm_111.txt'
    content = '123123'
    FileUtil.write(file,content)