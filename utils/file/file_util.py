import os

import qiskit
from qiskit import QuantumCircuit
import os
import shutil
import zipfile

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
        root_dir = root_dir[:-11]
        return root_dir
    @staticmethod
    def write(file,content):
        try:
            directory = os.path.dirname(file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Open file in write mode; if file doesn't exist, it will be created
            file = open(file, "w",encoding="utf-8")
            file.write(content)
            file.close()
            print(f"Successfully written to '{file}'.")

        except Exception as e:
            print(f"Error occurred: {e}")

    @staticmethod
    def compress_folder(path1):
        # 检查 path1 文件夹是否存在
        if not os.path.exists(path1):
            print(f"文件夹 {path1} 不存在")
            return

        # 创建压缩文件
        zip_file_name = "compress.zip"
        zip_path = os.path.join(path1, zip_file_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 遍历 path1 文件夹下的所有文件和文件夹，逐个添加到压缩文件中
            for root, dirs, files in os.walk(path1):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 在压缩文件中创建相同的目录结构
                    zip_file.write(file_path, os.path.relpath(file_path, path1))

        return zip_path,zip_file_name

    @staticmethod
    def copy(path1,path2,file_name):
        # 检查 path2 文件夹是否存在，不存在则创建
        if not os.path.exists(path2):
            os.makedirs(path2)

        # 复制压缩文件到 path2
        shutil.copy(path1, os.path.join(path2, file_name))
        print(f"成功将文件夹 {path1} 下的文件打包为 {file_name} 并复制到 {path2}")
if __name__ == '__main__':
    # 调用示例
    path1 = "d:/test"
    path2 = "e:/"
    zip_path,zip_file_name = FileUtil.compress_folder(path1)
    FileUtil.copy(zip_path,path2,zip_file_name)






