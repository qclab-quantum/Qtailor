import csv
import datetime
import os

from utils.file.file_util import FileUtil

sep = os.path.sep
rootdir = FileUtil.get_root_dir()
class CSVUtil:
    @staticmethod
    def write_data(file_path, data):
        # 确保文件夹存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 写入数据
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    @staticmethod
    def append_data(file_path, data):
        # 确保文件夹存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 追加数据
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    @staticmethod
    def read_data(file_path):
        data = []
        # 读取数据
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        return data

    @staticmethod
    def to_dataframe(abs_path=None, relative_path=None):
        import pandas as pd
        if  abs_path:
            path=abs_path
        else :
            path=rootdir+sep+relative_path

        df = pd.read_csv(path)
        # Display the dataframe
        return  df
    @staticmethod
    def wirte2darray(filepath,data):
        for row in data:
            CSVUtil.append_data(filepath,row)

    def new_csv(time_str, header=None):
        sep = '/'
        csv_path = FileUtil.get_root_dir() + sep + 'benchmark' + sep + 'a-result' + sep + time_str + '.csv'
        if header is None:
            header = [['datetime', 'qasm', 'rl', 'qiskit', 'mix', 'result', 'iter', 'checkpoint', 'remark', ]]
        CSVUtil.write_data(csv_path, data=header)
        return csv_path

def test():
    # 写入数据
    data = [['Name', 'Age'], ['Alice', 25], ['Bob', 30]]
    CSVUtil.write_data('d:/data.csv', data)

    # 追加数据
    new_data = [['Charlie', 35]]
    CSVUtil.append_data('d:/data.csv', new_data)

    # 读取数据
    data = CSVUtil.read_data('d:/data.csv')
    for row in data:
        print(row)

def demo():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M')
    rootdir = FileUtil.get_root_dir()
    sep = '/'
    csv_path = rootdir + sep + 'benchmark' + sep + 'a-result' + sep + formatted_datetime + '.csv'
    print(csv_path)
    CSVUtil.write_data(csv_path,[['datetime', 'qasm', 'rl', 'qiskit', 'rl_qiskit', 'result', 'iter', 'remark', 'checkpoint'] ])

if __name__ == '__main__':

    CSVUtil.to_dataframe(relative_path=r'data/train_metric/ae40/env6.csv')