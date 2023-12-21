import csv
import datetime
import os

from utils.file_util import FileUtil


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

if __name__ == '__main__':
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M')
    rootdir = FileUtil.get_root_dir()
    sep = '/'
    csv_path = rootdir + sep + 'benchmark' + sep + 'a-result' + sep + formatted_datetime + '.csv'
    print(csv_path)
    CSVUtil.write_data(csv_path,[['datetime', 'qasm', 'rl', 'qiskit', 'rl_qiskit', 'result', 'iter', 'remark', 'checkpoint'] ])