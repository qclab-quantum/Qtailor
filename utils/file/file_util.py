import os
from pathlib import Path
import shutil
import zipfile

import qiskit
from qiskit import QuantumCircuit

class FileUtil:

    @staticmethod
    def read_all(path: str) -> str:
        root_dir = FileUtil.get_root_dir()
        file_path = root_dir / path
        # 使用 pathlib 的 read_text 方法
        return file_path.read_text(encoding='utf-8')

    @staticmethod
    def get_root_dir() -> Path:
        # 使用 pathlib 获取当前文件的父目录
        current_file = Path(__file__).resolve()
        # 向上查找到项目根目录 (假设需要向上11级)
        root_dir = current_file.parents[2]  # 根据实际目录结构调整
        return root_dir

    @staticmethod
    def write(file_path: str | Path, content: str):
        try:
            file_path = Path(file_path)
            # 创建父目录（如果不存在）
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # 写入文件
            file_path.write_text(content, encoding="utf-8")
            print(f"Successfully written to '{file_path}'.")

        except Exception as e:
            print(f"Error occurred: {e}")

    @staticmethod
    def compress_folder(path1: str):
        folder_path = Path(path1)
        
        # 检查文件夹是否存在
        if not folder_path.exists():
            print(f"文件夹 {folder_path} 不存在")
            return

        # 创建压缩文件
        zip_file_name = "compress.zip"
        zip_path = folder_path / zip_file_name
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 遍历文件夹下的所有文件
            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and file_path.name != zip_file_name:
                    # 计算相对路径
                    relative_path = file_path.relative_to(folder_path)
                    zip_file.write(file_path, relative_path)

        return zip_path, zip_file_name

    @staticmethod
    def copy(source_path: str | Path, dest_dir: str | Path, file_name: str):
        source_path = Path(source_path)
        dest_dir = Path(dest_dir)
        
        # 创建目标目录（如果不存在）
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 复制文件到目标目录
        dest_path = dest_dir / file_name
        shutil.copy2(source_path, dest_path)
        print(f"成功将文件 {source_path} 复制到 {dest_path}")

if __name__ == '__main__':
    # 调用示例
    path1 = "d:/test"
    path2 = "e:/"
    zip_path, zip_file_name = FileUtil.compress_folder(path1) # type: ignore
    FileUtil.copy(zip_path, path2, zip_file_name)






