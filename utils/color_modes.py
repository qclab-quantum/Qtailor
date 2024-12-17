import os
from PIL import Image

def get_image_modes(directory):
    # 遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, file)
            try:
                # 打开图片文件
                with Image.open(file_path) as img:
                    # 输出图片路径和颜色模式
                    print(f"Image: {file_path}, Mode: {img.mode}")
            except IOError:
                # 如果文件不是图片，跳过
                #print(f"File {file_path} is not an image or cannot be opened.")
                pass

# 使用函数，指定要检查的目录
directory_path = r'D:\sync\camera ready\main\figure'
get_image_modes(directory_path)