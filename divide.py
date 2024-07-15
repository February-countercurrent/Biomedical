import os
import shutil

# 创建目标文件夹
os.makedirs('Data/T1', exist_ok=True)
os.makedirs('Data/Lesion', exist_ok=True)

# 定义源文件夹
source_folder = '../Dataset/Aims-Tbi'

# 获取文件列表
files = os.listdir(source_folder)

# 遍历文件并移动
for file in files:
    if 'T1' in file:
        shutil.move(os.path.join(source_folder, file), 'Data/T1/' + file)
    elif 'Lesion' in file:
        shutil.move(os.path.join(source_folder, file), 'Data/Lesion/' + file)

# 打印T1文件夹中的文件
print("T1 files:", os.listdir('Data/T1'))

# 打印Lesion文件夹中的文件
print("Lesion files:", os.listdir('Data/Lesion'))
