import os
import random
import shutil

# 定义相对数据目录
data_dir = 'Merged'
t1_dir = os.path.join(data_dir, 'T1')
lesion_dir = os.path.join(data_dir, 'Lesion')

# 获取所有T1和Lesion文件列表
t1_files = [f for f in os.listdir(t1_dir) if os.path.isfile(os.path.join(t1_dir, f))]
lesion_files = [f for f in os.listdir(lesion_dir) if os.path.isfile(os.path.join(lesion_dir, f))]

# 确保T1和Lesion文件数量一致
assert len(t1_files) == len(lesion_files), "T1 and Lesion file counts do not match."

# 打乱文件列表
combined_files = list(zip(t1_files, lesion_files))
random.shuffle(combined_files)

# 分割比例
total_files = len(combined_files)
train_count = 272
val_count = 58
test_count = total_files - train_count - val_count

# 分割文件列表
train_files = combined_files[:train_count]
val_files = combined_files[train_count:train_count + val_count]
test_files = combined_files[train_count + val_count:]

print(f"训练集文件数量: {len(train_files)}")
print(f"验证集文件数量: {len(val_files)}")
print(f"测试集文件数量: {len(test_files)}")

# 定义新的目录
prepared_data_dir = 'PreparedData'
new_train_t1_dir = os.path.join(prepared_data_dir, 'new_train', 'T1')
new_train_lesion_dir = os.path.join(prepared_data_dir, 'new_train', 'Lesion')
new_val_t1_dir = os.path.join(prepared_data_dir, 'new_val', 'T1')
new_val_lesion_dir = os.path.join(prepared_data_dir, 'new_val', 'Lesion')
new_test_t1_dir = os.path.join(prepared_data_dir, 'test', 'T1')
new_test_lesion_dir = os.path.join(prepared_data_dir, 'test', 'Lesion')

# 创建新的目录
os.makedirs(new_train_t1_dir, exist_ok=True)
os.makedirs(new_train_lesion_dir, exist_ok=True)
os.makedirs(new_val_t1_dir, exist_ok=True)
os.makedirs(new_val_lesion_dir, exist_ok=True)
os.makedirs(new_test_t1_dir, exist_ok=True)
os.makedirs(new_test_lesion_dir, exist_ok=True)

# 复制文件到新的目录
def copy_files(file_pairs, src_dirs, dest_dirs):
    for t1_file, lesion_file in file_pairs:
        t1_src_path = os.path.join(src_dirs[0], t1_file)
        lesion_src_path = os.path.join(src_dirs[1], lesion_file)
        t1_dest_path = os.path.join(dest_dirs[0], t1_file)
        lesion_dest_path = os.path.join(dest_dirs[1], lesion_file)
        shutil.copy(t1_src_path, t1_dest_path)
        shutil.copy(lesion_src_path, lesion_dest_path)

# 复制训练集文件
copy_files(train_files, [t1_dir, lesion_dir], [new_train_t1_dir, new_train_lesion_dir])

# 复制验证集文件
copy_files(val_files, [t1_dir, lesion_dir], [new_val_t1_dir, new_val_lesion_dir])

# 复制测试集文件
copy_files(test_files, [t1_dir, lesion_dir], [new_test_t1_dir, new_test_lesion_dir])

# 将文件列表写入到txt文件中
def write_files_to_txt(file_pairs, txt_file_path):
    with open(txt_file_path, 'w') as file:
        for idx, (t1_file, lesion_file) in enumerate(file_pairs):
            file.write(f"{idx+1}: {t1_file}, {lesion_file}\n")

# 写入训练集文件列表
write_files_to_txt(train_files, os.path.join(prepared_data_dir, 'train_files.txt'))

# 写入验证集文件列表
write_files_to_txt(val_files, os.path.join(prepared_data_dir, 'val_files.txt'))

# 写入测试集文件列表
write_files_to_txt(test_files, os.path.join(prepared_data_dir, 'test_files.txt'))

print("文件复制和列表写入完成")
