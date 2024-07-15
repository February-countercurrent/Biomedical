import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data.astype(np.float32)  # 变更数据类型为 float32

def resample(image, target_shape):
    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=1)  # 线性插值

def load_data_generator(data_dir, demographics_file, target_shape=(256, 256, 256), batch_size=2):
    t1_dir = os.path.join(data_dir, "T1")
    lesion_dir = os.path.join(data_dir, "Lesion")

    # 加载人口统计信息
    demographics_data = pd.read_excel(demographics_file)
    demographics_data['RandID'] = demographics_data['RandID'].str.replace('scan_', '')
    demographics_data.set_index('RandID', inplace=True)

    # 获取所有 T1 文件名
    t1_files = [f for f in os.listdir(t1_dir) if f.endswith(".nii.gz")]

    for i in range(0, len(t1_files), batch_size):
        t1_images = []
        lesion_masks = []
        extra_features = []

        batch_files = t1_files[i:i + batch_size]

        for file_name in batch_files:
            rand_id = file_name.split('_')[1]
            t1_file = os.path.join(t1_dir, file_name)
            lesion_file = os.path.join(lesion_dir, f"scan_{rand_id}_Lesion.nii.gz")

            if os.path.exists(lesion_file):
                t1_data = load_nifti(t1_file)
                lesion_data = load_nifti(lesion_file)

                # 归一化 T1 数据
                t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data))

                # 重新采样 T1 和病灶数据
                t1_data = resample(t1_data, target_shape)
                lesion_data = resample(lesion_data, target_shape)

                t1_images.append(t1_data)
                lesion_masks.append(lesion_data)

                # 获取对应的人口统计信息
                demographics_info = demographics_data.loc[rand_id]
                extra_features.append(demographics_info[['Age', 'Sex', 'TSI']].values.astype(np.float32))

        # 如果 batch 不为空，则转换为 numpy 数组，再转换为 PyTorch 张量，并优化数据类型
        if t1_images:
            t1_tensors = torch.tensor(np.array(t1_images), dtype=torch.float32).unsqueeze(1)  # (N, 1, D, H, W)
            lesion_tensors = torch.tensor(np.array(lesion_masks), dtype=torch.float32).unsqueeze(1)  # (N, 1, D, H, W)
            extra_features_tensors = torch.tensor(np.array(extra_features), dtype=torch.float32)  # (N, num_features)

            yield t1_tensors, lesion_tensors, extra_features_tensors
