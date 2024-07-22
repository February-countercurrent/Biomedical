import torch
import numpy as np
import pandas as pd
from model import UNet
from data_loader import load_nifti, resample  # 确保从正确的位置导入数据加载器

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义移除 module. 前缀的函数
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# 加载模型
print("Loading model...")
model = UNet().to(device)
checkpoint = torch.load('result/RTX4090/final_model.pth', map_location=device)
state_dict = remove_module_prefix(checkpoint)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

def load_and_preprocess_image(nifti_path):
    t1_data = load_nifti(nifti_path)
    t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data))  # 归一化
    t1_data = resample(t1_data, (256, 256, 256))  # 重新采样
    t1_tensor = torch.tensor(t1_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256, 256)
    return t1_tensor.to(device)

def load_and_preprocess_features(demographics_info):
    extra_features = torch.tensor(demographics_info[['Age', 'Sex', 'TSI']].values.astype(np.float32), dtype=torch.float32)
    return extra_features.to(device)

def predict_lesion(nifti_path, demographics_info):
    # 预处理图像和人口统计信息
    t1_tensor = load_and_preprocess_image(nifti_path)
    extra_features = load_and_preprocess_features(demographics_info)

    # 进行预测
    with torch.no_grad():
        outputs = model(t1_tensor, extra_features)
        preds = (outputs > 0.5).float()  # 二值化预测结果

    # 判断是否有病灶
    has_lesion = preds.sum().item() > 0
    return has_lesion

def main():
    # 示例输入
    nifti_path = 'PreparedData/val/T1/scan_0237_T1.nii.gz'
    demographics_info = pd.DataFrame({
        'Age': [13.66666667],
        'Sex': [2],
        'TSI': [56.57142857],
        'ScanManufacturer': ['Philips']
    })

    # 执行预测
    has_lesion = predict_lesion(nifti_path, demographics_info)
    if has_lesion:
        print("Prediction: Lesion detected")
    else:
        print("Prediction: No lesion detected")

if __name__ == '__main__':
    main()