import torch
import numpy as np
from model import UNet
from data_loader import load_nifti, resample

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def remove_module_prefix(state_dict):
    """移除state_dict中键的'module.'前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # remove 'module.' prefix
        new_state_dict[k] = v
    return new_state_dict

# 加载模型
print("Loading model...")
model = UNet().to(device)
checkpoint = torch.load('result/Vnet/final_model.pth', map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

# 预处理T1图像
def load_and_preprocess_image(nifti_path):
    t1_data = load_nifti(nifti_path)
    t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data))  # 归一化
    t1_data = resample(t1_data, (128, 128, 128))  # 重新采样
    t1_tensor = torch.tensor(t1_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128, 128)
    print(f"Preprocessed image tensor: {t1_tensor}")
    return t1_tensor.to(device)

# 预处理人口统计信息
def load_and_preprocess_features(age, sex, tsi):
    extra_features = torch.tensor([age, sex, tsi], dtype=torch.float32).unsqueeze(0)  # 转换为二维张量，形状为 (1, num_features)
    print(f"Preprocessed demographic features: {extra_features}")
    return extra_features.to(device)

# 预测是否存在病灶
def predict_lesion(nifti_path, age, sex, tsi):
    # 预处理图像和人口统计信息
    t1_tensor = load_and_preprocess_image(nifti_path)
    extra_features = load_and_preprocess_features(age, sex, tsi)

    # 进行预测
    with torch.no_grad():
        outputs = model(t1_tensor, extra_features)
        preds = (outputs > 0.5).float()  # 二值化预测结果

    # 打印输出张量和二值化预测结果
    print("Model output tensor:")
    print(outputs)
    print("Binarized prediction tensor:")
    print(preds)

    # 打印二值化张量中为1的元素数量
    num_ones = preds.sum().item()
    print(f"Number of elements predicted as lesion (value > 0.5): {num_ones}")

    # 判断是否有病灶
    if num_ones > 0:
        has_lesion = True
    else:
        has_lesion = False
    return has_lesion


def main():
    # 示例图像路径和人口统计信息
    # nifti_path = 'PreparedData/test/T1/scan_1099_T1.nii.gz'
    # age = 42  # 示例年龄
    # sex = 2  # 示例性别（1表示男性，0表示女性）
    # tsi = 780  # 示例TSI

    nifti_path = 'PreparedData/test/T1/scan_0093_T1.nii.gz'
    age = 18.22438356  # 示例年龄
    sex = 1  # 示例性别（1表示男性，0表示女性）
    tsi = 81.34794521  # 示例TSI

    # 预测是否存在病灶
    has_lesion = predict_lesion(nifti_path, age, sex, tsi)

    # 输出结果
    if has_lesion:
        print("The image has a lesion.")
    else:
        print("The image does not have a lesion.")


if __name__ == "__main__":
    main()