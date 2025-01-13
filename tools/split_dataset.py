import os
import random

# 设置图像文件夹路径
root_path = '/home/fjj/Code/UNetMamba/data/SWJTU-R'
image_folder = 'SWJTU_RGB'

# 获取所有图像文件的名称
image_path = os.path.join(root_path, image_folder)
all_image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

# 随机打乱图像文件列表
random.shuffle(all_image_files)

# 计算训练集和验证集的大小
train_size = int(len(all_image_files) * 0.7)
val_size = len(all_image_files) - train_size

# 分割训练集和验证集
train_files = all_image_files[:train_size]
val_files = all_image_files[train_size:]

# 保存图像名称到txt文件，不包括文件后缀
def save_to_txt(file_list, txt_file_name):
    with open(txt_file_name, 'w') as f:
        for file in file_list:
            f.write(os.path.splitext(file)[0] + '\n')

# 保存训练集和验证集的图像名称
save_train_txt_path = os.path.join(root_path, 'train_images.txt')
save_val_txt_path = os.path.join(root_path, 'val_images.txt')
save_to_txt(train_files, save_train_txt_path)
save_to_txt(val_files, save_val_txt_path)

print(f'Training images saved to {save_train_txt_path}')
print(f'Validation images saved to {save_val_txt_path}')