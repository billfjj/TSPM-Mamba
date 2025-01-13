from torch.utils.tensorboard.summary import image

from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random


CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]



ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


class CityscapesTrainDataset(Dataset):
    def __init__(self, data_root='/home/fjj/Data/City_scapes/cityscape/',
                 img_dir='train', mosaic_ratio=0.25,
                 mask_dir='trainlabel', img_suffix='_leftImg8bit.png', mask_suffix='_gtFine_labelTrainIds.png',
                 transform=None, img_size=(1024, 1024)):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mosaic_ratio = mosaic_ratio

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.names = self.get_img_ids()

    def __getitem__(self, index):
        p_ratio = random.random()
        img, mask = self.load_img_and_mask(index)
        if p_ratio < self.mosaic_ratio:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        name = self.names[index]
        img_path = os.path.join(self.data_root, self.img_dir, (name + self.img_suffix))
        mask_path = os.path.join(self.data_root, self.mask_dir, (name + self.mask_suffix))
        results = {'img': img, 'gt_semantic_seg': mask, 'img_path': img_path, 'mask_path': mask_path}

        return results

    def __len__(self):
        length = len(self.names)
        return length

    def get_img_ids(self):
        names = []
        img_path = os.path.join(self.data_root, self.img_dir)
        # 获取文件夹中所有文件的名称
        img_names = [
            '_'.join(file.split('_')[:-1]) for file in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, file))
        ]

        # 打印文件名称
        for img_name in img_names:
            names.append(img_name)

        return names

    def load_img_and_mask(self, index):
        name = self.names[index]
        img_path = os.path.join(self.data_root, self.img_dir, (str(name) + self.img_suffix))
        mask_path = os.path.join(self.data_root, self.mask_dir, (str(name) + self.mask_suffix))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # 将掩码图像转换为 numpy 数组
        mask_array = np.array(mask)

        # 将数组中所有的 255 值替换为 19
        mask_array[mask_array == 255] = 19

        # 将修改后的 numpy 数组转换回 PIL 图像
        mask = Image.fromarray(mask_array.astype(np.uint8))

        return img, mask

    def resize(self, img, mask):
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.names) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = self.resize(img_a, mask_a)
        img_b, mask_b = self.resize(img_b, mask_b)
        img_c, mask_c = self.resize(img_c, mask_c)
        img_d, mask_d = self.resize(img_d, mask_d)

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask


class CityscapesTestDataset(Dataset):
    def __init__(self, data_root='/home/fjj/Data/City_scapes/cityscape/',
                 img_dir='val', mosaic_ratio=0.0,
                 mask_dir='vallabel', img_suffix='_leftImg8bit.png', mask_suffix='_gtFine_labelTrainIds .png',
                 img_size=(1024, 1024)):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.names = self.get_img_ids()

    def __getitem__(self, index):
        img = self.load_img(index)

        img = np.array(img)
        resize = albu.Resize(height=self.img_size[0], width=self.img_size[1])(image=img)
        img = resize['image']
        aug = albu.Normalize()(image=img)
        img = aug['image']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        name = self.names[index]
        img_path = os.path.join(self.data_root, self.img_dir, (str(name) + self.img_suffix))
        mask_path = os.path.join(self.data_root, self.mask_dir, (str(name) + self.mask_suffix))
        results = {'img': img, 'img_path': img_path, 'mask_path': mask_path}

        return results

    def __len__(self):
        length = len(self.names)

        return length

    def get_img_ids(self):
        names = []
        img_path = os.path.join(self.data_root, self.img_dir)
        # 获取文件夹中所有文件的名称
        img_names = [
            '_'.join(file.split('_')[:-1]) for file in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, file))
        ]

        # 打印文件名称
        for img_name in img_names:
            names.append(img_name)

        return names

    def load_img(self, index):
        name = self.names[index]
        img_path = os.path.join(self.data_root, self.img_dir, (str(name) + self.img_suffix))
        img_name = osp.join(self.data_root, img_path)
        img = Image.open(img_name).convert('RGB')

        return img
