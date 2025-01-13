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


CLASSES = ('Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
           'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist',
           # 'Void'
           )

PALETTE = [[128, 128, 128],
           [128, 0, 0],
           [192, 192, 128],
           [128, 64, 128],
           [0, 0, 192],
           [128, 128, 0],
           [192, 128, 128],
           [64, 64, 128],
           [64, 0, 128],
           [64, 64, 0],
           [0, 128, 192],
           # [0, 0, 0]
           ]


ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


class CamVidTrainDataset(Dataset):
    def __init__(self, data_root='data/CamVid/',
                 txt_dir='camvid_train.txt',
                 img_dir='CamVid_RGB', mosaic_ratio=0.25,
                 mask_dir='CamVidGray', img_suffix='.png', mask_suffix='.png',
                 transform=None, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.txt_dir = os.path.join(data_root, txt_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mosaic_ratio = mosaic_ratio

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.paths = self.get_img_ids(self.txt_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        img, mask = self.load_img_and_mask(index)
        if p_ratio < self.mosaic_ratio:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_path, mask_path = self.paths[index]
        results = {'img': img, 'gt_semantic_seg': mask, 'img_path': img_path, 'mask_path': mask_path}

        return results

    def __len__(self):
        length = len(self.paths)
        return length

    def get_img_ids(self, txt_dir):
        paths = []
        with open(txt_dir, 'r') as file:
            for line in file:
                line = line.strip()
                path1, path2 = line.split()
                paths.append((path1, path2))

        return paths

    def load_img_and_mask(self, index):
        img_path, mask_path = self.paths[index]
        img_name = osp.join(self.data_root, img_path)
        mask_name = osp.join(self.data_root, mask_path)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        # 将掩码图像转换为 numpy 数组
        mask_array = np.array(mask)

        # 将数组中所有的 255 值替换为 11
        mask_array[mask_array == 255] = 11

        # 将修改后的 numpy 数组转换回 PIL 图像
        mask = Image.fromarray(mask_array.astype(np.uint8))

        return img, mask

    def resize(self, img, mask):
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.paths) - 1) for _ in range(3)]
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


# camvid_val_dataset = CamVidTrainDataset(txt_dir='camvid_val.txt', mosaic_ratio=0.0,
#                                         transform=val_aug)


class CamVidTestDataset(Dataset):
    def __init__(self, data_root='data/CamVid/', txt_dir='camvid_val.txt',
                 img_suffix='.png',  mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.path_dir = os.path.join(data_root, txt_dir)

        self.img_suffix = img_suffix
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.paths = self.get_img_ids(self.path_dir)

    def __getitem__(self, index):
        img = self.load_img(index)

        img = np.array(img)
        resize = albu.Resize(height=self.img_size[0], width=self.img_size[1])(image=img)
        img = resize['image']
        aug = albu.Normalize()(image=img)
        img = aug['image']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_path, mask_path = self.paths[index]

        results = {'img': img, 'img_path': img_path, 'mask_path': mask_path}

        return results

    def __len__(self):
        length = len(self.paths)

        return length

    def get_img_ids(self, txt_dir):
        paths = []
        with open(txt_dir, 'r') as file:
            for line in file:
                line = line.strip()
                path1, path2 = line.split()
                paths.append((path1, path2))

        return paths

    def load_img(self, index):
        img_path, mask_path = self.paths[index]
        img_name = osp.join(self.data_root, img_path)
        img = Image.open(img_name).convert('RGB')

        return img
