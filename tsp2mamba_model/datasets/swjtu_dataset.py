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


CLASSES = ('background', 'dotted_line', 'car', 'bicyclist', 'road', 'guide_arrow',
           'person', 'small_truck', 'solid_line', 'diversion_line', 'motorcycle',
           'large_truck', 'deceleration_line', 'crosswalk', 'bus', 'tricycle')

PALETTE = [[0, 0, 0],
           [128, 0, 0],
           [0, 0, 64],
           [128, 0, 64],
           [0, 0, 128],
           [128, 0, 128],
           [0, 0, 192],
           [128, 128, 192],
           [0, 128, 0],
           [128, 128, 0],
           [128, 0, 192],
           [128, 128, 64],
           [0, 128, 128],
           [128, 128, 128],
           [0, 128, 192],
           [0, 128, 64]
           ]


# def get_training_transform():
#     train_transform = [
#         albu.Resize(height=1024, width=1024),
#         albu.HorizontalFlip(p=0.5),
#         albu.VerticalFlip(p=0.5),
#         albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
#         # albu.RandomRotate90(p=0.5),
#         # albu.OneOf([
#         #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
#         #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
#         # ], p=0.25),
#         albu.Normalize()
#     ]
#     return albu.Compose(train_transform)
#
#
# def train_aug(img, mask):
#     # multi-scale training and crop
#     crop_aug = Compose([Resize(size=(1024, 1024)),
#                         RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
#                         SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)])
#     img, mask = crop_aug(img, mask)
#
#     img, mask = np.array(img), np.array(mask)
#     aug = get_training_transform()(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask
#
#
# def get_val_transform():
#     val_transform = [
#         albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
#         albu.Normalize()
#     ]
#     return albu.Compose(val_transform)
#
#
# def val_aug(img, mask):
#     img, mask = np.array(img), np.array(mask)
#     aug = get_val_transform()(image=img.copy(), mask=mask.copy())
#     img, mask = aug['image'], aug['mask']
#     return img, mask


class SWJTUTrainDataset(Dataset):
    def __init__(self, data_root='data/SWJTU-R/', txt_dir='train_images.txt',
                 img_dir='SWJTU_RGB', mosaic_ratio=0.25,
                 mask_dir='SWJTU_Gray', img_suffix='.jpg', mask_suffix='.png',
                 transform=None, img_size=(1024, 1024)):
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
        name = self.paths[index]
        img_path = os.path.join(self.data_root, self.img_dir, (name + self.img_suffix))
        mask_path = os.path.join(self.data_root, self.mask_dir, (name + self.mask_suffix))
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
                name = line.split()[0]
                paths.append(name)

        return paths

    def load_img_and_mask(self, index):
        name = self.paths[index]
        img_path = os.path.join(self.data_root, self.img_dir, (str(name) + self.img_suffix))
        mask_path = os.path.join(self.data_root, self.mask_dir, (str(name) + self.mask_suffix))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # 将掩码图像转换为 numpy 数组
        mask_array = np.array(mask)

        # 将数组中所有的 255 值替换为 11
        mask_array[mask_array == 255] = 16

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


class SWJTUTestDataset(Dataset):
    def __init__(self, data_root='data/SWJTU-R/', txt_dir='val_images.txt',
                 img_dir='SWJTU_RGB', mosaic_ratio=0.0,
                 mask_dir='SWJTU_Gray', img_suffix='.jpg', mask_suffix='.png',
                 img_size=(1024, 1024)):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.path_dir = os.path.join(data_root, txt_dir)

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
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
        name = self.paths[index]
        img_path = os.path.join(self.data_root, self.img_dir, (str(name) + self.img_suffix))
        mask_path = os.path.join(self.data_root, self.mask_dir, (str(name) + self.mask_suffix))
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
                name = line.split()[0]
                paths.append(name)

        return paths

    def load_img(self, index):
        name = self.paths[index]
        img_path = os.path.join(self.data_root, self.img_dir, (str(name) + self.img_suffix))
        img_name = osp.join(self.data_root, img_path)
        img = Image.open(img_name).convert('RGB')

        return img


def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.png')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
