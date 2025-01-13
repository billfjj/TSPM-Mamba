import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from PIL import Image
from shapely.ops import orient

from train import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [244, 35, 232]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [70, 70, 70]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [80, 90, 40]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [180, 165,180]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [107, 142, 35]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [152, 251,152]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [70, 130, 180]
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 9, axis=0)] = [255, 100, 0]
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [0, 0, 142]
    mask_rgb[np.all(mask_convert == 11, axis=0)] = [0, 0, 70]
    mask_rgb[np.all(mask_convert == 12, axis=0)] = [0, 60, 100]
    mask_rgb[np.all(mask_convert == 13, axis=0)] = [0, 0, 230]
    mask_rgb[np.all(mask_convert == 14, axis=0)] = [119, 11, 32]
    mask_rgb[np.all(mask_convert == 15, axis=0)] = [250, 170, 160]
    mask_rgb[np.all(mask_convert == 16, axis=0)] = [250, 200, 160]
    mask_rgb[np.all(mask_convert == 17, axis=0)] = [250, 240, 180]
    mask_rgb[np.all(mask_convert == 18, axis=0)] = [153, 153, 153]
    mask_rgb[np.all(mask_convert == 19, axis=0)] = [250, 170, 30]
    mask_rgb[np.all(mask_convert == 20, axis=0)] = [220, 220, 0]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, ori_img, rgb, img_size) = inp
    if rgb:
        image = cv2.imread(ori_img)
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        (w, h) = mask_tif.shape[:2]
        image_resized = cv2.resize(image, (h, w), interpolation=cv2.INTER_AREA)
        # overlay = np.zeros_like(image_resized)
        alpha = 0.5  # 透明度
        # alpha = 1  # 透明度
        combined_image = (1 - alpha) * image_resized + alpha * mask_tif
        combined_image = cv2.resize(combined_image, (h, w), interpolation=cv2.INTER_AREA)
        cv2.imwrite(mask_name_tif, combined_image)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-p", "--test_weights_path", type=Path, required=True, help="Path to model state dicts")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval Val set", action='store_true')
    return parser.parse_args()


def slip_predict(img, crop_size, stride, num_classes, model):
    # 滑块推理
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.size()
    out_channels = num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # 初始化输出张量
    preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_seg_logit = model(crop_img)
            # preds += F.pad(crop_seg_logit,
            #                (int(x1), int(preds.shape[3] - x2), int(y1),
            #                 int(preds.shape[2] - y2)))
            preds[:, :, y1:y2, x1:x2] = crop_seg_logit
            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    prediction = preds / count_mat
    # pre_mask = nn.Softmax(dim=1)(prediction)
    # pre_mask = pre_mask.argmax(dim=1)
    return prediction


def visualizing_feature_maps():
    pass


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(args.test_weights_path, config=config)
    model.cuda()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        # results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            # raw_predictions = model(input['img'].cuda())
            crop_size = (512, 512)
            stride = (256, 256)
            raw_predictions = slip_predict(input['img'].cuda(), crop_size, stride, 21, model)

            image_path = input["img_path"]
            filename = os.path.basename(image_path[0])  # 获取路径的最后一部分
            image_ids = os.path.splitext(filename)[0]  # 去除扩展名
            if args.val:
                masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                results = []
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids
                ori_img =  image_path[0]
                if args.val:
                    if not os.path.exists(os.path.join(args.output_path)):
                        os.mkdir(os.path.join(args.output_path))
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    results.append((mask, str(args.output_path / mask_name), ori_img, args.rgb, config.image_size))
                else:
                    results.append((mask, str(args.output_path / mask_name), ori_img, args.rgb, config.image_size))

                mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)

    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        ac_per_class = evaluator.Pixel_Accuracy_Class()
        dice_per_class = evaluator.Dice()
        precision_per_class = evaluator.Precision()
        recall_per_class = evaluator.Recall()
        OA = evaluator.OA()
        # for class_name, class_iou, class_f1, class_ac in zip(config.classes, iou_per_class, f1_per_class, ac_per_class):
        #     print('mF1_{}:{}, IOU_{}:{}, AP_{}:{}'.format(class_name, class_f1,
        #                                                   class_name, class_iou,
        #                                                   class_name, class_ac))
        print('aAcc:{}, mIOU:{}, mAcc:{}, mDice:{}, '
              'mFscore:{}, mPrecision:{}, mRecall:{}'.format(OA,
                                                             np.nanmean(iou_per_class),
                                                             np.nanmean(ac_per_class),
                                                             np.nanmean(dice_per_class),
                                                             np.nanmean(f1_per_class),
                                                             np.nanmean(precision_per_class),
                                                             np.nanmean(recall_per_class)))


    # t0 = time.time()
    # mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    # t1 = time.time()
    # img_write_time = t1 - t0
    # print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
    # visualizing_feature_maps()
    # model = torch.load('model_weights/tsp6k/mfossmamba-tsp6k-512-e200-202411220000/model-epoch=197-val_mIoU=0.69.ckpt')
    # print(model)