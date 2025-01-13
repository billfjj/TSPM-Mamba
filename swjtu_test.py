import multiprocessing as mp
import multiprocessing.pool as mpp
import time

import ttach as tta
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import *


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 0, 64]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 0, 64]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 0, 128]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [0, 0, 192]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [128, 128, 192]
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 9, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 0, 192]
    mask_rgb[np.all(mask_convert == 11, axis=0)] = [128, 128, 64]
    mask_rgb[np.all(mask_convert == 12, axis=0)] = [0, 128, 128]
    mask_rgb[np.all(mask_convert == 13, axis=0)] = [128, 128, 128]
    mask_rgb[np.all(mask_convert == 14, axis=0)] = [0, 128, 192]
    mask_rgb[np.all(mask_convert == 15, axis=0)] = [0, 128, 64]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, ori_img, rgb, img_size) = inp
    if rgb:
        image = cv2.imread(ori_img)
        (w, h) = image.shape[:2]
        image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        # overlay = np.zeros_like(image_resized)
        alpha = 0.5  # 透明度
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
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())

            image_path = input["img_path"]
            filename = os.path.basename(image_path[0])  # 获取路径的最后一部分
            image_ids = os.path.splitext(filename)[0]  # 去除扩展名
            if args.val:
                masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids
                ori_img =  'data/SWJTU-R/SWJTU_RGB/' + image_ids + '.jpg'
                if args.val:
                    if not os.path.exists(os.path.join(args.output_path)):
                        os.mkdir(os.path.join(args.output_path))
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    results.append((mask, str(args.output_path / mask_name), ori_img, args.rgb, config.image_size))
                else:
                    results.append((mask, str(args.output_path / mask_name), ori_img, args.rgb, config.image_size))
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


    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
