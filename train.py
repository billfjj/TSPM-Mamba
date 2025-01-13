import argparse
import io
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn

from tools.cfg import py2cfg
from tools.metric import Evaluator

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss
        self.crop_size = config.crop_size
        self.stride = config.stride

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        # 记录loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU*100.0,
                      'F1': F1*100.0,
                      'OA': OA*100.0}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou*100.0
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        if self.crop_size:
            # 滑块推理
            h_stride, w_stride = self.stride
            h_crop, w_crop = self.crop_size
            batch_size, _, h_img, w_img = img.size()
            out_channels = self.config.num_classes
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
                    crop_seg_logit = self.forward(crop_img)
                    # preds += F.pad(crop_seg_logit,
                    #                (int(x1), int(preds.shape[3] - x2), int(y1),
                    #                 int(preds.shape[2] - y2)))
                    preds[:, :, y1:y2, x1:x2] = crop_seg_logit
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            prediction = preds / count_mat
            pre_mask = nn.Softmax(dim=1)(prediction)
            pre_mask = pre_mask.argmax(dim=1)
        else:
            prediction = self.forward(img)
            pre_mask = nn.Softmax(dim=1)(prediction)
            pre_mask = pre_mask.argmax(dim=1)

        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())
            mAcc = np.nanmean(self.metrics_val.Pixel_Accuracy_Class())
            mDice = np.nanmean(self.metrics_val.Dice())
            mPrecision = np.nanmean(self.metrics_val.Precision())
            mRecall = np.nanmean(self.metrics_val.Recall())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {
            'mIoU': mIoU*100.0,
            'F1': F1*100.0,
            'OA': OA*100.0,
            'mAcc': mAcc*100.0,
            'Dice': mDice*100.0,
            'mPrecision': mPrecision*100.0,
            'mRecall': mRecall*100.0,
        }
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou*100.0
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {
            'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA ,
            'val_mAcc': mAcc, 'val_Dice': mDice, 'val_mPrecision': mPrecision,
            'val_mRecall': mRecall
        }
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


class LoggingProgressBar(RichProgressBar):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.log_stream = io.StringIO()

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.log_stream = io.StringIO()

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        with open(self.log_file, 'a') as f:
            f.write(self.log_stream.getvalue())
        self.log_stream.close()

    def print(self, *args, **kwargs):
        super().print(*args, **kwargs)
        print(*args, file=self.log_stream, **kwargs)


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        # save_top_k=-1,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        every_n_epochs=1,
        filename='model-{epoch:02d}-{val_mIoU:.2f}'
    )
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)

    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)


    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator='auto',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        strategy='auto',
        logger=logger,
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
