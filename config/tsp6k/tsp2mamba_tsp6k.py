import copy
import datetime

from catalyst import utils
from catalyst.contrib.nn import Lookahead
from fvcore.nn import flop_count, parameter_count
from torch.utils.data import DataLoader

from tsp2mamba_model.datasets.tsp6k_dataset import *
from tsp2mamba_model.losses import *
from tsp2mamba_model.models.TSP2Mamba import TSP2Mamba

# training hparam
max_epoch = 400
ignore_index = len(CLASSES)
train_batch_size = 3
val_batch_size = 3
lr = 8e-4
weight_decay = 1e-4
backbone_lr = 8e-5
backbone_weight_decay = 1e-4
num_classes = len(CLASSES)
classes = CLASSES

image_size = 512
INPUT_IMG_SIZE = (image_size, image_size)
TEST_IMG_SIZE = INPUT_IMG_SIZE
crop_size = (512, 512)
stride=(256, 256)

extract_model_name='rest_lite'
now = datetime.datetime.now()
weights_name = ("TSP2Mamba-tsp6k-" + str(image_size) + "-e" + str(max_epoch) + "-" +
                extract_model_name + "-" + "d8" + "-" + now.strftime("%Y%m%d%H%M"))
weights_path = "model_weights/tsp6k/{}".format(weights_name)
test_weights_name = "last-v1"
log_name = 'tsp6k/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 2
pretrained_ckpt_path = None  # the path for the pretrained TSP2Mamba weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
# VSSM parameters
PATCH_SIZE = 4
IN_CHANS = 3
DEPTHS = [2, 2, 9, 2]
EMBED_DIM = 96
SSM_D_STATE = 16
SSM_RATIO = 2.0
SSM_RANK_RATIO = 2.0
SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"
SSM_CONV = 3
SSM_CONV_BIAS = True
SSM_DROP_RATE = 0.0
SSM_INIT = "v0"
SSM_FORWARDTYPE = "v4"
MLP_RATIO = 4.0
MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
PATCH_NORM = True
NORM_LAYER = "ln"
DOWNSAMPLE = "v2"
PATCHEMBED = "v2"
GMLP = False
USE_CHECKPOINT = False

#  define the network
net = TSP2Mamba(
    extract_model_name=extract_model_name,
    pretrained=pretrained_ckpt_path,
    input_size=INPUT_IMG_SIZE,
    num_classes=num_classes,
    patch_size=PATCH_SIZE,
    in_chans=IN_CHANS,
    depths=DEPTHS,
    dims=EMBED_DIM,
    ssm_d_state=SSM_D_STATE,
    ssm_ratio=SSM_RATIO,
    ssm_rank_ratio=SSM_RANK_RATIO,
    ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
    ssm_act_layer=SSM_ACT_LAYER,
    ssm_conv=SSM_CONV,
    ssm_conv_bias=SSM_CONV_BIAS,
    ssm_drop_rate=SSM_DROP_RATE,
    ssm_init=SSM_INIT,
    forward_type=SSM_FORWARDTYPE,
    mlp_ratio=MLP_RATIO,
    mlp_act_layer=MLP_ACT_LAYER,
    mlp_drop_rate=MLP_DROP_RATE,
    drop_path_rate=DROP_PATH_RATE,
    patch_norm=PATCH_NORM,
    norm_layer=NORM_LAYER,
    downsample_version=DOWNSAMPLE,
    patchembed_version=PATCHEMBED,
    gmlp=GMLP,
    use_checkpoint=USE_CHECKPOINT
)

# define the loss
loss = TSP2MambaLoss(ignore_index=ignore_index)
use_aux_loss = True


# define the dataloader
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose(
        [
            Resize(size=(2048, 1024)),
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
            SmartCropV1(crop_size=INPUT_IMG_SIZE[0], max_ratio=0.75, ignore_index=ignore_index, nopad=False)
        ]
    )
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform(img_size):
    height, width = img_size
    val_transform = [
        # albu.Resize(height=2048, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    crop_aug = Compose(
        [
            Resize(size=(2048, 1024))
        ]
    )
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform(INPUT_IMG_SIZE)(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = TSPTrainDataset(transform=train_aug, img_size=INPUT_IMG_SIZE)

val_dataset = TSPTrainDataset(img_dir='images/val',
                              mask_dir='labels/val',
                              mosaic_ratio=0.0, transform=val_aug, img_size=INPUT_IMG_SIZE)
# val_dataset = TSPTrainDataset(img_dir='images/train',
#                               mask_dir='labels/train',
#                               mosaic_ratio=0.0, transform=val_aug, img_size=INPUT_IMG_SIZE)

test_dataset = val_dataset

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

model = copy.deepcopy(net)
model.cuda().eval()
input = torch.randn((1, 3, image_size, image_size), device=next(model.parameters()).device)
params = parameter_count(model)[""]
Gflops, unsupported = flop_count(model=model, inputs=(input,))
print('GFLOPs: ', sum(Gflops.values()), 'G')
print('Params: ', params / 1e6, 'M')
