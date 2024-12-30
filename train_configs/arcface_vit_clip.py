import os
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
from torchvision import transforms as T

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from forwards import ArcfaceOnlyForward
from models import ArcfaceOnlyModel


RUN_NAME = 'arcface_1_0_vit_clip_bs64_100epochs_cosinelr_lr1e-3_backbonelr1e-5_hardmargin_s30_m0.15'
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)

CONFIG = {
    'loaders': {
        'batch_size': 64,
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': True,
        'train_transforms': T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.1),
            T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value=0),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value='random'),
            T.RandomGrayscale(p=0.3),
            T.Normalize(mean=tensor([0.4815, 0.4578, 0.4082]), 
                        std=tensor([0.2686, 0.2613, 0.2758]))
        ]),
        'test_transforms': T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=tensor([0.4815, 0.4578, 0.4082]), 
                        std=tensor([0.2686, 0.2613, 0.2758]))
        ]),
    },
    'model': ArcfaceOnlyModel,
    'model_params': {
        'backbone': 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
        'num_classes': 1200,
        'pretrained': True,
        'features_only': False,
        'frozen_backbone': True,
        'head_dropout_rate': 0.2,
        'easy_margin': False,
        's': 30.0,
        'm': 0.15,
        'neck_type': 'LongLinear',
    },
    'forward': ArcfaceOnlyForward,
    'forward_params': {
        'loss': CrossEntropyLoss(),
        'optimizer_config': {
            'optimizer': AdamW,
            'optimizer_params': {
                'lr': 1e-3,
            },
        },
        'scheduler_config': {
            'scheduler': CosineAnnealingLR,
            'scheduler_params': {
                'T_max': 100,
                'eta_min': 1e-5,
            },
        },
        'expand_labels': False,
    },
    'trainer_params': {
        'max_epochs': 100,
        'callbacks': [
            ModelCheckpoint(monitor='val/map_at_5', dirpath=DIR_PATH, mode='max'),
            EarlyStopping(monitor='val/map_at_5', patience=10, mode='max'),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model=False),
    },
}