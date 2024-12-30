import os
from timm.loss import AsymmetricLossSingleLabel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from forwards import StandardForward
from models import MulticlassModel


RUN_NAME = 'multiclass_standard_2_1_resnet50_bs64_100epochs_cosinelr_assymetric'
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)

CONFIG = {
    'loaders': {
        'batch_size': 64,
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': True,
    },
    'model': MulticlassModel,
    'model_params': {
        'backbone': 'resnet50.a1_in1k',
        'num_classes': 1200,
        'pretrained': True,
        'frozen_backbone': False,
        'head_dropout_rate': 0.2,
    },
    'forward': StandardForward,
    'forward_params': {
        'loss': AsymmetricLossSingleLabel(),
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
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model="all"),
    },
}