import os
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from forwards import StandardForward
from models import MulticlassModel
from models.schedulers import CosineAnnealingLRWithWarmup


RUN_NAME = 'multiclass_standard_3_1_convnext_nano_tiny_bs32_100epochs_lr1e-4_cosinelrwarmup'
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)
BS = 32
EPOCHS = 100
STEPS_PER_EPOCH = 3597 // BS + 1

CONFIG = {
    'loaders': {
        'batch_size': BS,
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': True,
    },
    'model': MulticlassModel,
    'model_params': {
        'backbone': 'convnext_nano.in12k_ft_in1k',
        'num_classes': 1200,
        'pretrained': True,
        'frozen_backbone': False,
        'head_dropout_rate': 0.2,
    },
    'forward': StandardForward,
    'forward_params': {
        'loss': BCEWithLogitsLoss(),
        'optimizer_config': {
            'optimizer': AdamW,
            'optimizer_params': {
                'lr': 1e-4,
            },
        },
        'scheduler_config': {
            'scheduler': CosineAnnealingLRWithWarmup,
            'scheduler_params': {
                'T_max': STEPS_PER_EPOCH * EPOCHS,
                'eta_min': 1e-5,
                'warmup_steps': STEPS_PER_EPOCH // 2,
            },
        },
        'scheduler_interval': 'step',
        'expand_labels': True,
    },
    'trainer_params': {
        'max_epochs': EPOCHS,
        'callbacks': [
            ModelCheckpoint(monitor='val/map_at_5', dirpath=DIR_PATH, mode='max'),
            EarlyStopping(monitor='val/map_at_5', patience=10, mode='max'),
            LearningRateMonitor(logging_interval='step'),
        ],
        'log_every_n_steps': 20,
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model="all"),
    },
}