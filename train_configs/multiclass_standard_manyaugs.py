import os
import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from forwards import StandardForward
from models import MulticlassModel
from models.schedulers import CosineAnnealingLRWithWarmup

EPOCHS = 100
BS = 64
RUN_NAME = f'multiclass_standard_2_2_resnet50_bs{BS}_{EPOCHS}epochs_cosinelrwarm_manyaugsgray'
STEPS_PER_EPOCH = 3597 // BS + 1
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)

CONFIG = {
    'loaders': {
        'batch_size': 64,
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': True,
        'train_transforms': T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.1),
            T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.RandomApply([
                T.RandomChoice([
                    T.ElasticTransform(alpha=5.0, sigma=10.0), 
                    T.ElasticTransform(alpha=3.0, sigma=5.0)
                    ]),
                ], 
                p=0.3),
            T.RandomGrayscale(p=0.3),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value=0),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value='random'),
            # T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), 
            #             std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ]),
        'test_transforms': T.Compose([
            T.Resize((288, 288)),
            T.ToTensor(),
            # T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), 
            #             std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ]),
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
        'loss': BCEWithLogitsLoss(),
        'optimizer_config': {
            'optimizer': AdamW,
            'optimizer_params': {
                'lr': 1e-3,
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
        'max_epochs': 100,
        'callbacks': [
            ModelCheckpoint(monitor='val/map_at_5', dirpath=DIR_PATH, mode='max'),
            EarlyStopping(monitor='val/map_at_5', patience=12, mode='max'),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model="all"),
    },
}