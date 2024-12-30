import os
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms as T
from torch import tensor

from forwards import StandardForward
from models import MulticlassModel
from models.schedulers import CosineAnnealingLRWithWarmup


RUN_NAME = 'multiclass_standard_4_0_vit_clip_bs64_100epochs_cosinelrwarmup'
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)
BS = 64
EPOCHS = 100
STEPS_PER_EPOCH = 3597 // BS + 1

CONFIG = {
    'loaders': {
        'batch_size': BS,
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
    'model': MulticlassModel,
    'model_params': {
        'backbone': 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
        'num_classes': 1200,
        'pretrained': True,
        'frozen_backbone': True,
        'head_type': 'LongLinear',
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
        'max_epochs': EPOCHS,
        'callbacks': [
            ModelCheckpoint(monitor='val/map_at_5', dirpath=DIR_PATH, mode='max'),
            EarlyStopping(monitor='val/map_at_5', patience=10, mode='max'),
            LearningRateMonitor(logging_interval='step'),
        ],
        'log_every_n_steps': 20,
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model=True),
    },
}