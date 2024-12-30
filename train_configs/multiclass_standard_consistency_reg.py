import os
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from forwards import ConsistencyRegularizationForward
from models import MulticlassModel
from utils.data import TwoTransformsParallelTransform, GaussianBlur


RUN_NAME = 'multiclass_standard_consistency_reg_1_1_resnet50_bs24_lr1e-3_100epochs_cosinelr_lambda100.0'
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)

CONFIG = {
    'loaders': {
        'batch_size': 24,
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': True,
        'semisupervised': True,
        'train_transforms': TwoTransformsParallelTransform(
            T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.6),
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.ToTensor(),
            ]), 
            T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.3),
                T.ToTensor(),
                T.RandomGrayscale(p=0.3),
                T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value=0),
                T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value='random'),
                T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            ])
        ),
        'test_transforms': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
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
    'forward': ConsistencyRegularizationForward,
    'forward_params': {
        'supervised_loss': BCEWithLogitsLoss(),
        'consistency_loss': MSELoss(),
        'lambda_consistency': 100.0,
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
        'expand_labels': True,
    },
    'trainer_params': {
        'max_epochs': 100,
        'callbacks': [
            ModelCheckpoint(monitor='val/map_at_5', dirpath=DIR_PATH, mode='max'),
            EarlyStopping(monitor='val/map_at_5', patience=5, mode='max'),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model="all"),
        'log_every_n_steps': 20,
    },
    # 'start_checkpoint': './checkpoints/multiclass_standard_consistency_reg_1_1_resnet50_bs24_lr1e-3_100epochs_cosinelr_lambda2.0/epoch=6-step=1001.ckpt',
}