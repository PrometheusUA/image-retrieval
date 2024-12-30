import os

from torch.nn import CosineSimilarity
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from forwards import SimSiamForward
from models import SimSiamModel
from utils.data import TwoTransformsParallelTransform, GaussianBlur


RUN_NAME = 'simsiam_1_1_resnet50_bs64_100epochs_cosinelr_lr3e-3'
DIR_PATH = os.path.join('./checkpoints', RUN_NAME)


CONFIG = {
    'loaders': {
        'batch_size': 64,
        'num_workers': 16,
        'pin_memory': True,
        'persistent_workers': True,
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
        'train_on_test_too': True,
    },
    'model': SimSiamModel,
    'model_params': {
        'backbone': 'resnet50.a1_in1k',
        'pretrained': True,
        'frozen_backbone': False,\
        'embedding_dim': 1024,
        'predictor_dim': 256,
    },
    'forward': SimSiamForward,
    'forward_params': {
        'loss': CosineSimilarity(dim=1),
        'optimizer_config': {
            'optimizer': AdamW,
            'optimizer_params': {
                'lr': 3e-3,
                'weight_decay': 1e-4,
            },
        },
        'scheduler_config': {
            'scheduler': CosineAnnealingLR,
            'scheduler_params': {
                'T_max': 100,
                'eta_min': 1e-5,
            },
        },
        # 'transforms': T.Compose([
        #     T.RandomHorizontalFlip(),
        #     T.RandomRotation(15),
        #     T.RandomApply([
        #         T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.3),
        #     T.RandomGrayscale(p=0.3),
        #     T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value=0),
        #     T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(1/3, 3), value='random'),
        #     T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        # ]),
    },
    'trainer_params': {
        'max_epochs': 100,
        'callbacks': [
            ModelCheckpoint(monitor='val/loss', dirpath=DIR_PATH, mode='min'),
            EarlyStopping(monitor='val/loss', patience=20, mode='min'),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model="all"),
        'log_every_n_steps': 5,
    },
}
