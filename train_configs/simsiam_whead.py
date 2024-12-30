import os
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms as T
from torch import tensor

from forwards import SimSiamClassificationForward
from models import SimSiamModel, SimSiamMultilabelHead
from models.schedulers import CosineAnnealingLRWithWarmup


RUN_NAME = 'multiclass_simsiam_based_1_0_bs64_100epochs_cosinelrwarmup'
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
        ]),
        'test_transforms': T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
        ]),
    },
    'model': SimSiamMultilabelHead,
    'model_params': {
        'num_classes': 1200,
        'embedding_dim': 1024,
        'head_type': 'MLP',
        'head_dropout_rate': 0.2,
    },
    'forward': SimSiamClassificationForward,
    'forward_params': {
        'simsiam_model': SimSiamModel(**{
            'backbone': 'resnet50.a1_in1k',
            'pretrained': True,
            'frozen_backbone': False,\
            'embedding_dim': 1024,
            'predictor_dim': 256,
        }),
        'simsiam_model_checkpoint': 'checkpoints/simsiam_1_1_resnet50_bs64_100epochs_cosinelr_lr3e-3/epoch=18-step=4921.ckpt',
        'freeze_simsiam': True,
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
        'logger': WandbLogger(name=RUN_NAME, project='image_retrieval', log_model=False),
    },
}