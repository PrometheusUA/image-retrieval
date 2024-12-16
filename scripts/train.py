import argparse
import sys

sys.path.append('./')

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from utils.data import get_loaders, get_default_transforms
from utils.main import read_py, get_device


def train(config_path):
    config = read_py(config_path)

    loaders_config = config.CONFIG['loaders']
    if 'train_transforms' not in loaders_config:
        loaders_config['train_transforms'] = get_default_transforms('train')
    if 'test_transforms' not in loaders_config:
        loaders_config['test_transforms'] = get_default_transforms('test')

    train_loader, val_loader, test_loader = get_loaders(**loaders_config)

    model = config.CONFIG['model'](**config.CONFIG['model_params'])
    forward = config.CONFIG['forward'](model, **config.CONFIG['forward_params'])
    trainer_params = config.CONFIG['trainer_params']
    if 'accelerator' not in trainer_params:
        trainer_params['accelerator'] = get_device(lightning=True)

    if 'logger' in trainer_params and isinstance(trainer_params['logger'], WandbLogger):
        trainer_params['logger'].experiment.config.update(config.CONFIG)
        trainer_params['logger'].watch(model)

    trainer = Trainer(**trainer_params)
    trainer.fit(forward, train_loader, val_loader)
    
    if 'logger' in trainer_params and isinstance(trainer_params['logger'], WandbLogger):
        trainer_params['logger'].experiment.unwatch(model)

    trainer.validate(forward, val_loader)
    # trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    train(args.config)
