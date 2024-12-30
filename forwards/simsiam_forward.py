from typing import Dict, Optional

import torch
from torch.nn import Module
from torch.nn.functional import one_hot, sigmoid, normalize
from lightning import LightningModule
from time import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import mean_average_precision_at_k


class SimSiamForward(LightningModule):
    def __init__(self, 
                 model: Module, 
                 loss: Module, 
                 optimizer_config: Dict, 
                 scheduler_config: Optional[Dict] = None, 
                 scheduler_interval: str = 'epoch', 
                #  transforms = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        # self.transforms = transforms
        # if self.transforms is None:
        #     self.transforms = lambda x: x
        #     print("Warning: No transforms provided!")
        self.loss = loss
        # self.automatic_optimization = False

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def training_step(self, input):
        imgs, _ = input
        z1, p1 = self(imgs[0])
        z2, p2 = self(imgs[1])

        loss = -(self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) * 0.5

        self.log('train/loss', loss, on_step=True, on_epoch=True)

        # opt = self.optimizers()
        # opt.zero_grad()
        # start_time = time()
        # loss.backward()
        # backward_time = time()
        # print(f"Backward time: {backward_time - start_time}")
        # opt.step()
        # step_time = time()
        # print(f"Optimizer step time: {step_time - backward_time}")

        return loss

    def validation_step(self, input):
        imgs, _ = input
        z1, p1 = self(imgs[0])
        z2, p2 = self(imgs[1])

        loss = -(self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) * 0.5
        self.log('val/loss', loss, on_epoch=True)



        # calculate metrics
        # y_true = labels.cpu().numpy()
        # y_preds = preds.cpu().numpy().argmax(axis=1)
        # acc = accuracy_score(y_true, y_preds)
        # self.log('val/acc', acc)
        # precision = precision_score(y_true, y_preds, average='macro', zero_division=0)
        # self.log('val/precision', precision)
        # recall = recall_score(y_true, y_preds, average='macro', zero_division=0)
        # self.log('val/recall', recall)
        # f1 = f1_score(y_true, y_preds, average='macro', zero_division=0)
        # self.log('val/f1', f1)
        # map_at_5 = mean_average_precision_at_k(y_true, preds.cpu().numpy(), k=5)
        # self.log('val/map_at_5', map_at_5)

        return loss

    def test_step(self, input):
        imgs = input
        preds = self(imgs)
        return sigmoid(preds)
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_config['optimizer'](self.model.parameters(), **self.optimizer_config['optimizer_params'])
        if self.scheduler_config is not None:
            self.scheduler = self.scheduler_config['scheduler'](self.optimizer, **self.scheduler_config['scheduler_params'])
            return [self.optimizer], [{
                    'scheduler': self.scheduler,
                    'interval': self.scheduler_interval,
                }]
        return [self.optimizer]