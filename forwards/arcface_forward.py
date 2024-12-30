from typing import Dict, Optional

from torch.nn import Module
from torch.nn.functional import one_hot, sigmoid, softmax
from torch.nn import CrossEntropyLoss
from lightning import LightningModule

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import mean_average_precision_at_k

class ArcfaceForward(LightningModule):
    def __init__(self, 
                 model: Module, 
                 loss: Module, 
                 optimizer_config: Dict, 
                 scheduler_config: Optional[Dict] = None, 
                 scheduler_interval: str = 'epoch', 
                 expand_labels:bool=False, 
                 arcface_loss_weight: float = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.arcface_loss = CrossEntropyLoss()
        self.loss = loss
        self.expand_labels = expand_labels
        self.arcface_loss_weight = arcface_loss_weight

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def training_step(self, input):
        imgs, labels = input
        preds, arcface_preds = self(imgs, labels)
        if self.expand_labels:
            labels_loss = one_hot(labels, num_classes=preds.size(1)).float()
        loss = self.loss(preds, labels_loss)

        arcface_labels = labels
        arcface_loss = self.arcface_loss(arcface_preds, arcface_labels)

        self.log('train/head_loss', loss, on_step=True, on_epoch=True)
        self.log('train/arcface_loss', arcface_loss, on_step=True, on_epoch=True)
        full_loss = loss + self.arcface_loss_weight * arcface_loss
        self.log('train/loss', full_loss, on_step=True, on_epoch=True)
        return full_loss

    def validation_step(self, input):
        imgs, labels = input
        preds, arcface_preds = self(imgs, labels)
        if self.expand_labels:
            labels_ = one_hot(labels, num_classes=preds.size(1)).float()
            loss = self.loss(preds, labels_)
        else:
            loss = self.loss(preds, labels)
        arcface_labels = labels
        arcface_loss = self.arcface_loss(arcface_preds, arcface_labels)

        self.log('val/head_loss', loss, on_epoch=True)
        self.log('val/arcface_loss', arcface_loss, on_epoch=True)
        full_loss = loss + self.arcface_loss_weight * arcface_loss
        self.log('val/loss', full_loss, on_epoch=True)

        # calculate metrics
        y_true = labels.cpu().numpy()
        y_preds = preds.cpu().numpy().argmax(axis=1)
        acc = accuracy_score(y_true, y_preds)
        self.log('val/acc', acc)
        precision = precision_score(y_true, y_preds, average='macro', zero_division=0)
        self.log('val/precision', precision)
        recall = recall_score(y_true, y_preds, average='macro', zero_division=0)
        self.log('val/recall', recall)
        f1 = f1_score(y_true, y_preds, average='macro', zero_division=0)
        self.log('val/f1', f1)
        map_at_5 = mean_average_precision_at_k(y_true, preds.cpu().numpy(), k=5)
        self.log('val/map_at_5', map_at_5)

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


class ArcfaceOnlyForward(LightningModule):
    def __init__(self, 
                 model: Module, 
                 loss: Module, 
                 optimizer_config: Dict, 
                 scheduler_config: Optional[Dict] = None, 
                 scheduler_interval: str = 'epoch', 
                 expand_labels:bool=False, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.loss = loss
        self.expand_labels = expand_labels

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def training_step(self, input):
        imgs, labels = input
        preds = self(imgs, labels)
        if self.expand_labels:
            labels = one_hot(labels, num_classes=preds.size(1)).float()
        loss = self.loss(preds, labels)

        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, input):
        imgs, labels = input
        preds = self(imgs)
        if self.expand_labels:
            labels_ = one_hot(labels, num_classes=preds.size(1)).float()
            loss = self.loss(preds, labels_)
        else:
            loss = self.loss(preds, labels)

        self.log('val/loss', loss, on_epoch=True)

        # calculate metrics
        y_true = labels.cpu().numpy()
        y_preds = preds.cpu().numpy().argmax(axis=1)
        acc = accuracy_score(y_true, y_preds)
        self.log('val/acc', acc)
        precision = precision_score(y_true, y_preds, average='macro', zero_division=0)
        self.log('val/precision', precision)
        recall = recall_score(y_true, y_preds, average='macro', zero_division=0)
        self.log('val/recall', recall)
        f1 = f1_score(y_true, y_preds, average='macro', zero_division=0)
        self.log('val/f1', f1)
        map_at_5 = mean_average_precision_at_k(y_true, preds.cpu().numpy(), k=5)
        self.log('val/map_at_5', map_at_5)

        return loss

    def test_step(self, input):
        imgs = input
        preds = self(imgs)
        return softmax(preds)
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_config['optimizer'](self.model.parameters(), **self.optimizer_config['optimizer_params'])
        if self.scheduler_config is not None:
            self.scheduler = self.scheduler_config['scheduler'](self.optimizer, **self.scheduler_config['scheduler_params'])
            return [self.optimizer], [{
                    'scheduler': self.scheduler,
                    'interval': self.scheduler_interval,
                }]
        return [self.optimizer]
