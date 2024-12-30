from typing import Dict, Optional

import torch
from torch.nn import Module
from torch.nn.functional import one_hot, sigmoid, softmax
from lightning import LightningModule

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import mean_average_precision_at_k


class ConsistencyRegularizationForward(LightningModule):
    def __init__(self, 
                 model: Module, 
                 supervised_loss: Module,
                 consistency_loss: Module, 
                 optimizer_config: Dict, 
                 lambda_consistency: float = 0.5,
                 scheduler_config: Optional[Dict] = None, 
                 scheduler_interval: str = 'epoch', 
                 expand_labels:bool=False, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_interval = scheduler_interval
        self.supervised_loss = supervised_loss
        self.consistency_loss = consistency_loss
        self.lambda_consistency = lambda_consistency
        self.expand_labels = expand_labels

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def training_step(self, input):
        labeled_batch, unlabeled_batch = input
        (labeled_imgs_weak_aug, labeled_images_strong_aug), labels = labeled_batch
        unlabeled_imgs_weak_aug, unlabeled_imgs_strong_aug = unlabeled_batch

        preds = self(labeled_imgs_weak_aug, labels)
        if self.expand_labels:
            labels = one_hot(labels, num_classes=preds.size(1)).float()
            supervised_loss = self.supervised_loss(preds, labels)
        else:
            supervised_loss = self.supervised_loss(preds, labels)
        
        self.log('train/supervised_loss', supervised_loss, on_step=True, on_epoch=True)

        weak_aug = torch.cat([labeled_imgs_weak_aug, unlabeled_imgs_weak_aug], dim=0)
        strong_aug = torch.cat([labeled_images_strong_aug, unlabeled_imgs_strong_aug], dim=0)

        # consistency regularization
        outputs_weak = self(weak_aug)
        outputs_strong = self(strong_aug)

        if self.expand_labels:
            consistency_loss = self.consistency_loss(sigmoid(outputs_weak.detach()), sigmoid(outputs_strong))
        else:
            consistency_loss = self.consistency_loss(softmax(outputs_weak.detach(), dim=1), softmax(outputs_strong, dim=1))

        self.log('train/consistency_loss', consistency_loss, on_step=True, on_epoch=True)

        loss = supervised_loss + self.lambda_consistency * consistency_loss
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, input):
        labeled_batch, unlabeled_batch = input
        (labeled_imgs_weak_aug, labeled_images_strong_aug), labels = labeled_batch
        unlabeled_imgs_weak_aug, unlabeled_imgs_strong_aug = unlabeled_batch

        preds = self(labeled_imgs_weak_aug)
        if self.expand_labels:
            labels_ = one_hot(labels, num_classes=preds.size(1)).float()
            supervised_loss = self.supervised_loss(preds, labels_)
        else:
            supervised_loss = self.supervised_loss(preds, labels)
        
        self.log('val/supervised_loss', supervised_loss, on_step=False, on_epoch=True)

        strong_aug = torch.cat([labeled_images_strong_aug, unlabeled_imgs_strong_aug], dim=0)

        # consistency regularization
        outputs_weak = torch.cat([preds, self(unlabeled_imgs_weak_aug)], dim=0)
        outputs_strong = self(strong_aug)

        if self.expand_labels:
            consistency_loss = self.consistency_loss(sigmoid(outputs_weak.detach()), sigmoid(outputs_strong))
        else:
            consistency_loss = self.consistency_loss(softmax(outputs_weak.detach(), dim=1), softmax(outputs_strong, dim=1))

        self.log('val/consistency_loss', consistency_loss, on_step=False, on_epoch=True)

        loss = supervised_loss + self.lambda_consistency * consistency_loss
        self.log('val/loss', loss, on_step=False, on_epoch=True)

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
        if self.expand_labels:
            return sigmoid(preds)
        else:
            return softmax(preds, dim=1)
    
    def configure_optimizers(self):
        if 'backbone_lr_coef' in self.optimizer_config:
            lr_coef = self.optimizer_config['backbone_lr_coef']

            backbone_params = set(self.model.backbone.parameters())
            other_params = set(param for name, param in self.model.named_parameters() if param not in backbone_params)

            self.optimizer = self.optimizer_config['optimizer']([
                {"params": list(backbone_params), **self.optimizer_config['optimizer_params'], "lr": self.optimizer_config['optimizer_params']['lr'] * lr_coef},
                {"params": list(other_params), **self.optimizer_config['optimizer_params']}
            ])
        self.optimizer = self.optimizer_config['optimizer'](self.model.parameters(), **self.optimizer_config['optimizer_params'])
        if self.scheduler_config is not None:
            self.scheduler = self.scheduler_config['scheduler'](self.optimizer, **self.scheduler_config['scheduler_params'])
            return [self.optimizer], [{
                    'scheduler': self.scheduler,
                    'interval': self.scheduler_interval,
                }]
        return [self.optimizer]
