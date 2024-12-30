from typing import Dict, Optional

import torch
import timm
import peft
import math
from torch.nn import functional as F
from torch import nn


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        if easy_margin:
            self.th = 0
        else:
            self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        if labels is None: # eval/test mode
            return cos_theta * self.s
        
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        output = output * self.s
        return output


class ArcfaceModel(nn.Module):
    def __init__(self, 
                 backbone: str, 
                 num_classes: int, 
                 pretrained: bool = True, 
                 frozen_backbone: bool = True, 
                 timm_kwargs: Optional[Dict] = dict(), 
                 head_dropout_rate: float = 0.2, 
                 s: float = 30.0,
                 m: float = 0.50,
                 easy_margin: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            **timm_kwargs,
        )

        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.num_classes = num_classes
        
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.head = nn.Sequential(
            nn.Dropout(p=head_dropout_rate),
            nn.Linear(self.backbone.feature_info.channels()[-1], self.num_classes)
        )

        self.arcface_head = ArcFaceHead(self.backbone.feature_info.channels()[-1], 
                                        self.num_classes, s, m,
                                        easy_margin)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_embedding: bool = False):
        x = self.backbone(x)[-1]
        x = self.neck(x)
        if return_embedding:
            return x
        output = self.head(x)
        if labels is None:
            return output
        arc_output = self.arcface_head(x, labels)
        return output, arc_output


class ArcfaceOnlyModel(nn.Module):
    def __init__(self, 
                 backbone: str, 
                 num_classes: int, 
                 pretrained: bool = True, 
                 frozen_backbone: bool = True,
                 unfreeze_after: Optional[str] = None, 
                 timm_kwargs: Optional[Dict] = dict(), 
                 head_dropout_rate: float = 0.2, 
                 s: float = 30.0,
                 m: float = 0.50,
                 neck_type: str = 'Simple',
                 easy_margin: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            **timm_kwargs,
        )

        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                if unfreeze_after in name:
                    break
                param.requires_grad = False
        
        self.num_classes = num_classes
        
        prev_dim = self.backbone.feature_info.channels()[-1]

        if neck_type == 'Simple':
            self.neck = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        elif neck_type == 'ReLU_BN':
            self.neck = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
            )
        elif neck_type == 'LongLinear':
            self.neck = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError(f"{neck_type} not implemented")

        self.dropout = nn.Dropout(p=head_dropout_rate)

        self.head = ArcFaceHead(prev_dim, self.num_classes, s, m, easy_margin)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_embedding: bool = False):
        x = self.backbone(x)[-1]
        x = self.neck(x)
        if return_embedding:
            return x
        x = self.dropout(x)
        output = self.head(x, labels)
        return output


class ArcfaceOnlyModelLORA(nn.Module):
    def __init__(self, 
                 backbone: str, 
                 num_classes: int, 
                 lora_config_params: Dict,
                 pretrained: bool = True, 
                 frozen_backbone: bool = True,
                 unfreeze_after: Optional[str] = None, 
                 timm_kwargs: Optional[Dict] = dict(), 
                 head_dropout_rate: float = 0.2, 
                 s: float = 30.0,
                 m: float = 0.50,
                 neck_type: str = 'Simple',
                 easy_margin: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            **timm_kwargs,
        )

        lora_config = peft.LoraConfig(**lora_config_params)

        self.backbone = peft.get_peft_model(self.backbone, lora_config)

        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                if unfreeze_after in name:
                    break
                param.requires_grad = False
        
        self.num_classes = num_classes
        
        prev_dim = self.backbone.feature_info.channels()[-1]

        if neck_type == 'Simple':
            self.neck = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        elif neck_type == 'ReLU_BN':
            self.neck = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
            )
        elif neck_type == 'LongLinear':
            self.neck = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError(f"{neck_type} not implemented")

        self.dropout = nn.Dropout(p=head_dropout_rate)

        self.head = ArcFaceHead(prev_dim, self.num_classes, s, m, easy_margin)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, return_embedding: bool = False):
        x = self.backbone(x)[-1]
        x = self.neck(x)
        if return_embedding:
            return x
        x = self.dropout(x)
        output = self.head(x, labels)
        return output
