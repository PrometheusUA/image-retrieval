from typing import Dict, Optional

import timm
from torch import nn


class MulticlassModel(nn.Module):
    def __init__(self, backbone: str, 
                 num_classes: int, 
                 pretrained: bool = True, 
                 frozen_backbone: bool = True, 
                 timm_kwargs: Optional[Dict] = dict(), 
                 head_dropout_rate: float = 0.2, 
                 head_type: str = 'Simple',
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
        
        prev_dim = self.backbone.feature_info.channels()[-1]
        if head_type == 'Simple':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(prev_dim, self.num_classes)
            )
        elif head_type == 'LongLinear':
            self.head = nn.Sequential(
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
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(prev_dim, self.num_classes)
            )
        else:
            raise NotImplementedError(f"{head_type} not implemented")


    def forward(self, x, return_embedding: bool = False):
        x = self.backbone(x)[-1]
        if return_embedding:
            return self.head[:2](x)
        x = self.head(x)
        return x
