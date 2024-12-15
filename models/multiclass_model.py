from typing import Dict, Optional

import timm
from torch import nn


class MulticlassModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, frozen_backbone: bool = True, timm_kwargs: Optional[Dict] = dict(), head_dropout_rate: float = 0.2, *args, **kwargs):
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
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=head_dropout_rate),
            nn.Linear(self.backbone.feature_info.channels()[-1], self.num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.head(x)
        return x
