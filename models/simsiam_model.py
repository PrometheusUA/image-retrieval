from typing import Dict, Optional

import timm
from torch import nn


class SimSiamModel(nn.Module):
    def __init__(self, backbone: str, embedding_dim: int, predictor_dim: int, pretrained: bool = True, frozen_backbone: bool = True, timm_kwargs: Optional[Dict] = dict(), *args, **kwargs):
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
        
        self.embedding_dim = embedding_dim
        
        prev_dim = self.backbone.feature_info.channels()[-1]
        self.projector = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        # nn.ReLU(inplace=True), # first layer
                                        # nn.Linear(prev_dim, prev_dim, bias=False),
                                        # nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, embedding_dim, bias=False),
                                        nn.BatchNorm1d(embedding_dim, affine=False)) # output layer

        self.predictor = nn.Sequential(nn.Linear(embedding_dim, predictor_dim, bias=False),
                                        nn.BatchNorm1d(predictor_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(predictor_dim, embedding_dim)) # output layer

    def forward(self, x, return_embedding: bool = False):
        x = self.backbone(x)[-1]
        z = self.projector(x)
        if return_embedding:
            return z
        p = self.predictor(z)
        return p, z.detach()
    

class SimSiamMultilabelHead(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 num_classes: int, 
                 head_type: str = 'Linear',
                 head_dropout_rate: float = 0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if head_type == 'Linear':
            self.head = nn.Linear(embedding_dim, num_classes)
        elif head_type == 'MLP':
            self.head = nn.Sequential(
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(embedding_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(p=head_dropout_rate),
                nn.Linear(embedding_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, num_classes),
            )
        else:
            raise ValueError(f'Invalid head_type: {head_type}')

    def forward(self, x):
        return self.head(x)
