# -*- coding: utf-8 -*-
#
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.swin_transformer import SwinTransformer
from models.cross_view_attention import CrossViewAttention
import logging


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # ResNet Backbone
        # For capturing local, spatial features due to its convolutional nature and 
        # residual connections. 
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:7])  # Up to layer3

        # Swin Transformer Backbone
        # Efficient Vision Transformer, which is good at capturing global dependencies through their unique 
        # shifted window attention mechanism.
        self.swin_transformer = SwinTransformer(
            self.cfg,
            in_channels=3, 
            img_size=224, 
            pretrained=True
        )

        # Channel Reduction - ResNet 1024 --> 256
        self.resnet_reduce = nn.Conv2d(1024, 256, kernel_size=1)
        # Channel Reduction - SwinT X --> 256
        # If USE_SWIN_T_MULTI_STAGE is enabled, it processes features from different stages of the 
        # Swin Transformer, reducing their channels to 256. 
        # Otherwise, it's a single reduction for the final Swin Transformer output
        if self.cfg.NETWORK.USE_SWIN_T_MULTI_STAGE:
          self.swin_stage_reduces = nn.ModuleList([
              nn.Conv2d(ch, 256, kernel_size=1)
              for ch in self.swin_transformer.out_channels
          ])
          # downsample the features from different stages to a consistent 7x7 spatial resolution
          # to ensure all features are aligned before concatenation.
          self.swin_downsamples = nn.ModuleList([
              nn.Sequential(
                  nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [56, 56] -> [28, 28]
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [28, 28] -> [14, 14]
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [14, 14] -> [7, 7]
                  nn.BatchNorm2d(256),
                  nn.ReLU()
              ) if i == 0 else
              nn.Sequential(
                  nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [28, 28] -> [14, 14]
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [14, 14] -> [7, 7]
                  nn.BatchNorm2d(256),
                  nn.ReLU()
              ) if i == 1 else
              nn.Sequential(
                  nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [14, 14] -> [7, 7]
                  nn.BatchNorm2d(256),
                  nn.ReLU()
              ) if i == 2 else
              nn.Identity()
              for i in self.cfg.NETWORK.SWIN_T_STAGES
          ])
        else:
            self.swin_reduce = nn.Conv2d(768, 256, kernel_size=1)


        # Cross-View Attention Layer
        # Helps the model understand the consistency and relationships between different 2D perspectives
        if self.cfg.NETWORK.USE_CROSS_VIEW_ATTENTION:
          self.cross_view_attention = CrossViewAttention(cfg, in_channels=512)  # 256 + 256
        else:
            self.cross_view_attention = None

        # Fusion Layer
        # Takes the combined features (512 channels) and further refines them down to 256 channels
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Additional Layers
        # Further refine the extracted 2D features
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, rendering_images):
        # Input: [batch_size, n_views, img_c, img_h, img_w]
        batch_size, n_views, img_c, img_h, img_w = rendering_images.shape
        img = rendering_images.view(batch_size * n_views, img_c, img_h, img_w)

        # ResNet Features
        resnet_features = self.resnet(img)                                          # [batch_size * n_views, 1024, 14, 14]
        # Channels are reduced to 256
        resnet_features = self.resnet_reduce(resnet_features)                       # [batch_size * n_views, 256, 14, 14]
        # Average pooling further reduces the spatial size to 7x7
        resnet_features = F.avg_pool2d(resnet_features, kernel_size=2, stride=2)    # [batch_size * n_views, 256, 7, 7]


        # Swin Transformer Features
        # Depending on the configuration, swin_features can be a list 
        # of features from multiple stages or a single tensor
        swin_features = self.swin_transformer(img)                                  # List or Single tensor [batch_size * n_views, 768, 7, 7]
        if self.cfg.NETWORK.USE_SWIN_T_MULTI_STAGE:
            # If USE_SWIN_T_MULTI_STAGE is active, a loop processes each 
            # feature map from different Swin Transformer stages
            swin_processed = []
            for i, (feat, reduce, upsample) in enumerate(zip(swin_features, self.swin_stage_reduces, self.swin_downsamples)):
                feat = reduce(feat)                                                 # [batch_size * n_views, 256, H_i, W_i]
                feat = upsample(feat)                                               # [batch_size * n_views, 256, 7, 7]
                swin_processed.append(feat)
            swin_features = torch.sum(torch.stack(swin_processed), dim=0)           # [batch_size * n_views, 256, 7, 7]
        else:
            swin_features = self.swin_reduce(swin_features)                         # [batch_size * n_views, 256, 7, 7]

        # Concatenate Features along the channel dimension
        fused_features = torch.cat((resnet_features, swin_features), dim=1)         # [batch_size * n_views, 512, 7, 7]
        fused_features = fused_features.view(batch_size, n_views, 512, 7, 7)

        # Cross-View Attention (If enabled)
        if self.cfg.NETWORK.USE_CROSS_VIEW_ATTENTION:
            attended_features = self.cross_view_attention(fused_features)           # [batch_size, n_views, 512, 7, 7]
        else:
            attended_features = fused_features

        # Process Features
        attended_features = attended_features.view(batch_size * n_views, 512, 7, 7)
        # Pass through the fusion_layer to reduce channels to 256
        fused_features = self.fusion_layer(attended_features)                       # [batch_size * n_views, 256, 7, 7]
        # Further refinement using sequential convolutional blocks, 
        # maintaining 256 channels and 7x7 spatial resolution
        fused_features = self.layer1(fused_features)                                # [batch_size * n_views, 256, 7, 7]
        fused_features = self.layer2(fused_features)                                # [batch_size * n_views, 256, 7, 7]
        fused_features = self.layer3(fused_features)                                # [batch_size * n_views, 256, 7, 7]

        # Reshape Output back to [batch_size, n_views, 256, 7, 7]
        fused_features = fused_features.view(batch_size, n_views, 256, 7, 7)        # [batch_size, n_views, 256, 7, 7]
        return fused_features                                                       # torch.Size([64, 1, 256, 7, 7])
