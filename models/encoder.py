# -*- coding: utf-8 -*-
#
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.swin_transformer import SwinTransformer
from models.cross_view_attention import CrossViewAttention


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # ResNet Backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:7])  # Up to layer3

        # Swin Transformer Backbone
        self.swin_transformer = SwinTransformer(
            in_channels=3, 
            img_size=224, 
            pretrained=True
        )

        # Channel Reduction for ResNet and Swin Features
        self.resnet_reduce = nn.Conv2d(1024, 512, kernel_size=1)  # Reduce ResNet channels
        self.swin_reduce = nn.Conv2d(768, 512, kernel_size=1)     # Reduce Swin channels

        # Cross-View Attention Layer
        if self.cfg.NETWORK.USE_CROSS_VIEW_ATTENTION:
          self.cross_view_attention = CrossViewAttention(cfg, in_channels=1024)  # 512 + 512
        else:
            self.cross_view_attention = None

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Additional Layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, rendering_images):
        # Input shape: [batch_size, n_views, img_c, img_h, img_w]
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()  # [n_views, batch_size, img_c, img_h, img_w]
        rendering_images = torch.split(rendering_images, 1, dim=0)  # List of [1, batch_size, img_c, img_h, img_w]
        image_features = []

        for i, img in enumerate(rendering_images):
            img = img.squeeze(dim=0)  # [batch_size, img_c, img_h, img_w]

            # ResNet Features
            resnet_features = self.resnet(img)  # [batch_size, 1024, 14, 14]
            resnet_features = self.resnet_reduce(resnet_features)  # [batch_size, 512, 14, 14]

            # Swin Transformer Features
            swin_features = self.swin_transformer(img)  # [batch_size, 768, 7, 7]

            swin_features = self.swin_reduce(swin_features)  # [batch_size, 512, 7, 7]

            swin_features = F.interpolate(swin_features, size=resnet_features.shape[2:], mode='bilinear', align_corners=False)  # [batch_size, 512, 14, 14]

            # Concatenate Features
            fused_features = torch.cat((resnet_features, swin_features), dim=1)  # [batch_size, 1024, 14, 14]
            image_features.append(fused_features)

        # Stack Features from All Views
        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()  # [batch_size, n_views, 1024, 14, 14]

        # Apply Cross-View Attention
        if self.cfg.NETWORK.USE_CROSS_VIEW_ATTENTION:
          attended_features = self.cross_view_attention(image_features)  # [batch_size, n_views, 1024, 14, 14]
        else:
          attended_features = image_features

        batch_size, n_views, channels, height, width = attended_features.shape
        attended_features = attended_features.view(batch_size * n_views, channels, height, width)  # [batch_size * n_views, 1024, 14, 14]

        # Process Attended Features
        fused_features = self.fusion_layer(attended_features)  # [batch_size * n_views, 512, 14, 14]
        fused_features = self.layer1(fused_features)  # [batch_size * n_views, 512, 14, 14]
        fused_features = self.layer2(fused_features)  # [batch_size * n_views, 256, 7, 7]
        fused_features = self.layer3(fused_features)  # [batch_size * n_views, 256, 7, 7]

        # Reshape Back to [batch_size, n_views, channels, height, width]
        fused_features = fused_features.view(batch_size, n_views, 256, 7, 7)  # [batch_size, n_views, 256, 7, 7]

        return fused_features