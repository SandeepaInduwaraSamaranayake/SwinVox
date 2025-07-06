# -*- coding: utf-8 -*-
#
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import timm
import logging

class SwinTransformer(nn.Module):
    def __init__(self, cfg, in_channels=3, img_size=224, pretrained=True):
        super(SwinTransformer, self).__init__()
        self.cfg = cfg
        self.img_size = img_size

        # Load pretrained Swin Transformer
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=self.cfg.NETWORK.SWIN_T_STAGES  # Use config stages - multi-scale features
        )

        # Modify patch_embed.proj for custom in_channels
        embed_dim = self.model.patch_embed.proj.out_channels
        original_proj = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(
            in_channels, 
            embed_dim,
            kernel_size=original_proj.kernel_size, 
            stride=original_proj.stride, 
            padding=original_proj.padding
        )

        # Initialize weights for custom in_channels( 4 like RGBA instead of 3 for RGB)
        if in_channels != 3 and pretrained:
            pretrained_weights = original_proj.weight.clone()  # [embed_dim, 3, 4, 4]
            new_weights = torch.zeros([embed_dim, in_channels, 4, 4], device=pretrained_weights.device)
            for i in range(min(in_channels, 3)):
                new_weights[:, i, :, :] = pretrained_weights[:, i, :, :]
            if in_channels > 3:
                new_weights[:, 3:, :, :] = pretrained_weights[:, :in_channels-3, :, :].mean(dim=1, keepdim=True)
            self.model.patch_embed.proj.weight = nn.Parameter(new_weights)
            if original_proj.bias is not None:
                self.model.patch_embed.proj.bias = nn.Parameter(original_proj.bias.clone())
        else:
            if not pretrained:
                nn.init.xavier_uniform_(self.model.patch_embed.proj.weight)
                if self.model.patch_embed.proj.bias is not None:
                    nn.init.zeros_(self.model.patch_embed.proj.bias)

        # Output channels and spatial dims
        self.out_channels = [self.model.feature_info.channels()[i] for i in range(len(self.cfg.NETWORK.SWIN_T_STAGES))]
        self.out_spatial = [self.img_size // (4 * 2**i) for i in self.cfg.NETWORK.SWIN_T_STAGES]  # e.g., stage 3: 14, stage 4: 7

        # Layer normalization for each stage
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm([self.out_channels[i], self.out_spatial[i], self.out_spatial[i]])
            for i in range(len(self.cfg.NETWORK.SWIN_T_STAGES))
        ])
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        # Resize input if needed
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Extract features
        features = self.model(x)  # List of selected stages: e.g., [[batch_size, H, W, C]]

        # Process each stage
        final_features = []
        for i, feat in enumerate(features):
            # Permute to [batch_size, C, H, W]
            feat = feat.permute(0, 3, 1, 2)  # [batch_size, C, H, W]
            # Apply normalization and dropout
            feat = self.layer_norm[i](feat)
            feat = self.dropout(feat)
            final_features.append(feat)

        if not self.cfg.NETWORK.USE_SWIN_T_MULTI_STAGE:
          return final_features[-1]     # [batch_size, C_i, H_i, W_i]
        return final_features           # List of [batch_size, C_i, H_i, W_i]