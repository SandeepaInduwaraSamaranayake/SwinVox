# -*- coding: utf-8 -*-
#
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import timm


class SwinTransformer(nn.Module):
    def __init__(self, in_channels=3, img_size=224, pretrained=True):
        super(SwinTransformer, self).__init__()

        # Load the pretrained Swin Transformer from the timm library
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=pretrained, 
            features_only=True,  # Extract feature maps from multiple stages
        )

        # Modify the input layer to accept the required number of channels
        self.model.patch_embed.proj = nn.Conv2d(
            in_channels, 
            self.model.feature_info.channels()[0],  # Use the first layer's channel count
            kernel_size=(4, 4), 
            stride=(4, 4), 
            padding=0
        )

        # Get the output channel count of the last layer
        self.out_channels = self.model.feature_info.channels()[-1]

        # Optional: Add layer normalization and dropout for stability
        self.layer_norm = nn.LayerNorm(self.out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Print input shape for debugging
        # print(f"Input shape to Swin Transformer: {x.shape}")  # [batch_size, in_channels, img_size, img_size]

        # Extract features from the Swin Transformer
        features = self.model(x)  # List of feature maps from different stages

        # Use only the final feature map (highest-level features)
        final_features = features[-1]  # Shape: [batch_size, H, W, C]

        # Permute to [batch_size, C, H, W] if needed
        final_features = final_features.permute(0, 3, 1, 2)  # Shape: [batch_size, C, H, W]

        # Optional: Apply layer normalization and dropout
        final_features = self.layer_norm(final_features.permute(0, 2, 3, 1))  # Normalize over the channel dimension
        final_features = final_features.permute(0, 3, 1, 2)  # Reshape back to [batch_size, C, H, W]
        final_features = self.dropout(final_features)

        return final_features  # Output shape: [batch_size, out_channels, H, W]