# -*- coding: utf-8 -*-
#
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class CrossViewAttention(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CrossViewAttention, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.num_heads = cfg.NETWORK.CROSS_ATT_NUM_HEADS
        self.reduced_channels = in_channels // cfg.NETWORK.CROSS_ATT_REDUCTION_RATIO  # Lower reduction for efficiency
        self.attention_spatial_downsample_ratio = cfg.NETWORK.ATT_SPATIAL_DOWNSAMPLE_RATIO 

        # Ensure reduced_channels is divisible by num_heads
        assert self.reduced_channels % self.num_heads == 0, f"reduced_channels ({self.reduced_channels}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.reduced_channels // self.num_heads

        # Downsample for Q, K, V
        if self.attention_spatial_downsample_ratio > 1:
            self.downsample_qkv = nn.Conv2d(
                in_channels, in_channels,
                kernel_size=self.attention_spatial_downsample_ratio,
                stride=self.attention_spatial_downsample_ratio,
                groups=in_channels
            )
        else:
            self.downsample_qkv = None

        # QKV Projection
        self.qkv_conv = nn.Conv2d(in_channels, 3 * self.reduced_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Projection Back to in_channels
        self.proj_conv = nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1)

        # Feed-Forward Network (reduced expansion for efficiency)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # BatchNorm and Dropout
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Input: [batch_size, n_views, channels, height, width]
        batch_size, n_views, channels, height, width = x.size()
        x_flat_views = x.view(batch_size * n_views, channels, height, width)

        # Downsample Q, K, V
        if self.downsample_qkv:
            x_for_qkv = self.downsample_qkv(x_flat_views) # [batch_size * n_views, channels, new_h, new_w]
            new_height, new_width = x_for_qkv.shape[2:]
        else:
            x_for_qkv = x_flat_views
            new_height, new_width = height, width

        # QKV Projection
        qkv = self.qkv_conv(x_for_qkv)                                 # [batch_size * n_views, 3 * reduced_channels, new_h, new_w]
        q, k, v = torch.split(qkv, [self.reduced_channels] * 3, dim=1) # Each: [batch_size * n_views, reduced_channels, new_h, new_w]

        # Reshape for Multi-Head Attention
        q = q.view(batch_size, n_views, self.num_heads, self.head_dim * new_height * new_width)  # [batch_size, n_views, num_heads, head_dim * new_h * new_w]
        k = k.view(batch_size, n_views, self.num_heads, self.head_dim * new_height * new_width)
        v = v.view(batch_size, n_views, self.num_heads, self.head_dim, new_height, new_width)    # [batch_size, n_views, num_heads, head_dim, new_h, new_w]

        # Compute Attention Scores
        q = q.permute(0, 2, 1, 3)                                    # [batch_size, num_heads, n_views, head_dim * new_h * new_w]
        k = k.permute(0, 2, 3, 1)                                    # [batch_size, num_heads, head_dim * new_h * new_w, n_views]
        attention_scores = torch.matmul(q, k) / (self.head_dim * n_views)**0.5  # [batch_size, num_heads, n_views, n_views]
        attention_scores = self.softmax(attention_scores)

        # Apply Attention
        v = v.permute(0, 2, 1, 3, 4, 5)                              # [batch_size, num_heads, n_views, head_dim, new_h, new_w]
        v = v.reshape(batch_size, self.num_heads, n_views, -1)       # [batch_size, num_heads, n_views, head_dim * new_h * new_w]
        attended_values = torch.matmul(attention_scores, v)          # [batch_size, num_heads, n_views, head_dim * new_h * new_w]
        attended_values = attended_values.view(batch_size, self.num_heads, n_views, self.head_dim, new_height, new_width)
        attended_values = attended_values.permute(0, 2, 1, 3, 4, 5)  # [batch_size, n_views, num_heads, head_dim, new_h, new_w]
        attended_values = attended_values.reshape(batch_size, n_views, self.reduced_channels, new_height, new_width)

        # Project Back to in_channels
        attended_values = attended_values.view(batch_size * n_views, self.reduced_channels, new_height, new_width)
        attended_values = self.proj_conv(attended_values)            # [batch_size * n_views, in_channels, new_h, new_w]

        # Upsample if Downsampled
        if self.downsample_qkv:
            attended_values = F.interpolate(
                attended_values, size=(height, width), mode='bilinear', align_corners=False
            )

        # Residual Connection
        attended_values = attended_values.view(batch_size, n_views, self.in_channels, height, width)
        attended_values = attended_values + x  # Skip connection

        # Feed-Forward Network
        attended_values_flat = attended_values.view(batch_size * n_views, self.in_channels, height, width)
        attended_values_flat = self.ffn(attended_values_flat)
        attended_values = attended_values_flat.view(batch_size, n_views, self.in_channels, height, width)

        # BatchNorm and Dropout
        attended_values_flat = attended_values.view(batch_size * n_views, self.in_channels, height, width)
        attended_values_flat = self.batch_norm(attended_values_flat)
        attended_values_flat = self.dropout(attended_values_flat)
        attended_values = attended_values_flat.view(batch_size, n_views, self.in_channels, height, width)

        return attended_values