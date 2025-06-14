# -*- coding: utf-8 -*-
#
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewAttention(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CrossViewAttention, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.reduced_channels = in_channels // cfg.CONST.CROSS_ATTENTION_REDUCTION_RATIO
        self.attention_spatial_downsample_ratio = cfg.CONST.ATTENTION_SPATIAL_DOWNSAMPLE_RATIO

        # Downsample layer for QKV features if needed
        if self.attention_spatial_downsample_ratio > 1:
            self.downsample_qkv = nn.Conv2d(in_channels, in_channels,
                                            kernel_size=self.attention_spatial_downsample_ratio,
                                            stride=self.attention_spatial_downsample_ratio,
                                            groups=in_channels) # Depthwise conv for efficient downsampling
        else:
            self.downsample_qkv = None


        # Combine query, key, and value convolutions into a single layer
        # The input channels to qkv_conv might change if downsample_qkv is used
        # However, it's applied *after* the initial `x.view` which maintains `channels`
        self.qkv_conv = nn.Conv2d(in_channels, self.reduced_channels * 2 + in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Add layer normalization and dropout for stability
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, n_views, channels, height, width]
        batch_size, n_views, channels, height, width = x.size()

        # Reshape for efficient computation
        x_flat_views = x.view(batch_size * n_views, channels, height, width)  # [batch_size * n_views, channels, height, width]

        # Initialize variables that will be used in both branches
        current_q_features = None
        current_k_features = None
        current_v_features = None
        current_output_height = height
        current_output_width = width

        # --- Apply spatial downsampling to features before QKV projection ---
        if self.downsample_qkv is not None:
        
          # Apply downsampling to features that will form Q, K, V
          # This happens before the qkv_conv.
          # This might require some careful design choice here:

          # Option A: Downsample x_flat_views directly.
          # Option B: Apply qkv_conv on full resolution, then downsample q, k, v (this is often better for preserving info)

          # Let's consider Option B for now, as it's more common in vision transformers,
          # though it means qkv_conv runs on full resolution.

          # Let's refine the idea: downsample the *features for attention calculation only*.
          # This means we'll generate q_small, k_small for attention scores,
          # but apply attention to full-res V. This is more complex but often effective.

          # For simplicity and to match original structure, let's keep qkv_conv at original H,W
          # and suggest reducing input H,W to the entire module if possible in your main network.
          # Or, as a direct change:
          # Downsample *before* qkv_conv only if `in_channels` matches expected `qkv_conv` input.

          # For performance improvement, we must downsample *before* flattening.
          # Let's make the downsample operate *on the features used to derive Q/K*.
          x_for_qkv = x_flat_views
          if self.downsample_qkv:
            x_for_qkv = self.downsample_qkv(x_flat_views) # Downsample spatial dims for Q, K, V calculation.

          # Compute query, key, and value in one forward pass
          qkv = self.qkv_conv(x_for_qkv)  # [batch_size * n_views, reduced_channels * 2 + channels, height, width]
          # New h, w from downsampling
          new_height, new_width = qkv.shape[2:]

          q, k, v_reduced = torch.split(qkv, [self.reduced_channels, self.reduced_channels, self.in_channels], dim=1)

          # Reshape query and key for attention computation
          q = q.view(batch_size, n_views, -1)  # [batch_size, n_views, reduced_channels * height * width]
          k = k.view(batch_size, n_views, -1)  # [batch_size, n_views, reduced_channels * height * width]

          # Original V is still full resolution, if we want to apply attention to it
          # v = v.view(batch_size, n_views, self.in_channels, height, width) # [batch_size, n_views, channels, height, width] Use original V if not downsampled with Q,K

          # If v_reduced is derived from downsampled features, then its shape will be [B*Nv, C, new_H, new_W]
          # To apply attention to full resolution V, the split logic for QKV would need to be changed
          # where V comes from the original x.

          # Compute attention scores using batch matrix multiplication
          attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, n_views, n_views]
          attention_scores = self.softmax(attention_scores / (self.reduced_channels * new_height * new_width)**0.5)  # Scaled dot-product attention

          # Apply attention to values
          v = v_reduced.view(batch_size, n_views, -1)  # [batch_size, n_views, channels * height * width]
          attended_values = torch.matmul(attention_scores, v)  # [batch_size, n_views, channels * height * width]

          # Reshape back to original dimensions
          attended_values = attended_values.view(batch_size, n_views, self.in_channels, new_height, new_width)

          # If the output needs to be original H, W, then you need an Upsample layer here:
          if self.downsample_qkv is not None:
              # Considered simple interpolation or learned upsampling.
              # For maintaining accuracy, learned upsampling (e.g., ConvTranspose2d) or
              # integrating this into a decoder that handles resolution differences is better.
              attended_values = F.interpolate(attended_values.view(batch_size * n_views, self.in_channels, new_height, new_width),
                                              size=(height, width), mode='bilinear', align_corners=False)
              attended_values = attended_values.view(batch_size, n_views, self.in_channels, height, width)

        else: # Original logic if no downsampling
          qkv = self.qkv_conv(x_flat_views)
          q, k, v = torch.split(qkv, [self.reduced_channels, self.reduced_channels, self.in_channels], dim=1)

          q = q.view(batch_size, n_views, -1)
          k = k.view(batch_size, n_views, -1)
          v = v.view(batch_size, n_views, self.in_channels, height, width)

          attention_scores = torch.matmul(q, k.transpose(-1, -2))
          attention_scores = self.softmax(attention_scores / (self.reduced_channels * height * width)**0.5)

          v = v.view(batch_size, n_views, -1)
          attended_values = torch.matmul(attention_scores, v)

          attended_values = attended_values.view(batch_size, n_views, self.in_channels, height, width)

        # Reshape for layer normalization
        attended_values = attended_values.permute(0, 1, 3, 4, 2)  # [batch_size, n_views, height, width, channels]
        attended_values = attended_values.reshape(-1, self.in_channels)  # [batch_size * n_views * height * width, channels]

        # Apply layer normalization
        attended_values = self.layer_norm(attended_values)  # [batch_size * n_views * height * width, channels]

        # Reshape back to original shape
        attended_values = attended_values.view(batch_size, n_views, height, width, self.in_channels)
        attended_values = attended_values.permute(0, 1, 4, 2, 3)  # [batch_size, n_views, channels, height, width]

        # Add dropout
        attended_values = self.dropout(attended_values)

        return attended_values