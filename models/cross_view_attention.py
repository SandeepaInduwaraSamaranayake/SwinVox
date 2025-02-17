import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewAttention(nn.Module):
    def __init__(self, cfg, in_channels, reduction_ratio=128):
        super(CrossViewAttention, self).__init__()
        self.cfg = cfg
        self.reduction_ratio = cfg.CONST.CROSS_ATTENTION_REDUCTION_RATIO
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction_ratio

        # Combine query, key, and value convolutions into a single layer
        self.qkv_conv = nn.Conv2d(in_channels, self.reduced_channels * 2 + in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # Optional: Add layer normalization and dropout for stability
        self.layer_norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, n_views, channels, height, width]
        batch_size, n_views, channels, height, width = x.size()

        # Reshape for efficient computation
        x = x.view(batch_size * n_views, channels, height, width)  # [batch_size * n_views, channels, height, width]

        # Compute query, key, and value in one forward pass
        qkv = self.qkv_conv(x)  # [batch_size * n_views, reduced_channels * 2 + channels, height, width]
        q, k, v = torch.split(qkv, [self.reduced_channels, self.reduced_channels, self.in_channels], dim=1)

        # Reshape query and key for attention computation
        q = q.view(batch_size, n_views, -1)  # [batch_size, n_views, reduced_channels * height * width]
        k = k.view(batch_size, n_views, -1)  # [batch_size, n_views, reduced_channels * height * width]
        v = v.view(batch_size, n_views, self.in_channels, height, width)  # [batch_size, n_views, channels, height, width]

        # Compute attention scores using batch matrix multiplication
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, n_views, n_views]
        attention_scores = self.softmax(attention_scores / (self.reduced_channels ** 0.5))  # Scaled dot-product attention

        # Apply attention to values
        v = v.view(batch_size, n_views, -1)  # [batch_size, n_views, channels * height * width]
        attended_values = torch.matmul(attention_scores, v)  # [batch_size, n_views, channels * height * width]

        # Reshape back to original dimensions
        attended_values = attended_values.view(batch_size, n_views, self.in_channels, height, width)

        # Reshape for layer normalization
        attended_values = attended_values.permute(0, 1, 3, 4, 2)  # [batch_size, n_views, height, width, channels]
        attended_values = attended_values.reshape(-1, self.in_channels)  # [batch_size * n_views * height * width, channels]

        # Apply layer normalization
        attended_values = self.layer_norm(attended_values)  # [batch_size * n_views * height * width, channels]

        # Reshape back to original shape
        attended_values = attended_values.view(batch_size, n_views, height, width, self.in_channels)
        attended_values = attended_values.permute(0, 1, 4, 2, 3)  # [batch_size, n_views, channels, height, width]

        # Optional: Add dropout
        attended_values = self.dropout(attended_values)

        return attended_values








