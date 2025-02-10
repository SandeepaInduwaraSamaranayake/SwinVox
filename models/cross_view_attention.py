import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossViewAttention, self).__init__()
        # Reduces the number of channels to in_channels // 8
        self.query_conv = nn.Conv2d(in_channels, in_channels // 256, kernel_size=1)
        # Reduces the number of channels to in_channels // 8
        self.key_conv = nn.Conv2d(in_channels, in_channels // 256, kernel_size=1)
        # Keeps the number of channels the same as the input
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape [batch_size, n_views, channels, height, width]
        # torch.Size([64, 1, 1792, 14, 14])
        batch_size, n_views, channels, height, width = x.size()
        print(f"batch size: {batch_size}  , n_views : {n_views} , channels : {channels} , height : {height} , width : {width}")
        
        # Reshape for attention computation
        # [64, 1792, 14, 14]
        x = x.view(batch_size * n_views, channels, height, width)
        # print(f"Reshaped attention computation : {x}")

        # Compute query, key, and value
        # reduces the number of channels from 1792 to 1792 // 256 = 7
        # [64, 1, 7 * 14 * 14] = [64, 1, 980]
        query = self.query_conv(x).view(batch_size, n_views, -1)
        # [64, 1, 980]
        key = self.key_conv(x).view(batch_size, n_views, -1)
        # [64, 1, 1792, 14, 14]
        value = self.value_conv(x).view(batch_size, n_views, channels, height, width)

        # Print shapes for debugging
        print(f"Query shape: {query.shape}")
        print(f"Key shape: {key.shape}")
        print(f"Value shape: {value.shape}")

        # Compute attention scores using einsum for efficiency
        # [64, 1, 1]
        attention_scores = F.softmax(torch.einsum('bik,bjk->bij', query, key), dim=-1)

        # Print attention scores shape
        print(f"Attention scores shape: {attention_scores.shape}")

        # Reshape value for bmm
        # [64, 1, 1792 * 14 * 14] = [64, 1, 351232]
        value_reshaped = value.view(batch_size, n_views, -1)  # Shape: [batch_size, n_views, channels * height * width]

        print(f"value_reshaped shape: {value_reshaped.shape}")

        # Apply attention to values
        # [64, 1, 351232]
        attended_values = torch.bmm(attention_scores, value_reshaped)  # Shape: [batch_size, features, channels * height * width]
        
        # [64, 1, 1792, 14, 14]
        attended_values = attended_values.view(batch_size, n_views, channels, height, width)  # Reshape back to original dimensions

        return attended_values