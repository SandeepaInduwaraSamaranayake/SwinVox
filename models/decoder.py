# # -*- coding: utf-8 -*-
# #
# # Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint

# class ResidualBlock3D(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(channels, channels, 3, padding=1, bias=False),
#             nn.InstanceNorm3d(channels),
#             nn.LeakyReLU(0.02),
#             nn.Conv3d(channels, channels, 3, padding=1, bias=False),
#             nn.InstanceNorm3d(channels)
#         )
        
#     def forward(self, x):
#         return x + self.conv(x)

# class ChannelAttention3D(nn.Module):
#     def __init__(self, channels, reduction=8):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         b, c, *_ = x.shape
#         weights = self.fc(x.view(b, c, -1).mean(-1)).view(b, c, 1, 1, 1)
#         return x * weights

# class Decoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
        
#         # Layer Definition
#         self.layer1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='trilinear'),
#             nn.Conv3d(1568, 512, 3, padding=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
#             ResidualBlock3D(512),
#             ChannelAttention3D(512, cfg.NETWORK.ATTENTION_REDUCTION),
#             nn.InstanceNorm3d(512),
#             nn.LeakyReLU(0.02),
#             nn.Dropout3d(cfg.NETWORK.DECODER_DROPOUT)
#         )
        
#         self.layer2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='trilinear'),
#             nn.Conv3d(512, 128, 3, padding=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
#             ResidualBlock3D(128),
#             ChannelAttention3D(128, cfg.NETWORK.ATTENTION_REDUCTION),
#             nn.InstanceNorm3d(128),
#             nn.LeakyReLU(0.02),
#             nn.Dropout3d(cfg.NETWORK.DECODER_DROPOUT)
#         )
        
#         self.layer3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='trilinear'),
#             nn.Conv3d(128, 32, 3, padding=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
#             ResidualBlock3D(32),
#             ChannelAttention3D(32, cfg.NETWORK.ATTENTION_REDUCTION),
#             nn.InstanceNorm3d(32),
#             nn.LeakyReLU(0.02),
#             nn.Dropout3d(cfg.NETWORK.DECODER_DROPOUT)
#         )
        
#         self.layer4 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='trilinear'),
#             nn.Conv3d(32, 8, 3, padding=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
#             ResidualBlock3D(8),
#             ChannelAttention3D(8, cfg.NETWORK.ATTENTION_REDUCTION),
#             nn.InstanceNorm3d(8),
#             nn.LeakyReLU(0.02)
#         )
        
#         self.layer5 = nn.Sequential(
#             nn.utils.spectral_norm(
#                 nn.Conv3d(8, 1, 1, bias=cfg.NETWORK.TCONV_USE_BIAS)),
#             nn.Sigmoid()
#         )

#     def forward(self, image_features):
#         image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
#         image_features = torch.split(image_features, 1, dim=0)
        
#         raw_features = []
#         gen_volumes  = []

#         for features in image_features:
#             gen_volume = features.view(-1, 1568, 2, 2, 2)
            
#             # Through layers by checkpointing intermediate volumes
#             gen_volume = checkpoint.checkpoint(self.layer1, gen_volume, use_reentrant=False)
#             gen_volume = checkpoint.checkpoint(self.layer2, gen_volume, use_reentrant=False)
#             gen_volume = checkpoint.checkpoint(self.layer3, gen_volume, use_reentrant=False)
#             gen_volume = checkpoint.checkpoint(self.layer4, gen_volume, use_reentrant=False)
            
#             # Store raw features before final projection
#             raw_feature = gen_volume
            
#             # Final projection
#             gen_volume = self.layer5(gen_volume)
            
#             # Concatenate features for refinement
#             raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
            
#             gen_volumes.append(torch.squeeze(gen_volume, dim=1))
#             raw_features.append(raw_feature)

#         gen_volumes  = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
#         raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        
#         return raw_features, gen_volumes







# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1568, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        raw_features = []
        gen_volumes = []

        for features in image_features:
            gen_volume = features.view(-1, 1568, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 1568, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
            # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])
            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes
