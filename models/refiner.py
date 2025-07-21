# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import torch


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

        # Layer Definition
        # 3D Downsampling Path (Layer 1 --> Layer 3)
        # Conv3d : These extract features from the 3D volume.
        # MaxPool3d(kernel_size=2) : progressively downsamples the spatial resolution of the 3D volume by a 
        #                            factor of 2 at each step, while increasing the number of channels.
        # LeakyReLU   : Introduces non-linearity, allowing the network to learn complex relationships.
        # BatchNorm3d : Stabilize training by normalizing the feature maps.
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        # Fully Connected Layers (Layers 4 --> Layer 5)
        # These are standard Linear (fully connected) layers with ReLU activations
        # Layer 4 : Reduces this high-dimensional vector to 2048 dimensions.
        # Layer 5 : Expands it back to 8192 dimensions. These layers act as a bottleneck for global feature extraction, 
        #           allowing the network to capture global context and refine features that might be ambiguous from 
        #           purely local convolutions.
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU()
        ) 
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU()
        )
        # 3D Upsampling Path (Layers 6 --> Layer 8)
        # Progressively upsample the volume back to the original resolution.
        # ConvTranspose3d(stride=2) : Each layer doubles the spatial dimensions.
        # BatchNorm3d : For network stability.
        # ReLU : Introduces non-linearity.
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
        )

    def forward(self, coarse_volumes):
        # Add a channel dimension making it [batch_size, 1, 32, 32, 32], which is the expected input format for the 
        # 3D convolutional layers.
        volumes_32_l = coarse_volumes.unsqueeze(dim=1)
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        # DOWNSAMPLING PATH
        # Reducing spatial size to 16x16x16 and increasing channels to 32
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])

        # FULLY CONNECTED LAYERS (To capture Global Information)
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])

        # UPSAMPLING PATH with skip connections
        # This is a skip connection, which allows the network to combine local and global information, improving the refinement process.
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        # The volumes_4_r (now enhanced with global context) is upsampled by layer6 to 8x8x8 resolution
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        # The final upsampling step by layer8 brings the resolution back to 32x32x32 and 1 channel
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        return volumes_32_r.squeeze(dim=1)
