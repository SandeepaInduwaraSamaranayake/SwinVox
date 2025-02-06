# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models as models

#--------------------SWINT INTEGRATION---------------------
from models.swin_transformer import SwinTransformer
import torch.nn.functional as F
#--------------------SWINT INTEGRATION---------------------


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        # ResNet Backbone
        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

        #--------------------SWINT INTEGRATION COMMENTED THIS-------------------
        # self.resnet = torch.nn.Sequential(*[
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
        #     resnet.layer4
        # ])[:6]

        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3
        ])

        #--------------------SWINT INTEGRATION COMMENTED THIS-------------------

        #-----------------------------------------------------------------------
        # Swin Transformer Backbone
        self.swin_transformer = SwinTransformer(in_channels=3, img_size=224, pretrained=True)

        # Fusion Layer
        fusion_in_channels = 512 + self.swin_transformer.out_channels
        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1031, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        #-----------------------------------------------------------------------


        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )


    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
          #--------------------SWINT INTEGRATION COMMENTED THIS---------------

            # features = self.resnet(img.squeeze(dim=0))
            # # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            # features = self.layer1(features)
            # # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            # features = self.layer2(features)
            # # print(features.size())    # torch.Size([batch_size, 256, 14, 14])
            # features = self.layer3(features)
            # # print(features.size())    # torch.Size([batch_size, 256, 7, 7])
            # image_features.append(features)

          #--------------------SWINT INTEGRATION COMMENTED THIS---------------

          #---------------------------------------------------------------------
            img = img.squeeze(dim=0)  # Remove the batch dimension
        
            # Get features from ResNet
            resnet_features = self.resnet(img)
            
            # Get features from Swin Transformer
            swin_features = self.swin_transformer(img)

            
            # Resize swin_features to match the size of resnet_features
            # Assuming resnet_features has shape (batch_size, 512, 14, 14)
            # and swin_features has shape (batch_size, 96, 7, 7)
            swin_features_resized = F.interpolate(swin_features, size=resnet_features.shape[2:], mode='bilinear', align_corners=False)

            print(f"ResNet feature size: {resnet_features.shape}")
            print(f"Swin feature size: {swin_features_resized.shape}")

            # Now you can concatenate along dim=1 (channel dimension)
            fused_features = torch.cat((resnet_features, swin_features_resized), dim=1)
            
            # Pass the fused features through the subsequent layers
            fused_features = self.fusion_layer(fused_features)
            fused_features = self.layer1(fused_features)
            fused_features = self.layer2(fused_features)
            
            # Append the fused features to the list
            image_features.append(fused_features)
          #---------------------------------------------------------------------

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7])
        return image_features
