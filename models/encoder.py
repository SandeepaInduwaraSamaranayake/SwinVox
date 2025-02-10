# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models as models
from models.swin_transformer import SwinTransformer
from models.cross_view_attention import CrossViewAttention
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # ResNet Backbone Layer Definition
        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3
        ])

        # Swin Transformer Backbone
        self.swin_transformer = SwinTransformer(in_channels=3, img_size=224, pretrained=True)

        # Cross-View Attention Layer (chanells 1024 + 768 = 1792)
        self.cross_view_attention = CrossViewAttention(in_channels = 1792)

        # Fusion Layer ( channells 1024 + 768 = 1792)
        fusion_in_channels = 1024 + self.swin_transformer.out_channels

        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Conv2d(fusion_in_channels, 1568, kernel_size=3, padding=1), # changed 512 to 1568
            torch.nn.BatchNorm2d(1568), # changed 512 to 1568
            torch.nn.ReLU()
        )

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1568, 512, kernel_size=3, padding=1), # changed 512 to 1568
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
        print(f"No of rendering images received : {len(rendering_images)}" )
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        # torch.Size([n_views, batch_size, img_c, img_h, img_w])
        rendering_images = torch.split(rendering_images, 1, dim=0)
        # A list of tensors, each with shape [1, batch_size, img_c, img_h, img_w]
        image_features = []

        print(f"No of rendering images received : {len(rendering_images)}" )

        for img in rendering_images:

            img = img.squeeze(dim=0)
            # removes the first dimension [batch_size, img_c, img_h, img_w]
            # torch.Size([64, 3, 224, 224])
            print(f'------> after squeece image shape : {img.shape}')
            
            # Get features from ResNet [batch, 1024, 14, 14]
            resnet_features = self.resnet(img)
            # torch.Size([64, 1024, 14, 14])
            print(f'-------> resnet features shape : {resnet_features.shape}')

            # Get features from Swin Transformer [batch, 768, 7, 7]
            swin_features = self.swin_transformer(img)
            # torch.Size([64, 768, 7, 7])
            print(f'-------> swin features shape : {swin_features.shape}')
            

            # Resize swin_features to match the size of resnet_features
            # resnet_features has shape (batch_size, 1024, 14, 14)
            # and swin_features has shape (batch_size, 768, 7, 7)
            # Upsampling / resizing swin_features (initially 7 x 7) tensor to match the spatial dimensions of resnet_features (14 x 14).
            swin_features_resized = F.interpolate(
              swin_features, size=resnet_features.shape[2:], mode='bilinear', align_corners=False
            )

            print(f"ResNet feature size: {resnet_features.shape}")
            # torch.Size([batch_size, 1024, 14, 14])
            
            print(f"Swin feature size: {swin_features_resized.shape}")
            # torch.Size([batch_size, 768, 14, 14])

            # Concatenate along dim=1 (channel dimension)
            # Concatenate the two tensors along the channel dimension
            # Will result [batch_size, 1024 + 768, 14, 14]
            fused_features = torch.cat((resnet_features, swin_features_resized), dim=1)

            # [batch_size, 1792, 14, 14]
            print(f"fused_features size: {fused_features.shape}")

            image_features.append(fused_features)

        
        print(f"Number of views in image_features: {len(image_features)}")
        # Stack features from all views (After stacking shape will be [n_views, batch_size, channels, height, width])
        # permute changes the order of dimensions to [batch_size, n_views, channels, height, width]
        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()

        #  torch.Size([64, 1, 1792, 14, 14])
        print(f"Image features shape : {image_features.shape}")

        # Apply cross-view attention
        # torch.Size([64, 1, 1792, 14, 14])
        attended_features = self.cross_view_attention(image_features)

        batch_size, n_views, channels, height, width = attended_features.shape
        attended_features = attended_features.view(batch_size * n_views, channels, height, width)

        print(f"attended_features shape : {attended_features.shape}")

        # Process attended features through the fusion layer
        # torch.Size([64, 512, 14, 14])
        fused_features = self.fusion_layer(attended_features)

        print(f"fused_features shape : {fused_features.shape}")

        # Ensure the output shape is compatible with the decoder
        # After the fusion layer, the shape should be [64, 1568, 14, 14]
        # You may need to add a pooling layer to reduce the spatial dimensions to 2x2
        fused_features = F.adaptive_avg_pool2d(fused_features, (2, 2))  # Reduce to 2x2

        # Continue with additional layers
        fused_features = self.layer1(fused_features)
        fused_features = self.layer2(fused_features)
        fused_features = self.layer3(fused_features)

        # Reshape back to [batch_size, n_views, channels, height, width]
        batch_size, channels, height, width = fused_features.shape
        fused_features = fused_features.view(batch_size, n_views, channels, height, width)

        # torch.Size([64, 1, 256, 3, 3])
        print(f"final fused_features shape : {fused_features.shape}")

        return fused_features
