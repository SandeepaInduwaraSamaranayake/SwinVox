import torch
import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    def __init__(self, in_channels=3, img_size=224, pretrained=True):
        super(SwinTransformer, self).__init__()

        # Load the pretrained Swin Transformer from the timm library
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)

        # Modify the input layer to accept the required number of channels (3 for RGB images)
        self.model.patch_embed.proj = nn.Conv2d(in_channels, self.model.feature_info.channels()[0], kernel_size=(4, 4), stride=(4, 4), padding=0)

        # Get the output channel count of the last layer
        self.out_channels = self.model.feature_info.channels()[-1]

    def forward(self, x):
        # Extract features from the Swin Transformer
        features = self.model(x)
        # Use only the final feature map (or others depending on your use case)
        return features[-1]  # Output shape: [batch_size, out_channels, H, W]