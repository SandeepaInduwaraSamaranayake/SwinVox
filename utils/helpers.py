# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

from mpl_toolkits.mplot3d import Axes3D


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
       type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_volume_views(volume, save_dir, prefix, sample_idx, epoch_idx):

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    # Create a figure for 3D plotting
    fig = plt.figure()
    # Standard way to create a 3D subplot in new versions of Matplotlib
    ax = fig.add_subplot(111, projection='3d')
    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])
    # Plot the voxels with black edges
    ax.voxels(volume, edgecolor="k")
    # Convert the figure canvas to an image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    # Transpose the image to be in [C, H, W] format, which is expected by TensorBoard. This ensures that the returned image is in [C, H, W] format, where C is the number of channels (e.g., 3 for RGB), and H and W are the height and width, respectively. This is the format expected by TensorBoard's add_image method.
    img = np.transpose(img, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)

    # Save Plot
    # Ensure the filename is unique by adding the sample index and epoch index
    save_path = os.path.join(save_dir, f"{prefix}_sample{sample_idx}_epoch{epoch_idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    # Close the figure to free up resources
    plt.close(fig)

    return img
