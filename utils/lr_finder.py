# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import os
import logging
import random
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from time import time

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter

# Import for Automatic Mixed Precision
from torch.amp import autocast, GradScaler


def find_lr(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader (only need train_data_loader for LR finding)
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True, # Shuffle is good even for LR finding to get diverse batches
        drop_last=True)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger  = Merger(cfg)
    logging.info(f'Parameters in Encoder: {utils.helpers.count_parameters(encoder)}.')
    logging.info(f'Parameters in Decoder: {utils.helpers.count_parameters(decoder)}.')
    logging.info(f'Parameters in Refiner: {utils.helpers.count_parameters(refiner)}.')
    logging.info(f'Parameters in Merger: {utils.helpers.count_parameters(merger)}.')

    # Initialize weights of networks - IMPORTANT: We will reset these later
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    refiner.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)

    # Save initial state for reset
    initial_encoder_state = encoder.state_dict()
    initial_decoder_state = decoder.state_dict()
    initial_refiner_state = refiner.state_dict() if cfg.NETWORK.USE_REFINER else None
    initial_merger_state = merger.state_dict() if cfg.NETWORK.USE_MERGER else None

    # Set up solver - will be modified by LR finder
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-7, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=1e-7, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(), lr=1e-7, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=1e-7, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-7, momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(), lr=1e-7, momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(), lr=1e-7, momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(merger.parameters(), lr=1e-7, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(init_scale=2**16)

    # Move models to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        encoder = torch.nn.DataParallel(encoder).to(device)
        decoder = torch.nn.DataParallel(decoder).to(device)
        refiner = torch.nn.DataParallel(refiner).to(device)
        merger  = torch.nn.DataParallel(merger).to(device)

    # Set up loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # Learning Rate Finder specific parameters
    start_lr = 1e-7
    end_lr = 1.0 # Or higher if you suspect very high LRs might be optimal
    num_batches_to_test = 200 # Number of batches to run LR finder for. Adjust based on dataset size.
    lr_multiplier = (end_lr / start_lr) ** (1 / (num_batches_to_test - 1))

    # Lists to store results
    lrs = []
    losses = []
    smoothed_losses = []
    best_loss = float('inf')
    avg_beta = 0.98 # For exponential moving average of loss

    logging.info(f'Starting Learning Rate Finder. Range: {start_lr} to {end_lr} over {num_batches_to_test} batches.')

    # Set initial learning rates for all optimizers
    for solver in [encoder_solver, decoder_solver, refiner_solver, merger_solver]:
        for param_group in solver.param_groups:
            param_group['lr'] = start_lr

    # Switch models to training mode
    encoder.train()
    decoder.train()
    merger.train()
    refiner.train()

    # LR Finder loop
    for batch_idx, (taxonomy_names, sample_names, rendering_images,
                    ground_truth_volumes) in enumerate(train_data_loader):
        if batch_idx >= num_batches_to_test:
            break

        current_lr = encoder_solver.param_groups[0]['lr'] # Assuming all LRs are updated proportionally
        lrs.append(current_lr)

        rendering_images = utils.helpers.var_or_cuda(rendering_images)
        ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
        rendering_images = torch.clamp(rendering_images, min=-1, max=1)
        ground_truth_volumes = torch.clamp(ground_truth_volumes, min=0, max=1)

        with autocast(device_type=device.type, enabled=True):
            image_features = encoder(rendering_images)
            raw_features, generated_volumes = decoder(image_features)

            if cfg.NETWORK.USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)

            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes)

            if cfg.NETWORK.USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                refiner_loss = bce_loss(generated_volumes, ground_truth_volumes)
            else:
                refiner_loss = encoder_loss
            
            total_loss = encoder_loss + (refiner_loss if cfg.NETWORK.USE_REFINER else 0)

        # Gradient descent
        encoder.zero_grad()
        decoder.zero_grad()
        refiner.zero_grad()
        merger.zero_grad()

        scaler.scale(total_loss).backward()

        scaler.unscale_(encoder_solver)
        scaler.unscale_(decoder_solver)
        scaler.unscale_(refiner_solver)
        scaler.unscale_(merger_solver)

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(merger.parameters(), max_norm=1.0)

        scaler.step(encoder_solver)
        scaler.step(decoder_solver)
        if cfg.NETWORK.USE_REFINER:
            scaler.step(refiner_solver)
        if cfg.NETWORK.USE_MERGER:
            scaler.step(merger_solver)
        scaler.update()

        # Record loss and update smoothed loss
        loss_item = total_loss.item()
        losses.append(loss_item)
        
        if batch_idx == 0:
            smoothed_losses.append(loss_item)
        else:
            smoothed_losses.append(smoothed_losses[-1] * avg_beta + loss_item * (1 - avg_beta))

        # Check for divergence
        if smoothed_losses[-1] > 4 * best_loss: # If loss explodes, stop early
            logging.warning(f'Loss diverged at LR {current_lr:.2e}. Stopping LR finder.')
            break
        if loss_item < best_loss:
            best_loss = loss_item

        # Increase learning rate for the next batch
        for solver in [encoder_solver, decoder_solver, refiner_solver, merger_solver]:
            for param_group in solver.param_groups:
                param_group['lr'] *= lr_multiplier

        logging.info(f'LR Finder Batch {batch_idx + 1}/{num_batches_to_test}, LR: {current_lr:.2e}, Loss: {loss_item:.4f}')

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, smoothed_losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Smoothed Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, which="both", ls="-")
    
    # Suggest an LR (heuristic: steepest gradient)
    # This is a simple heuristic, often you pick visually
    # Find where the gradient of the smoothed loss is steepest
    grad = np.gradient(np.array(smoothed_losses), np.array(lrs))
    # Filter out initial noisy points and potentially diverging points
    valid_indices = np.where(np.array(smoothed_losses) < 2 * best_loss)[0] # Only consider points before divergence
    if len(valid_indices) > 5: # Need enough points to compute gradient meaningfully
        valid_lrs = np.array(lrs)[valid_indices]
        valid_grads = np.array(grad)[valid_indices]
        
        # Take the steepest negative gradient
        steepest_idx = np.argmin(valid_grads) # min because we want most negative slope
        suggested_lr = valid_lrs[steepest_idx]
        
        plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
        logging.info(f'Suggested Learning Rate: {suggested_lr:.2e}')
    else:
        suggested_lr = None
        logging.warning("Could not suggest an optimal LR automatically due to insufficient valid points.")

    plt.legend()
    plt.savefig(os.path.join(cfg.DIR.OUT_PATH, f'lr_finder_plot_{dt.now().strftime("%Y%m%d-%H%M%S")}.png'))
    plt.show()

    # Reset model and optimizer states to initial
    encoder.load_state_dict(initial_encoder_state)
    decoder.load_state_dict(initial_decoder_state)
    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(initial_refiner_state)
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(initial_merger_state)
    
    # Important: Reset optimizers as well, as their internal states (e.g., Adam's momentum buffers)
    # would be corrupted by the LR finder run.
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    
    logging.info('Model and optimizer states reset to initial values.')

    return suggested_lr