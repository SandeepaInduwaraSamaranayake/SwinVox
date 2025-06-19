# -*- coding: utf-8 -*-
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import os
import logging
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler
import utils.data_loaders
import utils.data_transforms
import utils.helpers
from datetime import datetime as dt
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

def find_lr(cfg):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print header
    logging.info('=' * 40)
    logging.info(' Starting Learning Rate Finder ')
    logging.info('=' * 40)
    
    # Ensure output directory exists for LR finder plot
    os.makedirs(cfg.DIR.OUT_PATH, exist_ok=True) 

    logging.info(f"[LR Finder] Using device: {device}")

    # Data transforms
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

    # Data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    # Models
    encoder = Encoder(cfg).to(device)
    decoder = Decoder(cfg).to(device)
    merger = Merger(cfg).to(device)
    refiner = Refiner(cfg).to(device)
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)
    refiner.apply(utils.helpers.init_weights)
    if device.type == 'cuda':
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)
        merger = torch.nn.DataParallel(merger)
        refiner = torch.nn.DataParallel(refiner)
    logging.info(f'Parameters in Encoder: {utils.helpers.count_parameters(encoder)}')
    logging.info(f'Parameters in Decoder: {utils.helpers.count_parameters(decoder)}')
    logging.info(f'Parameters in Merger: {utils.helpers.count_parameters(merger)}')
    logging.info(f'Parameters in Refiner: {utils.helpers.count_parameters(refiner)}')

    # Save initial states
    initial_encoder_state = encoder.state_dict()
    initial_decoder_state = decoder.state_dict()
    initial_merger_state = merger.state_dict() if cfg.NETWORK.USE_MERGER else None
    initial_refiner_state = refiner.state_dict() if cfg.NETWORK.USE_REFINER else None

    # Optimizers
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.LR_FINDER.START_LR, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(
            decoder.parameters(), lr=cfg.LR_FINDER.START_LR, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(
            merger.parameters(), lr=cfg.LR_FINDER.START_LR / 10.0, betas=cfg.TRAIN.BETAS) if cfg.NETWORK.USE_MERGER else None
        refiner_solver = torch.optim.Adam(
            refiner.parameters(), lr=cfg.LR_FINDER.START_LR / 10.0, betas=cfg.TRAIN.BETAS) if cfg.NETWORK.USE_REFINER else None
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(
            filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.LR_FINDER.START_LR, momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(
            decoder.parameters(), lr=cfg.LR_FINDER.START_LR, momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(
            merger.parameters(), lr=cfg.LR_FINDER.START_LR / 10.0, momentum=cfg.TRAIN.MOMENTUM) if cfg.NETWORK.USE_MERGER else None
        refiner_solver = torch.optim.SGD(
            refiner.parameters(), lr=cfg.LR_FINDER.START_LR / 10.0, momentum=cfg.TRAIN.MOMENTUM) if cfg.NETWORK.USE_REFINER else None
    else:
        raise Exception(f'[FATAL] Unknown optimizer: {cfg.TRAIN.POLICY}')

    # GradScaler
    scaler = GradScaler(init_scale=2**16)

    # Loss
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # LR Finder parameters
    start_lr = cfg.LR_FINDER.START_LR
    end_lr = cfg.LR_FINDER.END_LR
    num_batches_to_test = cfg.LR_FINDER.NUM_BATCHES_TO_TEST
    lr_multiplier = (end_lr / start_lr) ** (1 / (num_batches_to_test - 1))
    lrs = []
    losses = []
    smoothed_losses = []
    best_loss = float('inf')
    avg_beta = cfg.LR_FINDER.AVG_BETA

    logging.info(f'Starting LR Finder: {start_lr} to {end_lr} over {num_batches_to_test} batches')

    # Training mode
    encoder.train()
    decoder.train()
    merger.train()
    refiner.train()

    # LR Finder loop
    for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
        if batch_idx >= num_batches_to_test:
            break

        current_lr = encoder_solver.param_groups[0]['lr']
        lrs.append(current_lr)

        rendering_images = utils.helpers.var_or_cuda(rendering_images)
        ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
        rendering_images = torch.clamp(rendering_images, min=-1, max=1)
        ground_truth_volumes = torch.clamp(ground_truth_volumes, min=0, max=1)

        with autocast(device.type, enabled=True):
            image_features = encoder(rendering_images)
            raw_features, generated_volumes = decoder(image_features)
            if cfg.NETWORK.USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)
            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes)
            total_loss = encoder_loss
            if cfg.NETWORK.USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                refiner_loss = bce_loss(generated_volumes, ground_truth_volumes)
                total_loss += refiner_loss

        encoder_solver.zero_grad()
        decoder_solver.zero_grad()
        if cfg.NETWORK.USE_MERGER:
            merger_solver.zero_grad()
        if cfg.NETWORK.USE_REFINER:
            refiner_solver.zero_grad()

        scaler.scale(total_loss).backward()
        scaler.unscale_(encoder_solver)
        scaler.unscale_(decoder_solver)
        if cfg.NETWORK.USE_MERGER:
            scaler.unscale_(merger_solver)
        if cfg.NETWORK.USE_REFINER:
            scaler.unscale_(refiner_solver)

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        if cfg.NETWORK.USE_MERGER:
            torch.nn.utils.clip_grad_norm_(merger.parameters(), max_norm=1.0)
        if cfg.NETWORK.USE_REFINER:
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)

        scaler.step(encoder_solver)
        scaler.step(decoder_solver)
        if cfg.NETWORK.USE_MERGER:
            scaler.step(merger_solver)
        if cfg.NETWORK.USE_REFINER:
            scaler.step(refiner_solver)
        scaler.update()

        loss_item = total_loss.item()
        losses.append(loss_item)
        smoothed_loss = loss_item if batch_idx == 0 else smoothed_losses[-1] * avg_beta + loss_item * (1 - avg_beta)
        smoothed_losses.append(smoothed_loss)

        if loss_item < best_loss:
            best_loss = loss_item

        if smoothed_losses[-1] > 10 * best_loss and batch_idx > 10:
            logging.warning(f'Loss diverged at LR {current_lr:.2e}. Stopping.')
            break

        for param_group in encoder_solver.param_groups:
            param_group['lr'] *= lr_multiplier
        for param_group in decoder_solver.param_groups:
            param_group['lr'] *= lr_multiplier
        if cfg.NETWORK.USE_MERGER:
            for param_group in merger_solver.param_groups:
                param_group['lr'] *= lr_multiplier
        if cfg.NETWORK.USE_REFINER:
            for param_group in refiner_solver.param_groups:
                param_group['lr'] *= lr_multiplier

        logging.info(f'Batch {batch_idx + 1}/{num_batches_to_test}, LR: {current_lr:.2e}, Loss: {loss_item:.4f}')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, smoothed_losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Smoothed Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, which="both", ls="-")

    # Suggest LR
    suggested_lr = None
    if len(smoothed_losses) > 5:
        min_loss_idx = np.argmin(smoothed_losses)
        search_window_end_idx = min_loss_idx
        search_window_start_idx = max(0, min_loss_idx - 50)
        if search_window_start_idx < search_window_end_idx:
            segment_lrs = np.array(lrs)[search_window_start_idx:search_window_end_idx]
            segment_smoothed_losses = np.array(smoothed_losses)[search_window_start_idx:search_window_end_idx]
            if len(segment_lrs) > 1:
                segment_grad = np.gradient(segment_smoothed_losses, segment_lrs)
                steepest_idx_in_segment = np.argmin(segment_grad)
                suggested_lr = segment_lrs[max(0, steepest_idx_in_segment - 5)]
                plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested Encoder/Decoder LR: {suggested_lr:.2e}')
                logging.info('=' * 40)
                logging.info(' Learning Rate Finder Finished ')
                logging.info(f'Suggested Encoder/Decoder LR: {suggested_lr:.2e}')
                logging.info(f'Suggested Merger/Refiner LR: {suggested_lr/10:.2e}')
                logging.info('Please review the generated plot in your output directory and update your config file.')
                logging.info('=' * 40)
    if suggested_lr is None:
        logging.warning("Could not suggest an optimal LR.")

    plt.legend()
    plt.show() 
    plt.savefig(os.path.join(cfg.DIR.OUT_PATH, f'lr_finder_plot_{dt.now().strftime("%Y%m%d-%H%M%S")}.png'))
    plt.close()

    # Reset models and optimizers
    encoder.load_state_dict(initial_encoder_state)
    decoder.load_state_dict(initial_decoder_state)
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(initial_merger_state)
    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(initial_refiner_state)

    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(
            decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(
            merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS) if cfg.NETWORK.USE_MERGER else None
        refiner_solver = torch.optim.Adam(
            refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS) if cfg.NETWORK.USE_REFINER else None
    else:
        raise Exception(f'[FATAL] Unknown optimizer: {cfg.TRAIN.POLICY}')
    return suggested_lr