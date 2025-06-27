# # -*- coding: utf-8 -*-
# #
# # Developed by Haozhe Xie <cshzxie@gmail.com>
# # Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

# import os
# import logging
# import random
# import torch
# import torch.backends.cudnn
# import torch.utils.data
# import torch.nn as nn

# import utils.data_loaders
# import utils.data_transforms
# import utils.helpers

# from datetime import datetime as dt
# from tensorboardX import SummaryWriter
# from time import time

# from core.test import test_net
# from models.encoder import Encoder
# from models.decoder import Decoder
# from models.refiner import Refiner
# from models.merger import Merger
# from utils.average_meter import AverageMeter

# # Import for Automatic Mixed Precision
# from torch.amp import autocast, GradScaler


# def train_net(cfg):
#     # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
#     torch.backends.cudnn.benchmark = True

#     # Set up data augmentation
#     IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
#     CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
#     train_transforms = utils.data_transforms.Compose([
#         utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
#         utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
#         utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
#         utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
#         utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
#         utils.data_transforms.RandomFlip(),
#         utils.data_transforms.RandomPermuteRGB(),
#         utils.data_transforms.ToTensor(),
#     ])
#     val_transforms = utils.data_transforms.Compose([
#         utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
#         utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
#         utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
#         utils.data_transforms.ToTensor(),
#     ])

#     # Set up data loader
#     train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
#     val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
#     train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
#         utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
#                                                     batch_size=cfg.CONST.BATCH_SIZE,
#                                                     num_workers=cfg.CONST.NUM_WORKER,
#                                                     pin_memory=True,
#                                                     shuffle=True,
#                                                     drop_last=True)
#     val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
#         utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
#                                                   batch_size=1,
#                                                   num_workers=cfg.CONST.NUM_WORKER,
#                                                   pin_memory=True,
#                                                   shuffle=False)

#     # Set up networks
#     encoder = Encoder(cfg)
#     decoder = Decoder(cfg)
#     refiner = Refiner(cfg)
#     merger  = Merger(cfg)
#     logging.info(f'Parameters in Encoder: {utils.helpers.count_parameters(encoder)}.')
#     logging.info(f'Parameters in Decoder: {utils.helpers.count_parameters(decoder)}.')
#     logging.info(f'Parameters in Refiner: {utils.helpers.count_parameters(refiner)}.')
#     logging.info(f'Parameters in Merger: {utils.helpers.count_parameters(merger)}.')

#     # Initialize weights of networks
#     encoder.apply(utils.helpers.init_weights)
#     decoder.apply(utils.helpers.init_weights)
#     refiner.apply(utils.helpers.init_weights)
#     merger.apply(utils.helpers.init_weights)

#     # Set up solver
#     if cfg.TRAIN.POLICY == 'adam':
#         encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
#                                           lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
#                                           betas=cfg.TRAIN.BETAS,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#         decoder_solver = torch.optim.Adam(decoder.parameters(),
#                                           lr=cfg.TRAIN.DECODER_LEARNING_RATE,
#                                           betas=cfg.TRAIN.BETAS,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#         refiner_solver = torch.optim.Adam(refiner.parameters(),
#                                           lr=cfg.TRAIN.REFINER_LEARNING_RATE,
#                                           betas=cfg.TRAIN.BETAS,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#         merger_solver = torch.optim.Adam(merger.parameters(),
#                                           lr=cfg.TRAIN.MERGER_LEARNING_RATE,
#                                           betas=cfg.TRAIN.BETAS,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#     elif cfg.TRAIN.POLICY == 'sgd':
#         encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
#                                           lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
#                                           momentum=cfg.TRAIN.MOMENTUM,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#         decoder_solver = torch.optim.SGD(decoder.parameters(),
#                                           lr=cfg.TRAIN.DECODER_LEARNING_RATE,
#                                           momentum=cfg.TRAIN.MOMENTUM,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#         refiner_solver = torch.optim.SGD(refiner.parameters(),
#                                           lr=cfg.TRAIN.REFINER_LEARNING_RATE,
#                                           momentum=cfg.TRAIN.MOMENTUM,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#         merger_solver = torch.optim.SGD(merger.parameters(),
#                                           lr=cfg.TRAIN.MERGER_LEARNING_RATE,
#                                           momentum=cfg.TRAIN.MOMENTUM,
#                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
#     else:
#         raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

#     # Set up learning rate scheduler to decay learning rates dynamically
#     encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
#                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
#                                                                gamma=cfg.TRAIN.GAMMA)
#     decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
#                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
#                                                                gamma=cfg.TRAIN.GAMMA)
#     refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
#                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
#                                                                gamma=cfg.TRAIN.GAMMA)
#     merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
#                                                                milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
#                                                                gamma=cfg.TRAIN.GAMMA)

#     # Initialize GradScaler for mixed precision training
#     scaler = GradScaler(init_scale=2**16)

#     # Move models to CUDA if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if device.type == 'cuda':
#         encoder = torch.nn.DataParallel(encoder).to(device)
#         decoder = torch.nn.DataParallel(decoder).to(device)
#         refiner = torch.nn.DataParallel(refiner).to(device)
#         merger  = torch.nn.DataParallel(merger).to(device)

#     # Set up loss functions
#     bce_loss = torch.nn.BCEWithLogitsLoss()

#     # Load pretrained model if exists
#     init_epoch = 0
#     best_iou = -1
#     best_epoch = -1
#     if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
#         logging.info(f'Recovering from {cfg.CONST.WEIGHTS} ...')
#         checkpoint = torch.load(cfg.CONST.WEIGHTS, weights_only=False)
#         init_epoch = checkpoint['epoch_idx']
#         best_iou = checkpoint['best_iou']
#         best_epoch = checkpoint['best_epoch']

#         encoder.load_state_dict(checkpoint['encoder_state_dict'])
#         decoder.load_state_dict(checkpoint['decoder_state_dict'])
#         if cfg.NETWORK.USE_REFINER:
#             refiner.load_state_dict(checkpoint['refiner_state_dict'])
#         if cfg.NETWORK.USE_MERGER:
#             merger.load_state_dict(checkpoint['merger_state_dict'])
#         if 'scaler_state_dict' in checkpoint:
#             scaler.load_state_dict(checkpoint['scaler_state_dict'])

#         logging.info(f'Recover complete. Current epoch #{init_epoch}, Best IoU = {best_iou:.4f} at epoch #{best_epoch}.')

#     # Summary writer for TensorBoard
#     output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
#     cfg.DIR.LOGS = output_dir % 'logs'
#     cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
#     train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
#     val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

#     # Training loop
#     for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
#         # Tick / tock
#         epoch_start_time = time()

#         # Batch average metrics
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         encoder_losses = AverageMeter()
#         refiner_losses = AverageMeter()

#         # Switch models to training mode
#         encoder.train()
#         decoder.train()
#         merger.train()
#         refiner.train()

#         batch_end_time = time()
#         n_batches = len(train_data_loader)
#         for batch_idx, (taxonomy_names, sample_names, rendering_images,
#                         ground_truth_volumes) in enumerate(train_data_loader):
#             # Measure data time
#             data_time.update(time() - batch_end_time)

#             # Get data from data loader
#             rendering_images = utils.helpers.var_or_cuda(rendering_images)
#             ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

#             # Clip input images to prevent extreme values
#             rendering_images = torch.clamp(rendering_images, min=-1, max=1)
#             ground_truth_volumes = torch.clamp(ground_truth_volumes, min=0, max=1)

#             # Check device consistency
#             assert rendering_images.device == ground_truth_volumes.device == next(encoder.parameters()).device, \
#                 "Device mismatch between data and model"

#             # Wrap forward pass in autocast context manager
#             with autocast(device_type=device.type, enabled=True):
#                 # Train the encoder, decoder, refiner, and merger
#                 image_features = encoder(rendering_images)
#                 raw_features, generated_volumes = decoder(image_features)

#                 if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
#                     generated_volumes = merger(raw_features, generated_volumes)
#                 else:
#                     generated_volumes = torch.mean(generated_volumes, dim=1)

#                 encoder_loss = bce_loss(generated_volumes, ground_truth_volumes)

#                 if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
#                     generated_volumes = refiner(generated_volumes)
#                     refiner_loss = bce_loss(generated_volumes, ground_truth_volumes)
#                 else:
#                     refiner_loss = encoder_loss

#                 # Combine losses for single backward pass
#                 total_loss = encoder_loss + (refiner_loss if cfg.NETWORK.USE_REFINER and 
#                                              epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER else 0)

#             # Gradient descent
#             encoder.zero_grad()
#             decoder.zero_grad()
#             refiner.zero_grad()
#             merger.zero_grad()

#             # Single backward pass
#             scaler.scale(total_loss).backward()

#             # Unscale gradients for clipping
#             scaler.unscale_(encoder_solver)
#             scaler.unscale_(decoder_solver)
#             scaler.unscale_(refiner_solver)
#             scaler.unscale_(merger_solver)

#             # Clip gradients to prevent explosion
#             torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(merger.parameters(), max_norm=1.0)

#             # Optimizer steps
#             scaler.step(encoder_solver)
#             scaler.step(decoder_solver)
#             if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
#                 scaler.step(refiner_solver)
#             if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
#                 scaler.step(merger_solver)
#             scaler.update()

#             # Append loss to average metrics
#             encoder_losses.update(encoder_loss.item())
#             refiner_losses.update(refiner_loss.item())

#             # Append loss to TensorBoard
#             n_itr = epoch_idx * n_batches + batch_idx
#             train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
#             train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

#             # Tick / tock
#             batch_time.update(time() - batch_end_time)
#             batch_end_time = time()
#             logging.info(
#                 f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}][Batch {batch_idx + 1}/{n_batches}] '
#                 f'BatchTime = {batch_time.val:.3f} (s) DataTime = {data_time.val:.3f} (s) '
#                 f'EDLoss = {encoder_loss.item():.4f} RLoss = {refiner_loss.item():.4f}')

#         # Adjust learning rate
#         encoder_lr_scheduler.step()
#         decoder_lr_scheduler.step()
#         refiner_lr_scheduler.step()
#         merger_lr_scheduler.step()

#         # Append epoch loss to TensorBoard
#         train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
#         train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

#         # Tick / tock
#         epoch_end_time = time()
#         logging.info(
#                 f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}] EpochTime = {epoch_end_time - epoch_start_time:.3f} (s) '
#                 f'EDLoss = {encoder_losses.avg:.4f} RLoss = {refiner_losses.avg:.4f}')

#         # Update Rendering Views
#         if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
#             n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
#             train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
#             logging.info(
#                   f'Epoch [{epoch_idx + 2}/{cfg.TRAIN.NUM_EPOCHS}] Update #RenderingViews to {n_views_rendering}')

#         # Validate the training models
#         iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)

#         # Save weights to file
#         if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
#             file_name = f'checkpoint-epoch-{epoch_idx + 1:03d}.pth'
#             if iou > best_iou:
#                 best_iou = iou
#                 best_epoch = epoch_idx
#                 file_name = 'checkpoint-best.pth'

#             output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
#             if not os.path.exists(cfg.DIR.CHECKPOINTS):
#                 os.makedirs(cfg.DIR.CHECKPOINTS)

#             checkpoint = {
#                 'epoch_idx': int(epoch_idx),
#                 'best_iou': float(best_iou),
#                 'best_epoch': int(best_epoch),
#                 'encoder_state_dict': encoder.state_dict(),
#                 'decoder_state_dict': decoder.state_dict(),
#                 'scaler_state_dict': scaler.state_dict(),
#             }
#             if cfg.NETWORK.USE_REFINER:
#                 checkpoint['refiner_state_dict'] = refiner.state_dict()
#             if cfg.NETWORK.USE_MERGER:
#                 checkpoint['merger_state_dict'] = merger.state_dict()

#             torch.save(checkpoint, output_path)
#             logging.info(f'Saved checkpoint to {output_path} ...')

#     # Close SummaryWriter for TensorBoard
#     train_writer.close()
#     val_writer.close()





















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

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time
from tqdm.notebook import tqdm # Import tqdm for Jupyter/Colab
import sys # Import sys to configure logging stream

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter

# Import for Automatic Mixed Precision
from torch.amp import autocast, GradScaler


def train_net(cfg):
    # --- START Logging Configuration ---
    # Create a custom logger instance for your training
    custom_logger = logging.getLogger(__name__)
    custom_logger.setLevel(logging.INFO) # Set the minimum logging level

    # Ensure no handlers from previous runs or root logger are attached to avoid duplicate output
    if custom_logger.handlers:
        for handler in custom_logger.handlers[:]:
            custom_logger.removeHandler(handler)
    if logging.root.handlers: # Also remove default handlers from root logger if necessary
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    # 1. File Handler: This will write all INFO messages (and above) to a log file
    log_file_path = os.path.join(cfg.DIR.OUT_PATH, 'training_details.log')
    # Make sure the directory exists for the log file
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    custom_logger.addHandler(file_handler)

    # 2. Console Handler: This will write messages to sys.stdout (your Kaggle console/Logs tab)
    #    You can adjust the level here to control verbosity in the console.
    #    For *all* INFO messages, set to logging.INFO. For less, set to logging.WARNING.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Display INFO messages in console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s') # Simpler format for console
    console_handler.setFormatter(console_formatter)
    custom_logger.addHandler(console_handler)

    # Prevent the logger from sending messages to the root logger's handlers
    # This is crucial to stop output from appearing in notebook cells directly.
    custom_logger.propagate = False

    # Replace all 'logging.' calls with 'custom_logger.'
    # For example, logging.info(...) becomes custom_logger.info(...)
    # This ensures your custom handlers are used.
    # --- END Logging Configuration ---


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
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                     batch_size=cfg.CONST.BATCH_SIZE,
                                                     num_workers=cfg.CONST.NUM_WORKER,
                                                     pin_memory=True,
                                                     shuffle=True,
                                                     drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.NUM_WORKER,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger  = Merger(cfg)
    custom_logger.info(f'Parameters in Encoder: {utils.helpers.count_parameters(encoder)}.')
    custom_logger.info(f'Parameters in Decoder: {utils.helpers.count_parameters(decoder)}.')
    custom_logger.info(f'Parameters in Refiner: {utils.helpers.count_parameters(refiner)}.')
    custom_logger.info(f'Parameters in Merger: {utils.helpers.count_parameters(merger)}.')

    # Initialize weights of networks
    encoder.apply(utils.helpers.init_weights)
    decoder.apply(utils.helpers.init_weights)
    refiner.apply(utils.helpers.init_weights)
    merger.apply(utils.helpers.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        refiner_solver = torch.optim.Adam(refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        merger_solver = torch.optim.Adam(merger.parameters(),
                                          lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          momentum=cfg.TRAIN.MOMENTUM,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          momentum=cfg.TRAIN.MOMENTUM,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        refiner_solver = torch.optim.SGD(refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                          momentum=cfg.TRAIN.MOMENTUM,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        merger_solver = torch.optim.SGD(merger.parameters(),
                                          lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                          momentum=cfg.TRAIN.MOMENTUM,
                                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                 milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                 gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                 milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                 gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                 milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                 gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                                 milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                                 gamma=cfg.TRAIN.GAMMA)

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

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        custom_logger.info(f'Recovering from {cfg.CONST.WEIGHTS} ...')
        checkpoint = torch.load(cfg.CONST.WEIGHTS, weights_only=False)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        custom_logger.info(f'Recover complete. Current epoch #{init_epoch}, Best IoU = {best_iou:.4f} at epoch #{best_epoch}.')

    # Summary writer for TensorBoard
    # Ensure cfg.DIR.OUT_PATH is defined (e.g., /kaggle/working/output)
    output_base_dir = os.path.join(cfg.DIR.OUT_PATH, dt.now().strftime('%Y-%m-%d_%H-%M-%S'))
    cfg.DIR.LOGS = os.path.join(output_base_dir, 'logs')
    cfg.DIR.CHECKPOINTS = os.path.join(output_base_dir, 'checkpoints')
    
    os.makedirs(cfg.DIR.LOGS, exist_ok=True)
    os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Training loop
    # Wrap the outer epoch loop with tqdm
    for epoch_idx in tqdm(range(init_epoch, cfg.TRAIN.NUM_EPOCHS), desc="Epochs"):
        # Tick / tock
        epoch_start_time = time()

        # Batch average metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()

        # Switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        # Wrap the inner batch loop with tqdm
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in tqdm(enumerate(train_data_loader),
                                                     total=n_batches,
                                                     desc=f"Epoch {epoch_idx + 1} Batches",
                                                     leave=False):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)

            # Clip input images to prevent extreme values
            rendering_images = torch.clamp(rendering_images, min=-1, max=1)
            ground_truth_volumes = torch.clamp(ground_truth_volumes, min=0, max=1)

            # Check device consistency
            assert rendering_images.device == ground_truth_volumes.device == next(encoder.parameters()).device, \
                "Device mismatch between data and model"

            # Wrap forward pass in autocast context manager
            with autocast(device_type=device.type, enabled=True):
                # Train the encoder, decoder, refiner, and merger
                image_features = encoder(rendering_images)
                raw_features, generated_volumes = decoder(image_features)

                if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                    generated_volumes = merger(raw_features, generated_volumes)
                else:
                    generated_volumes = torch.mean(generated_volumes, dim=1)

                encoder_loss = bce_loss(generated_volumes, ground_truth_volumes)

                if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                    generated_volumes = refiner(generated_volumes)
                    refiner_loss = bce_loss(generated_volumes, ground_truth_volumes)
                else:
                    refiner_loss = encoder_loss

                # Combine losses for single backward pass
                total_loss = encoder_loss + (refiner_loss if cfg.NETWORK.USE_REFINER and 
                                             epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER else 0)

            # Gradient descent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()

            # Single backward pass
            scaler.scale(total_loss).backward()

            # Unscale gradients for clipping
            scaler.unscale_(encoder_solver)
            scaler.unscale_(decoder_solver)
            scaler.unscale_(refiner_solver)
            scaler.unscale_(merger_solver)

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(merger.parameters(), max_norm=1.0)

            # Optimizer steps
            scaler.step(encoder_solver)
            scaler.step(decoder_solver)
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                scaler.step(refiner_solver)
            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                scaler.step(merger_solver)
            scaler.update()

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            # Use custom_logger.info instead of logging.info
            custom_logger.info(
                f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}][Batch {batch_idx + 1}/{n_batches}] '
                f'BatchTime = {batch_time.val:.3f} (s) DataTime = {data_time.val:.3f} (s) '
                f'EDLoss = {encoder_loss.item():.4f} RLoss = {refiner_loss.item():.4f}')

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        custom_logger.info(
                f'[Epoch {epoch_idx + 1}/{cfg.TRAIN.NUM_EPOCHS}] EpochTime = {epoch_end_time - epoch_start_time:.3f} (s) '
                f'EDLoss = {encoder_losses.avg:.4f} RLoss = {refiner_losses.avg:.4f}')

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            custom_logger.info(
                    f'Epoch [{epoch_idx + 2}/{cfg.TRAIN.NUM_EPOCHS}] Update #RenderingViews to {n_views_rendering}')

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_base_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            file_name = f'checkpoint-epoch-{epoch_idx + 1:03d}.pth'
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'checkpoint-best.pth'

            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            # Directory check and creation handled by output_base_dir earlier

            checkpoint = {
                'epoch_idx': int(epoch_idx),
                'best_iou': float(best_iou),
                'best_epoch': int(best_epoch),
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            if cfg.NETWORK.USE_REFINER:
                checkpoint['refiner_state_dict'] = refiner.state_dict()
            if cfg.NETWORK.USE_MERGER:
                checkpoint['merger_state_dict'] = merger.state_dict()

            torch.save(checkpoint, output_path)
            custom_logger.info(f'Saved checkpoint to {output_path} ...')

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()

    # Final message, useful for the console output
    custom_logger.info(f"Training finished. Best IoU: {best_iou:.4f} at epoch {best_epoch}.")
    custom_logger.info(f"Full logs are available in {log_file_path}")

