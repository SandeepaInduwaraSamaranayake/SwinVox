# -*- coding: utf-8 -*-
# Developed by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import os
import logging
import copy
import torch
import torch.utils.data
import optuna
import shutil
from torch.amp import autocast, GradScaler
from datetime import datetime as dt
from tensorboardX import SummaryWriter
import utils.data_loaders
import utils.data_transforms
import utils.helpers
import utils.average_meter
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from core.test import test_net


def optuna_tune(cfg):
    """
    Run Optuna hyperparameter tuning for SwinVox model.
    
    Args:
        cfg: Configuration object from config.py
    """
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##############################################################################################
    ################################Optimize Configuration########################################
    n_trials = 3
    timeout = 72000
    ##############################################################################################
    ##############################################################################################

    # Print header
    logging.info('=' * 40)
    logging.info(' Starting Optuna Hyperparameter Tuning ')
    logging.info('=' * 40)
    logging.info(f"[Optuna Tune] Using device: {device}")

    # Ensure output directory exists
    os.makedirs(cfg.DIR.OUT_PATH, exist_ok=True)

    # Define a shared log file for all trial results
    all_trials_log_file = os.path.join(cfg.DIR.OUT_PATH, 'optuna_trial_results.txt')

    def objective(trial):
        """Optuna objective function."""
        # Deep copy the config
        trial_cfg = copy.deepcopy(cfg)

        ##############################################################################################
        ##############################################################################################
        #################################selected hyperparameters#####################################
        config_params = {
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [32, 64]),

            # 'ENCODER_LR': trial.suggest_float('ENCODER_LR', 1e-7, 5e-4, log=True),
            # 'DECODER_LR': trial.suggest_float('DECODER_LR', 1e-7, 5e-4, log=True),
            # 'MERGER_LR': trial.suggest_float('MERGER_LR', 1e-8, 1e-3, log=True),
            # 'REFINER_LR': trial.suggest_float('REFINER_LR', 1e-8, 1e-3, log=True),

            # 'GAMMA': trial.suggest_float('GAMMA', 0.1, 0.9),

            # # 'USE_MERGER': trial.suggest_categorical('USE_MERGER', [True, False]),
            # # 'USE_REFINER': trial.suggest_categorical('USE_REFINER', [True, False]),

            # 'POLICY': trial.suggest_categorical('POLICY', ['adam', 'sgd']),
            # 'BETA1': trial.suggest_float('BETA1', 0.8, 0.9, step=0.05),
            # 'BETA2': trial.suggest_float('BETA2', 0.99, 0.999, step=0.003),

            # 'BRIGHTNESS': trial.suggest_float('BRIGHTNESS', 0.1, 0.5),
            # 'CONTRAST': trial.suggest_float('CONTRAST', 0.1, 0.5),
            # 'SATURATION': trial.suggest_float('SATURATION', 0.1, 0.5),
            # 'NOISE_STD': trial.suggest_float('NOISE_STD', 0.01, 0.1),

            # 'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', 1e-6, 1e-3, log=True),

            # 'USE_SWIN_T_MULTI_STAGE': trial.suggest_categorical('USE_SWIN_T_MULTI_STAGE', [True, False]),
            # 'SWIN_T_STAGES': trial.suggest_categorical('SWIN_T_STAGES', [[0,1,2,3],[1,2,3],[2,3],[3]]),
            # 'USE_CROSS_VIEW_ATTENTION': trial.suggest_categorical('USE_CROSS_VIEW_ATTENTION', [True, False]),

            # 'CROSS_ATT_REDUCTION_RATIO': trial.suggest_categorical('CROSS_ATT_REDUCTION_RATIO', [2,4,8]),
            # 'ATT_SPATIAL_DOWNSAMPLE_RATIO': trial.suggest_categorical('ATT_SPATIAL_DOWNSAMPLE_RATIO', [2,4]),
            # 'CROSS_ATT_NUM_HEADS': trial.suggest_categorical('CROSS_ATT_NUM_HEADS', [2,4,8]),
        }
        ##############################################################################################
        ##############################################################################################
        ##############################################################################################

        # Update trial config
        trial_cfg.CONST.BATCH_SIZE = config_params['BATCH_SIZE']

        # trial_cfg.TRAIN.ENCODER_LEARNING_RATE = config_params['ENCODER_LR']
        # trial_cfg.TRAIN.DECODER_LEARNING_RATE = config_params['DECODER_LR']
        # trial_cfg.TRAIN.MERGER_LEARNING_RATE = config_params['MERGER_LR']
        # trial_cfg.TRAIN.REFINER_LEARNING_RATE = config_params['REFINER_LR']

        # trial_cfg.TRAIN.GAMMA = config_params['GAMMA']

        # # trial_cfg.NETWORK.USE_MERGER = config_params['USE_MERGER']
        # # trial_cfg.NETWORK.USE_REFINER = config_params['USE_REFINER']

        # trial_cfg.TRAIN.POLICY = config_params['POLICY']
        # trial_cfg.TRAIN.BETAS = (config_params['BETA1'], config_params['BETA2'])

        # trial_cfg.TRAIN.BRIGHTNESS = config_params['BRIGHTNESS']
        # trial_cfg.TRAIN.CONTRAST = config_params['CONTRAST']
        # trial_cfg.TRAIN.SATURATION = config_params['SATURATION']
        # trial_cfg.TRAIN.NOISE_STD = config_params['NOISE_STD']

        # trial_cfg.TRAIN.WEIGHT_DECAY = config_params['WEIGHT_DECAY']

        # trial_cfg.NETWORK.USE_SWIN_T_MULTI_STAGE = config_params['USE_SWIN_T_MULTI_STAGE']
        # trial_cfg.NETWORK.SWIN_T_STAGES = config_params['SWIN_T_STAGES']
        # trial_cfg.NETWORK.USE_CROSS_VIEW_ATTENTION = config_params['USE_CROSS_VIEW_ATTENTION']
        # trial_cfg.NETWORK.CROSS_ATT_REDUCTION_RATIO = config_params['CROSS_ATT_REDUCTION_RATIO']
        # trial_cfg.NETWORK.ATT_SPATIAL_DOWNSAMPLE_RATIO = config_params['ATT_SPATIAL_DOWNSAMPLE_RATIO']
        # trial_cfg.NETWORK.CROSS_ATT_NUM_HEADS = config_params['CROSS_ATT_NUM_HEADS']

        ##############################################################################################
        ##############################################################################################
        
        trial_cfg.TRAIN.NUM_EPOCHS = 20  # Short trials
        trial_cfg.TRAIN.ENCODER_LR_MILESTONES = [3]
        trial_cfg.TRAIN.DECODER_LR_MILESTONES = [3]
        trial_cfg.TRAIN.MERGER_LR_MILESTONES = [3]
        trial_cfg.TRAIN.REFINER_LR_MILESTONES = [3]

        # trial_cfg.TRAIN.EPOCH_START_USE_REFINER = 0
        # trial_cfg.TRAIN.EPOCH_START_USE_MERGER = 0

        ##############################################################################################
        ##############################################################################################

        # Set unique output directory
        log_dir = os.path.join(trial_cfg.DIR.OUT_PATH, f'trial_{trial.number}_{dt.now().strftime("%Y%m%d_%H%M%S")}')
        trial_cfg.DIR.LOGS = os.path.join(log_dir, 'logs')
        trial_cfg.DIR.CHECKPOINTS = os.path.join(log_dir, 'checkpoints')
        os.makedirs(trial_cfg.DIR.LOGS, exist_ok=True)
        os.makedirs(trial_cfg.DIR.CHECKPOINTS, exist_ok=True)

        logging.info(f"Trial {trial.number}: Config={config_params}")

        # Data transforms
        IMG_SIZE = trial_cfg.CONST.IMG_H, trial_cfg.CONST.IMG_W
        CROP_SIZE = trial_cfg.CONST.CROP_IMG_H, trial_cfg.CONST.CROP_IMG_W
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(trial_cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ColorJitter(trial_cfg.TRAIN.BRIGHTNESS, trial_cfg.TRAIN.CONTRAST, trial_cfg.TRAIN.SATURATION),
            utils.data_transforms.RandomNoise(trial_cfg.TRAIN.NOISE_STD),
            utils.data_transforms.Normalize(mean=trial_cfg.DATASET.MEAN, std=trial_cfg.DATASET.STD),
            utils.data_transforms.RandomFlip(),
            utils.data_transforms.RandomPermuteRGB(),
            utils.data_transforms.ToTensor(),
        ])
        val_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(trial_cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=trial_cfg.DATASET.MEAN, std=trial_cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        # Data loaders
        train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[trial_cfg.DATASET.TRAIN_DATASET](trial_cfg)
        val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[trial_cfg.DATASET.TEST_DATASET](trial_cfg)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_loader.get_dataset(
                utils.data_loaders.DatasetType.TRAIN, trial_cfg.CONST.N_VIEWS_RENDERING, train_transforms),
            batch_size=trial_cfg.CONST.BATCH_SIZE,
            num_workers=trial_cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset_loader.get_dataset(
                utils.data_loaders.DatasetType.VAL, trial_cfg.CONST.N_VIEWS_RENDERING, val_transforms),
            batch_size=1,
            num_workers=trial_cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False
        )

        # Models
        encoder = Encoder(trial_cfg).to(device)
        decoder = Decoder(trial_cfg).to(device)
        merger = Merger(trial_cfg).to(device)
        refiner = Refiner(trial_cfg).to(device)
        encoder.apply(utils.helpers.init_weights)
        decoder.apply(utils.helpers.init_weights)
        merger.apply(utils.helpers.init_weights)
        refiner.apply(utils.helpers.init_weights)
        if device.type == 'cuda':
            encoder = torch.nn.DataParallel(encoder)
            decoder = torch.nn.DataParallel(decoder)
            merger = torch.nn.DataParallel(merger)
            refiner = torch.nn.DataParallel(refiner)
        logging.info(f'Trial {trial.number} Parameters in Encoder: {utils.helpers.count_parameters(encoder)}')
        logging.info(f'Trial {trial.number} Parameters in Decoder: {utils.helpers.count_parameters(decoder)}')
        logging.info(f'Trial {trial.number} Parameters in Merger: {utils.helpers.count_parameters(merger)}')
        logging.info(f'Trial {trial.number} Parameters in Refiner: {utils.helpers.count_parameters(refiner)}')

        # Optimizers
        if trial_cfg.TRAIN.POLICY == 'adam':
            encoder_solver = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=trial_cfg.TRAIN.ENCODER_LEARNING_RATE,
                betas=trial_cfg.TRAIN.BETAS,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            )
            decoder_solver = torch.optim.Adam(
                decoder.parameters(),
                lr=trial_cfg.TRAIN.DECODER_LEARNING_RATE,
                betas=trial_cfg.TRAIN.BETAS,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            )
            merger_solver = torch.optim.Adam(
                merger.parameters(),
                lr=trial_cfg.TRAIN.MERGER_LEARNING_RATE,
                betas=trial_cfg.TRAIN.BETAS,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            ) if trial_cfg.NETWORK.USE_MERGER else None
            refiner_solver = torch.optim.Adam(
                refiner.parameters(),
                lr=trial_cfg.TRAIN.REFINER_LEARNING_RATE,
                betas=trial_cfg.TRAIN.BETAS,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            ) if trial_cfg.NETWORK.USE_REFINER else None
        elif trial_cfg.TRAIN.POLICY == 'sgd':
            encoder_solver = torch.optim.SGD(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=trial_cfg.TRAIN.ENCODER_LEARNING_RATE,
                momentum=trial_cfg.TRAIN.MOMENTUM,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            )
            decoder_solver = torch.optim.SGD(
                decoder.parameters(),
                lr=trial_cfg.TRAIN.DECODER_LEARNING_RATE,
                momentum=trial_cfg.TRAIN.MOMENTUM,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            )
            merger_solver = torch.optim.SGD(
                merger.parameters(),
                lr=trial_cfg.TRAIN.MERGER_LEARNING_RATE,
                momentum=trial_cfg.TRAIN.MOMENTUM,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            ) if trial_cfg.NETWORK.USE_MERGER else None
            refiner_solver = torch.optim.SGD(
                refiner.parameters(),
                lr=trial_cfg.TRAIN.REFINER_LEARNING_RATE,
                momentum=trial_cfg.TRAIN.MOMENTUM,
                weight_decay=trial_cfg.TRAIN.WEIGHT_DECAY
            ) if trial_cfg.NETWORK.USE_REFINER else None
        else:
            raise Exception(f'[FATAL] Unknown optimizer: {trial_cfg.TRAIN.POLICY}')

        # Schedulers
        encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            encoder_solver, milestones=trial_cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=trial_cfg.TRAIN.GAMMA)
        decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            decoder_solver, milestones=trial_cfg.TRAIN.DECODER_LR_MILESTONES, gamma=trial_cfg.TRAIN.GAMMA)
        merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            merger_solver, milestones=trial_cfg.TRAIN.MERGER_LR_MILESTONES, gamma=trial_cfg.TRAIN.GAMMA
        ) if trial_cfg.NETWORK.USE_MERGER else None
        refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            refiner_solver, milestones=trial_cfg.TRAIN.REFINER_LR_MILESTONES, gamma=trial_cfg.TRAIN.GAMMA
        ) if trial_cfg.NETWORK.USE_REFINER else None

        # GradScaler and Loss
        scaler = GradScaler(init_scale=2**16)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        # TensorBoard writer
        val_writer = SummaryWriter(log_dir=trial_cfg.DIR.LOGS)

        # Training loop
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()
        best_iou = -1.0

        for epoch_idx in range(trial_cfg.TRAIN.NUM_EPOCHS):
            encoder_losses = utils.average_meter.AverageMeter()
            refiner_losses = utils.average_meter.AverageMeter()

            for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
                rendering_images = utils.helpers.var_or_cuda(rendering_images)
                ground_truth_volumes = utils.helpers.var_or_cuda(ground_truth_volumes)
                rendering_images = torch.clamp(rendering_images, min=-1, max=1)
                ground_truth_volumes = torch.clamp(ground_truth_volumes, min=0, max=1)

                with autocast(device.type, enabled=True):
                    image_features = encoder(rendering_images)
                    raw_features, generated_volumes = decoder(image_features)
                    if trial_cfg.NETWORK.USE_MERGER and epoch_idx >= trial_cfg.TRAIN.EPOCH_START_USE_MERGER:
                        generated_volumes = merger(raw_features, generated_volumes)
                    else:
                        generated_volumes = torch.mean(generated_volumes, dim=1)
                    encoder_loss = bce_loss(generated_volumes, ground_truth_volumes)
                    if trial_cfg.NETWORK.USE_REFINER and epoch_idx >= trial_cfg.TRAIN.EPOCH_START_USE_REFINER:
                        generated_volumes = refiner(generated_volumes)
                        refiner_loss = bce_loss(generated_volumes, ground_truth_volumes)
                    else:
                        refiner_loss = encoder_loss
                    total_loss = encoder_loss + (refiner_loss if trial_cfg.NETWORK.USE_REFINER and
                                                 epoch_idx >= trial_cfg.TRAIN.EPOCH_START_USE_REFINER else 0)

                encoder_solver.zero_grad()
                decoder_solver.zero_grad()
                if trial_cfg.NETWORK.USE_MERGER:
                    merger_solver.zero_grad()
                if trial_cfg.NETWORK.USE_REFINER:
                    refiner_solver.zero_grad()

                scaler.scale(total_loss).backward()
                scaler.unscale_(encoder_solver)
                scaler.unscale_(decoder_solver)
                if trial_cfg.NETWORK.USE_MERGER:
                    scaler.unscale_(merger_solver)
                if trial_cfg.NETWORK.USE_REFINER:
                    scaler.unscale_(refiner_solver)

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                if trial_cfg.NETWORK.USE_MERGER:
                    torch.nn.utils.clip_grad_norm_(merger.parameters(), max_norm=1.0)
                if trial_cfg.NETWORK.USE_REFINER:
                    torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)

                scaler.step(encoder_solver)
                scaler.step(decoder_solver)
                if trial_cfg.NETWORK.USE_MERGER:
                    scaler.step(merger_solver)
                if trial_cfg.NETWORK.USE_REFINER:
                    scaler.step(refiner_solver)
                scaler.update()

                encoder_losses.update(encoder_loss.item())
                refiner_losses.update(refiner_loss.item())

                logging.info(f"Trial {trial.number} Epoch {epoch_idx+1}/{trial_cfg.TRAIN.NUM_EPOCHS} Batch {batch_idx+1}: "
                              f"Encoder Loss={encoder_loss.item():.4f}, Refiner Loss={refiner_loss.item():.4f}")

            # Adjust learning rate
            encoder_lr_scheduler.step()
            decoder_lr_scheduler.step()
            if trial_cfg.NETWORK.USE_MERGER:
                merger_lr_scheduler.step()
            if trial_cfg.NETWORK.USE_REFINER:
                refiner_lr_scheduler.step()

            # Validation
            iou = test_net(trial_cfg, epoch_idx + 1, log_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)
            logging.info(f"Trial {trial.number} Epoch {epoch_idx+1}: IoU={iou:.4f}")

            # Pruning
            trial.report(iou, epoch_idx)
            if trial.should_prune():
                logging.info(f"Trial {trial.number} pruned at epoch {epoch_idx+1}, IoU={iou:.4f}")
                # Log pruned trial to the shared file before raising TrialPruned
                with open(all_trials_log_file, 'a') as f_log:
                    f_log.write(f"--- Pruned Trial {trial.number} ---\n")
                    f_log.write(f"IoU: {iou:.4f} (pruned at epoch {epoch_idx+1})\n")
                    f_log.write("Parameters:\n")
                    for k, v in trial.params.items():
                        f_log.write(f"  {k}: {v}\n")
                    f_log.write("\n")

                raise optuna.TrialPruned()

            # Early stopping
            if iou < 0.2 and epoch_idx >= 2:
                logging.info(f"Trial {trial.number} early stopped at epoch {epoch_idx+1}, IoU={iou:.4f}")
                 # Log early stopped trial to the shared file
                with open(all_trials_log_file, 'a') as f_log:
                    f_log.write(f"--- Early Stopped Trial {trial.number} ---\n")
                    f_log.write(f"IoU: {iou:.4f} (early stopped at epoch {epoch_idx+1})\n")
                    f_log.write("Parameters:\n")
                    for k, v in trial.params.items():
                        f_log.write(f"  {k}: {v}\n")
                    f_log.write("\n")
                # Break out of the epoch loop for this trial
                break

            best_iou = max(best_iou, iou)

        val_writer.close()

        # Log completed trial to the shared file
        with open(all_trials_log_file, 'a') as f_log:
            f_log.write(f"--- Completed Trial {trial.number} ---\n")
            f_log.write(f"Final IoU: {best_iou:.4f}\n")
            f_log.write("Parameters:\n")
            for k, v in trial.params.items():
                f_log.write(f"  {k}: {v}\n")
            f_log.write("\n")

        # Cleanup
        if best_iou < 0.2:
            shutil.rmtree(log_dir, ignore_errors=True)

        return best_iou

    # Create study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    logging.info(f"Starting Optuna study with {n_trials} trials")
    study.optimize(objective, n_trials = n_trials, timeout = timeout)

    # Log best trial
    best_trial = study.best_trial
    logging.info(f"Best Trial: Number={best_trial.number}, IoU={best_trial.value:.4f}, Params={best_trial.params}")

    # Save best config
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(cfg.DIR.OUT_PATH, 'config_best.py'), 'w') as f:
        for line in lines:
            if 'BATCH_SIZE =' in line:
                f.write(f'__C.CONST.BATCH_SIZE = {best_trial.params["BATCH_SIZE"]}\n')
            elif 'USE_MERGER =' in line:
                f.write(f'__C.NETWORK.USE_MERGER = {best_trial.params["USE_MERGER"]}\n')
            elif 'USE_REFINER =' in line:
                f.write(f'__C.NETWORK.USE_REFINER = {best_trial.params["USE_REFINER"]}\n')
            elif 'POLICY =' in line:
                f.write(f'__C.TRAIN.POLICY = "{best_trial.params["POLICY"]}"\n')
            elif 'BETAS =' in line:
                f.write(f'__C.TRAIN.BETAS = ({best_trial.params["BETA1"]}, {best_trial.params["BETA2"]})\n')
            elif 'BRIGHTNESS =' in line:
                f.write(f'__C.TRAIN.BRIGHTNESS = {best_trial.params["BRIGHTNESS"]}\n')
            elif 'CONTRAST =' in line:
                f.write(f'__C.TRAIN.CONTRAST = {best_trial.params["CONTRAST"]}\n')
            elif 'SATURATION =' in line:
                f.write(f'__C.TRAIN.SATURATION = {best_trial.params["SATURATION"]}\n')
            elif 'NOISE_STD =' in line:
                f.write(f'__C.TRAIN.NOISE_STD = {best_trial.params["NOISE_STD"]}\n')
            elif 'ENCODER_LEARNING_RATE =' in line:
                f.write(f'__C.TRAIN.ENCODER_LEARNING_RATE = {best_trial.params["ENCODER_LR"]}\n')
            elif 'DECODER_LEARNING_RATE =' in line:
                f.write(f'__C.TRAIN.DECODER_LEARNING_RATE = {best_trial.params["DECODER_LR"]}\n')
            elif 'MERGER_LEARNING_RATE =' in line:
                f.write(f'__C.TRAIN.MERGER_LEARNING_RATE = {best_trial.params["MERGER_LR"]}\n')
            elif 'REFINER_LEARNING_RATE =' in line:
                f.write(f'__C.TRAIN.REFINER_LEARNING_RATE = {best_trial.params["REFINER_LR"]}\n')
            elif 'WEIGHT_DECAY =' in line:
                f.write(f'__C.TRAIN.WEIGHT_DECAY = {best_trial.params["WEIGHT_DECAY"]}\n')
            elif 'GAMMA =' in line:
                f.write(f'__C.TRAIN.GAMMA = {best_trial.params["GAMMA"]}\n')
            elif 'NUM_EPOCHS =' in line:
                f.write(f'__C.TRAIN.NUM_EPOCHS = 50\n')
            elif 'ENCODER_LR_MILESTONES =' in line:
                f.write(f'__C.TRAIN.ENCODER_LR_MILESTONES = [30]\n')
            elif 'DECODER_LR_MILESTONES =' in line:
                f.write(f'__C.TRAIN.DECODER_LR_MILESTONES = [30]\n')
            elif 'MERGER_LR_MILESTONES =' in line:
                f.write(f'__C.TRAIN.MERGER_LR_MILESTONES = [30]\n')
            elif 'REFINER_LR_MILESTONES =' in line:
                f.write(f'__C.TRAIN.REFINER_LR_MILESTONES = [30]\n')
            else:
                f.write(line)

    logging.info('=' * 40)
    logging.info(' Optuna Hyperparameter Tuning Finished ')
    logging.info(f'Best IoU: {best_trial.value:.4f}')
    logging.info('Best Hyperparameters:')
    for key, value in best_trial.params.items():
        logging.info(f'  {key}: {value}')
    logging.info(f'Saved best configuration to {os.path.join(cfg.DIR.OUT_PATH, "config_best.py")}')
    logging.info('Please review the results and update your config file for final training.')
    logging.info('=' * 40)