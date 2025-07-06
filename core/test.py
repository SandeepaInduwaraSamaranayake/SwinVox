# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified  by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data
import os

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter

def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        logging.info(f'Loading weights from {cfg.CONST.WEIGHTS} ...')
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    test_fscore = dict()
    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # Apply sigmoid to generated_volume BEFORE calculating IoU and F-score
            generated_volume_prob = torch.sigmoid(generated_volume)

            # IoU and F-score per sample
            sample_iou = []
            sample_fscore = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume_prob, th).float()
                # Calculate IoU
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                # Handle edge case: if union is 0, IoU is 1.0 if intersection is also 0, else 0.0
                sample_iou.append(1.0 if union.item() == 0 and intersection.item() == 0 else (intersection / union).item() if union.item() > 0 else 0.0)

                # Calculate F-score components (TP, FP, FN)
                tp = torch.sum(_volume * ground_truth_volume).float()        # True Positives: _volume == 1 and ground_truth_volume == 1
                fp = torch.sum(_volume * (1 - ground_truth_volume)).float()  # False Positives: _volume == 1 and ground_truth_volume == 0
                fn = torch.sum((1 - _volume) * ground_truth_volume).float()  # False Negatives: _volume == 0 and ground_truth_volume == 1
                
                # Calculate Precision and Recall with epsilon for numerical stability
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                sample_fscore.append(f1.item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # F-score per taxonomy
            if taxonomy_id not in test_fscore:
                test_fscore[taxonomy_id] = {'n_samples': 0, 'fscore': []}
            test_fscore[taxonomy_id]['n_samples'] += 1
            test_fscore[taxonomy_id]['fscore'].append(sample_fscore)

            # Append generated volumes to TensorBoard
            if output_dir and test_writer and sample_idx < 3:
                img_dir = os.path.join(output_dir, 'images')
                os.makedirs(img_dir, exist_ok=True)
                # Volume Visualization: Use probability volume for visualization
                gv = utils.helpers.get_volume_views(generated_volume_prob.cpu().numpy(), img_dir, 'GV', sample_idx, epoch_idx)
                test_writer.add_image(f'Model{sample_idx:02d}/Reconstructed', gv, epoch_idx)

                gt = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy(), img_dir, 'GT', sample_idx, epoch_idx)
                test_writer.add_image(f'Model{sample_idx:02d}/GroundTruth', gt, epoch_idx)

            # Print sample loss, IoU, and F-score
            logging.info(f'Test[{sample_idx + 1}/{n_samples}] Taxonomy = {taxonomy_id} Sample = {sample_name} '
                         f'EDLoss = {encoder_loss.item():.4f} RLoss = {refiner_loss.item():.4f} '
                         f'IoU = {[f"{si:.4f}" for si in sample_iou]} F-score = {[f"{sf:.4f}" for sf in sample_fscore]}')

    # Output testing results for IoU
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Output testing results for F-score
    mean_fscore = []
    for taxonomy_id in test_fscore:
        test_fscore[taxonomy_id]['fscore'] = np.mean(test_fscore[taxonomy_id]['fscore'], axis=0)
        mean_fscore.append(test_fscore[taxonomy_id]['fscore'] * test_fscore[taxonomy_id]['n_samples'])
    mean_fscore = np.sum(mean_fscore, axis=0) / n_samples

    # Print header for IoU
    print('============================ TEST RESULTS (IoU) ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print(f't={th:.2f}', end='\t')
    print()
    # Print body for IoU
    for taxonomy_id in test_iou:
        print(f'{taxonomies[taxonomy_id]["taxonomy_name"].ljust(8)}', end='\t')
        print(f'{test_iou[taxonomy_id]["n_samples"]}', end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print(f'{taxonomies[taxonomy_id]["baseline"][f"{cfg.CONST.N_VIEWS_RENDERING}-view"]:.4f}', end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print(f'{ti:.4f}', end='\t')
        print()

    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print(f'{mi:.4f}', end='\t')
    print('\n')

    # Print header for F-score
    print('========================== TEST RESULTS (F-score) ==========================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print(f't={th:.2f}', end='\t')
    print()
    # Print body for F-score
    for taxonomy_id in test_fscore:
        print(f'{taxonomies[taxonomy_id]["taxonomy_name"].ljust(8)}', end='\t')
        print(f'{test_fscore[taxonomy_id]["n_samples"]}', end='\t')
        # No baseline for F-score
        print('N/a', end='\t\t')

        for sf in test_fscore[taxonomy_id]['fscore']:
            print(f'{sf:.4f}', end='\t')
        print()

    # Print mean F-score for each threshold
    print('Overall ', end='\t\t\t\t')
    for mf in mean_fscore:
        print(f'{mf:.4f}', end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    max_fscore = np.max(mean_fscore)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)
        test_writer.add_scalar('Refiner/F-score', max_fscore, epoch_idx)

    return max_iou