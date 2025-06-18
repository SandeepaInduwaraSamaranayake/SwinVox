# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified  by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()

# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet.json'                                                                                  # for colab original json file
#__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet_aeroplane_category.json'                                                                 # for colab test json file
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = '/kaggle/working/SwinVox/datasets/ShapeNet_aeroplane_category.json'                                          # for kaggle

# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/PascalShapeNet.json'

#__C.DATASETS.SHAPENET.RENDERING_PATH        = '/content/ShapeNetRendering/%s/%s/rendering/%02d.png'                                                         # for colab
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/kaggle/input/shapenet/ShapeNetRendering/ShapeNetRendering/%s/%s/rendering/%02d.png'                          # for kaggle

# __C.DATASETS.SHAPENET.RENDERING_PATH      = '/home/hzxie/Datasets/ShapeNet/PascalShapeNetRendering/%s/%s/render_%04d.jpg'

#__C.DATASETS.SHAPENET.VOXEL_PATH            = '/content/ShapeNetVox32/%s/%s/model.binvox'                                                                   # for colab
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/kaggle/input/shapenet/ShapeNetVox32/ShapeNetVox32/%s/%s/model.binvox'                                      # for kaggle

__C.DATASETS.PASCAL3D                       = edict()
__C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D                          = edict()
__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/Pix3D.json'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/content/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/content/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/content/model/%s/%s/%s.binvox'


#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
# __C.DATASET.TEST_DATASET                  = 'Pascal3D'
# __C.DATASET.TEST_DATASET                  = 'Pix3D'


#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.BATCH_SIZE                        = 64        # default is 64. 
__C.CONST.N_VIEWS_RENDERING                 = 1         
__C.CONST.CROP_IMG_W                        = 128       
__C.CONST.CROP_IMG_H                        = 128       
__C.CONST.NUM_WORKER                        = 4         # number of data workers -- suggested max is 2, but already initialized 4 as default.
__C.CONST.WEIGHTS                           = '/content/drive/MyDrive/Colab Git Clones/SwinVox/SwinVox/output/checkpoints/2025-06-13T16:07:22.579164/checkpoint-best.pth'   # if training is resuming, uncomment this replace 'path' with weight path.


#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'


#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True

__C.NETWORK.USE_SWIN_T_MULTI_STAGE          = True
__C.NETWORK.SWIN_T_STAGES                   = [0,1,2,3]          # Single or multiple stage(s) [0, 1, 2, 3]. -1 for only final stage
__C.NETWORK.USE_CROSS_VIEW_ATTENTION        = True
__C.NETWORK.CROSS_ATT_REDUCTION_RATIO       = 4
__C.NETWORK.ATT_SPATIAL_DOWNSAMPLE_RATIO    = 2
__C.NETWORK.CROSS_ATT_NUM_HEADS             = 4


#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_EPOCHS                        = 10           # default 250
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 7.5e-4          # default is 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 7.5e-4          # default is 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 7.5e-5          # default is 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 7.5e-5          # default is 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False


#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
