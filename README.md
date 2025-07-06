[![CodeFactor](https://www.codefactor.io/repository/github/sandeepainduwarasamaranayake/swinvox/badge)](https://www.codefactor.io/repository/github/sandeepainduwarasamaranayake/swinvox)

# SwinVox


This repository contains the source code for SwinVox, a novel deep learning model for reconstructing 3D voxel-based shapes from multiple 2D input images (views). Building upon the foundations of the Pix2Vox++ (https://gitlab.com/hzxie/Pix2Vox) architecture, SwinVox integrates the powerful Swin Transformer (https://github.com/microsoft/Swin-Transformer) for robust feature extraction and introduces an optional  Cross-View Attention mechanism to enhance the fusion of multi-view information. This project aims to improve the fidelity and detail of 3D reconstructions, particularly in complex scenarios.

![Algorithm_Design drawio](https://github.com/user-attachments/assets/e102be2a-d767-4719-bc4d-d979595e4186)

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) dataset in our experiments, which are available below:

### Extracted (Recommended)
- [ShapeNet rendering images & voxelized models](https://www.kaggle.com/api/v1/datasets/download/gabrielescognamiglio/shapenet)

### Archieved (Alternative)
- [ShapeNet rendering images & voxelized models](https://www.kaggle.com/api/v1/datasets/download/sirish001/shapenet-3dr2n2)

### Separate Rendering images & voxelized models (Extracted | Alternative)
- [ShapeNet rendering images](https://www.kaggle.com/api/v1/datasets/download/ronak555/shapenetcorerendering-part1)
- [ShapeNet voxelized models](https://www.kaggle.com/api/v1/datasets/download/ronak555/shapenetvox32)



## Pretrained Models

The pretrained models on ShapeNet are available as follows:

- [SwinVox-A](https://gateway.infinitescript.com/?fileName=Pix2Vox-A-ShapeNet.pth) (457.0 MB)

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/SandeepaInduwaraSamaranayake/SwinVox.git
```

#### Install Python Denpendencies

```
cd SwinVox
pip install -r requirements.txt
```

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/path/to/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/path/to/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/path/to/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/path/to/Datasets/Pix3D/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/path/to/Datasets/Pix3D/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/path/to/Datasets/Pix3D/model/%s/%s/%s.binvox'
```

## Get Started

To train SwinVox, you can simply use the following command:

```
python3 runner.py
```

To test SwinVox, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```

## License

This project is open sourced under MIT license.
