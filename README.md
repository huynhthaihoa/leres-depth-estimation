# Metric LeReS Depth Estimation

A PyTorch implementation of metric monocular depth estimation based on the [LeReS](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS) architecture.

Given a single RGB image, the model predicts an absolute depth value in meters for every pixel. The repository supports model training, evaluation, inference, quantization, and deployment through ONNX and TFLite.

## Features

- Metric monocular depth estimation
- Training and evaluation on NYUv2 and KITTI
- Multiple encoder families, including:
  - EfficientNet
  - ResNet and RegNet
  - ConvNeXt
  - DINOv2
  - YOLO-based backbones
  - RT-DETR, D-FINE, DEIM, and others
- Multiple decoder families, including:
  - MetricLeReS
  - NeWCRFs
  - DPT
  - FPN
  - PixelFormer
- PyTorch, ONNX, and TFLite inference
- Image, video, and webcam inference
- Optional test-time augmentation
- Automatic mixed precision and multi-GPU training
- Quantization-aware training utilities
- TI/TIDL-oriented deployment support

## Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Pretrained models](#pretrained-models)
- [Inference](#inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model conversion](#model-conversion)
- [Supported encoders and decoders](#supported-encoders-and-decoders)
- [Repository structure](#repository-structure)
- [Known limitations](#known-limitations)
- [References](#references)
- [License](#license)

## Architecture

The model follows an encoder-decoder design:

1. An encoder extracts multi-scale image features.
2. A depth decoder combines those features.
3. The decoder produces a dense depth map.
4. The output is scaled by the configured maximum depth.

![MetricLeReS architecture](assets/MetricLeReS.png "MetricLeReS architecture")

The main model is implemented in [`networks/MetricLeRes.py`](networks/MetricLeRes.py).

The architecture is configurable through the `encoder` and `decoder` values in the YAML configuration files.

## Requirements

The repository provides Conda environments for standard PyTorch usage and TI-oriented workflows.

The TI environment uses Python 3.8, PyTorch 2.0.1, CUDA 11.8, and torchvision 0.15.2.

```bash
conda env create -f envs/ti_env.yml
conda activate ti
```

For a standard environment:

```bash
conda env create -f envs/standard_env.yml
conda activate <environment-name>
```

A CUDA-capable GPU is recommended for training and PyTorch inference. ONNX CPU inference is also supported through ONNX Runtime.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/huynhthaihoa/leres-depth-estimation.git
cd leres-depth-estimation
```

### 2. Create the environment

```bash
conda env create -f envs/ti_env.yml
conda activate ti
```

### 3. Download a pretrained model

Pretrained checkpoints and exported models are listed in the [model zoo](assets/modelzoo.md).

### 4. Run inference

Update `configs/infer.yml` with the input path and model path, then run:

```bash
python infer.py -c configs/infer.yml
```

For a single image, the configuration can contain:

```yaml
input: path/to/image.jpg
weight_path: path/to/model.ckpt
normalize: False
use_cuda: True
use_tta: False
show: True
output_depth_path: outputs/depth.npy
output_image_path: outputs/depth.png
```

The input may be:

- An image: `.jpg`, `.jpeg`, `.png`, or `.bmp`
- A video: `.mp4`, `.avi`, `.mkv`, or `.mov`
- A camera index such as `0`

For video or webcam inference, press `q` to stop the stream.

## Pretrained models

Available PyTorch and ONNX models are documented in the [model zoo](assets/modelzoo.md).

The model zoo includes models trained for:

- NYUv2 with a maximum depth of 10 meters
- KITTI with a maximum depth of 80 meters
- NYUv2 grayscale input
- ImageNet-normalized and non-normalized inputs
- PyTorch and ONNX deployment

When using a pretrained model, make sure the inference configuration matches the model's preprocessing settings:

- `normalize`
- `to_grayscale`
- `input_height`
- `input_width`
- `max_depth`

## Inference

The `infer.py` script supports PyTorch checkpoints, ONNX models, and TFLite models.

### PyTorch inference

```bash
python infer.py \
    --input path/to/image.jpg \
    --weight_path path/to/model.ckpt \
    --use_cuda \
    --show
```

### ONNX inference

```bash
python infer.py \
    --input path/to/image.jpg \
    --weight_path path/to/model.onnx \
    --show
```

### TFLite inference

```bash
python infer.py \
    --input path/to/image.jpg \
    --weight_path path/to/model.tflite \
    --show
```

### Save the raw depth map and visualization

```bash
python infer.py \
    --input path/to/image.jpg \
    --weight_path path/to/model.onnx \
    --output_depth_path outputs/depth.npy \
    --output_image_path outputs/depth.png
```

The raw depth output is saved as a NumPy array. Each element represents the estimated distance from the camera in meters.

### ONNX integration

For integrating ONNX inference into another Python application, use:

```python
from networks.ONNXDepthModel import ONNXDepthModel
import cv2

model = ONNXDepthModel(
    weight_path="path/to/model.onnx",
    use_cuda=False,
    normalize=False,
    use_tta=False,
)

image = cv2.imread("image.jpg")
depth = model(image)
```

The returned depth map is a floating-point array with the same spatial dimensions as the input image.

See the complete [inference and integration guide](assets/infer.md).

## Training

Training is configured through YAML files in [`configs/`](configs/).

Available examples include:

- [`configs/train_nyu.yml`](configs/train_nyu.yml)
- [`configs/train_kitti.yml`](configs/train_kitti.yml)
- [`configs/train_nyu_normalize.yml`](configs/train_nyu_normalize.yml)
- [`configs/train_kitti_normalize.yml`](configs/train_kitti_normalize.yml)

Before training, update at least:

- `data_path`
- `data_path_eval`
- `filenames_file`
- `filenames_file_eval`
- `log_dir`
- `encoder`
- `decoder`
- `input_height`
- `input_width`
- `max_depth`

The provided configuration files contain machine-specific dataset and output paths. Replace those paths with locations on your system.

### Train on NYUv2

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    -c configs/train_nyu.yml \
    --comment nyu_metric_depth
```

### Train on KITTI

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    -c configs/train_kitti.yml \
    --comment kitti_metric_depth
```

The training script supports:

- Automatic mixed precision
- Data parallel and distributed training
- Early stopping
- Exponential moving average
- Quantization-aware training
- TensorBoard or DVC logging
- Multiple optimizers and loss functions
- Optional data augmentation

Training outputs, checkpoints, and logs are written below the configured `log_dir`.

### Monitor training with TensorBoard

```bash
tensorboard \
    --logdir path/to/log_dir/model_name/timestamp/train \
    --port 3003
```

To monitor evaluation metrics:

```bash
tensorboard \
    --logdir path/to/log_dir/model_name/timestamp/eval \
    --port 3003
```

See the complete [training guide](assets/train.md).

## Evaluation

Evaluation supports PyTorch, ONNX, and TFLite models.

### Evaluate a PyTorch checkpoint

For NYUv2:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    -c configs/eval_nyu.yml
```

For KITTI:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    -c configs/eval_kitti.yml
```

Set `weight_path` and `data_path_eval` in the selected configuration before running evaluation.

### Evaluate an ONNX model

```bash
CUDA_VISIBLE_DEVICES=0 python eval_onnx.py \
    -c configs/eval_nyu.yml
```

### Evaluate a TFLite model

```bash
CUDA_VISIBLE_DEVICES=0 python eval_tflite.py \
    -c configs/eval_nyu.yml
```

The evaluation pipeline reports common depth-estimation metrics, including:

- SILog
- Absolute relative error
- Log10 error
- RMSE
- Squared relative error
- Log RMSE
- Accuracy under threshold δ1, δ2, and δ3

Batch evaluation scripts are also available:

```text
eval_batch.py
eval_batch_onnx.py
eval_batch_tflite.py
```

See the complete [evaluation guide](assets/eval.md).

## Model conversion

> For TI-compatible deployment, use the TI Conda environment whenever possible.

### Convert a single PyTorch checkpoint to ONNX

Update:

```text
configs/convert_onnx.yml
```

Important parameters include:

- `weight_path`
- `output_path`
- `input_width`
- `input_height`
- `simplify`

Input dimensions must be divisible by 32.

Run:

```bash
python convert_onnx.py \
    -c configs/convert_onnx.yml
```

### Convert multiple checkpoints to ONNX

Update:

```text
configs/convert_batch_onnx.yml
```

Then run:

```bash
python convert_batch_onnx.py \
    -c configs/convert_batch_onnx.yml
```

Each checkpoint is converted to an ONNX file with the corresponding filename.

### Convert a single model to TFLite

Update:

```text
configs/convert_tflite.yml
```

Then run:

```bash
python convert_tflite.py \
    -c configs/convert_tflite.yml
```

### Convert multiple models to TFLite

Update:

```text
configs/convert_batch_tflite.yml
```

Then run:

```bash
python convert_batch_tflite.py \
    -c configs/convert_batch_tflite.yml
```

## Quantization and edge deployment

The [`xnn/`](xnn/) directory contains quantization utilities and documentation related to post-training quantization and quantization-aware training.

Supported workflows include:

- Floating-point PyTorch training
- Quantization-aware training
- ONNX export
- TFLite conversion
- TI/TIDL-oriented deployment preparation

Additional documentation is available in:

- [`xnn/README.md`](xnn/README.md)
- [`xnn/quantization/README.md`](xnn/quantization/README.md)
- [`xnn/quantization/docs/`](xnn/quantization/docs/)

## Supported encoders and decoders

The repository supports a large collection of classification- and detection-based encoders, along with several depth decoders.

For the complete list, compatibility notes, source implementations, and licenses, see:

- [Supported encoders and decoders](assets/encoders_decoders.md)

The main encoder and decoder selection is made in the model configuration:

```yaml
encoder: tiyolov8s
decoder: metricleres
```

Common decoder families include:

```text
metricleres
newcrfs
dpt
fpn
pixelformer
```

## Repository structure

```text
.
├── assets/                 Documentation, diagrams, and model references
├── configs/                Training, evaluation, inference, and export configs
├── data_splits/            NYUv2 and KITTI split files
├── dataloaders/            Dataset loading and preprocessing
├── envs/                   Conda environment definitions
├── lora/                   LoRA injection utilities
├── networks/               Encoders, decoders, and model wrappers
├── optimizer/              Additional optimizer implementations
├── xnn/                    Quantization and deployment utilities
├── convert_onnx.py         Single-model ONNX conversion
├── convert_tflite.py       Single-model TFLite conversion
├── eval.py                PyTorch evaluation
├── infer.py               Image, video, and webcam inference
├── train.py               Training entry point
└── utils.py                Shared preprocessing, losses, and metrics
```

## Known limitations

- Input height and width should be divisible by 32 for model conversion.
- The preprocessing configuration must match the preprocessing used during training.
- Some encoders are not compatible with TI deployment. See the compatibility table in [`assets/encoders_decoders.md`](assets/encoders_decoders.md).
- The example YAML files contain environment-specific dataset and output paths that must be updated.
- Larger encoders and decoders require significantly more GPU memory.
- Test-time augmentation may improve accuracy but approximately doubles inference work.
- The model zoo contains results from specific environments and hardware; use them as reference values rather than universal benchmarks.
- Some supported backbones are derived from external projects with different licenses. Review the relevant license before redistributing a model or deployment package.

## References

- [MetricLeReS](https://github.com/DeltaX-AI-Lab/MetricLeReS)
- [LeReS](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS)
- [NeWCRFs](https://github.com/aliyun/NeWCRFs)
- [PixelFormer](https://github.com/ashutosh1807/PixelFormer)
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)
- [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch/)
- [TI Edge AI tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

## License

See [`LICENSE`](LICENSE).

This repository integrates or derives components from multiple upstream projects. Refer to the upstream projects and the compatibility table for their respective licenses and attribution requirements.
