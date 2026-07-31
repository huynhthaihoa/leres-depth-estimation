# SUPPORTED ENCODERS & DECODERS

*(last updated: May 14th, 2025)*

## Contents
1. [Supported encoders](#1-supported-encoders)

    1.1. [Classification-based](#11-classification-based)

    1.2. [Detection-based](#12-classification-based)

2. [Supported decoders](#2-supported-decoders)

## 1. Supported encoders
**Notes**
- :x: means the encoder is incompatible with TI
- :heavy_check_mark: means the encoder is compatible with TI
- :grey_question: means the TI compatibility has not been checked 
- Detection-based encoders are derived from the corresponding object detection models, whereas other decoders are derived from classification models
- **Reference** column refers to the original source/repo, whereas **Source** column refers to the implementation in this repo
<!-- - We will update the license for each encoder later -->
<!-- Support 5-level features (Y/N)| -->
<!-- ------------:| -->

### 1.1. Classification-based

|Reference|Source|Encoder|TI compatibility|License|
|------|------:|------:|------------:|------:|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb0|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb1|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb2|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb3|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb4|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb5|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb6|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetb7|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b0|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b1|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b2|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b3|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b4|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b5|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b6|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetap_b7|:x:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetlite0|:heavy_check_mark:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetlite1|:heavy_check_mark:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetlite2|:heavy_check_mark:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetlite3|:heavy_check_mark:|Apache-2.0|
|[geffnet](https://github.com/rwightman/gen-efficientnet-pytorch/tree/master/geffnet)|[geffnet_encoder.py](../networks/encoders/geffnet_encoder.py)|efficientnetlite4|:heavy_check_mark:|Apache-2.0|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb0|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb1|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb2|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb3|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb4|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb5|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb6|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|torchefficientnetb7|:x:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|mobilenetv2|:heavy_check_mark:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|mobilenetv3small|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|mobilenetv3large|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnet18|:heavy_check_mark:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnet34|:heavy_check_mark:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnet50|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnet101|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnet152|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnext50_32x4d|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnext101_32x8d|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|resnext101_64x4d|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_400mf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_800mf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_1_6gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_3_2gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_8gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_16gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_y_32gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_400mf|:heavy_check_mark:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_800mf|:heavy_check_mark:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_1_6gf|:heavy_check_mark:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_3_2gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_8gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_16gf|:grey_question:|BSD 3-Clause|
|[torchvision](https://github.com/pytorch/vision)|[torchvision_encoder.py](../networks/encoders/torchvision_encoder.py)|regnet_x_32gf|:grey_question:|BSD 3-Clause|Apple|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|[mobileone_encoder.py](../networks/encoders/mobileone_encoder.py)|mobileone_s0|:x:|Apple|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|[mobileone_encoder.py](../networks/encoders/mobileone_encoder.py)|mobileone_s1|:x:|Apple|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|[mobileone_encoder.py](../networks/encoders/mobileone_encoder.py)|mobileone_s2|:x:|Apple|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|[mobileone_encoder.py](../networks/encoders/mobileone_encoder.py)|mobileone_s3|:x:|Apple|
|[apple ml-mobileone](https://github.com/apple/ml-mobileone)|[mobileone_encoder.py](../networks/encoders/mobileone_encoder.py)|mobileone_s4|:x:|Apple|
|[timm encoder*](https://huggingface.co/docs/timm/feature_extraction)|[timm_encoder.py](../networks/encoders/timm_encoder.py)|-|:grey_question:|Apache-2.0|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|[convnext_encoder.py](../networks/encoders/convnext_encoder.py)|convnext_tiny|:x:|MIT|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|[convnext_encoder.py](../networks/encoders/convnext_encoder.py)|convnext_small|:x:|MIT|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|[convnext_encoder.py](../networks/encoders/convnext_encoder.py)|convnext_base|:x:|MIT|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|[convnext_encoder.py](../networks/encoders/convnext_encoder.py)|convnext_large|:x:|MIT|
|[facebook ConvNeXt](https://github.com/facebookresearch/ConvNeXt)|[convnext_encoder.py](../networks/encoders/convnext_encoder.py)|convnext_xlarge|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_atto|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_femto|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_pico|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_nano|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_tiny|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_base|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_large|:x:|MIT|
|[facebook ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)|[convnextv2_encoder.py](../networks/encoders/convnextv2_encoder.py)|convnextv2_huge|:x:|MIT|
<!-- |[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_t8|:grey_question:|Apple|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_t12|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_s12|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_sa12|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_sa24|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_sa36|:grey_question:|
|[apple ml-fastvit](https://github.com/apple/ml-fastvit)|[fastvit_encoder.py](../networks/encoders/fastvit_encoder.py)|fastvit_ma36|:grey_question:| -->

### 1.2. Detection-based

|Reference|Source|Encoder|TI compatibility|License|
|------|------:|------:|------------:|------:|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|tiyoloxt|:heavy_check_mark:|Apache-2.0|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|tiyoloxn|:heavy_check_mark:|Apache-2.0|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|tiyoloxs|:heavy_check_mark:|Apache-2.0|
|[edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|tiyoloxm|:heavy_check_mark:|Apache-2.0|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|yoloxt|:x:|Apache-2.0|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|yoloxn|:x:|Apache-2.0|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|yoloxs|:x:|Apache-2.0|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|yoloxm|:x:|Apache-2.0|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|yoloxl|:x:|Apache-2.0|
|[Megviii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)|[yolox_encoder.py](../networks/encoders/yolox_encoder.py)|yoloxx|:x:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_ns|:grey_question:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_nm|:grey_question:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_nl|:grey_question:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_t|:grey_question:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_s|:grey_question:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_m|:grey_question:|Apache-2.0|
|[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)|[damoyolo_encoder.py](../networks/encoders/damoyolo_encoder.py)|damoyolo_l|:grey_question:|Apache-2.0|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_s|:x:|Apache-2.0|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_m|:x:|Apache-2.0|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_l|:x:|Apache-2.0|
|[PPYOLOE backbone + neck](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_x|:x:|Apache-2.0|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_s_noattn|:heavy_check_mark:|Apache-2.0|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_m_noattn|:heavy_check_mark:|Apache-2.0|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_l_noattn|:grey_question:|Apache-2.0|
|[PPYOLOE backbone + neck (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_x_noattn|:grey_question:|Apache-2.0|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_s_truncate|:x:|Apache-2.0|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_m_truncate|:x:|Apache-2.0|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_l_truncate|:x:|Apache-2.0|
|[PPYOLOE backbone](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_x_truncate|:x:|Apache-2.0|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_s_noattn_truncate|:heavy_check_mark:|Apache-2.0|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_m_noattn_truncate|:heavy_check_mark:|Apache-2.0|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_l_noattn_truncate|:grey_question:|Apache-2.0|
|[PPYOLOE backbone (removed Squeeze-Excitation block)](https://github.com/Nioolek/PPYOLOE_pytorch)|[ppyoloe_encoder.py](../networks/encoders/ppyoloe_encoder.py)|ppyoloe_x_noattn_truncate|:grey_question:|Apache-2.0|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|yolov5n|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|yolov5s|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|yolov5m|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|yolov5l|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|yolov5x|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt (custom)](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|tiyolov5n|:heavy_check_mark:|AGPL-3.0|
|[Jahongir YOLOv5-pt (custom)](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|tiyolov5s|:heavy_check_mark:|AGPL-3.0|
|[Jahongir YOLOv5-pt (custom)](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|tiyolov5m|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt (custom)](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|tiyolov5l|:x:|AGPL-3.0|
|[Jahongir YOLOv5-pt (custom)](https://github.com/jahongir7174/YOLOv5-pt)|[yolov5_encoder.py](../networks/encoders/nets/yolov5_encoder.py)|tiyolov5x|:x:|AGPL-3.0|
|[ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov5n|:x:|AGPL-3.0|
|[ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov5s|:x:|AGPL-3.0|
|[ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov5m|:x:|AGPL-3.0|
|[ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov5l|:x:|AGPL-3.0|
|[ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov5x|:x:|AGPL-3.0|
|[ultralytics YOLOv5 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov5n|:x:|AGPL-3.0|
|[ultralytics YOLOv5 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov5s|:x:|AGPL-3.0|
|[ultralytics YOLOv5 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov5m|:x:|AGPL-3.0|
|[ultralytics YOLOv5 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov5l|:x:|AGPL-3.0|
|[ultralytics YOLOv5 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov5x|:x:|AGPL-3.0|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6n|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6s|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6m|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P5](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6l|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6n6|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6s6|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6m6|:grey_question:|GPL-3.0|
|[meituan YOLOv6 P6](https://github.com/meituan/YOLOv6)|[yolov6_encoder.py](../networks/encoders/yolov6_encoder.py)|yolov6l6|:grey_question:|GPL-3.0|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|[yolov6lite_encoder.py](../networks/encoders/yolov6lite_encoder.py)|yolov6lites|:grey_question:|GPL-3.0|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|[yolov6lite_encoder.py](../networks/encoders/yolov6lite_encoder.py)|yolov6litem|:grey_question:|GPL-3.0|
|[meituan YOLOv6 lite](https://github.com/meituan/YOLOv6)|[yolov6lite_encoder.py](../networks/encoders/yolov6lite_encoder.py)|yolov6litel|:grey_question:|GPL-3.0|
|[ultralytics YOLOv6](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov6n|:x:|AGPL-3.0|
|[ultralytics YOLOv6](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov6s|:x:|AGPL-3.0|
|[ultralytics YOLOv6](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov6m|:x:|AGPL-3.0|
|[ultralytics YOLOv6](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov6l|:x:|AGPL-3.0|
|[ultralytics YOLOv6](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov6x|:x:|AGPL-3.0|
|[ultralytics YOLOv6 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov6n|:x:|AGPL-3.0|
|[ultralytics YOLOv6 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov6s|:x:|AGPL-3.0|
|[ultralytics YOLOv6 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov6m|:x:|AGPL-3.0|
|[ultralytics YOLOv6 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov6l|:x:|AGPL-3.0|
|[ultralytics YOLOv6 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov6x|:x:|AGPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7x|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7-w6|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7-tiny|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7-e6e|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7-e6|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main)|[yolov7_encoder.py](../networks/encoders/yolov7_encoder.py)|yolov7-d6|:grey_question:|GPL-3.0|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|yolov8n|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|yolov8s|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|yolov8m|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|yolov8l|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|yolov8x|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt (custom)](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|tiyolov8n|:heavy_check_mark:|AGPL-3.0|
|[Jahongir YOLOv8-pt (custom)](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|tiyolov8s|:heavy_check_mark:|AGPL-3.0|
|[Jahongir YOLOv8-pt (custom)](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|tiyolov8m|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt (custom)](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|tiyolov8l|:x:|AGPL-3.0|
|[Jahongir YOLOv8-pt (custom)](https://github.com/jahongir7174/YOLOv8-pt)|[yolov8_encoder.py](../networks/encoders/nets/yolov8_encoder.py)|tiyolov8x|:x:|AGPL-3.0|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov8n|:x:|AGPL-3.0|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov8s|:x:|AGPL-3.0|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov8m|:x:|AGPL-3.0|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov8l|:x:|AGPL-3.0|
|[ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov8x|:x:|AGPL-3.0|
|[ultralytics YOLOv8 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov8n|:x:|AGPL-3.0|
|[ultralytics YOLOv8 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov8s|:x:|AGPL-3.0|
|[ultralytics YOLOv8 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov8m|:x:|AGPL-3.0|
|[ultralytics YOLOv8 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov8l|:x:|AGPL-3.0|
|[ultralytics YOLOv8 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov8x|:x:|AGPL-3.0|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|[yolov9_encoder.py](../networks/encoders/yolov9_encoder.py)|yolov9-t|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|[yolov9_encoder.py](../networks/encoders/yolov9_encoder.py)|yolov9-s|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|[yolov9_encoder.py](../networks/encoders/yolov9_encoder.py)|yolov9-c|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|[yolov9_encoder.py](../networks/encoders/yolov9_encoder.py)|yolov9-e|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|[yolov9_encoder.py](../networks/encoders/yolov9_encoder.py)|gelan-c|:grey_question:|GPL-3.0|
|[WongKinYiu YOLOv9](https://github.com/WongKinYiu/yolov9)|[yolov9_encoder.py](../networks/encoders/yolov9_encoder.py)|gelan-e|:grey_question:|GPL-3.0|
|[ultralytics YOLOv9](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov9c|:x:|AGPL-3.0|
|[ultralytics YOLOv9](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_yolov9e|:x:|AGPL-3.0|
|[ultralytics YOLOv9 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov9c|:x:|AGPL-3.0|
|[ultralytics YOLOv9 (custom)](https://github.com/ultralytics/ultralytics)|[ultralytics_encoder.py](../networks/encoders/ultralytics_encoder.py)|ultralytics_tiyolov9e|:x:|AGPL-3.0|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|yolov10n|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|yolov10s|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|yolov10m|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|yolov10l|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|yolov10x|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10 (custom)](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|tiyolov10n|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10 (custom)](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|tiyolov10s|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10 (custom)](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|tiyolov10m|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10 (custom)](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|tiyolov10l|:grey_question:|AGPL-3.0|
|[THU-MIG YOLOv10 (custom)](https://github.com/THU-MIG/yolov10)|[yolov10_encoder.py](../networks/encoders/yolov10_encoder.py)|tiyolov10x|:grey_question:|AGPL-3.0|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|[yolov11_encoder.py](../networks/encoders/yolov11_encoder.py)|yolov11n|:grey_question:|AGPL-3.0|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|[yolov11_encoder.py](../networks/encoders/yolov11_encoder.py)|yolov11s|:grey_question:|AGPL-3.0|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|[yolov11_encoder.py](../networks/encoders/yolov11_encoder.py)|yolov11m|:grey_question:|AGPL-3.0|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|[yolov11_encoder.py](../networks/encoders/yolov11_encoder.py)|yolov11l|:grey_question:|AGPL-3.0|
|[Jahongir YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)|[yolov11_encoder.py](../networks/encoders/yolov11_encoder.py)|yolov11x|:grey_question:|AGPL-3.0|
|[sunsmarterjie YOLOv12](https://github.com/sunsmarterjie/yolov12/tree/main)|[yolov12_encoder.py](../networks/encoders/yolov12_encoder.py)|yolov12n|:grey_question:|AGPL-3.0|
|[sunsmarterjie YOLOv12](https://github.com/sunsmarterjie/yolov12/tree/main)|[yolov12_encoder.py](../networks/encoders/yolov12_encoder.py)|yolov12s|:grey_question:|AGPL-3.0|
|[sunsmarterjie YOLOv12](https://github.com/sunsmarterjie/yolov12/tree/main)|[yolov12_encoder.py](../networks/encoders/yolov12_encoder.py)|yolov12m|:grey_question:|AGPL-3.0|
|[sunsmarterjie YOLOv12](https://github.com/sunsmarterjie/yolov12/tree/main)|[yolov12_encoder.py](../networks/encoders/yolov12_encoder.py)|yolov12l|:grey_question:|AGPL-3.0|
|[sunsmarterjie YOLOv12](https://github.com/sunsmarterjie/yolov12/tree/main)|[yolov12_encoder.py](../networks/encoders/yolov12_encoder.py)|yolov12x|:grey_question:|AGPL-3.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r18vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r34vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r50vd_truncate|23,474,016|4.35|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r101vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r18vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r34vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r50vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r101vd_truncate|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r18vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r34vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r50vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|rtdetr_r101vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder (pretrained on COCO + Object365 dataset)](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r18vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder (pretrained on COCO + Object365 dataset)](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r34vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder (pretrained on COCO + Object365 dataset)](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r50vd|:grey_question:|Apache-2.0|
|[RT-DETR PResNet backbone + HybridEncoder (pretrained on COCO + Object365 dataset)](https://github.com/lyuwenyu/RT-DETR)|[rtdetr_encoder.py](../networks/encoders/rtdetr_encoder.py)|cocoobject365_rtdetr_r101vd|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_hgnetv2_b0|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_hgnetv2_b2|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_hgnetv2_b4|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_hgnetv2_b5|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_hgnetv2_b0|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_hgnetv2_b2|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_hgnetv2_b4|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_hgnetv2_b5|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO + Object365  dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b0|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO + Object365  dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b2|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO + Object365  dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b4|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone (pretrained on COCO + Object365  dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_hgnetv2_b5|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_dfine_s|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_dfine_m|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_dfine_l|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|coco_dfine_x|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_dfine_s|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_dfine_m|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_dfine_l|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|object365_dfine_x|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO + Object365  dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_dfine_s|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO + Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_dfine_m|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO + Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_dfine_l|:grey_question:|Apache-2.0|
|[D-FINE HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO + Object365 dataset)](https://github.com/Peterande/D-FINE)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|cocoobject365_dfine_x|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|deim_hgnetv2_b0|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|deim_hgnetv2_b2|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|deim_hgnetv2_b4|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[hgnetv2_encoder.py](../networks/encoders/hgnetv2_encoder.py)|deim_hgnetv2_b5|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[deim_encoder.py](../networks/encoders/deim_encoder.py)|deim_s|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[deim_encoder.py](../networks/encoders/deim_encoder.py)|deim_m|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[deim_encoder.py](../networks/encoders/deim_encoder.py)|deim_l|:grey_question:|Apache-2.0|
|[DEIM HGNetv2 backbone + HybridEncoder encoder (pretrained on COCO dataset)](https://github.com/ShihuaHuang95/DEIM)|[deim_encoder.py](../networks/encoders/deim_encoder.py)|deim_x|:grey_question:|Apache-2.0|

## 2. Supported decoders
- [dpt](../networks/decoders/dpt_decoder.py)
- [dpttruncate](../networks/decoders/dpt_decoder.py)
- [dpt2](../networks/decoders/dpt_decoder.py)
- [dpt2truncate](../networks/decoders/dpt_decoder.py)
- [dptdeconv](../networks/decoders/dptdeconv_decoder.py)
- [dptdeconvtruncate](../networks/decoders/dptdeconv_decoder.py)
- [dptdeconv2](../networks/decoders/dptdeconv_decoder.py)
- [dptdeconv2truncate](../networks/decoders/dptdeconv_decoder.py)
- [fpn](../networks/decoders/fpn_decoder.py)
- [fpntruncate](../networks/decoders/fpn_decoder.py)
- [fpn2](../networks/decoders/fpn_decoder.py)
- [fpn2truncate](../networks/decoders/fpn_decoder.py)
- [fpndeconv](../networks/decoders/fpndeconv_decoder.py)
- [fpndeconvupsample](../networks/decoders/fpndeconv_decoder.py)
- [fpndeconvinterpolate](../networks/decoders/fpndeconv_decoder.py)
- [metricleres](../networks/decoders/metricleres_decoder.py)
- [metriclerestruncate](../networks/decoders/metriclerestruncate_decoder.py)
- [metricleres2](../networks/decoders/metricleres2_decoder.py)
- [metricleres2truncate](../networks/decoders/metricleres2truncate_decoder.py)
- [metricleresdeconv](../networks/decoders/metricleresdeconv_decoder.py)
- [metricleresdeconvupsample](../networks/decoders/metricleresdeconvupsample_decoder.py)
- [metricleresdeconvupsampletruncate](../networks/decoders/metricleresdeconvupsampletruncate_decoder.py)
- [newcrfs](../networks/decoders/newcrfs_decoder.py)
- [newcrfstruncate](../networks/decoders/newcrfstruncate_decoder.py)
- [newcrfs2](../networks/decoders/newcrfs2_decoder.py)
- [newcrfs2truncate](../networks/decoders/newcrfs2truncate_decoder.py)
- [newcrfsdpt](../networks/decoders/newcrfsdpt_decoder.py)
- [newcrfsdpttruncate](../networks/decoders/newcrfsdpt_decoder.py)
- [newcrfsdptdeconv](../networks/decoders/newcrfsdptdeconv_decoder.py)
- [newcrfs2dpt](../networks/decoders/newcrfsdpt_decoder.py)
- [newcrfs2dpttruncate](../networks/decoders/newcrfsdpt_decoder.py)
- [newcrfsdeconv](../networks/decoders/newcrfsdeconv_decoder.py)
- [newcrfsdeconvupsample](../networks/decoders/newcrfsdeconvupsample_decoder.py)
- [newcrfsdeconvupsampletruncate](../networks/decoders/newcrfsdeconvupsampletruncate.py)
- [pixelformer](../networks/decoders/pixelformer_decoder.py)
- [pixelformerdeconv](../networks/decoders/pixelformerdeconv_decoder.py)
- [pixelformerdeconvupsample](../networks/decoders/pixelformerdeconvupsample_decoder.py)
<!-- *TBA* -->