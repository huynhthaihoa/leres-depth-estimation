# Model zoo
*Latest update: February 19th, 2024*

1. [NYUv2 dataset (10m)](#1-nyuv2-dataset-10m)
    
    1.1. [With ImageNet normalization](#11-with-imagenet-normalization)

    1.1.1. [Pytorch](#111-pytorch)
    
    1.1.2. [ONNX](#112-onnx)

    1.2. [Without ImageNet normalization](#12-without-imagenet-normalization)
    
    1.2.1. [Pytorch](#121-pytorch)
    
    1.2.2. [ONNX](#122-onnx)
2. [KITTI dataset (80m)](#2-kitti-dataset-80m)
    
    2.1. [With ImageNet normalization](#21-with-imagenet-normalization)
    
    2.1.1. [Pytorch](#211-pytorch)
    
    2.1.2. [ONNX](#212-onnx)
    
    2.2. [Without ImageNet normalization](#22-without-imagenet-normalization)
    
    2.2.1. [Pytorch](#221-pytorch)
    
    2.2.2. [ONNX](#222-onnx)
3. [NYUv2 grayscale dataset (10m)](#3-nyuv2-grayscale-dataset-10m)
    
    3.1. [With ImageNet normalization](#31-with-imagenet-normalization)

    3.1.1. [Pytorch](#311-pytorch)
    
    3.1.2. [ONNX](#312-onnx)

    3.2. [Without ImageNet normalization](#32-without-imagenet-normalization)
    
    3.2.1. [Pytorch](#121-pytorch)
    
    3.2.2. [ONNX](#122-onnx)
    
**Notes**: 
- These models were evaluated without using test-time augmentation.
- The ONNX models were made using [our Pytorch-to-ONNX conversion tool](../README.md#7-converting-Pytorch-to-ONNX) with `simplified` set as `True`.
- The FPS (frame per second) was calculated by taking average FPS of 10000 inference iterations using Venus's GPU 2.
- The arrow symbol on each metric shows the desired direction: ↓ means smaller value is better, ↑ means larger value is better.
- When using with-ImageNet-normalization model, please set `normalize` as `True`, otherwise is `False`

## 1. NYUv2 dataset (10m)

### 1.1. With ImageNet normalization

#### 1.1.1. Pytorch
<!-- |[efficientnetb4 (with neck)](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.3.0-beta/nyu_efficientnetb4_base.ckpt)|26,645,689|57.02|0.1044|0.3701|0.0549|0.9008|0.9858|0.9975| -->
|Encoder|Number of params|FPS (480 $\times$ 640)↑|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[efficientnetb4](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.3.0-beta/nyu_efficientnetb4_lite.ckpt)|18,888,745|64.77|0.1079|0.3774|0.0561|0.8982|0.9862|0.9973|
|[efficientnetb0](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.3.1/nyu_efficientnetb0_lite_batch16.ckpt)|4,731,845|94.97|0.1300|0.4490|0.0787|0.8353|0.9731|0.9955|
|[tiyolov8m](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/tiyolov8m_nyu_1018/tiyolov8m_nyu_1018.ckpt)|25,776,145|74.83|0.0956|0.3467|0.0482|0.9130|0.9870|0.9977|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8s_nyu.ckpt)|10,810,465|104.96|0.1056|0.3752|0.0599|0.8907|0.9821|0.9954|
|[tiyolov8n](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8n_nyu.ckpt)|2,708,017|98.79|0.1189|0.4050|0.0682|0.8636|0.9785|0.9961|
|[regnet_y_3_2gf](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/regnet_y_3_2gf_nyu.ckpt)|21,635,875|60.22|0.1259|0.4227|0.0731|0.8531|0.9769|0.9956|
|[regnet_x_3_2gf](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/regnet_x_3_2gf_nyu.ckpt)|18,771,041|74.18|0.1160|0.3979|0.0643|0.8713|0.9809|0.9966|
|[efficientnetlite4](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/efficientnetlite4_nyu.ckpt)|12,770,785|112.05|0.1184|0.4096|0.0678|0.8666|0.9797|0.9962|
|[resnet50](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/resnet50_nyu.ckpt)|52,126,529|63.99|0.1180|0.4087|0.0707|0.8693|0.9779|0.9956|
|[resnet34](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/resnet34_nyu.ckpt)|23,075,585|174.10|0.1166|0.4057|0.0688|0.8712|0.9786|0.9956|
|[resnet18](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/resnet18_nyu.ckpt)|12,967,425|223.17|0.1299|0.4428|0.0809|0.8446|0.9728|0.9944|
|[*NeWCRFs*](https://github.com/aliyun/NeWCRFs)|*270,444,877*|*14.32*|*0.0958*|*0.3331*|*0.0451*|*0.9215*|*0.9915*|*0.9980*|
|[*PixelFormer*](https://github.com/ashutosh1807/PixelFormer)|*270,895,920*|*15.59*|*0.0905*|*0.3242*|*0.0435*|*0.9289*|*0.9906*|*0.9977*|

#### 1.1.2. ONNX
<!-- |[efficientnetb4 with neck](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.3.0-beta/nyu_efficientnetb4_base.onnx)|26,577,830|480 $\times$ 640|0.1044|0.3708|0.0551|0.9004|0.9858|0.9975| -->
|Encoder|Number of params|Input size|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[efficientnetb4](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.3.0-beta/nyu_efficientnetb4_lite.onnx)|18,825,606|480 $\times$ 640|0.1078|0.3779|0.0562|0.898|0.9862|0.9973|
|[efficientnetb0](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/efficientnetb0lite_nyu_upgrade/nyu_efficientnetb0_lite_double.onnx)|4,710,434|480 $\times$ 640|0.1319|0.4493|0.0805|0.8334|0.9730|0.9953|
|[tiyolov8m](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/tiyolov8m_nyu_1018/tiyolov8m_nyu_1018.onnx)|25,759,494|480 $\times$ 640|0.1016|0.3643|0.0529|0.9004|0.9850|0.9974|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8s_nyu.onnx)|10,800,518|480 $\times$ 640|0.1149|0.4053|0.0689|0.8653|0.9760|0.9945|
|[tiyolov8n](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8n_nyu.onnx)|2,703,046|480 $\times$ 640|0.1285|0.4380|0.0796|0.8402|0.9707|0.9941|
|[regnet_y_3_2gf](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/regnet_y_3_2gf_nyu.onnx)|21,601,576|480 $\times$ 640|0.1396|0.4655|0.0876|0.8166|0.9688|0.9944|
|[regnet_x_3_2gf](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/regnet_x_3_2gf_nyu.onnx)|18,738,134|480 $\times$ 640|0.1289|0.4492|0.0787|0.8340|0.9717|0.9958|
|[resnet50](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/resnet50_nyu.onnx)|52,095,622|480 $\times$ 640|0.1329|0.4633|0.0865|0.8259|0.9661|0.9923|
|[resnet34](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/resnet34_nyu.onnx)|23,065,990|480 $\times$ 640|0.1310|0.4548|0.0844|0.8341|0.9690|0.9933|
|[resnet18](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/resnet18_nyu.onnx)|12,961,542|480 $\times$ 640|0.1466|0.4967|0.0992|0.7957|0.9598|0.9925|
|[efficientnetlite4](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-alpha/efficientnetlite4_nyu.onnx)|12,714,064|480 $\times$ 640|0.1297|0.4443|0.0794|0.8374|0.9731|0.9952|
|[YOLOv8 small (Ammar)](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/tiboard-tbv/yolov8_depth_TI_sim.onnx)|10,800,518|480 $\times$ 640|0.1421|0.4749|0.0929|0.8125|0.9654|0.9921|
|[YOLOv8 large (Ammar)](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/tiboard-tbv/yolov8_large_dept_sim.onnx)|44,001,542|480 $\times$ 640|0.1110|0.3912|0.0629|0.8774|0.9800|0.9958|
|[*NeWCRFs*](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/extra/newcrfs_nyu.onnx)|271,971,053|480 $\times$ 640|0.0958|0.3337|0.0453|0.9210|0.9914|0.9980|
|[*PixelFormer*](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/extra/pixelformer_nyu.onnx)|261,458,134|480 $\times$ 640|0.0911|0.3262|0.0440|0.9284|0.9904|0.9976|

### 1.2. Without ImageNet normalization

#### 1.2.1. Pytorch

|Encoder|Number of params|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.0/nyu_tiyolov8s_nonorm.ckpt)|10,810,465|0.1033|0.3717|0.0578|0.8965|0.9814|0.9958|
|[tiyolov5s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.1/nyu_tiyolov5s_metricleres_customsilu.ckpt)|8,797,057|0.1120|0.3885|0.0633|0.8817|0.9810|0.9955|
|[tiyoloxslite](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.1/nyu_tiyoloxslite_metricleres_cutmix.ckpt)|8,838,621|0.1138|0.3922|0.0641|0.8760|0.9794|0.9964|

#### 1.2.2. ONNX

|Encoder|Number of params|Input size|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.0/nyu_tiyolov8s_nonorm.onnx)|10,800,518|480 $\times$ 640|0.1033|0.3716|0.0578|0.8965|0.9813|0.9958|
|[tiyolov5s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.1/nyu_tiyolov5s_metricleres_customsilu.onnx)|8,786,470|480 $\times$ 640|0.1120|0.3886|0.0634|0.8819|0.9810|0.9954|
|[tiyoloxslite](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.1/nyu_tiyoloxslite_metricleres_cutmix.onnx)|8,827,894|480 $\times$ 640|0.1136|0.3914|0.0639|0.8765|0.9794|0.9963|

## 2. KITTI dataset (80m)

### 2.1. With ImageNet normalization

#### 2.1.1. Pytorch
<!-- |[efficientnetb4 with neck](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-kitti/kitti_efficientnetb4_base.ckpt)|26,645,689|52.16|0.0522|2.3859|0.1840|0.9696|0.9954|0.9986| -->
|Encoder|Number of params|FPS (352 $\times$ 1120)↑|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[efficientnetb4](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-kitti/kitti_efficientnetb4_lite.ckpt)|18,888,745|65.40|0.0526|2.4374|0.1913|0.9684|0.9949|0.9986|
|[efficientnetb0](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v.1.0.0-kitti/kitti_efficientnetb0_lite.ckpt)|4,731,845|104.17|0.0593|2.7452|0.2345|0.9603|0.9935|0.0.998|
|[tiyolov8m](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8m_kitti.ckpt)|25,776,145|67.13|0.0455|2.1825|0.1540|0.9795|0.9960|0.9982|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8s_kitti.ckpt)|10,810,465|101.35|0.0479|2.4401|0.1858|0.9741|0.9946|0.9981|
|[tiyolov8n](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8n_kitti.ckpt)|2,708,017|92.85|0.0542|2.4449|0.1940|0.9725|0.9946|0.9979|
|[*NeWCRFs*](https://github.com/aliyun/NeWCRFs)|*270,444,877*|*14.08*|*0.048*|*2.2319*|*0.1584*|*0.9767*|*0.9962*|*0.9986*|
|[*PixelFormer*](https://github.com/ashutosh1807/PixelFormer)|*270,895,920*|*15.29*|*0.0534*|*2.4173*|*0.184*|*0.9711*|*0.9952*|*0.9982*|

#### 2.1.2. ONNX
<!-- |[efficientnetb4 with neck](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-kitti/kitti_efficientnetb4_base.onnx)|26,577,830|352 $\times$ 1120|0.0545|2.8055|0.2360|0.9627|0.9935|0.9978| -->
|Encoder|Number of params|Input size|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[efficientnetb4](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0-kitti/kitti_efficientnetb4_lite.onnx)|18,825,606|352 $\times$ 1120|0.0578|2.7219|0.2337|0.9617|0.9941|0.9977|
|[efficientnetb0](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v.1.0.0-kitti/kitti_efficientnetb0_lite.onnx)|4,710,434|352 $\times$ 1120|0.0683|3.3592|0.3351|0.9460|0.9871|0.9961|
|[tiyolov8m](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8m_kitti.onnx)|25,759,494|352 $\times$ 1120|0.0512|2.6906|0.2195|0.9665|0.9934|0.9976|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8s_kitti.onnx)|10,800,518|352 $\times$ 1120|0.0562|2.9756|0.2668|0.9580|0.9908|0.9969|
|[tiyolov8n](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/v1.4.0/tiyolov8n_kitti.onnx)|2,703,046|352 $\times$ 1120|0.0602|3.0310|0.2854|0.9542|0.9890|0.9967|
|[*NeWCRFs*](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/extra/newcrfs_kitti.onnx)|271,971,053|352 $\times$ 1120|0.0509|2.1871|0.1608|0.9762|0.9963|0.9987|
|[*PixelFormer*](https://github.com/DeltaX-AI-Lab/MetricLeReS/releases/download/extra/pixelformer_kitti.onnx)|261,458,134|352 $\times$ 1120|0.0542|2.3748|0.1817|0.9724|0.9955|0.9984|

### 2.2. Without ImageNet normalization

#### 2.2.1. Pytorch

|Encoder|Number of params|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.0/kitti_tiyolov8s_nonorm.ckpt)|10,810,465|0.0479|2.3879|0.1798|0.9721|0.9952|0.9982|

#### 2.2.2. ONNX
|Encoder|Number of params|Input size|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.0/kitti_tiyolov8s_nonorm.onnx)|10,800,518|352 $\times$ 1120|0.0475|2.2880|0.1738|0.9742|0.9952|0.9981|

## 3. NYUv2 grayscale dataset (10m)

> Note: this dataset was made by transforming NYUv2 images from RGB to grayscale

### 3.1. With ImageNet normalization

#### 3.1.1. Pytorch

*(to be added)*

#### 3.1.2. ONNX

*(to be added)*

### 3.2. Without ImageNet normalization

#### 3.2.1. Pytorch

|Encoder|Number of params|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.0/nyugrayscale_tiyolov8s_nonorm.ckpt)|10,810,465|0.1058|0.3732|0.0592|0.8917|0.9821|0.9958|

#### 3.2.2. ONNX

|Encoder|Number of params|Input size|Abs.Rel↓ |RMSE↓|Sqr.Rel↓|$\delta1$↑|$\delta2$↑|$\delta3$↑| 
|----------|------:|------:|------:|------:|------:|------:|------:|------:|
|[tiyolov8s](https://github.com/DeltaX-AI-Lab/mobis-oms-depth-estimation/releases/download/v1.0/nyugrayscale_tiyolov8s_nonorm.onnx)|10,800,518|480 $\times$ 640|0.1056|0.3749|0.0592|0.8913|0.9826|0.9959|
