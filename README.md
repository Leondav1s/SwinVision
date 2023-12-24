# SwinVision: detecting small objects in low-light environments
## Introduction
![SwinVision](resources/SwinVision.jpg)

Neural networks have been widely used in the field of object detection. Transformers enable effective object detection through global context awareness, modular design, scalability, and adaptability to diverse target scales. However, small object detection requires careful consideration due to its complex computations, data requirements, and real-time performance challenges. We present SwinVision, an innovative framework for small object detection in low-light environments. We first introduce a Swin Transformer-based computing network optimized for real-time UAV monitoring in large areas. The framework strikes a balance between computational power and resource efficiency, surpassing conventional transformers. Then we present the STLE module, which enhances low-light image features for better object detection. Last is a specialized Swin-based detection block for accurate detection of small, detailed objects in resource-constrained scenarios.

## Install

```bash
# Clone the SwinVision repository
git clone https://github.com/Leondav1s/SwinVision.git

# Navigate to the cloned directory
# Use ultralytics framework
cd SwinVision/ultralytics

# Install the package in editable mode for development
pip install -e .
```

## Getting Started

**step1: use STLE to enhance your Images**

```bash
# Firstly put your images into SwinVision/SwinVision/STLE/enhance_dataset
cd SwinVision/SwinVision/STLE

# The enhanced images will be saved to the 'result' folder
python enhance.py
```

**step2: use SOF detector to predict**

```bash
cd SwinVision/SwinVision/ultalytics

# Remember change the source to your enhanced images result path
yolo detect predict model=SwinVision.pt source=../STLE/result/your/path
```

## Training

- **STLE**

```bash
# Prepare the training set of the LOL dataset before training
# Navigate to the directory
cd SwinVision/SwinVision/STLE

# Start training
python train.py --trainset_path /your/path/to/LoLdataset/
```

- **SOF detector**

``` bash
cd SwinVision/SwinVision/ultalytics

# Start training
yolo detect train data=VisDrone.yaml model=./ultralytics/models/sof_detector.yaml epochs=300 batch=12 half=True
```

## Results and models

The results of experiments are saved in 'experiments' folder.

|   Model    |         Backbone          |  mAP  | AP50  |                            Config                            |                         Google Drive                         |
| :--------: | :-----------------------: | :---: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| SwinVision | Improved Swin Transformer | 34.05 | 59.47 | [SwinVision](./SwinVision/ultralytics/ultralytics/models/sof_detector.yaml) | [model](https://drive.google.com/file/d/1xF0-Pu07z39uleSkdIgZunzQl1LZOpzn/view?usp=sharing) |
|  YOLOv8x   |       CSPDarkNet-53       | 27.74 | 46.92 | [YOLOv8x](./SwinVision/ultralytics/ultralytics/models/v8/yolov8.yaml) | [model](https://drive.google.com/file/d/1zoPEnSzpT1Zve3Yi1twqrHwy3ilxxxpd/view?usp=sharing) |

## Acknowledgements

A great thanks to [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) , [ultralytics](https://github.com/ultralytics/ultralytics) for providing the basis for this code.

