# 🚧 Road Damage Detection using YOLOv5

A custom-trained YOLOv5m object detection model for detecting road surface damage including cracks, potholes, waterlogged potholes, and garbage. The dataset was built by merging multiple Roboflow datasets into a unified training set.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

---

## Overview

This project uses **YOLOv5m** (medium variant) from [Ultralytics](https://github.com/ultralytics/yolov5) to detect and classify road damage in images. The model was trained on a combined dataset sourced from multiple Roboflow datasets and merged into a single unified YAML configuration.

**Model:** YOLOv5m  
**Framework:** Ultralytics YOLOv5  
**Task:** Object Detection  
**Number of Classes:** 6

---

## Classes

| ID | Class Name       | Description                                      |
|----|------------------|--------------------------------------------------|
| 0  | `crack`          | Surface cracks on the road                      |
| 1  | `damage`         | General road surface damage                     |
| 2  | `pothole`        | Standard potholes                               |
| 3  | `pothole_water`  | Potholes filled with water                      |
| 4  | `pothole_water_m`| Multiple waterlogged potholes                   |
| 5  | `garbage`        | Garbage/debris on or near the road              |

---

## Dataset

The dataset was created by combining multiple Roboflow datasets into a single merged dataset.

**Dataset YAML Configuration:**

```yaml
path: C:\Users\Siddhesh\OneDrive\Desktop\detection\combined_dataset
train: train/images
val: valid/images
test: test/images
nc: 6
names:
  - crack
  - damage
  - pothole
  - pothole_water
  - pothole_water_m
  - garbage
```

The `dataset/combined_dataset/` directory contains three splits:
- **train/** — Training images and labels
- **valid/** — Validation images and labels
- **test/** — Test images and labels

---

## Project Structure

```
DETECTION_PROJECT/
│
├── dataset/
│   └── combined_dataset/
│       ├── test/
│       ├── train/
│       ├── valid/
│       └── data.yaml
│
├── testing/
│   ├── images/
│   │   ├── test.jpg
│   │   ├── test_1.jpg
│   │   ├── test_2.jpg
│   │   └── test_3.jpg
│   ├── outputs/
│   │   ├── result.jpg
│   │   ├── result1.jpg
│   │   ├── result2.jpg
│   │   └── result3.jpg
│   └── weights/
│       └── best.pt
│
├── training/
│   └── yolov5_road_damage_training.ipynb
│
├── yolov5/                    # Ultralytics YOLOv5 repository (clone separately)
├── requirements.txt
├── test_custom_yolov5.py
└── README.MD
```

---

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
pip install seaborn
pip install Pillow
```

> **Note:** It is recommended to use a virtual environment or conda environment. CUDA-enabled GPU is strongly recommended for training.

---

## Training

Training was performed using the Ultralytics YOLOv5 framework. See the notebook for the full training pipeline:

```
training/yolov5_road_damage_training.ipynb
```

**Training command (example):**

```bash
python yolov5/train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data dataset/combined_dataset/data.yaml \
  --weights yolov5m.pt \
  --name road_damage_yolov5m
```

The best model weights are saved at:

```
testing/weights/best.pt
```

---

## Inference

Run inference on test images using the provided script:

```bash
python test_custom_yolov5.py
```

Or use the YOLOv5 detect script directly:

```bash
python yolov5/detect.py \
  --weights testing/weights/best.pt \
  --source testing/images/ \
  --conf 0.25 \
  --save-txt \
  --project testing/outputs
```

---

## Results

Sample detection output images are available in `testing/outputs/`:

| Input | Output |
|-------|--------|
| test.jpg | result.jpg |
| test_1.jpg | result1.jpg |
| test_2.jpg | result2.jpg |
| test_3.jpg | result3.jpg |

---

## 🔧 Improvements & Future Work

The current model is a baseline trained on **YOLOv5m**. The following improvements are planned for future iterations:

### Model Upgrades
- [ ] Migrate to **YOLOv8m / YOLOv8x** (Ultralytics) for improved accuracy and faster convergence
- [ ] Experiment with **YOLOv9** and **YOLOv10** architectures for state-of-the-art detection performance
- [ ] Try **YOLO11** (Ultralytics latest) once stable, which offers better efficiency at higher scales
- [ ] Benchmark YOLOv5l and YOLOv5x as intermediate steps before full migration

### Dataset Improvements
- [ ] Expand the dataset with more diverse road conditions (night, rain, highway vs. urban)
- [ ] Add more samples for underrepresented classes like `pothole_water_m` and `garbage`
- [ ] Apply advanced augmentations (mosaic, MixUp, copy-paste) during training
- [ ] Refine and re-annotate noisy labels from merged Roboflow datasets



---

## Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com/) — for dataset hosting and augmentation tools