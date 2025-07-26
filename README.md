# 🚗 Lane Keeping Assistance (LKA) System

A **deep IoT‑fused** framework for real‑time lane detection and steering prediction, built from scratch.  
Supports two modes:
1. **Classic LKA** (`lane.py`): fuses camera, radar, LiDAR, ultrasonic for segmentation + steering.
2. **SLAM‑Enhanced LKA** (`Lane_SLAM.py`): adds a lightweight Visual SLAM module for improved localization, mapping & control.

---

## 🌟 Key Features

- **Multi‑Sensor Fusion**  
  Early fusion of 4 modalities → **6‑channel input**  
  RGB (3) + Radar (1) + LiDAR (1) + Ultrasonic (1)

- **Dual‑Head Output**  
  - **Lane Segmentation**: 3‑class mask (left, right lanes, background)  
  - **Steering Regression**: continuous angle (°)

- **Residual & Dilated CNN**  
  - Stacked residual blocks with dilations [2, 4, 6]  
  - No pretrained backbones—proof of concept from scratch

- **SLAM Integration (optional)**  
  - ORB + pose‑guided feature warping  
  - Loop‑closure support for drift correction  
  - Pose‑MSE loss for end‑to‑end refinement

- **Training Utilities**  
  - Multi‑task loss: `BCEWithLogitsLoss` + `MSELoss` (+ SLAM‑pose MSE)  
  - Cosine LR scheduler + Early stopping  
  - IoU & RMSE metrics

---

## 📁 Project Structure

```text
Lane_Keep_Assistance/
├── CODE/
│   ├── lane.py            # Classic LKA mode
│   ├── Lane_SLAM.py       # SLAM‑enhanced mode
│   ├── dataset.py         # Multi‑sensor PyTorch Dataset
│   ├── train.py           # Trainer (choose mode via --mode)
│   ├── inference.py       # Real‑time inference & ROS node
│   ├── utils.py           # Losses, metrics (IoU, RMSE)
│   └── requirements.txt   # pip dependencies
├── DATA/
│   ├── train/             # camera/, radar/, lidar/, ultrasonic/, masks/, steer/
│   └── val/               # same structure
├── RESULTS/               # checkpoints & logs (in .gitignore)
├── IMAGES/                # architecture diagrams & sample outputs
├── run_training.sh        # launch training for either mode
├── README.md              # this file
└── .gitignore
```
--- 
## 🧠 Architecture Overview graph LR
  subgraph Fusion
    rgb[RGB] --> F[SensorFusion]
    radar[Radar] --> F
    lidar[LiDAR] --> F
    ultra[Ultra] --> F
  end
  F --> E[ResBlock×3 (dil[2,4,6])]
  E --> B[1×1 Conv → 512 ch]
  B --> Seg[Segmentation Head → 3‑class mask]
  B --> Steer[Steering Head → angle]
  classDef heads fill:#f9f,stroke:#333,stroke-width:1px;
  class Seg,Steer heads


---
##📊 Sample Metrics
**Classic LKA**
| Split | mIoU | Pixel Acc | Steering RMSE (°) |
| :---: | :--: | :-------: | :---------------: |
| Train | 0.88 |    0.96   |        2.5        |
|  Val  | 0.84 |    0.93   |        2.9        |
**SLAM‑Enhanced LKA**
| Split | mIoU | Pixel Acc | Steering RMSE (°) |
| :---: | :--: | :-------: | :---------------: |
| Train | 0.90 |    0.97   |        2.2        |
|  Val  | 0.86 |    0.94   |        2.6        |
