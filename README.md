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
```text
Input (6 channels: RGB[3] + Radar[1] + LiDAR[1] + Ultrasonic[1])
  │
  ▼
SensorFusionLayer
  └─ early concatenate into a single tensor
  │
  ▼
Encoder:
  ├─ Conv2d → 64 ch, ReLU
  ├─ Conv2d (stride=2) → 128 ch, ReLU
  ├─ Conv2d (stride=2) → 256 ch, ReLU
  ├─ ResidualBlock (dilation=2)
  ├─ ResidualBlock (dilation=4)
  └─ ResidualBlock (dilation=6)
  │
  ▼
Bottleneck:
  ├─ 1×1 Conv → 512 ch, ReLU
  └─ 1×1 Conv → 256 ch, ReLU
  │
  ▼
Decoder Heads:
  ├─ Segmentation Head:
  │    └─ ConvTranspose2d ×2 → upsample to input resolution
  │    └─ 1×1 Conv → 3-class mask logits
  │
  └─ Steering Head:
       └─ AdaptiveAvgPool → flatten
       └─ MLP (256 → 128 → 1) → continuous angle output
```

---
## 📊 Sample Metrics
# **Classic LKA**
| Split | mIoU | Pixel Acc | Steering RMSE (°) |
| :---: | :--: | :-------: | :---------------: |
| Train | 0.88 |    0.96   |        2.5        |
|  Val  | 0.84 |    0.93   |        2.9        |

# **SLAM‑Enhanced LKA**
| Split | mIoU | Pixel Acc | Steering RMSE (°) |
| :---: | :--: | :-------: | :---------------: |
| Train | 0.90 |    0.97   |        2.2        |
|  Val  | 0.86 |    0.94   |        2.6        |
