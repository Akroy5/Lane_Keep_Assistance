# 🚗 Lane Keeping Assistance using Deep IoT Sensor Fusion

A lightweight deep learning model that fuses **RGB, Radar, LiDAR, and Ultrasonic** sensor data for robust real-time **lane detection** in autonomous driving systems. The model is fully built **from scratch** (no pretrained backbone) to be deployable on edge systems.

---

## 🔧 Key Features

- **4-sensor fusion**: Early fusion of RGB (3-ch) + Radar + LiDAR + Ultrasonic = 6-channel input  
- **Custom CNN**: Encoder-decoder architecture (no pretrained backbones)  
- **Binary segmentation output**: Predicts lane masks  
- **IoT-ready**: Edge-deployable, real-time performance  
- **BCE Loss + IoU Evaluation**

---

## 🧠 Architecture
RGB (3) + Radar (1) + LiDAR (1) + Ultrasonic (1) → 6-channel Input
↓
Encoder (Conv → ReLU → BN) × 3
↓
Bottleneck (Conv)
↓
Decoder (TransposeConv → ReLU) × 2
↓
Sigmoid Output (1-channel mask)
