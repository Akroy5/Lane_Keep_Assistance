import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

# ----------------------
# MODEL ARCHITECTURE
# ----------------------
class SensorFusionLayer(nn.Module):
    """Multi-modal sensor fusion layer with attention mechanism"""
    def __init__(self, channels: Tuple[int, int, int, int] = (3, 1, 1, 1)):
        super().__init__()
        self.rgb_conv = nn.Conv2d(channels[0], 64, kernel_size=3, padding=1)
        self.radar_conv = nn.Conv2d(channels[1], 32, kernel_size=3, padding=1)
        self.lidar_conv = nn.Conv2d(channels[2], 32, kernel_size=3, padding=1)
        self.ultra_conv = nn.Conv2d(channels[3], 32, kernel_size=3, padding=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, rgb, radar, lidar, ultra):
        rgb_feat = F.relu(self.rgb_conv(rgb))
        radar_feat = F.relu(self.radar_conv(radar))
        lidar_feat = F.relu(self.lidar_conv(lidar))
        ultra_feat = F.relu(self.ultra_conv(ultra))
        
        fused = torch.cat([rgb_feat, radar_feat, lidar_feat, ultra_feat], dim=1)
        attn_weights = self.attention(fused)
        attn_weights = attn_weights.unsqueeze(2)
        
        feats = torch.stack([rgb_feat, radar_feat, lidar_feat, ultra_feat], dim=1)
        weighted = torch.sum(attn_weights * feats, dim=1)
        return weighted

class ResidualBlock(nn.Module):
    """Residual block with dilation support"""
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class LaneDetectionHead(nn.Module):
    """Lane detection head with multi-scale feature fusion"""
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.up1(x)
        
        x = F.relu(self.conv2(x))
        x = self.up2(x)
        
        x = F.relu(self.conv3(x))
        x = self.up3(x)
        
        return self.output(x)

class LKANet(nn.Module):
    """Complete LKA system with sensor fusion and lane detection"""
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.sensor_fusion = SensorFusionLayer()
        
        # Feature extractor backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            self.encoder = nn.Sequential(*list(base_model.children())[:7])
            self.lateral_conv = nn.Conv2d(1024, 512, kernel_size=1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Residual blocks with dilation
        self.res_blocks = nn.Sequential(
            ResidualBlock(512, 512, dilation=2),
            ResidualBlock(512, 512, dilation=4),
            ResidualBlock(512, 512, dilation=6)
        )
        
        # Lane detection head
        self.lane_head = LaneDetectionHead(512, num_classes=3)
        
        # Steering angle regression
        self.steering_reg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, rgb, radar, lidar, ultra):
        # Sensor fusion
        x = self.sensor_fusion(rgb, radar, lidar, ultra)
        
        # Feature extraction
        features = self.encoder(x)
        features = self.lateral_conv(features)
        
        # Residual processing
        features = self.res_blocks(features)
        
        # Lane segmentation
        lane_out = self.lane_head(features)
        
        # Steering prediction
        steering = self.steering_reg(features)
        
        return lane_out, steering

# ----------------------
# DATA HANDLING
# ----------------------
class LKADataset(Dataset):
    """Multi-sensor dataset for lane keeping"""
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform or self.default_transform()
        self.samples = self.load_samples()
        
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((384, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_samples(self):
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        
        for scene_id in os.listdir(split_dir):
            scene_path = os.path.join(split_dir, scene_id)
            sensor_path = os.path.join(scene_path, 'sensors')
            
            # Get all timestamps
            timestamps = set()
            for sensor in ['camera', 'radar', 'lidar', 'ultrasonic']:
                sensor_dir = os.path.join(sensor_path, sensor)
                if os.path.exists(sensor_dir):
                    timestamps.update(
                        fname.split('.')[0] for fname in os.listdir(sensor_dir)
                    )
            
            # Create samples
            for ts in timestamps:
                sample = {
                    'camera': os.path.join(sensor_path, 'camera', f'{ts}.jpg'),
                    'radar': os.path.join(sensor_path, 'radar', f'{ts}.png'),
                    'lidar': os.path.join(sensor_path, 'lidar', f'{ts}.png'),
                    'ultrasonic': os.path.join(sensor_path, 'ultrasonic', f'{ts}.png'),
                    'label': os.path.join(scene_path, 'labels', f'{ts}.png'),
                    'steering': float(open(os.path.join(scene_path, 'steering', f'{ts}.txt')).read()
                }
                # Check all files exist
                if all(os.path.exists(v) for k, v in sample.items() if k != 'steering'):
                    samples.append(sample)
                    
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        rgb = Image.open(sample['camera']).convert('RGB')
        radar = Image.open(sample['radar']).convert('L')
        lidar = Image.open(sample['lidar']).convert('L')
        ultrasonic = Image.open(sample['ultrasonic']).convert('L')
        label = Image.open(sample['label']).convert('L')
        
        # Apply transforms
        if self.transform:
            rgb = self.transform(rgb)
            radar = self.transform(radar)
            lidar = self.transform(lidar)
            ultrasonic = self.transform(ultrasonic)
            label = self.transform(label)
            
        # Convert labels to tensor
        label = (label > 0.5).float()
        steering = torch.tensor(sample['steering'], dtype=torch.float32)
        
        return {
            'rgb': rgb,
            'radar': radar,
            'lidar': lidar,
            'ultrasonic': ultrasonic,
            'label': label,
            'steering': steering
        }

# ----------------------
# LOSS FUNCTIONS
# ----------------------
class LaneLoss(nn.Module):
    """Multi-task loss for lane segmentation and steering prediction"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for segmentation loss
        self.beta = beta    # Weight for steering loss
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.MSELoss()
        
    def forward(self, seg_pred, steer_pred, seg_target, steer_target):
        seg_loss = self.seg_loss(seg_pred, seg_target)
        reg_loss = self.reg_loss(steer_pred.squeeze(), steer_target)
        return self.alpha * seg_loss + self.beta * reg_loss, seg_loss, reg_loss

# ----------------------
# METRICS CALCULATION
# ----------------------
def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union for lane segmentation"""
    pred = (torch.sigmoid(pred) > threshold
    target = target > threshold
    
    intersection = torch.logical_and(pred, target).sum(dim=(1, 2, 3))
    union = torch.logical_or(pred, target).sum(dim=(1, 2, 3))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def calculate_accuracy(pred, target, threshold=0.5):
    """Calculate pixel accuracy for segmentation"""
    pred = (torch.sigmoid(pred) > threshold
    correct = (pred == target).sum()
    total = target.numel()
    return correct.float() / total

# ----------------------
# TRAINING UTILITIES
# ----------------------
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class LKATrainer:
    """Training pipeline for LKA system"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.optimizer = self.configure_optimizer()
        self.criterion = LaneLoss(alpha=0.8, beta=0.2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=7)
        self.train_loader, self.val_loader = self.prepare_dataloaders()
        
    def build_model(self):
        model = LKANet(backbone=self.config['backbone'])
        if self.config['pretrained']:
            model.load_state_dict(torch.load(self.config['pretrained_path']))
        return model.to(self.device)
    
    def configure_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
    
    def prepare_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.Resize((384, 640)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((384, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_set = LKADataset(
            self.config['data_root'], 
            split='train',
            transform=train_transform
        )
        
        val_set = LKADataset(
            self.config['data_root'], 
            split='val',
            transform=val_transform
        )
        
        train_loader = DataLoader(
            train_set,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_reg_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0
        
        for i, batch in enumerate(self.train_loader):
            # Move data to device
            rgb = batch['rgb'].to(self.device)
            radar = batch['radar'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            ultrasonic = batch['ultrasonic'].to(self.device)
            labels = batch['label'].to(self.device)
            steering = batch['steering'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            seg_pred, steer_pred = self.model(rgb, radar, lidar, ultrasonic)
            
            # Calculate loss
            loss, seg_loss, reg_loss = self.criterion(
                seg_pred, steer_pred, labels, steering
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            iou = calculate_iou(seg_pred, labels)
            acc = calculate_accuracy(seg_pred, labels)
            
            # Update stats
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_reg_loss += reg_loss.item()
            total_iou += iou.item()
            total_acc += acc.item()
            
            # Log progress
            if i % self.config['log_interval'] == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | IoU: {iou:.4f} | "
                      f"Acc: {acc:.4f} | Steering Err: {reg_loss.item():.4f}")
        
        # Calculate epoch averages
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_reg_loss = total_reg_loss / len(self.train_loader)
        avg_iou = total_iou / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'reg_loss': avg_reg_loss,
            'iou': avg_iou,
            'accuracy': avg_acc
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_reg_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                rgb = batch['rgb'].to(self.device)
                radar = batch['radar'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                ultrasonic = batch['ultrasonic'].to(self.device)
                labels = batch['label'].to(self.device)
                steering = batch['steering'].to(self.device)
                
                # Forward pass
                seg_pred, steer_pred = self.model(rgb, radar, lidar, ultrasonic)
                
                # Calculate loss
                loss, seg_loss, reg_loss = self.criterion(
                    seg_pred, steer_pred, labels, steering
                )
                
                # Metrics
                iou = calculate_iou(seg_pred, labels)
                acc = calculate_accuracy(seg_pred, labels)
                
                # Update stats
                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                total_reg_loss += reg_loss.item()
                total_iou += iou.item()
                total_acc += acc.item()
        
        # Calculate epoch averages
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = total_seg_loss / len(self.val_loader)
        avg_reg_loss = total_reg_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'reg_loss': avg_reg_loss,
            'iou': avg_iou,
            'accuracy': avg_acc
        }
    
    def train(self):
        best_val_loss = float('inf')
        history = {'train': [], 'val': []}
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            history['val'].append(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Check early stopping
            self.early_stopping(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), 
                          f"{self.config['save_dir']}/best_model.pth")
                print(f"Saved new best model with val loss: {best_val_loss:.4f}")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"IoU: {train_metrics['iou']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"IoU: {val_metrics['iou']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            
            # Visualize results periodically
            if epoch % self.config['vis_interval'] == 0:
                self.visualize_results(epoch)
            
            # Early stopping check
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  f"{self.config['save_dir']}/final_model.pth")
        print("Training completed")
        return history
    
    def visualize_results(self, epoch):
        # Get sample batch
        batch = next(iter(self.val_loader))
        rgb = batch['rgb'][:4].to(self.device)
        radar = batch['radar'][:4].to(self.device)
        lidar = batch['lidar'][:4].to(self.device)
        ultrasonic = batch['ultrasonic'][:4].to(self.device)
        labels = batch['label'][:4].to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            seg_pred, steer_pred = self.model(rgb, radar, lidar, ultrasonic)
            seg_pred = torch.sigmoid(seg_pred)
        
        # Visualize
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        titles = ['RGB Input', 'Radar', 'LiDAR', 'Ultrasonic', 'Ground Truth', 'Prediction']
        
        for i in range(4):
            # Input sensors
            axes[i, 0].imshow(rgb[i].cpu().permute(1, 2, 0).numpy())
            axes[i, 1].imshow(radar[i].cpu().squeeze(), cmap='viridis')
            axes[i, 2].imshow(lidar[i].cpu().squeeze(), cmap='plasma')
            axes[i, 3].imshow(ultrasonic[i].cpu().squeeze(), cmap='inferno')
            
            # Labels and predictions
            axes[i, 4].imshow(labels[i].cpu().squeeze(), cmap='gray')
            axes[i, 5].imshow(seg_pred[i, 0].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            
            # Set titles
            if i == 0:
                for j, title in enumerate(titles):
                    axes[i, j].set_title(title)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['save_dir']}/visualization_epoch_{epoch}.png")
        plt.close()

# ----------------------
# ROS INTEGRATION (PSEUDO-CODE)
# ----------------------
class ROSLKAIntegration:
    """ROS node for LKA system integration"""
    def __init__(self, model_path):
        # Load trained model
        self.model = LKANet().eval()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to('cuda')
        
        # ROS initialization
        rospy.init_node('lka_system')
        
        # Publishers
        self.steering_pub = rospy.Publisher('/vehicle/steering_cmd', Float32, queue_size=1)
        self.lane_viz_pub = rospy.Publisher('/lka/lane_detection', Image, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/camera/rgb', Image, self.rgb_callback)
        rospy.Subscriber('/sensors/radar', PointCloud2, self.radar_callback)
        rospy.Subscriber('/sensors/lidar', PointCloud2, self.lidar_callback)
        rospy.Subscriber('/sensors/ultrasonic', RangeArray, self.ultrasonic_callback)
        
        # Sensor data buffers
        self.rgb_data = None
        self.radar_data = None
        self.lidar_data = None
        self.ultrasonic_data = None
        
    def sensor_data_ready(self):
        return all(data is not None for data in [
            self.rgb_data, self.radar_data, 
            self.lidar_data, self.ultrasonic_data
        ])
    
    def rgb_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Preprocess and store
        self.rgb_data = preprocess_rgb(cv_image)
    
    def radar_callback(self, msg):
        # Process radar point cloud
        self.radar_data = process_radar(msg)
    
    def lidar_callback(self, msg):
        # Process lidar point cloud
        self.lidar_data = process_lidar(msg)
    
    def ultrasonic_callback(self, msg):
        # Process ultrasonic data
        self.ultrasonic_data = process_ultrasonic(msg)
    
    def process_frame(self):
        if not self.sensor_data_ready():
            return
        
        # Convert to tensors
        rgb_tensor = torch.tensor(self.rgb_data).float().to('cuda')
        radar_tensor = torch.tensor(self.radar_data).float().to('cuda')
        lidar_tensor = torch.tensor(self.lidar_data).float().to('cuda')
        ultra_tensor = torch.tensor(self.ultrasonic_data).float().to('cuda')
        
        # Inference
        with torch.no_grad():
            seg_pred, steer_pred = self.model(
                rgb_tensor.unsqueeze(0),
                radar_tensor.unsqueeze(0),
                lidar_tensor.unsqueeze(0),
                ultra_tensor.unsqueeze(0)
            )
        
        # Process results
        steering_angle = steer_pred.item()
        lane_mask = (torch.sigmoid(seg_pred) > 0.5).cpu().numpy().squeeze()
        
        # Publish steering command
        self.steering_pub.publish(steering_angle)
        
        # Publish visualization
        viz_msg = self.create_visualization(lane_mask)
        self.lane_viz_pub.publish(viz_msg)
        
        # Reset sensor data
        self.rgb_data = None
        self.radar_data = None
        self.lidar_data = None
        self.ultrasonic_data = None
    
    def run(self):
        rate = rospy.Rate(10)  # 10Hz processing
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()

# ----------------------
# MAIN EXECUTION
# ----------------------
if __name__ == "__main__":
    # Configuration
    config = {
        'data_root': '/path/to/dataset',
        'backbone': 'resnet50',
        'pretrained': False,
        'pretrained_path': '',
        'batch_size': 8,
        'epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'log_interval': 20,
        'vis_interval': 5,
        'save_dir': './checkpoints',
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize and run training
    trainer = LKATrainer(config)
    history = trainer.train()
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot([m['loss'] for m in history['train']], label='Train')
    plt.plot([m['loss'] for m in history['val']], label='Validation')
    plt.title('Total Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot([m['iou'] for m in history['train']], label='Train')
    plt.plot([m['iou'] for m in history['val']], label='Validation')
    plt.title('IoU')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot([m['accuracy'] for m in history['train']], label='Train')
    plt.plot([m['accuracy'] for m in history['val']], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot([m['reg_loss'] for m in history['train']], label='Train')
    plt.plot([m['reg_loss'] for m in history['val']], label='Validation')
    plt.title('Steering MSE Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_history.png'))