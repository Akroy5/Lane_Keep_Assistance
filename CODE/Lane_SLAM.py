import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import cv2
import time
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import open3d as o3d  # For point cloud processing

# ----------------------
# SLAM MODULE
# ----------------------
class VisualSLAM(nn.Module):
    """Visual SLAM module integrating ORB features and deep loop closure detection"""
    def __init__(self, input_size=(384, 640)):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), 
            dict(checks=50)
        )
        
        self.loop_closure = nn.Sequential(
            nn.Linear(256 * 12 * 20, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.pose_regressor = nn.Sequential(
            nn.Linear(256 + 1000, 128),  # Features + ORB descriptors
            nn.ReLU(),
            nn.Linear(128, 6)  # 6-DOF pose
        )
        
        self.map_points = o3d.geometry.PointCloud()
        self.keyframes = []
        self.current_pose = np.eye(4)
        
    def extract_features(self, image):
        """Extract both deep features and ORB features"""
        # Convert to OpenCV format
        img_cv = np.array(image.permute(1, 2, 0).cpu().numpy() * 255
        img_cv = img_cv.astype(np.uint8)
        
        # Extract ORB features
        kp, des = self.orb.detectAndCompute(img_cv, None)
        
        # Extract deep features
        with torch.no_grad():
            deep_feat = self.feature_extractor(image.unsqueeze(0))
            deep_feat = F.adaptive_avg_pool2d(deep_feat, (12, 20)).view(-1)
        
        return kp, des, deep_feat
    
    def detect_loop_closure(self, current_feat):
        """Detect loop closures using deep features"""
        if not self.keyframes:
            return -1, 0.0
        
        similarities = []
        for kf in self.keyframes:
            sim = F.cosine_similarity(current_feat, kf['deep_feat'], dim=0)
            similarities.append(sim.item())
        
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        if max_sim > 0.85:  # High similarity threshold
            return max_idx, max_sim
        return -1, 0.0
    
    def track_features(self, prev_kp, prev_des, curr_kp, curr_des):
        """Match features between frames"""
        if prev_des is None or curr_des is None:
            return []
            
        matches = self.flann.knnMatch(prev_des, curr_des, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches
    
    def estimate_pose(self, prev_frame, curr_frame):
        """Estimate camera pose using PnP"""
        prev_kp, prev_des, prev_deep = prev_frame
        curr_kp, curr_des, curr_deep = curr_frame
        
        matches = self.track_features(prev_kp, prev_des, curr_kp, curr_des)
        if len(matches) < 50:
            return None
        
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Camera intrinsic parameters (assumed)
        K = np.array([[500, 0, 320], [0, 500, 192], [0, 0, 1]])
        dist_coeffs = np.zeros(4)
        
        # Find essential matrix and recover pose
        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        _, R, t, mask = cv2.recoverPose(E, curr_pts, prev_pts, K, mask=mask)
        
        # Combine features for pose regression
        combined_feat = torch.cat([
            prev_deep, 
            torch.from_numpy(prev_des.flatten()[:1000]).float()
        ])
        
        # Deep pose regression
        with torch.no_grad():
            pose_delta = self.pose_regressor(combined_feat).numpy()
        
        # Combine geometric and learned pose
        pose_matrix = self.delta_to_matrix(pose_delta)
        return pose_matrix
    
    def delta_to_matrix(self, delta):
        """Convert 6-DOF pose delta to transformation matrix"""
        R = cv2.Rodrigues(delta[:3])[0]
        t = delta[3:]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def update_map(self, frame, pose):
        """Update SLAM map with new keyframe"""
        keyframe = {
            'kp': frame[0],
            'des': frame[1],
            'deep_feat': frame[2],
            'pose': pose
        }
        self.keyframes.append(keyframe)
        
        # Triangulate new points (simplified)
        # In a real system, we'd do proper triangulation
        if len(self.keyframes) > 1:
            prev_frame = self.keyframes[-2]
            matches = self.track_features(
                prev_frame['kp'], prev_frame['des'], 
                frame[0], frame[1]
            )
            
            # Add matched points to map
            if matches:
                # This is simplified - real implementation would use triangulation
                for m in matches:
                    new_point = np.random.rand(3)  # Placeholder
                    self.map_points.points.append(new_point)
    
    def step(self, image):
        """Process new frame through SLAM pipeline"""
        # Extract features
        kp, des, deep_feat = self.extract_features(image)
        current_frame = (kp, des, deep_feat)
        
        # Initialize with identity if first frame
        if not self.keyframes:
            self.update_map(current_frame, np.eye(4))
            return np.eye(4)
        
        # Detect loop closure
        loop_idx, loop_sim = self.detect_loop_closure(deep_feat)
        if loop_idx >= 0:
            # Use loop closure pose
            self.current_pose = self.keyframes[loop_idx]['pose']
        else:
            # Estimate new pose
            prev_frame = self.keyframes[-1]
            pose_delta = self.estimate_pose(prev_frame, current_frame)
            
            if pose_delta is not None:
                self.current_pose = self.current_pose @ pose_delta
        
        # Update map with new keyframe
        self.update_map(current_frame, self.current_pose)
        
        return self.current_pose

# ----------------------
# ENHANCED LKA MODEL WITH SLAM
# ----------------------
class SLAMEnhancedLKA(nn.Module):
    """LKA system with integrated SLAM for improved localization and mapping"""
    def __init__(self, slam_config=None):
        super().__init__()
        # Original LKA components
        self.sensor_fusion = SensorFusionLayer()
        self.encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:7])
        self.lateral_conv = nn.Conv2d(1024, 512, kernel_size=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock(512, 512, dilation=2),
            ResidualBlock(512, 512, dilation=4),
            ResidualBlock(512, 512, dilation=6)
        )
        self.lane_head = LaneDetectionHead(512, num_classes=3)
        self.steering_reg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # SLAM module
        self.slam = VisualSLAM()
        
        # SLAM-enhanced components
        self.slam_fusion = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Localization refinement
        self.loc_refinement = nn.Sequential(
            nn.Linear(6, 32),  # 6-DOF pose
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
        # Map-based prediction module
        self.map_guidance = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Buffer for previous frame data
        self.prev_frame = None
        
    def forward(self, rgb, radar, lidar, ultra):
        # Process SLAM on RGB stream
        current_pose = self.slam.step(rgb)
        
        # Refine pose with learned model
        pose_tensor = torch.tensor(current_pose[:3, :].flatten()[:6], dtype=torch.float32)
        refined_pose = self.loc_refinement(pose_tensor)
        
        # Sensor fusion
        x = self.sensor_fusion(rgb, radar, lidar, ultra)
        
        # Feature extraction
        features = self.encoder(x)
        features = self.lateral_conv(features)
        
        # Fuse with SLAM information
        if self.prev_frame is not None:
            # Create map guidance features
            map_guidance = self.map_guidance(self.prev_frame['features'])
            
            # Warp previous features using pose
            warped_prev = self.warp_features(
                self.prev_frame['features'], 
                refined_pose,
                self.prev_frame['pose']
            )
            
            # Fuse with current features
            features = torch.cat([features, warped_prev], dim=1)
            features = self.slam_fusion(features)
        
        # Store current frame for next iteration
        self.prev_frame = {
            'features': features.clone(),
            'pose': refined_pose.detach()
        }
        
        # Residual processing
        features = self.res_blocks(features)
        
        # Lane segmentation
        lane_out = self.lane_head(features)
        
        # Steering prediction (now enhanced with SLAM)
        steering = self.steering_reg(features)
        
        # Apply map-based correction
        steering += 0.1 * refined_pose[5]  # Adjust based on yaw
        
        return lane_out, steering, current_pose
    
    def warp_features(self, features, current_pose, prev_pose):
        """Warp previous features to current viewpoint"""
        # Calculate relative transformation
        rel_pose = np.linalg.inv(prev_pose) @ current_pose
        
        # Convert to homography (simplified for 2D)
        # In real implementation, we'd use 3D warping
        H = np.eye(3)
        H[:2, :2] = rel_pose[:2, :2]
        H[:2, 2] = rel_pose[:2, 3] * 10  # Scaling factor
        
        # Apply homography transformation
        warped = F.grid_sample(
            features, 
            self.create_grid(features.size(), H),
            align_corners=True
        )
        return warped
    
    def create_grid(self, size, H):
        """Create grid for feature warping"""
        _, _, h, w = size
        grid = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack(grid, axis=-1).reshape(-1, 2)
        
        # Apply homography
        ones = np.ones((grid.shape[0], 1))
        grid_hom = np.concatenate([grid, ones], axis=1)
        warped = H @ grid_hom.T
        warped = warped[:2] / warped[2]
        warped = warped.T.reshape(h, w, 2)
        
        # Normalize to [-1, 1]
        warped[..., 0] = 2 * warped[..., 0] / w - 1
        warped[..., 1] = 2 * warped[..., 1] / h - 1
        
        return torch.tensor(warped, dtype=torch.float32).unsqueeze(0)

# ----------------------
# ENHANCED DATASET WITH GROUND TRUTH POSES
# ----------------------
class SLAMLKADataset(LKADataset):
    """Extended dataset with SLAM ground truth poses"""
    def __init__(self, data_root, split='train', transform=None):
        super().__init__(data_root, split, transform)
        self.pose_data = self.load_poses()
        
    def load_poses(self):
        poses = {}
        split_dir = os.path.join(self.data_root, self.split)
        
        for scene_id in os.listdir(split_dir):
            scene_path = os.path.join(split_dir, scene_id)
            pose_file = os.path.join(scene_path, 'poses.txt')
            
            if os.path.exists(pose_file):
                with open(pose_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        ts = parts[0]
                        pose = np.array([float(x) for x in parts[1:]]).reshape(4, 4)
                        poses[f"{scene_id}/{ts}"] = pose
        return poses
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        scene_ts = f"{os.path.basename(os.path.dirname(os.path.dirname(sample['camera'])))}/{os.path.splitext(os.path.basename(sample['camera']))[0]}"
        
        if scene_ts in self.pose_data:
            sample['pose'] = self.pose_data[scene_ts]
        else:
            sample['pose'] = np.eye(4)
            
        return sample

# ----------------------
# ENHANCED LOSS FUNCTION FOR SLAM
# ----------------------
class SLAMLoss(nn.Module):
    """Loss function incorporating SLAM pose estimation"""
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # Segmentation weight
        self.beta = beta    # Steering weight
        self.gamma = gamma  # Pose weight
        
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.MSELoss()
        self.pose_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        seg_pred, steer_pred, pose_pred = outputs
        seg_target, steer_target, pose_target = targets
        
        # Calculate individual losses
        seg_loss = self.seg_loss(seg_pred, seg_target)
        steer_loss = self.reg_loss(steer_pred.squeeze(), steer_target)
        
        # Convert poses to 6-DOF representation
        pose_pred_vec = self.matrix_to_vec(pose_pred)
        pose_target_vec = self.matrix_to_vec(pose_target)
        pose_loss = self.pose_loss(pose_pred_vec, pose_target_vec)
        
        # Combined loss
        total_loss = (
            self.alpha * seg_loss + 
            self.beta * steer_loss + 
            self.gamma * pose_loss
        )
        
        return total_loss, seg_loss, steer_loss, pose_loss
    
    def matrix_to_vec(self, matrix):
        """Convert 4x4 matrix to 6-DOF vector (3 rotation + 3 translation)"""
        rvec = cv2.Rodrigues(matrix[:3, :3])[0].flatten()
        tvec = matrix[:3, 3]
        return torch.cat([torch.tensor(rvec), torch.tensor(tvec)])

# ----------------------
# ENHANCED TRAINER WITH SLAM SUPPORT
# ----------------------
class SLAMLKATrainer(LKATrainer):
    """Training pipeline with SLAM-enhanced LKA system"""
    def __init__(self, config):
        super().__init__(config)
        self.criterion = SLAMLoss(alpha=0.6, beta=0.2, gamma=0.2)
        self.train_loader, self.val_loader = self.prepare_dataloaders()
        
    def build_model(self):
        model = SLAMEnhancedLKA()
        if self.config['pretrained']:
            model.load_state_dict(torch.load(self.config['pretrained_path']))
        return model.to(self.device)
    
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
        
        train_set = SLAMLKADataset(
            self.config['data_root'], 
            split='train',
            transform=train_transform
        )
        
        val_set = SLAMLKADataset(
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
        total_steer_loss = 0.0
        total_pose_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0
        total_pose_err = 0.0
        
        for i, batch in enumerate(self.train_loader):
            # Move data to device
            rgb = batch['rgb'].to(self.device)
            radar = batch['radar'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            ultrasonic = batch['ultrasonic'].to(self.device)
            labels = batch['label'].to(self.device)
            steering = batch['steering'].to(self.device)
            pose_gt = batch['pose'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            seg_pred, steer_pred, pose_pred = self.model(
                rgb, radar, lidar, ultrasonic
            )
            
            # Calculate loss
            loss, seg_loss, steer_loss, pose_loss = self.criterion(
                (seg_pred, steer_pred, pose_pred),
                (labels, steering, pose_gt)
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            iou = calculate_iou(seg_pred, labels)
            acc = calculate_accuracy(seg_pred, labels)
            pose_err = F.mse_loss(pose_pred, pose_gt)
            
            # Update stats
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_steer_loss += steer_loss.item()
            total_pose_loss += pose_loss.item()
            total_iou += iou.item()
            total_acc += acc.item()
            total_pose_err += pose_err.item()
            
            # Log progress
            if i % self.config['log_interval'] == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | IoU: {iou:.4f} | "
                      f"Acc: {acc:.4f} | Pose Err: {pose_err:.4f}")
        
        # Calculate epoch averages
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_steer_loss = total_steer_loss / len(self.train_loader)
        avg_pose_loss = total_pose_loss / len(self.train_loader)
        avg_iou = total_iou / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        avg_pose_err = total_pose_err / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'steer_loss': avg_steer_loss,
            'pose_loss': avg_pose_loss,
            'iou': avg_iou,
            'accuracy': avg_acc,
            'pose_error': avg_pose_err
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_steer_loss = 0.0
        total_pose_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0
        total_pose_err = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                rgb = batch['rgb'].to(self.device)
                radar = batch['radar'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                ultrasonic = batch['ultrasonic'].to(self.device)
                labels = batch['label'].to(self.device)
                steering = batch['steering'].to(self.device)
                pose_gt = batch['pose'].to(self.device)
                
                # Forward pass
                seg_pred, steer_pred, pose_pred = self.model(
                    rgb, radar, lidar, ultrasonic
                )
                
                # Calculate loss
                loss, seg_loss, steer_loss, pose_loss = self.criterion(
                    (seg_pred, steer_pred, pose_pred),
                    (labels, steering, pose_gt)
                )
                
                # Metrics
                iou = calculate_iou(seg_pred, labels)
                acc = calculate_accuracy(seg_pred, labels)
                pose_err = F.mse_loss(pose_pred, pose_gt)
                
                # Update stats
                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                total_steer_loss += steer_loss.item()
                total_pose_loss += pose_loss.item()
                total_iou += iou.item()
                total_acc += acc.item()
                total_pose_err += pose_err.item()
        
        # Calculate epoch averages
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = total_seg_loss / len(self.val_loader)
        avg_steer_loss = total_steer_loss / len(self.val_loader)
        avg_pose_loss = total_pose_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        avg_pose_err = total_pose_err / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'steer_loss': avg_steer_loss,
            'pose_loss': avg_pose_loss,
            'iou': avg_iou,
            'accuracy': avg_acc,
            'pose_error': avg_pose_err
        }

# ----------------------
# ROS INTEGRATION WITH SLAM
# ----------------------
class ROSLKAWithSLAM:
    """ROS node for SLAM-enhanced LKA system"""
    def __init__(self, model_path):
        # Load trained model
        self.model = SLAMEnhancedLKA().eval()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to('cuda')
        
        # ROS initialization
        rospy.init_node('slam_lka_system')
        
        # Publishers
        self.steering_pub = rospy.Publisher('/vehicle/steering_cmd', Float32, queue_size=1)
        self.lane_viz_pub = rospy.Publisher('/lka/lane_detection', Image, queue_size=1)
        self.map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=1)
        self.pose_pub = rospy.Publisher('/slam/pose', PoseStamped, queue_size=1)
        
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
            seg_pred, steer_pred, pose_pred = self.model(
                rgb_tensor.unsqueeze(0),
                radar_tensor.unsqueeze(0),
                lidar_tensor.unsqueeze(0),
                ultra_tensor.unsqueeze(0)
            )
        
        # Process results
        steering_angle = steer_pred.item()
        lane_mask = (torch.sigmoid(seg_pred) > 0.5).cpu().numpy().squeeze()
        current_pose = pose_pred.cpu().numpy()
        
        # Publish steering command
        self.steering_pub.publish(steering_angle)
        
        # Publish visualization
        viz_msg = self.create_visualization(lane_mask)
        self.lane_viz_pub.publish(viz_msg)
        
        # Publish SLAM results
        self.publish_map()
        self.publish_pose(current_pose)
        
        # Reset sensor data
        self.reset_sensors()
    
    def publish_map(self):
        """Publish SLAM map as point cloud"""
        if self.model.slam.map_points:
            map_msg = self.pointcloud_to_msg(self.model.slam.map_points)
            self.map_pub.publish(map_msg)
    
    def publish_pose(self, pose_matrix):
        """Publish current vehicle pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        
        # Extract position
        pose_msg.pose.position.x = pose_matrix[0, 3]
        pose_msg.pose.position.y = pose_matrix[1, 3]
        pose_msg.pose.position.z = pose_matrix[2, 3]
        
        # Extract orientation (quaternion)
        r = R.from_matrix(pose_matrix[:3, :3])
        quat = r.as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)
    
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
        'data_root': '/path/to/slam_dataset',
        'backbone': 'resnet50',
        'pretrained': False,
        'pretrained_path': '',
        'batch_size': 8,
        'epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'log_interval': 20,
        'vis_interval': 5,
        'save_dir': './slam_checkpoints',
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize and run training
    trainer = SLAMLKATrainer(config)
    history = trainer.train()
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot([m['loss'] for m in history['train']], label='Train')
    plt.plot([m['loss'] for m in history['val']], label='Validation')
    plt.title('Total Loss')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot([m['iou'] for m in history['train']], label='Train')
    plt.plot([m['iou'] for m in history['val']], label='Validation')
    plt.title('IoU')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot([m['pose_error'] for m in history['train']], label='Train')
    plt.plot([m['pose_error'] for m in history['val']], label='Validation')
    plt.title('Pose Error')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot([m['steer_loss'] for m in history['train']], label='Train')
    plt.plot([m['steer_loss'] for m in history['val']], label='Validation')
    plt.title('Steering Loss')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot([m['seg_loss'] for m in history['train']], label='Train')
    plt.plot([m['seg_loss'] for m in history['val']], label='Validation')
    plt.title('Segmentation Loss')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot([m['pose_loss'] for m in history['train']], label='Train')
    plt.plot([m['pose_loss'] for m in history['val']], label='Validation')
    plt.title('Pose Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_history.png'))