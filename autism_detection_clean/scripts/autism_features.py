"""
Autism-Specific Behavioral Feature Engineering
==============================================
Extract domain-specific features that capture autism movement patterns:
- Motion concentration (stimming location)
- Temporal periodicity (repetitiveness)
- Bilateral asymmetry
- Jerkiness vs fluidity
- Amplitude dynamics
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy


def extract_autism_features(video_tensor, motion_feats, pose_coords):
    """
    Extract autism-specific behavioral features from video and pose data
    
    Args:
        video_tensor: (T, H, W) video frames
        motion_feats: (T-1, H, W) optical flow magnitude
        pose_coords: (T, 34) pose keypoints (17 joints * 2 coords)
    
    Returns:
        features: (13,) behavioral feature vector
    """
    
    T = pose_coords.shape[0]
    features = []
    
    # ===== 1. MOTION CONCENTRATION (Where on body is movement?) =====
    # Split body into regions: upper (head+arms), lower (torso+legs)
    upper_indices = [0, 1, 2, 3, 4, 5, 6, 16]  # Head, shoulders, arms, neck
    lower_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15]  # Torso, hips, legs
    
    pose_motion = np.diff(pose_coords, axis=0)  # (T-1, 34) velocity
    pose_speed = np.sqrt(np.sum(pose_motion**2, axis=1))  # (T-1,) per-frame speed
    
    # Create flattened indices for x,y coordinates
    upper_coord_indices = np.array([[2*i, 2*i+1] for i in upper_indices]).flatten()
    lower_coord_indices = np.array([[2*i, 2*i+1] for i in lower_indices]).flatten()
    
    upper_motion = np.sqrt(np.sum(pose_motion[:, upper_coord_indices]**2, axis=1))
    lower_motion = np.sqrt(np.sum(pose_motion[:, lower_coord_indices]**2, axis=1))
    
    upper_concentration = np.mean(upper_motion) / (np.mean(pose_speed) + 1e-6)  # Ratio
    lower_concentration = np.mean(lower_motion) / (np.mean(pose_speed) + 1e-6)
    
    features.append(upper_concentration)  # 1: High in autism (stimming in hands/head)
    features.append(lower_concentration)  # 2: Low in autism
    
    # ===== 2. TEMPORAL PERIODICITY (Repetitive movements?) =====
    # Compute autocorrelation of motion speed to detect rhythmic patterns
    if len(pose_speed) > 10:
        # Normalize speed
        speed_norm = (pose_speed - np.mean(pose_speed)) / (np.std(pose_speed) + 1e-6)
        # Autocorrelation at typical stimming frequencies (1-3 Hz at 30fps)
        acf_vals = [np.correlate(speed_norm, speed_norm, mode='full')[len(speed_norm)-1-lag] 
                    for lag in [10, 15, 20, 30]]  # Lags corresponding to 0.33-1Hz at 30fps
        periodicity = np.mean(acf_vals)
    else:
        periodicity = 0.0
    
    features.append(periodicity)  # 3: High in autism (rhythmic stimming)
    
    # ===== 3. BILATERAL ASYMMETRY (Left vs Right symmetry) =====
    # Left side: indices 4, 5, 6, 7, 11, 12, 13
    # Right side: indices 1, 2, 3, 8, 9, 10, 14
    left_joint_indices = np.array([4, 5, 6, 7, 11, 12, 13])
    right_joint_indices = np.array([1, 2, 3, 8, 9, 10, 14])
    
    left_coord_indices = np.array([[2*i, 2*i+1] for i in left_joint_indices]).flatten()
    right_coord_indices = np.array([[2*i, 2*i+1] for i in right_joint_indices]).flatten()
    
    left_coords = pose_coords[:, left_coord_indices]
    right_coords = pose_coords[:, right_coord_indices]
    
    left_motion_mag = np.sqrt(np.sum(np.diff(left_coords, axis=0)**2, axis=1))
    right_motion_mag = np.sqrt(np.sum(np.diff(right_coords, axis=0)**2, axis=1))
    
    asymmetry = np.abs(np.mean(left_motion_mag) - np.mean(right_motion_mag)) / (np.mean(pose_speed) + 1e-6)
    
    features.append(asymmetry)  # 4: High in autism (asymmetric stimming)
    
    # ===== 4. JERKINESS (Smoothness of motion) =====
    # Acceleration (second derivative) - high acceleration = jerky
    if len(pose_speed) > 2:
        accel = np.diff(pose_speed, n=2)  # Second derivative
        jerkiness = np.mean(np.abs(accel)) / (np.mean(pose_speed) + 1e-6)
    else:
        jerkiness = 0.0
    
    features.append(jerkiness)  # 5: High in autism (jerky movements)
    
    # ===== 5. MOTION CONSISTENCY (How similar are successive frames?) =====
    # Entropy of optical flow direction - low entropy = consistent direction
    motion_feats_flat = motion_feats.reshape(-1, motion_feats.shape[-1])  # Flatten spatial dims
    if motion_feats_flat.size > 0:
        # Compute motion angles
        dy_flat, dx_flat = motion_feats_flat[:, 0], motion_feats_flat[:, 1] if motion_feats_flat.shape[1] > 1 else 0
        angles = np.arctan2(dy_flat, dx_flat + 1e-6)
        # Quantize angles to 8 bins
        angle_bins = np.digitize(angles, np.linspace(-np.pi, np.pi, 9))
        # Entropy of angle distribution
        angle_entropy = entropy(np.bincount(angle_bins, minlength=9) + 1e-6)
        consistency = 1.0 - (angle_entropy / np.log(9))  # Normalize to [0, 1]
    else:
        consistency = 0.0
    
    features.append(consistency)  # 6: High in autism (repetitive direction)
    
    # ===== 6. AMPLITUDE DISTRIBUTION (How variable are movements?) =====
    # Coefficient of variation of movement magnitudes
    amplitude_cv = np.std(pose_speed) / (np.mean(pose_speed) + 1e-6)
    features.append(amplitude_cv)  # 7: Low in autism (stereotyped amplitude)
    
    # ===== 7. HAND FOCUS (How much motion in hands vs body?) =====
    # Hand indices: 4, 5 (left), 1, 2 (right) in 17-joint model
    hand_joint_indices = [1, 2, 4, 5]
    hand_coord_indices = np.array([[2*i, 2*i+1] for i in hand_joint_indices]).flatten()
    body_coord_indices = np.setdiff1d(np.arange(34), hand_coord_indices)
    
    if len(hand_coord_indices) > 0 and len(body_coord_indices) > 0:
        hand_motion = np.sqrt(np.sum(pose_motion[:, hand_coord_indices]**2, axis=1))
        body_motion = np.sqrt(np.sum(pose_motion[:, body_coord_indices]**2, axis=1))
        hand_focus = np.mean(hand_motion) / (np.mean(body_motion) + 1e-6)
    else:
        hand_focus = 0.0
    
    features.append(hand_focus)  # 8: High in autism (hand stimming)
    
    # ===== 8. MOTION BURST PATTERNS (Sudden changes?) =====
    # Detect motion bursts (rapid increases in speed)
    if len(pose_speed) > 5:
        speed_diff = np.diff(pose_speed)
        positive_bursts = np.sum(speed_diff > np.std(speed_diff)) / len(speed_diff)
        burst_intensity = np.mean(np.maximum(speed_diff, 0)) / (np.mean(pose_speed) + 1e-6)
    else:
        positive_bursts = 0.0
        burst_intensity = 0.0
    
    features.append(positive_bursts)  # 9: Pattern of motion bursts
    features.append(burst_intensity)  # 10: Intensity of bursts
    
    # ===== 9. OVERALL MOTION MAGNITUDE =====
    mean_speed = np.mean(pose_speed)
    features.append(mean_speed)  # 11: Overall activity level
    
    # ===== 10. MOTION ENTROPY (Unpredictability) =====
    # High entropy = varied movements; Low = repetitive
    if len(pose_speed) > 10:
        speed_bins = np.digitize(pose_speed, np.percentile(pose_speed, np.linspace(0, 100, 11)))
        motion_entropy = entropy(np.bincount(speed_bins, minlength=11) + 1e-6)
    else:
        motion_entropy = 0.0
    
    features.append(motion_entropy)  # 12: Motion predictability
    
    # ===== 11. NORMALIZED OPTICAL FLOW MAGNITUDE =====
    optical_flow_mean = np.mean(np.abs(motion_feats))
    features.append(optical_flow_mean)  # 13: Raw motion intensity
    
    return np.array(features)


def extract_all_autism_features(videos_list, pose_data_list, motion_data_list):
    """
    Batch extract autism features for all videos
    
    Returns:
        X_autism: (N, 13) feature matrix
    """
    X_autism = []
    
    for i, (video, pose, motion) in enumerate(zip(videos_list, pose_data_list, motion_data_list)):
        try:
            features = extract_autism_features(video, motion, pose)
            X_autism.append(features)
        except Exception as e:
            print(f"  Error extracting features from video {i}: {e}")
            X_autism.append(np.zeros(13))
    
    return np.array(X_autism)


if __name__ == "__main__":
    print("Autism-Specific Feature Extraction Module")
    print("Features: Motion concentration, Periodicity, Asymmetry, Jerkiness, Consistency,")
    print("          Amplitude CV, Hand focus, Motion bursts, Speed, Entropy, Optical flow")
