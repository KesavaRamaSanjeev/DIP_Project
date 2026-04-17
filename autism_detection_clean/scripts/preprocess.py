import cv2
import os
import numpy as np
import torch
import random


def compute_optical_flow_features(frames_gray):
    """
    Compute optical flow between consecutive grayscale frames.
    Returns motion statistics: mean magnitude, std, max, periodicity estimate.
    """
    flow_magnitudes = []
    flow_angles = []
    
    for i in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(mag.mean())
        flow_angles.append(ang.mean())
    
    if len(flow_magnitudes) == 0:
        return np.zeros(8, dtype=np.float32)
    
    mags = np.array(flow_magnitudes)
    angs = np.array(flow_angles)
    
    # Motion statistics
    features = [
        mags.mean(),           # Average motion intensity
        mags.std(),            # Motion variability
        mags.max(),            # Peak motion
        np.median(mags),       # Median motion
        angs.std(),            # Direction variability (high = spinning/flapping)
    ]
    
    # Periodicity: autocorrelation of motion signal
    if len(mags) > 4:
        mags_centered = mags - mags.mean()
        autocorr = np.correlate(mags_centered, mags_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
            # Find first peak after zero crossing (periodicity strength)
            peaks = []
            for j in range(1, len(autocorr) - 1):
                if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1] and autocorr[j] > 0.1:
                    peaks.append(autocorr[j])
            periodicity = max(peaks) if peaks else 0.0
        else:
            periodicity = 0.0
    else:
        periodicity = 0.0
    
    features.append(periodicity)  # Repetitive motion indicator
    
    # Acceleration: changes in motion magnitude
    if len(mags) > 1:
        accel = np.diff(mags)
        features.append(np.abs(accel).mean())   # Jerkiness
        features.append(accel.std())             # Acceleration variability
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


def extract_frames(video_path, num_frames=16, training=True):
    """
    Extracts uniformly sampled frames and motion features.
    Now using Two-Stream approach: Channel 0 is Diff, Channel 1 is Dense Flow.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros((2, num_frames, 112, 112)), torch.zeros(8)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return torch.zeros((2, num_frames, 112, 112)), torch.zeros(8)
    
    # Read ALL frames for motion analysis (subsample later for CNN)
    all_frames_gray = []
    all_frames_color = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (112, 112))
        color = cv2.resize(frame, (112, 112))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        all_frames_gray.append(gray)
        all_frames_color.append(color)
    
    cap.release()
    
    if len(all_frames_gray) == 0:
        return torch.zeros((2, num_frames, 112, 112)), torch.zeros(8)
    
    # 1. Compute or Load motion features (Shape Validation)
    cache_path = video_path + ".motion.npy"
    motion_features = None
    if os.path.exists(cache_path):
        try:
            temp = np.load(cache_path)
            if hasattr(temp, 'shape') and temp.shape == (8,):
                motion_features = temp
            else:
                os.remove(cache_path) # Purge incompatible cache
        except:
            pass
            
    if motion_features is None:
        motion_features = compute_optical_flow_features(all_frames_gray)
        np.save(cache_path, motion_features)
    
    # 2. Sequential Sampling for CNN
    n = len(all_frames_color)
    if training:
        if n > num_frames:
            start_idx = random.randint(0, n - num_frames)
            indices = list(range(start_idx, start_idx + num_frames))
        else:
            indices = list(range(n))
    else:
        if n >= num_frames:
            indices = np.linspace(0, n - 1, num_frames, dtype=int)
        else:
            indices = list(range(n))
    
    frames = [all_frames_color[i] for i in indices]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]
    
    # Augmentation
    if training:
        if random.random() > 0.5:
            frames = [np.fliplr(f).copy() for f in frames]
        if random.random() > 0.5:
            alpha = 1.0 + random.uniform(-0.2, 0.2)
            beta = random.uniform(-20, 20)
            frames = [cv2.convertScaleAbs(f, alpha=alpha, beta=beta) for f in frames]
    
    # 3. Two-Stream Generation: Temporal Difference + Dense Flow
    diff_frames = []
    flow_frames = []
    
    for i in range(len(frames) - 1):
        # Difference
        diff = cv2.absdiff(frames[i], frames[i+1])
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_frames.append(diff_gray)
        
        # Flow
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_frames.append(mag_norm)

    if len(diff_frames) > 0:
        diff_frames.append(diff_frames[-1])
        flow_frames.append(flow_frames[-1])
    else:
        diff_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        flow_frames = [np.zeros_like(diff_frames[0]) for _ in frames]

    stacked = [np.stack([d, f], axis=0) for d, f in zip(diff_frames, flow_frames)]
    video_array = np.array(stacked, dtype=np.float32) / 255.0 
    video_tensor = torch.from_numpy(video_array).permute(1, 0, 2, 3).float()
    motion_tensor = torch.from_numpy(motion_features).float()
    
    return video_tensor, motion_tensor


if __name__ == "__main__":
    if os.path.exists('dataset/normal/normal_0.mp4'):
        tensor, motion = extract_frames('dataset/normal/normal_0.mp4')
        print(f"Video tensor shape: {tensor.shape}")
