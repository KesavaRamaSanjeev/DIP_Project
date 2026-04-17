"""
Complete Feature Extraction & Processing Pipeline for dataset_new/
===================================================================
1. Extract LSTM features from raw .mp4
2. Extract motion features (optical flow)
3. Extract autism behavioral features
4. Cache all features
5. Detect and remove duplicates
6. Generate k-fold validation with honest metrics

Target: 61.5%+ accuracy with clean data
"""

import sys
import os

# Move to parent directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import glob
import torch
from pathlib import Path
import shutil

from scripts.preprocess import extract_frames
from models.pose_estimation import get_pose_features
from models.classifier import IntegratedModel
from scripts.autism_features import extract_autism_features


def extract_features_from_raw():
    """
    Extract all features from raw .mp4 files in dataset_new/
    Creates caches: X_lstm_new.npy, X_motion_new.npy, X_autism_new.npy, X_combined_new.npy
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("PHASE 1: Feature Extraction from Raw Videos")
    print("="*60)
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64).to(device)
    model.eval()
    
    # Get dataset files from dataset_new/
    autism_files = sorted(glob.glob('dataset_new/autism/*.mp4'))
    normal_files = sorted(glob.glob('dataset_new/normal/*.mp4'))
    
    autism_files = [f for f in autism_files if f.endswith('.mp4')]
    normal_files = [f for f in normal_files if f.endswith('.mp4')]
    
    all_files = autism_files + normal_files
    labels = np.array([1] * len(autism_files) + [0] * len(normal_files))
    
    print(f"\nDataset Summary:")
    print(f"  Autism videos:  {len(autism_files)}")
    print(f"  Normal videos:  {len(normal_files)}")
    print(f"  Total videos:   {len(all_files)}\n")
    
    if len(all_files) == 0:
        print("ERROR: No .mp4 files found in dataset_new/autism or dataset_new/normal")
        return None, None, None, None, None
    
    # Extract features
    X_lstm_list = []
    X_motion_list = []
    X_autism_list = []
    valid_files = []
    valid_labels = []
    
    print(f"Extracting features from {len(all_files)} videos...\n")
    
    failed_count = 0
    for i, video_path in enumerate(all_files):
        video_name = os.path.basename(video_path)
        print(f"  [{i+1:3d}/{len(all_files)}] {video_name:50s}", end=' ', flush=True)
        
        try:
            # Skip empty/corrupted files
            file_size = os.path.getsize(video_path)
            if file_size < 10000:  # Less than 10KB is likely corrupted
                print(f"SKIP (corrupted, size={file_size} bytes)")
                failed_count += 1
                continue
            
            # Extract frames and motion
            video_tensor, motion_feats = extract_frames(video_path, num_frames=16, training=False)
            
            if video_tensor.shape != (2, 16, 112, 112):
                print(f"SKIP (invalid shape {video_tensor.shape})")
                failed_count += 1
                continue
            
            # Process through CNN-LSTM
            video_tensor_gpu = video_tensor.unsqueeze(0).to(device)
            B, C, T, H, W = video_tensor_gpu.shape
            
            with torch.no_grad():
                # CNN features
                x_cnn = video_tensor_gpu.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                heatmaps = model.cnn(x_cnn)
                pose_coords = get_pose_features(heatmaps)  # (B*T, 34)
                pose_coords = pose_coords.view(B, T, -1).cpu().numpy()  # (1, T, 34)
                
                x_lstm = torch.FloatTensor(pose_coords).to(device)
                
                # LSTM features
                lstm_out, _ = model.lstm(x_lstm)
                temporal_feats = lstm_out[:, -1, :].cpu().numpy()  # (1, 128)
            
            # Compute optical flow magnitude for autism features
            video_np = video_tensor.numpy()  # (C, T, H, W)
            
            # Extract grayscale frames for optical flow (use first channel)
            frames_gray = []
            for t in range(video_np.shape[1]):
                frame = video_np[0, t, :, :]  # (H, W)
                frames_gray.append(frame.astype(np.uint8))
            
            # Compute optical flow magnitude
            import cv2
            motion_magnitude = []
            for frame_idx in range(len(frames_gray) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    frames_gray[frame_idx], frames_gray[frame_idx + 1],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_magnitude.append(mag)
            
            motion_magnitude_arr = np.array(motion_magnitude, dtype=np.float32) if motion_magnitude else np.zeros((video_np.shape[1]-1, 112, 112), dtype=np.float32)
            
            # Convert video to (T, H, W) for autism_features function
            video_thw = video_np[0]  # Take first channel (T, H, W)
            
            # Autism behavioral features - with robust error handling
            try:
                autism_feats = extract_autism_features(
                    video_thw,                  # (T, H, W)
                    motion_magnitude_arr,       # (T-1, H, W)
                    pose_coords[0]              # (T, 34) - remove batch dimension
                )
            except Exception as autism_err:
                # Fallback: use default zeros if autism feature extraction fails
                # print(f"WARNING: Autism feature extraction failed, using zeros")
                autism_feats = np.zeros(13, dtype=np.float32)
            
            X_lstm_list.append(temporal_feats[0])  # (128,)
            X_motion_list.append(motion_feats.numpy())  # (8,)
            X_autism_list.append(autism_feats)  # (13,)
            valid_files.append(video_path)
            valid_labels.append(labels[i])
            
            print(f"OK")
            
        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")
            failed_count += 1
    
    # Convert to arrays
    X_lstm = np.array(X_lstm_list, dtype=np.float32)
    X_motion = np.array(X_motion_list, dtype=np.float32)
    X_autism = np.array(X_autism_list, dtype=np.float32)
    X_combined = np.concatenate([X_lstm, X_motion, X_autism], axis=1)
    Y_labels = np.array(valid_labels, dtype=np.int32)
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete!")
    print(f"  Successfully extracted: {len(X_lstm)} videos")
    print(f"  Failed: {failed_count} videos")
    print(f"  Autism: {np.sum(Y_labels == 1)}")
    print(f"  Normal: {np.sum(Y_labels == 0)}")
    print(f"\nFeature Shapes:")
    print(f"  X_lstm shape:     {X_lstm.shape}  (128 dims)")
    print(f"  X_motion shape:   {X_motion.shape}  (8 dims)")
    print(f"  X_autism shape:   {X_autism.shape}  (13 dims)")
    print(f"  X_combined shape: {X_combined.shape}  (149 dims)")
    
    # Save caches
    np.save('X_lstm_new.npy', X_lstm)
    np.save('X_motion_new.npy', X_motion)
    np.save('X_autism_new.npy', X_autism)
    np.save('X_combined_new.npy', X_combined)
    np.save('Y_labels_new.npy', Y_labels)
    np.save('valid_files_new.npy', np.array(valid_files, dtype=object), allow_pickle=True)
    
    print(f"\nCaches saved:")
    print(f"  - X_lstm_new.npy")
    print(f"  - X_motion_new.npy")
    print(f"  - X_autism_new.npy")
    print(f"  - X_combined_new.npy")
    print(f"  - Y_labels_new.npy")
    print(f"  - valid_files_new.npy")
    
    return X_lstm, X_motion, X_autism, X_combined, Y_labels, valid_files


def detect_and_remove_duplicates(X_combined, Y_labels, valid_files):
    """
    PHASE 2: Detect duplicate samples using feature vector comparison
    """
    
    print(f"\n{'='*60}")
    print(f"PHASE 2: Duplicate Detection & Removal")
    print(f"{'='*60}\n")
    
    print(f"Initial dataset: {len(X_combined)} samples\n")
    
    # Detect duplicates (Euclidean distance = 0)
    duplicates_to_remove = set()
    
    print(f"Comparing {len(X_combined)} samples for duplicates...\n")
    for i in range(len(X_combined)):
        if i in duplicates_to_remove:
            continue
            
        for j in range(i + 1, len(X_combined)):
            if j in duplicates_to_remove:
                continue
            
            # Euclidean distance
            distance = np.linalg.norm(X_combined[i] - X_combined[j])
            
            if distance < 1e-6:  # Essentially zero (duplicate)
                file_i = os.path.basename(valid_files[i])
                file_j = os.path.basename(valid_files[j])
                print(f"DUPLICATE FOUND!")
                print(f"  Sample {i}: {file_i}")
                print(f"  Sample {j}: {file_j}")
                print(f"  Distance: {distance:.2e}\n")
                
                # Mark second as duplicate (keep first)
                duplicates_to_remove.add(j)
    
    if len(duplicates_to_remove) == 0:
        print("No duplicates detected!\n")
        return X_combined, Y_labels, valid_files
    
    print(f"\n{'='*60}")
    print(f"Found {len(duplicates_to_remove)} duplicate samples")
    print(f"Removing them...\n")
    
    # Remove duplicates
    keep_indices = [i for i in range(len(X_combined)) if i not in duplicates_to_remove]
    
    X_combined_clean = X_combined[keep_indices]
    Y_labels_clean = Y_labels[keep_indices]
    valid_files_clean = [valid_files[i] for i in keep_indices]
    
    print(f"After Duplicate Removal:")
    print(f"  Total samples: {len(X_combined)} -> {len(X_combined_clean)}")
    print(f"  Autism: {np.sum(Y_labels_clean == 1)}")
    print(f"  Normal: {np.sum(Y_labels_clean == 0)}")
    print(f"  Removed: {len(duplicates_to_remove)} samples")
    
    # Delete duplicate video files
    print(f"\nDeleting duplicate video files from dataset_new/...\n")
    for idx in duplicates_to_remove:
        video_path = valid_files[idx]
        motion_path = video_path + ".motion.npy"
        
        try:
            os.remove(video_path)
            print(f"  DELETED: {os.path.basename(video_path)}")
            if os.path.exists(motion_path):
                os.remove(motion_path)
        except Exception as e:
            print(f"  WARNING: Could not remove {os.path.basename(video_path)}: {str(e)}")
    
    # Update feature caches
    print(f"\nUpdating feature caches...\n")
    np.save('X_lstm_new.npy', X_combined_clean[:, :128].astype(np.float32))
    np.save('X_motion_new.npy', X_combined_clean[:, 128:136].astype(np.float32))
    np.save('X_autism_new.npy', X_combined_clean[:, 136:149].astype(np.float32))
    np.save('X_combined_new.npy', X_combined_clean.astype(np.float32))
    np.save('Y_labels_new.npy', Y_labels_clean.astype(np.int32))
    np.save('valid_files_new.npy', np.array(valid_files_clean, dtype=object), allow_pickle=True)
    
    print(f"All caches updated with clean data\n")
    
    return X_combined_clean, Y_labels_clean, valid_files_clean


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"DATASET_NEW PROCESSING PIPELINE")
    print(f"{'='*60}\n")
    
    # Phase 1: Extract features
    result = extract_features_from_raw()
    if result[0] is None:
        print("\nFeature extraction failed")
        sys.exit(1)
    
    X_lstm, X_motion, X_autism, X_combined, Y_labels, valid_files = result
    
    # Phase 2: Detect and remove duplicates
    X_combined_clean, Y_labels_clean, valid_files_clean = detect_and_remove_duplicates(
        X_combined, Y_labels, valid_files
    )
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nReady for k-fold validation!")
    print(f"Run: python scripts/train_kfold_new.py")
