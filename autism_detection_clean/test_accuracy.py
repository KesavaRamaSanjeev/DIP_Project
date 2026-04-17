#!/usr/bin/env python
"""
Test accuracy of the ensemble autism detection model on entire dataset
Usage: python test_accuracy.py
"""

import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models.classifier import IntegratedModel
from models.pose_estimation import get_pose_features

def load_ensemble_model(model_path='checkpoints/ensemble_model.joblib'):
    """Load the trained ensemble (SVM + RF) model"""
    if not os.path.exists(model_path):
        print(f"✗ Ensemble model not found at {model_path}")
        return None
    
    model = joblib.load(model_path)
    print(f"✓ Loaded ensemble model from {model_path}")
    return model

def load_feature_extractor(model_path='checkpoints/best_model.pth', device='cpu'):
    """Load the neural network for feature extraction"""
    model = IntegratedModel(num_classes=1, num_frames=16, hidden_size=64)
    
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt)
            print(f"✓ Loaded feature extractor from {model_path}")
        except:
            print(f"⚠ Could not load weights from {model_path}, using random initialization")
    
    model.eval()
    return model.to(device)

def extract_video_features(video_path, extractor_model, device, num_frames=16):
    """Extract combined features from video (LSTM + Motion)"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames
    sampled_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []
    
    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) < num_frames:
        return None
    
    frames = np.array(frames, dtype=np.float32)
    
    # Compute optical flow
    optical_flow = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i+1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        optical_flow.append(flow)
    
    optical_flow = np.array(optical_flow, dtype=np.float32)
    
    # Get LSTM features
    with torch.no_grad():
        frames_tensor = torch.from_numpy(frames[:-1]).unsqueeze(1).to(device)
        flow_tensor = torch.from_numpy(optical_flow).permute(0, 3, 1, 2).to(device)
        
        lstm_features = extractor_model.lstm_extractor(
            frames_tensor, flow_tensor
        ).cpu().numpy()
    
    # Get motion features
    motion_features = extract_motion_features_from_flow(optical_flow)
    
    # Combine features
    features = np.concatenate([lstm_features, motion_features])
    
    return features

def extract_motion_features_from_flow(optical_flow):
    """Extract 8 motion statistics from optical flow"""
    
    flow_mag = np.sqrt(optical_flow[..., 0]**2 + optical_flow[..., 1]**2)
    
    features = np.array([
        np.mean(flow_mag),                              # 0: magnitude mean
        np.std(flow_mag),                               # 1: magnitude std
        np.max(flow_mag),                               # 2: magnitude max
        np.mean(optical_flow[..., 0]),                  # 3: flow_x mean
        np.mean(optical_flow[..., 1]),                  # 4: flow_y mean
        np.mean(flow_mag > 0),                          # 5: motion proportion
        np.var(flow_mag),                               # 6: temporal variance
        np.mean(np.abs(np.diff(flow_mag, axis=0)))      # 7: smoothness
    ], dtype=np.float32)
    
    return features

def get_all_videos(base_path):
    """Get all video paths with labels"""
    videos = []
    
    # Normal videos (label = 0)
    normal_dir = os.path.join(base_path, 'normal')
    if os.path.exists(normal_dir):
        for fname in os.listdir(normal_dir):
            if fname.endswith('.avi'):
                videos.append((os.path.join(normal_dir, fname), 0, 'NORMAL'))
    
    # Autism videos (label = 1)
    autism_dir = os.path.join(base_path, 'autism')
    if os.path.exists(autism_dir):
        for category in os.listdir(autism_dir):
            cat_path = os.path.join(autism_dir, category)
            if os.path.isdir(cat_path):
                for fname in os.listdir(cat_path):
                    if fname.endswith('.mp4') or fname.endswith('.avi'):
                        videos.append((os.path.join(cat_path, fname), 1, f'AUTISM ({category})'))
    
    return videos

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load models
    ensemble_model = load_ensemble_model()
    feature_extractor = load_feature_extractor(device=device)
    
    if ensemble_model is None or feature_extractor is None:
        print("✗ Failed to load models")
        return
    
    # Get all videos
    base_path = 'cleanest/cleanest'
    videos = get_all_videos(base_path)
    
    print(f"\n📊 Found {len(videos)} videos to test")
    print(f"   Normal: {sum(1 for _, label, _ in videos if label == 0)}")
    print(f"   Autism: {sum(1 for _, label, _ in videos if label == 1)}\n")
    
    # Test predictions
    y_true = []
    y_pred = []
    y_proba = []
    predictions_list = []
    
    successful = 0
    failed = 0
    
    print("=" * 80)
    print("TESTING VIDEOS")
    print("=" * 80)
    
    for video_path, true_label, label_name in videos:
        try:
            # Extract features
            features = extract_video_features(video_path, feature_extractor, device, num_frames=16)
            
            if features is None:
                print(f"⚠ SKIP: {os.path.basename(video_path)} - Could not extract features")
                failed += 1
                continue
            
            # Reshape for sklearn
            features_2d = features.reshape(1, -1)
            
            # Predict
            pred = ensemble_model.predict(features_2d)[0]
            proba = ensemble_model.predict_proba(features_2d)[0]
            
            y_true.append(true_label)
            y_pred.append(pred)
            y_proba.append(proba[1])  # Probability of autism
            
            pred_name = "AUTISM" if pred == 1 else "NORMAL"
            confidence = max(proba) * 100
            
            match = "✓" if pred == true_label else "✗"
            predictions_list.append({
                'video': os.path.basename(video_path),
                'true_label': label_name,
                'predicted': pred_name,
                'confidence': confidence,
                'correct': pred == true_label
            })
            
            print(f"{match} {os.path.basename(video_path)[:50]:50s} | TRUE: {label_name:20s} | PRED: {pred_name:20s} ({confidence:.1f}%)")
            
            successful += 1
            
        except Exception as e:
            print(f"✗ ERROR: {os.path.basename(video_path)} - {str(e)[:60]}")
            failed += 1
    
    print("=" * 80)
    
    # Calculate metrics
    if len(y_true) == 0:
        print("✗ No successful predictions")
        return
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = None
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    
    print(f"\n✓ Successful predictions: {successful}/{len(videos)}")
    print(f"✗ Failed predictions: {failed}/{len(videos)}")
    
    print(f"\n📈 METRICS:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}  (True Positives / Predicted Positives)")
    print(f"  Recall:    {recall:.2%}  (True Positives / Actual Positives) - Sensitivity")
    print(f"  F1-Score:  {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC:   {auc:.4f}")
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"  True Negatives (Normal correctly classified):   {cm[0, 0]}")
    print(f"  False Positives (Normal misclassified as Autism): {cm[0, 1]}")
    print(f"  False Negatives (Autism misclassified as Normal): {cm[1, 0]}")
    print(f"  True Positives (Autism correctly classified):   {cm[1, 1]}")
    
    # Specificity
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    print(f"\n  Specificity: {specificity:.2%}  (True Negatives / Actual Negatives)")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(videos),
        'successful': successful,
        'failed': failed,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'roc_auc': float(auc) if auc is not None else None
        },
        'confusion_matrix': {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        },
        'predictions': predictions_list
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to test_results.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
