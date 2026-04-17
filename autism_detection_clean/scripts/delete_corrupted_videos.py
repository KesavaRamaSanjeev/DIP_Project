"""
Find and delete corrupted MP4 files in dataset_new/autism/ and dataset_new/normal/
Corrupted files: can't be opened by OpenCV (moov atom not found, etc.)
"""

import os
import cv2

def is_video_corrupted(video_path):
    """Check if video file is corrupted by trying to open it"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Try to read some basic properties
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        cap.release()
        
        # If can't get properties, file is corrupted
        if frame_count == 0 or fps == 0 or width == 0 or height == 0:
            return True
        
        return False
    except Exception as e:
        return True

def find_and_delete_corrupted(folder_path):
    """Find corrupted videos and delete them"""
    corrupted_videos = []
    valid_videos = []
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return corrupted_videos, valid_videos
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    print(f"\nProcessing {folder_path}...")
    print(f"Found {len(video_files)} videos")
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(folder_path, video_file)
        
        if is_video_corrupted(video_path):
            try:
                os.remove(video_path)
                corrupted_videos.append(video_file)
                print(f"  [{i+1}/{len(video_files)}] [DELETED] {video_file} (CORRUPTED)")
            except Exception as e:
                print(f"  [{i+1}/{len(video_files)}] [ERROR] {video_file} - Could not delete: {e}")
        else:
            valid_videos.append(video_file)
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(video_files)}] [OK] Checked {i+1} videos... (valid so far: {len(valid_videos)})")
    
    return corrupted_videos, valid_videos

# Process both folders
print("="*70)
print("FINDING AND DELETING CORRUPTED MP4 FILES")
print("="*70)

autism_folder = 'dataset_new/autism'
normal_folder = 'dataset_new/normal'

autism_corrupted, autism_valid = find_and_delete_corrupted(autism_folder)
normal_corrupted, normal_valid = find_and_delete_corrupted(normal_folder)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nAutism Folder:")
print(f"  Valid: {len(autism_valid)} videos")
print(f"  Corrupted & Deleted: {len(autism_corrupted)} videos")

print(f"\nNormal Folder:")
print(f"  Valid: {len(normal_valid)} videos")
print(f"  Corrupted & Deleted: {len(normal_corrupted)} videos")

total_valid = len(autism_valid) + len(normal_valid)
total_deleted = len(autism_corrupted) + len(normal_corrupted)

print(f"\nTotal:")
print(f"  Valid Videos: {total_valid}")
print(f"  Deleted (Corrupted): {total_deleted}")

if total_deleted > 0:
    print(f"\nDeleted Videos:")
    for video in autism_corrupted:
        print(f"  autism/{video}")
    for video in normal_corrupted:
        print(f"  normal/{video}")
else:
    print(f"\nNo corrupted videos found!")
