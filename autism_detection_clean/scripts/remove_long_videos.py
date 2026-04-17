"""
Remove videos longer than 10 seconds from dataset_new/autism/ and dataset_new/normal/
"""

import os
import cv2

def get_video_duration(video_path):
    """Get video duration in seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps == 0:
            return None
        
        duration = frame_count / fps
        return duration
    except:
        return None

def remove_long_videos(folder_path, max_duration=10):
    """Remove videos longer than max_duration seconds"""
    deleted_videos = []
    kept_videos = []
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return deleted_videos, kept_videos
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    print(f"\nProcessing {folder_path}...")
    print(f"Found {len(video_files)} videos")
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        duration = get_video_duration(video_path)
        
        if duration is None:
            print(f"  [SKIP] {video_file} - Could not read duration")
            continue
        
        if duration > max_duration:
            try:
                os.remove(video_path)
                deleted_videos.append((video_file, duration))
                print(f"  [DELETED] {video_file} ({duration:.2f}s)")
            except Exception as e:
                print(f"  [ERROR] {video_file} - {e}")
        else:
            kept_videos.append((video_file, duration))
            print(f"  [KEPT] {video_file} ({duration:.2f}s)")
    
    return deleted_videos, kept_videos

# Process autism folder
autism_folder = 'dataset_new/autism'
autism_deleted, autism_kept = remove_long_videos(autism_folder, max_duration=10)

# Process normal folder
normal_folder = 'dataset_new/normal'
normal_deleted, normal_kept = remove_long_videos(normal_folder, max_duration=10)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nAutism Folder:")
print(f"  Deleted: {len(autism_deleted)} videos")
print(f"  Kept: {len(autism_kept)} videos")

print(f"\nNormal Folder:")
print(f"  Deleted: {len(normal_deleted)} videos")
print(f"  Kept: {len(normal_kept)} videos")

print(f"\nTotal:")
total_deleted = len(autism_deleted) + len(normal_deleted)
total_kept = len(autism_kept) + len(normal_kept)
print(f"  Total Deleted: {total_deleted} videos")
print(f"  Total Kept: {total_kept} videos")

if total_deleted > 0:
    print(f"\nDeleted Videos:")
    for video, duration in autism_deleted:
        print(f"  autism/{video} ({duration:.2f}s)")
    for video, duration in normal_deleted:
        print(f"  normal/{video} ({duration:.2f}s)")
