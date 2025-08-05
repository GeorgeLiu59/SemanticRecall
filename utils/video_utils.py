"""Video processing utilities for the semantic video understanding system."""

import cv2
import numpy as np
from typing import List, Tuple
import os


def get_video_info(video_path: str) -> dict:
    """Get basic video information."""
    cap = cv2.VideoCapture(video_path)
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info


def extract_frames(video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
    """Extract specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames


def detect_shot_boundaries(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """Detect shot boundaries using frame difference."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    boundaries = []
    prev_frame = None
    scene_start = 0.0
    
    for frame_idx in range(0, total_frames, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            mean_diff = np.mean(diff)
            
            if mean_diff > threshold:
                boundaries.append((scene_start, current_time))
                scene_start = current_time
        
        prev_frame = frame.copy()
    
    cap.release()
    return boundaries 