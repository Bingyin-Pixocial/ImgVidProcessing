"""
Video Analyzer Module

This module provides functions to analyze video files and extract statistics
about frame rates and frame counts.
"""

import os
import cv2
import glob
from typing import Dict, List, Tuple, Optional


def get_video_info(video_path: str) -> Tuple[float, int]:
    """
    Extract FPS and frame count from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (fps, frame_count)
        
    Raises:
        ValueError: If video cannot be opened or read
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        # Get FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get total frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return fps, frame_count
    
    finally:
        cap.release()


def analyze_videos_in_folder(folder_path: str, recursive: bool = True) -> Dict:
    """
    Analyze all MP4 files in a folder and collect statistics.
    
    Args:
        folder_path: Path to the folder containing MP4 files
        recursive: Whether to search subdirectories recursively
        
    Returns:
        Dictionary with video statistics in the format:
        {
            "num_videos": int,
            "max_fps": float,
            "min_fps": float,
            "avg_fps": float,
            "max_frames": int,
            "min_frames": int,
            "avg_frames": float
        }
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all MP4 files (case insensitive)
    if recursive:
        mp4_patterns = [
            os.path.join(folder_path, "**", "*.mp4"),
            os.path.join(folder_path, "**", "*.MP4")
        ]
    else:
        mp4_patterns = [
            os.path.join(folder_path, "*.mp4"),
            os.path.join(folder_path, "*.MP4")
        ]
    
    video_files = []
    for pattern in mp4_patterns:
        video_files.extend(glob.glob(pattern, recursive=recursive))
    
    # Remove duplicates (in case of case-insensitive filesystem)
    video_files = list(set(video_files))
    
    if not video_files:
        return {
            "num_videos": 0,
            "max_fps": 0.0,
            "min_fps": 0.0,
            "avg_fps": 0.0,
            "max_frames": 0,
            "min_frames": 0,
            "avg_frames": 0.0
        }
    
    fps_values = []
    frame_counts = []
    
    for video_path in video_files:
        try:
            fps, frame_count = get_video_info(video_path)
            fps_values.append(fps)
            frame_counts.append(frame_count)
        except Exception:
            # Skip files that can't be processed
            continue
    
    if not fps_values:
        return {
            "num_videos": 0,
            "max_fps": 0.0,
            "min_fps": 0.0,
            "avg_fps": 0.0,
            "max_frames": 0,
            "min_frames": 0,
            "avg_frames": 0.0
        }
    
    # Calculate statistics
    stats = {
        "num_videos": len(fps_values),
        "max_fps": max(fps_values),
        "min_fps": min(fps_values),
        "avg_fps": sum(fps_values) / len(fps_values),
        "max_frames": max(frame_counts),
        "min_frames": min(frame_counts),
        "avg_frames": sum(frame_counts) / len(frame_counts)
    }
    
    return stats


def analyze_video_list(video_paths: List[str]) -> Dict:
    """
    Analyze a list of video files and collect statistics.
    
    Args:
        video_paths: List of paths to video files
        
    Returns:
        Dictionary with video statistics in the same format as analyze_videos_in_folder
    """
    fps_values = []
    frame_counts = []
    
    for video_path in video_paths:
        try:
            fps, frame_count = get_video_info(video_path)
            fps_values.append(fps)
            frame_counts.append(frame_count)
        except Exception:
            # Skip files that can't be processed
            continue
    
    if not fps_values:
        return {
            "num_videos": 0,
            "max_fps": 0.0,
            "min_fps": 0.0,
            "avg_fps": 0.0,
            "max_frames": 0,
            "min_frames": 0,
            "avg_frames": 0.0
        }
    
    # Calculate statistics
    stats = {
        "num_videos": len(fps_values),
        "max_fps": max(fps_values),
        "min_fps": min(fps_values),
        "avg_fps": sum(fps_values) / len(fps_values),
        "max_frames": max(frame_counts),
        "min_frames": min(frame_counts),
        "avg_frames": sum(frame_counts) / len(frame_counts)
    }
    
    return stats