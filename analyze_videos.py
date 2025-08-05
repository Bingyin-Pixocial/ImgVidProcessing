#!/usr/bin/env python3
"""
Video Analysis Script

This script analyzes MP4 files in a given folder and collects statistics about
their frame rates (FPS) and frame counts.

Usage:
    python analyze_videos.py --folder /path/to/videos
    
Returns:
    Dictionary with statistics about the videos found
"""

import os
import argparse
import cv2
import glob
from typing import Dict, List, Tuple


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


def analyze_videos_in_folder(folder_path: str) -> Dict:
    """
    Analyze all MP4 files in a folder and collect statistics.
    
    Args:
        folder_path: Path to the folder containing MP4 files
        
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
    mp4_patterns = [
        os.path.join(folder_path, "*.mp4"),
        os.path.join(folder_path, "*.MP4"),
        os.path.join(folder_path, "**", "*.mp4"),
        os.path.join(folder_path, "**", "*.MP4")
    ]
    
    video_files = []
    for pattern in mp4_patterns:
        video_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates (in case of case-insensitive filesystem)
    video_files = list(set(video_files))
    
    if not video_files:
        print(f"No MP4 files found in {folder_path}")
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
    failed_videos = []
    
    print(f"Found {len(video_files)} MP4 files. Analyzing...")
    
    for i, video_path in enumerate(video_files):
        try:
            print(f"Processing {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
            fps, frame_count = get_video_info(video_path)
            fps_values.append(fps)
            frame_counts.append(frame_count)
            print(f"  FPS: {fps:.2f}, Frames: {frame_count}")
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
            failed_videos.append(video_path)
    
    if failed_videos:
        print(f"\nFailed to process {len(failed_videos)} videos:")
        for failed_video in failed_videos:
            print(f"  - {failed_video}")
    
    if not fps_values:
        print("No videos could be processed successfully.")
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


def print_statistics(stats: Dict):
    """
    Print video statistics in a formatted way.
    
    Args:
        stats: Dictionary containing video statistics
    """
    print("\n" + "="*50)
    print("VIDEO ANALYSIS RESULTS")
    print("="*50)
    print(f"Number of videos processed: {stats['num_videos']}")
    print("\nFPS Statistics:")
    print(f"  Maximum FPS: {stats['max_fps']:.2f}")
    print(f"  Minimum FPS: {stats['min_fps']:.2f}")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    print("\nFrame Count Statistics:")
    print(f"  Maximum frames: {stats['max_frames']}")
    print(f"  Minimum frames: {stats['min_frames']}")
    print(f"  Average frames: {stats['avg_frames']:.1f}")
    print("="*50)


def main():
    """
    Main function that handles command line arguments and executes the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze MP4 files in a folder to collect FPS and frame count statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_videos.py --folder /path/to/videos
  python analyze_videos.py --folder ./video_dataset --output stats.txt
        """
    )
    
    parser.add_argument(
        "--folder", 
        type=str, 
        required=True,
        help="Path to the folder containing MP4 files"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Optional: Save results to a text file"
    )
    
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Search for MP4 files recursively in subdirectories (default: enabled)"
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze videos
        stats = analyze_videos_in_folder(args.folder)
        
        # Print results
        print_statistics(stats)
        
        # Optionally save to file
        if args.output:
            with open(args.output, 'w') as f:
                f.write("Video Analysis Results\n")
                f.write("="*50 + "\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            print(f"\nResults saved to: {args.output}")
        
        # Also print the raw dictionary for programmatic use
        print(f"\nRaw dictionary result:")
        print(stats)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())