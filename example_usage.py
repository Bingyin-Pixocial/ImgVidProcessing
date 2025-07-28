#!/usr/bin/env python3
"""
Example usage of the frame extraction and video creation functions.
"""

from extract_frame import extract_frames_from_video, create_video_from_frames

def main():
    # Example 1: Extract frames from a video
    print("=== Example 1: Extracting frames from video ===")
    try:
        extract_frames_from_video(
            video_path="input_video.mp4",  # Replace with your video file
            output_dir="extracted_frames",
            filename_prefix="frame"
        )
    except FileNotFoundError:
        print("Video file not found. Please provide a valid video file path.")
    
    # Example 2: Create video from frames
    print("\n=== Example 2: Creating video from frames ===")
    try:
        create_video_from_frames(
            frames_dir="extracted_frames",  # Directory containing frame images
            output_video_path="output_video.mp4",
            filename_pattern="frame_*.jpg",  # Pattern to match frame files
            fps=30,  # Frames per second
            frame_size=(1920, 1080)  # Optional: specify output resolution
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the frames directory exists and contains frame images.")

if __name__ == "__main__":
    main() 