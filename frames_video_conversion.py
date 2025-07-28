import os
import argparse
from PIL import Image
import torch
from torchvision.io import read_video
import cv2
import glob
import numpy as np

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    filename_prefix: str = "frame"
):
    """
    Reads a video file and writes each frame to disk as a JPEG.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # read_video returns: (video_frames, audio_frames, info_dict)
    # video_frames: Tensor[T, H, W, C] with dtype uint8, in RGB order
    video_frames, _, _ = read_video(video_path, pts_unit="sec")

    # Loop over all frames and save
    for i, frame in enumerate(video_frames):
        # frame is a torch Tensor of shape (H, W, C)
        img = Image.fromarray(frame.numpy())
        out_path = os.path.join(output_dir, f"{filename_prefix}_{i:05d}.jpg")
        img.save(out_path)

    print(f"Saved {len(video_frames)} frames to {output_dir}/")

def create_video_from_frames(
    frames_dir: str,
    output_video_path: str,
    filename_pattern: str = "frame_*.jpg",
    fps: int = 30,
    frame_size: tuple | None = None
):
    """
    Reads frames from a directory and combines them into a video file.
    
    Args:
        frames_dir: Directory containing the frame images
        output_video_path: Path for the output video file
        filename_pattern: Pattern to match frame files (default: "frame_*.jpg")
        fps: Frames per second for the output video
        frame_size: Tuple of (width, height) for output video. If None, uses first frame size
    """
    # Validate output path
    if not output_video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # If no extension provided, assume it's a directory and create a default filename
        if os.path.isdir(output_video_path) or not os.path.exists(output_video_path):
            output_video_path = os.path.join(output_video_path, "output.mp4")
            print(f"No video extension provided, using: {output_video_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame files sorted by name
    frame_files = glob.glob(os.path.join(frames_dir, filename_pattern))
    frame_files.sort()
    
    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir} matching pattern {filename_pattern}")
    
    print(f"Found {len(frame_files)} frame files")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")
    
    height, width = first_frame.shape[:2]
    
    # Use provided frame_size if specified, otherwise use first frame size
    if frame_size is not None:
        width, height = frame_size
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise ValueError(f"Could not open video writer for {output_video_path}. Make sure the path includes a video file extension (e.g., .mp4)")
    
    # Process each frame
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}, skipping...")
            continue
        
        # Resize frame if necessary
        if frame_size is not None:
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames")
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video or create video from frames")
    parser.add_argument("--mode", choices=["extract", "create"], required=True,
                       help="Mode: 'extract' to extract frames from video, 'create' to create video from frames")
    
    # Arguments for frame extraction
    parser.add_argument("--video_path", help="Path to the input video file (for extract mode)")
    parser.add_argument("--output", help="Directory to save extracted frames (for extract mode)")
    parser.add_argument("--prefix", default="frame", help="Filename prefix for extracted frames (default: frame)")
    
    # Arguments for video creation
    parser.add_argument("--frames_dir", help="Directory containing frame images (for create mode)")
    parser.add_argument("--output_video", help="Path for the output video file (for create mode)")
    parser.add_argument("--output_filename", help="Output video filename (e.g., 'my_video.mp4'). If provided, will be combined with --output_video directory")
    parser.add_argument("--pattern", default="frame_*.jpg", help="Pattern to match frame files (default: frame_*.jpg)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video (default: 30)")
    parser.add_argument("--width", type=int, help="Width of output video (optional)")
    parser.add_argument("--height", type=int, help="Height of output video (optional)")
    
    args = parser.parse_args()
    
    if args.mode == "extract":
        if not args.video_path or not args.output:
            parser.error("extract mode requires --video_path and --output")
        extract_frames_from_video(args.video_path, args.output, args.prefix)
    
    elif args.mode == "create":
        if not args.frames_dir:
            parser.error("create mode requires --frames_dir")
        
        # Handle output path logic
        output_video_path = args.output_video
        
        if not output_video_path:
            parser.error("create mode requires --output_video")
        
        # If custom filename is provided, combine it with the output directory
        if args.output_filename:
            if os.path.isdir(output_video_path):
                # If output_video is a directory, combine with custom filename
                output_video_path = os.path.join(output_video_path, args.output_filename)
            else:
                # If output_video is a file path, replace the filename
                output_dir = os.path.dirname(output_video_path)
                if output_dir:
                    output_video_path = os.path.join(output_dir, args.output_filename)
                else:
                    output_video_path = args.output_filename
            print(f"Using custom filename: {output_video_path}")
        
        frame_size = None
        if args.width and args.height:
            frame_size = (args.width, args.height)
        
        create_video_from_frames(args.frames_dir, output_video_path, args.pattern, args.fps, frame_size)
