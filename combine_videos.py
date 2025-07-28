import cv2
import numpy as np
import os
import argparse
from typing import List, Tuple, Union
from PIL import Image
import imageio


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an image, False otherwise
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions


def load_image_as_frame(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    """
    Load an image and resize it to target dimensions as a video frame.
    
    Args:
        image_path: Path to the image file
        target_width: Target width
        target_height: Target height
        
    Returns:
        Resized image as a numpy array in BGR format
    """
    # Load image using PIL first to handle more formats
    pil_image = Image.open(image_path)
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Resize image
    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and BGR format for OpenCV
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array


def get_video_info(video_path: str) -> Tuple[float, int, int, int]:
    """
    Get video information including duration, width, height, and fps.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (duration, width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration, width, height, fps


def get_image_info(image_path: str) -> Tuple[float, int, int, int]:
    """
    Get image information. For images, we return a default duration and fps.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (duration, width, height, fps) - duration is set to 5 seconds, fps to 30
    """
    # Load image to get dimensions
    pil_image = Image.open(image_path)
    width, height = pil_image.size
    
    # For images, we'll use a default duration and fps
    duration = 5.0  # 5 seconds default
    fps = 30.0      # 30 fps default
    
    return duration, width, height, fps


def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize frame to target dimensions.
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
        
    Returns:
        Resized frame
    """
    resized = cv2.resize(frame, (target_width, target_height))
    return resized


def combine_media(input_paths: List[str], output_path: str, target_height: int = 768, target_width: int = 576, output_format: str = "mp4"):
    """
    Combine multiple videos and images side by side.
    
    Args:
        input_paths: List of paths to input videos and images
        output_path: Path for the output video
        target_height: Target height for all media (default: 768)
        target_width: Target width for each media (default: 576)
        output_format: Output format - "mp4" or "gif" (default: "mp4")
    """
    if not input_paths:
        raise ValueError("No input paths provided")
    
    # Classify inputs as images or videos
    media_infos = []
    video_caps = []
    image_frames = []
    
    for input_path in input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        if is_image_file(input_path):
            # Handle image
            duration, width, height, fps = get_image_info(input_path)
            media_infos.append({
                'path': input_path,
                'type': 'image',
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps
            })
            # Load and resize image frame
            image_frame = load_image_as_frame(input_path, target_width, target_height)
            image_frames.append(image_frame)
            video_caps.append(None)  # No video capture for images
        else:
            # Handle video
            duration, width, height, fps = get_video_info(input_path)
            media_infos.append({
                'path': input_path,
                'type': 'video',
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps
            })
            # Open video capture
            cap = cv2.VideoCapture(input_path)
            video_caps.append(cap)
            image_frames.append(None)  # No image frame for videos
    
    # Find minimum duration among videos (ignore image durations for this)
    video_durations = [info['duration'] for info in media_infos if info['type'] == 'video']
    if video_durations:
        min_duration = min(video_durations)
    else:
        # If only images, use the shortest image duration
        min_duration = min(info['duration'] for info in media_infos)
    
    # Use maximum fps from videos, or default to 30 if only images
    video_fps = [info['fps'] for info in media_infos if info['type'] == 'video']
    if video_fps:
        output_fps = max(video_fps)
    else:
        output_fps = 30.0  # Default fps for image-only combinations
    
    print(f"Minimum duration: {min_duration:.2f} seconds")
    print(f"Output FPS: {output_fps:.2f}")
    
    # Calculate total width for output video
    total_width = len(input_paths) * target_width
    
    # Calculate frame count based on minimum duration
    total_frames = int(min_duration * output_fps)
    
    print(f"Processing {total_frames} frames...")
    
    # Initialize output based on format
    if output_format.lower() == "gif":
        frames_list = []
        print("Creating GIF frames...")
    else:
        # Create video writer for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (total_width, target_height))
    
    try:
        for frame_idx in range(total_frames):
            # Calculate current time in seconds
            current_time = frame_idx / output_fps
            
            # Create combined frame
            combined_frame = np.zeros((target_height, total_width, 3), dtype=np.uint8)
            current_x = 0
            
            for i, (media_info, cap, image_frame) in enumerate(zip(media_infos, video_caps, image_frames)):
                if media_info['type'] == 'image':
                    # Use the static image frame
                    combined_frame[:, current_x:current_x + target_width] = image_frame
                    current_x += target_width
                else:
                    # Handle video with frame rate synchronization
                    if current_time <= min_duration:
                        # Calculate the frame index for this video at the current time
                        video_frame_idx = int(current_time * media_info['fps'])
                        
                        # Ensure we don't exceed the video's frame count
                        frame_count = int(media_info['duration'] * media_info['fps'])
                        video_frame_idx = min(video_frame_idx, frame_count - 1)
                        
                        # Seek to the specific frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                        ret, frame = cap.read()
                        
                        if ret:
                            # Resize frame to target dimensions
                            resized_frame = resize_frame(frame, target_width, target_height)
                            
                            # Place frame in combined frame
                            combined_frame[:, current_x:current_x + target_width] = resized_frame
                            current_x += target_width
                        else:
                            # If seeking failed, use black frame
                            combined_frame[:, current_x:current_x + target_width] = 0
                            current_x += target_width
                    else:
                        # Video has ended (beyond minimum duration), use black frame
                        combined_frame[:, current_x:current_x + target_width] = 0
                        current_x += target_width
            
            # Handle frame based on output format
            if output_format.lower() == "gif":
                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames_list.append(pil_image)
            else:
                # Write combined frame to video
                out.write(combined_frame)
            
            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
    
    finally:
        # Clean up video captures
        for cap in video_caps:
            if cap is not None:
                cap.release()
        
        # Handle final output based on format
        if output_format.lower() == "gif":
            # Calculate duration per frame for GIF
            duration_per_frame = 1000 / output_fps  # milliseconds
            print(f"Saving GIF with {len(frames_list)} frames at {duration_per_frame:.1f}ms per frame...")
            imageio.mimsave(output_path, frames_list, duration=duration_per_frame/1000)
        else:
            # Release video writer
            out.release()
    
    print(f"Media combination completed! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Combine multiple videos and images side by side')
    parser.add_argument('--inputs', nargs='+', required=True, 
                       help='List of input video and image paths')
    parser.add_argument('--output', required=True, 
                       help='Output video path')
    parser.add_argument('--height', type=int, default=768,
                       help='Target height for media (default: 768)')
    parser.add_argument('--width', type=int, default=576,
                       help='Target width for each media (default: 576)')
    parser.add_argument('--format', choices=['mp4', 'gif'], default='mp4',
                       help='Output format (default: mp4)')
    
    args = parser.parse_args()
    
    try:
        combine_media(args.inputs, args.output, args.height, args.width, args.format)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage (uncomment to use directly without command line arguments)
    # input_paths = ["video1.mp4", "image1.jpg", "video2.mp4", "image2.png"]
    # output_path = "combined_media.mp4"  # or "combined_media.gif"
    # combine_media(input_paths, output_path, target_height=768, target_width=576, output_format="mp4")
    
    exit(main())
