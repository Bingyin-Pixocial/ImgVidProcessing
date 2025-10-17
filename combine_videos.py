import cv2
import numpy as np
import os
import argparse
from typing import List, Tuple
from PIL import Image
import imageio


def is_image_file(file_path: str) -> bool:
    """Check if a file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions


def load_image_as_frame(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    """Load an image and resize it to target dimensions as a video frame."""
    pil_image = Image.open(image_path)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def get_video_info(video_path: str) -> Tuple[float, int, int, int]:
    """Get video duration, width, height, and FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, width, height, fps


def get_image_info(image_path: str) -> Tuple[float, int, int, int]:
    """Return dummy video-like info for an image."""
    pil_image = Image.open(image_path)
    width, height = pil_image.size
    return 5.0, width, height, 30.0  # duration, width, height, fps


def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize a frame to the target dimensions."""
    return cv2.resize(frame, (target_width, target_height))


def combine_media(input_paths: List[str], output_path: str,
                  target_height: int = 768, target_width: int = 576,
                  output_format: str = "mp4"):
    """Combine multiple videos and images side by side into a single output."""
    if not input_paths:
        raise ValueError("No input paths provided")

    media_infos = []
    video_caps = []
    image_frames = []

    for input_path in input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        if is_image_file(input_path):
            duration, width, height, fps = get_image_info(input_path)
            media_infos.append({'path': input_path, 'type': 'image',
                                'duration': duration, 'width': width,
                                'height': height, 'fps': fps})
            image_frame = load_image_as_frame(input_path, target_width, target_height)
            image_frames.append(image_frame)
            video_caps.append(None)
        else:
            duration, width, height, fps = get_video_info(input_path)
            media_infos.append({'path': input_path, 'type': 'video',
                                'duration': duration, 'width': width,
                                'height': height, 'fps': fps})
            cap = cv2.VideoCapture(input_path)
            video_caps.append(cap)
            image_frames.append(None)

    # Compute min duration and output fps
    video_durations = [m['duration'] for m in media_infos if m['type'] == 'video']
    min_duration = min(video_durations) if video_durations else min(m['duration'] for m in media_infos)
    video_fps = [m['fps'] for m in media_infos if m['type'] == 'video']
    output_fps = max(video_fps) if video_fps else 30.0

    total_width = len(input_paths) * target_width
    total_frames = int(min_duration * output_fps)
    print(f"Combining {len(input_paths)} media sources, {total_frames} frames at {output_fps:.2f} FPS")

    if output_format.lower() == "gif":
        frames_list = []
    else:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # safer H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (total_width, target_height))

    try:
        for frame_idx in range(total_frames):
            combined_frame = np.zeros((target_height, total_width, 3), dtype=np.uint8)
            current_x = 0

            for media_info, cap, image_frame in zip(media_infos, video_caps, image_frames):
                if media_info['type'] == 'image':
                    combined_frame[:, current_x:current_x + target_width] = image_frame
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    else:
                        frame = resize_frame(frame, target_width, target_height)
                    combined_frame[:, current_x:current_x + target_width] = frame
                current_x += target_width

            if output_format.lower() == "gif":
                rgb_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                frames_list.append(Image.fromarray(rgb_frame))
            else:
                out.write(combined_frame)

            if frame_idx % 100 == 0:
                print(f"Progress: {frame_idx / total_frames * 100:.1f}%")

    finally:
        for cap in video_caps:
            if cap is not None:
                cap.release()
        if output_format.lower() == "gif":
            imageio.mimsave(output_path, frames_list, duration=1 / output_fps)
        else:
            out.release()

    print(f"✅ Combination complete! Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Combine multiple videos and images side by side.')
    parser.add_argument('--inputs', nargs='+', required=True, help='List of input video/image paths')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--height', type=int, default=768, help='Target height')
    parser.add_argument('--width', type=int, default=576, help='Target width')
    parser.add_argument('--format', choices=['mp4', 'gif'], default='mp4', help='Output format')

    args = parser.parse_args()
    combine_media(args.inputs, args.output, args.height, args.width, args.format)


if __name__ == "__main__":
    main()
