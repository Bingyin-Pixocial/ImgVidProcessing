# Video Analysis Tools

This directory contains tools for analyzing MP4 video files and extracting statistics about their frame rates (FPS) and frame counts.

## Files

- `analyze_videos.py` - Command-line script for analyzing videos in a folder
- `video_analyzer.py` - Python module with reusable functions
- `example_usage.py` - Examples showing how to use all the tools

## Quick Start

### Command Line Usage

```bash
# Analyze all MP4 files in a folder
python analyze_videos.py --folder /path/to/videos

# Save results to a file
python analyze_videos.py --folder /path/to/videos --output results.txt
```

### Python Module Usage

```python
from video_analyzer import analyze_videos_in_folder

# Analyze videos in a folder
stats = analyze_videos_in_folder("/path/to/videos")
print(stats)
```

## Output Format

The analysis returns a dictionary with the following structure:

```python
{
    "num_videos": int,      # Number of videos processed
    "max_fps": float,       # Highest FPS found
    "min_fps": float,       # Lowest FPS found
    "avg_fps": float,       # Average FPS
    "max_frames": int,      # Highest frame count
    "min_frames": int,      # Lowest frame count
    "avg_frames": float     # Average frame count
}
```

## Examples

### Example 1: Command Line
```bash
python analyze_videos.py --folder ./my_videos
```

Output:
```
Found 3 MP4 files. Analyzing...
Processing 1/3: video1.mp4
  FPS: 30.00, Frames: 900
Processing 2/3: video2.mp4
  FPS: 25.00, Frames: 1500
Processing 3/3: video3.mp4
  FPS: 60.00, Frames: 1800

==================================================
VIDEO ANALYSIS RESULTS
==================================================
Number of videos processed: 3

FPS Statistics:
  Maximum FPS: 60.00
  Minimum FPS: 25.00
  Average FPS: 38.33

Frame Count Statistics:
  Maximum frames: 1800
  Minimum frames: 900
  Average frames: 1400.0
==================================================

Raw dictionary result:
{'num_videos': 3, 'max_fps': 60.0, 'min_fps': 25.0, 'avg_fps': 38.33333333333334, 'max_frames': 1800, 'min_frames': 900, 'avg_frames': 1400.0}
```

### Example 2: Python Code
```python
from video_analyzer import analyze_videos_in_folder, get_video_info

# Analyze a folder
stats = analyze_videos_in_folder("./videos")
print(f"Found {stats['num_videos']} videos")
print(f"Average FPS: {stats['avg_fps']:.2f}")

# Analyze a single video
fps, frames = get_video_info("video.mp4")
print(f"Video has {fps} FPS and {frames} frames")
```

## Requirements

The video analysis tools require the following packages (already in requirements.txt):
- opencv-python
- numpy

These tools are designed to work with the existing ImgVidProcessing toolkit and follow the same patterns and dependencies.