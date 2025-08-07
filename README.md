# User Guide

## Extract Frames from Videos
```
python frames_video_conversion.py --mode extract --video_path /path/to/videos --output /path/to/extracted_frames  --prefix frame
```

## Create Videos from Frames
```
python frames_video_conversion.py --mode create --frames_dir /path/to/frames --output_video /path/to/output_video --output_filename output_video_name --fps 30 
```

## Combine Videos
```
python combine_videos.py --inputs input1 input2 input3 --output /path/to/combined_video --format mp4 (or gif)
```

## Video Analysis
```
# Analyze all MP4 files in a folder
python analyze_videos.py --folder /path/to/videos

# Save results to a file
python analyze_videos.py --folder /path/to/videos --output results.txt
```
