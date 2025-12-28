# Video Generation Jobs

This directory contains a modular framework for running InfiniteTalk video generation jobs. Each job is organized in its own folder with dedicated argument classes, making the codebase clean and scalable.

## Project Structure

```
video-generation-jobs/
├── base_job_runner/           # Shared runner framework
│   ├── __init__.py
│   ├── base_job_runner.py     # BaseJobRunner class (handles subprocess execution)
│   ├── base_args.py           # BaseJobArgs dataclass (common arguments)
│   └── job_config.py          # JobConfig dataclass (video generation config)
│
├── from_picture_job/          # Image-to-video generation job
│   ├── __init__.py
│   ├── job_args.py            # PictureJobArgs dataclass (image-specific args)
│   └── run.py                 # Entry point for image-to-video
│
├── from_video_job/            # Video-to-video generation job
│   ├── __init__.py
│   ├── job_args.py            # VideoJobArgs dataclass (video-specific args)
│   └── run.py                 # Entry point for video-to-video
│
└── README.md                  # This file
```

## Core Components

### BaseJobRunner
The `BaseJobRunner` class in `base_job_runner/base_job_runner.py` handles:
- Argument parsing via argparse
- Temporary JSON config file creation
- Subprocess execution of `generate_infinitetalk.py`
- Automatic cleanup of temporary files

### Dataclasses

#### JobConfig
Represents the video generation configuration that gets serialized to JSON:
- `prompt`: Text description of the video
- `cond_video`: Path to input image or video
- `cond_audio`: Dictionary mapping person names to audio file paths

#### BaseJobArgs
Common arguments shared across all jobs:
- `resolution`: Video resolution (480 or 720)
- `output`: Output filename prefix
- `steps`: Sampling steps (default: 40)
- `low_vram`: Enable low VRAM mode (default: False)

#### Job-Specific Args

**PictureJobArgs** (from_picture_job/job_args.py)
- Extends `BaseJobArgs`
- Adds: `image`, `audio`, `prompt`

**VideoJobArgs** (from_video_job/job_args.py)
- Extends `BaseJobArgs`
- Adds: `video`, `audio`, `prompt`

## Usage

### Create Video from Picture

```bash
python video-generation-jobs/from_picture_job/run.py \
  --image path/to/image.png \
  --audio path/to/audio.wav \
  --output my_video_result \
  --prompt "A person singing" \
  --resolution 480
```

### Create Video from Video

```bash
python video-generation-jobs/from_video_job/run.py \
  --video path/to/video.mp4 \
  --audio path/to/audio.wav \
  --output my_video_result \
  --prompt "A person dancing" \
  --resolution 720
```

### Optional Arguments

Both scripts support:
- `--resolution`: 480 or 720 (default: 480)
- `--prompt`: Text description (default: "A person talking")
- `--output`: Output filename (required)
- `--low-vram`: Enable low VRAM mode (only add flag if needed)

## How It Works

1. Each job script (`run.py`) parses command-line arguments
2. Arguments are packaged into job-specific dataclass instances
3. A `JobConfig` object is created from the arguments
4. `BaseJobRunner.run()` is called with both objects
5. A temporary JSON config file is created
6. `generate_infinitetalk.py` is executed as a subprocess with the config
7. The temporary config file is automatically cleaned up

## Adding a New Job

To add a new video generation job:

1. Create a new folder: `video-generation-jobs/new_job/`
2. Create `__init__.py` (empty file)
3. Create `job_args.py` with your specific args dataclass:
   ```python
   from dataclasses import dataclass
   from base_job_runner.base_args import BaseJobArgs
   
   @dataclass
   class NewJobArgs(BaseJobArgs):
       your_param1: str
       your_param2: str
   ```
4. Create `run.py` following the pattern from `from_picture_job/run.py` or `from_video_job/run.py`
5. Import and use the new args class in your `run.py`

## Requirements

- InfiniteTalk repository (parent directory)
- Model weights downloaded (see main README)
- Python packages from requirements.txt

## Notes

- Temporary config JSON files are created in the same directory as the job script
- The runner automatically changes to the repository root before executing `generate_infinitetalk.py`
- Model paths are hardcoded relative to the repository root
- Low VRAM mode requires `--num_persistent_param_in_dit` to be set to 0
