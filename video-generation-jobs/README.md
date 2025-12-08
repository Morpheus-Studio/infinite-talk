# Video Generation Jobs

Modular job system for InfiniteTalk video generation. All jobs use simple functions with hardcoded optimized defaults and dynamic parameters.

## Quick Start

### Image to Video
```bash
python image_to_video.py face.jpg audio.m4a output.mp4
```

### Video to Video (Dubbing)
```bash
python video_to_video.py input.mp4 new_audio.m4a dubbed.mp4
```

### Avatar Video
```bash
python avatar_video.py my_avatar.safetensors audio.m4a avatar.mp4
```

## Architecture

All jobs use `base_video_runner.py` with simple functions:

- `run_video_generation()` - Inference with hardcoded optimized defaults
- `create_input_json()` - Helper to create input configuration

### Fixed Arguments (Hardcoded Defaults)
- Model paths, quantization (FP8), VRAM optimization
- Size: 480P, Mode: streaming
- Default LoRA: FusionX (5x speed acceleration)

### Dynamic Parameters (Per-Job Tuning)
- `sample_steps` - Diffusion steps (default: 8)
- `sample_text_guide_scale` - Text guidance (default: 1.0)
- `sample_audio_guide_scale` - Audio guidance (default: 2.0)
- `lora_scale` - LoRA strength (default: 1.0)
- `lora_dir` - Custom LoRA path (override for avatars)

## Jobs

### 1. Image to Video
Generate talking head video from image + audio.

```python
from image_to_video import main
success = main('face.jpg', 'audio.m4a', 'output.mp4')
```

### 2. Video to Video
Dub existing video with new audio using first frame as reference.

```python
from video_to_video import main
success = main('input.mp4', 'new_audio.m4a', 'dubbed.mp4')
```

### 3. Avatar Video
Generate video using custom trained avatar LoRA.

```python
from avatar_video import main
success = main('avatar.safetensors', 'audio.m4a', 'output.mp4')
```

### 4. Train Avatar
Placeholder for LoRA training. Implement as needed.

## Performance

- FusionX LoRA: 40 → 8 steps (5x speedup)
- FP8 quantization: Reduces VRAM without quality loss
- Audio caching: Faster re-runs via `save_audio/`
- Typical inference: 30-60 seconds on A100
