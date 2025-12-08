#!/usr/bin/env python3
"""
Video to Video Generation Job
Converts a video + new audio into a dubbed video with lip sync.
Uses first frame of input video as reference.
"""

import sys
import subprocess
from pathlib import Path
from base_video_runner import run_video_generation, create_input_json


def extract_first_frame(video_path: str, output_frame: str) -> bool:
    """Extract first frame from video using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            output_frame,
            '-y'
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error extracting frame: {e}")
        return False
    except FileNotFoundError:
        print("✗ FFmpeg not found. Please install FFmpeg.")
        return False


def main(video_path: str, audio_path: str, output_video: str,
         prompt: str = "A person speaks naturally with clear lip sync.",
         size: str = 'infinitetalk-480',
         sample_steps: int = 8,
         sample_text_guide_scale: float = 1.0,
         sample_audio_guide_scale: float = 2.0,
         lora_scale: float = 1.0,
         lora_dir: str = 'weights/FusionX/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors'):
    """
    Generate video with new audio (dubbing/lip-sync).
    
    Args:
        video_path: Path to input video
        audio_path: Path to new audio
        output_video: Path for output MP4
        prompt: Text prompt (optional)
        sample_steps: Number of diffusion steps (default: 8 with FusionX)
        sample_text_guide_scale: Text guidance strength
        sample_audio_guide_scale: Audio guidance strength
        lora_scale: LoRA strength
        lora_dir: Path to LoRA weights
    
    Returns:
        True if successful
    """
    # Extract first frame
    print("Extracting first frame from video...")
    frame_path = Path('temp_frame.jpg')
    if not extract_first_frame(video_path, str(frame_path)):
        return False
    print(f"✓ Frame extracted: {frame_path}")
    print()
    
    # Create input JSON
    input_json = create_input_json(str(frame_path), audio_path, prompt)
    
    # Run inference
    success = run_video_generation(
        input_json,
        output_video,
        size=size,
        sample_steps=sample_steps,
        sample_text_guide_scale=sample_text_guide_scale,
        sample_audio_guide_scale=sample_audio_guide_scale,
        lora_scale=lora_scale,
        lora_dir=lora_dir,
    )
    
    # Cleanup
    Path(input_json).unlink(missing_ok=True)
    frame_path.unlink(missing_ok=True)
    
    return success


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python video_to_video.py <video_path> <audio_path> [output_video]")
        print()
        print("Example:")
        print("  python video_to_video.py input.mp4 audio.m4a output.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_video = sys.argv[3] if len(sys.argv) > 3 else 'output.mp4'
    
    success = main(video_path, audio_path, output_video)
    sys.exit(0 if success else 1)
