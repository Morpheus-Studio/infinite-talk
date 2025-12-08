#!/usr/bin/env python3
"""
Avatar Video Generation Job
Generates video using a trained custom avatar LoRA.
"""

import sys
from pathlib import Path
from base_video_runner import run_video_generation, create_input_json


def main(avatar_lora: str, audio_path: str, output_video: str,
         avatar_image: str = 'assets/face.jpeg',
         prompt: str = "A person speaks naturally with clear lip sync.",
         size: str = 'infinitetalk-480',
         sample_steps: int = 8,
         sample_text_guide_scale: float = 1.0,
         sample_audio_guide_scale: float = 2.0,
         lora_scale: float = 1.0):
    """
    Generate video using custom avatar LoRA.
    
    Args:
        avatar_lora: Path to trained avatar LoRA safetensors
        audio_path: Path to input audio
        output_video: Path for output MP4
        avatar_image: Reference image for avatar (default: assets/face.jpeg)
        prompt: Text prompt (optional)
        sample_steps: Number of diffusion steps (default: 8 with FusionX)
        sample_text_guide_scale: Text guidance strength
        sample_audio_guide_scale: Audio guidance strength
        lora_scale: LoRA strength for avatar
    
    Returns:
        True if successful
    """
    # Validate inputs
    if not Path(avatar_lora).exists():
        print(f"✗ Avatar LoRA not found: {avatar_lora}")
        print("  Train an avatar first using: python train_avatar.py")
        return False
    
    if not Path(audio_path).exists():
        print(f"✗ Audio not found: {audio_path}")
        return False
    
    if not Path(avatar_image).exists():
        print(f"✗ Avatar image not found: {avatar_image}")
        return False
    
    # Create input JSON
    input_json = create_input_json(avatar_image, audio_path, prompt)
    
    # Run inference with custom LoRA
    success = run_video_generation(
        input_json,
        output_video,
        size=size,
        sample_steps=sample_steps,
        sample_text_guide_scale=sample_text_guide_scale,
        sample_audio_guide_scale=sample_audio_guide_scale,
        lora_scale=lora_scale,
        lora_dir=avatar_lora,  # Override with custom avatar LoRA
    )
    
    # Cleanup
    Path(input_json).unlink(missing_ok=True)
    
    return success


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python avatar_video.py <lora_path> <audio_path> [output_video]")
        print()
        print("Example:")
        print("  python avatar_video.py my_avatar.safetensors audio.m4a output.mp4")
        sys.exit(1)
    
    avatar_lora = sys.argv[1]
    audio_path = sys.argv[2]
    output_video = sys.argv[3] if len(sys.argv) > 3 else 'output.mp4'
    
    success = main(avatar_lora, audio_path, output_video)
    sys.exit(0 if success else 1)
