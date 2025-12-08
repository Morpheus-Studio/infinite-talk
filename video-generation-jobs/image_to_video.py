#!/usr/bin/env python3
"""
Image to Video Generation Job
Converts a single image + audio into a talking head video.
"""

import sys
from pathlib import Path
from base_video_runner import run_video_generation, create_input_json


def main(image_path: str, audio_path: str, output_video: str, 
         prompt: str = "A person speaks naturally to the camera with clear lip sync.",
         size: str = 'infinitetalk-480',
         sample_steps: int = 8,
         sample_text_guide_scale: float = 1.0,
         sample_audio_guide_scale: float = 2.0,
         lora_scale: float = 1.0,
         lora_dir: str = 'weights/FusionX/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors'):
    """
    Generate video from image and audio.
    
    Args:
        image_path: Path to input image
        audio_path: Path to input audio
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
    input_json = create_input_json(image_path, audio_path, prompt)
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
    
    return success


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python image_to_video.py <image_path> <audio_path> [output_video]")
        print()
        print("Example:")
        print("  python image_to_video.py face.jpg audio.m4a output.mp4")
        sys.exit(1)
    
    image_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_video = sys.argv[3] if len(sys.argv) > 3 else 'output.mp4'
    
    success = main(image_path, audio_path, output_video)
    sys.exit(0 if success else 1)
