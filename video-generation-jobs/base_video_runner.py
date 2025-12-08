#!/usr/bin/env python3
"""
Base Video Runner - Simple functions for video generation with hardcoded defaults.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path to import generate_infinitetalk
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_infinitetalk import _parse_args, generate


def run_video_generation(
    input_json: str,
    output_path: str,
    size: str = 'infinitetalk-480',
    sample_steps: int = 8,
    sample_text_guide_scale: float = 1.0,
    sample_audio_guide_scale: float = 2.0,
    lora_scale: float = 1.0,
    lora_dir: str = 'weights/FusionX/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors',
) -> bool:
    """
    Run InfiniteTalk video generation with fixed and dynamic arguments.
    
    Fixed arguments (hardcoded):
    - Model paths (ckpt_dir, wav2vec_dir, infinitetalk_dir)
    - Quantization settings (fp8)
    - VRAM optimization (num_persistent_param_in_dit=0)
    - Mode (streaming), motion_frame, etc.
    
    Dynamic arguments (function parameters):
    - size: Video resolution (default: infinitetalk-480)
    - sample_steps: Number of diffusion steps
    - sample_text_guide_scale: Text guidance strength
    - sample_audio_guide_scale: Audio guidance strength
    - lora_scale: LoRA adapter strength
    - lora_dir: Path to LoRA safetensors (can override for custom avatars)
    
    Args:
        input_json: Path to input config JSON
        output_path: Path for output MP4
        size: Video size (default: infinitetalk-480, also supports infinitetalk-720)
        sample_steps: Diffusion steps (default: 8 with FusionX)
        sample_text_guide_scale: Text CFG scale (default: 1.0 with LoRA)
        sample_audio_guide_scale: Audio CFG scale (default: 2.0 with LoRA)
        lora_scale: LoRA strength (default: 1.0)
        lora_dir: Path to LoRA weights (default: FusionX)
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Build args object matching generate_infinitetalk.py expectations
        args = argparse.Namespace(
            task='infinitetalk-14B',
            size=size,
            frame_num=81,
            max_frame_num=1000,
            ckpt_dir='weights/Wan2.1-I2V-14B-480P',
            infinitetalk_dir='weights/InfiniteTalk/single/infinitetalk.safetensors',
            quant_dir='weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors',
            wav2vec_dir='weights/chinese-wav2vec2-base',
            dit_path=None,
            lora_dir=[lora_dir],
            lora_scale=[lora_scale],
            offload_model=True,
            ulysses_size=1,
            ring_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            save_file=output_path.replace('.mp4', ''),
            audio_save_dir='save_audio',
            base_seed=42,
            input_json=input_json,
            motion_frame=9,
            mode='streaming',
            sample_steps=sample_steps,
            sample_shift=None,  # Will be set in _validate_args
            sample_text_guide_scale=sample_text_guide_scale,
            sample_audio_guide_scale=sample_audio_guide_scale,
            num_persistent_param_in_dit=0,
            audio_mode='localfile',
            use_teacache=False,
            teacache_thresh=0.2,
            use_apg=False,
            apg_momentum=-0.75,
            apg_norm_threshold=55,
            color_correction_strength=1.0,
            scene_seg=False,
            quant='fp8',
        )
        
        print(f"Generating video from {input_json}")
        print(f"  Size: {size}")
        print(f"  Steps: {sample_steps}")
        print(f"  Text scale: {sample_text_guide_scale}")
        print(f"  Audio scale: {sample_audio_guide_scale}")
        print(f"  LoRA scale: {lora_scale}")
        print()
        
        # Call generate directly
        generate(args)
        
        print(f"\n✓ Video saved to: {output_path}.mp4")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_input_json(
    image_path: str,
    audio_path: str,
    prompt: str = "A person speaks naturally to the camera with clear lip sync."
) -> str:
    """
    Create input JSON config for inference.
    
    Args:
        image_path: Path to input image
        audio_path: Path to input audio
        prompt: Text prompt for generation
    
    Returns:
        Path to created JSON file
    """
    config = {
        "prompt": prompt,
        "cond_video": image_path,
        "cond_audio": {
            "person1": audio_path
        }
    }
    
    json_path = Path('tmp_input.json')
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(json_path)
        for key, value in self.fixed_args.items():
            print(f"  {key:30s}: {value}")
        
        print()
        print("=" * 60)
        print("Dynamic Arguments (per-job override):")
        print("=" * 60)
        for key, value in self.dynamic_args.items():
            print(f"  {key:30s}: {value}")


if __name__ == "__main__":
    # Example usage
    runner = BaseVideoRunner()
    runner.print_config()
