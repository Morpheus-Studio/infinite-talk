#!/usr/bin/env python3
"""
InfiniteTalk Inference Test
Runs a quick inference test to verify the model works correctly.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run InfiniteTalk inference test"""
    print("\n" + "=" * 60)
    print("InfiniteTalk Inference Test")
    print("=" * 60)
    print()
    
    # Check if weights exist
    weights_dir = Path("weights")
    required_paths = [
        weights_dir / "Wan2.1-I2V-14B-480P",
        weights_dir / "chinese-wav2vec2-base",
        weights_dir / "InfiniteTalk/single/infinitetalk.safetensors",
    ]
    
    print("Checking model weights...")
    for path in required_paths:
        if not path.exists():
            print(f"✗ Missing: {path}")
            print("\nPlease run: bash install_scripts/download_weights.sh")
            return 1
        print(f"✓ Found: {path}")
    
    print()
    print("Starting inference with example image...")
    print("This will generate a talking head video from an image and audio.")
    print("=" * 60)
    print()
    
    # Run the inference command with FusionX LoRA + quantization for fast, low-VRAM inference
    cmd = [
        "python", "generate_infinitetalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
        "--lora_dir", "weights/FusionX/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
        "--input_json", "examples/single_example_image.json",
        "--lora_scale", "1.0",
        "--size", "infinitetalk-480",
        "--sample_text_guide_scale", "1.0",
        "--sample_audio_guide_scale", "2.0",
        "--sample_steps", "8",
        "--mode", "streaming",
        "--quant", "fp8",
        "--quant_dir", "weights/InfiniteTalk/quant_models/t5_fp8.safetensors",
        "--motion_frame", "9",
        "--sample_shift", "2",
        "--num_persistent_param_in_dit", "0",
        "--save_file", "test_output"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("✓ Inference completed successfully!")
        print("=" * 60)
        print()
        print("Output video saved as: test_output.mp4")
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("✗ Inference failed!")
        print("=" * 60)
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("Inference interrupted by user")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
