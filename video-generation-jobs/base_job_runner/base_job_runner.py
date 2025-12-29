import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import asdict
from base_job_runner.job_config import JobConfig
from base_job_runner.base_args import BaseJobArgs

class BaseJobRunner:
    def __init__(self, description: str):
        self.parser = argparse.ArgumentParser(description=description)

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        return self.parser.parse_args()

    def run(self, args: BaseJobArgs, config: JobConfig, temp_config_name: str = "temp_config.json"):
        # Setup paths
        current_dir = Path(__file__).parent.absolute()
        # Project root is one level above video-generation-jobs
        repo_root = current_dir.parent.parent
        
        # Default weight paths (relative to repo root)
        ckpt_dir = repo_root / "weights" / "Wan2.1-I2V-14B-480P"
        wav2vec_dir = repo_root / "weights" / "chinese-wav2vec2-base"
        infinitetalk_dir = repo_root / "weights" / "InfiniteTalk" / "single" / "infinitetalk.safetensors"

        # Create temporary config json
        config_path = current_dir / temp_config_name
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=4)

        # Construct command
        cmd = [
            sys.executable,
            str(repo_root / "generate_infinitetalk.py"),
            "--ckpt_dir", str(ckpt_dir),
            "--wav2vec_dir", str(wav2vec_dir),
            "--infinitetalk_dir", str(infinitetalk_dir),
            "--input_json", str(config_path),
            "--size", f"infinitetalk-{args.resolution}",
            "--sample_steps", str(args.steps),
            "--mode", "streaming",
            "--motion_frame", "9",
            "--save_file", args.output
        ]

        if args.low_vram:
            # Reduce GPU memory: disable persistent params and use fp8 quant weights
            cmd.extend([
                "--num_persistent_param_in_dit", "0",
                "--quant", "fp8",
                "--quant_dir", str(repo_root / "weights" / "InfiniteTalk" / "quant_models" / "infinitetalk_single_fp8.safetensors"),
            ])

        print(f"Running command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
