from dataclasses import dataclass, field
import json
import subprocess
import sys
from pathlib import Path
from base_job_runner.base_args import BaseJobArgs
from utils.s3_handler import S3Handler

@dataclass
class BaseJobRunner:
    
    s3_handler: S3Handler = field(init=False, default_factory=S3Handler)
        
    def run(self, args: BaseJobArgs):
        # Setup paths
        current_dir = Path(__file__).parent.absolute()
        # Project root is one level above video-generation-jobs
        repo_root = current_dir.parent.parent
        
        # Hardcoded temporary paths
        temp_config_name = "temp_config.json"
        temp_output_path = "/tmp/infinitetalk_output"

        # read audio and image/video from s3
        
        # Create temporary config json
        config_data = {
            "prompt": args.prompt,
            "cond_video": args.video_path,
            "cond_audio": {
                "person1": args.audio_path
            }
        }
        
        config_path = current_dir / temp_config_name
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        # Construct command
        cmd = [
            sys.executable,
            str(repo_root / "generate_infinitetalk.py"),
            "--ckpt_dir", str(repo_root / "weights" / "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir", str(repo_root / "weights" / "chinese-wav2vec2-base"),
            "--infinitetalk_dir", str(repo_root / "weights" / "InfiniteTalk" / "single" / "infinitetalk.safetensors"),
            "--input_json", str(config_path),
            "--size", f"infinitetalk-{args.resolution}",
            "--sample_steps", str(args.steps),
            "--mode", "streaming",
            "--motion_frame", "9",
            "--save_file", temp_output_path
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
            # Upload output to S3
            self.s3_handler.write_to_s3(temp_output_path, args.s3_output_path)
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
            if Path(temp_output_path).exists():
                Path(temp_output_path).unlink()
