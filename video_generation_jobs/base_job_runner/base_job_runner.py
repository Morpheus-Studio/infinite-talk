from dataclasses import dataclass, field
import json
import subprocess
import sys
import time
from pathlib import Path
from video_generation_jobs.base_job_runner.base_args import BaseJobArgs
from video_generation_jobs.utils.s3_handler import S3Handler
from video_generation_jobs.utils.job_status_reporter import JobStatusReporter, JobStatusReport

@dataclass
class BaseJobRunner:
    
    s3_handler: S3Handler = field(init=False, default_factory=S3Handler)
    job_status_reporter: JobStatusReporter = field(init=False, default_factory=JobStatusReporter)
    
    def download_model_weights(self, repo_root: Path):
        """Download model weights"""
        download_script = repo_root / "install_scripts" / "download_weights.sh"
        
        try:
            subprocess.run(
                ["/bin/bash", str(download_script)],
                check=True,
                cwd=repo_root,
                capture_output=False
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download model weights: {e}")
        
    def apply_lora(self, args: BaseJobArgs, repo_root: Path):
        
        def resolve_lora_path(lora_path: str) -> str:
            if not lora_path:
                return str(repo_root / "weights" / "FusionX" / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors")
            
            return self.s3_handler.read_from_s3(lora_path, "/tmp/infinitetalk_lora.safetensors")
                    
        local_lora_path = resolve_lora_path(args.lora_path)
        
        return [
            "--lora_dir", local_lora_path,
            "--lora_scale", "1.0",
            "--sample_text_guide_scale", "1.0",
            "--sample_audio_guide_scale", "2.0"
        ]

    def run(self, args: BaseJobArgs):
        # Setup paths
        current_dir = Path(__file__).parent.absolute()
        # Project root is one level above video-generation-jobs
        repo_root = current_dir.parent.parent
        
        # download model weights before proceeding
        self.download_model_weights(repo_root)
        
        # Hardcoded temporary paths
        temp_config_name = "temp_config.json"
        temp_output_path = "/tmp/infinitetalk_output"

        # Download video and audio files from S3 (preserve extensions so video detection works)
        video_suffix = Path(args.video_path).suffix or ".mp4"
        audio_suffix = Path(args.audio_path).suffix or ".wav"

        local_video_path = self.s3_handler.read_from_s3(args.video_path, f"/tmp/infinitetalk_video{video_suffix}")        
        local_audio_path = self.s3_handler.read_from_s3(args.audio_path, f"/tmp/infinitetalk_audio{audio_suffix}")
        
        # Create temporary config json
        config_data = {
            "prompt": args.prompt,
            "cond_video": local_video_path,
            "cond_audio": {
                "person1": local_audio_path
            }
        }
        
        config_path = current_dir / temp_config_name
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        cmd = [
            sys.executable,
            str(repo_root / "generate_infinitetalk.py"),
            "--ckpt_dir", str(repo_root / "weights" / "Wan2.1-I2V-14B-480P"),
            "--wav2vec_dir", str(repo_root / "weights" / "chinese-wav2vec2-base"),
            "--infinitetalk_dir", str(repo_root / "weights" / "InfiniteTalk" / "single" / "infinitetalk.safetensors"),
            "--input_json", str(config_path),
            "--size", f"infinitetalk-{args.resolution}",
            "--sample_steps", "40" if args.lora_path else "8", # 40 for custom LoRA, 8 for default FusionX
            "--mode", "streaming",
            
            "--use_teacache",                      # Enable inference acceleration
            "--teacache_thresh", "0.2",            # Set TeaCache efficiency
            "--num_persistent_param_in_dit", "50",  # Keep more weights in VRAM for speed (default is usually low for consumer cards)
            "--offload_model", "False",
            "--motion_frame", "9",   # Increase overlap for smoother motion (default is 9)            
            "--frame_num", "81",                  # Larger per-chunk processing (must be 4n+1)
            
            # "--use_apg",                # Enable higher quality sampling
            # "--apg_momentum", "-0.75",   # Standard stable setting
            # "--apg_norm_threshold", "55",# Standard stable setting
            # "--color_correction_strength", "1.0", # Keep identity colors accurate
            # "--sample_shift", "7.0",          # Standard stable setting


            "--save_file", temp_output_path
        ]

        # Apply LoRA settings
        cmd.extend(self.apply_lora(args, repo_root))

        if args.quantized:
            cmd.extend([
                "--quant", "fp8",
                "--quant_dir", str(repo_root / "weights" / "InfiniteTalk" / "quant_models" / "infinitetalk_single_fp8.safetensors"),
            ])
        
        if args.scene_seg:
            cmd.append("--scene_seg")

        print(f"Running command: {' '.join(cmd)}")
        
        # Track execution time
        start_time = time.time()
        
        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
            # Upload output to S3 (the script adds .mp4 extension)
            actual_output_path = f"{temp_output_path}.mp4"
            self.s3_handler.write_to_s3(actual_output_path, args.s3_output_path)
            
            # Report success
            execution_time = time.time() - start_time
            model_type = "Custom LoRA" if args.lora_path else "FusionX"
            
            report = JobStatusReport(
                job_id=args.job_id,
                status="COMPLETED",
                execution_time=execution_time,
                resolution=args.resolution,
                quantized=args.quantized,
                scene_seg=args.scene_seg,
                model_type=model_type
            )
            self.job_status_reporter.report_status(report)
            
        except Exception as e:
            # Report failure
            execution_time = time.time() - start_time
            model_type = "Custom LoRA" if args.lora_path else "FusionX"

            report = JobStatusReport(
                job_id=args.job_id,
                status="FAILED",
                execution_time=execution_time,
                error_message=str(e),
                resolution=args.resolution,
                quantized=args.quantized,
                scene_seg=args.scene_seg,
                model_type=model_type
            )
            self.job_status_reporter.report_status(report)
            raise e
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()
            actual_output_path = f"{temp_output_path}.mp4"
            if Path(actual_output_path).exists():
                Path(actual_output_path).unlink()
