from video_generation_jobs.base_job_runner import BaseJobRunner
from video_generation_jobs.from_lora_job.job_args import LoRAJobArgs


def run(
    image_path: str,
    audio_path: str,
    s3_output_path: str,
    lora_path: str,
    prompt: str,
    resolution: str,
    steps: int = 8,
):
    """
    Run LoRA-optimized video generation with FusionX or Lightx2v
    
    Args:
        image_path: Path to input image
        audio_path: Path to input audio
        s3_output_path: S3 path where the output video will be uploaded
        lora_path: Path to LoRA weights file
        prompt: Text prompt describing the video
        resolution: Video resolution ("480" or "720")
        steps: Number of sampling steps (8 for FusionX, 4 for Lightx2v)
    """
    runner = BaseJobRunner()
    
    job_args = LoRAJobArgs(
        resolution=resolution,
        s3_output_path=s3_output_path,
        prompt=prompt,
        video_path=image_path,
        audio_path=audio_path,
        steps=steps,
        lora_dir=lora_path,
    )
    
    runner.run(job_args)
