from typing import Optional
from video_generation_jobs.base_job_runner import BaseJobRunner
from video_generation_jobs.from_video_job.job_args import VideoJobArgs

def run(job_id: str, video_path: str, audio_path: str, s3_output_path: str, prompt: str, resolution: str, lora_path: Optional[str] = None):
    """
    Run video-to-video generation
    
    Args:
        job_id: Unique identifier for the job
        video_path: Path to input video
        audio_path: Path to input audio
        s3_output_path: S3 path where the output video will be uploaded
        prompt: Text prompt describing the video
        resolution: Video resolution ("480" or "720")
        lora_path: Optional path to LoRA weights
    """
    runner = BaseJobRunner()
    
    job_args = VideoJobArgs(
        job_id=job_id,
        resolution=resolution,
        s3_output_path=s3_output_path,
        prompt=prompt,
        video_path=video_path,
        audio_path=audio_path,
        lora_path=lora_path
    )
    
    runner.run(job_args)
