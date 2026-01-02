import os
from base_job_runner import BaseJobRunner
from from_video_job.job_args import VideoJobArgs

def run(video_path: str, audio_path: str, s3_output_path: str, prompt: str, resolution: str, steps: int):
    """
    Run video-to-video generation
    
    Args:
        video_path: Path to input video
        audio_path: Path to input audio
        s3_output_path: S3 path where the output video will be uploaded
        prompt: Text prompt describing the video
        resolution: Video resolution ("480" or "720")
        steps: Number of sampling steps
    """
    runner = BaseJobRunner()
    
    job_args = VideoJobArgs(
        resolution=resolution,
        s3_output_path=s3_output_path,
        prompt=prompt,
        video_path=video_path,
        audio_path=audio_path,
        steps=steps
    )
    
    runner.run(job_args)
