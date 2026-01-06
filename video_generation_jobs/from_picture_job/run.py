from video_generation_jobs.base_job_runner import BaseJobRunner
from video_generation_jobs.from_picture_job.job_args import PictureJobArgs



def run(job_id: str, image_path: str, audio_path: str, s3_output_path: str, prompt: str, resolution: str, steps: int):
    """
    Run picture-to-video generation
    
    Args:
        job_id: Unique identifier for the job
        image_path: Path to input image
        audio_path: Path to input audio
        s3_output_path: S3 path where the output video will be uploaded
        prompt: Text prompt describing the video
        resolution: Video resolution ("480" or "720")
        steps: Number of sampling steps
    """
    runner = BaseJobRunner()
    
    job_args = PictureJobArgs(
        job_id=job_id,
        resolution=resolution,
        s3_output_path=s3_output_path,
        prompt=prompt,
        video_path=image_path,
        audio_path=audio_path,
        steps=steps,
    )
    
    runner.run(job_args)