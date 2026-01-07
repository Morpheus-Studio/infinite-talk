"""
RunPod Serverless Handler for InfiniteTalk Video Generation
"""
import runpod
import sys
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

# Add project root to path (parent of run_pod folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from video_generation_jobs.from_lora_job.run import run as run_lora_job
from video_generation_jobs.from_picture_job.run import run as run_picture_job
from video_generation_jobs.from_video_job.run import run as run_video_job


class JobInput(BaseModel):
    """Pydantic model for job input validation"""
    job_id: str = Field(..., description="Unique identifier for the job")
    job_type: Literal["from_picture", "from_video", "from_lora"] = Field(..., description="Type of video generation job")
    input_media: str = Field(..., description="Path to input image or video")
    audio: str = Field(..., description="Path to input audio file")
    prompt: str = Field(default="A person talking", description="Text prompt describing the video")
    resolution: Literal["480", "720"] = Field(default="480", description="Video resolution")
    steps: int = Field(default=40, ge=1, le=100, description="Number of sampling steps")
    s3_output_path: str = Field(default="", description="S3 path for output video")
    lora_path: Optional[str] = Field(default=None, description="Path to LoRA weights (required for from_lora)")


def handler(job_args: dict):
    try:
        job_input = JobInput(**job_args.get("input"))
    except ValidationError as e:
        return {"error": "Validation failed", "details": e.errors()}
    
    if job_input.job_type == "from_lora" and not job_input.lora_path:
        return {"error": "lora_path is required for job_type 'from_lora'"}
    
    try:
        if job_input.job_type == "from_picture":
            run_picture_job(
                job_id=job_input.job_id,
                image_path=job_input.input_media,
                audio_path=job_input.audio,
                s3_output_path=job_input.s3_output_path,
                prompt=job_input.prompt,
                resolution=job_input.resolution,
                steps=job_input.steps
            )
        if job_input.job_type == "from_video":
            run_video_job(
                job_id=job_input.job_id,
                video_path=job_input.input_media,
                audio_path=job_input.audio,
                s3_output_path=job_input.s3_output_path,
                prompt=job_input.prompt,
                resolution=job_input.resolution,
                steps=job_input.steps
            )
        if job_input.job_type == "from_lora":
            run_lora_job(
                job_id=job_input.job_id,
                image_path=job_input.input_media,
                audio_path=job_input.audio,
                s3_output_path=job_input.s3_output_path,
                lora_path=job_input.lora_path,
                prompt=job_input.prompt,
                resolution=job_input.resolution,
                steps=job_input.steps
            )
        
        return {"status": "success", "job_type": job_input.job_type, "s3_output_path": job_input.s3_output_path}
        
    except Exception as e:
        return {"error": str(e), "job_type": job_input.job_type}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
