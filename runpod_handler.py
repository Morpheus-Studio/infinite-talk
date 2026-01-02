"""
RunPod Serverless Handler for InfiniteTalk Video Generation
"""
import runpod
import sys
from pathlib import Path
from video_generation_jobs.from_picture_job.run import run as run_picture_job
from video_generation_jobs.from_video_job.run import run as run_video_job
# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def handler(job):
    """
    RunPod handler for video generation
    
    Input format:
    {
        "input": {
            "job_type": "from_picture" or "from_video",
            "input_media": "/path/to/image.jpg" or "/path/to/video.mp4",
            "audio": "/path/to/audio.wav",
            "prompt": "A person talking",
            "resolution": "480" or "720",
            "steps": 40
        }
    }
    """
    job_input = job.get("input", {})
    job_type = job_input.get("job_type")
    
    # Validate job type
    if job_type not in ["from_picture", "from_video"]:
        return {"error": f"Invalid job_type: {job_type}. Must be 'from_picture' or 'from_video'"}
    
    # Validate required inputs
    if not job_input.get("input_media"):
        return {"error": "input_media is required"}
    
    if not job_input.get("audio"):
        return {"error": "audio is required"}
    
    # Extract parameters
    output_file = f"/tmp/output_{job.get('id', 'unknown')}"
    prompt = job_input.get("prompt", "A person talking")
    resolution = job_input.get("resolution", "480")
    steps = job_input.get("steps", 40)
    
    try:
        if job_type == "from_picture":
            run_picture_job(
                image_path=job_input["input_media"],
                audio_path=job_input["audio"],
                s3_output_path=job_input.get("s3_output_path", ""),
                prompt=prompt,
                resolution=resolution,
                steps=steps
            )
        if job_type == "from_video":
            run_video_job(
                video_path=job_input["input_media"],
                audio_path=job_input["audio"],
                s3_output_path=job_input.get("s3_output_path", ""),
                prompt=prompt,
                resolution=resolution,
                steps=steps
            )
        
        return {"status": "success", "job_type": job_type, "s3_output_path": job_input.get("s3_output_path", "")}
        
    except Exception as e:
        return {"error": str(e), "job_type": job_type}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
