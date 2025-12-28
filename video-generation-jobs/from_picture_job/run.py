import os
from base_job_runner import BaseJobRunner
from base_job_runner.job_config import JobConfig
from from_picture_job.job_args import PictureJobArgs

def main():
    runner = BaseJobRunner(description="Create video from picture using InfiniteTalk")
    runner.add_argument("--image", required=True, help="Path to input image")
    runner.add_argument("--audio", required=True, help="Path to input audio")
    runner.add_argument("--output", required=True, help="Output filename prefix (e.g. result)")
    runner.add_argument("--prompt", default="A person talking", help="Text prompt describing the video")
    runner.add_argument("--resolution", choices=["480", "720"], default="480", help="Video resolution")
    
    args = runner.parse_args()

    job_args = PictureJobArgs(
        resolution=args.resolution,
        output=args.output,
        image=os.path.abspath(args.image),
        audio=os.path.abspath(args.audio),
        prompt=args.prompt
    )

    config = JobConfig(
        prompt=job_args.prompt,
        cond_video=job_args.image,
        cond_audio={
            "person1": job_args.audio
        }
    )
    
    runner.run(job_args, config, "temp_config_image.json")

if __name__ == "__main__":
    main()
