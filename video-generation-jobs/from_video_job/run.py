import os
from base_job_runner import BaseJobRunner
from base_job_runner.job_config import JobConfig
from from_video_job.job_args import VideoJobArgs

def main():
    runner = BaseJobRunner(description="Create video from video using InfiniteTalk")
    runner.add_argument("--video", required=True, help="Path to input video")
    runner.add_argument("--audio", required=True, help="Path to input audio")
    runner.add_argument("--output", required=True, help="Output filename prefix (e.g. result)")
    runner.add_argument("--prompt", default="A person talking", help="Text prompt describing the video")
    runner.add_argument("--resolution", choices=["480", "720"], default="480", help="Video resolution")
    
    args = runner.parse_args()

    job_args = VideoJobArgs(
        resolution=args.resolution,
        output=args.output,
        video=os.path.abspath(args.video),
        audio=os.path.abspath(args.audio),
        prompt=args.prompt
    )

    config = JobConfig(
        prompt=job_args.prompt,
        cond_video=job_args.video,
        cond_audio={
            "person1": job_args.audio
        }
    )
    
    runner.run(job_args, config, "temp_config_video.json")

if __name__ == "__main__":
    main()
