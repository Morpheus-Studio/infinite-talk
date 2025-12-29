from dataclasses import dataclass
from base_job_runner.base_args import BaseJobArgs

@dataclass
class VideoJobArgs(BaseJobArgs):
    video: str = ""
    audio: str = ""
    prompt: str = ""
