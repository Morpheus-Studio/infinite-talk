from dataclasses import dataclass
from base_job_runner.base_args import BaseJobArgs

@dataclass
class PictureJobArgs(BaseJobArgs):
    image: str
    audio: str
    prompt: str
