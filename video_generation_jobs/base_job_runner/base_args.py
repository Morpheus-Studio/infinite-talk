from dataclasses import dataclass
from typing import Literal

@dataclass
class BaseJobArgs:
    resolution: Literal["480", "720"]
    s3_output_path: str
    prompt: str
    video_path: str
    audio_path: str
    steps: int = 40
    low_vram: bool = False
