from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class BaseJobArgs:
    job_id: str
    resolution: Literal["480", "720"]
    s3_output_path: str
    prompt: str
    video_path: str
    audio_path: str
    low_vram: bool = False
    lora_path: Optional[str] = None
