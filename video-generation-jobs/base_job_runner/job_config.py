from dataclasses import dataclass
from typing import Dict

@dataclass
class JobConfig:
    prompt: str
    cond_video: str
    cond_audio: Dict[str, str]
