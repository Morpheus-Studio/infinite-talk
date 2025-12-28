from dataclasses import dataclass

@dataclass
class BaseJobArgs:
    resolution: str
    output: str
    steps: int = 40
    low_vram: bool = False
