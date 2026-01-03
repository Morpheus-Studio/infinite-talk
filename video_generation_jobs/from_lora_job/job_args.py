from dataclasses import dataclass, field
from video_generation_jobs.base_job_runner.base_args import BaseJobArgs

@dataclass
class LoRAJobArgs(BaseJobArgs):
    """
    LoRA-specific job arguments for faster inference with optimized quality
    
    Supported LoRA models:
    - FusionX: Requires 8 steps
    - Lightx2v: Requires 4 steps
    """
    lora_dir: str = ""  # Path to LoRA weights
