import os
import requests
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class JobStatusReport:
    """Data class for job status reporting"""
    job_id: str
    status: Literal["COMPLETED", "FAILED"]
    execution_time: float
    resolution: str
    quantized: bool
    scene_seg: bool
    model_type: str
    error_message: Optional[str] = None


@dataclass
class JobStatusReporter:
    """Handle job status reporting to the job manager API"""
    api_key: str = field(default_factory=lambda: os.getenv("AWS_JOB_STATUS_API_KEY", ""))
    api_url: str = field(default_factory=lambda: os.getenv("AWS_JOB_STATUS_URL", ""))
    
    def report_status(self, report: JobStatusReport) -> bool:
        """
        Report job completion status to the job manager API.
        
        Args:
            report: JobStatusReport containing job status details
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                "job_id": report.job_id,
                "status": report.status,
                "execution_time": report.execution_time
            }
            
            if report.error_message:
                payload["error_message"] = report.error_message
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            print(f"Successfully reported job status: {report.status} for job_id: {report.job_id}")
            return True
        except Exception as e:
            print(f"Failed to report job status: {e}")
            return False
