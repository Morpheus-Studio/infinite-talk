import boto3
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse
from botocore.exceptions import ClientError

@dataclass
class S3Handler:
    """Handle S3 operations for video generation jobs"""
    # Initialize the S3 client once for the instance
    s3_client: any = field(default_factory=lambda: boto3.client('s3'))
        
    def _parse_s3_path(self, s3_path: str):
        """Helper to extract bucket and key from an s3:// URL"""
        parsed = urlparse(s3_path)
        return parsed.netloc, parsed.path.lstrip('/')

    def read_from_s3(self, s3_path: str, local_path: str) -> str:
        """
        Downloads a file from S3 to the specified local path
        
        Args:
            s3_path: S3 URL (e.g., s3://bucket/key)
            local_path: Local file path to save to
            
        Returns:
            The local_path if successful, empty string on failure
        """
        try:
            bucket, key = self._parse_s3_path(s3_path)
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
            
            self.s3_client.download_file(bucket, key, local_path)
            print(f"Successfully downloaded {s3_path} to {local_path}")
            return local_path
        except ClientError as e:
            raise RuntimeError(f"Failed to download audio from {e}")
    
    def write_to_s3(self, local_file_path: str, s3_path: str) -> bool:
        """Uploads a local file to the specified S3 path"""
        try:
            bucket, key = self._parse_s3_path(s3_path)
            
            self.s3_client.upload_file(local_file_path, bucket, key)
            print(f"Successfully uploaded {local_file_path} to {s3_path}")
            return True
        except ClientError as e:
            raise RuntimeError(f"Failed to upload file to S3: {e}")