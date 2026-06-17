import os
import io
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings

def get_s3_client():
    """
    Returns a boto3 S3 client configured with settings from core.config.
    Falls back to a default session if the custom profile fails or isn't set.
    """
    profile = settings.AWS_PROFILE
    region = settings.AWS_REGION
    
    try:
        if profile:
            # Try loading session with custom profile name
            session = boto3.Session(profile_name=profile, region_name=region)
            return session.client("s3")
    except Exception as e:
        print(f"Warning: Failed to initialize AWS session with profile '{profile}': {e}. Falling back to default.")
    
    # Fallback to default session
    return boto3.client("s3", region_name=region)

def s3_object_exists(bucket: str, key: str) -> bool:
    """
    Checks if an object exists in a given S3 bucket using head_object.
    """
    s3_client = get_s3_client()
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        # 404 indicates the file does not exist
        if e.response['Error']['Code'] == '404':
            return False
        # Other client errors should be logged/raised
        print(f"Error checking S3 key '{key}' in bucket '{bucket}': {e}")
        return False
    except Exception as e:
        print(f"Error checking S3 key '{key}' in bucket '{bucket}': {e}")
        return False

def upload_to_s3(buffer: io.BytesIO, bucket: str, key: str) -> bool:
    """
    Uploads an in-memory BytesIO buffer directly to S3.
    """
    s3_client = get_s3_client()
    try:
        buffer.seek(0)
        s3_client.upload_fileobj(buffer, bucket, key)
        print(f"Successfully uploaded s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        return False

def download_from_s3(bucket: str, key: str) -> io.BytesIO:
    """
    Downloads an S3 object into an in-memory BytesIO buffer.
    """
    s3_client = get_s3_client()
    buffer = io.BytesIO()
    try:
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        print(f"Successfully downloaded s3://{bucket}/{key}")
        return buffer
    except Exception as e:
        print(f"Failed to download from S3: {e}")
        raise e
