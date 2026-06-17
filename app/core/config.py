# Application-wide configuration and environment variables loader
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Settings:
    AWS_PROFILE: str = os.getenv("AWS_PROFILE", "duy")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    AWS_S3_MODELS_BUCKET: str = os.getenv(
        "AWS_S3_MODELS_BUCKET", 
        "healthsync-ml-models-dev-819109476069-ap-southeast-1-an"
    )

settings = Settings()
