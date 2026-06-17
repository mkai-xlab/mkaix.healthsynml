resource "aws_s3_bucket" "models_bucket" {
  bucket           = "healthsync-ml-models-${var.environment}-819109476069-ap-southeast-1-an"
  bucket_namespace = "account-regional"

  tags = {
    Name        = "healthsync-ml-models-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "models_bucket_public_block" {
  bucket = aws_s3_bucket.models_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
