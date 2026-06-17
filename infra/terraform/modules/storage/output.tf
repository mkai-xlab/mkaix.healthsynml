output "models_bucket_name" {
  value       = aws_s3_bucket.models_bucket.id
  description = "The name of the ML models S3 bucket"
}

output "models_bucket_arn" {
  value       = aws_s3_bucket.models_bucket.arn
  description = "The ARN of the ML models S3 bucket"
}
