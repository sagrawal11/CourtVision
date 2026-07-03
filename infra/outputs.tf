output "backend_url" {
  description = "Public URL for the FastAPI backend (set as NEXT_PUBLIC_API_URL in Vercel)"
  value       = "http://${module.ecs.alb_dns_name}"
}

output "alerts_sns_topic_arn" {
  description = "SNS topic that receives CloudWatch alarm notifications (DLQ failures, backend down)"
  value       = module.monitoring.sns_topic_arn
}

output "ecr_backend_repo" {
  description = "ECR repository URL — used in docker push and CI/CD"
  value       = module.ecr.backend_repo_url
}

output "sqs_queue_url" {
  description = "SQS queue URL — set as SQS_QUEUE_URL in both backend and CV worker .env"
  value       = module.sqs.queue_url
}

output "models_s3_bucket" {
  description = "S3 bucket for model weights — run: aws s3 sync models/ s3://<bucket>/models/"
  value       = module.ec2_worker.models_bucket_name
}

output "cv_worker_instance_id" {
  description = "EC2 CV worker instance ID — use SSM to connect: aws ssm start-session --target <id>"
  value       = module.ec2_worker.instance_id
}

output "cv_worker_public_ip" {
  description = "CV worker public IP (changes on stop/start — use SSM for stable access)"
  value       = module.ec2_worker.instance_public_ip
}
