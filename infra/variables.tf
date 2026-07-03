variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "app_name" {
  type    = string
  default = "courtvision"
}

variable "environment" {
  type    = string
  default = "production"
}

variable "backend_image_tag" {
  type        = string
  description = "Docker image tag to deploy (e.g. git SHA). Defaults to 'latest'."
  default     = "latest"
}

variable "key_pair_name" {
  type        = string
  description = "EC2 key pair name for optional SSH access to the CV worker. Leave empty to skip."
  default     = ""
}

# ── Backend environment variables ─────────────────────────────────────────────
# Sensitive values (Supabase keys) should be set via TF_VAR_* environment
# variables or a .tfvars file that is NOT committed to git.

variable "supabase_url" {
  type      = string
  sensitive = true
}

variable "supabase_service_role_key" {
  type      = string
  sensitive = true
}

variable "supabase_anon_key" {
  type      = string
  sensitive = true
}

variable "frontend_url" {
  type        = string
  description = "Vercel frontend URL (for CORS). Set after first Vercel deploy."
  default     = "https://your-app.vercel.app"
}

variable "internal_job_secret" {
  type        = string
  sensitive   = true
  description = "Shared secret authenticating server→server CV-job callbacks (INTERNAL_JOB_SECRET). Must match the value in the EC2 worker's /app/.env."
}

variable "acm_certificate_arn" {
  type        = string
  default     = ""
  description = "ACM certificate ARN for the backend API domain. When set, the ALB serves HTTPS (443) and redirects HTTP→HTTPS. Requires a custom API domain (ACM cannot certify the *.elb.amazonaws.com name)."
}

variable "alert_email" {
  type        = string
  default     = ""
  description = "Email address to receive CloudWatch alarm notifications (DLQ failures, backend down). Leave empty to create the SNS topic without a subscription. Email subscriptions require confirming the AWS opt-in email."
}
