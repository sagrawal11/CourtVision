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
