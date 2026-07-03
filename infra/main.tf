terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Recommended: enable S3 backend for shared state once the bucket exists.
  # Run once without this block, then migrate:
  #   terraform init -migrate-state
  #
  # backend "s3" {
  #   bucket         = "courtvision-terraform-state"
  #   key            = "production/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "courtvision-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
}

# ── Modules ───────────────────────────────────────────────────────────────────

module "networking" {
  source      = "./modules/networking"
  app_name    = var.app_name
  environment = var.environment
}

module "ecr" {
  source   = "./modules/ecr"
  app_name = var.app_name
}

module "sqs" {
  source      = "./modules/sqs"
  app_name    = var.app_name
  environment = var.environment
}

module "ecs" {
  source      = "./modules/ecs"
  app_name    = var.app_name
  environment = var.environment
  aws_region  = var.aws_region

  vpc_id            = module.networking.vpc_id
  public_subnet_ids = module.networking.public_subnet_ids
  alb_sg_id         = module.networking.alb_sg_id
  backend_sg_id     = module.networking.backend_sg_id

  backend_image_url      = "${module.ecr.backend_repo_url}:${var.backend_image_tag}"
  sqs_publish_policy_arn = module.sqs.sqs_publish_policy_arn
  acm_certificate_arn    = var.acm_certificate_arn

  env_vars = {
    SUPABASE_URL              = var.supabase_url
    SUPABASE_SERVICE_ROLE_KEY = var.supabase_service_role_key
    SUPABASE_ANON_KEY         = var.supabase_anon_key
    ALLOWED_ORIGINS           = var.frontend_url
    BACKEND_URL               = "http://${module.ecs.alb_dns_name}"
    SQS_QUEUE_URL             = module.sqs.queue_url
    AWS_REGION_NAME           = var.aws_region
    INTERNAL_JOB_SECRET       = var.internal_job_secret
  }
}

module "ec2_worker" {
  source      = "./modules/ec2-worker"
  app_name    = var.app_name
  environment = var.environment
  aws_region  = var.aws_region

  vpc_id          = module.networking.vpc_id
  subnet_id       = module.networking.public_subnet_ids[0]
  cv_worker_sg_id = module.networking.cv_worker_sg_id

  sqs_consume_policy_arn = module.sqs.sqs_consume_policy_arn
  sqs_queue_url          = module.sqs.queue_url
  key_pair_name          = var.key_pair_name
}

module "monitoring" {
  source      = "./modules/monitoring"
  app_name    = var.app_name
  environment = var.environment

  dlq_name         = module.sqs.dlq_name
  ecs_cluster_name = module.ecs.cluster_name
  ecs_service_name = module.ecs.service_name
  alert_email      = var.alert_email
}
