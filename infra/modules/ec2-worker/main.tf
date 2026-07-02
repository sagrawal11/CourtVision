variable "app_name"          { type = string }
variable "environment"        { type = string }
variable "aws_region"         { type = string }
variable "vpc_id"             { type = string }
variable "subnet_id"          { type = string }
variable "cv_worker_sg_id"    { type = string }
variable "sqs_consume_policy_arn" { type = string }
variable "sqs_queue_url"      { type = string }
variable "key_pair_name"      {
  type        = string
  description = "EC2 key pair name for SSH access (must already exist in AWS)"
  default     = ""
}

# ── S3 bucket for model weights ───────────────────────────────────────────────
# Upload model weights once: aws s3 sync models/ s3://<bucket>/models/

resource "aws_s3_bucket" "models" {
  bucket_prefix = "${var.app_name}-models-"
  tags          = { Environment = var.environment }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM for EC2 worker ────────────────────────────────────────────────────────

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = "${var.app_name}-cv-worker"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "worker_sqs" {
  role       = aws_iam_role.worker.name
  policy_arn = var.sqs_consume_policy_arn
}

# S3 read access for model weights
resource "aws_iam_role_policy" "worker_s3" {
  name = "${var.app_name}-worker-s3"
  role = aws_iam_role.worker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
    }]
  })
}

# SSM access for remote shell (no inbound SSH needed)
resource "aws_iam_role_policy_attachment" "worker_ssm" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "worker" {
  name = "${var.app_name}-cv-worker"
  role = aws_iam_role.worker.name
}

# ── AMI — Deep Learning OSS AMI (PyTorch + CUDA pre-installed) ────────────────

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04) *"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ── EC2 instance ──────────────────────────────────────────────────────────────

locals {
  user_data = <<-USERDATA
    #!/bin/bash
    set -euxo pipefail
    exec > /var/log/cv-worker-init.log 2>&1

    # The DL AMI uses a conda env; activate it for all subsequent commands
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch

    # Extra Python deps not in the DL AMI
    pip install --quiet boto3 supabase catboost scipy tqdm yt-dlp beautifulsoup4

    # Sync model weights from S3 (run at boot; idempotent)
    mkdir -p /app/models
    aws s3 sync s3://${aws_s3_bucket.models.bucket}/models/ /app/models/ --region ${var.aws_region}

    # Create systemd service
    cat > /etc/systemd/system/cv-worker.service <<'SERVICE'
    [Unit]
    Description=Courtvision CV SQS Worker
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    User=ubuntu
    WorkingDirectory=/app
    EnvironmentFile=/app/.env
    ExecStart=/opt/conda/envs/pytorch/bin/python cv/sqs_worker.py
    Restart=on-failure
    RestartSec=15

    [Install]
    WantedBy=multi-user.target
    SERVICE

    systemctl daemon-reload
    systemctl enable cv-worker

    echo "Init complete — deploy code to /app, populate /app/.env, then: systemctl start cv-worker"
  USERDATA
}

resource "aws_instance" "worker" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = "g4dn.xlarge"
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.cv_worker_sg_id]
  iam_instance_profile   = aws_iam_instance_profile.worker.name
  key_name               = var.key_pair_name != "" ? var.key_pair_name : null

  root_block_device {
    volume_type = "gp3"
    volume_size = 100   # models + video scratch space
    encrypted   = true
  }

  user_data = local.user_data

  # Stop the instance when not processing to save money (~$0.53/hr on-demand)
  # Use spot for up to 70% discount (update instance_market_options below)
  # instance_market_options {
  #   market_type = "spot"
  #   spot_options { spot_instance_type = "one-time" }
  # }

  tags = {
    Name        = "${var.app_name}-cv-worker"
    Environment = var.environment
    StopWhenIdle = "true"
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "instance_id"       { value = aws_instance.worker.id }
output "instance_public_ip" { value = aws_instance.worker.public_ip }
output "models_bucket_name" { value = aws_s3_bucket.models.bucket }
