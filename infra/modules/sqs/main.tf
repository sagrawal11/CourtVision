variable "app_name"   { type = string }
variable "environment" { type = string }

# Dead-letter queue — receives messages after 3 failed processing attempts
resource "aws_sqs_queue" "dlq" {
  name                      = "${var.app_name}-analysis-jobs-dlq"
  message_retention_seconds = 1209600  # 14 days
  tags = { Environment = var.environment }
}

# Main analysis job queue
resource "aws_sqs_queue" "analysis_jobs" {
  name                       = "${var.app_name}-analysis-jobs"
  visibility_timeout_seconds = 14400   # 4 hours — must exceed longest analysis time
  message_retention_seconds  = 345600  # 4 days
  receive_wait_time_seconds  = 20      # long polling

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })

  tags = { Environment = var.environment }
}

# IAM policy document for publishing (used by backend/ECS task role)
data "aws_iam_policy_document" "publish" {
  statement {
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.analysis_jobs.arn]
  }
}

# IAM policy document for consuming (used by EC2 worker instance role)
data "aws_iam_policy_document" "consume" {
  statement {
    actions   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"]
    resources = [aws_sqs_queue.analysis_jobs.arn]
  }
}

resource "aws_iam_policy" "sqs_publish" {
  name   = "${var.app_name}-sqs-publish"
  policy = data.aws_iam_policy_document.publish.json
}

resource "aws_iam_policy" "sqs_consume" {
  name   = "${var.app_name}-sqs-consume"
  policy = data.aws_iam_policy_document.consume.json
}

output "queue_url"          { value = aws_sqs_queue.analysis_jobs.url }
output "queue_arn"          { value = aws_sqs_queue.analysis_jobs.arn }
output "dlq_url"            { value = aws_sqs_queue.dlq.url }
output "sqs_publish_policy_arn" { value = aws_iam_policy.sqs_publish.arn }
output "sqs_consume_policy_arn" { value = aws_iam_policy.sqs_consume.arn }
