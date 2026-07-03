variable "app_name" { type = string }
variable "environment" { type = string }
variable "dlq_name" { type = string }
variable "ecs_cluster_name" { type = string }
variable "ecs_service_name" { type = string }

# Email to notify on alarm. Leave empty to create the topic without a
# subscription (alarms still fire; wire up a subscriber later).
variable "alert_email" {
  type    = string
  default = ""
}

# ── Notification topic ──────────────────────────────────────────────────────

resource "aws_sns_topic" "alerts" {
  name = "${var.app_name}-alerts"
  tags = { Environment = var.environment }
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── Alarms ──────────────────────────────────────────────────────────────────

# Any message on the DLQ means an analysis job failed all its retries.
resource "aws_cloudwatch_metric_alarm" "dlq_not_empty" {
  alarm_name          = "${var.app_name}-analysis-dlq-not-empty"
  alarm_description   = "Analysis jobs are landing in the dead-letter queue (failed after retries)."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = var.dlq_name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
  tags                = { Environment = var.environment }
}

# Backend service has no running tasks — the API is down. Uses the
# Container Insights RunningTaskCount metric (Insights is enabled on the cluster).
resource "aws_cloudwatch_metric_alarm" "backend_down" {
  alarm_name        = "${var.app_name}-backend-no-running-tasks"
  alarm_description = "The backend ECS service has fewer than one running task."
  namespace         = "ECS/ContainerInsights"
  metric_name       = "RunningTaskCount"
  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = var.ecs_service_name
  }
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
  tags                = { Environment = var.environment }
}

output "sns_topic_arn" { value = aws_sns_topic.alerts.arn }
