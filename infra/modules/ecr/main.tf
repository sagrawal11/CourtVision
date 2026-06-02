variable "app_name" { type = string }

resource "aws_ecr_repository" "backend" {
  name                 = "${var.app_name}-backend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration { scan_on_push = true }

  lifecycle {
    prevent_destroy = false
  }
}

# Keep only the 10 most recent images to control storage costs
resource "aws_ecr_lifecycle_policy" "backend" {
  repository = aws_ecr_repository.backend.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

output "backend_repo_url"  { value = aws_ecr_repository.backend.repository_url }
output "backend_repo_name" { value = aws_ecr_repository.backend.name }
output "registry_id"       { value = aws_ecr_repository.backend.registry_id }
