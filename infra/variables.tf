variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "routing-optimizer"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "openai_api_key" {
  description = "OpenAI API key (will be stored in Secrets Manager)"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to SSH (your IP)"
  type        = string
  default     = "0.0.0.0/0" # Restringir em producao!
}

variable "github_repo" {
  description = "GitHub repository URL"
  type        = string
  default     = "https://github.com/Zagari/routing-optimizer-ai.git"
}
