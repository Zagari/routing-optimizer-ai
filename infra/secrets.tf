# Secret para OpenAI API Key
resource "aws_secretsmanager_secret" "openai_key" {
  name        = "${var.project_name}/openai-api-key"
  description = "OpenAI API key for Routing Optimizer"

  recovery_window_in_days = 7

  tags = {
    Name = "${var.project_name}-openai-secret"
  }
}

resource "aws_secretsmanager_secret_version" "openai_key" {
  secret_id     = aws_secretsmanager_secret.openai_key.id
  secret_string = var.openai_api_key
}
