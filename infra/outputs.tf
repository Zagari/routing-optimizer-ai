output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.app.id
}

output "public_ip" {
  description = "Public IP address"
  value       = aws_eip.app.public_ip
}

output "app_url_http" {
  description = "Application URL (HTTP - temporary)"
  value       = "http://${aws_eip.app.public_ip}:8501"
}

output "app_url_https" {
  description = "Application URL (HTTPS - after nginx setup)"
  value       = var.domain_name != "" ? "https://${var.domain_name}" : "Configure domain for HTTPS"
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/id_rsa ec2-user@${aws_eip.app.public_ip}"
}

output "secret_arn" {
  description = "Secrets Manager ARN"
  value       = aws_secretsmanager_secret.openai_key.arn
}
