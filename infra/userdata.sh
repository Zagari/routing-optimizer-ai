#!/bin/bash
set -e

# Log setup
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "=== Starting user data script at $(date) ==="

# Variables from Terraform
AWS_REGION="${aws_region}"
PROJECT_NAME="${project_name}"
GITHUB_REPO="${github_repo}"
SECRET_NAME="${secret_name}"

# Update system
echo "=== Updating system packages ==="
dnf update -y

# Install dependencies
echo "=== Installing dependencies ==="
dnf install -y python3.11 python3.11-pip git nginx

# Create app user
useradd -m -s /bin/bash streamlit || true

# Clone repository
echo "=== Cloning repository ==="
cd /opt
git clone $GITHUB_REPO app
chown -R streamlit:streamlit /opt/app

# Setup Python environment
echo "=== Setting up Python environment ==="
cd /opt/app
sudo -u streamlit python3.11 -m venv venv
sudo -u streamlit /opt/app/venv/bin/pip install --upgrade pip
sudo -u streamlit /opt/app/venv/bin/pip install -e ".[all]"
sudo -u streamlit /opt/app/venv/bin/pip install boto3

# Create Streamlit config directory
mkdir -p /home/streamlit/.streamlit
chown -R streamlit:streamlit /home/streamlit/.streamlit

# Create Streamlit config (headless mode)
cat > /home/streamlit/.streamlit/config.toml <<EOF
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = true

[theme]
base = "dark"

[browser]
gatherUsageStats = false
EOF

chown streamlit:streamlit /home/streamlit/.streamlit/config.toml

# Create environment file with region info
cat > /opt/app/.env.aws <<EOF
AWS_REGION=$AWS_REGION
SECRET_NAME=$SECRET_NAME
EOF

chown streamlit:streamlit /opt/app/.env.aws

# Create systemd service
echo "=== Creating systemd service ==="
cat > /etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit Routing Optimizer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=streamlit
Group=streamlit
WorkingDirectory=/opt/app
EnvironmentFile=/opt/app/.env.aws
ExecStart=/opt/app/venv/bin/streamlit run src/routing_optimizer/app/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Streamlit
systemctl daemon-reload
systemctl enable streamlit.service
systemctl start streamlit.service

# Configure nginx (basic, without SSL for now)
echo "=== Configuring nginx ==="
cat > /etc/nginx/conf.d/streamlit.conf <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF

# Start nginx
systemctl enable nginx
systemctl start nginx

echo "=== User data script completed at $(date) ==="
echo "Application should be available at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
