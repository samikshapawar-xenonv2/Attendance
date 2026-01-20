#!/bin/bash
# ============================================
# EC2 Deployment Script for Attendance App
# ============================================
# Run this script on a fresh EC2 Ubuntu instance
# Usage: chmod +x deploy.sh && ./deploy.sh
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Attendance App - EC2 Deployment Script${NC}"
echo -e "${GREEN}============================================${NC}"

# Configuration - UPDATE THESE VALUES
DOMAIN_NAME="${DOMAIN_NAME:-your-domain.com}"  # Set via environment or replace here
APP_DIR="/home/ubuntu/attendance"

# Step 1: Update system
echo -e "\n${YELLOW}[1/8] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install Docker
echo -e "\n${YELLOW}[2/8] Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    rm get-docker.sh
    echo -e "${GREEN}Docker installed successfully${NC}"
else
    echo -e "${GREEN}Docker already installed${NC}"
fi

# Step 3: Install Docker Compose
echo -e "\n${YELLOW}[3/8] Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed successfully${NC}"
else
    echo -e "${GREEN}Docker Compose already installed${NC}"
fi

# Step 4: Create app directory
echo -e "\n${YELLOW}[4/8] Setting up application directory...${NC}"
mkdir -p $APP_DIR
mkdir -p $APP_DIR/ssl
mkdir -p $APP_DIR/certbot/www
cd $APP_DIR

# Step 5: Clone or pull latest code
echo -e "\n${YELLOW}[5/8] Pulling latest code...${NC}"
if [ -d ".git" ]; then
    git pull origin main
else
    # Clone your repository - UPDATE THIS URL
    git clone https://github.com/samikshapawar-xenonv2/Attendance.git .
fi

# Step 6: Create .env file if not exists
echo -e "\n${YELLOW}[6/8] Checking environment configuration...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}WARNING: .env file not found!${NC}"
    echo "Please create .env file with the following variables:"
    echo "---"
    cat << EOF
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ENCODINGS_FILE=face_encodings.pkl
FLASK_SECRET_KEY=your_secret_key_here
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=your_admin_password
EOF
    echo "---"
    echo "Create the file and run this script again."
    exit 1
else
    echo -e "${GREEN}.env file found${NC}"
fi

# Step 7: Generate self-signed SSL certificate (for testing)
# For production, use Let's Encrypt (see below)
echo -e "\n${YELLOW}[7/8] Setting up SSL certificates...${NC}"
if [ ! -f "ssl/fullchain.pem" ]; then
    echo "Generating self-signed SSL certificate for testing..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/privkey.pem \
        -out ssl/fullchain.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN_NAME"
    echo -e "${GREEN}Self-signed certificate created${NC}"
    echo -e "${YELLOW}NOTE: For production, use Let's Encrypt. See instructions below.${NC}"
else
    echo -e "${GREEN}SSL certificates already exist${NC}"
fi

# Step 8: Build and start containers
echo -e "\n${YELLOW}[8/8] Building and starting Docker containers...${NC}"
sudo docker-compose down 2>/dev/null || true
sudo docker-compose build --no-cache
sudo docker-compose up -d

# Wait for services to start
echo -e "\n${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Check status
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN} Deployment Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
sudo docker-compose ps

echo -e "\n${GREEN}Your app is now running!${NC}"
echo -e "Access it at:"
echo -e "  - HTTP:  http://$DOMAIN_NAME (redirects to HTTPS)"
echo -e "  - HTTPS: https://$DOMAIN_NAME"
echo ""
echo -e "${YELLOW}For Let's Encrypt SSL (production):${NC}"
echo "1. Ensure your domain points to this server's IP"
echo "2. Run: sudo certbot certonly --webroot -w ./certbot/www -d $DOMAIN_NAME"
echo "3. Copy certificates: sudo cp /etc/letsencrypt/live/$DOMAIN_NAME/* ./ssl/"
echo "4. Restart: sudo docker-compose restart nginx"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  View logs:     sudo docker-compose logs -f"
echo "  Restart:       sudo docker-compose restart"
echo "  Stop:          sudo docker-compose down"
echo "  Rebuild:       sudo docker-compose up --build -d"
