# 🚀 EC2 Deployment Guide - Attendance App

## Overview

This guide explains how to deploy the Face Recognition Attendance app on AWS EC2 so users can access it from their mobile phones.

**Key Changes for Cloud Deployment:**
- ✅ Camera now runs on **client-side** (user's phone) instead of server
- ✅ Uses **HTTPS** (required for mobile camera access)
- ✅ **Docker + Nginx** for production-ready deployment
- ✅ Mobile-optimized interface at `/mobile_attendance`

---

## Prerequisites

1. **AWS Account** with EC2 access
2. **Domain name** (recommended) OR use EC2's public IP
3. **Supabase** account with database set up
4. **face_encodings.pkl** file (generated from `train_model.py`)

---

## Step 1: Launch EC2 Instance

### Recommended Instance:
- **Type:** t3.medium (2 vCPU, 4GB RAM) - minimum for face recognition
- **AMI:** Ubuntu 22.04 LTS
- **Storage:** 20GB SSD

### Security Group Rules:
| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| HTTP | 80 | 0.0.0.0/0 | Web (redirects to HTTPS) |
| HTTPS | 443 | 0.0.0.0/0 | Secure web access |

---

## Step 2: Connect to EC2

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

---

## Step 3: Install Docker

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again to apply docker group
exit
```

---

## Step 4: Deploy the Application

```bash
# Clone your repository
git clone https://github.com/samikshapawar-xenonv2/Attendance.git
cd Attendance

# Create .env file with your credentials
cat > .env << EOF
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
ENCODINGS_FILE=face_encodings.pkl
FLASK_SECRET_KEY=$(openssl rand -hex 32)
ADMIN_EMAIL=your_admin_email
ADMIN_PASSWORD=your_admin_password
EOF

# Create SSL directory
mkdir -p ssl certbot/www
```

---

## Step 5: SSL Certificate Setup

### Option A: Self-Signed Certificate (Quick Testing)

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/privkey.pem \
    -out ssl/fullchain.pem \
    -subj "/CN=your-domain-or-ip"
```

⚠️ **Note:** Self-signed certs will show browser warnings. Use Let's Encrypt for production.

### Option B: Let's Encrypt (Production)

1. Point your domain to EC2's public IP
2. Install certbot:
```bash
sudo apt install certbot -y
```

3. Get certificate:
```bash
sudo certbot certonly --standalone -d yourdomain.com
```

4. Copy certificates:
```bash
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/
sudo chown ubuntu:ubuntu ssl/*
```

---

## Step 6: Build and Run

```bash
# Build and start the containers
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## Step 7: Access Your App

- **HTTPS:** `https://your-domain.com` or `https://your-ec2-ip`
- **Mobile Attendance:** `https://your-domain.com/mobile_attendance`

### Mobile Usage:
1. Open the URL on your phone
2. Allow camera access when prompted
3. Point camera at students
4. Faces are detected and matched automatically
5. Click "Save Attendance" when done

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Mobile Phone                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Browser (Camera Access via WebRTC/getUserMedia)    │    │
│  │  - Captures video frames                             │    │
│  │  - Sends base64 images to server                     │    │
│  │  - Displays detection results                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS (Port 443)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         EC2 Instance                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Nginx (SSL Proxy)                  │    │
│  │  - SSL termination                                    │    │
│  │  - Rate limiting                                      │    │
│  │  - Gzip compression                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Flask App (Gunicorn)                     │    │
│  │  - /api/process_frame - Face recognition              │    │
│  │  - /mobile_attendance - Mobile UI                     │    │
│  │  - MediaPipe + dlib face detection                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                Supabase (Database)                    │    │
│  │  - Students table                                     │    │
│  │  - Attendance records                                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/mobile_attendance` | GET | Mobile camera interface |
| `/live_attendance` | GET | Server webcam (local only) |
| `/api/process_frame` | POST | Process image for face detection |
| `/api/session/status` | GET | Get detected students |
| `/api/session/reset` | POST | Clear current session |
| `/finalize_attendance` | POST | Save attendance to database |

---

## Troubleshooting

### Camera not working on mobile?
- Ensure you're using **HTTPS** (required for getUserMedia)
- Check browser permissions for camera access
- Try a different browser (Chrome works best)

### Face recognition slow?
- EC2 t3.medium recommended (t2.micro is too slow)
- Check if Docker has enough memory
- Consider reducing frame resolution

### SSL certificate issues?
- Verify certificate files exist in `ssl/` folder
- Check nginx logs: `docker-compose logs nginx`
- For Let's Encrypt, ensure domain DNS is correctly configured

### Container won't start?
- Check logs: `docker-compose logs app`
- Verify .env file has all required variables
- Ensure face_encodings.pkl exists

---

## Useful Commands

```bash
# View all logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up --build -d

# Check container resource usage
docker stats

# Enter app container for debugging
docker exec -it attendance-app bash
```

---

## Cost Estimate (AWS)

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| EC2 t3.medium | 2 vCPU, 4GB RAM | ~$30 |
| EBS Storage | 20GB SSD | ~$2 |
| Data Transfer | 100GB/month | ~$9 |
| **Total** | | **~$41/month** |

💡 **Tip:** Use EC2 Spot Instances for ~70% savings if occasional interruptions are acceptable.

---

## Security Recommendations

1. ✅ Always use HTTPS
2. ✅ Change default admin credentials
3. ✅ Use strong FLASK_SECRET_KEY
4. ✅ Keep Supabase credentials private
5. ✅ Regularly update Docker images
6. ✅ Enable AWS CloudWatch for monitoring
7. ✅ Set up automated backups

---

## Support

For issues, check:
1. Container logs: `docker-compose logs`
2. Nginx access logs: `docker-compose logs nginx`
3. Browser console for JavaScript errors
