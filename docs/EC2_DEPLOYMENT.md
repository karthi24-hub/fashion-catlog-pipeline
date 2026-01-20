# EC2 Deployment Guide for Fashion Visual Search API

## Prerequisites
- EC2 instance with GPU (recommended: g4dn.xlarge or larger)
- Ubuntu 20.04/22.04 AMI
- Security group allowing port 8000 (or your chosen port)

## Step 1: Connect to EC2

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

## Step 2: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3.10 python3.10-venv python3-pip git

# For GPU: Install NVIDIA drivers (if using GPU instance)
sudo apt install -y nvidia-driver-535
# Reboot after driver install
sudo reboot
```

## Step 3: Clone Repository

```bash
cd ~
git clone <your-repo-url> fashion-catlog-pipeline
cd fashion-catlog-pipeline
```

## Step 4: Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For GPU PyTorch (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 5: Configure Environment

```bash
# Create .env file
cat > .env << EOF
API_KEY=your_secure_api_key_here
S3_BUCKET=shoptainment-dev-fashion-dataset-bucket
S3_PREFIX=dataset/products/
S3_REGION=ap-south-1
EOF
```

## Step 6: Configure AWS Credentials

**Option A: IAM Role (Recommended for EC2)**
- Attach an IAM role to your EC2 instance with S3 read access
- No credentials needed in code

**Option B: AWS Credentials File**
```bash
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
EOF

cat > ~/.aws/config << EOF
[default]
region = ap-south-1
EOF
```

## Step 7: Download YOLO Weights

```bash
mkdir -p models
cd models
# Download YOLOv8n weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
cd ..
```

## Step 8: Build/Verify FAISS Index

```bash
# Check if index exists
ls -la faiss/

# If not, build from S3 embeddings
python src/build_faiss_index.py
```

## Step 9: Test Locally

```bash
source venv/bin/activate
uvicorn src.api:app --host 127.0.0.1 --port 8000

# In another terminal, test:
curl http://127.0.0.1:8000/health
```

## Step 10: Run Production Server

**Using Screen (Simple)**
```bash
screen -S api
source venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2
# Ctrl+A, D to detach
```

**Using Systemd (Recommended)**
```bash
sudo cat > /etc/systemd/system/fashion-api.service << EOF
[Unit]
Description=Fashion Visual Search API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fashion-catlog-pipeline
Environment="PATH=/home/ubuntu/fashion-catlog-pipeline/venv/bin"
ExecStart=/home/ubuntu/fashion-catlog-pipeline/venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fashion-api
sudo systemctl start fashion-api
sudo systemctl status fashion-api
```

## Step 11: Test from Outside

```bash
# From your local machine
curl -X POST http://your-ec2-public-ip:8000/search \
  -H "x-api-key: your_api_key" \
  -F "file=@test_image.jpg"
```

## Troubleshooting

### S3 Access Denied
- Verify IAM role/credentials have `s3:GetObject` permission
- Check bucket name and prefix are correct
- Pre-signed URLs should work for private buckets

### CUDA/GPU Errors
```bash
# Check GPU is detected
nvidia-smi

# Test PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce `MAX_ITEMS_PER_IMAGE` in api.py
- Use smaller batch sizes
- Consider using `faiss-cpu` instead of `faiss-gpu`

## Optional: Run Catalog Enrichment

```bash
# Test with 10 products first
python scripts/enrich_catalog.py --test

# Full enrichment (run overnight)
nohup python scripts/enrich_catalog.py --workers=4 > enrich.log 2>&1 &
```

## Security Checklist

- [ ] Use strong API_KEY
- [ ] Restrict security group to needed IPs
- [ ] Use HTTPS with nginx/certbot
- [ ] Enable CloudWatch logging
- [ ] Set up auto-scaling if needed
