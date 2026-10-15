# Deployment Guide
## Agentic Multimodal Explainable RAG Framework for Personalized Bone Marrow Assessment

**Version**: 1.0  
**Last Updated**: October 2026  
**Status**: Production Ready  

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Local)](#quick-start-local)
3. [Environment Configuration](#environment-configuration)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)
9. [Security Considerations](#security-considerations)
10. [Monitoring & Logging](#monitoring--logging)

---

## Prerequisites

### **System Requirements**:

**Minimum**:
- OS: Windows 10/11, Ubuntu 20.04+, macOS 11+
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB free space
- Internet: Required (for Groq API)

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- GPU: Optional (CUDA-compatible for future enhancements)
- Storage: 10GB+ for models and data

### **Software Requirements**:

- **Python**: 3.10 or 3.11 (3.14 tested and working)
- **pip**: Latest version
- **Git**: For cloning repository
- **Web Browser**: Chrome, Firefox, Edge, or Safari (latest versions)

---

## Quick Start (Local)

### **5-Minute Setup**:

```bash
# 1. Clone the repository
git clone <repository-url>
cd Multimodal_AI_Bone_Marrow

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Verify setup
python setup_groq_rag.py

# 6. Run the application
streamlit run app/streamlit_app.py
# Or: python -m streamlit run app/streamlit_app.py
```

### **Access the Application**:
- Local URL: http://localhost:8501
- The app will automatically open in your default browser

---

## Environment Configuration

### **1. Create `.env` File**:

```bash
# Copy the example file
cp .env.example .env
```

### **2. Required Variables**:

```bash
# .env file
# =============================================================================
# LLM Provider (Groq)
# =============================================================================
GROQ_API_KEY=your_groq_api_key_here

# Get your free API key at: https://console.groq.com
# Free tier: 14,400 requests/day (600 patients)

# =============================================================================
# Optional Configuration
# =============================================================================

# Synthesizer mode: "true" for LLM, "false" for deterministic
SYNTHESIZER_USE_LLM=true

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Model checkpoint path (optional, defaults to models/checkpoints/)
MODEL_CHECKPOINT_PATH=models/checkpoints/best_vit_lora_morphology.pt

# FAISS index path (optional, defaults to data/processed/retrieval_index/)
FAISS_INDEX_PATH=data/processed/retrieval_index/dense_index.faiss
```

### **3. Verify Configuration**:

```bash
python setup_groq_rag.py
```

**Expected Output**:
```
✓ Configuration loaded
✓ GROQ_API_KEY found
✓ ViT model checkpoint exists
✓ FAISS index loaded (12 chunks)
✓ All checks passed
```

---

## Production Deployment

### **1. Pre-Deployment Checklist**:

- [ ] All tests passing: `pytest tests/ -v`
- [ ] Dependencies pinned in `requirements.txt`
- [ ] `.env` configured with production keys
- [ ] Logs directory created: `mkdir -p logs/`
- [ ] SSL/TLS certificates obtained (for HTTPS)
- [ ] Backup strategy in place
- [ ] Monitoring configured

### **2. Production Environment Variables**:

```bash
# .env.production
GROQ_API_KEY=<production_key>
SYNTHESIZER_USE_LLM=true
LOG_LEVEL=WARNING
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### **3. Run in Production Mode**:

```bash
# Load production environment
export $(cat .env.production | xargs)

# Run with production settings
streamlit run app/streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
```

### **4. Process Management (Linux/macOS)**:

Using **systemd** service:

```bash
# Create service file
sudo nano /etc/systemd/system/bone-marrow-ai.service
```

```ini
[Unit]
Description=Bone Marrow AI Assessment System
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/Multimodal_AI_Bone_Marrow
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run app/streamlit_app.py --server.port=8501 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable bone-marrow-ai
sudo systemctl start bone-marrow-ai
sudo systemctl status bone-marrow-ai
```

### **5. Process Management (Windows)**:

Using **NSSM** (Non-Sucking Service Manager):

```powershell
# Install NSSM
choco install nssm

# Create Windows service
nssm install BoneMarrowAI "C:\path\to\python.exe" "-m streamlit run app/streamlit_app.py"
nssm set BoneMarrowAI AppDirectory "C:\path\to\Multimodal_AI_Bone_Marrow"
nssm start BoneMarrowAI
```

---

## Docker Deployment

### **1. Build Docker Image**:

```bash
# Build the image
docker build -t bone-marrow-ai:latest .

# Verify build
docker images | grep bone-marrow-ai
```

### **2. Run Container**:

```bash
# Run with environment variables
docker run -d \
  --name bone-marrow-ai \
  -p 8501:8501 \
  -e GROQ_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  bone-marrow-ai:latest

# Check logs
docker logs -f bone-marrow-ai
```

### **3. Using Docker Compose**:

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **4. Docker Health Check**:

```bash
# Check if container is healthy
docker inspect --format='{{.State.Health.Status}}' bone-marrow-ai
```

---

## Cloud Deployment

### **AWS Deployment (EC2)**:

```bash
# 1. Launch EC2 instance
# - AMI: Ubuntu 22.04 LTS
# - Instance type: t3.large (8GB RAM)
# - Security group: Allow port 8501

# 2. SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 3. Install dependencies
sudo apt update
sudo apt install -y python3.10 python3-pip git

# 4. Clone and setup
git clone <repository-url>
cd Multimodal_AI_Bone_Marrow
pip3 install -r requirements.txt

# 5. Configure and run
cp .env.example .env
nano .env  # Add GROQ_API_KEY
streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### **Google Cloud Platform (Cloud Run)**:

```bash
# 1. Build and push Docker image
gcloud builds submit --tag gcr.io/PROJECT-ID/bone-marrow-ai

# 2. Deploy to Cloud Run
gcloud run deploy bone-marrow-ai \
  --image gcr.io/PROJECT-ID/bone-marrow-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=your_key_here

# 3. Get service URL
gcloud run services describe bone-marrow-ai --format='value(status.url)'
```

### **Azure (Container Instances)**:

```bash
# 1. Create resource group
az group create --name bone-marrow-rg --location eastus

# 2. Create container instance
az container create \
  --resource-group bone-marrow-rg \
  --name bone-marrow-ai \
  --image your-registry/bone-marrow-ai:latest \
  --dns-name-label bone-marrow-ai \
  --ports 8501 \
  --environment-variables GROQ_API_KEY=your_key_here

# 3. Get FQDN
az container show --resource-group bone-marrow-rg --name bone-marrow-ai --query ipAddress.fqdn
```

---

## Troubleshooting

### **Issue 1: "GROQ_API_KEY not found"**

**Symptom**: Error on startup about missing API key

**Solution**:
```bash
# Check if .env file exists
ls -la .env

# Verify key is set
cat .env | grep GROQ_API_KEY

# Ensure no quotes around the key value
# WRONG: GROQ_API_KEY="your_key"
# RIGHT: GROQ_API_KEY=your_key
```

### **Issue 2: "Model checkpoint not found"**

**Symptom**: `FileNotFoundError: models/checkpoints/best_vit_lora_morphology.pt`

**Solution**:
```bash
# Check if model file exists
ls -la models/checkpoints/

# If missing, you need to train the model first
python vision/train.py

# Or download pre-trained weights (if available)
```

### **Issue 3: "FAISS index not found"**

**Symptom**: `FileNotFoundError: data/processed/retrieval_index/dense_index.faiss`

**Solution**:
```bash
# Build the FAISS index
python retrieval/index.py

# Verify index was created
ls -la data/processed/retrieval_index/
```

### **Issue 4: "Port 8501 already in use"**

**Symptom**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find and kill process using port 8501
# Linux/macOS:
lsof -ti:8501 | xargs kill -9

# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or use a different port
streamlit run app/streamlit_app.py --server.port=8502
```

### **Issue 5: Memory errors during inference**

**Symptom**: `RuntimeError: CUDA out of memory` or system slowdown

**Solution**:
```bash
# Reduce batch size in configs/default.yaml
# Change: batch_size: 64
# To: batch_size: 32 (or 16 for low RAM systems)

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""

# Increase system swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### **Issue 6: Slow performance**

**Symptom**: Analysis takes >30 seconds per image

**Solution**:
1. Check internet connection (Groq API requires internet)
2. Verify GROQ_API_KEY is valid
3. Use GPU if available
4. Reduce image size before upload
5. Check system resources: `top` or `htop`

---

## Performance Tuning

### **1. Optimize Configuration**:

Edit `configs/default.yaml`:

```yaml
# For faster inference (lower quality)
vision:
  batch_size: 32  # Reduce from 64

training:
  precision: fp16  # Use instead of bf16 on older GPUs

# For better accuracy (slower)
retrieval:
  top_k_reranked: 10  # Increase from 5

llm:
  temperature: 0.05  # More deterministic (default: 0.1)
  max_tokens: 4096   # Longer reports (default: 2048)
```

### **2. Caching Strategy**:

```python
# Enable Streamlit caching (already implemented)
# In app/streamlit_app.py, functions use @st.cache_resource
# This caches model loading and FAISS index
```

### **3. Database Optimization** (if using database):

```bash
# Create indexes on patient_id, case_id
# Optimize query performance
# Use connection pooling
```

---

## Security Considerations

### **1. API Key Management**:

```bash
# NEVER commit .env to git
echo ".env" >> .gitignore

# Use environment-specific keys
# .env.development (dev key)
# .env.production (prod key)

# Rotate keys regularly
# https://console.groq.com/keys
```

### **2. Access Control**:

```python
# Add authentication to Streamlit (if needed)
# Option 1: Streamlit-Authenticator
# Option 2: Reverse proxy (nginx) with basic auth
# Option 3: OAuth integration
```

**Example nginx reverse proxy**:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        
        # Basic authentication
        auth_basic "Restricted Access";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

### **3. HTTPS/SSL**:

```bash
# Using Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### **4. Input Validation**:

- Maximum file size: 50MB (configured in `configs/default.yaml`)
- Allowed formats: PNG, JPG, TIFF only
- Image dimension limits: 224×224 to 4000×4000 px

### **5. Rate Limiting**:

```python
# Implement rate limiting for production
# Using Flask-Limiter or similar
# Limit: 10 requests per minute per IP
```

---

## Monitoring & Logging

### **1. Application Logs**:

```bash
# View real-time logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log

# Rotate logs (Linux)
sudo apt install logrotate
sudo nano /etc/logrotate.d/bone-marrow-ai
```

**Logrotate configuration**:
```
/path/to/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### **2. Performance Monitoring**:

```bash
# CPU and memory usage
htop

# Disk usage
df -h
du -sh data/ models/ logs/

# Network monitoring
netstat -tunlp | grep 8501
```

### **3. Application Health Checks**:

```bash
# Create health check endpoint
curl http://localhost:8501/_stcore/health

# Monitor Groq API status
curl https://api.groq.com/openai/v1/models
```

### **4. Error Tracking** (Optional):

Consider integrating:
- **Sentry**: For error tracking and monitoring
- **Prometheus + Grafana**: For metrics visualization
- **ELK Stack**: For log aggregation

---

## Backup & Recovery

### **1. Critical Files to Backup**:

```bash
# Models
models/checkpoints/*.pt

# Data
data/processed/splits.json
data/knowledge_base/*.json
data/processed/retrieval_index/*

# Configuration
configs/*.yaml
.env.production
```

### **2. Backup Script**:

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/bone-marrow-ai"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

tar -czf $BACKUP_DIR/backup_$DATE.tar.gz \
  models/checkpoints/ \
  data/processed/ \
  data/knowledge_base/ \
  configs/ \
  .env.production

# Keep only last 7 backups
ls -t $BACKUP_DIR/backup_*.tar.gz | tail -n +8 | xargs rm -f
```

### **3. Automated Backups**:

```bash
# Add to crontab (daily at 2 AM)
crontab -e
0 2 * * * /path/to/backup.sh
```

---

## Scaling Considerations

### **For High Traffic (>1000 patients/day)**:

1. **Horizontal Scaling**:
   - Deploy multiple instances behind load balancer
   - Use Redis for session management
   - Implement request queuing (Celery + RabbitMQ)

2. **Vertical Scaling**:
   - Upgrade to larger instance (16GB+ RAM)
   - Use GPU for faster inference
   - Optimize batch processing

3. **Database**:
   - Move from file-based to PostgreSQL/MongoDB
   - Implement caching layer (Redis)

4. **CDN**:
   - Serve static assets via CDN
   - Cache responses where appropriate

---

## Support & Maintenance

### **Regular Maintenance Tasks**:

- [ ] Weekly: Check logs for errors
- [ ] Weekly: Monitor disk space usage
- [ ] Weekly: Review API usage (Groq dashboard)
- [ ] Monthly: Update dependencies: `pip install -U -r requirements.txt`
- [ ] Monthly: Review and rotate API keys
- [ ] Monthly: Backup verification
- [ ] Quarterly: Security audit
- [ ] Yearly: Model retraining with new data

### **Getting Help**:

- **Documentation**: `docs/` folder
- **Issues**: GitHub Issues
- **Email**: [Your support email]
- **Community**: [Discord/Slack link if available]

---

## Version Information

**Current Version**: 1.0.0  
**Release Date**: October 2026  
**Python Version**: 3.10+  
**Streamlit Version**: 1.56.0  
**Key Dependencies**: See `requirements.txt`

---

## License

See `LICENSE` file for details.

---

**Document Version**: 1.0  
**Last Reviewed**: October 2026  
**Next Review**: January 2027
