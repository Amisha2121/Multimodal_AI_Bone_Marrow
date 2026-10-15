# Production Deployment Checklist
## Bone Marrow AI Assessment System

**Version**: 1.0  
**Date**: October 2026  

---

## 📋 Pre-Deployment Checklist

### **1. Code Quality** ✅

- [ ] All unit tests passing (`pytest tests/ -v`)
- [ ] Integration tests passing
- [ ] Code coverage ≥80%
- [ ] No critical linting errors
- [ ] All TODOs resolved or documented
- [ ] Code reviewed by team members

### **2. Dependencies** ✅

- [ ] `requirements.txt` up to date and pinned
- [ ] All dependencies vulnerability-scanned
- [ ] No deprecated packages
- [ ] Optional dependencies documented
- [ ] Python version specified (3.10+)

### **3. Configuration** ✅

- [ ] `.env.production` created with production values
- [ ] All required environment variables documented
- [ ] Sensitive data NOT in version control
- [ ] `configs/default.yaml` reviewed for production
- [ ] Logging configured appropriately (WARN level)

### **4. Security** ✅

- [ ] API keys secured and not exposed
- [ ] Input validation implemented
- [ ] File upload size limits enforced (50MB)
- [ ] HTTPS/TLS configured for production
- [ ] CORS settings configured (if applicable)
- [ ] Rate limiting implemented
- [ ] Security headers configured

### **5. Data & Models** ✅

- [ ] Model checkpoint exists and tested
- [ ] FAISS index built and verified
- [ ] Knowledge base (12 WHO chunks) loaded
- [ ] Patient-level splits verified (no leakage)
- [ ] Test images available for smoke testing

### **6. Performance** ✅

- [ ] Load testing completed
- [ ] Memory usage profiled (<8GB recommended)
- [ ] Response time acceptable (<15s per case)
- [ ] Concurrent user limit tested
- [ ] Database queries optimized (if applicable)

### **7. Monitoring & Logging** ✅

- [ ] Logging configured (file + console)
- [ ] Log rotation configured
- [ ] Health check endpoint working
- [ ] Error tracking set up (Sentry, etc.)
- [ ] Performance monitoring configured
- [ ] Alerting rules defined

### **8. Backup & Recovery** ✅

- [ ] Backup strategy documented
- [ ] Automated backups configured
- [ ] Recovery procedure tested
- [ ] Backup retention policy defined (7 days minimum)

### **9. Documentation** ✅

- [ ] README.md comprehensive
- [ ] Deployment guide complete (`docs/DEPLOYMENT.md`)
- [ ] API documentation available
- [ ] Troubleshooting guide included
- [ ] Architecture diagrams up to date
- [ ] Change log maintained

### **10. Legal & Compliance** ⚠️

- [ ] License file included
- [ ] Privacy policy prepared (if handling PHI)
- [ ] Terms of service prepared
- [ ] FDA compliance reviewed (research use only)
- [ ] HIPAA compliance (if applicable)
- [ ] GDPR compliance (if EU users)
- [ ] Disclaimer clearly visible in UI

---

## 🚀 Deployment Steps

### **Phase 1: Pre-Production Testing**

#### **Step 1: Local Verification** (30 minutes)

```bash
# 1. Clean environment
rm -rf venv/
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Fresh install
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v --cov=. --cov-report=html

# 4. Verify setup
python setup_groq_rag.py

# 5. Test application
streamlit run app/streamlit_app.py
# Upload test image and verify full workflow
```

**Checklist**:
- [ ] All tests pass (green)
- [ ] Coverage ≥80%
- [ ] Setup script successful
- [ ] App runs without errors
- [ ] Test image processes correctly
- [ ] Report generation works
- [ ] Attention maps display correctly

---

#### **Step 2: Docker Build** (15 minutes)

```bash
# 1. Build image
docker build -t bone-marrow-ai:latest .

# 2. Check image size (should be <2GB)
docker images | grep bone-marrow-ai

# 3. Test run
docker run -d \
  --name test-container \
  -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  bone-marrow-ai:latest

# 4. Wait 30 seconds for startup
sleep 30

# 5. Check health
docker exec test-container python -c "import requests; print(requests.get('http://localhost:8501/_stcore/health').status_code)"

# 6. Test in browser
# Visit http://localhost:8501

# 7. Cleanup
docker stop test-container
docker rm test-container
```

**Checklist**:
- [ ] Docker build successful
- [ ] Image size acceptable (<2GB)
- [ ] Container starts without errors
- [ ] Health check returns 200
- [ ] Web interface accessible
- [ ] Test case processes successfully

---

#### **Step 3: Staging Deployment** (1 hour)

```bash
# 1. Deploy to staging environment
# (Use same procedure as production, but with staging configs)

# 2. Run smoke tests
python tests/smoke_test.py --env=staging

# 3. Load testing (optional)
# Use tools like Locust, JMeter, or k6
locust -f tests/load_test.py --host=https://staging.yourdomain.com

# 4. Monitor for 24-48 hours
# Check logs, memory usage, error rates
```

**Checklist**:
- [ ] Staging deployment successful
- [ ] Smoke tests pass
- [ ] Load tests meet requirements
- [ ] No memory leaks detected
- [ ] Error rate <1%
- [ ] Response times acceptable

---

### **Phase 2: Production Deployment**

#### **Step 4: Production Setup** (2 hours)

**Option A: Cloud Deployment (AWS/GCP/Azure)**

```bash
# AWS EC2 Example:
# 1. Launch instance (t3.large, 8GB RAM)
# 2. SSH into instance
# 3. Clone repository
# 4. Run deployment script
```

**Option B: Docker Compose**

```bash
# 1. Copy files to production server
scp -r . user@prod-server:/path/to/app/

# 2. SSH into server
ssh user@prod-server

# 3. Configure environment
cd /path/to/app/
cp .env.example .env.production
nano .env.production  # Add production keys

# 4. Deploy
docker-compose up -d

# 5. Verify
docker-compose ps
docker-compose logs -f
```

**Option C: Kubernetes**

```bash
# 1. Build and push image
docker build -t registry/bone-marrow-ai:v1.0 .
docker push registry/bone-marrow-ai:v1.0

# 2. Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 3. Check status
kubectl get pods
kubectl logs -f pod/bone-marrow-ai-xxxxx
```

**Checklist**:
- [ ] Production server provisioned
- [ ] SSL/TLS certificates installed
- [ ] DNS configured
- [ ] Firewall rules set
- [ ] Application deployed
- [ ] Health checks passing

---

#### **Step 5: Post-Deployment Verification** (30 minutes)

```bash
# 1. Health check
curl https://your-domain.com/_stcore/health

# 2. Smoke test
# Visit https://your-domain.com
# Upload test image
# Verify report generation

# 3. Check logs
docker-compose logs -f | grep -i error

# 4. Monitor metrics
# CPU usage
# Memory usage
# Response times
# Error rates
```

**Checklist**:
- [ ] Application accessible via domain
- [ ] HTTPS working (valid certificate)
- [ ] Health check returns 200
- [ ] Test case processes correctly
- [ ] Logs show no errors
- [ ] Performance metrics acceptable

---

#### **Step 6: Enable Monitoring** (30 minutes)

```bash
# 1. Configure monitoring alerts
# - Email/Slack notifications
# - Error rate threshold
# - Response time threshold
# - Resource usage threshold

# 2. Set up log aggregation
# - Centralized logging (e.g., ELK, Loki)
# - Log retention policy
# - Search and analysis tools

# 3. Configure uptime monitoring
# - Pingdom, UptimeRobot, or similar
# - Check interval: 5 minutes
# - Alert on downtime >2 minutes
```

**Checklist**:
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Uptime monitoring active
- [ ] Alert channels tested (email, Slack)

---

### **Phase 3: Go-Live**

#### **Step 7: Production Launch** (15 minutes)

```bash
# 1. Announce maintenance window (if updating existing system)
# 2. Final smoke test
# 3. Enable production traffic
# 4. Monitor closely for first hour
# 5. Be ready to rollback if issues occur
```

**Go/No-Go Checklist**:
- [ ] All tests passing
- [ ] Staging environment stable
- [ ] Team available for support
- [ ] Rollback plan ready
- [ ] Communication channels open
- [ ] Monitoring active

**Launch**:
- [ ] Traffic directed to production
- [ ] No errors in first 15 minutes
- [ ] Response times normal
- [ ] No user complaints
- [ ] Success metrics met

---

## 🔍 Post-Deployment Monitoring

### **First Hour**:
- [ ] Monitor error logs every 5 minutes
- [ ] Check health endpoint every minute
- [ ] Verify user traffic is flowing
- [ ] Response times within SLA
- [ ] No memory/CPU spikes

### **First Day**:
- [ ] Review all error logs
- [ ] Check resource utilization trends
- [ ] Verify backup ran successfully
- [ ] Review user feedback (if any)
- [ ] Update status page

### **First Week**:
- [ ] Daily log review
- [ ] Performance trend analysis
- [ ] Capacity planning review
- [ ] User satisfaction survey
- [ ] Incident retrospective (if any)

---

## 🚨 Rollback Procedure

**If critical issues occur, follow this rollback plan:**

### **Step 1: Assess Severity** (2 minutes)
- Is the issue blocking all users? → ROLLBACK IMMEDIATELY
- Is it affecting <10% of users? → Fix forward if possible
- Is it a minor bug? → Create ticket, fix in next release

### **Step 2: Execute Rollback** (5 minutes)

**Docker Compose**:
```bash
# 1. Stop current version
docker-compose down

# 2. Switch to previous version
docker tag bone-marrow-ai:v1.0 bone-marrow-ai:latest

# 3. Restart
docker-compose up -d

# 4. Verify
docker-compose ps
curl https://your-domain.com/_stcore/health
```

**Kubernetes**:
```bash
# Rollback to previous revision
kubectl rollout undo deployment/bone-marrow-ai

# Check status
kubectl rollout status deployment/bone-marrow-ai
```

### **Step 3: Communicate** (5 minutes)
- [ ] Notify users of the rollback
- [ ] Update status page
- [ ] Inform team members
- [ ] Document issue for post-mortem

### **Step 4: Post-Mortem** (1 hour)
- [ ] What happened?
- [ ] Why did it happen?
- [ ] How was it detected?
- [ ] How was it resolved?
- [ ] How do we prevent it in the future?

---

## 📊 Success Metrics

### **Technical Metrics**:
- **Uptime**: ≥99.5% (target: 99.9%)
- **Response Time**: <10 seconds per analysis (target: <5s)
- **Error Rate**: <1% (target: <0.1%)
- **CPU Usage**: <70% average
- **Memory Usage**: <80% average
- **Disk Usage**: <90%

### **Business Metrics**:
- **Daily Active Users**: Track growth
- **Cases Processed**: Track volume
- **User Satisfaction**: Survey score ≥4.0/5.0
- **Time Saved**: Compare to manual differential (45 min → 10 sec)

---

## 🔒 Security Review

### **Weekly**:
- [ ] Review access logs for suspicious activity
- [ ] Check for failed authentication attempts
- [ ] Verify API key is not exposed
- [ ] Review file upload patterns

### **Monthly**:
- [ ] Dependency vulnerability scan
- [ ] Penetration testing (if budget allows)
- [ ] Review and rotate API keys
- [ ] Update security patches

### **Quarterly**:
- [ ] Full security audit
- [ ] Review access control policies
- [ ] Update security documentation
- [ ] Team security training

---

## 📝 Sign-Off

### **Pre-Deployment Sign-Off**:

**Technical Lead**: _________________ Date: _______  
**Project Manager**: _________________ Date: _______  
**DevOps Engineer**: _________________ Date: _______  

### **Post-Deployment Sign-Off**:

**System Stable (24h)**: _________________ Date: _______  
**Monitoring Active**: _________________ Date: _______  
**Documentation Complete**: _________________ Date: _______  
**Team Trained**: _________________ Date: _______  

---

## 📞 Emergency Contacts

**Technical Support**:
- On-Call Engineer: [Phone/Email]
- Backup Engineer: [Phone/Email]

**Business**:
- Project Manager: [Phone/Email]
- Product Owner: [Phone/Email]

**External**:
- Groq Support: support@groq.com
- Hosting Provider: [Contact info]

---

**Checklist Version**: 1.0  
**Last Updated**: October 2026  
**Next Review**: January 2027
