# AI Chatbot - Deployment Guide

## Overview
This modern AI Chatbot has been upgraded to use **FastAPI** for better performance, scalability, and deployment flexibility.

## Local Setup & Running

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone/Navigate to project directory**
```bash
cd /path/to/AI-Chatbot
```

2. **Create a virtual environment (recommended)**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data (required for chatbot)**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

5. **Run the application**
```bash
python app.py
```

The chatbot will be available at: **http://localhost:8000**

---

## Deployment Options

### Option 1: Heroku Deployment (Free Tier)

1. **Install Heroku CLI** from https://devcenter.heroku.com/articles/heroku-cli

2. **Create Heroku app**
```bash
heroku login
heroku create your-chatbot-app-name
```

3. **Create Procfile** (in project root)
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

4. **Create runtime.txt** (in project root)
```
python-3.11.0
```

5. **Deploy**
```bash
git add .
git commit -m "Deploy chatbot"
git push heroku main
```

6. **View logs**
```bash
heroku logs --tail
```

Your app will be live at: `https://your-chatbot-app-name.herokuapp.com`

---

### Option 2: Render.com Deployment (Recommended)

1. **Sign up at** https://render.com

2. **Create Render configuration file** `.render/render.yaml`
```yaml
services:
  - type: web
    name: ai-chatbot
    runtime: python
    runtimeVersion: 3.11.0
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

3. **Push to GitHub** (Render integrates with GitHub)

4. **In Render Dashboard:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository
   - Configure:
     - **Name:** ai-chatbot
     - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
     - **Python Version:** 3.11

5. **Deploy** and get your public URL

---

### Option 3: Railway.app Deployment

1. **Sign up at** https://railway.app

2. **Install Railway CLI**
```bash
npm i -g @railway/cli
```

3. **Login and deploy**
```bash
railway login
railway init
railway up
```

4. **Your app will be deployed automatically**

---

### Option 4: AWS EC2 Deployment (Production)

1. **Launch EC2 instance** (Ubuntu 20.04 LTS recommended)

2. **SSH into instance**
```bash
ssh -i your-key.pem ec2-user@your-instance-ip
```

3. **Install dependencies**
```bash
sudo apt update
sudo apt install python3-pip python3-venv git
```

4. **Clone repository**
```bash
git clone your-repo-url
cd AI-Chatbot
```

5. **Setup virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

6. **Install and configure Gunicorn**
```bash
pip install gunicorn
```

7. **Create systemd service file** `/etc/systemd/system/chatbot.service`
```
[Unit]
Description=AI Chatbot FastAPI Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AI-Chatbot
ExecStart=/home/ubuntu/AI-Chatbot/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

8. **Start service**
```bash
sudo systemctl daemon-reload
sudo systemctl start chatbot
sudo systemctl enable chatbot
```

9. **Configure Nginx as reverse proxy** (optional but recommended)
```bash
sudo apt install nginx
```

Create `/etc/nginx/sites-available/chatbot`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

### Option 5: Docker Deployment

1. **Create Dockerfile** in project root
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Create .dockerignore**
```
venv
__pycache__
*.pyc
.git
.gitignore
README.md
```

3. **Build image**
```bash
docker build -t ai-chatbot:latest .
```

4. **Run container locally**
```bash
docker run -p 8000:8000 ai-chatbot:latest
```

5. **Deploy to Docker Hub or container registry**
```bash
docker login
docker tag ai-chatbot:latest yourname/ai-chatbot:latest
docker push yourname/ai-chatbot:latest
```

---

## Environment Variables (Optional)

Create a `.env` file for production settings:
```
WORKERS=4
LOG_LEVEL=info
CORS_ORIGINS=*
```

---

## API Endpoints

### Send Message
```
POST /api/chat
Content-Type: application/json

{
    "msg": "your message here"
}

Response:
{
    "response": "chatbot response",
    "confidence": 0.95
}
```

### Health Check
```
GET /health

Response:
{
    "status": "healthy",
    "model": "loaded"
}
```

### Web Interface
```
GET /
Returns: HTML interface
```

---

## Performance Tips

1. **Use Gunicorn for production** instead of Uvicorn development server
2. **Run multiple workers**: `gunicorn -w 4 app:app`
3. **Enable gzip compression** in reverse proxy
4. **Use CDN** for static files (CSS, JS)
5. **Monitor logs** for errors and performance issues
6. **Scale horizontally** by deploying multiple instances behind load balancer

---

## Monitoring & Maintenance

### View Logs (Heroku)
```bash
heroku logs --tail
```

### Monitor Performance (Railway/Render)
Check dashboard for memory usage, CPU, and request metrics

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

---

## Troubleshooting

**Port already in use:**
```bash
kill $(lsof -t -i:8000)
```

**Model loading errors:**
- Ensure `chatbot_model.h5` is in project root
- Ensure `words.pkl` and `classes.pkl` exist

**NLTK data missing:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

**Static files not loading:**
- Check that `static/` directory exists
- Verify file paths in `index.html`

---

## Support & Resources

- FastAPI Docs: https://fastapi.tiangolo.com/
- Render Deployment: https://render.com/docs
- Railway Deployment: https://docs.railway.app/
- AWS EC2: https://docs.aws.amazon.com/ec2/
