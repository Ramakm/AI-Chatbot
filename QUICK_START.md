# 🚀 Quick Reference Guide

## Local Development

### First Time Setup
```bash
# macOS/Linux
bash start.sh

# Windows
start.bat
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
python app.py
```

Visit: http://localhost:8000

---

## Docker Quick Start

### Run with Docker
```bash
docker build -t ai-chatbot .
docker run -p 8000:8000 ai-chatbot
```

### Run with Docker Compose
```bash
docker-compose up
```

Visit: http://localhost:8000 (or http://localhost for Nginx)

---

## One-Click Cloud Deployments

### Heroku (2 minutes)
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

### Railway (3 clicks)
1. Go to https://railway.app
2. Connect GitHub repo
3. Deploy
4. Get URL from dashboard

### Render (3 clicks)
1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect GitHub and deploy
4. Get public URL

---

## API Usage

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"msg": "Hello!"}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## File Locations

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server |
| `templates/index.html` | Chat UI |
| `static/style.css` | Styling |
| `intents.json` | Chatbot responses |
| `chatbot_model.h5` | ML model |
| `requirements.txt` | Dependencies |
| `Dockerfile` | Container image |
| `docker-compose.yml` | Multi-container setup |

---

## Environment Variables

```bash
# Optional: Create .env file
WORKERS=4
LOG_LEVEL=info
CORS_ORIGINS=*
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| Port 8000 in use | Kill process: `kill $(lsof -t -i:8000)` |
| Missing model | Ensure `chatbot_model.h5` in root |
| NLTK errors | Run: `python -c "import nltk; nltk.download('punkt')"` |
| Static files 404 | Check `static/` directory exists |

---

## Performance Tuning

```bash
# Increase workers
gunicorn -w 8 -k uvicorn.workers.UvicornWorker app:app

# Production server with Gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

---

## Monitoring

```bash
# Docker logs
docker logs ai-chatbot

# Health check
curl http://localhost:8000/health

# Python logs (local)
python app.py --log-level debug
```

---

## Update/Redeploy

```bash
# Pull latest changes
git pull

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Restart service
docker-compose restart chatbot
```

---

## Advanced Features

### WebSocket (Future Enhancement)
```python
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Real-time messaging
```

### Rate Limiting
```bash
pip install slowapi
```

### Database Integration
```bash
pip install sqlalchemy postgresql
```

---

## Cost Estimation (Monthly)

| Platform | Free Tier | Paid Starting |
|----------|-----------|---------------|
| Heroku | ❌ (removed) | $7/month |
| Render | ✅ 750hrs/mo | $7/month |
| Railway | ✅ $5 credit/mo | Pay-as-you-go |
| AWS EC2 | ✅ 12 months | $3-10/month |

---

## Useful Links

- 📖 [FastAPI Docs](https://fastapi.tiangolo.com/)
- 🐳 [Docker Docs](https://docs.docker.com/)
- 🚀 [Railway Docs](https://docs.railway.app/)
- 🔧 [Render Docs](https://render.com/docs)
- 🔐 [AWS EC2 Guide](https://docs.aws.amazon.com/ec2/)

---

## Support

For detailed deployment guide: See `DEPLOYMENT.md`
For full documentation: See `README_UPDATED.md`
