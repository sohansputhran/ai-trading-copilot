# Streamlit Cloud Deployment Guide

## 🚀 Deploy Your AI Trading Copilot in 5 Minutes

Streamlit Cloud is the **easiest and recommended** way to deploy this project. It's free, handles HTTPS automatically, and deploys directly from your GitHub repo.

---

## Prerequisites

- ✅ Code pushed to GitHub repository
- ✅ HuggingFace API token ([get one FREE here](https://huggingface.co/settings/tokens))
- ✅ Streamlit Cloud account ([sign up here](https://share.streamlit.io))

---

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

Ensure you have these files in your repo:

**`requirements.txt`** (required)
```txt
streamlit>=1.28.0
huggingface-hub>=0.22.0
langgraph>=0.0.20
langchain>=0.0.300
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
# ... other dependencies
```

**`.streamlit/config.toml`** (optional but recommended)
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

**`.streamlit/secrets.toml.example`** (template for users)
```toml
# Copy this to .streamlit/secrets.toml locally
# In Streamlit Cloud, add these via the UI

HUGGINGFACE_API_TOKEN = "your-key-here"
DATABASE_URL = "sqlite:///./trading.db"  # or PostgreSQL URL
```

**Add `.streamlit/secrets.toml` to `.gitignore`:**
```gitignore
.streamlit/secrets.toml
.env
*.db
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub** (grants Streamlit access to your repos)

3. **Click "New app"**

4. **Configure your app:**
   - **Repository**: `your-username/ai-trading-copilot`
   - **Branch**: `main` (or `master`)
   - **Main file path**: `app.py`
   - **App URL**: `your-app-name` (becomes `your-app-name.streamlit.app`)

5. **Advanced settings** → **Secrets**:
   ```toml
   HUGGINGFACE_API_TOKEN = "hf_your-actual-key-here"
   PAPER_TRADING = "true"
   MAX_POSITION_SIZE = "0.05"
   MAX_DAILY_LOSS = "0.02"
   ```

6. **Click "Deploy!"**

### Step 3: Wait for Deployment

Streamlit Cloud will:
- ✅ Clone your repository
- ✅ Install dependencies from `requirements.txt`
- ✅ Run your `app.py`
- ✅ Provide a public URL with HTTPS

This typically takes **2-3 minutes**.

### Step 4: Verify Deployment

Once deployed, your app will be live at:
```
https://your-app-name.streamlit.app
```

Test the deployment:
1. Navigate to the URL
2. Try the market scanner
3. Verify API key is working (no errors)
4. Test all pages/features

---

## Managing Your Deployment

### Update Your App

Streamlit Cloud **auto-deploys** when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "feat: add new feature"
git push origin main

# Streamlit Cloud automatically rebuilds and redeploys!
```

### View Logs

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click "Manage app" → "Logs"
4. View real-time application logs

### Update Secrets

1. Go to app settings
2. Click "Advanced settings" → "Secrets"
3. Update the TOML content
4. Click "Save"
5. App will automatically restart

### Reboot App

If your app becomes unresponsive:

1. Click "⋮" (three dots) on app dashboard
2. Select "Reboot app"
3. Wait 30-60 seconds

---

## Environment Variables & Secrets

### Required Secrets

Add these in Streamlit Cloud settings:

```toml
# Required
HUGGINGFACE_API_TOKEN = "hf_your-key-here"

# Optional (defaults work fine)
DATABASE_URL = "sqlite:///./trading.db"
PAPER_TRADING = "true"
LOG_LEVEL = "INFO"
MAX_POSITION_SIZE = "0.05"
MAX_DAILY_LOSS = "0.02"
MAX_OPEN_POSITIONS = "5"
```

### Access Secrets in Code

Streamlit makes secrets available via `st.secrets`:

```python
import streamlit as st
import os

# Access secrets
api_key = st.secrets.get("HUGGINGFACE_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

# Use fallback for local development
database_url = st.secrets.get("DATABASE_URL", "sqlite:///./trading.db")
```

---

## Database Considerations

### SQLite (Default - Simple)

**Pros:**
- ✅ No setup required
- ✅ Works out of the box
- ✅ Perfect for demo/portfolio

**Cons:**
- ⚠️ Data resets on reboot (Streamlit Cloud)
- ⚠️ Not suitable for production trading

**Recommendation:** Use SQLite for demo purposes. Data will persist between normal requests but may reset on app reboots.

### PostgreSQL (Production)

For persistent data across reboots:

1. **Create a free PostgreSQL database:**
   - [ElephantSQL](https://www.elephantsql.com/) (free tier)
   - [Supabase](https://supabase.com/) (free tier)
   - [Neon](https://neon.tech/) (free tier)

2. **Get connection string:**
   ```
   postgresql://user:password@host:5432/database
   ```

3. **Add to Streamlit secrets:**
   ```toml
   DATABASE_URL = "postgresql://user:password@host:5432/database"
   ```

4. **Update code to use it:**
   ```python
   from sqlalchemy import create_engine
   
   engine = create_engine(st.secrets["DATABASE_URL"])
   ```

---

## Custom Domain (Optional)

Streamlit Cloud supports custom domains on paid plans:

1. Upgrade to Streamlit Cloud Pro
2. Go to app settings → "Custom domain"
3. Add your domain (e.g., `trading.yourdomain.com`)
4. Configure DNS settings as shown
5. Wait for SSL certificate provisioning

**Free tier URL:** Always works: `https://your-app-name.streamlit.app`

---

## Resource Limits

### Streamlit Cloud Free Tier

- **Memory**: 1 GB RAM
- **CPU**: Shared
- **Storage**: Ephemeral (resets on reboot)
- **Apps**: Up to 3 public apps
- **Auto-sleep**: After 7 days of inactivity

**Optimization tips:**
- Use caching (`@st.cache_data`, `@st.cache_resource`)
- Lazy-load data
- Avoid large in-memory datasets
- Use pagination for large result sets

### Upgrade if Needed

If you hit limits:
- **Streamlit Cloud Pro**: More resources, private apps
- **Self-host with Docker**: Full control, unlimited resources

---

## Troubleshooting

### "App failed to load"

**Check logs:**
1. Go to app dashboard → Logs
2. Look for Python errors

**Common issues:**
- Missing dependency in `requirements.txt`
- Syntax error in code
- Missing API key in secrets

**Fix:**
```bash
# Update requirements.txt
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "fix: update dependencies"
git push
```

### "Import error: No module named X"

**Solution:** Add to `requirements.txt`:
```bash
# Check local environment
pip freeze | grep module-name

# Add to requirements.txt
echo "module-name==x.y.z" >> requirements.txt
git add requirements.txt && git commit -m "fix: add missing dependency" && git push
```

### "App is slow"

**Solutions:**
1. **Add caching:**
   ```python
   @st.cache_data(ttl=300)  # Cache for 5 minutes
   def fetch_market_data(symbol):
       return expensive_api_call(symbol)
   ```

2. **Reduce API calls:**
   - Cache results
   - Batch requests
   - Use pagination

3. **Optimize queries:**
   - Add database indexes
   - Limit result sizes
   - Use query optimization

### "Data disappeared after reboot"

**Expected behavior with SQLite** on Streamlit Cloud.

**Solutions:**
1. **Use PostgreSQL** for persistent data
2. **Export/import** data manually
3. **Accept ephemeral nature** for demo purposes

---

## Security Best Practices

### Secrets Management

✅ **DO:**
- Store API keys in Streamlit secrets
- Use `.gitignore` for local secrets
- Rotate keys periodically
- Use environment-specific keys

❌ **DON'T:**
- Commit secrets to Git
- Share secrets in screenshots
- Use production keys in demos
- Hardcode credentials

### Access Control

For public demos:
```python
# Add read-only mode
import streamlit as st

DEMO_MODE = st.secrets.get("DEMO_MODE", "true") == "true"

if DEMO_MODE:
    st.warning("⚠️ Demo mode: Paper trading only, data resets daily")
    # Disable real trading features
```

### API Rate Limiting

Protect your API keys:
```python
# Add request throttling
from time import sleep
from functools import lru_cache

@lru_cache(maxsize=100)
@st.cache_data(ttl=60)  # Cache for 1 minute
def throttled_api_call(symbol):
    sleep(0.1)  # 10 requests/second max
    return api.fetch(symbol)
```

---

## Monitoring Your App

### Built-in Analytics

Streamlit Cloud provides:
- Page views
- User activity
- Error rates
- Resource usage

Access via: App dashboard → "Analytics"

### Custom Logging

Add structured logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log important events
logger.info(f"Market scan completed: {len(results)} stocks found")
logger.error(f"API error: {error_message}")
```

View in Streamlit Cloud logs dashboard.

---

## Next Steps After Deployment

1. ✅ **Test thoroughly**
   - Try all features
   - Check error handling
   - Verify API integration

2. ✅ **Add to README**
   - Update demo link
   - Add "View Live Demo" button
   - Include screenshots

3. ✅ **Share it**
   - LinkedIn post with live link
   - GitHub README badge
   - Portfolio website

4. ✅ **Monitor it**
   - Check logs regularly
   - Watch for errors
   - Monitor API usage

---

## Example: Complete Streamlit Cloud Setup

### 1. Project Structure
```
ai-trading-copilot/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml          # App configuration
│   └── secrets.toml.example # Secrets template
├── .gitignore               # Ignore secrets
├── src/                     # Your source code
├── pages/                   # Streamlit pages
└── README.md               # With live demo link
```

### 2. requirements.txt
```txt
streamlit==1.31.0
huggingface-hub==0.22.0
langgraph==0.0.40
langchain==0.1.10
yfinance==0.2.37
pandas==2.2.0
numpy==1.26.4
sqlalchemy==2.0.27
psycopg2-binary==2.9.9
python-dotenv==1.0.1
```

### 3. Streamlit Secrets (in UI)
```toml
HUGGINGFACE_API_TOKEN = "hf_your-key-here"
PAPER_TRADING = "true"
DATABASE_URL = "sqlite:///./trading.db"
```

### 4. Deploy & Share
```markdown
<!-- In your README.md -->
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

[🚀 **View Live Demo**](https://your-app.streamlit.app)
```

---

## Conclusion

Streamlit Cloud is the **fastest and easiest** way to deploy your AI Trading Copilot:

✅ **5-minute deployment**  
✅ **Free tier available**  
✅ **Auto-deploys on git push**  
✅ **HTTPS included**  
✅ **Perfect for portfolios**  

**Deploy now:** [share.streamlit.io](https://share.streamlit.io)

---

**Questions?** Check the [Streamlit Cloud docs](https://docs.streamlit.io/streamlit-community-cloud) or reach out!
