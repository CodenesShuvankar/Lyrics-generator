# � AI Lyrics Generator - Deployment Guide

## 🌟 Recommended: Streamlit Community Cloud (FREE)

### Why This is Perfect for You:
- ✅ **Completely FREE**
- ✅ **Public URL** (e.g., `https://your-app.streamlit.app`)
- ✅ **Auto-deploys from GitHub**
- ✅ **HTTPS enabled by default**
- ✅ **Perfect for AI/ML apps**

## Quick Setup Steps:

### 1. Push Your Code to GitHub
```powershell
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit - AI Lyrics Generator"

# Create repository on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/lyrics-generator.git
git branch -M main
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/lyrics-generator`
5. Set main file path: `app.py`
6. Click "Deploy"

### 3. Your App Goes Live!
- URL: `https://YOUR_USERNAME-lyrics-generator-app-xyz123.streamlit.app`
- Auto-updates when you push to GitHub
- Supports up to 1GB RAM (sufficient for your model)

## Limitations:
- **Resource limits**: 1GB RAM, shared CPU
- **Sleep mode**: App sleeps after 7 days of inactivity
- **No custom domain** (unless you upgrade)

---

## Alternative: Hugging Face Spaces (Also FREE)

Perfect for ML apps with model hosting:

### Setup:
1. Create account at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space with Streamlit
3. Upload your code
4. Gets URL like: `https://huggingface.co/spaces/YOUR_USERNAME/lyrics-generator`

---

## If You Need More Resources: Azure App Service

More powerful than Streamlit Cloud but costs ~$10-15/month:

### Benefits:
- **2GB+ RAM**
- **Custom domain support**
- **Always-on** (no sleep mode)
- **Professional scaling**
