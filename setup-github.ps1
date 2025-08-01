# Quick GitHub Setup for Streamlit Cloud Deployment

Write-Host "🚀 Setting up GitHub repository for Streamlit Cloud deployment..." -ForegroundColor Green

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "📁 Initializing Git repository..." -ForegroundColor Yellow
    git init
}

# Add all files
Write-Host "📝 Adding files to git..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
git commit -m "Prepare for Streamlit Cloud deployment"

Write-Host ""
Write-Host "🌟 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Create a GitHub repository at: https://github.com/new" -ForegroundColor White
Write-Host "2. Copy the repository URL (e.g., https://github.com/USERNAME/lyrics-generator.git)" -ForegroundColor White
Write-Host "3. Run these commands with YOUR repository URL:" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/USERNAME/lyrics-generator.git" -ForegroundColor Gray
Write-Host "   git branch -M main" -ForegroundColor Gray
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Go to https://share.streamlit.io and deploy!" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Your app will be live at: https://USERNAME-lyrics-generator-app-xyz123.streamlit.app" -ForegroundColor Green
