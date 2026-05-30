@echo off
git pull origin main
git add .
git commit -m "Updated project" || echo No changes to commit
git push origin main
pause