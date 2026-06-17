@echo off
cd /d "%~dp0"
echo Running the full Dry Bean ML pipeline...
python main.py --mode all
pause

