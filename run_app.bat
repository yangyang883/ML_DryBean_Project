@echo off
cd /d "%~dp0"
echo Starting Dry Bean Streamlit UI...
echo If the browser does not open automatically, visit:
echo http://127.0.0.1:8502
python -m streamlit run app\streamlit_app.py --server.address=127.0.0.1 --server.port=8502 --server.enableCORS=false --server.enableXsrfProtection=false
pause
