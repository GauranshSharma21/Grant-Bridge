@echo off
echo ============================================
echo  GrantBridge - AI Grant Finder for Startups
echo  IBM Granite powered by WatsonX AI
echo ============================================
echo.
echo Starting GrantBridge on http://localhost:8501
echo.
echo Before starting:
echo  1. Copy env.example to .env
echo  2. Add your IBM_API_KEY and IBM_PROJECT_ID
echo  3. Run: pip install -r requirements.txt
echo.
python -m streamlit run app.py
