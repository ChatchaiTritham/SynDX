@echo off 
REM Startup script for SynDX-Hybrid Framework 
  
echo Starting SynDX-Hybrid Framework Demo...  
echo.  
call syn_dx_env\Scripts\activate.bat  
cd /d "%%~dp0"  
echo Running implementation verification...  
python verify_implementation.py  
echo.  
echo To run the Jupyter notebook:  
echo 1. Make sure you have activated the virtual environment  
echo 2. Run: jupyter notebook notebooks/syn_dx_hybrid_implementation_demo.ipynb  
echo.  
pause 
