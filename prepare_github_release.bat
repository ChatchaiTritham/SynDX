@echo off
REM SynDX GitHub Release Preparation Script (Windows)
REM Prepares repository for initial publication

echo ===============================================
echo SynDX GitHub Release Preparation
echo ===============================================
echo.
echo WARNING: This is preliminary work without clinical validation
echo.

REM Check if we're in SynDX directory
if not exist "setup.py" (
    echo Error: Must run from SynDX root directory
    exit /b 1
)

REM Step 1: Initialize git repository
echo Step 1: Initializing Git repository...
git init

REM Step 2: Add all files
echo Step 2: Adding files to Git...
git add .

REM Step 3: Create .gitattributes
echo Step 3: Creating .gitattributes...
(
echo # Python
echo *.py diff=python
echo.
echo # Jupyter Notebooks
echo *.ipynb diff=jupyternotebook
echo.
echo # Documentation
echo *.md diff=markdown
echo.
echo # Data files ^(treat as binary^)
echo *.csv binary
echo *.json binary
echo *.parquet binary
) > .gitattributes

git add .gitattributes

REM Step 4: Initial commit
echo Step 4: Creating initial commit...
git commit -m "Initial commit: SynDX v0.1.0" -m "" -m "- Phase 1: Clinical knowledge extraction (TiTrATE/Barany)" -m "- Phase 2: XAI-driven synthesis (NMF implemented, VAE/SHAP/CF placeholders)" -m "- Phase 3: Multi-level validation (statistical metrics implemented)" -m "- Example dataset: 500 archetypes, 1000 synthetic patients" -m "- Docker deployment ready" -m "- Comprehensive documentation" -m "" -m "WARNING: Preliminary work without clinical validation" -m "NOT for clinical use - research purposes only"

REM Step 5: Create annotated tag
echo Step 5: Creating release tag v0.1.0...
git tag -a v0.1.0 -m "v0.1.0 - Initial Release (Preliminary Work)"

REM Display next steps
echo.
echo ===============================================
echo Git repository prepared successfully!
echo ===============================================
echo.
echo Next steps:
echo.
echo 1. Create GitHub repository:
echo    Go to: https://github.com/new
echo    Repository name: SynDX
echo    Description: Explainable AI-Driven Synthetic Data Generation (Preliminary)
echo    Public repository
echo    Don't initialize with README
echo.
echo 2. Add remote and push:
echo    git remote add origin https://github.com/chatchai.tritham/SynDX.git
echo    git branch -M main
echo    git push -u origin main
echo    git push origin v0.1.0
echo.
echo 3. Create GitHub Release
echo 4. Get DOI from Zenodo
echo 5. Update README.md with DOI
echo.
echo ===============================================
echo Repository is ready for publication!
echo WARNING: Remember disclaimer in all communications
echo ===============================================
echo.
pause
