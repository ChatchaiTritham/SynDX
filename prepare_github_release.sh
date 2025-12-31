#!/bin/bash

# SynDX GitHub Release Preparation Script
# Prepares repository for initial publication

echo "==============================================="
echo "SynDX GitHub Release Preparation"
echo "==============================================="
echo ""
echo "⚠️  IMPORTANT: This is preliminary work without clinical validation"
echo ""

# Check if we're in SynDX directory
if [ ! -f "setup.py" ]; then
    echo "Error: Must run from SynDX root directory"
    exit 1
fi

# Step 1: Initialize git repository
echo "Step 1: Initializing Git repository..."
git init

# Step 2: Add all files
echo "Step 2: Adding files to Git..."
git add .

# Step 3: Create .gitattributes for better diffs
echo "Step 3: Creating .gitattributes..."
cat > .gitattributes << EOF
# Python
*.py diff=python

# Jupyter Notebooks
*.ipynb diff=jupyternotebook

# Documentation
*.md diff=markdown

# Data files (treat as binary)
*.csv binary
*.json binary
*.parquet binary
EOF

git add .gitattributes

# Step 4: Initial commit
echo "Step 4: Creating initial commit..."
git commit -m "Initial commit: SynDX v0.1.0

- Phase 1: Clinical knowledge extraction (TiTrATE/Bárány)
- Phase 2: XAI-driven synthesis (NMF implemented, VAE/SHAP/CF placeholders)
- Phase 3: Multi-level validation (statistical metrics implemented)
- Example dataset: 500 archetypes, 1000 synthetic patients
- Docker deployment ready
- Comprehensive documentation

⚠️  Preliminary work without clinical validation
NOT for clinical use - research purposes only"

# Step 5: Create annotated tag
echo "Step 5: Creating release tag v0.1.0..."
git tag -a v0.1.0 -m "v0.1.0 - Initial Release (Preliminary Work)

This release includes:
- Core SynDX framework with Phase 1-3 modules
- NMF latent archetype extraction
- TiTrATE constraint validation
- Example synthetic dataset generation
- Docker deployment support
- Comprehensive documentation

Pending implementation:
- VAE training loop
- SHAP reweighting
- Counterfactual validation
- Differential privacy
- FHIR export

⚠️  IMPORTANT NOTICE:
This is preliminary work without clinical validation.
All validation uses synthetic data only.
Do NOT use for clinical decision-making.

For details, see DEPLOYMENT_GUIDE.md and PROJECT_SUMMARY.md"

# Step 6: Display next steps
echo ""
echo "==============================================="
echo "✅ Git repository prepared successfully!"
echo "==============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create GitHub repository:"
echo "   Go to: https://github.com/new"
echo "   Repository name: SynDX"
echo "   Description: Explainable AI-Driven Synthetic Data Generation (Preliminary)"
echo "   Public repository"
echo "   Don't initialize with README"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/chatchai.tritham/SynDX.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo "   git push origin v0.1.0"
echo ""
echo "3. Create GitHub Release:"
echo "   - Go to repository → Releases → Draft a new release"
echo "   - Choose tag: v0.1.0"
echo "   - Title: v0.1.0 - Initial Release (Preliminary Work)"
echo "   - Description: Copy from CHANGELOG.md"
echo "   - Attach: outputs/synthetic_patients/*.csv (zip first)"
echo "   - Publish release"
echo ""
echo "4. Get DOI from Zenodo:"
echo "   - Go to: https://zenodo.org"
echo "   - Link GitHub repository"
echo "   - Enable Zenodo integration"
echo "   - Upload release"
echo "   - Get DOI"
echo ""
echo "5. Update badges:"
echo "   - Edit README.md"
echo "   - Replace XXXXXXX with actual DOI"
echo "   - Commit and push"
echo ""
echo "6. Verify repository:"
echo "   - Test: git clone https://github.com/chatchai.tritham/SynDX.git"
echo "   - Test: pip install -e ."
echo "   - Test: python scripts/generate_example_dataset.py"
echo ""
echo "==============================================="
echo "Repository is ready for publication!"
echo "⚠️  Remember to add disclaimer in all communications"
echo "==============================================="
