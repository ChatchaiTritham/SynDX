# SynDX Quick Start Guide

Get up and running with SynDX in 5 minutes!

## ⚡ Fast Track (For Experienced Users)

```bash
# Clone and setup
git clone https://github.com/ChatchaiTritham/SynDX.git && cd SynDX
python -m venv venv && source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

# Generate datasets and run analysis
python scripts/generate_example_dataset.py # Quick test (2 min)
python scripts/generate_full_dataset_for_paper.py # Full dataset (10 min)
python scripts/generate_publication_figures.py # All figures (5 min)
python scripts/compute_table_c5_metrics.py # Validation (3 min)

# Done! Check outputs/ directory
```

**Total time:** ~20 minutes on standard laptop

---

## 📚 Step-by-Step Guide (For New Users)

### Step 1: Installation (2 minutes)

#### Option A: Using pip (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/ChatchaiTritham/SynDX.git
cd SynDX

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install SynDX
pip install -e .
```

#### Option B: Using Docker (Even Easier!)

```bash
# Build and run
docker-compose up

# Access Jupyter at: http://localhost:8888
```

### Step 2: Generate Synthetic Dataset (10 minutes)

```bash
# Generate full dataset for paper
python scripts/generate_full_dataset_for_paper.py

# This will create:
# - data/synthetic_patients_train.json (5,600 cases)
# - data/synthetic_patients_test.json (2,800 cases)
# - data/urgency_dras5_mapping.json
# - outputs/dataset_summary.csv
```

**What it does:** Creates synthetic patient cases for dizziness/vertigo with realistic clinical features, urgency tiers (0-2), and DRAS-5 states (G1-G5).

### Step 3: Generate Publication Figures (5 minutes)

```bash
# Generate all publication-quality figures (300 DPI)
python scripts/generate_publication_figures.py

# This will create in outputs/figures/:
# - figure_3x_transformation_pipeline.{pdf,png,svg}
# - figure_mapping_matrix.{pdf,png,svg}
# - figure_state_distribution.{pdf,png,svg}
# - figure_uncertainty_impact.{pdf,png,svg}
# - figure_validation_dashboard.{pdf,png,svg}
# - figure_context_escalation_heatmap.{pdf,png,svg}
```

**What it does:** Generates publication-ready figures in PDF (vector), PNG (300 DPI), and SVG formats following IEEE/Nature standards.

### Step 4: Run Validation Metrics (3 minutes)

```bash
# Compute Table C.5 validation metrics
python scripts/compute_table_c5_metrics.py

# This will generate:
# - outputs/table_c5_validation_metrics.csv
# - outputs/validation_report.txt
```

**What it does:** Validates urgency-to-DRAS5 mapping against 8 safety criteria (emergency preservation, uncertainty override, etc.).

### Step 5: Run Automated Tests (5 minutes)

```bash
# Run full test suite
python automated_test.py

# This will test:
# - Data generation pipeline
# - Urgency-DRAS5 mapping logic
# - Safety constraint validation
# - Figure generation
```

**What it does:** Ensures all components work correctly and meet safety requirements.

---

## 🚀 Quick Usage Example

### Using SynDX in Python:

```python
from syndx.pipeline import SynDXPipeline
from syndx.urgency_dras5_transformer import UrgencyToDRAS5Transformer

# Initialize pipeline
pipeline = SynDXPipeline()

# Patient data (from clinical guidelines)
patient = {
 'patient_id': 'P001',
 'age': 72,
 'sex': 'F',
 'chief_complaint': 'acute_vertigo',
 'onset_type': 'sudden',
 'red_flags': ['new_focal_neurological_deficit'],
 'risk_factors': {
 'cardiovascular_disease': True,
 'hypertension': True,
 'diabetes': False
 },
 'vital_signs': {
 'blood_pressure_systolic': 170,
 'heart_rate': 95
 },
 'nystagmus_type': 'central',
 'data_completeness': 0.92
}

# Step 1: Get urgency tier (from TiTrATE clinical guidelines)
urgency_result = pipeline.get_urgency_tier(patient)
print(f"Urgency Tier: {urgency_result['urgency']}") # Output: 2 (Emergency)

# Step 2: Transform to DRAS-5 state
transformer = UrgencyToDRAS5Transformer()
dras5_result = transformer.transform(
 urgency=urgency_result['urgency'],
 uncertainty=urgency_result['uncertainty'],
 patient_context=patient
)

print(f"DRAS-5 State: {dras5_result['dras5_state']}")
print(f"Confidence: {dras5_result['confidence']:.3f}")
print(f"Explanation: {dras5_result['explanation']}")
```

**Output:**
```
Urgency Tier: 2 (Emergency)
DRAS-5 State: G4 (Emergency Escalation)
Confidence: 0.963
Explanation: Emergency escalation due to:
 • Urgency tier 2 (emergency)
 • Red flag detected: new_focal_neurological_deficit
 • Central nystagmus (high stroke risk)
 • Age > 65 with cardiovascular disease
→ Immediate emergency department evaluation required
```

---

## 📊 Expected Results

After running all scripts, you should see:

### Performance Metrics (Synthetic Data)
- ✅ Emergency Preservation: **100%** (All emergency cases → G4)
- ✅ Uncertainty Override: **98.7%** (High uncertainty → G5)
- ✅ Safe Escalation: **96.3%** (No inappropriate downgrades)
- ✅ Critical Flag Escalation: **100%** (Red flags properly escalated)
- ✅ Monotonic Escalation: **94.8%** (Context only increases risk)
- ✅ State Validity: **100%** (All states in {G1, G2, G3, G4, G5})
- ⚡ Transformation Time: **0.45 ms** (Real-time capable)

### Generated Files
```
data/
 ├── synthetic_patients_train.json (5,600 cases, ~12 MB)
 ├── synthetic_patients_test.json (2,800 cases, ~6 MB)
 └── urgency_dras5_mapping.json

outputs/
 ├── dataset_summary.csv
 ├── table_c5_validation_metrics.csv
 ├── validation_report.txt
 └── figures/
 ├── pdf/ (vector format - for journal submission)
 ├── png/ (300 DPI - for presentations, thesis)
 └── svg/ (vector format - for editing)

notebooks/
 ├── 01_data_exploration.ipynb
 ├── 02_urgency_analysis.ipynb
 └── 03_dras5_validation.ipynb
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: `ModuleNotFoundError: No module named 'syndx'`
```bash
# Solution: Make sure you installed SynDX package
pip install -e .
```

#### Issue 2: Missing output directories
```bash
# Solution: Create directories manually
mkdir -p outputs/figures/{pdf,png,svg}
mkdir -p data models
```

#### Issue 3: Matplotlib backend errors
```python
# Add to top of script
import matplotlib
matplotlib.use('Agg') # For non-interactive backend
```

#### Issue 4: Out of memory
```python
# Reduce dataset size in scripts
n_train = 1000 # Instead of 5,600
n_test = 500 # Instead of 2,800
```

---

## 📖 Next Steps

### For Researchers:
1. Read the full paper: [Paper Title](link-to-paper)
2. Explore [docs/](docs/) for detailed documentation
3. Check [validation_system/](validation_system/) for safety validation
4. Customize urgency-DRAS5 mapping in `syndx/urgency_dras5_transformer.py`

### For Developers:
1. Read [CONTRIBUTING.md](CONTRIBUTING.md) (if available)
2. Run tests: `pytest tests/`
3. Check code style: `black syndx/ scripts/`
4. Submit pull requests on GitHub

### For Clinicians:
1. Review safety guarantees in [README.md](README.md)
2. Understand limitations section
3. Validate against your institution's clinical guidelines
4. Check regulatory requirements (FDA, local authorities)

---

## ⚠️ Important Reminders

🚨 **NOT FOR CLINICAL USE** - This is research software only
- Requires IRB approval for retrospective studies
- Requires FDA/regulatory clearance for prospective use
- Always maintain human clinical oversight
- Uncertainty threshold (u > 0.7) triggers system abstention (G5)

🔬 **Research Context:**
- Part of PhD dissertation on clinical decision support for emergency triage
- Integrates TiTrATE clinical guidelines with SRGL risk gates
- Novel contribution: G5 (System Abstention) state for high uncertainty

---

## 🆘 Getting Help

- 📧 Email: chatchai.t@example.edu
- 🐛 Issues: https://github.com/ChatchaiTritham/SynDX/issues
- 💬 Discussions: https://github.com/ChatchaiTritham/SynDX/discussions
- 📚 Full Documentation: [README.md](README.md)

---

## 🎯 Success Checklist

- [ ] Environment set up (venv/docker)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data generated (5,600 train + 2,800 test cases)
- [ ] Publication figures generated (6 main figures, 3 formats each)
- [ ] Validation metrics computed (Table C.5)
- [ ] All tests passed (`python automated_test.py`)
- [ ] Figures match expected results (~96%+ validation rates)
- [ ] Understood safety disclaimers and limitations
- [ ] Ready to explore, validate, or extend!

---

## 🔗 Related Projects

- **SURgul**: Safety-first Urgency Risk Gating with Uncertainty Logic
 - Repository: https://github.com/ChatchaiTritham/SURgul
 - Implements SRGL gates (G1-G3) with uncertainty-driven abstention

- **ORASR**: Optimized Risk-Aware State Router
 - Maps DRAS-5 states to clinical care pathways

---

**Enjoy exploring SynDX! Transforming urgency to actionable risk states. Every decision traceable. Every patient protected.** 🛡️

**Citation:**
```bibtex
@software{syndx2026,
 author = {Tritham, Chatchai},
 title = {SynDX: Synthetic Diagnosis and Urgency-to-DRAS5 Transformation Pipeline},
 year = {2026},
 url = {https://github.com/ChatchaiTritham/SynDX},
 note = {Part of PhD dissertation on clinical decision support for emergency triage}
}
```
