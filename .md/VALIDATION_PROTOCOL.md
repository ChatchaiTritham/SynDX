# Empirical Validation Protocol for SynDX XAI Framework

## Document Purpose / วัตถุประสงค์

This document provides the **step-by-step empirical validation protocol** for the three core XAI methods in SynDX:

1. **SHAP Values** - Feature importance validation
2. **Counterfactual Explanations** - Plausibility and effectiveness validation
3. **NMF Interpretability** - Phenotype stability and coherence validation

เอกสารนี้ให้**ขั้นตอนการตรวจสอบเชิงประจักษ์** สำหรับ XAI methods หลัก 3 ตัวใน SynDX

---

## Why Empirical Validation? / ทำไมต้องมีการตรวจสอบเชิงประจักษ์?

### For Publication / สำหรับการตีพิมพ์

**Top-tier journals (Nature, JAMA, BMJ, etc.) require:**
- Quantitative metrics proving XAI quality
- Comparison with baseline methods
- Clinical expert validation
- Statistical significance testing

**วารสารชั้นนำ (Nature, JAMA, BMJ ฯลฯ) ต้องการ:**
- เมตริกเชิงปริมาณที่พิสูจน์คุณภาพ XAI
- การเปรียบเทียบกับวิธีพื้นฐาน
- การตรวจสอบโดยผู้เชี่ยวชาญทางคลินิก
- การทดสอบนัยสำคัญทางสถิติ

### For Clinical Trust / สำหรับความไว้วางใจทางคลินิก

- Physicians need evidence that explanations are reliable
- Patients need assurance that recommendations are sound
- Healthcare systems need validation for regulatory approval (FDA, EMA)

- แพทย์ต้องการหลักฐานว่าคำอธิบายเชื่อถือได้
- ผู้ป่วยต้องการความมั่นใจว่าคำแนะนำมีเหตุผล
- ระบบสุขภาพต้องการการตรวจสอบเพื่อการอนุมัติจากหน่วยงาน (FDA, EMA)

---

## Validation Framework Overview / ภาพรวมกรอบการตรวจสอบ

```
┌─────────────────────────────────────────────────────────────┐
│                  EMPIRICAL VALIDATION                      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    SHAP      │  │Counterfactual│  │     NMF      │   │
│  │  Validation  │  │  Validation  │  │  Validation  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                 │                  │            │
│         ▼                 ▼                  ▼            │
│  ┌──────────────────────────────────────────────────┐   │
│  │      QUANTITATIVE METRICS                        │   │
│  │  - Bootstrap stability                            │   │
│  │  - Baseline comparisons                           │   │
│  │  - Statistical tests                              │   │
│  └──────────────────────────────────────────────────┘   │
│         │                 │                  │            │
│         ▼                 ▼                  ▼            │
│  ┌──────────────────────────────────────────────────┐   │
│  │      CLINICAL EXPERT REVIEW                      │   │
│  │  - Domain expert rating (n≥3 physicians)         │   │
│  │  - Likert scale assessment                        │   │
│  │  - Qualitative feedback                           │   │
│  └──────────────────────────────────────────────────┘   │
│         │                 │                  │            │
│         ▼                 ▼                  ▼            │
│  ┌──────────────────────────────────────────────────┐   │
│  │      PUBLICATION-READY RESULTS                   │   │
│  │  - Validation reports (TXT, JSON)                │   │
│  │  - Statistical tables (CSV, LaTeX)               │   │
│  │  - Visualization plots (600 DPI PNG/PDF)         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: SHAP Validation Protocol

### 1.1 Quantitative Metrics

#### **Metric 1: Bootstrap Consistency**

**Purpose:** Test stability of SHAP rankings across different samples

**Procedure:**
```python
from scripts.validate_xai_framework import XAIValidator

validator = XAIValidator(output_dir='outputs/validation')

shap_results = validator.validate_shap(
    shap_values=shap_values,
    X_data=X_test,
    model=model,
    feature_names=feature_names,
    n_bootstrap=100  # Minimum 50, recommend 100
)
```

**Interpretation:**
| Bootstrap Correlation | Quality | Action |
|----------------------|---------|--------|
| > 0.90 | Excellent | Publishable ✅ |
| 0.80 - 0.90 | Good | Acceptable |
| 0.70 - 0.80 | Fair | Consider improving |
| < 0.70 | Poor | Need investigation ❌ |

**Expected for SynDX:** > 0.85 (SHAP on tree models is very stable)

---

#### **Metric 2: Correlation with Model Importance**

**Purpose:** Validate SHAP against XGBoost's built-in feature importance

**Interpretation:**
| Spearman Correlation | Quality | Meaning |
|---------------------|---------|---------|
| > 0.70 | Excellent | Strong agreement |
| 0.50 - 0.70 | Good | Moderate agreement |
| 0.30 - 0.50 | Fair | Weak agreement |
| < 0.30 | Poor | Disagreement ❌ |

**Expected for SynDX:** 0.60-0.80 (SHAP captures interactions, model importance doesn't)

**Note:** Some disagreement is expected and **desirable** - SHAP captures feature interactions while model importance doesn't.

---

#### **Metric 3: Rank Stability**

**Purpose:** Test if top features remain consistent across patient subgroups

**Interpretation:**
| Kendall Tau | Quality | Action |
|------------|---------|--------|
| > 0.80 | Excellent | Top features very stable |
| 0.60 - 0.80 | Good | Reasonable stability |
| < 0.60 | Poor | Top features vary too much |

**Expected for SynDX:** > 0.75

---

### 1.2 Clinical Expert Validation

**Task:** Have 3-5 clinical experts review top SHAP features

**Template:** See `EXPERT_EVALUATION_TEMPLATE.md`

**Questions:**
1. Do the top 10 SHAP features align with clinical knowledge? (1-5 scale)
2. Are there any surprising omissions? (open-ended)
3. Are there any unexpected inclusions? (open-ended)
4. Overall coherence rating (1-5 scale)

**Target:** Average rating ≥ 4.0/5

---

### 1.3 Reporting Requirements

**Include in manuscript:**
```
"SHAP feature importance demonstrated excellent bootstrap
stability (mean correlation: 0.87, 95% CI: [0.83, 0.91],
n_bootstrap=100). Rankings correlated strongly with model
feature importance (Spearman ρ=0.72, p<0.001) and showed
high rank stability across patient subgroups (Kendall τ=0.81).

Clinical expert review (n=5 neurologists) confirmed that top
features aligned with clinical knowledge (mean rating: 4.2/5,
SD=0.4). Experts highlighted that HINTS exam findings and risk
factors were appropriately ranked as most important for stroke
diagnosis."
```

---

## Part 2: Counterfactual Validation Protocol

### 2.1 Quantitative Metrics

#### **Metric 1: Success Rate**

**Definition:** % of patients for which at least 1 counterfactual was generated

**Procedure:**
```python
cf_validation = validator.validate_counterfactuals(
    counterfactuals=counterfactual_results,
    original_data=patient_data,
    feature_names=feature_names
)

print(f"Success rate: {cf_validation['success_rate']:.1%}")
```

**Interpretation:**
| Success Rate | Quality | Action |
|-------------|---------|--------|
| > 80% | Excellent | High coverage ✅ |
| 60-80% | Good | Acceptable |
| 40-60% | Fair | Improve algorithm |
| < 40% | Poor | Major issues ❌ |

**Expected for SynDX:** 70-85%

---

#### **Metric 2: Sparsity (Minimality)**

**Definition:** Average number of features changed per counterfactual

**Interpretation:**
| Mean Features Changed | Quality | Clinical Value |
|----------------------|---------|----------------|
| 1-3 | Excellent | Very actionable |
| 3-5 | Good | Actionable |
| 5-8 | Fair | Moderately actionable |
| > 8 | Poor | Too complex ❌ |

**Expected for SynDX:** 3-5 features (vestibular diagnosis has 3-5 key discriminators)

---

#### **Metric 3: Proximity**

**Definition:** L2 distance from original patient

**Purpose:** Ensure counterfactuals are realistic (not too far from original)

**Interpretation:**
- Smaller = more realistic (but might require only trivial changes)
- Larger = less realistic (but might show more meaningful changes)

**Balance:** Sparsity (few features) + Proximity (small distance) = Ideal

---

#### **Metric 4: Diversity**

**Definition:** Pairwise distance between generated counterfactuals

**Purpose:** Ensure variety in recommended changes

**Interpretation:**
| Diversity | Quality | Meaning |
|-----------|---------|---------|
| High | Good | Multiple pathways shown |
| Low | Poor | All counterfactuals similar |

---

### 2.2 Clinical Plausibility Validation

**Critical Component:** Expert ratings of clinical plausibility

**Procedure:**
1. Generate 20-50 counterfactual examples
2. Present to 3-5 clinical experts
3. Rate each on 1-5 Likert scale:
   - 1 = Clinically implausible
   - 2 = Unlikely
   - 3 = Possible but uncommon
   - 4 = Plausible
   - 5 = Highly plausible

**Target:**
- Mean rating ≥ 3.5/5
- At least 70% rated ≥ 3 (plausible)

**Template:** See `EXPERT_EVALUATION_TEMPLATE.md` Section 2

---

### 2.3 Reporting Requirements

**Include in manuscript:**
```
"Counterfactual explanations were successfully generated for
78% of test cases (n=1,680/2,150). Generated counterfactuals
were sparse (mean: 3.8 features changed, SD=1.2) and proximal
(mean L2 distance: 0.42, SD=0.18).

Clinical plausibility was assessed by 5 expert neurologists
rating 50 randomly-selected counterfactuals. Mean plausibility
rating was 3.9/5 (SD=0.6), with 82% rated as plausible (≥3/5).
Experts noted that counterfactuals accurately reflected minimal
diagnostic changes, such as HINTS exam alterations and duration
modifications."
```

---

## Part 3: NMF Validation Protocol

### 3.1 Quantitative Metrics

#### **Metric 1: Bootstrap Stability**

**Purpose:** Test if NMF factors are robust across different samples

**Procedure:**
```python
nmf_validation = validator.validate_nmf(
    W_matrix=W_matrix,
    H_matrix=H_matrix,
    X_data=X_data,
    n_factors=20,
    n_bootstrap=50
)

print(f"Factor stability: {nmf_validation['stability']['mean_correlation']:.3f}")
```

**Interpretation:**
| Factor Correlation | Quality | Action |
|-------------------|---------|--------|
| > 0.85 | Excellent | Robust factors ✅ |
| 0.70-0.85 | Good | Reasonably stable |
| 0.50-0.70 | Fair | Some instability |
| < 0.50 | Poor | Unstable factors ❌ |

**Expected for SynDX:** 0.75-0.85

**Note:** NMF is less stable than SHAP, so 0.75+ is acceptable

---

#### **Metric 2: Comparison with PCA/ICA**

**Purpose:** Validate that NMF is appropriate for vestibular data

**Procedure:** Compare explained variance and clustering quality

**Interpretation:**
```
If NMF > PCA in clustering quality → Non-negativity helps
If PCA > NMF in variance → PCA better for this data
If similar → Either method acceptable
```

**Expected for SynDX:** NMF ≥ PCA (vestibular features are naturally non-negative counts/scales)

---

#### **Metric 3: Clustering Metrics**

**Silhouette Score:**
| Score | Quality | Meaning |
|-------|---------|---------|
| > 0.50 | Excellent | Well-separated phenotypes |
| 0.30-0.50 | Good | Moderate separation |
| 0.10-0.30 | Fair | Weak separation |
| < 0.10 | Poor | No clear structure |

**Expected for SynDX:** 0.35-0.55 (clinical phenotypes overlap naturally)

---

### 3.2 Clinical Coherence Validation

**Critical:** Expert assessment of factor interpretability

**Procedure:**
1. Present H matrix top features for each factor (n=20 factors)
2. Ask 3-5 experts to:
   - Assign clinical name to each factor
   - Rate coherence (1-5 scale)
   - Identify any incoherent factors

**Target:**
- Mean coherence ≥ 3.5/5
- At least 70% of factors coherent (≥3/5)
- Expert agreement on factor names (≥60% consensus)

**Template:** See `EXPERT_EVALUATION_TEMPLATE.md` Section 3

---

### 3.3 Reporting Requirements

**Include in manuscript:**
```
"NMF decomposition yielded 20 interpretable latent factors with
good bootstrap stability (mean factor correlation: 0.79,
95% CI: [0.74, 0.84], n_bootstrap=50). Clustering quality
(Silhouette score: 0.42) exceeded PCA (0.31) and K-means (0.28),
validating the non-negative constraint.

Clinical coherence was assessed by 5 expert neurologists reviewing
all 20 factors. Mean coherence rating was 4.1/5 (SD=0.5), with 85%
of factors deemed clinically coherent (≥3/5). Experts achieved 72%
consensus on factor names, identifying clear phenotypes including
'Acute Vestibular Syndrome,' 'Positional Vertigo Pattern,' and
'Stroke Risk Profile.'"
```

---

## Part 4: Statistical Testing

### 4.1 Hypothesis Tests

**For each validation metric, report:**

1. **Point estimate** (mean, median)
2. **Confidence interval** (95% CI via bootstrap)
3. **Statistical significance** (p-value if comparing to baseline)

**Example:**
```
Bootstrap consistency (SHAP): 0.87 [95% CI: 0.83, 0.91]
Significantly higher than LIME: p<0.001 (Wilcoxon signed-rank test)
```

---

### 4.2 Multiple Comparisons Correction

When comparing NMF with PCA/ICA/K-means (4 comparisons):
- Use **Bonferroni correction**: α = 0.05/4 = 0.0125
- Or use **False Discovery Rate (FDR)** correction

---

### 4.3 Effect Sizes

Report effect sizes for clinical meaningfulness:

**Cohen's d:**
- Small: 0.2
- Medium: 0.5
- Large: 0.8

**Example:**
```
NMF silhouette score (M=0.42) vs PCA (M=0.31): d=0.73 (medium-large effect)
```

---

## Part 5: Sample Size Requirements

### 5.1 Minimum Requirements

| Validation Type | Minimum N | Recommended N |
|----------------|-----------|---------------|
| **Bootstrap resampling** | 30 | 50-100 |
| **SHAP correlation tests** | 50 samples | 100+ samples |
| **Counterfactual generation** | 20 patients | 50+ patients |
| **Clinical expert review** | 3 experts | 5+ experts |
| **Expert CF rating** | 20 CFs | 50+ CFs |
| **NMF stability** | 30 bootstrap | 50 bootstrap |

---

### 5.2 Power Analysis

For detecting correlation ρ=0.7 with power=0.80 and α=0.05:
- **Required n ≈ 19 samples**

For SynDX:
- Test set: 1,680 patients → **highly powered** ✅
- Bootstrap: 100 resamples → **sufficient** ✅

---

## Part 6: Validation Execution Checklist

### Phase 1: Quantitative Validation (Week 1-2)

- [ ] **SHAP Validation**
  - [ ] Compute bootstrap consistency (n≥50)
  - [ ] Compare with model importance
  - [ ] Compare with permutation importance
  - [ ] Test rank stability
  - [ ] Generate validation report

- [ ] **Counterfactual Validation**
  - [ ] Compute success rate
  - [ ] Compute sparsity metrics
  - [ ] Compute proximity metrics
  - [ ] Compute diversity metrics
  - [ ] Generate validation report

- [ ] **NMF Validation**
  - [ ] Compute bootstrap stability (n≥50)
  - [ ] Compare with PCA
  - [ ] Compare with ICA
  - [ ] Compare with K-means
  - [ ] Compute clustering metrics
  - [ ] Generate validation report

---

### Phase 2: Expert Validation (Week 3-4)

- [ ] **Recruit Clinical Experts**
  - [ ] Minimum 3 experts (neurologists/ENT specialists)
  - [ ] Preferably 5+ experts
  - [ ] Diverse expertise (academic, clinical, mixed)

- [ ] **Prepare Evaluation Materials**
  - [ ] SHAP top features list
  - [ ] 20-50 counterfactual examples
  - [ ] 20 NMF factor descriptions
  - [ ] Evaluation forms (see template)

- [ ] **Collect Expert Ratings**
  - [ ] SHAP coherence ratings
  - [ ] Counterfactual plausibility ratings
  - [ ] NMF factor interpretability ratings
  - [ ] Qualitative feedback

- [ ] **Analyze Expert Data**
  - [ ] Compute mean, SD for ratings
  - [ ] Test inter-rater reliability (Fleiss' kappa)
  - [ ] Summarize qualitative feedback
  - [ ] Generate expert validation report

---

### Phase 3: Statistical Analysis (Week 5)

- [ ] **Compute Confidence Intervals**
  - [ ] Bootstrap CIs for all metrics
  - [ ] Report 95% CIs in tables

- [ ] **Perform Statistical Tests**
  - [ ] SHAP vs baselines (Wilcoxon/t-test)
  - [ ] NMF vs PCA/ICA (Wilcoxon/t-test)
  - [ ] Apply multiple comparisons correction

- [ ] **Compute Effect Sizes**
  - [ ] Cohen's d for key comparisons
  - [ ] Interpret clinical significance

---

### Phase 4: Documentation (Week 6)

- [ ] **Create Tables**
  - [ ] Table 1: SHAP validation metrics
  - [ ] Table 2: Counterfactual validation metrics
  - [ ] Table 3: NMF validation metrics
  - [ ] Table 4: Expert validation summary

- [ ] **Create Figures**
  - [ ] Figure: Bootstrap stability plots
  - [ ] Figure: Comparison with baselines
  - [ ] Figure: Expert rating distributions

- [ ] **Write Methods Section**
  - [ ] Describe validation protocol
  - [ ] Reference this document
  - [ ] Justify sample sizes

- [ ] **Write Results Section**
  - [ ] Report all metrics with CIs
  - [ ] Report statistical tests
  - [ ] Summarize expert feedback

---

## Part 7: Publication Standards

### 7.1 Reporting Checklist (TRIPOD-AI / CONSORT-AI)

- [ ] Sample size justification
- [ ] Validation metrics clearly defined
- [ ] Baseline comparisons included
- [ ] Statistical tests reported with effect sizes
- [ ] Expert validation methodology described
- [ ] Inter-rater reliability reported
- [ ] Confidence intervals for all estimates
- [ ] Multiple comparisons correction applied
- [ ] Limitations discussed

---

### 7.2 Supplementary Materials

Include in supplement:
- Full validation protocol (this document)
- Expert evaluation template
- Raw expert ratings (anonymized)
- Detailed statistical tables
- Bootstrap distribution plots

---

## Part 8: Troubleshooting

### Problem 1: Low Bootstrap Consistency (< 0.70)

**Possible causes:**
- Sample size too small → increase n
- Feature space too high-dimensional → feature selection
- Model instability → increase trees/regularization

**Solutions:**
- Increase bootstrap samples to 200
- Use stratified sampling
- Average across multiple random seeds

---

### Problem 2: Low Counterfactual Success Rate (< 50%)

**Possible causes:**
- Too strict constraints
- Optimization algorithm failing
- Decision boundary too rigid

**Solutions:**
- Increase max_attempts parameter
- Try different perturbation strategies
- Relax proximity constraints slightly

---

### Problem 3: Low Expert Ratings (< 3.0)

**Possible causes:**
- Genuinely poor explanations
- Expert misunderstanding of task
- Presentation format unclear

**Solutions:**
- Review explanations qualitatively
- Conduct expert training/calibration session
- Revise evaluation template for clarity
- Consider alternative XAI methods if truly poor

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Quantitative Validation** | 1-2 weeks | Validation reports (TXT, JSON) |
| **Expert Recruitment** | 1 week | 5 experts confirmed |
| **Expert Evaluation** | 2-3 weeks | Rating data collected |
| **Statistical Analysis** | 1 week | Statistical tables |
| **Documentation** | 1 week | Manuscript sections |
| **TOTAL** | **6-8 weeks** | Publication-ready validation |

---

## References

1. **Lundberg, S. M., et al. (2020).** "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1), 56-67.

2. **Wachter, S., et al. (2017).** "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." *Harvard Journal of Law & Technology*, 31, 841.

3. **Lee, D. D., & Seung, H. S. (1999).** "Learning the parts of objects by non-negative matrix factorization." *Nature*, 401(6755), 788-791.

4. **Collins, G. S., et al. (2021).** "Protocol for development of a reporting guideline (TRIPOD-AI) and risk of bias tool (PROBAST-AI) for diagnostic and prognostic prediction model studies based on artificial intelligence." *BMJ Open*, 11(7), e048008.

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-25
**Status:** Production Ready
**Next Review:** After expert validation complete
