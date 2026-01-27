# SynDX XAI Framework - Publication Readiness Checklist

## Document Purpose

This checklist ensures that the SynDX XAI framework is ready for submission to top-tier journals (Nature Machine Intelligence, JAMA Network Open, npj Digital Medicine, etc.).

เอกสารนี้ทำให้มั่นใจว่า SynDX XAI framework พร้อมสำหรับการส่งไปยังวารสารชั้นนำ

---

## Pre-Submission Checklist

### Phase 1: Implementation ✅ COMPLETE

- [x] **SHAP Implementation**
  - [x] TreeExplainer integrated
  - [x] Summary plots (beeswarm, bar, waterfall)
  - [x] Local and global explanations
  - [x] 600 DPI PNG + vector PDF outputs

- [x] **Counterfactual Implementation**
  - [x] 3 perturbation strategies (flip, extreme, noise)
  - [x] Comparison visualizations
  - [x] Clinical reports (TXT, JSON, CSV)
  - [x] 600 DPI outputs

- [x] **NMF Implementation**
  - [x] Factor interpretation
  - [x] Patient profiling
  - [x] Factor-disease associations (ANOVA)
  - [x] Feature-factor networks (NetworkX)
  - [x] 600 DPI outputs

- [x] **Documentation**
  - [x] XAI_FEATURES_SUMMARY.md
  - [x] NMF_INTERPRETABILITY_SUMMARY.md
  - [x] XAI_INTEGRATION_GUIDE.md
  - [x] XAI_README.md

---

### Phase 2: Empirical Validation ⏳ IN PROGRESS

#### **SHAP Validation**

- [x] **Validation script created** (validate_xai_framework.py)
- [ ] **Bootstrap consistency** (n≥50)
  - [ ] Run bootstrap analysis
  - [ ] Target: mean correlation > 0.85
  - [ ] Document: 95% CI

- [ ] **Correlation with model importance**
  - [ ] Spearman correlation computed
  - [ ] Target: ρ > 0.60
  - [ ] P-value < 0.05

- [ ] **Permutation importance comparison**
  - [ ] Permutation test completed
  - [ ] Correlation computed
  - [ ] P-value < 0.05

- [ ] **Rank stability**
  - [ ] Kendall tau computed
  - [ ] Target: τ > 0.75
  - [ ] Across patient subgroups

- [ ] **Statistical testing**
  - [ ] Confidence intervals (95% CI)
  - [ ] Effect sizes (Cohen's d)
  - [ ] Multiple comparisons correction

---

#### **Counterfactual Validation**

- [x] **Validation script created**
- [ ] **Success rate**
  - [ ] Computed for all test patients
  - [ ] Target: > 70%
  - [ ] Document failures

- [ ] **Sparsity metrics**
  - [ ] Mean features changed
  - [ ] Target: 3-5 features
  - [ ] SD, min, max reported

- [ ] **Proximity metrics**
  - [ ] L2 distance computed
  - [ ] L1 distance computed
  - [ ] Distribution plots

- [ ] **Diversity metrics**
  - [ ] Pairwise distances
  - [ ] Target: high diversity
  - [ ] Visual inspection

- [ ] **Clinical plausibility**
  - [ ] N≥3 experts recruited
  - [ ] N≥20 counterfactuals rated
  - [ ] Target: mean ≥ 3.5/5
  - [ ] Inter-rater reliability (Fleiss' kappa)

- [ ] **Statistical testing**
  - [ ] CIs for all metrics
  - [ ] Expert agreement statistics
  - [ ] Qualitative feedback summarized

---

#### **NMF Validation**

- [x] **Validation script created**
- [ ] **Bootstrap stability**
  - [ ] N≥50 bootstrap samples
  - [ ] Factor correlation computed
  - [ ] Target: mean > 0.75
  - [ ] 95% CI

- [ ] **Reconstruction error**
  - [ ] RMSE computed
  - [ ] MAE computed
  - [ ] Frobenius norm

- [ ] **Comparison with PCA**
  - [ ] Explained variance compared
  - [ ] Reconstruction error compared
  - [ ] Statistical test (Wilcoxon)

- [ ] **Comparison with ICA**
  - [ ] Silhouette scores compared
  - [ ] Statistical test

- [ ] **Comparison with K-means**
  - [ ] Clustering quality compared
  - [ ] Davies-Bouldin index

- [ ] **Clustering metrics**
  - [ ] Silhouette score > 0.35
  - [ ] Calinski-Harabasz index
  - [ ] Davies-Bouldin index

- [ ] **Clinical coherence**
  - [ ] N≥3 experts recruited
  - [ ] All 20 factors reviewed
  - [ ] Target: mean coherence ≥ 3.5/5
  - [ ] Factor naming consensus ≥ 60%

- [ ] **Statistical testing**
  - [ ] CIs for stability
  - [ ] Effect sizes vs baselines
  - [ ] Bonferroni correction applied

---

### Phase 3: Expert Validation ⏳ PENDING

- [ ] **Expert Recruitment**
  - [ ] N≥3 experts confirmed (Target: 5)
  - [ ] Diverse specializations:
    - [ ] Neurology
    - [ ] ENT
    - [ ] Emergency Medicine
  - [ ] Diverse practice settings:
    - [ ] Academic
    - [ ] Community
    - [ ] Mixed

- [ ] **Evaluation Materials Prepared**
  - [ ] SHAP top features list
  - [ ] 20-50 counterfactual examples
  - [ ] 20 NMF factor descriptions
  - [ ] Evaluation forms (EXPERT_EVALUATION_TEMPLATE.md)
  - [ ] IRB approval (if required)

- [ ] **Data Collection**
  - [ ] SHAP coherence ratings
  - [ ] Counterfactual plausibility ratings
  - [ ] NMF interpretability ratings
  - [ ] Qualitative feedback

- [ ] **Data Analysis**
  - [ ] Mean, SD for all ratings
  - [ ] Inter-rater reliability (Fleiss' kappa > 0.40)
  - [ ] Qualitative themes identified
  - [ ] Expert validation report generated

---

### Phase 4: Manuscript Preparation ⏳ PENDING

#### **Title**

- [ ] **Draft title**
  - [ ] Concise (< 150 characters)
  - [ ] Describes contribution clearly
  - [ ] Keywords included

**Suggested:**
> "SynDX: A Multi-Dimensional Explainable AI Framework for Synthetic Data Generation and Clinical Phenotype Discovery in Vestibular Disorder Diagnosis"

---

#### **Abstract**

- [ ] **Structured abstract** (250 words)
  - [ ] Background (2-3 sentences)
  - [ ] Methods (3-4 sentences)
  - [ ] Results (4-5 sentences)
  - [ ] Conclusions (2-3 sentences)

- [ ] **Key results quantified**
  - [ ] Model accuracy: ___%
  - [ ] SHAP consistency: ___
  - [ ] CF success rate: ___%
  - [ ] NMF stability: ___
  - [ ] Expert ratings: ___/5

---

#### **Introduction**

- [ ] **Clinical problem** defined
  - [ ] Vestibular disorder prevalence
  - [ ] Diagnostic challenges (stroke misdiagnosis rate)
  - [ ] Need for explainability in medical AI

- [ ] **Literature review**
  - [ ] XAI in healthcare (SHAP, LIME, etc.)
  - [ ] Counterfactuals in medicine
  - [ ] Phenotype discovery (NMF, clustering)
  - [ ] Gap identified: No comprehensive XAI for vestibular disorders

- [ ] **Study objectives** clearly stated
  - [ ] Develop 3 complementary XAI methods
  - [ ] Validate empirically
  - [ ] Demonstrate clinical utility

---

#### **Methods**

- [ ] **Data generation** described
  - [ ] SynDX Phase 1 (parameter space)
  - [ ] N=8,400 synthetic patients
  - [ ] 7 vestibular disorders
  - [ ] 150 clinical features

- [ ] **Model training** described
  - [ ] XGBoost classifier
  - [ ] Train/test split (80/20)
  - [ ] Hyperparameters reported
  - [ ] Performance metrics

- [ ] **XAI methods** described
  - [ ] SHAP implementation
  - [ ] Counterfactual algorithm
  - [ ] NMF decomposition (20 factors)

- [ ] **Validation protocol** described
  - [ ] Reference VALIDATION_PROTOCOL.md
  - [ ] Quantitative metrics defined
  - [ ] Expert evaluation procedure
  - [ ] Statistical methods

- [ ] **Sample size** justified
  - [ ] Power analysis for correlations
  - [ ] N experts justified
  - [ ] N counterfactuals justified

- [ ] **Ethics** addressed
  - [ ] IRB approval (if required for expert study)
  - [ ] Data privacy
  - [ ] No patient data used (synthetic)

---

#### **Results**

- [ ] **Model performance**
  - [ ] Accuracy, precision, recall, F1
  - [ ] Confusion matrix
  - [ ] ROC-AUC per class

- [ ] **SHAP validation results**
  - [ ] Table 1: SHAP metrics (with CIs)
  - [ ] Figure 1: SHAP summary plots
  - [ ] Figure 2: Bootstrap stability
  - [ ] Expert ratings summarized
  - [ ] Qualitative feedback themes

- [ ] **Counterfactual validation results**
  - [ ] Table 2: CF metrics (with CIs)
  - [ ] Figure 3: CF examples
  - [ ] Figure 4: Sparsity distribution
  - [ ] Expert plausibility ratings
  - [ ] Success rate by diagnosis

- [ ] **NMF validation results**
  - [ ] Table 3: NMF metrics (with CIs)
  - [ ] Figure 5: Factor heatmap
  - [ ] Figure 6: Patient profiles
  - [ ] Figure 7: Factor-disease associations
  - [ ] Figure 8: Feature-factor network
  - [ ] Expert coherence ratings
  - [ ] Factor naming consensus

- [ ] **Statistical comparisons**
  - [ ] SHAP vs LIME/permutation
  - [ ] NMF vs PCA/ICA/K-means
  - [ ] Effect sizes reported
  - [ ] P-values with correction

---

#### **Discussion**

- [ ] **Main findings** summarized
  - [ ] 3 XAI methods successfully validated
  - [ ] Clinical expert approval
  - [ ] Empirical metrics meet benchmarks

- [ ] **Comparison with literature**
  - [ ] SHAP consistency vs prior work
  - [ ] CF plausibility vs prior work
  - [ ] NMF stability vs prior work

- [ ] **Clinical implications**
  - [ ] Emergency department use case
  - [ ] Medical education
  - [ ] Phenotype-based treatment
  - [ ] Regulatory considerations (FDA)

- [ ] **Strengths**
  - [ ] First comprehensive XAI for vestibular disorders
  - [ ] Multi-dimensional approach (WHY, HOW, WHAT)
  - [ ] Rigorous empirical validation
  - [ ] Clinical expert validation

- [ ] **Limitations**
  - [ ] Synthetic data (not real patients)
  - [ ] Single-center expert panel
  - [ ] English language only
  - [ ] Limited to 7 diagnoses

- [ ] **Future work**
  - [ ] Real-world validation
  - [ ] Prospective clinical trial
  - [ ] Additional XAI methods (prototypes, anchors)
  - [ ] Multi-modal data (imaging + clinical)

---

#### **Conclusion**

- [ ] **Concise summary** (1 paragraph)
- [ ] **Clinical impact** stated
- [ ] **Call to action** (adoption, further research)

---

#### **References**

- [ ] **N≥30 references**
  - [ ] XAI methodology papers
  - [ ] Clinical vestibular disorder papers
  - [ ] Medical AI papers
  - [ ] Validation methodology papers

- [ ] **Key citations included**:
  - [ ] Lundberg & Lee (2017) - SHAP
  - [ ] Wachter et al. (2017) - Counterfactuals
  - [ ] Lee & Seung (1999) - NMF
  - [ ] Seymour et al. (2019) - Clinical phenotypes
  - [ ] Collins et al. (2021) - TRIPOD-AI

- [ ] **Formatted correctly** for target journal

---

#### **Tables**

- [ ] **Table 1:** Model Performance Metrics
  - [ ] Accuracy, Precision, Recall, F1 per class
  - [ ] Overall metrics with CIs

- [ ] **Table 2:** SHAP Validation Metrics
  - [ ] Bootstrap consistency
  - [ ] Correlations with baselines
  - [ ] Rank stability
  - [ ] Expert ratings

- [ ] **Table 3:** Counterfactual Validation Metrics
  - [ ] Success rate
  - [ ] Sparsity
  - [ ] Proximity
  - [ ] Diversity
  - [ ] Expert plausibility ratings

- [ ] **Table 4:** NMF Validation Metrics
  - [ ] Bootstrap stability
  - [ ] Reconstruction error
  - [ ] Comparison with PCA/ICA
  - [ ] Clustering metrics
  - [ ] Expert coherence ratings

- [ ] **Table 5:** Expert Validation Summary
  - [ ] Demographics (specialty, experience)
  - [ ] Ratings by XAI method
  - [ ] Inter-rater reliability

---

#### **Figures**

- [ ] **Figure 1:** SHAP Summary Plots
  - [ ] Panel A: Beeswarm plot
  - [ ] Panel B: Bar plot (top 20 features)
  - [ ] 600 DPI, color-blind friendly

- [ ] **Figure 2:** Counterfactual Examples
  - [ ] Panel A: Stroke → VN example
  - [ ] Panel B: BPPV → Migraine example
  - [ ] Panel C: Sparsity distribution

- [ ] **Figure 3:** NMF Factor Heatmap
  - [ ] 20 factors × top 10 features each
  - [ ] Color-coded by weight

- [ ] **Figure 4:** NMF Patient Profiles
  - [ ] Panel A: Pie chart examples
  - [ ] Panel B: Radar chart examples
  - [ ] Representative of each diagnosis

- [ ] **Figure 5:** Factor-Disease Association Heatmap
  - [ ] Statistical significance marked
  - [ ] Dendrograms showing hierarchy

- [ ] **Figure 6:** Feature-Factor Network
  - [ ] Bipartite graph
  - [ ] Hub features highlighted

- [ ] **Figure 7:** Validation Metrics Comparison
  - [ ] Panel A: SHAP vs baselines
  - [ ] Panel B: NMF vs PCA/ICA
  - [ ] Error bars (95% CI)

---

#### **Supplementary Materials**

- [ ] **Supplementary Methods**
  - [ ] Detailed algorithm descriptions
  - [ ] Hyperparameter tuning
  - [ ] Complete validation protocol (VALIDATION_PROTOCOL.md)

- [ ] **Supplementary Tables**
  - [ ] S1: Full feature list (150 features)
  - [ ] S2: NMF factor interpretations (all 20)
  - [ ] S3: Expert ratings (detailed, anonymized)
  - [ ] S4: Statistical test details

- [ ] **Supplementary Figures**
  - [ ] S1: All SHAP waterfall plots (7 diagnoses)
  - [ ] S2: Bootstrap distribution plots
  - [ ] S3: All NMF factor bar plots
  - [ ] S4: Clustering comparison plots

- [ ] **Code Availability**
  - [ ] GitHub repository URL
  - [ ] DOI (via Zenodo)
  - [ ] Requirements.txt
  - [ ] README with instructions

- [ ] **Data Availability**
  - [ ] Synthetic data generation code
  - [ ] Archetype data (if shareable)
  - [ ] Validation results (JSON, CSV)

---

### Phase 5: Journal-Specific Requirements ⏳ PENDING

#### **Target Journal Selection**

**Tier 1 (High Impact):**
- [ ] Nature Machine Intelligence (IF: 25.9)
- [ ] Nature Medicine (IF: 82.9)
- [ ] JAMA Network Open (IF: 13.8)

**Tier 2 (Mid Impact):**
- [ ] npj Digital Medicine (IF: 15.2)
- [ ] The Lancet Digital Health (IF: 36.0)
- [ ] Journal of Medical Internet Research (IF: 5.8)

**Tier 3 (Domain-Specific):**
- [ ] Artificial Intelligence in Medicine (IF: 7.5)
- [ ] Computer Methods and Programs in Biomedicine (IF: 6.1)
- [ ] IEEE Journal of Biomedical and Health Informatics (IF: 6.7)

---

#### **Journal-Specific Formatting**

- [ ] **Word count** within limit
  - [ ] Abstract: ____ / ____ words
  - [ ] Main text: ____ / ____ words

- [ ] **Reference style** correct
  - [ ] Vancouver / AMA / Nature style

- [ ] **Figure format** requirements
  - [ ] File types (TIFF, EPS, PDF)
  - [ ] Resolution (300-600 DPI)
  - [ ] Color model (RGB/CMYK)

- [ ] **Author guidelines** followed
  - [ ] Structure (IMRaD)
  - [ ] Sections required
  - [ ] Supplementary limits

---

#### **Reporting Guidelines**

- [ ] **TRIPOD-AI Checklist** completed
  - [ ] Items 1-27 addressed
  - [ ] AI-specific items included
  - [ ] Checklist included in supplement

- [ ] **CONSORT-AI** (if applicable)
  - [ ] N/A (not a clinical trial)

- [ ] **STARD-AI** (if applicable)
  - [ ] N/A (not diagnostic accuracy study per se)

---

### Phase 6: Pre-Submission Review ⏳ PENDING

- [ ] **Internal Review**
  - [ ] Co-authors reviewed manuscript
  - [ ] All authors approved
  - [ ] Authorship criteria met (ICMJE)

- [ ] **Statistical Review**
  - [ ] Statistician consulted
  - [ ] Methods validated
  - [ ] Results verified

- [ ] **Language Editing**
  - [ ] Professional editing (if non-native English)
  - [ ] Grammar/spelling checked
  - [ ] Consistent terminology

- [ ] **Plagiarism Check**
  - [ ] iThenticate / Turnitin run
  - [ ] Similarity < 15%
  - [ ] No self-plagiarism issues

- [ ] **Conflict of Interest**
  - [ ] COI forms completed
  - [ ] Funding disclosed
  - [ ] Industry relationships declared

- [ ] **Cover Letter**
  - [ ] Novelty highlighted
  - [ ] Clinical impact emphasized
  - [ ] Suggested reviewers (3-5)
  - [ ] Excluded reviewers (if any)

---

### Phase 7: Submission ⏳ PENDING

- [ ] **Submission System**
  - [ ] Account created
  - [ ] Manuscript uploaded
  - [ ] Figures uploaded
  - [ ] Supplementary files uploaded

- [ ] **Metadata**
  - [ ] Title entered
  - [ ] Abstract entered
  - [ ] Keywords entered (5-8)
  - [ ] Author information complete

- [ ] **Declarations**
  - [ ] Ethics approval
  - [ ] Funding sources
  - [ ] Data availability
  - [ ] Code availability
  - [ ] Competing interests

- [ ] **Final Checks**
  - [ ] All files uploaded correctly
  - [ ] PDF preview reviewed
  - [ ] Co-authors notified
  - [ ] Submission fee ready (if applicable)

- [ ] **Submit!** 🚀

---

## Post-Submission Plan

### During Review

- [ ] **Monitor submission status**
- [ ] **Prepare for revisions**
  - [ ] Anticipate reviewer concerns
  - [ ] Prepare additional analyses
  - [ ] Draft rebuttal outline

### If Accepted

- [ ] **Proofs review** (carefully!)
- [ ] **Press release** (if high-impact)
- [ ] **Social media** announcement
- [ ] **Institution notification**
- [ ] **Preprint** (bioRxiv/medRxiv)

### If Rejected

- [ ] **Don't panic!** (Rejection rate ~70% for Nature/JAMA)
- [ ] **Read reviews carefully**
- [ ] **Improve manuscript**
- [ ] **Resubmit to Tier 2 journal**

---

## Success Criteria

### **Minimum Acceptable Standards**

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Model Accuracy** | > 85% | Comparable to physician accuracy |
| **SHAP Consistency** | > 0.80 | Good stability |
| **SHAP Expert Rating** | ≥ 3.5/5 | Clinical coherence |
| **CF Success Rate** | > 70% | High coverage |
| **CF Sparsity** | 3-5 features | Actionable |
| **CF Expert Rating** | ≥ 3.5/5 | Clinically plausible |
| **NMF Stability** | > 0.75 | Reasonably robust |
| **NMF Coherence Rating** | ≥ 3.5/5 | Interpretable |
| **Expert Sample** | ≥ 3 experts | Minimum for validity |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Implementation** | DONE ✅ | - |
| **Quantitative Validation** | 1-2 weeks | Need test data |
| **Expert Recruitment** | 1 week | IRB approval |
| **Expert Evaluation** | 2-3 weeks | Expert availability |
| **Data Analysis** | 1 week | Expert data complete |
| **Manuscript Writing** | 2-3 weeks | All results ready |
| **Internal Review** | 1 week | Co-authors |
| **Revisions** | 1 week | Feedback |
| **Submission** | 1 day | All materials ready |
| **TOTAL** | **8-12 weeks** | **From today** |

---

## Current Status Summary

### ✅ Completed

1. **SHAP Implementation** - Full functionality
2. **Counterfactual Implementation** - Full functionality
3. **NMF Implementation** - Full functionality
4. **Documentation** - Comprehensive (4 docs)
5. **Validation Framework** - Script created & tested

### ⏳ In Progress

6. **Quantitative Validation** - Script ready, need real data run
7. **Expert Evaluation** - Template ready, need recruitment

### ⏸️ Pending

8. **Statistical Analysis** - Awaiting validation data
9. **Manuscript Writing** - Awaiting results
10. **Journal Submission** - Final step

---

## Next Immediate Steps

1. **Run full validation** on real SynDX data
   ```bash
   python scripts/validate_xai_framework.py
   ```

2. **Review validation results**
   - Check if metrics meet targets
   - Identify any issues

3. **Recruit clinical experts** (N≥3)
   - Contact neurologists/ENT specialists
   - Obtain IRB approval if needed

4. **Prepare expert evaluation materials**
   - Finalize SHAP top features
   - Select 20-50 counterfactuals
   - Export NMF factor descriptions

5. **Begin manuscript draft** (parallel to expert eval)

---

## Contact / Support

For questions about this checklist:
- **PI:** Chatchai Tritham
- **Advisor:** [Advisor name]
- **GitHub:** https://github.com/ChatchaiTritham/SynDX

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-25
**Next Update:** After validation complete
**Target Submission:** [Target date]
