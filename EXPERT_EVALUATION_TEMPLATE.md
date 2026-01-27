# Clinical Expert Evaluation Template
# แบบประเมินสำหรับผู้เชี่ยวชาญทางคลินิก

## Document Information / ข้อมูลเอกสาร

**Project:** SynDX - Explainable AI for Vestibular Disorder Diagnosis
**Evaluation Version:** 1.0.0
**Date:** 2026-01-25
**Estimated Time:** 45-60 minutes

---

## Expert Information / ข้อมูลผู้ประเมิน

**Name:** _________________________________ (Optional / สมัครใจ)
**Specialization:**
- [ ] Neurology / ประสาทวิทยา
- [ ] Otolaryngology (ENT) / โสต ศอ นาสิก
- [ ] Emergency Medicine / เวชศาสตร์ฉุกเฉิน
- [ ] Internal Medicine / อายุรศาสตร์
- [ ] Other / อื่นๆ: _______________________

**Years of Experience:**
- [ ] < 5 years
- [ ] 5-10 years
- [ ] 10-20 years
- [ ] > 20 years

**Practice Setting:**
- [ ] Academic hospital / โรงพยาบาลศูนย์/มหาวิทยาลัย
- [ ] Community hospital / โรงพยาบาลชุมชน
- [ ] Private practice / คลินิกเอกชน
- [ ] Mixed / ผสม

**Vertigo/Dizziness Patients per Month:**
- [ ] < 10
- [ ] 10-30
- [ ] 30-50
- [ ] > 50

---

## Instructions / คำแนะนำ

### Purpose / วัตถุประสงค์

This evaluation assesses the **clinical validity** of three explainable AI (XAI) methods developed for vestibular disorder diagnosis:

1. **SHAP Values** - Feature importance rankings
2. **Counterfactual Explanations** - "What if" scenarios
3. **NMF Phenotypes** - Clinical syndrome patterns

การประเมินนี้วัด**ความถูกต้องทางคลินิก**ของ XAI methods สำหรับการวินิจฉัยโรคระบบการทรงตัว

### How to Complete / วิธีการกรอก

- Answer all questions honestly based on your clinical expertise
- Use the 1-5 Likert scale provided
- Provide qualitative comments where requested
- There are no "right" or "wrong" answers - we value your expert opinion

- ตอบคำถามทั้งหมดตามความรู้ทางคลินิกของท่าน
- ใช้ Likert scale 1-5 ตามที่กำหนด
- ให้ความเห็นเชิงคุณภาพตามที่ขอ
- ไม่มีคำตอบ "ถูก" หรือ "ผิด" - เราต้องการความเห็นของผู้เชี่ยวชาญ

### Scale Definitions / คำนิยาม

**1** = Strongly Disagree / ไม่เห็นด้วยอย่างยิ่ง
**2** = Disagree / ไม่เห็นด้วย
**3** = Neutral / ไม่แน่ใจ/กลางๆ
**4** = Agree / เห็นด้วย
**5** = Strongly Agree / เห็นด้วยอย่างยิ่ง

---

## Part 1: SHAP Feature Importance Evaluation

### Background / ความเป็นมา

SHAP (SHapley Additive exPlanations) ranks features by their importance in predicting each vestibular disorder diagnosis. Below are the **top 10 most important features** identified by SHAP analysis.

SHAP จัดอันดับคุณสมบัติตามความสำคัญในการทำนายการวินิจฉัยโรคระบบการทรงตัว ด้านล่างคือ **10 คุณสมบัติสำคัญที่สุด** ที่ SHAP วิเคราะห์ได้

---

### Top 10 SHAP Features (Example - Replace with Actual Results)

1. **HINTS Exam - Central Signs** (SHAP value: 0.42)
2. **Vertigo Type - Positional** (SHAP value: 0.38)
3. **Cardiovascular Risk Factors** (SHAP value: 0.35)
4. **Age** (SHAP value: 0.31)
5. **Hearing Loss** (SHAP value: 0.28)
6. **Nystagmus - Horizontal** (SHAP value: 0.26)
7. **Vertigo - Continuous** (SHAP value: 0.24)
8. **Duration - Seconds** (SHAP value: 0.22)
9. **Headache** (SHAP value: 0.19)
10. **Anxiety Symptoms** (SHAP value: 0.17)

---

### Questions

**Q1.1** The top 10 SHAP features align with my clinical understanding of important diagnostic features for vestibular disorders.

10 คุณสมบัติอันดับต้นสอดคล้องกับความเข้าใจทางคลินิกของฉันเกี่ยวกับคุณสมบัติที่สำคัญในการวินิจฉัย

- [ ] 1 - Strongly Disagree
- [ ] 2 - Disagree
- [ ] 3 - Neutral
- [ ] 4 - Agree
- [ ] 5 - Strongly Agree

**Comments / ความเห็น:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

**Q1.2** The relative ranking of these features makes clinical sense (e.g., HINTS exam ranked higher than anxiety).

การจัดอันดับสัมพัทธ์ของคุณสมบัติเหล่านี้สมเหตุสมผลทางคลินิก

- [ ] 1 - Strongly Disagree
- [ ] 2 - Disagree
- [ ] 3 - Neutral
- [ ] 4 - Agree
- [ ] 5 - Strongly Agree

**Comments:**
___________________________________________________________________
___________________________________________________________________

---

**Q1.3** Are there any important diagnostic features missing from this top 10 list?

มีคุณสมบัติสำคัญใดที่หายไปจากรายการ 10 อันดับนี้หรือไม่?

- [ ] No / ไม่มี
- [ ] Yes / มี → Please specify / โปรดระบุ:

___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

**Q1.4** Are there any features in this top 10 that you would NOT consider important for vestibular disorder diagnosis?

มีคุณสมบัติใดใน 10 อันดับที่ท่านไม่คิดว่าสำคัญสำหรับการวินิจฉัยหรือไม่?

- [ ] No / ไม่มี
- [ ] Yes / มี → Please specify / โปรดระบุ:

___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

**Q1.5** Overall, how would you rate the clinical coherence of these SHAP feature importance rankings?

โดยรวม ท่านให้คะแนนความสอดคล้องทางคลินิกของการจัดอันดับความสำคัญของ SHAP เท่าไร?

- [ ] 1 - Very poor
- [ ] 2 - Poor
- [ ] 3 - Acceptable
- [ ] 4 - Good
- [ ] 5 - Excellent

---

## Part 2: Counterfactual Explanation Evaluation

### Background

Counterfactual explanations show "what if" scenarios - minimal changes to a patient's features that would alter the diagnosis. These provide actionable insights for clinical decision-making.

Counterfactual explanations แสดง "what if" scenarios - การเปลี่ยนแปลงเล็กน้อยที่จะเปลี่ยนการวินิจฉัย

---

### Instructions

Please evaluate **10 counterfactual explanations** (provided separately) for clinical plausibility.

For each counterfactual, rate:
1. **Clinical Plausibility** - Could this change realistically occur?
2. **Actionability** - Is this insight useful for clinical decision-making?

โปรดประเมิน **10 counterfactual explanations** (แนบแยก) สำหรับความเป็นไปได้ทางคลินิก

---

### Example Counterfactual

**Original Patient:**
- Diagnosis: Stroke (Predicted: 94% confidence)
- Age: 68 years
- HINTS Exam: Central signs present (score: 0.89)
- Risk Factors: High (score: 0.73)
- Vertigo: Continuous (score: 0.92)

**Counterfactual:**
- **IF** HINTS Exam changed from 0.89 → 0.12 (central to peripheral signs)
- **AND** Risk Factors changed from 0.73 → 0.25 (high to low)
- **THEN** Diagnosis would flip to: Vestibular Neuritis (85% confidence)

**Changed Features:** 2 features
**Clinical Interpretation:** Patient with peripheral HINTS findings and low vascular risk is unlikely to have stroke; more consistent with Vestibular Neuritis.

---

### Rating Template (Repeat for Each of 10 Counterfactuals)

#### Counterfactual #1

**Clinical Plausibility:**
How realistic is this counterfactual scenario?

ความเป็นไปได้ทางคลินิกของสถานการณ์นี้เท่าไร?

- [ ] 1 - Clinically impossible / เป็นไปไม่ได้ทางคลินิก
- [ ] 2 - Highly unlikely / ไม่น่าจะเกิดขึ้น
- [ ] 3 - Possible but uncommon / เป็นไปได้แต่ไม่ค่อยพบ
- [ ] 4 - Plausible / เป็นไปได้
- [ ] 5 - Highly plausible / เป็นไปได้สูง

**Actionability:**
Is this insight useful for clinical decision-making or patient counseling?

ข้อมูลนี้มีประโยชน์สำหรับการตัดสินใจทางคลินิกหรือให้คำปรึกษาผู้ป่วยหรือไม่?

- [ ] 1 - Not useful at all
- [ ] 2 - Minimally useful
- [ ] 3 - Somewhat useful
- [ ] 4 - Useful
- [ ] 5 - Very useful

**Comments:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

*(Repeat sections for Counterfactuals #2-#10)*

---

### Summary Questions - Counterfactuals

**Q2.1** Overall, how would you rate the clinical plausibility of the counterfactual explanations?

โดยรวม ท่านให้คะแนนความเป็นไปได้ทางคลินิกของ counterfactual explanations เท่าไร?

- [ ] 1 - Very poor
- [ ] 2 - Poor
- [ ] 3 - Acceptable
- [ ] 4 - Good
- [ ] 5 - Excellent

---

**Q2.2** Would you use these counterfactual explanations in your clinical practice?

ท่านจะใช้ counterfactual explanations เหล่านี้ในการปฏิบัติทางคลินิกหรือไม่?

- [ ] Definitely not / ไม่แน่นอน
- [ ] Probably not / อาจจะไม่
- [ ] Unsure / ไม่แน่ใจ
- [ ] Probably yes / อาจจะใช้
- [ ] Definitely yes / ใช้แน่นอน

**For what purpose? / ใช้เพื่ออะไร?**
- [ ] Differential diagnosis / การวินิจฉัยแยกโรค
- [ ] Patient counseling / ให้คำปรึกษาผู้ป่วย
- [ ] Treatment planning / วางแผนการรักษา
- [ ] Medical education / การศึกษาแพทย์
- [ ] Other / อื่นๆ: ____________________

---

## Part 3: NMF Phenotype Evaluation

### Background

NMF (Non-negative Matrix Factorization) discovered **20 latent clinical phenotypes** from 150 vestibular disorder features. Each phenotype represents a coherent cluster of symptoms/signs.

NMF ค้นพบ **20 phenotypes ทางคลินิกแฝง** จาก 150 คุณสมบัติของโรคระบบการทรงตัว

---

### Instructions

Below are **5 representative phenotypes** (out of 20 total). For each phenotype:

1. **Review top contributing features**
2. **Assign a clinical name** if it represents a known syndrome
3. **Rate clinical coherence** (do features logically cluster together?)

---

### Example Phenotype

#### Phenotype #1

**Top Contributing Features:**
1. Vertigo - Continuous (weight: 0.42, 28%)
2. Nystagmus - Horizontal (weight: 0.38, 25%)
3. Nausea/Vomiting (weight: 0.31, 21%)
4. Imbalance (weight: 0.27, 18%)
5. Sudden Onset (weight: 0.19, 13%)

**Your Clinical Name for this Phenotype:**

___________________________________________________________________

**Suggested Name (Optional):** "Acute Vestibular Syndrome"

---

**Clinical Coherence:**
Do these features logically cluster together as a coherent clinical phenotype?

คุณสมบัติเหล่านี้รวมกลุ่มกันเป็น phenotype ทางคลินิกที่สอดคล้องกันหรือไม่?

- [ ] 1 - No coherence / ไม่สอดคล้องกันเลย
- [ ] 2 - Poor coherence / สอดคล้องน้อย
- [ ] 3 - Moderate coherence / สอดคล้องปานกลาง
- [ ] 4 - Good coherence / สอดคล้องดี
- [ ] 5 - Excellent coherence / สอดคล้องมาก

**Comments:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

*(Repeat sections for Phenotypes #2-#5)*

---

### Summary Questions - NMF Phenotypes

**Q3.1** Overall, how would you rate the clinical interpretability of the NMF phenotypes?

โดยรวม ท่านให้คะแนนการตีความทางคลินิกของ NMF phenotypes เท่าไร?

- [ ] 1 - Not interpretable
- [ ] 2 - Difficult to interpret
- [ ] 3 - Moderately interpretable
- [ ] 4 - Interpretable
- [ ] 5 - Highly interpretable

---

**Q3.2** Do these phenotypes represent clinically meaningful patterns that you recognize in practice?

Phenotypes เหล่านี้แสดงรูปแบบทางคลินิกที่มีความหมายที่ท่านพบในการปฏิบัติหรือไม่?

- [ ] 1 - Not at all
- [ ] 2 - Rarely
- [ ] 3 - Sometimes
- [ ] 4 - Often
- [ ] 5 - Very often

---

**Q3.3** Would these phenotypes be useful for:

Phenotypes เหล่านี้จะมีประโยชน์สำหรับ:

| Use Case | Rating (1-5) |
|----------|--------------|
| Understanding individual patient presentations | ____ |
| Differential diagnosis | ____ |
| Treatment planning | ____ |
| Medical education / teaching | ____ |
| Research (subgroup analysis) | ____ |
| Clinical trial stratification | ____ |

---

## Part 4: Overall XAI Framework Evaluation

### Q4.1 - Trust and Adoption

If this XAI system were available in your clinical setting, how likely would you be to use it?

ถ้าระบบ XAI นี้มีในที่ทำงานของท่าน ท่านจะใช้มากน้อยแค่ไหน?

- [ ] 1 - Would not use
- [ ] 2 - Unlikely to use
- [ ] 3 - Might use occasionally
- [ ] 4 - Would use regularly
- [ ] 5 - Would use very frequently

**What would increase your trust in this system?**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

### Q4.2 - Most Valuable XAI Component

Which XAI method do you find most clinically valuable?

XAI method ใดที่ท่านคิดว่ามีคุณค่าทางคลินิกมากที่สุด?

- [ ] SHAP feature importance
- [ ] Counterfactual explanations
- [ ] NMF phenotypes
- [ ] All equally valuable
- [ ] None particularly valuable

**Why? / ทำไม?**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

### Q4.3 - Concerns and Limitations

What are your main concerns about using AI explanations in clinical practice?

ข้อกังวลหลักของท่านเกี่ยวกับการใช้คำอธิบาย AI ในทางคลินิกคืออะไร?

- [ ] Accuracy/reliability
- [ ] Medicolegal liability
- [ ] Over-reliance on AI
- [ ] Difficulty interpreting
- [ ] Lack of transparency
- [ ] Other: _______________________

**Comments:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

### Q4.4 - Improvements

What improvements would make these XAI methods more useful for clinical practice?

การปรับปรุงอะไรจะทำให้ XAI methods เหล่านี้มีประโยชน์มากขึ้นสำหรับทางคลินิก?

___________________________________________________________________
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

### Q4.5 - Free-Form Feedback

Please provide any additional comments, suggestions, or concerns.

กรุณาให้ความคิดเห็น ข้อเสนอแนะ หรือข้อกังวลเพิ่มเติม

___________________________________________________________________
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

## Data Privacy / ความเป็นส่วนตัว

- Your responses will be anonymized
- Individual ratings will not be identified
- Only aggregate statistics will be reported
- You may withdraw at any time

- คำตอบของท่านจะไม่ระบุตัวตน
- การให้คะแนนแต่ละรายจะไม่ถูกระบุ
- จะรายงานเฉพาะสถิติรวม
- ท่านสามารถถอนตัวได้ตลอดเวลา

---

## Thank You! / ขอบคุณ!

Thank you for your valuable time and expert feedback. Your insights are critical for validating this XAI framework and ensuring it meets clinical needs.

ขอบคุณสำหรับเวลาอันมีค่าและความคิดเห็นจากผู้เชี่ยวชาญ ความเห็นของท่านมีความสำคัญต่อการตรวจสอบ XAI framework และทำให้มั่นใจว่าตอบสนองความต้องการทางคลินิก

**Contact Information:**
For questions or concerns, please contact:

- **PI Name:** Chatchai Tritham
- **Email:** [your email]
- **Institution:** [your institution]

---

**Document Version:** 1.0.0
**IRB Approval:** [Pending/Approved - IRB#______]
**Date:** 2026-01-25
