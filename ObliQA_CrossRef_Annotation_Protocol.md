# ObliQA-CrossRef Human Annotation and Gold Subset Protocol

## 1. Purpose and Rationale

The human annotation phase serves two interconnected objectives:

1. **Validate the automatic (LLM-as-a-Judge) curation process.**
   The LLM-as-a-Judge step filtered large amounts of automatically generated question–answer (QA) pairs.  
   Human validation is required to quantify how accurate those automated “Keep/Eliminate” decisions were, measuring:
   - the **precision** of the LLM-Kept subset,  
   - the **false-negative rate** within the LLM-Eliminated subset, and  
   - the **inter-annotator agreement** (reliability) among experts.

2. **Create a high-quality human-verified Gold subset.**
   The final, human-confirmed “Kept” items form the **ObliQA-CrossRef-Gold** dataset.  
   This set provides a trusted benchmark for evaluating question answering and reasoning models on cross-referential regulatory text.

This design balances **statistical robustness** with **practical feasibility**, ensuring scientific validity without over-burdening the limited pool of expert annotators.

---

## 2. Background: Data Generation and Automatic Curation

The ObliQA-CrossRef dataset was constructed in two stages:  
1️⃣ **Generation:** LLM-based question–answer creation for each cross-reference pair.  
2️⃣ **Curation:** Automatic filtering using the LLM-as-a-Judge method.

### Table 1. Generation and Curation Statistics

| Stage | **Method 1: Prompting** |  |  | **Method 2: Schema** |  |  |  
|:--|:--:|:--:|:--:|:--:|:--:|:--:|  
|  | Basic Persona | Professional Persona | **Total** | Basic Persona | Professional Persona | **Total** |  
| **Step 1 – Generation** | 1 715 | 1 719 | **3 434** | 856 | 861 | **1 717** |  
| **Step 2 – Curation (LLM-as-a-Judge)** |  |  |  |  |  |  |  
| Kept | 534 | 808 | **1 342** | 246 | 378 | **624** |  
| Kept % | 31.14 % | 47.0 % | **39.08 %** | 28.74 % | 43.9 % | **36.34 %** |  
| Rejected | 1 181 | 911 | **2 092** | 610 | 483 | **1 093** |  
| └─ Reject target-only | 566 | 395 | **961** | 396 | 259 | **655** |  
| └─ Reject source-only | 554 | 439 | **993** | 158 | 137 | **295** |  
| └─ Reject incorrect answer | 35 | 58 | **93** | 43 | 76 | **119** |  
| └─ Reject other | 26 | 19 | **45** | 13 | 11 | **24** |

**Interpretation.**  
- The LLM retained ≈ 39 % of Prompting and 36 % of Schema QA pairs as seemingly valid.  
- Rejections were dominated by *target-only* and *source-only* cases, where the answer relied on incomplete context.  
- Automatic curation is effective but imperfect—hence the need for **human verification** to confirm that “Kept” items are genuinely correct and to quantify the error among rejections.

---

## 3. Why These Annotation Numbers Were Chosen

### Statistical grounding

| Estimate | Desired half-width (E) | Expected proportion (p) | Formula | Result |
|-----------|----------------------:|------------------------:|---------|--------:|
| Precision of Kept | ± 6 % | 0.5 (worst-case) | n=(1.96² × p(1-p)) / E² | ≈ 270 |
| False-negative rate | ± 5 % | 0.10 | same formula | ≈ 120 |
| Reliability (κ) | ± 0.10 half-width | 3-rater design | empirical simulation | 120 triple-labeled |
| Human time | 3 min / item | ≤ 10.5 h / annotator | fits these values |

These counts ensure:  
- statistically meaningful confidence intervals (±5–6 %),  
- balanced representation across **Method × Persona**, and  
- feasible workload for 3 domain annotators.

---

## 4. Composition of the Human Annotation Subset

| Bucket | Unique Items | Prompt–Basic | Prompt–Prof | Schema–Basic | Schema–Prof |
|---|---:|---:|---:|---:|---:|
| **Kept (Triple-Labeled for κ)** | **120** | 30 | 30 | 30 | 30 |
| **Kept (Single-Labeled)** | **150** | 38 | 37 | 37 | 38 |
| **Kept Subtotal (Gold)** | **270** | **68** | **67** | **67** | **68** |
| **Eliminated (Single-Labeled)** | **120** | 30 | 30 | 30 | 30 |
| **Grand Total (Unique)** | **390** | **98** | **97** | **97** | **98** |

- Both **Basic** and **Professional** personas are preserved to maintain linguistic and conceptual diversity.  
- Internal vs External references remain proportionally balanced within each cell.

---

## 5. Annotation Load and Logistics

| Component | Unique Items | Raters / Item | Total Labels | Labels / Annotator | Time @ 3 min / item |
|---|---:|---:|---:|---:|---:|
| Kept (κ block) | 120 | 3 | 360 | 120 | 6 h |
| Kept (Single) | 150 | 1 | 150 | 50 | 2.5 h |
| Eliminated | 120 | 1 | 120 | 40 | 2 h |
| **Totals** | **390** | — | **630** | **210** | **≈ 10.5 h / annotator** |

Each annotator therefore labels:
- 120 shared items (κ block),  
- 50 unique Kept items, and  
- 40 unique Eliminated items.

---

## 6. Step-by-Step Procedure

### Step 1. Input preparation
Load two automatically curated files:
- `LLM_Kept.jsonl`
- `LLM_Eliminated.jsonl`  
Each record contains:  
`Question`, `SourceText`, `TargetText`, `Answer`, `Method`, `Persona`, `ReferenceType`.

### Step 2. Stratified sampling
Using a fixed random seed (e.g., 13):
- Sample 120 Kept → all three annotators (labelled for κ).  
- Sample 150 additional Kept → distributed 50 per annotator (single-label).  
- Sample 120 Eliminated → distributed 40 per annotator (single-label).  
Maintain equal representation for Method (Prompt / Schema) and Persona (Basic / Professional).

### Step 3. Assignment generation
Produce three input files:
- `Annotator_A.jsonl`  
- `Annotator_B.jsonl`  
- `Annotator_C.jsonl`  
Each includes metadata identifying whether the item belongs to the shared κ block or is unique.

### Step 4. Annotation interface
Use a consistent form (Google Form / Label Studio) with fields:
1. **Question validity** (well-formed / unrealistic).  
2. **Required context** (Source / Target / Both / More info).  
3. **Answer quality** (correct & complete / partly correct / incorrect).  
4. **Final decision** (Keep / Eliminate).  
5. **Comment** (short justification).

### Step 5. Post-processing and analysis
Aggregate all labels:
- **Precision (Kept)** = Human Keep / Total LLM-Kept → ± 6 %.  
- **False-Negative Rate (Eliminated)** = Human Keep / Total LLM-Eliminated → ± 5 %.  
- **Reliability:** compute pairwise Cohen’s κ and Fleiss’ κ (over 120 shared items).  
- Derive the **Gold subset** = Kept items confirmed by majority vote / consensus.

### Step 6. Deliverables
| Output | Description |
|---|---|
| `obliqa_crossref_annotation_assignments/` | Individual annotator input files |
| `results/precision_FN_kappa_report.csv` | Summary statistics |
| `ObliQA-CrossRef-Gold.jsonl` | Human-verified gold dataset |
| `README_AnnotationProtocol.md` | Full documentation |

---

## 7. Expected Outcomes

| Measure | Target | Achieved Through |
|---|---:|---|
| Precision of LLM Kept | ± 6 % @ 95 % CI | 270 Kept samples |
| False-Negative Rate | ± 5 % @ 95 % CI | 120 Eliminated samples |
| Inter-Annotator κ | half-width ≈ ± 0.1 | 120 triple-labeled Kept |
| Gold subset size | ≈ 270 items | Human consensus over Kept |
| Workload | ≈ 10.5 h / annotator | 630 labels total |

---

### Summary
This protocol ensures:
- statistically validated assessment of LLM-based filtering,  
- a reliable human-verified Gold benchmark, and  
- efficient use of expert time without compromising quality or representativeness.  

The result is a balanced, interpretable, and reproducible foundation for evaluating regulatory QA models on multi-passage, cross-reference reasoning tasks.
