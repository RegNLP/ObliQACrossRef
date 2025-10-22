# ObliQA-CrossRef Human Annotation and Gold Subset Protocol (Practical-min)

## 1. Purpose and Rationale

**Primary goals**

1. **Method comparison (Prompt vs Schema)** on precision of LLM-Kept items.
2. **Validation of LLM-as-a-Judge** (overall precision of Kept; false-negative rate in Eliminated; reliability).
3. **Produce a human-verified Gold subset** (ObliQA-CrossRef-Gold).

This setup balances statistical power for the **method comparison** with feasible expert workload.

---

## 2. Background: Data Generation and Automatic Curation

Two-stage construction:
1️⃣ **Generation** of QA pairs per cross-reference.
2️⃣ **Curation** with LLM-as-a-Judge → Kept vs Eliminated.

Key pool sizes (for proportional sampling):

* **Kept:** Prompt–Basic 534, Prompt–Prof 808, Schema–Basic 246, Schema–Prof 378 (total **1,966**)
* **Eliminated:** Prompt–Basic 1,181, Prompt–Prof 911, Schema–Basic 610, Schema–Prof 483 (total **3,185**)

---

## 3. Sample Size Rationale (Practical-min) **with formulas**

### 3.1 Overall estimates (Wald CI for a single proportion)

We use the Wald half-width formula for proportions to set overall sampling targets:

[
E = z_{1-\alpha/2} \cdot \sqrt{ \frac{p(1-p)}{n} } \quad \Rightarrow \quad n = \frac{z_{1-\alpha/2}^2, p(1-p)}{E^2}
]

* Confidence level: (95%) → (z_{1-\alpha/2}=1.96)
* Conservative proportion: (p=0.5) unless domain priors justify otherwise

**Targets used here**

* **Overall Kept precision (pooled):** larger than per-method (n=400) → CI narrower than per-method
* **Eliminated false-negative rate:** (n=100) → CI ≈ ±5–6% (for (p\approx 0.10))

### 3.2 Powered **Method comparison** (two-proportion design)

We power the Prompt vs Schema comparison on the **Kept** items.

**Null vs. alternative**
[
H_0: p_1 = p_2 \quad \text{vs} \quad H_1: p_1 \neq p_2
]

**Two-proportion z-test statistic (for reporting):**
[
\hat p_1 = \tfrac{x_1}{n_1},; \hat p_2 = \tfrac{x_2}{n_2},; \hat p = \tfrac{x_1+x_2}{n_1+n_2}
]
[
Z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p(1-\hat p)\left(\tfrac{1}{n_1}+\tfrac{1}{n_2}\right)}}
]

**Approximate per-arm sample size for detecting a difference (\Delta=|p_1-p_2|)** at 2-sided (\alpha) and power (1-\beta):
[
\boxed{; n_{\text{per method}} ;\approx; \frac{\left[ z_{1-\alpha/2},\sqrt{2,\bar p(1-\bar p)}; +; z_{1-\beta},\sqrt{p_1(1-p_1) + p_2(1-p_2)}\right]^2}{\Delta^2} ;}
]
with (\bar p=(p_1+p_2)/2). In absence of priors, set (p_1\approx p_2\approx 0.5) for a conservative bound.

**Practical-min choice**

* (n_1=n_2=200) Kept per method → **400 Kept total**
  ≈80% power to detect **(\Delta\approx 12)** percentage points at (\alpha=0.05).
  Per-method Wald CI (worst case (p\approx0.5)): **±6.9%**.

> If you need (\Delta=10) points, increase to ≈293 per method (see Notes).

---

## 4. Composition of the Human Annotation Subset (Proportional)

All allocations are **proportional to the original pools** (not equal quarters).

### 4.1 Kept (for precision & method comparison) — **400 total**

* **Per method totals:** Prompt **200**, Schema **200**
* **Within-method persona splits (proportional):**

  * Prompt: **Basic 80**, **Professional 120**
  * Schema: **Basic 79**, **Professional 121**

#### κ block (triple-labeled, **90** total; proportional to Kept pools)

* Prompt–Basic **25**, Prompt–Prof **37**, Schema–Basic **11**, Schema–Prof **17**

#### Kept singles (400 − 90 = **310**)

* Prompt–Basic **55**, Prompt–Prof **83**  → **Prompt singles 138**
* Schema–Basic **68**, Schema–Prof **104** → **Schema singles 172**

### 4.2 Eliminated (for FNR) — **100 total** (proportional to Eliminated pools)

* Prompt–Basic **37**, Prompt–Prof **29**, Schema–Basic **19**, Schema–Prof **15**

### 4.3 Summary table

| Bucket                          | Unique Items | Prompt–Basic | Prompt–Prof | Schema–Basic | Schema–Prof |
| ------------------------------- | -----------: | -----------: | ----------: | -----------: | ----------: |
| **Kept (κ, triple-labeled)**    |           90 |           25 |          37 |           11 |          17 |
| **Kept (single-labeled)**       |          310 |           55 |          83 |           68 |         104 |
| **Kept Total**                  |          400 |           80 |         120 |           79 |         121 |
| **Eliminated (single-labeled)** |          100 |           37 |          29 |           19 |          15 |
| **Grand Total (Unique)**        |          500 |          117 |         149 |           98 |         136 |

---

## 5. Annotation Load and Logistics

| Component      | Unique Items | Raters / Item | Total Labels | Labels / Annotator* | Time @ 3 min |
| -------------- | -----------: | ------------: | -----------: | ------------------: | -----------: |
| Kept (κ block) |           90 |             3 |          270 |                  90 |        4.5 h |
| Kept (Singles) |          310 |             1 |          310 |             103–104 |   ~5.2–5.2 h |
| Eliminated     |          100 |             1 |          100 |               33–34 |   ~1.7–1.8 h |
| **Totals**     |      **500** |             — |      **680** |         **226–227** |  **≈11.3 h** |

*3 annotators. Each labels all 90 κ items + an equal share of singles.

---

## 6. Step-by-Step Procedure

### Step 1 — Inputs

Use the four curated JSONLs:

* `outputs/qas_baseline_judgement_kept.jsonl`
* `outputs/qas_baseline_judgement_eliminated.jsonl`
* `outputs/qas_from_schema_V2_kept.jsonl`
* `outputs/qas_from_schema_V2_eliminated.jsonl`
  Fields: `Question`, `SourceText`, `TargetText`, `Answer`, `Method`, `Persona`, `ReferenceType`.

### Step 2 — Stratified, **Proportional** Sampling (seed=13)

* **Kept:** sample **200 per method**; split **within-method** across personas proportionally (Prompt: 80/120; Schema: 79/121).
* Inside Kept, reserve **90** items as a **shared κ block** with the counts above; the remainder are **Kept singles** per cell.
* **Eliminated:** sample **100 total**, proportional to Eliminated pools across Method×Persona (37/29/19/15).

### Step 3 — Assignment Generation

```bash
python make_annotation_subsets_per_method.py \
  --schema_kept outputs/qas_from_schema_V2_kept.jsonl \
  --schema_elim outputs/qas_from_schema_V2_eliminated.jsonl \
  --prompt_kept outputs/qas_baseline_judgement_kept.jsonl \
  --prompt_elim outputs/qas_baseline_judgement_eliminated.jsonl \
  --outdir ./annotation_subsets_per_method \
  --seed 13 \
  --plan practical_min
```

Outputs:

* `shared/kept_kappa_shared.jsonl` (90 items with cell tags matching the table)
* `annotators/Annotator_{A,B,C}.jsonl` (balanced singles + all κ items)

### Step 4 — Annotation Interface (consistent form)

1. **Question validity** (well-formed / unrealistic).
2. **Required context** (Source / Target / Both / More info).
3. **Answer quality** (correct & complete / partly correct / incorrect).
4. **Final decision** (Keep / Eliminate).
5. **Comment** (short justification).

### Step 5 — Post-Processing & Analysis

* **Precision (overall & per method):** human Keep / total LLM-Kept.

  * Report **per-method** CIs and a **two-proportion z-test** (Prompt vs Schema).
* **False-Negative Rate (overall):** human Keep / total LLM-Eliminated.
* **Reliability:** Cohen’s κ (pairwise) and Fleiss’ κ on the **90** shared Kept.
* **Gold set:** majority/consensus over Kept.

### Step 6 — Deliverables

| Output                           | Description                            |
| -------------------------------- | -------------------------------------- |
| `annotation_subsets_per_method/` | Stratified, proportional JSONL subsets |
| `Annotator_{A,B,C}.jsonl`        | Individual assignment files            |
| `kept_kappa_shared.jsonl`        | 90 shared Kept for κ                   |
| `precision_FN_kappa_report.csv`  | Overall & per-method stats + κ         |
| `ObliQA-CrossRef-Gold.jsonl`     | Human-verified gold dataset            |
| `README_AnnotationProtocol.md`   | This documentation                     |

---

## 7. Expected Outcomes (Practical-min)

| Measure                       | Target / Expectation                       | Achieved Through                            |
| ----------------------------- | ------------------------------------------ | ------------------------------------------- |
| **Method comparison**         | ~80% power to detect (\Delta\approx12) pts | 200 Kept per method + two-proportion z-test |
| **Per-method precision CI**   | ≈ ±6.9% (worst-case p≈0.5)                 | n=200 per method                            |
| **Overall Kept precision CI** | tighter than per-method (n=400)            | pooled estimate                             |
| **False-Negative Rate CI**    | ≈ ±5–6%                                    | n=100 Eliminated                            |
| **Inter-annotator κ**         | stable estimate                            | 90 triple-labeled Kept                      |
| **Gold subset size**          | ≈ 400 items (post-consensus)               | majority/consensus on Kept                  |
| **Workload**                  | ≈ 11.3 h / annotator                       | 226–227 labels @ 3 min/item                 |

---

### Notes & Options

* If you later need **(\Delta=10)-point** detection, increase Kept per method toward **~293** (total 586) and κ block to 120; per-annotator time ≈ **15.8 h**.
* All splits above are **proportional** to preserve the original Method×Persona composition.
