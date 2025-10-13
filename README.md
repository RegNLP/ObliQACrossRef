# ObliQA-CrossRef Schema Guide (v2 - Merged Pair)

*A compact, practical specification for structuring cross-reference passage pairs to generate 1‑hop, dependency-enforced questions.*

---

## 1) Purpose

This schema captures the relationship between two linked regulatory passages (**source** and **target**) in a single, merged record. It defines the types of clauses, the key phrases (hooks) that link them, and the location of the gold answer within the target text. This structure is designed specifically for the task of generating high-quality, cross-reference-dependent question–answer pairs.

---

## 2) Record Layout

```json
{
  "item_id": "uuid",
  "reference_type": "Internal",
  "reference_text": "4.12.1",
  
  "semantic_hook": "A Clearing Member must at all times comply with",
  "citation_hook": "4.12.1",
  
  "source_passage_id": "uuid",
  "source_text": "A Clearing Member must at all times comply with the requirements of Rule 4.12...",
  "target_passage_id": "uuid",
  "target_text": "4.12.1 The Clearing House will keep separate records...",
  
  "source_item_type": "Obligation|Prohibition|Permission|Definition|Scope|Procedure|Other",
  "target_item_type": "Obligation|Prohibition|Permission|Definition|Scope|Procedure|Other",
  
  "answer_spans": [
    {"text": "separate records", "start": 38, "end": 54, "type": "FREEFORM"}
  ],
  
  "target_is_title": false,
  "provenance": {"model": "gpt-4o", "ts": "iso-timestamp"}
}
```

---

## 3) Field Semantics

### Identity & Text
- **item_id** — Stable UUID for this merged pair record.
- **reference_type** — The type of cross-reference (e.g., *Internal*, *External*), taken from the source data.
- **reference_text** — The literal text of the cross-reference (e.g., `4.12.1`), taken from the source data.
- **source_passage_id** / **target_passage_id** — Canonical IDs for the source and target passages.
- **source_text** / **target_text** — The full raw text of the source and target passages.

### Labels
- **source_item_type** / **target_item_type** (*what each clause is*) — One of: **Obligation** (must do), **Prohibition** (must not do), **Permission** (may do), **Definition**, **Scope** (coverage/applicability), **Procedure** (steps/filings), **Other**.
- **target_is_title** — Boolean flag indicating if the `target_text` is a short, non‑substantive heading.

### Gold Answer (spans in `target_text`)
- **answer_spans** — List of exact substrings within `target_text`.
  - Offsets are **0‑based, half‑open** `[start, end)`.
  - Prefer constant types: **DURATION, DATE, PERCENT, MONEY, TERM, SECTION**; else **FREEFORM**.
  - Multiple spans allowed. If no specific spans are found and the target is not a title, this may contain a single span covering the full `target_text`.

### Hooks (to enforce the 1‑hop dependency)
- **semantic_hook** — Short content phrase taken **verbatim** from `source_text` that captures the conceptual link between the passages.
- **citation_hook** — Literal reference string taken **verbatim** from the `source_text` (e.g., “Rule 3.4.1”, “Section 7.2”).

**Hook style:** Keep each hook concise (~6–10 tokens); **no paraphrase**; preserve punctuation/case as in the source text.

### Provenance
- **provenance** — Minimal trace of how the record was produced (e.g., `{ "model": "gpt-4o", "ts": "2025-10-07T..." }`).

---

## 4) Validation Rules

- **Span grounding** — Every `answer_spans[*].text` must equal `target_text[start:end]`.
- **Offsets** — `0 ≤ start < end ≤ len(target_text)`.
- **Typing** — `source_item_type`, `target_item_type`, and `answer_spans[*].type` must be from the allowed enums.
- **Completeness** — Records must have non‑empty `source_text`, `target_text`, `semantic_hook`, and `citation_hook` fields.
- **No paraphrase** — Hooks and spans must be **verbatim substrings** of their respective source texts.

---

## 5) Minimal Typing Cheat‑Sheet

| Example clause | `item_type` | Example `answer_spans` |
|---|---|---|
| “The Firm must notify the Authority within 10 business days.” | Obligation | `["within 10 business days"] → DURATION` |
| “A person must not hold more than 10% without approval.” | Prohibition | `["10%"] → PERCENT` |
| ““Authorised Person” means …” | Definition | `["Authorised Person"] → TERM` |
| “This Chapter applies to Authorised Persons in Category 3A.” | Scope | `["Authorised Persons in Category 3A"] → FREEFORM` |

---

## 6) Worked Example

*A single, merged record representing a cross-reference pair.*

```json
{
  "item_id": "90e8774e-0b01-4475-b3e5-8f3b25916053",
  "reference_type": "Internal",
  "reference_text": "4.12.1",
  "semantic_hook": "Rule 4.12 (Segregation and portability)",
  "citation_hook": "4.12.1",
  "source_passage_id": "6112097d-0f5d-40e8-9a64-1aa7b6b520a6",
  "source_text": "A Clearing Member must at all times comply with the requirements of Rule 4.12 (Segregation and portability).",
  "target_passage_id": "9ca78053-40eb-4073-b9b8-df05a7009571",
  "target_text": "4.12.1 The Clearing House will keep separate records of the Client Positions and Client Assets of each Clearing Member...",
  "source_item_type": "Obligation",
  "target_item_type": "Procedure",
  "answer_spans": [
    {
      "text": "separate records of the Client Positions and Client Assets",
      "start": 29,
      "end": 92,
      "type": "FREEFORM"
    }
  ],
  "target_is_title": false,
  "provenance": {
    "model": "gpt-4o",
    "ts": "2025-10-06T19:59:17+00:00"
  }
}
```

**Why this enforces 1‑hop:** A question generated from this record will be constructed using the hooks from the `source_text` (e.g., “Regarding Rule 4.12 on segregation and portability ...”). The answer to that question can only be found in the `target_text` (e.g., that the Clearing House “will keep separate records ...”).

---

## 7) Notes for Extraction

- The process reads a row from a source CSV and emits a single, merged JSON object for each valid source–target pair.
- The `semantic_hook` and `citation_hook` are **always** extracted from the `source_text`.
- The `answer_spans` are **always** extracted from the `target_text`.
- Pairs are skipped if `source_text` or `target_text` is missing, or if the `target_text` is flagged as a title.

---

## 8) Validation Enums (reference)

- `item_type ∈ {Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other}`
- `answer_spans[*].type ∈ {DURATION, DATE, MONEY, PERCENT, TERM, SECTION, FREEFORM}`

