#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_generate_qas_from_schema.py
------------------------------
Create QA pairs from *extracted schema* JSONL (output of 01_extract_schemas.py).

Requested behavior:
- INPUT: JSONL items with fields produced by Step-01 (merged pairs).
- Do NOT run title detection again; use the item's 'target_is_title' flag.
- Use 'semantic_hook' and ITEM_TYPES to anchor question focus & phrasing.
  * If target_item_type == "Other": do NOT constrain phrasing—leave it free.
  * Else: provide phrasing guidance based on the item type (see Prompting Strategy).
- Respect 'answer_spans' semantics:
  * If any span type ∈ {DURATION, DATE, MONEY, PERCENT, TERM, SECTION}, the expected answer
    MUST explicitly include those details (value/term/section label).
  * If only FREEFORM or spans are missing, produce a correct answer without forced slot copying.
- Output JSON lines with extra debug_context fields:
    semantic_hook, citation_hook, answer_spans, source_item_type, target_item_type

Requires: openai>=1.40.0 (ENV: OPENAI_API_KEY)


PROMPTING STRATEGY (detailed)
=============================

Overview
--------
We use a short, strict **system prompt** for format discipline and a structured **user prompt**
that embeds rules, persona styles, anchors (schema fields), and full SOURCE/TARGET texts.
The goal is to generate questions that *require both passages*, are centered on the
semantic_hook, and surface structured details when span types require it.

System Prompt (constant)
------------------------
"You generate regulatory Q&As. Output VALID JSON only—no markdown, no commentary."

User Prompt Structure
---------------------
1) **Rules block** (backbone constraints):
   - **Joint reliance**: Every question AND answer must need BOTH SOURCE and TARGET.
   - **Semantic anchor**: Center the question on `semantic_hook` substance (policy/action/actor/condition).
     Paraphrase; **do not** quote.
   - **Actor fidelity**: Use actor names exactly as written (e.g., "Authorised Person").
   - **No verbatim quotes/citations** inside Q/A (avoid pasting, no brackets).
   - **Answer minimalism**: One concise, directly responsive statement (short paragraph at most).

2) **ITEM_TYPES → Question phrasing guidance** (applied only when `target_item_type != "Other"`):
   - Obligation  → prefer "must/shall" formulations; focus required actions/deadlines.
   - Prohibition → prefer "must not/shall not/is prohibited"; highlight forbidden cases/exceptions.
   - Permission  → prefer "may/can/is permitted" with qualifying conditions.
   - Definition  → define precisely; anchor in term criteria/triggers; avoid copying long text.
   - Scope       → who/what/when applies or is excluded; emphasize applicability boundaries.
   - Procedure   → steps/approvals/calculations/sequence; keep minimal and clear.
   - Other       → **no guidance added** (leave phrasing free to the model).

   Rationale: The answer anchor is in TARGET; phrasing bias reflects TARGET type.
   Optionally mention SOURCE type as context (policy/authority/conditions).

3) **SPAN_TYPES → Answer-content constraints**:
   - If any span type in {DURATION, DATE, MONEY, PERCENT, TERM, SECTION}:
     *The expected answer MUST explicitly include those concrete details*
     (exact value, date, percentage, money amount, named term, or section label).
   - If only FREEFORM spans: Provide a correct, minimal answer; do not copy large chunks.
   - If no spans: Provide a correct, minimal answer without forced slot copying.

4) **Persona styles**:
   - **professional**: concise, precise compliance tone; natural "what/when/must/may" constructions.
   - **basic**: short, plain-language phrasing for non-experts; one clear idea per question.
   The user prompt includes both styles and requests separate arrays in the JSON output.

5) **Quantity control**:
   - "Brainstorm up to N internally, but OUTPUT ≤ max_per_persona per persona."
     This keeps diversity while limiting final items.

6) **Anchors (schema fields)**:
   - `semantic_hook`: the topical gravity; paraphrase, do not quote.
   - `citation_hook`: remind not to quote it; conceptually informs the linkage.
   - `source_item_type` / `target_item_type`: shown explicitly to drive phrasing rule.
   - `answer_spans`: included as compact JSON for the model to see span texts/types.

7) **Grounding**:
   - Full **SOURCE** and **TARGET** texts are included to prevent hallucinations and ensure both are needed.
   - We do not provide excerpts in the output; only Q/A pairs.

8) **Output contract**:
   - Strict JSON only, matching:
     {
       "professional": [{"question":"...","answer":"..."}],
       "basic": [{"question":"...","answer":"..."}]
     }

Recommended Inference Params
----------------------------
- temperature: 0.3  (low variance, stable phrasing)
- max_tokens: ~1600 (room for two personas)
- seed: settable for reproducibility
- dedup: post-hoc normalized-question dedup to reduce repeats

Why this works
--------------
- **Faithfulness**: Full texts + hook centering → on-topic and grounded.
- **Evaluability**: SPAN_TYPES enforce surfacing critical details when present.
- **Domain tone**: ITEM_TYPES steer question intent (except "Other" left free).
- **Clean format**: System prompt + explicit JSON schema reduce formatting errors.

Mini Examples (intent only)
---------------------------
- TARGET=Obligation + span=DURATION("within 10 business days"):
  Q: "When must the Authorised Person notify the Regulator if it cannot meet the requirement, and what timeline applies?"
  A: "It must notify immediately and follow up within 10 business days."

- TARGET=Definition + span=TERM("Client Money"):
  Q: "Under these rules, what exactly counts as Client Money when funds are held with a third-party bank?"
  A: "Client Money is money received or held by the firm for or on behalf of a client, including funds held with a third-party bank."

- TARGET=Procedure + spans=FREEFORM only:
  Q: "What steps must the Recognised Clearing House follow before suspending a member?"
  A: "It must issue notice, state reasons, allow the member an opportunity to respond, then decide and record the decision."
"""


import argparse
import json
import os
import re
import sys
import uuid
from typing import Any, Dict, List, Optional

# -----------------------------
# Constants
# -----------------------------
STRUCTURED_SPAN_TYPES = {"DURATION", "DATE", "MONEY", "PERCENT", "TERM", "SECTION"}
ITEM_TYPES = {"Obligation", "Prohibition", "Permission", "Definition", "Scope", "Procedure", "Other"}

# Style hints for item types (omit for "Other")
ITEM_STYLE_HINTS = {
    "Obligation":  "Prefer 'must/shall' formulations; focus on required actions or deadlines.",
    "Prohibition": "Prefer 'must not/shall not/is prohibited'; clarify forbidden cases or exceptions.",
    "Permission":  "Prefer 'may/can/is permitted' with qualifying conditions.",
    "Definition":  "Define precisely; anchor in term criteria or triggers; avoid copying long text.",
    "Scope":       "Ask who/what/when applies or is excluded; emphasize applicability boundaries.",
    "Procedure":   "Ask about steps, approvals, calculations, or sequencing; keep them minimal.",
    # "Other": intentionally absent → leave phrasing free to the model
}

# -----------------------------
# Utilities
# -----------------------------
def rand_uuid() -> str:
    return str(uuid.uuid4())

def norm_ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def looks_like_empty(s: Optional[str]) -> bool:
    return len(norm_ws(s)) == 0

def normalize_question_for_dedup(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s?]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q

# -----------------------------
# OpenAI call
# -----------------------------
def call_llm(model: str, system_prompt: str, user_prompt: str,
             temperature: float = 0.3, max_tokens: int = 1600,
             seed: Optional[int] = None) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        extra = {}
        if seed is not None:
            extra["seed"] = seed
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        sys.stderr.write(f"[LLM ERROR] {e}\n")
        return ""

def parse_llm_json(s: str) -> Dict[str, Any]:
    try:
        s = s.strip()
        s = re.sub(r"^```json\s*|\s*```$", "", s)
        return json.loads(s)
    except Exception:
        return {}

# -----------------------------
# Prompting
# -----------------------------
PROFESSIONAL_STYLE = (
    "Write concise, precise, natural questions like a compliance reviewer. "
    "Prefer 'what/when/under what conditions/may/can/must' constructions. "
    "Use actor names exactly as in the passages (e.g., 'Authorised Person')."
)
BASIC_STYLE = (
    "Write short, plain-language questions for non-experts. "
    "Keep one clear idea per question. Use the same actor names as in the texts."
)

SYSTEM_PROMPT_GEN = (
    "You generate regulatory Q&As. Output VALID JSON only—no markdown, no commentary."
)

def build_user_prompt(
    source_text: str,
    target_text: str,
    semantic_hook: str,
    citation_hook: str,
    source_item_type: str,
    target_item_type: str,
    answer_spans: List[Dict[str, Any]],
    max_per_persona: int,
    sample_n: int
) -> str:
    """
    Guide the model using schema fields. Enforce span semantics. Add item-type phrasing hints
    (omit when target_item_type == 'Other'). Incorporate baseline improvements:
    - TARGET specificity (with concrete examples)
    - Actor fidelity (exact actor names)
    - Naturalness (avoid awkward bridges)
    - Balance (vary question structures per persona)
    """
    # Span rules
    has_structured = any((sp.get("type") in STRUCTURED_SPAN_TYPES) for sp in (answer_spans or []))
    has_any_spans = len(answer_spans or []) > 0
    has_only_freeform = (has_any_spans and not has_structured)

    rules = []
    # From schema version
    rules.append("Every Q&A must require BOTH the SOURCE and the TARGET to answer correctly.")
    rules.append("Center the question on the semantic_hook’s substance (policy/action/actor/condition). Paraphrase; do NOT quote.")
    rules.append("Do NOT include verbatim quotes or citations; paraphrase naturally.")
    rules.append("Answers must be minimal, directly responsive single statements (single paragraph unless the question asks otherwise).")

    # ---- Baseline carry-overs (added) ----
    # TARGET specificity (with examples)
    rules.append(
        "TARGET specificity: Each question must clearly depend on at least one specific element from the TARGET "
        "(e.g., a named standard/appendix/rule/section, a method/approval/timeline/fee concept, or a defined actor/term). "
        "Do not quote verbatim; paraphrase naturally."
    )
    # Actor fidelity
    rules.append("Actor fidelity: Use the exact actor names as written in the texts (e.g., 'Authorised Person', 'Reporting Entity').")
    # Naturalness
    rules.append(
        "Naturalness: Make questions sound like real compliance questions. Avoid awkward bridges like "
        "“provided that they are also adhering to …” unless it truly improves clarity."
    )
    # Balance / variation
    rules.append(
        "Balance: For each persona, vary question structure across items (conditionals; 'when/under what conditions'; "
        "'what must…'; 'can/may…')."
    )
    # --------------------------------------

    # Item-type phrasing guidance (skip if target == Other)
    tgt_type = (target_item_type or "Other")
    src_type = (source_item_type or "Other")
    if tgt_type in ITEM_STYLE_HINTS:
        target_hint = ITEM_STYLE_HINTS[tgt_type]
        src_hint = ITEM_STYLE_HINTS.get(src_type)
        if src_hint:
            rules.append(f"For question form, prioritize TARGET type '{tgt_type}': {target_hint} (SOURCE '{src_type}' context: {src_hint}).")
        else:
            rules.append(f"For question form, prioritize TARGET type '{tgt_type}': {target_hint}.")

    # Span content constraints
    if has_structured:
        rules.append(
          "Structured spans are present (DURATION/DATE/MONEY/PERCENT/TERM/SECTION). "
          "Your expected answer MUST explicitly include those concrete details (exact value or term/section label)."
        )
    elif has_only_freeform:
        rules.append("Spans are FREEFORM only; provide a correct, minimal answer without copying long text.")
    else:
        rules.append("No spans provided; provide a correct, minimal answer without forced slot copying.")

    spans_block = json.dumps(answer_spans or [], ensure_ascii=False)

    return f"""
You are generating Q&As for a cross-referenced regulatory pair.

Follow ALL rules:
- {chr(10)}- ".join(rules)

Persona styles:
- professional: {PROFESSIONAL_STYLE}
- basic: {BASIC_STYLE}

Quantity:
- Brainstorm up to {sample_n} internally, but OUTPUT no more than {max_per_persona} per persona.

ANCHORS:
- semantic_hook (guides the practical substance): "{semantic_hook}"
- citation_hook (do not quote in Q/A; only use its concept as needed): "{citation_hook}"
- SOURCE item type: "{source_item_type}"
- TARGET item type: "{target_item_type}"
- answer_spans (with types): {spans_block}

SOURCE (full text):
\"\"\"{source_text}\"\"\"

TARGET (full text):
\"\"\"{target_text}\"\"\"

OUTPUT — strict JSON and nothing else:
{{
  "professional": [
    {{"question": "...", "answer": "..." }}
  ],
  "basic": [
    {{"question": "...", "answer": "..." }}
  ]
}}
""".strip()

# -----------------------------
# IO helpers (JSONL)
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Step-01 output JSONL (merged items)")
    ap.add_argument("--output_jsonl", required=True, help="QA JSONL output")
    ap.add_argument("--report_json", required=True, help="Summary report path")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_q_per_pair", type=int, default=2, help="Per persona, per pair (upper bound)")
    ap.add_argument("--sample_n", type=int, default=3, help="Brainstorm hint (per persona) to the LLM")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dedup", action="store_true", help="Global dedup by normalized questions")
    ap.add_argument("--drop_title_targets", action="store_true",
                    help="If set, skip items where target_is_title==True (usually already dropped in Step-01)")
    ap.add_argument("--row_sample_n", type=int, default=None)
    ap.add_argument("--row_sample_seed", type=int, default=13)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Scan/filter only; no model calls or writes")
    args = ap.parse_args()

    items = read_jsonl(args.input_jsonl)
    print(f"[info] loaded merged items: {len(items)} from {args.input_jsonl}", flush=True)
    if not items:
        print(json.dumps({
            "rows_loaded": 0,
            "kept_candidates": 0,
            "pairs_processed": 0,
            "qas_created": 0,
            "dropped_dupe_qs": 0,
            "skipped_empty_text": 0,
            "skipped_model_fail": 0,
            "skipped_title_targets": 0,
            "model": args.model,
            "note": "No items loaded—check input path/JSONL lines."
        }, indent=2), flush=True)
        return

    # Optional sampling
    if args.row_sample_n is not None and args.row_sample_n > 0 and len(items) > args.row_sample_n:
        import random
        rnd = random.Random(args.row_sample_seed)
        items = rnd.sample(items, args.row_sample_n)

    if args.max_pairs is not None and args.max_pairs > 0 and len(items) > args.max_pairs:
        items = items[:args.max_pairs]

    rows_loaded = len(items)
    kept_candidates = 0
    pairs_processed = 0
    qas_created = 0
    dropped_dupe_qs = 0
    skipped_empty_text = 0
    skipped_title_targets = 0
    skipped_model_fail = 0

    dedup_set = set() if args.dedup else None

    # Prepare output dirs
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)

    # Dry run (no LLM calls)
    if args.dry_run:
        for it in items:
            source_text = norm_ws(it.get("source_text"))
            target_text = norm_ws(it.get("target_text"))
            if looks_like_empty(source_text) or looks_like_empty(target_text):
                skipped_empty_text += 1
                continue
            if args.drop_title_targets and bool(it.get("target_is_title")):
                skipped_title_targets += 1
                continue
            kept_candidates += 1
        report = {
            "rows_loaded": rows_loaded,
            "kept_candidates": kept_candidates,
            "pairs_processed": kept_candidates,
            "qas_created": 0,
            "dropped_dupe_qs": 0,
            "skipped_empty_text": skipped_empty_text,
            "skipped_model_fail": 0,
            "skipped_title_targets": skipped_title_targets,
        }
        print(json.dumps(report, indent=2))
        return

    # Real generation
    with open(args.output_jsonl, "w", encoding="utf-8") as outf:
        for idx, it in enumerate(items):
            source_text = norm_ws(it.get("source_text"))
            target_text = norm_ws(it.get("target_text"))
            if looks_like_empty(source_text) or looks_like_empty(target_text):
                skipped_empty_text += 1
                continue

            if args.drop_title_targets and bool(it.get("target_is_title")):
                skipped_title_targets += 1
                continue

            kept_candidates += 1
            pairs_processed += 1

            # Schema anchors
            semantic_hook = norm_ws(it.get("semantic_hook"))
            citation_hook = norm_ws(it.get("citation_hook"))
            source_item_type = (it.get("source_item_type") or "Other")
            target_item_type = (it.get("target_item_type") or "Other")
            answer_spans = it.get("answer_spans") or []

            # Build prompt
            user_prompt = build_user_prompt(
                source_text=source_text,
                target_text=target_text,
                semantic_hook=semantic_hook,
                citation_hook=citation_hook,
                source_item_type=str(source_item_type),
                target_item_type=str(target_item_type),
                answer_spans=answer_spans,
                max_per_persona=args.max_q_per_pair,
                sample_n=args.sample_n
            )

            content = call_llm(
                model=args.model,
                system_prompt=SYSTEM_PROMPT_GEN,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=1600,
                seed=args.seed
            )
            if not content:
                skipped_model_fail += 2 * args.max_q_per_pair
                continue

            llm_obj = parse_llm_json(content)
            if not llm_obj:
                skipped_model_fail += 2 * args.max_q_per_pair
                continue

            # Collect per persona
            for persona in ["professional", "basic"]:
                items_p = llm_obj.get(persona, [])
                if not isinstance(items_p, list):
                    continue

                kept = 0
                for qa in items_p:
                    if not isinstance(qa, dict):
                        continue
                    q = norm_ws(qa.get("question"))
                    a = norm_ws(qa.get("answer"))
                    if looks_like_empty(q) or looks_like_empty(a):
                        continue

                    if dedup_set is not None:
                        key = normalize_question_for_dedup(q)
                        if key in dedup_set:
                            dropped_dupe_qs += 1
                            continue
                        dedup_set.add(key)

                    out = {
                        "qa_id": rand_uuid(),
                        "persona": persona,
                        "question": q,
                        "expected_answer": a,
                        "debug_context": {
                            "source_passage_id": it.get("source_passage_id"),
                            "target_passage_id": it.get("target_passage_id"),
                            "source_text": source_text,
                            "target_text": target_text,
                            "reference_type": it.get("reference_type"),
                            "reference_text": it.get("reference_text"),
                            "semantic_hook": semantic_hook,
                            "citation_hook": citation_hook,
                            "answer_spans": answer_spans,
                            "source_item_type": source_item_type,
                            "target_item_type": target_item_type,
                        },
                    }
                    outf.write(json.dumps(out, ensure_ascii=False) + "\n")
                    qas_created += 1
                    kept += 1
                    if kept >= args.max_q_per_pair:
                        break

            if args.verbose and pairs_processed % 50 == 0:
                print(
                    f"[progress] processed={pairs_processed} | kept_candidates={kept_candidates} | qas={qas_created}",
                    flush=True,
                )

    # Report
    report = {
        "rows_loaded": rows_loaded,
        "kept_candidates": kept_candidates,
        "pairs_processed": pairs_processed,
        "qas_created": qas_created,
        "dropped_dupe_qs": dropped_dupe_qs,
        "skipped_empty_text": skipped_empty_text,
        "skipped_model_fail": skipped_model_fail,
        "skipped_title_targets": skipped_title_targets,
        "model": args.model,
    }
    with open(args.report_json, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
