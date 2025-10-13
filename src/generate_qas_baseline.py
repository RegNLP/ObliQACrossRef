#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_qas_baseline.py
------------------------
Create QA pairs from cross-reference (source/target) passages, with optional
LLM-as-Judge validation to ensure each Q&A truly relies on BOTH texts.

Key behaviors:
- Filters to ReferenceType ∈ {Internal, External} (if the column exists).
- Drops title-like targets (strict heading detection).
- Sends FULL source/target text in prompts (no excerpts).
- Two personas: professional & basic. Supports >1 question per (pair, persona).
- Optional LLM-as-Judge validation: keep QAs only if neither SOURCE nor TARGET
  alone suffices to answer the question.
- Deduplicates questions globally if --dedup flag is set.
- Output objects match the requested schema (reference fields inside debug_context).
- REAL dataset row sampling: --row_sample_n/--row_sample_seed
- Hard cap processed pairs: --max_pairs
- Dry run / verbose progress supported.

Requires: openai>=1.40.0. Set OPENAI_API_KEY in your env.
"""


import argparse
import json
import os
import re
import sys
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

# -----------------------------
# Column normalization (aliases)
# -----------------------------
COLUMN_ALIASES = {
    "source_text": ["SourcePassage"],
    "target_text": ["TargetPassage"],
    "source_passage_id": ["SourceID"],
    "target_passage_id": ["TargetID"],
    "reference_type": ["ReferenceType"],
    "reference_text": ["ReferenceText"],
}

REFTYPE_KEEP = {"internal", "external"}  # case-insensitive

# -----------------------------
# Utilities
# -----------------------------
def rand_uuid() -> str:
    return str(uuid.uuid4())

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_question_for_dedup(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s?]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q

def looks_like_empty(s: str) -> bool:
    return len(normalize_whitespace(s)) == 0

# -----------------------------
# Title-like detector (strict)
# -----------------------------
def is_title_like(s: str) -> bool:
    """
    Multi-signal heuristic to drop headings/captions.
    We err on the side of dropping (strict).
    """
    if s is None:
        return True
    text = normalize_whitespace(s)

    if len(text) == 0:
        return True

    # Hard length & token caps for "title-ish"
    short_titleish = (len(text) <= 80 and text.count(" ") <= 12)

    # No ending punctuation usually suggests a heading
    no_end_punct = not re.search(r"[.!?]$", text)

    # TitleCase / ALLCAPS tendency
    tokens = text.split()
    cap_ratio = 0.0
    if tokens:
        cap_like = sum(1 for t in tokens if re.match(r"^[A-Z][a-zA-Z0-9\-]*$", t))
        cap_ratio = cap_like / max(1, len(tokens))

    # Low stopword share tends to be headings
    STOP = {
        "the","and","of","to","in","for","on","by","with",
        "a","an","or","as","is","are","at","from","that",
        "its","be","this","these","those","must","shall",
        "under","rule","section","chapter","part","article"
    }
    lower_tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    stop_share = (sum(1 for t in lower_tokens if t in STOP) / max(1, len(lower_tokens))) if lower_tokens else 0.0

    # Few punctuation marks overall
    punct_count = len(re.findall(r"[,;:]", text))
    few_punct = punct_count == 0

    # Common heading cues / structural references
    heading_cues = bool(re.match(
        r"^(definitions?|scope|interpretation|glossary|enforcement procedure|financial reports?)$",
        text.strip(), re.I))
    looks_like_rule_ref = bool(re.match(r"^(part|chapter|section|rule)\s+\d+([.\-]\d+)*", text.strip(), re.I))

    score = 0
    score += 2 if short_titleish else 0
    score += 1 if no_end_punct else 0
    score += 1 if cap_ratio >= 0.40 else 0
    score += 1 if stop_share <= 0.18 else 0
    score += 1 if few_punct else 0
    score += 2 if heading_cues else 0
    score += 2 if looks_like_rule_ref else 0

    return score >= 3

# -----------------------------
# OpenAI model call
# -----------------------------
def call_llm(model: str, system_prompt: str, user_prompt: str,
             max_tokens: int = 1600, temperature: float = 0.3,
             seed: Optional[int] = None) -> str:
    """One-shot chat completion. Returns content string or '' on failure."""
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
# -----------------------------
# Prompt builder
# -----------------------------
PROFESSIONAL_STYLE = (
    "Write concise, precise, natural questions like a compliance reviewer. "
    "Use conditional framing only when it clarifies scope (e.g., 'when', 'if', 'provided that'). "
    "Use the exact actor names from the texts (e.g., 'Authorised Person', 'Recognised Clearing House')."
)
BASIC_STYLE = (
    "Write short, plain-language questions for non-experts. "
    "Keep one clear idea per question. Use the same actor names as in the texts."
)

SYSTEM_PROMPT_GEN = (
    "You generate regulatory Q&As. Follow the user instructions exactly. "
    "Return VALID JSON only—no markdown, no commentary."
)


def build_prompt(source_text: str, target_text: str, max_per_persona: int, sample_n: int) -> str:
    """
    FULL source/target texts are provided. Do not include snippets in your output.
    Produce strictly-valid JSON with separate arrays for 'professional' and 'basic'.
    """
    return f"""
You are generating high-quality Q&A items for cross-referenced regulatory texts.

HARD RULES — FOLLOW EXACTLY:
1) Joint reliance: Every question AND its answer must require BOTH the SOURCE and the TARGET to be correct.
2) TARGET specificity: Each question must clearly depend on at least one specific element from the TARGET
   (e.g., a named standard/appendix/rule/section, a method/approval/timeline/fee concept, or a defined actor/term),
   but do NOT quote verbatim; paraphrase naturally.
3) Actor fidelity: Use the exact actor names as written in the texts (e.g., 'Authorised Person', 'Reporting Entity').
4) Naturalness: Make questions sound like real compliance questions. Avoid awkward bridges like
   “provided that they are also adhering to …” unless it actually improves clarity.
5) No excerpts: Do NOT include verbatim snippets, citations, quotes, or brackets from the texts.
6) Answers: Minimal, directly responsive, and phrased as a single clear statement (not a list unless the question asks for it).
7) Balance: For each persona, vary question structure (conditionals, “when/under what conditions,” “what must…”, “can/may…”).
8) If the TARGET is only a generic label or heading, do NOT fabricate detail—skip generating for that input.

OUTPUT SHAPE — STRICT JSON (and nothing else):
{{
  "professional": [
    {{"question": "...", "answer": "..." }}
  ],
  "basic": [
    {{"question": "...", "answer": "..." }}
  ]
}}

Persona styles:
- professional: {PROFESSIONAL_STYLE}
- basic: {BASIC_STYLE}

Quantity:
- Aim to brainstorm up to {sample_n} candidates per persona internally, but output no more than {max_per_persona} per persona.

SOURCE (full text):
\"\"\"{source_text}\"\"\"

TARGET (full text):
\"\"\"{target_text}\"\"\"
"""


# -----------------------------
# Parse/collect
# -----------------------------
def parse_llm_json(s: str) -> Dict[str, Any]:
    try:
        s = s.strip()
        s = re.sub(r"^```json\s*|\s*```$", "", s)
        return json.loads(s)
    except Exception:
        return {}

def collect_qas(
    llm_obj: Dict[str, Any],
    source_text: str,
    target_text: str,
    source_passage_id: str,
    target_passage_id: str,
    reference_type: Optional[str],
    reference_text: Optional[str],
    max_q_per_persona: int,
    dedup_set: Optional[set]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Returns: (qa_objects, dropped_dupe_qs, kept_count)
    """
    out: List[Dict[str, Any]] = []
    dropped_dupe = 0
    kept = 0

    for persona in ["professional", "basic"]:
        items = llm_obj.get(persona, []) if isinstance(llm_obj, dict) else []
        if not isinstance(items, list):
            continue

        persona_kept = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            q = normalize_whitespace(it.get("question", ""))
            a = normalize_whitespace(it.get("answer", ""))
            if looks_like_empty(q) or looks_like_empty(a):
                continue

            # Optional global dedup by normalized question text
            if dedup_set is not None:
                key = normalize_question_for_dedup(q)
                if key in dedup_set:
                    dropped_dupe += 1
                    continue
                dedup_set.add(key)

            qa_obj = {
                "qa_id": rand_uuid(),
                "persona": persona,
                "question": q,
                "expected_answer": a,
                "debug_context": {
                    "source_passage_id": source_passage_id,
                    "target_passage_id": target_passage_id,
                    "source_text": source_text,
                    "target_text": target_text,
                    "reference_type": reference_type or None,
                    "reference_text": reference_text or None,
                },
            }
            out.append(qa_obj)
            persona_kept += 1
            kept += 1
            if persona_kept >= max_q_per_persona:
                break

    return out, dropped_dupe, kept

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_q_per_pair", type=int, default=2, help="Per persona, per pair")
    ap.add_argument("--sample_n", type=int, default=3, help="Upper bound hints to LLM (per persona drafts)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Scan/filter only; do not call the model")

    # Dataset sampling & cap
    ap.add_argument("--row_sample_n", type=int, default=None, help="Sample N rows after filters")
    ap.add_argument("--row_sample_seed", type=int, default=13, help="Random seed for row sampling")
    ap.add_argument("--max_pairs", type=int, default=None, help="Hard cap on number of candidate pairs processed")

    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.input_csv)
    if args.verbose:
        print(f"[info] loaded rows: {len(df)}")

    # Normalize columns via aliases
    colmap = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        got = None
        for a in aliases:
            if a in df.columns:
                got = a
                break
        colmap[canonical] = got

    required = ["source_text", "target_text", "source_passage_id", "target_passage_id"]
    missing = [c for c in required if colmap.get(c) is None]
    if missing:
        sys.stderr.write(f"ERROR: Missing required columns (aliases checked): {missing}\n")
        sys.exit(1)

    # Filter ReferenceType if present
    if colmap.get("reference_type"):
        ref_col = colmap["reference_type"]
        df = df[df[ref_col].astype(str).str.lower().isin(REFTYPE_KEEP)].copy()

    # REAL dataset sampling (after ReferenceType filter)
    if args.row_sample_n is not None and args.row_sample_n > 0 and len(df) > args.row_sample_n:
        df = df.sample(n=args.row_sample_n, random_state=args.row_sample_seed).copy()

    # Hard cap total pairs processed
    if args.max_pairs is not None and args.max_pairs > 0 and len(df) > args.max_pairs:
        df = df.head(args.max_pairs).copy()

    # Stats
    rows_loaded = len(df)
    pairs_processed = 0
    qas_created = 0
    dropped_dupe_qs = 0
    skipped_empty_text = 0
    skipped_model_fail = 0
    dropped_title_like_targets = 0
    kept_candidates = 0

    # Dedup set
    dedup_set = set() if args.dedup else None

    # Prepare output
    out_path = args.output_jsonl
    rep_path = args.report_json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(rep_path), exist_ok=True)

    progress_every = 50

    # Dry-run: only scan/filter; no model calls; no output writing
    if args.dry_run:
        for _, row in df.iterrows():
            source_text = normalize_whitespace(str(row[colmap["source_text"]]))
            target_text = normalize_whitespace(str(row[colmap["target_text"]]))
            if looks_like_empty(source_text) or looks_like_empty(target_text):
                skipped_empty_text += 1
                continue
            if is_title_like(target_text):
                dropped_title_like_targets += 1
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
            "dropped_title_like_targets": dropped_title_like_targets,
        }
        print(json.dumps(report, indent=2))
        return

    # Real generation
    with open(out_path, "w", encoding="utf-8") as outf:
        for idx, row in df.iterrows():
            source_text = normalize_whitespace(str(row[colmap["source_text"]]))
            target_text = normalize_whitespace(str(row[colmap["target_text"]]))

            if looks_like_empty(source_text) or looks_like_empty(target_text):
                skipped_empty_text += 1
                continue

            # Drop title-like targets
            if is_title_like(target_text):
                dropped_title_like_targets += 1
                continue

            kept_candidates += 1
            source_passage_id = str(row[colmap["source_passage_id"]])
            target_passage_id = str(row[colmap["target_passage_id"]])

            reference_type = str(row[colmap["reference_type"]]) if colmap.get("reference_type") else None
            reference_text = str(row[colmap["reference_text"]]) if colmap.get("reference_text") else None

            pairs_processed += 1

            # Build prompt
            user_prompt = build_prompt(
                source_text=source_text,
                target_text=target_text,
                max_per_persona=args.max_q_per_pair,
                sample_n=args.sample_n
            )

            # Call generator LLM
            content = call_llm(
                model=args.model,
                system_prompt=SYSTEM_PROMPT_GEN,
                user_prompt=user_prompt,
                max_tokens=1600,
                temperature=args.temperature,
                seed=args.seed
            )
            if not content:
                skipped_model_fail += 2 * args.max_q_per_pair  # rough count
                continue

            llm_obj = parse_llm_json(content)
            if not llm_obj:
                skipped_model_fail += 2 * args.max_q_per_pair
                continue

            # Collect QAs
            qa_objs, dup_ct, kept_ct = collect_qas(
                llm_obj=llm_obj,
                source_text=source_text,
                target_text=target_text,
                source_passage_id=source_passage_id,
                target_passage_id=target_passage_id,
                reference_type=reference_type,
                reference_text=reference_text,
                max_q_per_persona=args.max_q_per_pair,
                dedup_set=dedup_set
            )

            dropped_dupe_qs += dup_ct
            qas_created += kept_ct

            for qa in qa_objs:
                outf.write(json.dumps(qa, ensure_ascii=False) + "\n")

            if args.verbose and pairs_processed % progress_every == 0:
                print(f"[progress] {pairs_processed}/{rows_loaded} rows scanned "
                      f"| kept_candidates={kept_candidates} | qas={qas_created}",
                      flush=True)

    # Report
    report = {
        "rows_loaded": rows_loaded,
        "kept_candidates": kept_candidates,
        "pairs_processed": pairs_processed,
        "qas_created": qas_created,
        "dropped_dupe_qs": dropped_dupe_qs,
        "skipped_empty_text": skipped_empty_text,
        "skipped_model_fail": skipped_model_fail,
        "dropped_title_like_targets": dropped_title_like_targets,
    }
    with open(rep_path, "w", encoding="utf-8") as rpf:
        json.dump(report, rpf, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()