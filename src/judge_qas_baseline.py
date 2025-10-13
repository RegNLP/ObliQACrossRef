#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
judge_qas_baseline.py
------------
Judges QA pair candidates using a Google Gemini model (LLM-as-a-Judge).

Reads a JSONL of QA candidates, evaluates joint reliance on BOTH source & target,
and writes kept/eliminated JSONLs plus a summary report.

Outputs:
- kept_jsonl:       QA pairs that passed validation.
- eliminated_jsonl: Failed QA pairs with judge verdict + rejection_reason.
- report_json:      Summary with per-persona and per-reason counts.

Requires: google-generativeai  (GOOGLE_API_KEY must be set)
"""

import argparse
import json
import os
import sys
import time
from typing import Tuple, Dict, Any
from collections import Counter, defaultdict

import google.generativeai as genai

# -----------------------------
# Gemini model call with retry
# -----------------------------
def call_gemini(model_name: str, prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    """
    Calls Gemini with simple exponential backoff.
    Returns model text or '' on failure.
    """
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as e:
            sys.stderr.write(f"\n[Gemini ERROR] attempt {attempt+1}/{max_retries}: {e}\n")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                return ""
    return ""

# -----------------------------
# Judge prompt
# -----------------------------
def build_judge_prompt(source_text: str, target_text: str, question: str, answer: str) -> str:
    return f"""
You are an expert auditor determining if a Q&A pair requires information from two separate texts.

SOURCE TEXT:
\"\"\"{source_text}\"\"\"

TARGET TEXT:
\"\"\"{target_text}\"\"\"

QUESTION: "{question}"
ANSWER: "{answer}"

First, perform a rigorous internal analysis to determine if BOTH texts are essential.
1) Can the question be answered completely using ONLY the Source Text?
2) Can the question be answered completely using ONLY the Target Text?
3) Is the provided ANSWER factually correct given the combined information?

Final output must be a single line starting with either:
- Yes
- No (very short reason)

Examples:
Yes
No (answer is entirely in the source text)
No (question only relates to the target)
No (answer is incorrect)
"""

# -----------------------------
# Verdict parsing & bucketing
# -----------------------------
def parse_verdict(verdict: str) -> Tuple[bool, str, str]:
    """
    Returns (keep_bool, normalized_verdict, rejection_reason).
    rejection_reason ∈ { '', 'source_only', 'target_only', 'both_suffice',
                         'incorrect_answer', 'malformed', 'api_fail', 'other' }
    """
    v = (verdict or "").strip()
    if not v:
        return (False, "No (judge LLM call failed or returned empty)", "api_fail")

    v_lower = v.lower()

    if v_lower.startswith("yes"):
        return (True, "Yes", "")

    if not v_lower.startswith("no"):
        return (False, f"No (malformed verdict: '{v}')", "malformed")

    # Try to map a short reason bucket
    reason = "other"
    if "source" in v_lower and ("only" in v_lower or "entirely" in v_lower):
        reason = "source_only"
    elif "target" in v_lower and ("only" in v_lower or "entirely" in v_lower):
        reason = "target_only"
    elif "both" in v_lower and ("suffice" in v_lower or "sufficient" in v_lower):
        reason = "both_suffice"
    elif "incorrect" in v_lower or "wrong" in v_lower or "hallucin" in v_lower:
        reason = "incorrect_answer"

    return (False, v, reason)

# -----------------------------
# Judge a single QA
# -----------------------------
def judge_qapair(model: str, qa_obj: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Returns (keep, normalized_verdict, rejection_reason)
    """
    ctx = qa_obj.get("debug_context", {})
    src = ctx.get("source_text")
    tgt = ctx.get("target_text")
    if not src or not tgt:
        return (False, "No (missing source/target in debug_context)", "malformed")

    prompt = build_judge_prompt(src, tgt, qa_obj.get("question", ""), qa_obj.get("expected_answer", ""))
    verdict = call_gemini(model, prompt)

    keep, verdict_norm, reason = parse_verdict(verdict)
    return keep, verdict_norm, reason

# -----------------------------
# Count lines for progress
# -----------------------------
def count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Judge QA candidates with Gemini (LLM-as-Judge)."
    )
    ap.add_argument("--input_jsonl", required=True, help="Path to QA candidates JSONL.")
    ap.add_argument("--kept_jsonl", required=True, help="Output JSONL for kept QAs.")
    ap.add_argument("--eliminated_jsonl", required=True, help="Output JSONL for eliminated QAs.")
    ap.add_argument("--report_json", required=True, help="Summary JSON.")
    ap.add_argument("--model", required=True, help="Gemini model (e.g. 'gemini-1.5-pro-latest').")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--rate_delay", type=float, default=0.0, help="Optional sleep between requests (seconds).")
    args = ap.parse_args()

    # Ensure outputs
    for p in [args.kept_jsonl, args.eliminated_jsonl, args.report_json]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # Configure Gemini
    if not os.getenv("GOOGLE_API_KEY"):
        sys.stderr.write("ERROR: GOOGLE_API_KEY not set.\n")
        sys.exit(1)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    total_candidates = count_lines(args.input_jsonl) if args.verbose else 0
    if args.verbose:
        print(f"Starting judgment with model: {args.model}")
        print(f"Found {total_candidates} candidates.")

    total_judged = 0
    total_kept = 0
    total_eliminated = 0

    # analytics
    per_persona = Counter()
    per_persona_kept = Counter()
    per_reason = Counter()
    persona_reason = defaultdict(Counter)

    start = time.time()

    with open(args.input_jsonl, "r", encoding="utf-8") as inf, \
         open(args.kept_jsonl, "w", encoding="utf-8") as keptf, \
         open(args.eliminated_jsonl, "w", encoding="utf-8") as elimf:

        for i, line in enumerate(inf, start=1):
            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                total_eliminated += 1
                per_reason["malformed"] += 1
                continue

            persona = qa.get("persona", "unknown")
            per_persona[persona] += 1

            keep, verdict_norm, reason = judge_qapair(args.model, qa)
            total_judged += 1

            if keep:
                total_kept += 1
                per_persona_kept[persona] += 1
                keptf.write(json.dumps(qa, ensure_ascii=False) + "\n")
            else:
                total_eliminated += 1
                per_reason[reason] += 1
                persona_reason[persona][reason] += 1
                qa["judge_verdict"] = verdict_norm
                qa["rejection_reason"] = reason
                elimf.write(json.dumps(qa, ensure_ascii=False) + "\n")

            if args.rate_delay > 0:
                time.sleep(args.rate_delay)

            if args.verbose and total_candidates:
                pct = 100.0 * i / total_candidates
                print(f"\r[Progress] {pct:5.1f}% ({i}/{total_candidates}) | "
                      f"Kept: {total_kept} | Eliminated: {total_eliminated}", end="", flush=True)

    if args.verbose:
        print()

    dur = round(time.time() - start, 2)
    # Build persona table
    persona_stats = {
        p: {
            "count": per_persona[p],
            "kept": per_persona_kept[p],
            "keep_rate": round((per_persona_kept[p] / per_persona[p] * 100), 2) if per_persona[p] else 0.0,
            "rejections_by_reason": dict(persona_reason[p]),
        }
        for p in sorted(per_persona.keys())
    }

    report = {
        "judge_model": args.model,
        "total_candidates_judged": total_judged,
        "total_kept": total_kept,
        "total_eliminated": total_eliminated,
        "pass_rate_pct": round((total_kept / total_judged * 100), 2) if total_judged else 0.0,
        "duration_seconds": dur,
        "per_reason_counts": dict(per_reason),
        "per_persona": persona_stats,
    }

    with open(args.report_json, "w", encoding="utf-8") as rpf:
        json.dump(report, rpf, indent=2, ensure_ascii=False)

    print("\n--- Judging Complete ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
