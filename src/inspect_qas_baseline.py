#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_qas_baseline.py
-----------------------
Analyze Q&A dataset quality and coverage across kept/rejected splits.

Typical pipeline:
  1) python src/generate_qas_baseline.py \
       --input_csv data/CrossReferenceData.csv \
       --output_jsonl outputs/qas_baseline_generation.jsonl \
       --report_json outputs/qas_baseline_generationreport.json \
       --model gpt-4o --max_q_per_pair 2 --sample_n 3 --dedup --verbose

  2) python judge_qas_baseline.py \
       --input_jsonl outputs/qas_baseline_generation.jsonl \
       --kept_jsonl outputs/qas_baseline_judgement_kept.jsonl \
       --eliminated_jsonl outputs/qas_baseline_judgement_eliminated.jsonl \
       --report_json outputs/qas_baseline_judging_report.json \
       --model gemini-1.5-pro-latest --verbose

Then run this inspector on the kept + eliminated JSONLs.

Inputs:
  --kept_jsonl       Path to kept QAs (from judge step)
  --rejected_jsonl   Path to eliminated QAs (from judge step) [optional but recommended]
  --out_summary_json Path to write overall summary JSON
  --persona_csv      (optional) Per-persona metrics CSV
  --reasons_csv      (optional) Persona x rejection_reason pivot CSV
  --length_bins_csv  (optional) Length histogram CSV (questions/answers)

Assumed JSONL fields (tolerant/robust):
  - persona: "professional" | "basic"
  - question: str
  - expected_answer: str
  - debug_context: { reference_type: "Internal"/"External", ... }
  - For rejected items (Gemini judge):
      * judge_verdict: "Yes" or "No (brief reason)"
    (Older judge format also supported:
      * rejection_reason: str
      * judge: { source_suffices: "Yes"/"No", target_suffices: "Yes"/"No", reason: str })

Metrics produced:
- Coverage & volume:
    total/kept/rejected counts, keep-rate
    per-persona counts + keep-rate
    per-reference_type counts + keep-rate
- Length & shape (kept only):
    word/char stats for Q and A (mean/median/std/p25/p75/min/max)
    sentence counts (avg, etc.)
    answer length buckets (≤10 / 11–25 / 26–50 / >50 words)
    histograms (word-count bins) for Q/A
- Quality & duplicates:
    exact-duplicate question rate within kept (normalized)
- Judge analytics (if rejected_jsonl provided):
    judged_total (= kept + rejected), dropped_by_judge (= rejected),
    counts by rejection_reason,
    persona × rejection_reason pivot,
    per-reason avg Q/A lengths
- Balance:
    internal vs external ratios (from debug_context.reference_type)
- Lexical diversity (type/token ratio) per persona (kept only)

Requires: pandas, numpy
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# ----------------- text utils -----------------
_SENT_SPLIT = re.compile(r"[.!?]+\s+")
_WORDS = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def word_list(s: str) -> List[str]:
    return _WORDS.findall(s or "")

def sent_count(s: str) -> int:
    s = norm_space(s)
    if not s:
        return 0
    return max(1, len(_SENT_SPLIT.split(s)))  # at least 1 if non-empty

def normalize_question_for_dedup(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s?]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q

def ttr_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / max(1, len(words))

# Buckets for answer-length (in words)
def answer_bucket(nwords: int) -> str:
    if nwords <= 10:
        return "≤10"
    if nwords <= 25:
        return "11–25"
    if nwords <= 50:
        return "26–50"
    return ">50"

# ----------------- Gemini verdict parsing -----------------
# Map free-text reasons into buckets we care about.
def bucket_from_verdict(verdict: Optional[str]) -> str:
    """
    Parse judge_verdict like:
      'Yes'
      'No (answer is found entirely in the source text)'
      'No (question only relates to the target text)'
      'No (answer is factually incorrect based on the texts)'
      'No (judge LLM call failed...)'  [if you used such fallback]
    Return canonical buckets: source_suffices / target_suffices / both_suffice / answer_incorrect / malformed / other
    """
    if not verdict:
        return "other"
    v = verdict.strip().lower()
    if v.startswith("yes"):
        # shouldn't appear in eliminated set, but just in case
        return "kept_yes"

    # try to extract the parenthetical reason
    # e.g., "No (reason here)"; if not present, treat generic "no"
    reason = ""
    m = re.search(r"no\s*\((.*?)\)\s*$", v, flags=re.IGNORECASE)
    if m:
        reason = m.group(1)
    else:
        # no parenthetical; use whole string after "no"
        reason = v[2:].strip(" :")

    # heuristic mapping
    if "source" in reason and "target" in reason:
        return "both_suffice"
    if "entirely in the source" in reason or "only in the source" in reason or "source text" in reason:
        return "source_suffices"
    if "entirely in the target" in reason or "only in the target" in reason or "target text" in reason:
        return "target_suffices"
    if "incorrect" in reason or "wrong" in reason or "hallucinat" in reason:
        return "answer_incorrect"
    if "malformed" in reason or "failed" in reason or "bad format" in reason:
        return "malformed"
    return "other"

# ----------------- IO helpers -----------------
def read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

# ----------------- core analysis -----------------
def frame_from_items(items: List[Dict[str, Any]], rejected: bool=False) -> pd.DataFrame:
    rows = []
    for it in items:
        persona = it.get("persona")
        q = norm_space(it.get("question", ""))
        a = norm_space(it.get("expected_answer", ""))
        dbg = it.get("debug_context") or {}
        ref_type = (dbg.get("reference_type") or "").strip() or None

        # judge / rejection (support both formats)
        # new (Gemini): judge_verdict = "Yes" or "No (reason)"
        verdict_raw = it.get("judge_verdict")
        # old: explicit fields
        rej_reason_field = it.get("rejection_reason")
        judge = it.get("judge") or {}
        judge_src = (str(judge.get("source_suffices") or "").strip().lower() or None)
        judge_tgt = (str(judge.get("target_suffices") or "").strip().lower() or None)

        # derive a canonical bucket
        if rej_reason_field:
            rej_bucket = rej_reason_field.strip().lower()
        elif verdict_raw:
            rej_bucket = bucket_from_verdict(verdict_raw)
        else:
            # fallback to older boolean flags if present
            if judge_src == "yes" and judge_tgt == "no":
                rej_bucket = "source_suffices"
            elif judge_src == "no" and judge_tgt == "yes":
                rej_bucket = "target_suffices"
            elif judge_src == "yes" and judge_tgt == "yes":
                rej_bucket = "both_suffice"
            else:
                rej_bucket = "other"

        rows.append({
            "persona": persona,
            "reference_type": (ref_type or "").capitalize() if ref_type else None,  # Internal/External/None
            "question": q,
            "answer": a,
            "q_words": len(word_list(q)),
            "a_words": len(word_list(a)),
            "q_chars": len(q),
            "a_chars": len(a),
            "q_sents": sent_count(q),
            "a_sents": sent_count(a),
            "a_bucket": answer_bucket(len(word_list(a))),
            "is_rejected": rejected,
            "rejection_reason": rej_bucket if rejected else None,
            "verdict_raw": verdict_raw if rejected else None,
            "judge_src": judge_src if rejected else None,
            "judge_tgt": judge_tgt if rejected else None,
        })
    return pd.DataFrame(rows)

def basic_stats(series: pd.Series) -> Dict[str, float]:
    if len(series) == 0:
        return dict(mean=0, median=0, std=0, p25=0, p75=0, min=0, max=0)
    return dict(
        mean=float(series.mean()),
        median=float(series.median()),
        std=float(series.std(ddof=0)),
        p25=float(series.quantile(0.25)),
        p75=float(series.quantile(0.75)),
        min=float(series.min()),
        max=float(series.max()),
    )

def histogram(series: pd.Series, bins: List[int]) -> Dict[str, int]:
    counts = {f"[{bins[i]},{bins[i+1]})": 0 for i in range(len(bins)-1)}
    counts[f"[{bins[-1]},∞)"] = 0
    for v in series.dropna().astype(int):
        placed = False
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                counts[f"[{bins[i]},{bins[i+1]})"] += 1
                placed = True
                break
        if not placed:
            counts[f"[{bins[-1]},∞)"] += 1
    return counts

def exact_dup_rate(df_kept: pd.DataFrame) -> Dict[str, Any]:
    if df_kept.empty:
        return {"duplicate_rate": 0.0, "duplicate_count": 0, "unique_count": 0}
    keys = df_kept["question"].map(normalize_question_for_dedup)
    total = len(keys)
    dup_count = int(total - len(set(keys)))
    rate = dup_count / total if total else 0.0
    return {"duplicate_rate": round(rate, 6), "duplicate_count": dup_count, "unique_count": int(total - dup_count)}

def persona_ttr(df_kept: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for persona, g in df_kept.groupby("persona", dropna=False):
        words = []
        for q, a in zip(g["question"].tolist(), g["answer"].tolist()):
            words.extend(word_list(q))
            words.extend(word_list(a))
        out[persona or "Unknown"] = round(ttr_ratio([w.lower() for w in words]), 6)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept_jsonl", required=True)
    ap.add_argument("--rejected_jsonl", default=None)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--persona_csv", default=None)
    ap.add_argument("--reasons_csv", default=None)
    ap.add_argument("--length_bins_csv", default=None)
    args = ap.parse_args()

    kept_items = read_jsonl(args.kept_jsonl)
    rej_items  = read_jsonl(args.rejected_jsonl) if args.rejected_jsonl else []

    df_kept = frame_from_items(kept_items, rejected=False)
    df_rej  = frame_from_items(rej_items, rejected=True)

    # ------- coverage & volume -------
    total_qas = len(df_kept) + len(df_rej)
    keep_rate = (len(df_kept) / total_qas) * 100.0 if total_qas else 0.0

    # per persona keep-rate
    def _rate(n_keep, n_all): return (n_keep / n_all * 100.0) if n_all else 0.0
    personas = sorted(set(df_kept["persona"].dropna().unique()).union(set(df_rej["persona"].dropna().unique())))
    persona_stats = []
    for persona in personas:
        k = int((df_kept["persona"] == persona).sum())
        r = int((df_rej["persona"] == persona).sum())
        persona_stats.append({
            "persona": persona,
            "kept": k,
            "rejected": r,
            "total": k + r,
            "keep_rate_pct": round(_rate(k, k + r), 2)
        })

    # per reference_type keep-rate (Internal/External/None)
    def _canon(x):
        x = (x or "").strip().capitalize()
        return x if x in {"Internal","External"} else "None"
    if not df_kept.empty:
        df_kept["reference_type_c"] = df_kept["reference_type"].map(_canon)
    if not df_rej.empty:
        df_rej["reference_type_c"]  = df_rej["reference_type"].map(_canon)
    ref_types = ["Internal","External","None"]
    reftype_stats = []
    for rt in ref_types:
        k = int((df_kept.get("reference_type_c", pd.Series(dtype=str)) == rt).sum()) if not df_kept.empty else 0
        r = int((df_rej.get("reference_type_c", pd.Series(dtype=str)) == rt).sum()) if not df_rej.empty else 0
        reftype_stats.append({
            "reference_type": rt,
            "kept": k,
            "rejected": r,
            "total": k + r,
            "keep_rate_pct": round(_rate(k, k + r), 2) if (k + r) else 0.0
        })

    # ------- length & shape (kept only) -------
    q_stats = basic_stats(df_kept["q_words"]) if not df_kept.empty else basic_stats(pd.Series(dtype=int))
    a_stats = basic_stats(df_kept["a_words"]) if not df_kept.empty else basic_stats(pd.Series(dtype=int))
    q_chars = basic_stats(df_kept["q_chars"]) if not df_kept.empty else basic_stats(pd.Series(dtype=int))
    a_chars = basic_stats(df_kept["a_chars"]) if not df_kept.empty else basic_stats(pd.Series(dtype=int))
    q_sents = basic_stats(df_kept["q_sents"]) if not df_kept.empty else basic_stats(pd.Series(dtype=int))
    a_sents = basic_stats(df_kept["a_sents"]) if not df_kept.empty else basic_stats(pd.Series(dtype=int))

    bucket_counts = df_kept["a_bucket"].value_counts().to_dict() if not df_kept.empty else {}

    # histograms (kept)
    q_hist = histogram(df_kept["q_words"], bins=[0,5,10,15,20,30,50]) if not df_kept.empty else {}
    a_hist = histogram(df_kept["a_words"], bins=[0,5,10,15,20,30,50]) if not df_kept.empty else {}

    # ------- quality & duplicates -------
    dup_info = exact_dup_rate(df_kept)

    # ------- judge analytics -------
    judged_total = int(total_qas)  # in the two-step pipeline, judge saw all candidates
    dropped_by_judge = int(len(df_rej))
    kept_after_judge = int(len(df_kept))

    if not df_rej.empty:
        reason_counts = df_rej["rejection_reason"].value_counts().to_dict()

        # persona x reason pivot
        pivot = (
            df_rej
            .pivot_table(index="persona", columns="rejection_reason", values="question", aggfunc="count", fill_value=0)
            .reset_index()
        )
        pivot_persona_reasons = pivot.to_dict(orient="records")

        # per reason avg lengths (Q/A words)
        reason_len = (
            df_rej
            .groupby("rejection_reason")[["q_words","a_words"]]
            .mean()
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        reason_counts = {}
        pivot_persona_reasons = []
        reason_len = []

    # ------- balance & diversity -------
    ttr_by_persona = persona_ttr(df_kept)

    # ------- optional CSVs -------
    if args.persona_csv:
        pd.DataFrame(persona_stats).to_csv(args.persona_csv, index=False)
    if args.reasons_csv and not df_rej.empty:
        wide = (
            df_rej
            .pivot_table(index="persona", columns="rejection_reason", values="question", aggfunc="count", fill_value=0)
            .reset_index()
        )
        wide.to_csv(args.reasons_csv, index=False)
    if args.length_bins_csv:
        pd.DataFrame([
            {"metric": "q_words_hist", **q_hist},
            {"metric": "a_words_hist", **a_hist},
        ]).to_csv(args.length_bins_csv, index=False)

    # ------- summary JSON -------
    summary = {
        "totals": {
            "total_qas": int(total_qas),
            "kept": int(len(df_kept)),
            "rejected": int(len(df_rej)),
            "keep_rate_pct": round(keep_rate, 2),
        },
        "per_persona": persona_stats,
        "per_reference_type": reftype_stats,
        "length_shape": {
            "question_words": q_stats,
            "answer_words": a_stats,
            "question_chars": q_chars,
            "answer_chars": a_chars,
            "question_sentences": q_sents,
            "answer_sentences": a_sents,
            "answer_length_buckets": bucket_counts,
            "histograms": {
                "q_words": q_hist,
                "a_words": a_hist,
            },
        },
        "quality_duplicates": dup_info,
        "judge_analytics": {
            "judged_total": int(judged_total),
            "kept_after_judge": int(kept_after_judge),
            "dropped_by_judge": int(dropped_by_judge),
            "rejection_reason_counts": reason_counts,
            "persona_x_reason": pivot_persona_reasons,
            "per_reason_avg_lengths": reason_len,
        },
        "diversity": {
            "ttr_by_persona": ttr_by_persona
        }
    }

    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Console glance
    print(json.dumps({
        "kept": int(len(df_kept)),
        "rejected": int(len(df_rej)),
        "keep_rate_pct": round(keep_rate, 2),
        "dup_rate": summary["quality_duplicates"]["duplicate_rate"],
        "ttr_by_persona": ttr_by_persona
    }, indent=2))


if __name__ == "__main__":
    main()
