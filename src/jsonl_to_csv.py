#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, sys
from typing import Any, Dict, Iterable, List

# ---------- tolerant getters ----------
def get_field(rec: Dict[str, Any], *names: str, default: str = "") -> str:
    # direct
    for n in names:
        if n in rec and rec[n] is not None:
            return str(rec[n])
    # case-insensitive
    lowered = {k.lower(): k for k in rec.keys()}
    for n in names:
        k = n.lower()
        if k in lowered and rec[lowered[k]] is not None:
            return str(rec[lowered[k]])
    return default

def get_from_paths(rec: Dict[str, Any], paths: List[List[str]], default: str = "") -> str:
    """Follow multiple candidate key paths (supports nested dicts)."""
    for path in paths:
        node: Any = rec
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node and node[key] is not None:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (str, int, float)):
            s = str(node).strip()
            if s:
                return s
    return default

def parse_jsonl(stream: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception as e:
            sys.stderr.write(f"[warn] skip bad JSONL line: {line[:120]}... ({e})\n")

# ---------- conversion ----------
def row_from_record(rec: Dict[str, Any]) -> Dict[str, str]:
    # input: either raw record or record+annotation
    # tolerate different layouts:
    # - top-level fields
    # - nested under debug_context.{source_text,target_text}
    # - annotation at rec['annotation']

    qa_id = get_field(rec, "qa_id", "qaId", "QA_ID", "id", "uuid", "QuestionID", "_source_id")
    if not qa_id:
        # synthesize from content for stability
        q = get_from_paths(rec, [["question"], ["Question"], ["q"]])
        s = get_from_paths(rec, [["debug_context","source_text"], ["SourceText"], ["source_text"], ["sourceText"], ["source"]])
        t = get_from_paths(rec, [["debug_context","target_text"], ["TargetText"], ["target_text"], ["targetText"], ["target"]])
        qa_id = f"auto::{hash((q[:64], s[:64], t[:64]))}"

    question = get_from_paths(rec, [["question"], ["Question"], ["q"]])
    expected_answer = get_from_paths(rec, [["expected_answer"], ["Answer"], ["answer"], ["a"]])
    source_text = get_from_paths(
        rec,
        [["debug_context","source_text"], ["SourceText"], ["source_text"], ["sourceText"], ["source"]]
    )
    target_text = get_from_paths(
        rec,
        [["debug_context","target_text"], ["TargetText"], ["target_text"], ["targetText"], ["target"]]
    )

    ann = rec.get("annotation", {}) or rec.get("annotations", {}) or {}
    q1 = get_field(ann, "question_validity", "q1")
    q2 = get_field(ann, "required_context", "q2")
    q3 = get_field(ann, "answer_correct_complete", "q3")
    final_ = get_field(ann, "final_decision", "final")
    comment = get_field(ann, "comment", "notes", "note")

    return {
        "qa_id": qa_id,
        "question": question,
        "expected_answer": expected_answer,
        "source_text": source_text,
        "target_text": target_text,
        "1_question_valid": q1,
        "2_needed": q2,
        "3_answer_correct": q3,
        "4_final_decision": final_,
        "5_comment": comment,
    }

def main():
    ap = argparse.ArgumentParser(description="Convert ObliQA JSONL (with annotations) to CSV.")
    ap.add_argument("input", help="input JSONL file (use '-' for stdin)")
    ap.add_argument("output", help="output CSV file (use '-' for stdout)")
    args = ap.parse_args()

    # open streams
    instream = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    outstream = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")

    fieldnames = [
        "qa_id", "question", "expected_answer", "source_text", "target_text",
        "1_question_valid", "2_needed", "3_answer_correct", "4_final_decision", "5_comment"
    ]

    writer = csv.DictWriter(outstream, fieldnames=fieldnames)
    writer.writeheader()

    try:
        for rec in parse_jsonl(instream):
            writer.writerow(row_from_record(rec))
    finally:
        if instream is not sys.stdin:
            instream.close()
        if outstream is not sys.stdout:
            outstream.close()

if __name__ == "__main__":
    main()
