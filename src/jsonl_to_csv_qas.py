#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jsonl_to_csv_qas.py

Reads a JSONL file where each line looks like:
{
  "qa_id": "...",
  "persona": "...",
  "question": "...",
  "expected_answer": "...",
  "debug_context": {
      "source_text": "...",
      "target_text": "..."
  }
}

Writes a CSV with columns:
qa_id, Question, Rule 1, Rule 2, Proposed Answer
"""

import json
import csv
import argparse
from typing import Any, Dict

def get_nested(d: Dict[str, Any], *keys, default=""):
    """Safely get nested keys from a dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default

def main():
    parser = argparse.ArgumentParser(description="Convert QA JSONL to CSV with specific columns.")
    parser.add_argument("--input", "-i", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", "-o", required=True, help="Path to output .csv file")
    args = parser.parse_args()

    out_fields = ["qa_id", "Question", "Rule 1", "Rule 2", "Proposed Answer"]

    total = 0
    written = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines but continue
                continue

            row = {
                "qa_id": item.get("qa_id", ""),
                "Question": item.get("question", ""),
                "Rule 1": get_nested(item, "debug_context", "source_text", default=""),
                "Rule 2": get_nested(item, "debug_context", "target_text", default=""),
                "Proposed Answer": item.get("expected_answer", ""),
            }

            writer.writerow(row)
            written += 1

    print(f"Done. Read {total} lines, wrote {written} rows to {args.output}")

if __name__ == "__main__":
    main()
