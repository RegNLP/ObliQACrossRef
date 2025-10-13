#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, collections, math

def each_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                # skip bad lines but keep count
                yield {"__parse_error__": True, "__line__": ln, "__err__": str(e)}

def main():
    ap = argparse.ArgumentParser(description="Inspect item_merged_clean.jsonl stats")
    ap.add_argument("--path", required=True, help="Path to item_merged_clean.jsonl")
    args = ap.parse_args()

    total = 0
    bad_lines = 0

    with_spans = 0
    total_spans = 0
    span_types = collections.Counter()
    span_len_sum = 0
    span_len_count = 0

    target_is_title = 0
    target_text_len_sum = 0
    target_text_len_count = 0

    ref_type = collections.Counter()
    src_item_type = collections.Counter()
    tgt_item_type = collections.Counter()

    semantic_hook_present = 0
    citation_hook_present = 0

    cross_semantic_true = 0
    cross_semantic_false = 0
    cross_cite_true = 0
    cross_cite_false = 0

    for obj in each_jsonl(args.path):
        if obj.get("__parse_error__"):
            bad_lines += 1
            continue
        total += 1

        # spans
        spans = obj.get("answer_spans") or []
        if isinstance(spans, list) and len(spans) > 0:
            with_spans += 1
            total_spans += len(spans)
            for sp in spans:
                span_types[sp.get("type") or "UNKNOWN"] += 1
                s, e = sp.get("start"), sp.get("end")
                if isinstance(s, int) and isinstance(e, int) and e >= s:
                    span_len_sum += (e - s)
                    span_len_count += 1

        # titles / lengths
        if obj.get("target_is_title"):
            target_is_title += 1
        ttl = obj.get("target_text_len")
        if isinstance(ttl, int):
            target_text_len_sum += ttl
            target_text_len_count += 1

        # simple categorical counts
        ref_type[obj.get("reference_type") or ""] += 1
        src_item_type[obj.get("source_item_type") or ""] += 1
        tgt_item_type[obj.get("target_item_type") or ""] += 1

        # hooks presence
        if (obj.get("semantic_hook") or "").strip():
            semantic_hook_present += 1
        if (obj.get("citation_hook") or "").strip():
            citation_hook_present += 1

        # crossref flags
        cro = obj.get("crossref_ok") or {}
        if isinstance(cro.get("semantic_in_source"), bool):
            if cro["semantic_in_source"]:
                cross_semantic_true += 1
            else:
                cross_semantic_false += 1
        if isinstance(cro.get("citation_matches_reference"), bool):
            if cro["citation_matches_reference"]:
                cross_cite_true += 1
            else:
                cross_cite_false += 1

    pct = lambda num, den: (100.0 * num / den) if den else 0.0
    avg = lambda s, n: (s / n) if n else 0.0

    summary = {
        "file": args.path,
        "totals": {
            "items_total": total,
            "bad_lines_skipped": bad_lines,
        },
        "answer_spans": {
            "items_with_answer_spans": with_spans,
            "items_with_answer_spans_pct": round(pct(with_spans, total), 2),
            "total_answer_spans": total_spans,
            "avg_spans_per_item_with_spans": round(avg(total_spans, with_spans), 3),
            "span_types_counts": dict(span_types),
            "avg_span_length_chars": round(avg(span_len_sum, span_len_count), 2),
        },
        "targets": {
            "items_target_is_title": target_is_title,
            "items_target_is_title_pct": round(pct(target_is_title, total), 2),
            "avg_target_text_len_chars": round(avg(target_text_len_sum, target_text_len_count), 2),
        },
        "categoricals": {
            "reference_type_counts": dict(ref_type),
            "source_item_type_counts": dict(src_item_type),
            "target_item_type_counts": dict(tgt_item_type),
        },
        "hooks_presence": {
            "semantic_hook_present": semantic_hook_present,
            "semantic_hook_present_pct": round(pct(semantic_hook_present, total), 2),
            "citation_hook_present": citation_hook_present,
            "citation_hook_present_pct": round(pct(citation_hook_present, total), 2),
        },
        "crossref_ok": {
            "semantic_in_source_true": cross_semantic_true,
            "semantic_in_source_false": cross_semantic_false,
            "citation_matches_reference_true": cross_cite_true,
            "citation_matches_reference_false": cross_cite_false,
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
