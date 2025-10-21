#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_annotation_sample_min.py

Minimal stratified sampler for 4 JSONL pools -> ONE combined JSONL + a text report.

Inputs:
  --schema-kept qas_from_schema_V2_kept.jsonl
  --schema-rejected qas_from_schema_V2_eliminated.jsonl
  --prompt-kept qas_baseline_judgement_kept.jsonl
  --prompt-rejected qas_baseline_judgement_eliminated.jsonl
  --sample-size 500
  --out-prefix outputs/ann_sample_v1

Behavior:
  - Strata = (method: schema|prompt) × (persona: basic|professional|unknown) × (status: kept|rejected)
  - Allocation across strata is PROPORTIONAL to available counts (largest remainder / Hamilton method).
  - Sampling is without replacement.
  - Outputs:
      <out_prefix>.jsonl
      <out_prefix>_report.txt
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

# --------- small helpers ---------

def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} invalid JSON: {e}")
    return items

def normalize_persona(v) -> str:
    if not v:
        return "unknown"
    t = str(v).strip().lower()
    if "prof" in t:
        return "professional"
    if "basic" in t:
        return "basic"
    if t in {"professional persona", "pro"}:
        return "professional"
    if t in {"basic persona"}:
        return "basic"
    return t if t in {"basic", "professional"} else "unknown"

def largest_remainder_quota(total: int, weights: List[int]) -> List[int]:
    """Hamilton apportionment given non-negative integer weights."""
    if total <= 0:
        return [0]*len(weights)
    s = sum(weights)
    if s == 0:
        base = total // len(weights)
        out = [base]*len(weights)
        for i in range(total - base*len(weights)):
            out[i] += 1
        return out
    raw = [w * total / s for w in weights]
    flo = [int(x) for x in map(float.__floor__, raw)] if hasattr(float, "__floor__") else [int(x) for x in raw]
    rem = total - sum(flo)
    # pair (fractional part, index); sort desc by fraction
    fracs = sorted([(raw[i] - flo[i], i) for i in range(len(raw))], reverse=True)
    for k in range(rem):
        _, idx = fracs[k]
        flo[idx] += 1
    return flo

# --------- core ---------

def load_buckets(schema_kept: Path, schema_rej: Path, prompt_kept: Path, prompt_rej: Path) -> Dict[Tuple[str,str,str], List[dict]]:
    """
    Returns dict keyed by (method, persona, status) -> list of raw items.
    """
    def push(pool: Dict[Tuple[str,str,str], List[dict]], items: List[dict], method: str, status: str):
        for obj in items:
            persona = normalize_persona(obj.get("persona"))
            key = (method, persona, status)
            pool.setdefault(key, []).append(obj)

    buckets: Dict[Tuple[str,str,str], List[dict]] = {}
    push(buckets, read_jsonl(schema_kept), "schema", "kept")
    push(buckets, read_jsonl(schema_rej),  "schema", "rejected")
    push(buckets, read_jsonl(prompt_kept), "prompt", "kept")
    push(buckets, read_jsonl(prompt_rej),  "prompt", "rejected")
    return buckets

def proportional_plan(buckets: Dict[Tuple[str,str,str], List[dict]], sample_size: int) -> Dict[Tuple[str,str,str], int]:
    keys = list(buckets.keys())
    caps = [len(buckets[k]) for k in keys]
    # initial proportional allocation across ALL buckets
    alloc = largest_remainder_quota(sample_size, caps)

    # ensure feasibility (cap at capacity and redistribute leftover)
    needs = dict(zip(keys, alloc))
    # cap
    leftover = 0
    for k in keys:
        cap = len(buckets[k])
        if needs[k] > cap:
            leftover += needs[k] - cap
            needs[k] = cap
    if leftover > 0:
        # redistribute leftover to buckets with spare capacity proportionally to spare
        spare_keys = [k for k in keys if len(buckets[k]) > needs[k]]
        if spare_keys:
            spares = [len(buckets[k]) - needs[k] for k in spare_keys]
            add = largest_remainder_quota(leftover, spares)
            for k, a in zip(spare_keys, add):
                needs[k] += a
        # if no spare at all, leftover silently drops (all pools exhausted)
    return needs

def sample_by_plan(buckets: Dict[Tuple[str,str,str], List[dict]], plan: Dict[Tuple[str,str,str], int], seed: int) -> List[Tuple[dict, Tuple[str,str,str]]]:
    rnd = random.Random(seed)
    picked: List[Tuple[dict, Tuple[str,str,str]]] = []
    for key, n in plan.items():
        pool = buckets.get(key, [])
        if not pool or n <= 0:
            continue
        # sample without replacement
        n_eff = min(n, len(pool))
        # random.sample raises on n > len(pool); we already min()'d
        idxs = rnd.sample(range(len(pool)), n_eff)
        for i in idxs:
            picked.append((pool[i], key))
    return picked

def write_outputs(picked: List[Tuple[dict, Tuple[str,str,str]]], plan: Dict[Tuple[str,str,str], int],
                  buckets: Dict[Tuple[str,str,str], List[dict]], out_prefix: Path, seed: int, in_paths: Dict[str, Path]):
    # .jsonl
    jsonl_path = out_prefix.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for sid, (obj, key) in enumerate(picked, 1):
            m, p, s = key
            out = dict(obj)
            out["_method"] = m
            out["_persona"] = p
            out["_status"]  = s
            out["_sample_id"] = sid
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # report.txt
    rpt_path = out_prefix.parent / (out_prefix.stem + "_report.txt")
    with rpt_path.open("w", encoding="utf-8") as r:
        r.write("Annotation Sample Report\n")
        r.write("=======================\n\n")
        r.write(f"Seed: {seed}\n")
        r.write("Inputs:\n")
        for k, p in in_paths.items():
            r.write(f"  {k}: {p}\n")
        r.write(f"\nTotal sample size (requested): {sum(plan.values())}\n")
        r.write(f"Total sampled (actual): {len(picked)}\n\n")

        # Summaries
        # input sizes per bucket
        r.write("Input sizes by (method, persona, status):\n")
        for key in sorted(buckets.keys()):
            r.write(f"  {key}: {len(buckets[key])}\n")
        r.write("\nPlanned allocations by (method, persona, status):\n")
        for key in sorted(plan.keys()):
            r.write(f"  {key}: {plan[key]}\n")
        # actual sampled by bucket
        counts: Dict[Tuple[str,str,str], int] = {}
        for _, key in picked:
            counts[key] = counts.get(key, 0) + 1
        r.write("\nActual sampled by (method, persona, status):\n")
        for key in sorted(plan.keys()):
            r.write(f"  {key}: {counts.get(key, 0)}\n")

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {rpt_path}")

# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Minimal stratified sampler -> one JSONL + text report.")
    ap.add_argument("--schema-kept",     type=Path, required=True)
    ap.add_argument("--schema-rejected", type=Path, required=True)
    ap.add_argument("--prompt-kept",     type=Path, required=True)
    ap.add_argument("--prompt-rejected", type=Path, required=True)
    ap.add_argument("--sample-size",     type=int, required=True, help="Total sample size across all buckets")
    ap.add_argument("--seed",            type=int, default=13)
    ap.add_argument("--out-prefix",      type=Path, required=True, help="e.g., outputs/ann_sample_v1")
    args = ap.parse_args()

    buckets = load_buckets(args.schema_kept, args.schema_rejected, args.prompt_kept, args.prompt_rejected)
    # If some buckets are missing entirely, that's fine—the plan will put 0 there.
    plan = proportional_plan(buckets, args.sample_size)
    picked = sample_by_plan(buckets, plan, seed=args.seed)

    write_outputs(
        picked=picked,
        plan=plan,
        buckets=buckets,
        out_prefix=args.out_prefix,
        seed=args.seed,
        in_paths={
            "schema_kept": args.schema_kept,
            "schema_rejected": args.schema_rejected,
            "prompt_kept": args.prompt_kept,
            "prompt_rejected": args.prompt_rejected,
        },
    )

if __name__ == "__main__":
    main()
