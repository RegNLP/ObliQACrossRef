#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_annotation_subsets_per_method.py

Stratified JSONL subset creator for ObliQA-CrossRef when inputs are provided
as FOUR separate files by method:
  - Schema kept
  - Schema eliminated
  - Prompt kept
  - Prompt eliminated

Goals (defaults mirror your 10.5h design, *per method*):
  • KEPT κ-block (shared by all):      --kappa_per_method         (default 60) from each method
  • KEPT singles (unique across ann.): --kept_single_per_method   (default 75) from each method
  • ELIM singles (unique across ann.): --elim_single_per_method   (default 60) from each method
Totals with defaults and 3 annotators:
  - Shared kept κ-block: 120 (60 Prompt + 60 Schema) → triple-labeled
  - Kept singles: 150 (75 Prompt + 75 Schema) → split evenly across annotators
  - Eliminated singles: 120 (60 Prompt + 60 Schema) → split evenly across annotators
  - Per annotator load: 120(shared) + 50(kept) + 40(elim) = 210 (~10.5h @ 3 min/item)

Persona handling:
  - By default, we keep BOTH personas and distribute quotas across personas PROPORTIONALLY to pool sizes.
  - Use --drop_basic to filter to Professional-only BEFORE sampling (stratify by Method only).

Input JSONL expectations (each line is a dict):
  - "Method": "Prompt" or "Schema" (if missing, the script will inject it from the file role)
  - "Persona": "Basic" or "Professional"  (ignored in --drop_basic mode where we pre-filter)
  - Other fields are passed through unchanged.

Usage example (both personas):
  python make_annotation_subsets_per_method.py \
    --schema_kept outputs/qas_from_schema_V2_kept.jsonl \
    --schema_elim outputs/qas_from_schema_V2_eliminated.jsonl \
    --prompt_kept outputs/qas_baseline_judgement_kept.jsonl \
    --prompt_elim outputs/qas_baseline_judgement_eliminated.jsonl \
    --outdir ./annotation_subsets_per_method \
    --seed 13

Professional-only:
  python make_annotation_subsets_per_method.py ... --drop_basic --outdir ./annotation_subsets_prof_only

"""

import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

METHODS = ["Prompt", "Schema"]
PERSONAS = ["Basic", "Professional"]
ANNOTATORS_DEFAULT = 3


def read_jsonl(path, force_method=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON on line {ln}: {e}")
            if force_method is not None:
                obj["Method"] = force_method
            elif obj.get("Method") not in METHODS:
                raise ValueError(f"{path}: record missing/invalid 'Method': {obj.get('Method')}")
            items.append(obj)
    return items


def filter_professional_only(items):
    return [x for x in items if x.get("Persona") == "Professional"]


def group_by_persona(items):
    bins = defaultdict(list)  # key: persona
    for obj in items:
        p = obj.get("Persona")
        if p not in PERSONAS:
            # if Persona missing in pro-only mode we tolerate (it was filtered already)
            p = obj.get("Persona", "Professional")
        bins[p].append(obj)
    return bins


def largest_remainder_quota(total, weights_dict):
    """Proportional integer quotas by largest-remainder method."""
    keys = list(weights_dict.keys())
    weights = [max(0, int(weights_dict[k])) for k in keys]
    s = sum(weights)
    if s == 0:
        # split evenly
        base = total // len(keys)
        rem = total - base * len(keys)
        q = {k: base for k in keys}
        for i in range(rem):
            q[keys[i % len(keys)]] += 1
        return q
    exact = [w * total / s for w in weights]
    floors = [math.floor(x) for x in exact]
    rem = total - sum(floors)
    fracs = sorted([(i, exact[i] - floors[i]) for i in range(len(keys))], key=lambda t: t[1], reverse=True)
    out = floors[:]
    for i in range(rem):
        out[fracs[i][0]] += 1
    return {keys[i]: out[i] for i in range(len(keys))}


def sample_from_bins(rng, bins, quotas):
    """bins: dict key->list; quotas: dict key->int"""
    sampled = []
    picked_by_key = {}
    for k, need in quotas.items():
        pool = bins.get(k, [])
        if len(pool) < need:
            raise ValueError(f"Not enough items in stratum {k}: need {need}, have {len(pool)}")
        chosen = rng.sample(pool, need)
        sampled.extend(chosen)
        picked_by_key[k] = chosen
        # remove chosen in-place
        bins[k] = [x for x in pool if x not in chosen]
    return sampled, picked_by_key


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def even_split(items, annotators, rng):
    items = items[:]
    rng.shuffle(items)
    splits = {a: [] for a in annotators}
    for i, obj in enumerate(items):
        splits[annotators[i % len(annotators)]].append(obj)
    return splits


def count_method_persona(items):
    c = Counter()
    for x in items:
        c[(x.get("Method"), x.get("Persona"))] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_kept", required=True)
    ap.add_argument("--schema_elim", required=True)
    ap.add_argument("--prompt_kept", required=True)
    ap.add_argument("--prompt_elim", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--annotators", type=int, default=ANNOTATORS_DEFAULT)
    ap.add_argument("--drop_basic", action="store_true", help="Filter to Professional persona only before sampling")
    # Per-method quotas (defaults mirror the 10.5h design)
    ap.add_argument("--kappa_per_method", type=int, default=60, help="Shared kept items PER METHOD")
    ap.add_argument("--kept_single_per_method", type=int, default=75, help="Unique kept items PER METHOD")
    ap.add_argument("--elim_single_per_method", type=int, default=60, help="Unique eliminated items PER METHOD")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    (outdir / "shared").mkdir(parents=True, exist_ok=True)
    (outdir / "singles").mkdir(parents=True, exist_ok=True)
    (outdir / "annotators").mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(parents=True, exist_ok=True)

    # Read inputs; inject Method where needed
    schema_kept = read_jsonl(args.schema_kept, force_method="Schema")
    schema_elim = read_jsonl(args.schema_elim, force_method="Schema")
    prompt_kept = read_jsonl(args.prompt_kept, force_method="Prompt")
    prompt_elim = read_jsonl(args.prompt_elim, force_method="Prompt")

    # Optional persona filtering
    if args.drop_basic:
        schema_kept = filter_professional_only(schema_kept)
        schema_elim = filter_professional_only(schema_elim)
        prompt_kept = filter_professional_only(prompt_kept)
        prompt_elim = filter_professional_only(prompt_elim)

    # ---- SAMPLE PER METHOD ----
    def sample_method(method_name, kept_pool, elim_pool):
        # Group by persona bins (or single bin if pro-only)
        if args.drop_basic:
            kept_bins = {"Professional": kept_pool}
            elim_bins = {"Professional": elim_pool}
        else:
            kept_bins = group_by_persona(kept_pool)
            elim_bins = group_by_persona(elim_pool)

        # Quotas across personas within the method (proportional)
        kept_kappa_quota = largest_remainder_quota(args.kappa_per_method, {k: len(v) for k, v in kept_bins.items()})
        kept_single_quota = largest_remainder_quota(args.kept_single_per_method, {k: len(v) for k, v in kept_bins.items()})
        elim_quota = largest_remainder_quota(args.elim_single_per_method, {k: len(v) for k, v in elim_bins.items()})

        # Sample kappa kept (shared)
        kept_kappa, _ = sample_from_bins(rng, kept_bins, kept_kappa_quota)
        # Sample kept singles
        kept_singles, _ = sample_from_bins(rng, kept_bins, kept_single_quota)
        # Sample eliminated singles
        elim_singles, _ = sample_from_bins(rng, elim_bins, elim_quota)

        # Sanity: label Method (already injected) and return
        for x in kept_kappa + kept_singles + elim_singles:
            x["Method"] = method_name
        return kept_kappa, kept_singles, elim_singles

    # Prompt method sampling
    p_kappa, p_kept_singles, p_elim_singles = sample_method("Prompt", prompt_kept, prompt_elim)
    # Schema method sampling
    s_kappa, s_kept_singles, s_elim_singles = sample_method("Schema", schema_kept, schema_elim)

    # Combine shared κ-block across methods
    kept_kappa_shared = p_kappa + s_kappa
    write_jsonl(outdir / "shared" / "kept_kappa_shared.jsonl", kept_kappa_shared)

    # Split singles evenly across annotators, separately for each method then merge per annotator
    annotators = [chr(ord('A') + i) for i in range(args.annotators)]

    p_kept_splits = even_split(p_kept_singles, annotators, rng)
    s_kept_splits = even_split(s_kept_singles, annotators, rng)
    p_elim_splits = even_split(p_elim_singles, annotators, rng)
    s_elim_splits = even_split(s_elim_singles, annotators, rng)

    # Write method-specific singles files (optional but handy)
    for a in annotators:
        write_jsonl(outdir / "singles" / f"kept_single_prompt_{a}.jsonl", p_kept_splits[a])
        write_jsonl(outdir / "singles" / f"kept_single_schema_{a}.jsonl", s_kept_splits[a])
        write_jsonl(outdir / "singles" / f"eliminated_prompt_{a}.jsonl", p_elim_splits[a])
        write_jsonl(outdir / "singles" / f"eliminated_schema_{a}.jsonl", s_elim_splits[a])

    # Merge per-annotator bundles: shared κ + kept singles (both methods) + elim (both methods)
    for a in annotators:
        bundle = []
        for obj in kept_kappa_shared:
            o = dict(obj); o["_assignment"] = "kept_kappa_shared"; bundle.append(o)
        for obj in p_kept_splits[a] + s_kept_splits[a]:
            o = dict(obj); o["_assignment"] = "kept_single"; bundle.append(o)
        for obj in p_elim_splits[a] + s_elim_splits[a]:
            o = dict(obj); o["_assignment"] = "eliminated_single"; bundle.append(o)
        # Save annotator package
        write_jsonl(outdir / "annotators" / f"Annotator_{a}.jsonl", bundle)

    # Reports
    def brief_counts(items):
        return {
            "total": len(items),
            "by_method_persona": count_method_persona(items)
        }

    rep_lines = []
    mode = "PROFESSIONAL-ONLY" if args.drop_basic else "BOTH-PERSONAS"
    rep_lines.append(f"=== Mode: {mode} ===")
    rep_lines.append(f"Annotators: {len(annotators)}")
    rep_lines.append(f"Per-method quotas (defaults): kappa={args.kappa_per_method}, kept_single={args.kept_single_per_method}, elim_single={args.elim_single_per_method}")
    rep_lines.append(f"Shared kept κ-block: total={len(kept_kappa_shared)} → {brief_counts(kept_kappa_shared)}")

    # Check per-annotator loads
    for a in annotators:
        total = len(kept_kappa_shared) + len(p_kept_splits[a]) + len(s_kept_splits[a]) + len(p_elim_splits[a]) + len(s_elim_splits[a])
        rep_lines.append(f"Annotator {a}: total items = {total} (shared + singles), kept singles={len(p_kept_splits[a])+len(s_kept_splits[a])}, elim singles={len(p_elim_splits[a])+len(s_elim_splits[a])}")

    (outdir / "reports" / "manifest_counts.txt").write_text("\n".join(rep_lines), encoding="utf-8")

    print("[OK] Subsets created in:", outdir)
    print("  Shared kept κ-block:", len(kept_kappa_shared))
    for a in annotators:
        total = len(kept_kappa_shared) + len(p_kept_splits[a]) + len(s_kept_splits[a]) + len(p_elim_splits[a]) + len(s_elim_splits[a])
        print(f"  Annotator {a}: total={total} (kept singles={len(p_kept_splits[a])+len(s_kept_splits[a])}, elim singles={len(p_elim_splits[a])+len(s_elim_splits[a])})")


if __name__ == "__main__":
    main()
