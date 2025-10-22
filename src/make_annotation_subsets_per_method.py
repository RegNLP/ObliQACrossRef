#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_annotation_subsets_per_method.py

Generates proportional, stratified annotation subsets for the ObliQA-CrossRef
"Practical-min" plan with powered Method comparison.

Inputs (JSONL):
  --prompt_kept        outputs/qas_baseline_judgement_kept.jsonl
  --prompt_elim        outputs/qas_baseline_judgement_eliminated.jsonl
  --schema_kept        outputs/qas_from_schema_V2_kept.jsonl
  --schema_elim        outputs/qas_from_schema_V2_eliminated.jsonl

Outputs (under --outdir):
  shared/kept_kappa_shared.jsonl       # 90 Kept items (triple-labeled κ block)
  annotators/Annotator_A.jsonl         # κ block + singles assigned to A
  annotators/Annotator_B.jsonl         # κ block + singles assigned to B
  annotators/Annotator_C.jsonl         # κ block + singles assigned to C
  summary_counts.json                  # machine-friendly summary
  summary_counts.txt                   # human-readable summary

Plan implemented: "practical_min"
  Kept totals (for method comparison & precision):
    - Prompt total = 200  → (Basic=80, Professional=120)
    - Schema total = 200  → (Basic=79, Professional=121)
  κ block (within Kept, triple-labeled = 90):
    - Prompt-Basic=25, Prompt-Prof=37, Schema-Basic=11, Schema-Prof=17
  Kept singles = 400 - 90 = 310
  Eliminated (for FNR): 100 total
    - Prompt-Basic=37, Prompt-Prof=29, Schema-Basic=19, Schema-Prof=15

Deterministic: sampling uses --seed (default=13).
"""

import argparse
import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# -----------------------------
# Configuration for Practical-min
# -----------------------------
METHODS = ("prompt", "schema")
PERSONAS = ("basic", "professional")
ANNOTATORS = ("Annotator_A", "Annotator_B", "Annotator_C")

# Kept totals per cell (Prompt/Schema × Basic/Prof)
KEPT_TOTALS = {
    ("prompt", "basic"): 80,
    ("prompt", "professional"): 120,
    ("schema", "basic"): 79,
    ("schema", "professional"): 121,
}

# κ (kappa) block counts within Kept (triple-labeled)
KAPPA_COUNTS = {
    ("prompt", "basic"): 25,
    ("prompt", "professional"): 37,
    ("schema", "basic"): 11,
    ("schema", "professional"): 17,
}

# Eliminated totals per cell
ELIM_TOTALS = {
    ("prompt", "basic"): 37,
    ("prompt", "professional"): 29,
    ("schema", "basic"): 19,
    ("schema", "professional"): 15,
}

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------- Helpers (use exactly once) ----------
def get_from_paths(rec: dict, paths: list, default: str = "") -> str:
    """
    Try multiple key paths, including nested dicts.
    Each path is a tuple/list of keys, e.g., ("debug_context","source_text").
    Returns the first non-empty string found; else default.
    """
    for path in paths:
        node = rec
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node and node[key] is not None:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (str, int, float)):
            s = str(node).strip()
            if s != "":
                return s
    return default

def get_field(rec: dict, *names: str, default: str = "") -> str:
    """
    Shallow (top-level) tolerant getter for variant casings.
    """
    for name in names:
        if name in rec and rec[name] is not None:
            return str(rec[name])
    lowered = {k.lower(): k for k in rec.keys()}
    for name in names:
        key = name.lower()
        if key in lowered:
            v = rec[lowered[key]]
            if v is not None:
                return str(v)
    return default
# ----------------------------------------------



def _normalize_persona(persona_raw: str) -> str:
    if not persona_raw:
        return "professional"
    s = persona_raw.lower().strip()
    if "basic" in s:
        return "basic"
    if "prof" in s or s in {"professional", "pro", "profesyonel"}:
        return "professional"
    return "professional"

def load_jsonl(path: str, default_method: str, bucket: str) -> List[dict]:
    """Load JSONL and inject normalized method/persona/bucket fields for your format."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            rec["method"] = default_method.lower().strip()
            rec["bucket"] = bucket

            # persona (already lower-case in your sample, but normalize anyway)
            persona_val = get_field(rec, "persona", "Persona", "persona_norm")
            rec["persona_norm"] = _normalize_persona(persona_val)

            # Prefer qa_id as the stable ID; then other candidates; else synthesize
            rec["_source_id"] = (
                get_field(rec, "qa_id", "qaId", "QA_ID")
                or get_field(rec, "id", "ID", "Id")
                or get_field(rec, "ItemID", "item_id", "itemId")
                or get_field(rec, "QuestionID", "question_id", "questionId")
                or get_field(rec, "uuid", "UID", "uid")
            )

            # Extract canonical content fields (top-level or nested in debug_context)
            q = get_from_paths(rec, [("question",), ("Question",), ("q",)], default="")
            a = get_from_paths(rec, [("expected_answer",), ("Answer",), ("answer",), ("a",)], default="")
            s = get_from_paths(rec, [("debug_context","source_text"),
                                     ("SourceText",), ("source_text",), ("sourceText",), ("source",)], default="")
            t = get_from_paths(rec, [("debug_context","target_text"),
                                     ("TargetText",), ("target_text",), ("targetText",), ("target",)], default="")

            # If no provided ID, synthesize from content + cell
            if not rec["_source_id"]:
                key = ((q or "")[:128], (s or "")[:128], (t or "")[:128], (a or "")[:128],
                       rec["method"], rec["persona_norm"], rec["bucket"])
                rec["_source_id"] = f"auto::{hash(key)}"

            # For convenience, store normalized content copies
            rec["_norm_question"] = q
            rec["_norm_answer"] = a
            rec["_norm_source"] = s
            rec["_norm_target"] = t

            items.append(rec)
    return items


def _sig_for_dedup(rec: dict):
    """
    Cell-aware signature for dedup across pools:
    distinct across (method, persona_norm, bucket).
    Prefer qa_id (or other ID) so duplicates collapse only within the same cell.
    Fallback to normalized content if no ID.
    """
    method = rec.get("method")
    persona = rec.get("persona_norm")
    bucket = rec.get("bucket")
    sid = rec.get("_source_id")

    if sid:
        return ("IDMBP", method, persona, bucket, str(sid))

    q = rec.get("_norm_question", "") or ""
    s = rec.get("_norm_source", "") or ""
    t = rec.get("_norm_target", "") or ""
    a = rec.get("_norm_answer", "") or ""
    return ("QSTAMB", method, persona, bucket, (q, s, t, a))


def dedup_by_signature(records: List[dict], name: str = "POOL") -> List[dict]:
    seen = set()
    out = []
    dropped = 0
    for r in records:
        s = _sig_for_dedup(r)
        if s in seen:
            dropped += 1
            continue
        seen.add(s)
        out.append(r)
    if dropped:
        print(f"[dedup] {name}: dropped {dropped} duplicate records; kept {len(out)}")
    return out

def group_by_cell(pool: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    by_cell = defaultdict(list)
    for rec in pool:
        by_cell[(rec["method"], rec["persona_norm"])].append(rec)
    return by_cell

def sample_without_replacement(rng: random.Random, population: List[dict], k: int) -> List[dict]:
    if k < 0:
        raise ValueError("k must be non-negative")
    if k > len(population):
        raise ValueError(f"Requested {k} samples but only {len(population)} available.")
    return rng.sample(population, k)

def round_robin_assign(items: List[dict], annotators=ANNOTATORS) -> Dict[str, List[dict]]:
    out = {a: [] for a in annotators}
    i = 0
    for rec in items:
        out[annotators[i % len(annotators)]].append(rec)
        i += 1
    return out

def write_jsonl(path: str, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# Borrowing (singles only)
# -----------------------------
def borrow_within_method_singles(
    rng: random.Random,
    need_map: Dict[Tuple[str, str], int],
    pools_by_cell: Dict[Tuple[str, str], List[dict]],
    method: str
):
    """
    For a given method, if any persona cell lacks singles, borrow from its sibling
    persona's remaining pool (same method). Mutates need_map and pools_by_cell.
    """
    cells = [(method, "basic"), (method, "professional")]
    for _ in range(2):  # two passes for cascading moves
        for m, p in cells:
            need = max(0, need_map.get((m, p), 0))
            if need <= 0:
                continue
            pool = pools_by_cell.get((m, p), [])
            have = len(pool)
            if have >= need:
                continue
            short = need - have
            sib = (m, "professional" if p == "basic" else "basic")
            sib_pool = pools_by_cell.get(sib, [])
            sib_have = len(sib_pool)
            if sib_have <= 0:
                continue
            take = min(short, sib_have)
            rng.shuffle(sib_pool)
            moved = sib_pool[:take]
            pools_by_cell[sib] = sib_pool[take:]
            pools_by_cell[(m, p)].extend(moved)

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_kept", required=True)
    ap.add_argument("--prompt_elim", required=True)
    ap.add_argument("--schema_kept", required=True)
    ap.add_argument("--schema_elim", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--plan", default="practical_min", choices=["practical_min"],
                    help="Currently only 'practical_min' is implemented.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load pools with explicit method/bucket tags from file role
    prompt_kept = load_jsonl(args.prompt_kept, default_method="prompt", bucket="kept")
    prompt_elim  = load_jsonl(args.prompt_elim,  default_method="prompt", bucket="eliminated")
    schema_kept  = load_jsonl(args.schema_kept,  default_method="schema", bucket="kept")
    schema_elim  = load_jsonl(args.schema_elim,  default_method="schema", bucket="eliminated")

    kept_pool = prompt_kept + schema_kept
    elim_pool = prompt_elim + schema_elim

    # Group by Method×Persona, then per-cell dedup (cell-aware)
    kept_by_cell = group_by_cell(kept_pool)
    elim_by_cell = group_by_cell(elim_pool)

    for cell in list(kept_by_cell.keys()):
        kept_by_cell[cell] = dedup_by_signature(kept_by_cell[cell], name=f"KEPT {cell}")
    for cell in list(elim_by_cell.keys()):
        elim_by_cell[cell] = dedup_by_signature(elim_by_cell[cell], name=f"ELIM {cell}")

    # Validate κ availability per cell (strict)
    for cell, need in KAPPA_COUNTS.items():
        have = len(kept_by_cell.get(cell, []))
        if have < need:
            raise RuntimeError(f"Not enough KEPT items for κ in {cell}: need {need}, have {have}.")

    # 1) Draw κ block from KEPT (signature-based removal)
    kappa_selected = []
    remaining_kept_after_kappa = {cell: list(recs) for cell, recs in kept_by_cell.items()}
    for cell, k_need in KAPPA_COUNTS.items():
        pool = remaining_kept_after_kappa[cell]
        chosen = sample_without_replacement(rng, pool, k_need)
        kappa_selected.extend(chosen)
        chosen_sigs = set(_sig_for_dedup(r) for r in chosen)
        remaining_kept_after_kappa[cell] = [r for r in pool if _sig_for_dedup(r) not in chosen_sigs]

    # 2) Draw KEPT singles per cell = KEPT_TOTALS - KAPPA_COUNTS
    kept_singles_selected = []
    kept_singles_need = {}
    for cell, total_need in KEPT_TOTALS.items():
        kappa_for_cell = KAPPA_COUNTS.get(cell, 0)
        singles_need = total_need - kappa_for_cell
        kept_singles_need[cell] = max(0, singles_need)

    # Borrow within same method if any cell lacks enough singles
    for method in METHODS:
        borrow_within_method_singles(
            rng,
            kept_singles_need,
            remaining_kept_after_kappa,
            method
        )

    # Now actually sample singles
    for cell, singles_need in kept_singles_need.items():
        pool = remaining_kept_after_kappa.get(cell, [])
        if len(pool) < singles_need:
            raise RuntimeError(f"Not enough KEPT singles in {cell}: need {singles_need}, have {len(pool)}.")
        chosen = sample_without_replacement(rng, pool, singles_need)
        kept_singles_selected.extend(chosen)
        chosen_sigs = set(_sig_for_dedup(r) for r in chosen)
        remaining_kept_after_kappa[cell] = [r for r in pool if _sig_for_dedup(r) not in chosen_sigs]

    # 3) Draw ELIMINATED singles per cell (with borrowing across personas within method if needed)
    elim_singles_selected = []
    elim_need = dict(ELIM_TOTALS)

    for method in METHODS:
        borrow_within_method_singles(
            rng,
            elim_need,
            elim_by_cell,
            method
        )

    for cell, need in elim_need.items():
        pool = elim_by_cell.get(cell, [])
        if len(pool) < need:
            raise RuntimeError(f"Not enough ELIMINATED in {cell}: need {need}, have {len(pool)}.")
        chosen = sample_without_replacement(rng, pool, need)
        elim_singles_selected.extend(chosen)

    # Final safety: ensure no overlap between κ and kept singles (by signature)
    kappa_sig = set(_sig_for_dedup(r) for r in kappa_selected)
    for r in kept_singles_selected:
        if _sig_for_dedup(r) in kappa_sig:
            raise AssertionError("Overlap between κ and kept singles detected (after signature-based removal).")

    # Prepare outputs
    outdir = args.outdir
    shared_dir = os.path.join(outdir, "shared")
    ann_dir = os.path.join(outdir, "annotators")
    ensure_dir(shared_dir)
    ensure_dir(ann_dir)

    # Write κ block (shared)
    write_jsonl(os.path.join(shared_dir, "kept_kappa_shared.jsonl"), kappa_selected)

    # Distribute singles to annotators round-robin within each cell to balance
    singles_by_cell = defaultdict(list)
    for r in kept_singles_selected:
        singles_by_cell[(r["method"], r["persona_norm"], "kept")].append(r)
    for r in elim_singles_selected:
        singles_by_cell[(r["method"], r["persona_norm"], "eliminated")].append(r)

    annotator_payloads = {a: [] for a in ANNOTATORS}

    # Everyone gets the κ block
    for a in ANNOTATORS:
        for rec in kappa_selected:
            rec_copy = dict(rec)
            rec_copy["assignment"] = "kappa_shared"
            annotator_payloads[a].append(rec_copy)

    # Within each (method, persona, bucket) cell, RR assignment of singles
    for cell, items in singles_by_cell.items():
        rng.shuffle(items)  # stable per seed
        rr = round_robin_assign(items, annotators=ANNOTATORS)
        for a, lst in rr.items():
            for rec in lst:
                rec_copy = dict(rec)
                rec_copy["assignment"] = "single"
                annotator_payloads[a].append(rec_copy)

    # Sort annotator files (nice diffs)
    def sort_key(r):
        return (
            0 if r["assignment"] == "kappa_shared" else 1,
            r.get("method", ""),
            r.get("persona_norm", ""),
            r.get("_source_id", ""),
        )

    for a in ANNOTATORS:
        annotator_payloads[a].sort(key=sort_key)
        write_jsonl(os.path.join(ann_dir, f"{a}.jsonl"), annotator_payloads[a])

    # Summaries
    def bucket_counts(records):
        c = Counter()
        for r in records:
            c[(r.get("assignment"), r.get("method"), r.get("persona_norm"), r.get("bucket"))] += 1
        return {f"{assn}|{m}|{p}|{b}": n for (assn, m, p, b), n in c.items()}

    summary = {
        "seed": args.seed,
        "plan": args.plan,
        "targets": {
            "kept_totals": {f"{m}|{p}": n for (m,p), n in KEPT_TOTALS.items()},
            "kappa_counts": {f"{m}|{p}": n for (m,p), n in KAPPA_COUNTS.items()},
            "elim_totals": {f"{m}|{p}": n for (m,p), n in ELIM_TOTALS.items()},
        },
        "produced": {
            "kappa_shared_total": len(kappa_selected),
            "kept_singles_total": len(kept_singles_selected),
            "elim_singles_total": len(elim_singles_selected),
        },
        "annotators": {a: {
            "total": len(annotator_payloads[a]),
            "breakdown": bucket_counts(annotator_payloads[a]),
        } for a in ANNOTATORS}
    }

    with open(os.path.join(outdir, "summary_counts.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable summary
    lines = []
    lines.append(f"Plan: {args.plan}  |  Seed: {args.seed}")
    lines.append("")
    lines.append("== Targets ==")
    lines.append(f"  Kept totals per cell: {KEPT_TOTALS}")
    lines.append(f"  Kappa (shared) per cell: {KAPPA_COUNTS}")
    lines.append(f"  Eliminated totals per cell: {ELIM_TOTALS}")
    lines.append("")
    lines.append("== Produced ==")
    lines.append(f"  Kappa shared: {len(kappa_selected)}")
    lines.append(f"  Kept singles: {len(kept_singles_selected)}")
    lines.append(f"  Eliminated singles: {len(elim_singles_selected)}")
    lines.append("")
    for a in ANNOTATORS:
        lines.append(f"-- {a} -- total {len(annotator_payloads[a])}")
        br = Counter(
            (r.get("assignment"), r.get("method"), r.get("persona_norm"), r.get("bucket"))
            for r in annotator_payloads[a]
        )
        for k, n in sorted(br.items()):
            lines.append(f"   {k}: {n}")
    with open(os.path.join(outdir, "summary_counts.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Done.")
    print(f"- shared/kept_kappa_shared.jsonl: {len(kappa_selected)}")
    for a in ANNOTATORS:
        print(f"- annotators/{a}.jsonl: {len(annotator_payloads[a])}")
    print("- summary_counts.json / summary_counts.txt")


if __name__ == "__main__":
    main()
