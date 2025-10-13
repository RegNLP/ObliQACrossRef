#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_generate_qas.py
------------------
Schema-driven QA generation (Method 2 — step 2).

Reads items from 01_extract_schemas.py (JSONL with merged pair schema) and
emits QAs per persona. Each QA must *require* BOTH the source and the target.

What this script does:
- Input: JSONL where each line is an object with at least:
    item_id, source_text, target_text,
    semantic_hook, citation_hook,
    source_item_type, target_item_type,
    answer_spans (list of {text,start,end,type}),
    target_is_title (bool),
    plus helpful context like reference_type/reference_text and passage IDs.
- Skips items flagged as titles (target_is_title == True).
- Builds prompts that *use* the hooks/types/spans to guide the LLM.
- Produces per-persona QAs (professional, basic).
- Optional global dedup on question text (--dedup).
- Writes a generation report JSON with counts and skips.

Requires: openai>=1.0.0. Set OPENAI_API_KEY in your env.
"""

import argparse
import json
import os
import re
import sys
import uuid
from typing import Dict, List, Tuple, Any, Optional

# -----------------------------
# Utilities
# -----------------------------
def rand_uuid() -> str:
    return str(uuid.uuid4())

def ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def looks_empty(s: Optional[str]) -> bool:
    return len(ws(s)) == 0

def normalize_question_for_dedup(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s?]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q



def minimally_consistent_with_spans(answer: str, answer_spans: list) -> bool:
    ans = (answer or "").lower()
    if not ans: return False
    hints = span_hints(answer_spans)
    ok = True
    if hints["want_time"] and not any(k in ans for k in ["within","before","after","by ","day","days","month","months","year","years","business day"]):
        ok = False
    if hints["want_num"] and not re.search(r"\d", ans):
        ok = False
    # TERM/SECTION remain soft: they’re conceptual and phrasing varies.
    return ok

def read_schema_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                pass
    return items

def ensure_dir_for(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
# -----------------------------
# OpenAI call
# -----------------------------
def call_llm(model: str, system_prompt: str, user_prompt: str,
             max_tokens: int = 1400, temperature: float = 0.3,
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
ITEM_TYPES = {"Obligation","Prohibition","Permission","Definition","Scope","Procedure","Other"}
SPAN_TYPES = {"DURATION","DATE","MONEY","PERCENT","TERM","SECTION","FREEFORM"}

PROFESSIONAL_STYLE = (
    "Write concise, precise questions like a compliance reviewer. "
    "Use exact actor names from the texts (e.g., 'Authorised Person'). "
    "Use conditional framing only when it clarifies scope ('when', 'if', 'provided that')."
)
BASIC_STYLE = (
    "Write short, plain-language questions for non-experts. "
    "Keep one clear idea per question. Use the same actor names as in the texts."
)

SYSTEM_PROMPT = (
    "You generate regulatory Q&A that *requires joint reliance* on two passages. "
    "Output VALID JSON only—no markdown, no commentary."
)

ITEM_TYPE_GUIDANCE = {
    "Obligation": {
        "nudge": "Focus on required actions and the conditions under which they apply. Prefer 'must' semantics but phrase naturally."
    },
    "Prohibition": {
        "nudge": "Emphasize what is not allowed and boundary conditions that trigger the restriction."
    },
    "Permission": {
        "nudge": "Surface when an actor may/can do something and any prerequisites or approvals."
    },
    "Definition": {
        "nudge": "Clarify meanings/criteria in context; avoid pure lookup—tie to how it’s applied here."
    },
    "Scope": {
        "nudge": "Highlight coverage/exclusions and who/what is within scope."
    },
    "Procedure": {
        "nudge": "Elicit how/when steps occur and any gating conditions; keep answers crisp."
    },
    "Other": {
        "nudge": "Use neutral compliance phrasing while ensuring both texts are required."
    },
}

def span_hints(answer_spans: list) -> dict:
    """Return soft constraints we’ll bake into the prompt + post-check."""
    want_time = any(s.get("type") in {"DATE","DURATION"} for s in answer_spans or [])
    want_num  = any(s.get("type") in {"MONEY","PERCENT"} for s in answer_spans or [])
    want_term = any(s.get("type") == "TERM" for s in answer_spans or [])
    want_sect = any(s.get("type") == "SECTION" for s in answer_spans or [])
    return {
        "want_time": want_time, "want_num": want_num,
        "want_term": want_term, "want_sect": want_sect
    }

def spans_to_natural_markers(answer_spans: list) -> str:
    """
    Turn spans into abstract anchors the model must cover—without quoting literal text.
    E.g., “a deadline/timing element”, “a fee/percentage threshold”, “a defined term”, “a referenced section”.
    """
    if not answer_spans:
        return "no explicit anchors; rely on core target concepts"
    kinds = sorted(set(s.get("type","FREEFORM") for s in answer_spans))
    labels = []
    if "DATE" in kinds or "DURATION" in kinds: labels.append("a deadline/timing element")
    if "MONEY" in kinds: labels.append("a monetary/fee amount")
    if "PERCENT" in kinds: labels.append("a percentage/threshold")
    if "TERM" in kinds: labels.append("a defined term")
    if "SECTION" in kinds: labels.append("a referenced section")
    if "FREEFORM" in kinds and not labels: labels.append("a key phrase or clause")
    return ", ".join(labels)

def span_preview(spans: List[Dict[str, Any]], max_chars: int = 180) -> str:
    """Tiny preview of spans for the model (not indices, just gist)."""
    outs = []
    for sp in (spans or [])[:3]:
        t = ws(sp.get("text", ""))
        outs.append(t[:max_chars])
    return "\n- ".join(outs) if outs else ""

def build_prompt_schema(
    source_text: str,
    target_text: str,
    semantic_hook: str,
    citation_hook: str,
    source_item_type: str,
    target_item_type: str,
    answer_spans: list,
    max_per_persona: int,
    sample_n: int
) -> str:
    src_type = source_item_type if source_item_type in ITEM_TYPES else "Other"
    tgt_type = target_item_type if target_item_type in ITEM_TYPES else "Other"
    nudge = ITEM_TYPE_GUIDANCE[tgt_type]["nudge"]
    hints = span_hints(answer_spans)
    anchors = spans_to_natural_markers(answer_spans)

    span_rules = []
    if hints["want_time"]:
        span_rules.append("- Reflect a timing element (deadline, window, or duration) in the answer.")
    if hints["want_num"]:
        span_rules.append("- Reflect a numeric/threshold element (amount, fee, or percentage) in the answer.")
    if hints["want_term"]:
        span_rules.append("- Clarify a defined term in plain language (no quoting).")
    if hints["want_sect"]:
        span_rules.append("- Refer conceptually to the relevant section (no raw citation tokens).")
    span_rules_text = "\n".join(span_rules) if span_rules else "- No special numeric/timing constraints are required."

    return f"""
ROLE: Create Q&A items that require BOTH the SOURCE and TARGET to be correct.

TYPE NUDGE (drives tone/intent, not phrasing): {nudge}

SCHEMA SIGNALS (use them as conceptual anchors, do not quote):
- semantic_hook: "{semantic_hook or ''}"
- citation_hook (concept pointer only): "{citation_hook or ''}"
- source_item_type={src_type}, target_item_type={tgt_type}
- target anchors: {anchors}

HARD RULES
1) Joint reliance: both texts must be needed for each Q and its answer.
2) Use actor labels exactly as written (e.g., "Authorised Person"). No invented actors.
3) No verbatim quotes, brackets, or raw citations; paraphrase naturally.
4) Answers: one concise statement (short list only if essential).
5) Span reflection from TARGET:
{span_rules_text}

DIVERSITY
- Vary interrogatives and structures (what/when/how/under what conditions/can/must).
- Avoid repeating the same opening clause across outputs.

OUTPUT JSON (exactly; no commentary):
{{
  "professional": [{{"question":"...","answer":"..."}} ],
  "basic":        [{{"question":"...","answer":"..."}} ]
}}

Quantity
- Brainstorm up to {sample_n} internally; output ≤ {max_per_persona} per persona.

SOURCE (full):
\"\"\"{source_text}\"\"\"

TARGET (full):
\"\"\"{target_text}\"\"\"
"""


# -----------------------------
# QA collection
# -----------------------------
def collect_qas(
    llm_obj: Dict[str, Any],
    schema: Dict[str, Any],
    max_q_per_persona: int,
    dedup_set: Optional[set]
) -> Tuple[List[Dict[str, Any]], int, int]:
    out: List[Dict[str, Any]] = []
    dropped_dupe = 0
    kept = 0

    for persona in ["professional", "basic"]:
        arr = llm_obj.get(persona, []) if isinstance(llm_obj, dict) else []
        if not isinstance(arr, list):
            continue

        persona_kept = 0
        for it in arr:
            if not isinstance(it, dict):
                continue
            q = ws(it.get("question", ""))
            a = ws(it.get("answer", ""))
            if looks_empty(q) or looks_empty(a):
                continue

            if dedup_set is not None:
                key = normalize_question_for_dedup(q)
                if key in dedup_set:
                    dropped_dupe += 1
                    continue
                dedup_set.add(key)

            qa = {
                "qa_id": rand_uuid(),
                "persona": persona,
                "question": q,
                "expected_answer": a,
                "debug_context": {
                    # raw passages
                    "source_text": schema.get("source_text"),
                    "target_text": schema.get("target_text"),
                    "source_passage_id": schema.get("source_passage_id"),
                    "target_passage_id": schema.get("target_passage_id"),
                    # schema guidance
                    "semantic_hook": schema.get("semantic_hook"),
                    "citation_hook": schema.get("citation_hook"),
                    "source_item_type": schema.get("source_item_type"),
                    "target_item_type": schema.get("target_item_type"),
                    "answer_spans": schema.get("answer_spans", []),
                    # reference context (nice to keep, cheap)
                    "reference_type": schema.get("reference_type"),
                    "reference_text": schema.get("reference_text"),
                    # provenance of item source
                    "schema_item_id": schema.get("item_id"),
                },
            }
            out.append(qa)
            persona_kept += 1
            kept += 1
            if persona_kept >= max_q_per_persona:
                break

    return out, dropped_dupe, kept

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Schema-driven QA generation (Method 2 — step 2).")
    ap.add_argument("--input_jsonl", required=True, help="JSONL from 01_extract_schemas.py")
    ap.add_argument("--output_jsonl", required=True, help="Where to write generated QAs (JSONL)")
    ap.add_argument("--report_json", required=True, help="Where to write a run report (JSON)")
    ap.add_argument("--model", required=True, help="OpenAI model, e.g., gpt-4o")

    # generation controls
    ap.add_argument("--max_q_per_pair", type=int, default=2, help="max per persona (professional/basic)")
    ap.add_argument("--sample_n", type=int, default=3, help="brainstorm cap hint per persona (LLM-internal)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=None)

    # dataset shaping
    ap.add_argument("--keep_reference_types", nargs="*", default=["internal", "external"],
                    help="case-insensitive filter on reference_type; empty => keep all")
    ap.add_argument("--item_sample_n", type=int, default=None, help="sample N items from input (after filters)")
    ap.add_argument("--item_sample_seed", type=int, default=13)
    ap.add_argument("--max_items", type=int, default=None, help="hard cap processed items")

    # behavior
    ap.add_argument("--dedup", action="store_true", help="global dedup over question text")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="no LLM calls; just count candidates")

    args = ap.parse_args()

    # load schemas
    items = read_schema_jsonl(args.input_jsonl)
    if args.verbose:
        print(f"[info] loaded schema items: {len(items)}", flush=True)

    # filter by reference_type (if present)
    keep_rt = set([s.lower() for s in (args.keep_reference_types or [])])
    if keep_rt:
        def _keep(item):
            rt = (item.get("reference_type") or "").lower()
            return (rt in keep_rt) if rt else True  # if missing, keep
        items = [it for it in items if _keep(it)]

    # drop titles
    pre_drop = len(items)
    items = [it for it in items if not bool(it.get("target_is_title"))]
    dropped_titles = pre_drop - len(items)

    # sampling
    if args.item_sample_n and args.item_sample_n > 0 and len(items) > args.item_sample_n:
        import random
        rnd = random.Random(args.item_sample_seed or 13)
        items = rnd.sample(items, args.item_sample_n)
    if args.max_items and args.max_items > 0 and len(items) > args.max_items:
        items = items[:args.max_items]

    ensure_dir_for(args.output_jsonl)
    ensure_dir_for(args.report_json)

    # stats
    items_loaded = len(items)
    items_processed = 0
    kept_candidates = 0
    qas_created = 0
    dropped_dupe_qs = 0
    skipped_empty_text = 0
    skipped_model_fail = 0
    span_inconsistency_drops = 0

    # NEW: observability counters (no hard filtering)
    items_with_empty_spans = 0
    items_with_empty_semhook = 0

    dedup_set = set() if args.dedup else None
    progress_every = 50

    if args.dry_run:
        # just count viable candidates (non-empty source/target) and observe schema holes
        for it in items:
            st = ws(it.get("source_text"))
            tt = ws(it.get("target_text"))
            if looks_empty(st) or looks_empty(tt):
                skipped_empty_text += 1
                continue
            kept_candidates += 1

            # observe empties
            if not (it.get("answer_spans") or []):
                items_with_empty_spans += 1
            if len(ws(it.get("semantic_hook"))) == 0:
                items_with_empty_semhook += 1

        report = {
            "items_loaded": items_loaded,
            "kept_candidates": kept_candidates,
            "items_processed": kept_candidates,
            "qas_created": 0,
            "dropped_dupe_qs": 0,
            "skipped_empty_text": skipped_empty_text,
            "skipped_model_fail": 0,
            "dropped_title_like_targets": dropped_titles,
            "span_inconsistency_drops": 0,
            "items_with_empty_spans": items_with_empty_spans,
            "items_with_empty_semhook": items_with_empty_semhook,
            "model": args.model,
        }
        print(json.dumps(report, indent=2))
        return

    with open(args.output_jsonl, "w", encoding="utf-8") as outf:
        for idx, it in enumerate(items, 1):
            source_text = ws(it.get("source_text"))
            target_text = ws(it.get("target_text"))
            if looks_empty(source_text) or looks_empty(target_text):
                skipped_empty_text += 1
                continue

            items_processed += 1
            kept_candidates += 1

            # observe empties (no filtering)
            spans = it.get("answer_spans") or []
            if not spans:
                items_with_empty_spans += 1
            semhook = ws(it.get("semantic_hook"))
            if not semhook:
                items_with_empty_semhook += 1

            user_prompt = build_prompt_schema(
                source_text=source_text,
                target_text=target_text,
                semantic_hook=semhook,
                citation_hook=ws(it.get("citation_hook")),
                source_item_type=it.get("source_item_type") or "Other",
                target_item_type=it.get("target_item_type") or "Other",
                answer_spans=spans,
                max_per_persona=args.max_q_per_pair,
                sample_n=args.sample_n
            )

            content = call_llm(
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1400,
                temperature=args.temperature,
                seed=args.seed
            )
            if not content:
                skipped_model_fail += 2 * args.max_q_per_pair
                if args.verbose:
                    print(f"[warn] empty model content for item {it.get('item_id')}", flush=True)
                continue

            obj = parse_llm_json(content)
            if not obj:
                skipped_model_fail += 2 * args.max_q_per_pair
                if args.verbose:
                    print(f"[warn] json parse fail for item {it.get('item_id')}", flush=True)
                continue

            qa_list, dup_ct, kept_ct = collect_qas(
                llm_obj=obj,
                schema=it,
                max_q_per_persona=args.max_q_per_pair,
                dedup_set=dedup_set
            )

            # span-aware post-check is lenient; only enforces numbers/timing when spans say so
            filtered = []
            for qa in qa_list:
                if minimally_consistent_with_spans(qa["expected_answer"], spans):
                    filtered.append(qa)
                else:
                    span_inconsistency_drops += 1

            dropped_dupe_qs += dup_ct
            qas_created += len(filtered)

            for qa in filtered:
                outf.write(json.dumps(qa, ensure_ascii=False) + "\n")

            if args.verbose and items_processed % progress_every == 0:
                print(f"[progress] {items_processed}/{items_loaded} items | qas={qas_created}", flush=True)

    report = {
        "items_loaded": items_loaded,
        "kept_candidates": kept_candidates,
        "items_processed": items_processed,
        "qas_created": qas_created,
        "dropped_dupe_qs": dropped_dupe_qs,
        "skipped_empty_text": skipped_empty_text,
        "skipped_model_fail": skipped_model_fail,
        "dropped_title_like_targets": dropped_titles,
        "span_inconsistency_drops": span_inconsistency_drops,
        "items_with_empty_spans": items_with_empty_spans,
        "items_with_empty_semhook": items_with_empty_semhook,
        "model": args.model,
    }
    with open(args.report_json, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2))
