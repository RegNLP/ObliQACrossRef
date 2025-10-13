#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ObliQACrossRefDataset — 01_extract_schemas.py (merged-pairs, lean schema)

Reads a cross-reference CSV and emits one merged JSONL item per (source,target) pair
with a lean schema tailored for QG and evaluation.

OUTPUT FIELDS (per item):
- item_id
- reference_type
- reference_text
- semantic_hook          # short verbatim, NO citations
- citation_hook          # the citation-like token (e.g., "Rule 9.7.5", "Section 58(2)")
- source_passage_id
- source_text
- target_passage_id
- target_text
- source_item_type       # {Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other}
- target_item_type       # same set
- answer_spans           # spans lie strictly within target_text (prefer TERM/SECTION/etc.)
- target_is_title        # heading-like targets are flagged True
- provenance             # {model, ts}

Policies:
- If LLM returns no valid answer_spans and target is NOT a title:
  • Provide ONE concise span; prefer a core clause; otherwise truncate to ≤220 chars (FREEFORM).
- If target looks like a title/heading, keep answer_spans empty.
- Dedup by a hash of core strings (source_text, target_text, hooks, reference_text).
- Optional: --drop_title_targets to skip heading-only targets.
- Default model: gpt-4o-mini (temperature 0.0)

CLI example:
python src/01_extract_schemas.py \
  --input_csv data/CrossReferenceData.csv \
  --output_jsonl outputs/items_merged_clean.jsonl \
  --sample_n 30 --sample_seed 13 \
  --model gpt-4o \
  --drop_title_targets
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---- OpenAI client (>=1.0.0 style) -------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ------------------------- Constants / Types ----------------------------------------
ALLOWED_ITEM_TYPES = {"Obligation", "Prohibition", "Permission", "Definition", "Scope", "Procedure", "Other"}
ALLOWED_SPAN_TYPES = {"DURATION", "DATE", "MONEY", "PERCENT", "TERM", "SECTION", "FREEFORM"}
MAX_FREEFORM_CHARS = 220  # cap for fallback span when target is long

# ------------------------- Text utils -----------------------------------------------
def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def safe_uuid() -> str:
    return str(uuid.uuid4())

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def token_trim(s: str, max_tokens: int = 16) -> str:
    if not s:
        return ""
    toks = s.split()
    if len(toks) <= max_tokens:
        return s.strip()
    return " ".join(toks[:max_tokens]).strip()

def make_hash_key(d: Dict[str, Any]) -> str:
    parts = [
        d.get("source_text", "") or "",
        d.get("target_text", "") or "",
        d.get("semantic_hook", "") or "",
        d.get("citation_hook", "") or "",
        d.get("reference_text", "") or "",
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()

def coerce_item_type(value: Optional[str]) -> str:
    if isinstance(value, str):
        v = value.strip()
        if v in ALLOWED_ITEM_TYPES:
            return v
    return "Other"

# ------------------------- Title detector (strict) -----------------------------------
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

# ------------------------- Citation helpers ------------------------------------------
def looks_like_citation(text: str) -> bool:
    if not text:
        return False
    # "Rule 3.4.1", "Section 58(2)(b)", "Part 4", "9.7.5", "FSMR 58(2)"
    return bool(re.search(r"""(?ix)
        \b(?:rule|section|part|chapter|appendix|schedule)\b
        |
        \bFSMR\b
        |
        \b\d+(?:\.\d+)+(?:\([^)]+\))*\b
    """, text))

def normalize_reftext_as_citation(ref_text: str) -> str:
    if not ref_text:
        return ""
    txt = ref_text.strip()
    return txt if looks_like_citation(txt) else txt

# ---- Strip citations from semantic_hook (while keeping it verbatim-ish & short) -----
CITE_PAT = re.compile(
    r"""(?ix)
    (?:\b(?:rule|section|part|chapter|appendix|schedule)\b
        \s*[:\-]?\s*
        (?:\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)  # 19.10.1(3) etc.
        (?:\s*[a-z])?
    )
    |
    \bFSMR\b
    |
    \b\d+(?:\.\d+)+(?:\([^)]+\))*\b
    """,
)

def sanitize_semantic_hook(hook: str) -> str:
    hook = (hook or "").strip()
    if not hook:
        return ""
    # remove trailing "subject to/under/per ..." clauses (often citation-bearing)
    hook = re.sub(r"\s*\(?(?:see|per|under|subject to)\s+[^)]*\)?$", "", hook, flags=re.I)
    # drop explicit citation tokens
    hook = CITE_PAT.sub("", hook)
    # collapse whitespace and trim punctuation
    hook = re.sub(r"\s+", " ", hook).strip(" ,.;:-")
    # keep a reasonable window (6–12 tokens) if it's too long
    toks = hook.split()
    if len(toks) > 12:
        hook = " ".join(toks[:12])
    return hook

# ---- Try to extract a "core clause" from TARGET when we need a concise fallback -----
CORE_CLAUSE_PAT = re.compile(
    r"""(?ix)
    (?:^|[.]\s+)
    (
      (?:[^.]{10,220}?)
      \b(?:must|shall|may|is\s+required\s+to|subject\s+to|provided\s+that)\b
      [^.]{0,220}
    )
    (?:[.]|$)
    """
)

def pick_core_clause(target_text: str) -> Optional[Tuple[int, int, str]]:
    if not target_text:
        return None
    m = CORE_CLAUSE_PAT.search(target_text)
    if not m:
        # fall back to first sentence up to ~220 chars
        s = target_text.strip()
        # find first sentence boundary
        end = re.search(r"[.?!]", s)
        if end:
            end_idx = min(end.end(), MAX_FREEFORM_CHARS)
        else:
            end_idx = min(len(s), MAX_FREEFORM_CHARS)
        frag = s[:end_idx].rstrip()
        return (0, len(frag), frag) if frag else None
    frag = m.group(1).strip()
    start = target_text.find(frag)
    end = start + len(frag)
    return (start, end, frag)

# ----------------------- LLM Prompts ------------------------------------------------
SYSTEM_PROMPT = """You extract a compact schema from two regulatory passages (SOURCE and TARGET).
Return ONLY a single valid JSON object (no markdown, no extra text).

STRICT RULES:
1) source_item_type and target_item_type ∈ {Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other}.
   • Obligation: must/shall/do required.
   • Prohibition: must not/shall not/forbidden.
   • Permission: may/can/allowed/discretionary authority.
   • Definition: defines a term/category or gives criteria (“X is … if …”).
   • Scope: applicability, exclusions, jurisdiction, who/when applies.
   • Procedure: steps, sequencing, approvals, calculations.
   • Other: everything else.

2) semantic_hook = a short VERBATIM phrase (6–12 tokens) copied from SOURCE that captures the practical substance (policy, action, condition, actor, or exception).
   • MUST NOT contain any citation tokens, numbers that look like rule IDs, section labels, cross-refs, or document scaffolding.
   Good:  "procedures for investigating complaints", "clear division between the Board's responsibility"
   Bad:   "information specified in Rule 2.15.4", "Subject to Rule 20.2.9", "Rule 11.11.1 requirements"

3) citation_hook = the best citation-like token.
   • If ReferenceText looks like a citation (e.g., "Rule 3.4.1", "Section 58(2)"), USE IT verbatim.
   • Else, extract a verbatim citation-like token/phrase from SOURCE (e.g., “Rule 9.7.5”, “Section 61 of FSMR”).

4) answer_spans: up to 3 spans from TARGET, each {text, start, end, type} with type ∈ {DURATION, DATE, MONEY, PERCENT, TERM, SECTION, FREEFORM}.
   • Spans MUST be exact substrings of TARGET with correct 0-based [start,end).
   • Prefer TERM/SECTION/DATE/DURATION/PERCENT/MONEY over FREEFORM.
   • Avoid selecting the entire TARGET unless TARGET is very short (≤220 chars). If TARGET is long, choose the most decision-critical fragment(s).
   • If unsure and TARGET is not a title: provide ONE concise FREEFORM span (≤220 chars). If TARGET is a title: answer_spans=[].

SCHEMA (return exactly this shape):
{{
  "source_item_type": "...",
  "semantic_hook": "...",
  "citation_hook": "...",
  "target_item_type": "...",
  "answer_spans": [
    {{"text": "...", "start": 0, "end": 0, "type": "TERM"}}
  ]
}}
"""


USER_PROMPT_TEMPLATE = """ReferenceType: {reference_type}
ReferenceText: {reference_text}

SOURCE:
passage_id={source_passage_id}
passage_ref={source_passage_ref}
text:
<<<
{source_text}
>>>

TARGET:
passage_id={target_passage_id}
passage_ref={target_passage_ref}
text:
<<<
{target_text}
>>>

Return JSON with this shape:
{{
  "source_item_type": "...",
  "semantic_hook": "...",
  "citation_hook": "...",
  "target_item_type": "...",
  "answer_spans": [
    {{"text": "...", "start": 0, "end": 0, "type": "TERM"}}
  ]
}}
"""

# ----------------------- LLM Wrapper ------------------------------------------------
def build_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai` (>=1.0.0)")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI()

def call_llm(client, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        # try to isolate the first JSON object
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            content = match.group(0)
        return json.loads(content)
    except Exception as e:
        sys.stderr.write(f"[LLM ERROR] {e}\n")
        return {}

# ----------------------- IO ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract merged cross-ref items (pairs) to JSONL (lean schema).")
    p.add_argument("--input_csv", required=True, help="Path to CrossReferenceData.csv")
    p.add_argument("--output_jsonl", required=True, help="Output JSONL file for merged items")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (e.g., gpt-4o, gpt-4o-mini)")
    p.add_argument("--sample_n", type=int, default=None, help="Optional: sample N rows for quick test")
    p.add_argument("--sample_seed", type=int, default=13, help="Seed for sampling")
    p.add_argument("--drop_title_targets", action="store_true",
                   help="If set, skip pairs where the target looks like a title/heading.")
    return p.parse_args()

def read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def sample_rows(rows: List[Dict[str, Any]], n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if not n or n <= 0 or n >= len(rows):
        return rows
    rnd = random.Random(seed)
    idxs = list(range(len(rows)))
    rnd.shuffle(idxs)
    pick = set(idxs[:n])
    return [rows[i] for i in range(len(rows)) if i in pick]

def ensure_outdir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ----------------------- Span validation --------------------------------------------
def answer_span_valid(span: Dict[str, Any], text: str) -> bool:
    try:
        if not isinstance(span, dict):
            return False
        if "text" not in span or "start" not in span or "end" not in span or "type" not in span:
            return False
        if span["type"] not in ALLOWED_SPAN_TYPES:
            return False
        start, end = int(span["start"]), int(span["end"])
        if not (0 <= start < end <= len(text)):
            return False
        return text[start:end] == span["text"]
    except Exception:
        return False

# ----------------------- Build item -------------------------------------------------
def build_merged_item(
    row: Dict[str, Any],
    llm_out: Dict[str, Any],
    model: str
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errs: List[str] = []

    # Pull CSV cols (lenient names to match your existing file)
    reference_type = (row.get("ReferenceType") or "").strip()
    reference_text = (row.get("ReferenceText") or "").strip()

    source_passage_id  = (row.get("SourceID") or "").strip()
    source_passage_ref = (row.get("SourcePassageID") or "").strip()
    source_text        = normalize_whitespace(row.get("SourcePassage") or "")

    target_passage_id  = (row.get("TargetID") or "").strip()
    target_passage_ref = (row.get("TargetPassageID") or "").strip()
    target_text        = normalize_whitespace(row.get("TargetPassage") or "")

    # Hard requirement: have both texts
    if not (source_text and target_text):
        errs.append("skipped: missing valid source or target")
        return None, errs

    # LLM outputs (coerced)
    source_item_type = coerce_item_type(llm_out.get("source_item_type")) if llm_out else "Other"
    target_item_type = coerce_item_type(llm_out.get("target_item_type")) if llm_out else "Other"

    # Hooks
    raw_sem_hook = token_trim((llm_out.get("semantic_hook") or ""), 16) if llm_out else ""
    semantic_hook = sanitize_semantic_hook(raw_sem_hook)

    citation_hook = (llm_out.get("citation_hook") or "") if llm_out else ""
    normalized_ref = normalize_reftext_as_citation(reference_text)
    if normalized_ref:
        citation_hook = normalized_ref  # prefer reference_text if it looks like a citation

    # If semantic_hook vanished after sanitization, try a soft fallback from SOURCE
    if not semantic_hook:
        # pick a mid-length natural phrase from SOURCE
        m = re.search(r"[A-Za-z][A-Za-z ,'\-\(\)]{20,120}", source_text)
        semantic_hook = normalize_whitespace(m.group(0)) if m else ""

    # Validate spans from model (if any)
    valid_spans: List[Dict[str, Any]] = []
    if llm_out:
        spans = llm_out.get("answer_spans") or []
        if isinstance(spans, list):
            for sp in spans[:3]:
                if not isinstance(sp, dict):
                    continue
                try:
                    sp["start"] = int(sp.get("start", -1))
                    sp["end"]   = int(sp.get("end", -1))
                except Exception:
                    continue
                if sp.get("type") not in ALLOWED_SPAN_TYPES:
                    sp["type"] = "FREEFORM"
                if answer_span_valid(sp, target_text):
                    # avoid full-target spans on long targets
                    if sp["type"] == "FREEFORM" and (sp["end"] - sp["start"] == len(target_text)) and len(target_text) > MAX_FREEFORM_CHARS:
                        # skip; we'll add a concise fallback below
                        continue
                    valid_spans.append(sp)

    # Title detection
    target_is_title = is_title_like(target_text)

    # Fallback spans
    if not valid_spans and not target_is_title:
        core = pick_core_clause(target_text)
        if core:
            s, e, frag = core
            if e - s > MAX_FREEFORM_CHARS:
                e = s + MAX_FREEFORM_CHARS
                frag = target_text[s:e].rstrip()
            valid_spans = [{
                "text": frag,
                "start": s,
                "end": e,
                "type": "FREEFORM",
            }]
        else:
            # final fallback: beginning chunk up to MAX_FREEFORM_CHARS
            frag = target_text[:MAX_FREEFORM_CHARS].rstrip()
            valid_spans = [{
                "text": frag,
                "start": 0,
                "end": len(frag),
                "type": "FREEFORM",
            }]

    item: Dict[str, Any] = {
        "item_id": safe_uuid(),
        "reference_type": reference_type or None,
        "reference_text": reference_text or None,

        "semantic_hook": semantic_hook or "",
        "citation_hook": (citation_hook or "").strip(),

        "source_passage_id": source_passage_id or None,
        "source_text": source_text or None,

        "target_passage_id": target_passage_id or None,
        "target_text": target_text or None,

        "source_item_type": source_item_type,
        "target_item_type": target_item_type,

        "answer_spans": valid_spans if not target_is_title else [],
        "target_is_title": bool(target_is_title),

        "provenance": {"model": model, "ts": now_iso_utc()},
    }

    return item, errs

# ----------------------- Runner -----------------------------------------------------
def main():
    args = parse_args()
    ensure_outdir(args.output_jsonl)

    rows = read_rows(args.input_csv)

    # Optional domain-specific filter example (kept from earlier discussions):
    # rows = [r for r in rows if (r.get("ReferenceType") or "").strip().lower() != "outsource"]

    # Sampling if requested
    if args.sample_n:
        rows = sample_rows(rows, args.sample_n, args.sample_seed or 13)

    client = build_client()
    rng = random.Random(args.sample_seed or 13)

    out_items: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    skipped_rows = 0
    title_targets = 0
    errors_found = 0
    seen_hashes: set = set()

    for idx, row in enumerate(rows):
        user_prompt = USER_PROMPT_TEMPLATE.format(
            reference_type=(row.get("ReferenceType") or "").strip(),
            reference_text=(row.get("ReferenceText") or "").strip(),
            source_passage_id=(row.get("SourceID") or "").strip(),
            source_passage_ref=(row.get("SourcePassageID") or "").strip(),
            source_text=(row.get("SourcePassage") or "").strip(),
            target_passage_id=(row.get("TargetID") or "").strip(),
            target_passage_ref=(row.get("TargetPassageID") or "").strip(),
            target_text=(row.get("TargetPassage") or "").strip(),
        )
        llm_json = call_llm(client, args.model, SYSTEM_PROMPT, user_prompt)

        item, errs = build_merged_item(row, llm_json, args.model)
        if item is None:
            details.append({"row": idx, "errors": errs or ["skipped"]})
            skipped_rows += 1
            continue

        # Optionally drop title targets
        if item.get("target_is_title"):
            title_targets += 1
            if args.drop_title_targets:
                details.append({"row": idx, "errors": ["skipped: target_is_title"]})
                skipped_rows += 1
                continue

        # Soft dedupe
        k = make_hash_key(item)
        if k in seen_hashes:
            details.append({"row": idx, "errors": ["deduped: identical content key"]})
            skipped_rows += 1
            continue
        seen_hashes.add(k)

        if errs:
            details.append({"row": idx, "errors": errs})
            errors_found += 1

        out_items.append(item)

        # Gentle pacing
        time.sleep(0.02 + rng.random() * 0.02)

    # Write JSONL
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Summary (light)
    summary = {
        "input_rows": len(rows),
        "items_emitted": len(out_items),
        "skipped_rows": skipped_rows,
        "title_targets_seen": title_targets,
        "drop_title_targets": bool(args.drop_title_targets),
        "model": args.model,
        "timestamp": now_iso_utc(),
        "errors_logged": errors_found,
    }
    report_path = os.path.join(os.path.dirname(os.path.abspath(args.output_jsonl)), "extract_report_merged.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(summary, rf, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
