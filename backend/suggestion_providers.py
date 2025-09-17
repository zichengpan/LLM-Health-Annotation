from __future__ import annotations

import json, os, re
from typing import Dict, Any, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from multiprocessing import resource_tracker
    resource_tracker._CLEANUP_FUNCS.pop('semaphore', None)
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELSET = {
    "entities": ["Disease","Medication","Symptom","Procedure"],
    "relations": [
        {"type":"treats","source":["Medication","Procedure"],"target":["Disease"]},
        {"type":"has_symptom","source":["Disease"],"target":["Symptom"]},
        {"type":"causes","source":["Disease","Medication"],"target":["Disease","Symptom"]},
        {"type":"worsens","source":["Disease","Medication","Symptom"],"target":["Disease","Symptom"]},
        {"type":"indicates","source":["Symptom"],"target":["Disease"]}
    ],
    "overlap_policy": "allow_nested"
}
ALLOWED_ENTITY_TYPES = set(LABELSET["entities"])
ALLOWED_REL_TYPES = {r["type"] for r in LABELSET["relations"]}

_MODEL: AutoModelForCausalLM | None = None
_TOKENIZER: AutoTokenizer | None = None
_CFG: Dict[str, Any] | None = None


def _backend_dir() -> str:
    return os.path.dirname(__file__)


def _cfg_path() -> str:
    return os.path.join(_backend_dir(), "config", "local_llm.json")


def _load_cfg() -> Dict[str, Any]:
    global _CFG
    if _CFG is not None:
        return _CFG
    path = _cfg_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            "Create backend/config/local_llm.json with {\"model_path\": \"../models/your-model\"}"
        )
    with open(path, "r", encoding="utf-8") as f:
        _CFG = json.load(f)
    return _CFG


def _resolve_model_dir(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_backend_dir(), p))


def _load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    cfg = _load_cfg()
    model_dir = _resolve_model_dir(cfg.get("model_path", "../models/llama-3.2-3b-instruct"))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Local model directory does not exist: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=_load_cfg().get("device_map", "auto"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _TOKENIZER, _MODEL = tokenizer, model
    return model, tokenizer


def _format_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a precise clinical IE model. Output only JSON."},
        {"role": "user", "content": prompt},
    ]


def infer_json_from_prompt(prompt: str) -> str:
    model, tokenizer = _load_model()
    cfg = _load_cfg()
    max_new = int(cfg.get("max_new_tokens", 512))

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(_format_messages(prompt), tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

def _read_prompt_template() -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", "prefill_entities_relations.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_relations_prompt_template() -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", "suggest_relations.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _build_prompt(text: str) -> str:
    # LABELSET is embedded into the template already by text; we still keep the object around if needed later.
    return _read_prompt_template().replace("{DOCUMENT_TEXT}", text)

def _build_relations_prompt(text: str, entities: List[Dict[str, Any]]) -> str:
    # Format entities list for the prompt
    entities_text = "\n".join([
        f"- {e['text']} ({e['type']}) [{e['start']}-{e['end']}]"
        for e in entities
    ])

    prompt = _read_relations_prompt_template()
    prompt = prompt.replace("{DOCUMENT_TEXT}", text)
    prompt = prompt.replace("{ENTITIES_LIST}", entities_text)
    return prompt

def _extract_json(s: str) -> Dict[str, Any]:
    """Best-effort JSON parse with fallbacks for partial output."""
    if not isinstance(s, str):
        return {}
    # Strip ```json fences
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)

    # Attempt 1: find first balanced {...}
    depth, start = 0, -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                chunk = s[start:i+1]
                try:
                    return json.loads(chunk)
                except Exception:
                    pass

    # Attempt 2: just the entities array
    m = re.search(r'"entities"\s*:\s*\[', s, flags=re.IGNORECASE)
    if m:
        j = m.end()  # after '['
        depth_br = 1
        k = j
        while k < len(s) and depth_br > 0:
            if s[k] == "[":
                depth_br += 1
            elif s[k] == "]":
                depth_br -= 1
            k += 1
        if depth_br == 0:
            arr_txt = s[j-1:k]
            try:
                ents = json.loads(arr_txt)
                if isinstance(ents, list):
                    return {"entities": ents, "relations": []}
            except Exception:
                pass

    # Attempt 3: naive regex
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _valid_rel_schema(src_type: str, tgt_type: str, rel_type: str) -> bool:
    for rule in LABELSET["relations"]:
        if rule["type"] == rel_type and src_type in rule["source"] and tgt_type in rule["target"]:
            return True
    return False

def _coerce_offsets(obj: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Normalize entity offsets and relation references returned by the model."""
    ents_in = obj.get("entities") or []
    rels_in = obj.get("relations") or []
    n = len(text)

    def find_span(q: str) -> tuple[int, int] | None:
        qq = q.strip().lower()
        if not qq:
            return None
        # exact substring (case-insensitive)
        m = re.search(re.escape(qq), text.lower())
        if m:
            return m.start(), m.end()
        # tolerant hyphen/space/underscore/slash
        toks = re.findall(r"[a-z0-9]+", qq)
        if toks:
            sep = r"[\s\-\_\/]+"
            m = re.search(r"\b" + sep.join(map(re.escape, toks)) + r"\b", text.lower())
            if m:
                return m.start(), m.end()
        return None

    ents: List[Dict[str, Any]] = []
    for e in ents_in:
        etype = str(e.get("type") or "")
        if etype not in ALLOWED_ENTITY_TYPES:
            continue

        # prefer text if given
        t = (e.get("text") or "").strip()
        span = None
        if t:
            span = find_span(t)

        # fallback to offsets if valid
        if not span and isinstance(e.get("start"), int) and isinstance(e.get("end"), int):
            s, k = int(e["start"]), int(e["end"])
            if 0 <= s < k <= n:
                span = (s, k)

        if not span:
            continue

        s, k = span
        ents.append({
            "text": text[s:k],
            "start": s,
            "end": k,
            "type": etype,
            "score": 1.0
        })

    # relations: expect {source,target,type} indices into the (original) array.
    # If model didn't return any, we'll add heuristics below.
    rels: List[Dict[str, Any]] = []
    for r in rels_in or []:
        try:
            src_idx = int(r.get("source"))
            tgt_idx = int(r.get("target"))
            rtype = str(r.get("type") or "")
        except Exception:
            continue
        if rtype not in ALLOWED_REL_TYPES:
            continue
        if not (0 <= src_idx < len(ents) and 0 <= tgt_idx < len(ents)):
            continue
        src, tgt = ents[src_idx], ents[tgt_idx]
        if src_idx == tgt_idx:
            continue
        if not _valid_rel_schema(src["type"], tgt["type"], rtype):
            continue
        rels.append({
            "type": rtype,
            "source_text": src["text"],
            "target_text": tgt["text"],
            "score": 1.0
        })

    return {"entities": ents, "relations": rels}

def _heuristic_relations(text: str, ents: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    rels: List[Dict[str,Any]] = []
    if not ents: return rels

    # sentence segments
    spans = []
    start = 0
    for m in re.finditer(r"[.!?]", text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))

    for s,e in spans:
        sent = text[s:e].lower()
        es = [en for en in ents if s <= en["start"] and en["end"] <= e]
        meds = [en for en in es if en["type"]=="Medication"]
        dis  = [en for en in es if en["type"]=="Disease"]
        syms = [en for en in es if en["type"]=="Symptom"]
        procs = [en for en in es if en["type"]=="Procedure"]

        # "treats" relations
        # Medication treats Disease
        if any(w in sent for w in (" for ", " treat", " prescribed", " started", " began", " given", " administered", " taking", " on ")):
            for m_en in meds:
                for d_en in dis:
                    rels.append({"type":"treats","source_text":m_en["text"],"target_text":d_en["text"],"score":0.7})

        # Procedure treats Disease
        if any(w in sent for w in (" surgery", " operation", " procedure", " treatment", " therapy")):
            for p_en in procs:
                for d_en in dis:
                    rels.append({"type":"treats","source_text":p_en["text"],"target_text":d_en["text"],"score":0.7})

        # "has_symptom" relations
        # Disease has_symptom Symptom
        if any(w in sent for w in (" with ", " presented", " presents", " reports", " reported", " complains", " experiencing", " symptoms", " showed")):
            for d_en in dis:
                for s_en in syms:
                    rels.append({"type":"has_symptom","source_text":d_en["text"],"target_text":s_en["text"],"score":0.7})

        # "causes" relations
        # Disease causes Symptom
        if any(w in sent for w in (" caused", " causes", " resulting", " due to", " leads to", " developed")):
            for d_en in dis:
                for s_en in syms:
                    rels.append({"type":"causes","source_text":d_en["text"],"target_text":s_en["text"],"score":0.6})

        # Medication causes Symptom (side effects)
        if any(w in sent for w in (" side effect", " adverse", " reaction", " after taking", " developed", " caused by")):
            for m_en in meds:
                for s_en in syms:
                    rels.append({"type":"causes","source_text":m_en["text"],"target_text":s_en["text"],"score":0.6})

        # "worsens" relations
        # Disease worsens Disease, Medication worsens Symptom, Symptom worsens Disease
        if any(w in sent for w in (" worsened", " worsens", " aggravated", " exacerbated", " deteriorated", " made worse")):
            for d_en in dis:
                for other_d in dis:
                    if d_en != other_d:
                        rels.append({"type":"worsens","source_text":d_en["text"],"target_text":other_d["text"],"score":0.6})
                for s_en in syms:
                    rels.append({"type":"worsens","source_text":s_en["text"],"target_text":d_en["text"],"score":0.6})
            for m_en in meds:
                for s_en in syms:
                    rels.append({"type":"worsens","source_text":m_en["text"],"target_text":s_en["text"],"score":0.6})

        # "indicates" relations
        # Symptom indicates Disease
        if any(w in sent for w in (" suggests", " indicates", " sign of", " diagnostic", " positive for", " confirms")):
            for s_en in syms:
                for d_en in dis:
                    rels.append({"type":"indicates","source_text":s_en["text"],"target_text":d_en["text"],"score":0.6})

    # dedupe
    seen, uniq = set(), []
    for r in rels:
        key = (r["type"], r["source_text"], r["target_text"])
        if key in seen: continue
        seen.add(key); uniq.append(r)
    return uniq

def hf_local(text: str) -> Dict[str, Any]:
    prompt = _build_prompt(text)
    try:
        raw = infer_json_from_prompt(prompt)
        print("[hf_local] raw (first 400):", raw[:400])
    except Exception as e:
        print("[hf_local] generation error:", repr(e))
        return {"entities": [], "relations": [], "raw": f"ERROR: {e}", "parsed": {}}

    parsed = _extract_json(raw) or {}
    coerced = _coerce_offsets(parsed, text)
    if not coerced["relations"]:
        coerced["relations"] = _heuristic_relations(text, coerced["entities"])
    return {**coerced, "raw": raw[:400], "parsed": parsed}


def suggest_relations_from_entities(text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entities:
        return {"entities": [], "relations": [], "raw": "", "parsed": {}}

    # Filter to only allowed entity types for relations
    relevant_entities = [
        e for e in entities
        if e.get("type") in ["Disease", "Medication", "Symptom", "Procedure"]
    ]

    if len(relevant_entities) < 2:
        return {"entities": entities, "relations": [], "raw": "", "parsed": {}}

    prompt = _build_relations_prompt(text, relevant_entities)
    try:
        raw = infer_json_from_prompt(prompt)
        print("[suggest_relations] raw (first 400):", raw[:400])
    except Exception as e:
        print("[suggest_relations] generation error:", repr(e))
        return {"entities": entities, "relations": [], "raw": f"ERROR: {e}", "parsed": {}}

    parsed = _extract_json(raw) or {}

    # Process suggested relations
    suggested_relations = []
    for rel in (parsed.get("relations") or []):
        source_text = str(rel.get("source_entity", "")).strip()
        target_text = str(rel.get("target_entity", "")).strip()
        rel_type = str(rel.get("type", "")).strip()
        confidence = float(rel.get("confidence", 0.7))

        # Find matching entities
        source_entity = None
        target_entity = None

        for e in relevant_entities:
            if e["text"].lower() == source_text.lower():
                source_entity = e
            if e["text"].lower() == target_text.lower():
                target_entity = e

        if source_entity and target_entity and rel_type in ALLOWED_REL_TYPES:
            # Validate relation schema
            if _valid_rel_schema(source_entity["type"], target_entity["type"], rel_type):
                suggested_relations.append({
                    "type": rel_type,
                    "source_text": source_entity["text"],
                    "target_text": target_entity["text"],
                    "score": confidence
                })

    # Add fallback heuristic relations if LLM didn't find any
    if not suggested_relations:
        heuristic_rels = _heuristic_relations(text, relevant_entities)
        suggested_relations.extend(heuristic_rels)

    return {
        "entities": entities,
        "relations": suggested_relations,
        "raw": raw[:400],
        "parsed": parsed
    }

PROVIDERS = {
    "hf_local": hf_local
}
