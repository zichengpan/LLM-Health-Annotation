import json, os, re
from typing import Optional, Dict, Any

_CACHE = None

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\W_]+$', '', s)
    return s

def _load() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = os.path.join(os.path.dirname(__file__), 'codes', 'lookup.json')
    if not os.path.exists(path):
        _CACHE = {}
    else:
        with open(path, 'r', encoding='utf-8') as f:
            _CACHE = json.load(f)
    return _CACHE

def map_text_to_codes(text: str, entity_type: str) -> Optional[Dict[str, Any]]:
    table = _load()
    etab = table.get(entity_type) or {}
    key = _normalize(text)
    # exact match
    if key in etab:
        return etab[key]
    return None
