import os, json, uuid
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from sqlalchemy import select
from .db import Base, engine, SessionLocal
from . import models, schemas
from .suggestion_providers import PROVIDERS, LABELSET, suggest_relations_from_entities
from .code_lookup import map_text_to_codes

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Medical Annotation Tool API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/labelset")
def get_labelset():
    return LABELSET


@app.get("/projects", response_model=List[schemas.ProjectRead])
def list_projects(db: Session = Depends(get_db)):
    rows = db.execute(select(models.Project)).scalars().all()
    return [schemas.ProjectRead(id=p.id, name=p.name, description=p.description) for p in rows]

@app.post("/projects", response_model=schemas.ProjectRead)
def create_project(payload: schemas.ProjectCreate, db: Session = Depends(get_db)):
    pid = payload.id or uid("prj_")
    proj = models.Project(id=pid, name=payload.name, description=payload.description or "")
    db.add(proj); db.commit(); db.refresh(proj)
    return schemas.ProjectRead(id=proj.id, name=proj.name, description=proj.description)

@app.get("/documents", response_model=List[schemas.DocumentRead])
def list_all_documents(db: Session = Depends(get_db)):
    rows = db.execute(select(models.Document)).scalars().all()
    return [schemas.DocumentRead(id=d.id, project_id=d.project_id, title=d.title, raw_text=d.raw_text, external_doc_id=d.external_doc_id, status=d.status, meta=d.meta) for d in rows]

@app.get("/projects/{project_id}/documents", response_model=List[schemas.DocumentRead])
def list_documents(project_id: str, db: Session = Depends(get_db)):
    rows = db.execute(select(models.Document).where(models.Document.project_id == project_id)).scalars().all()
    return [schemas.DocumentRead(id=d.id, project_id=d.project_id, title=d.title, raw_text=d.raw_text, external_doc_id=d.external_doc_id, status=d.status, meta=d.meta) for d in rows]

@app.post("/documents", response_model=schemas.DocumentRead)
def create_document(payload: schemas.DocumentCreate, db: Session = Depends(get_db)):
    did = payload.id or uid("doc_")
    doc = models.Document(
        id=did, project_id=payload.project_id, title=payload.title,
        raw_text=payload.raw_text, external_doc_id=payload.external_doc_id,
        status=payload.status or "new", meta=payload.meta
    )
    db.add(doc); db.commit(); db.refresh(doc)
    return schemas.DocumentRead(id=doc.id, project_id=doc.project_id, title=doc.title, raw_text=doc.raw_text, external_doc_id=doc.external_doc_id, status=doc.status, meta=doc.meta)

@app.get("/documents/{doc_id}", response_model=schemas.DocumentDetail)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.get(models.Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    entities = [
        schemas.EntityRead(
            id=e.id, document_id=e.document_id, start_offset=e.start_offset, end_offset=e.end_offset,
            type=e.type, codes=e.codes, annotator_id=e.annotator_id, created_at=e.created_at.isoformat() if e.created_at else None
        )
        for e in doc.entities
    ]
    relations = [
        schemas.RelationRead(
            id=r.id, document_id=r.document_id, source_entity_id=r.source_entity_id,
            target_entity_id=r.target_entity_id, type=r.type, annotator_id=r.annotator_id, created_at=r.created_at.isoformat() if r.created_at else None
        )
        for r in doc.relations
    ]
    return schemas.DocumentDetail(
        id=doc.id, project_id=doc.project_id, title=doc.title, raw_text=doc.raw_text, external_doc_id=doc.external_doc_id,
        status=doc.status, entities=entities, relations=relations, meta=doc.meta
    )

@app.post("/entities", response_model=schemas.EntityRead)
def create_entity(payload: schemas.EntityCreate, db: Session = Depends(get_db)):
    if payload.start_offset < 0 or payload.end_offset <= payload.start_offset:
        raise HTTPException(400, "Invalid offsets")
    doc = db.get(models.Document, payload.document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if payload.end_offset > len(doc.raw_text):
        raise HTTPException(400, "Offsets exceed document length")
    span_text = doc.raw_text[payload.start_offset:payload.end_offset]
    codes = payload.codes or map_text_to_codes(span_text, payload.type)
    e = models.Entity(
        id=payload.id or uid("ent_"), document_id=payload.document_id, start_offset=payload.start_offset,
        end_offset=payload.end_offset, type=payload.type, codes=codes, annotator_id=payload.annotator_id
    )
    db.add(e); db.commit(); db.refresh(e)
    return schemas.EntityRead(id=e.id, document_id=e.document_id, start_offset=e.start_offset, end_offset=e.end_offset, type=e.type, codes=e.codes, annotator_id=e.annotator_id, created_at=e.created_at.isoformat())

@app.delete("/entities/{entity_id}")
def delete_entity(entity_id: str, db: Session = Depends(get_db)):
    e = db.get(models.Entity, entity_id)
    if not e:
        raise HTTPException(404, "Entity not found")
    for r in list(e.outgoing_relations) + list(e.incoming_relations):
        db.delete(r)
    db.delete(e); db.commit()
    return {"ok": True}

def _valid_relation(src_type: str, tgt_type: str, rel_type: str) -> bool:
    for rule in LABELSET["relations"]:
        if rule["type"] == rel_type and src_type in rule["source"] and tgt_type in rule["target"]:
            return True
    return False

@app.post("/relations", response_model=schemas.RelationRead)
def create_relation(payload: schemas.RelationCreate, db: Session = Depends(get_db)):
    doc = db.get(models.Document, payload.document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    src = db.get(models.Entity, payload.source_entity_id)
    tgt = db.get(models.Entity, payload.target_entity_id)
    if not (src and tgt):
        raise HTTPException(400, "Source/Target entity not found")
    if src.document_id != doc.id or tgt.document_id != doc.id:
        raise HTTPException(400, "Entities must belong to the same document")
    if src.id == tgt.id:
        raise HTTPException(400, "Relation cannot point to the same entity")
    if not _valid_relation(src.type, tgt.type, payload.type):
        raise HTTPException(400, f"Invalid relation schema: {src.type} --{payload.type}--> {tgt.type}")
    r = models.Relation(
        id=payload.id or uid("rel_"), document_id=payload.document_id, source_entity_id=payload.source_entity_id,
        target_entity_id=payload.target_entity_id, type=payload.type, annotator_id=payload.annotator_id
    )
    db.add(r); db.commit(); db.refresh(r)
    return schemas.RelationRead(id=r.id, document_id=r.document_id, source_entity_id=r.source_entity_id, target_entity_id=r.target_entity_id, type=r.type, annotator_id=r.annotator_id, created_at=r.created_at.isoformat())

@app.delete("/relations/{relation_id}")
def delete_relation(relation_id: str, db: Session = Depends(get_db)):
    r = db.get(models.Relation, relation_id)
    if not r:
        raise HTTPException(404, "Relation not found")
    db.delete(r); db.commit()
    return {"ok": True}

def _export_project_dict(proj: models.Project):
    docs = []
    for d in proj.documents:
        docs.append({
            "document_id": d.id,
            "title": d.title,
            "text": d.raw_text,
            "entities": [
                {"id": e.id, "start": e.start_offset, "end": e.end_offset, "type": e.type, "codes": e.codes, "annotator_id": e.annotator_id, "created_at": (e.created_at.isoformat() if e.created_at else None)}
                for e in d.entities
            ],
            "relations": [
                {"id": r.id, "source": r.source_entity_id, "target": r.target_entity_id, "type": r.type, "direction": "source_to_target", "annotator_id": r.annotator_id, "created_at": (r.created_at.isoformat() if r.created_at else None)}
                for r in d.relations
            ],
            "meta": {"project_id": proj.id, **(d.meta or {})}
        })
    return docs

@app.get("/projects/{project_id}/export/json")
def export_project_json(project_id: str, db: Session = Depends(get_db)):
    proj = db.get(models.Project, project_id)
    if not proj:
        raise HTTPException(404, "Project not found")
    docs = _export_project_dict(proj)
    return JSONResponse(content=docs)

@app.post("/suggestions/run")
def run_suggestions(document_id: str, provider: str = "hf_local", db: Session = Depends(get_db)):
    provider = "hf_local"
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        print(f"[suggestions] provider={provider}, doc_id={document_id} start")
        result = PROVIDERS["hf_local"](doc.raw_text)
        ok = bool(result.get("entities") or result.get("relations"))
        print(f"[suggestions] done; entities={len(result.get('entities', []))}, relations={len(result.get('relations', []))}")
        return {"ok": ok, "provider": provider, "result": result}
    except Exception as e:
        print("[suggestions] error:", repr(e))
        return {"ok": False, "provider": provider, "error": str(e), "result": {"entities": [], "relations": [], "raw": ""}}

@app.post("/suggestions/relations")
def suggest_relations(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    existing_entities = [
        {
            "text": doc.raw_text[e.start_offset:e.end_offset],
            "start": e.start_offset,
            "end": e.end_offset,
            "type": e.type,
            "id": e.id
        }
        for e in doc.entities
    ]

    if len(existing_entities) < 2:
        return {
            "ok": False,
            "error": "Need at least 2 entities to suggest relations",
            "result": {"entities": existing_entities, "relations": [], "raw": ""}
        }

    try:
        print(f"[suggest_relations] doc_id={document_id}, entities={len(existing_entities)} start")
        result = suggest_relations_from_entities(doc.raw_text, existing_entities)
        ok = bool(result.get("relations"))
        print(f"[suggest_relations] done; relations={len(result.get('relations', []))}")
        return {"ok": ok, "result": result}
    except Exception as e:
        print("[suggest_relations] error:", repr(e))
        return {"ok": False, "error": str(e), "result": {"entities": existing_entities, "relations": [], "raw": ""}}


def _serialize_document(doc):
    entities, relations = [], []
    if hasattr(doc, "entities"):
        for e in doc.entities:
            entities.append({
                "id": getattr(e, "id", None),
                "text": getattr(e, "text", ""),
                "type": getattr(e, "type", ""),
                "start": getattr(e, "start_offset", getattr(e, "start", None)),
                "end": getattr(e, "end_offset", getattr(e, "end", None)),
                "codes": getattr(e, "codes", None),
            })
    if hasattr(doc, "relations"):
        for r in doc.relations:
            relations.append({
                "id": getattr(r, "id", None),
                "type": getattr(r, "type", ""),
                "source_text": getattr(getattr(r, "source", None), "text", None) or getattr(r, "source_text", ""),
                "target_text": getattr(getattr(r, "target", None), "text", None) or getattr(r, "target_text", ""),
            })
    return {
        "document_id": getattr(doc, "id", None),
        "title": getattr(doc, "title", None),
        "text": getattr(doc, "raw_text", ""),
        "entities": entities,
        "relations": relations,
        "meta": {
            "annotator": getattr(doc, "annotator", None),
            "project_id": getattr(doc, "project_id", "prj_default"),
        },
    }

@app.get("/export/json")
def export_json(db: Session = Depends(get_db)):
    docs = db.query(models.Document).all()
    return [_serialize_document(d) for d in docs]

@app.get("/projects/{project_id}/export/json")
def export_json_project(project_id: str, db: Session = Depends(get_db)):
    q = db.query(models.Document)
    if hasattr(models.Document, "project_id"):
        q = q.filter(models.Document.project_id == project_id)
    docs = q.all()
    return [_serialize_document(d) for d in docs]
