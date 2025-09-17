from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class EntityBase(BaseModel):
    document_id: str
    start_offset: int
    end_offset: int
    type: str
    codes: Optional[Dict[str, Any]] = None
    annotator_id: Optional[str] = None

class EntityCreate(EntityBase):
    id: Optional[str] = None

class EntityRead(EntityBase):
    id: str
    created_at: Optional[str] = None

class RelationBase(BaseModel):
    document_id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    annotator_id: Optional[str] = None

class RelationCreate(RelationBase):
    id: Optional[str] = None

class RelationRead(RelationBase):
    id: str
    created_at: Optional[str] = None

class DocumentBase(BaseModel):
    project_id: str
    title: Optional[str] = None
    raw_text: str
    external_doc_id: Optional[str] = None
    status: Optional[str] = "new"
    meta: Optional[Dict[str, Any]] = None

class DocumentCreate(DocumentBase):
    id: Optional[str] = None

class DocumentRead(DocumentBase):
    id: str

class DocumentDetail(DocumentRead):
    entities: List[EntityRead] = Field(default_factory=list)
    relations: List[RelationRead] = Field(default_factory=list)

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = ""

class ProjectCreate(ProjectBase):
    id: Optional[str] = None

class ProjectRead(ProjectBase):
    id: str
