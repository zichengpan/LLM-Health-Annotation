from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(String, nullable=True)  # human-readable title
    external_doc_id = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)
    status = Column(String, default="new")
    meta = Column(JSON, nullable=True)  # free-form metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="documents")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    relations = relationship("Relation", back_populates="document", cascade="all, delete-orphan")

class Entity(Base):
    __tablename__ = "entities"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    type = Column(String, nullable=False)
    codes = Column(JSON, nullable=True)  # standardized codes (SNOMED, RxNorm, etc.)
    annotator_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="entities")
    outgoing_relations = relationship("Relation", foreign_keys="Relation.source_entity_id", back_populates="source_entity")
    incoming_relations = relationship("Relation", foreign_keys="Relation.target_entity_id", back_populates="target_entity")

class Relation(Base):
    __tablename__ = "relations"
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    source_entity_id = Column(String, ForeignKey("entities.id"), nullable=False)
    target_entity_id = Column(String, ForeignKey("entities.id"), nullable=False)
    type = Column(String, nullable=False)
    annotator_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="relations")
    source_entity = relationship("Entity", foreign_keys=[source_entity_id], back_populates="outgoing_relations")
    target_entity = relationship("Entity", foreign_keys=[target_entity_id], back_populates="incoming_relations")
