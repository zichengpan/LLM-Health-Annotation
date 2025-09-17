from .db import Base, engine, SessionLocal
from pathlib import Path
from .models import Project, Document


def _load_sample_text() -> str:
    sample_path = Path(__file__).resolve().parent / "sample_documents" / "ten_paragraph_note.txt"
    if sample_path.exists():
        return sample_path.read_text(encoding="utf-8")
    return (
        "The patient was given amoxicillin for pneumonia. "
        "He reported a rash after two days. The physician discontinued the medication."
    )


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        prj = db.get(Project, "prj_default")
        if not prj:
            prj = Project(id="prj_default", name="Default Project", description="Seed project")
            db.add(prj)
            db.commit()
        if not db.query(Document).first():
            sample_text = _load_sample_text()
            doc = Document(
                id="doc_sample",
                project_id=prj.id,
                title="Sample Ward Note",
                raw_text=sample_text,
                status="in_progress",
                meta={"source": "seed", "language": "en"}
            )
            db.add(doc)
            db.commit()
    finally:
        db.close()
if __name__ == "__main__":
    seed()
