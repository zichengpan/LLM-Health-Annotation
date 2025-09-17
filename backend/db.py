from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQLITE_URL = "sqlite:///./app.db"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
