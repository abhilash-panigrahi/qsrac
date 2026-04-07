import config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# Construct the SQLAlchemy connection string
DATABASE_URL = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
# pool_pre_ping=True helps maintain connection stability [cite: 227]
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency for getting a database session in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()