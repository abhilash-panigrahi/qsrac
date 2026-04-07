import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, nullable=False, index=True)
    decision = Column(String, nullable=False)
    risk = Column(String, nullable=False)
    trust = Column(Float, nullable=False)
    # NORMAL   → standard Fast Path request
    # REPAIR   → MFA state-repair transition (/mfa/verify)
    # DEGRADED → Redis unavailable, sensitivity < 3, fail-open path
    event_type = Column(String, nullable=False, default="NORMAL")
    # Wall-clock milliseconds from middleware entry to policy decision.
    # NULL is permitted: REPAIR events and any caller that omits measurement.
    latency_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
