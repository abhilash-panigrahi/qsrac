
import asyncio
import logging
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from database import SessionLocal
from models import AuditLog
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def log_event(
    session_id: str,
    decision: str,
    risk: str,
    trust: float,
    event_type: str = "NORMAL",
    latency_ms: float | None = None,
):
    """
    Synchronous write to PostgreSQL.  Always called via run_in_executor —
    never on the event loop thread.

    SQLAlchemy errors are logged but not re-raised: audit must never block
    or crash the request path.  The log entry makes the failure observable
    for monitoring/alerting without surfacing to the caller.
    """
    db = SessionLocal()
    try:
        entry = AuditLog(
            session_id=session_id,
            decision=decision,
            risk=risk,
            trust=trust,
            event_type=event_type,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(entry)
        db.commit()
    except SQLAlchemyError as e:
        log.error(
            "Audit write failed [session=%s decision=%s event_type=%s]: %s",
            session_id, decision, event_type, e,
        )
        db.rollback()
    finally:
        db.close()


async def log_event_async(
    session_id: str,
    decision: str,
    risk: str,
    trust: float,
    event_type: str = "NORMAL",
    latency_ms: float | None = None,
):
    """
    Non-blocking audit write.  Offloads the synchronous DB call to the
    default thread-pool executor so the event loop is never blocked.

    All parameters default so existing call sites that omit event_type /
    latency_ms continue to work without modification.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        log_event,
        session_id,
        decision,
        risk,
        trust,
        event_type,
        latency_ms,
    )
