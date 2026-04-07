"""
rate_limiter.py — Redis token bucket rate limiter for QSRAC.

Design constraints
──────────────────
• Uses a *separate* Lua script from the Redis Lua gate in redis_lua.py.
  Key namespace:  rl:{session_id}:{bucket}
  Gate namespace: session:{session_id}
  These never overlap.

• Non-blocking: on Redis ConnectionError the check is skipped (fail-open).
  Callers decide whether that is acceptable for their endpoint.

• Two pre-configured buckets:
    "general" — 60 req / 60 s  (1 req/s sustained, burst up to 60)
    "mfa"     —  5 req / 60 s

• The Lua script is atomic: read-modify-write cannot race.
  It is loaded once via SCRIPT LOAD and cached by SHA.
"""

import time
from dataclasses import dataclass
from redis_lua import get_redis_client

# ── Bucket configurations ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class BucketConfig:
    capacity: int        # max tokens
    refill_rate: float   # tokens added per second
    window_seconds: int  # TTL for the Redis key


BUCKETS: dict[str, BucketConfig] = {
    "general": BucketConfig(capacity=60, refill_rate=1.0,        window_seconds=120),
    "mfa":     BucketConfig(capacity=5,  refill_rate=5.0 / 60.0, window_seconds=120),
}

# ── Lua token-bucket script ────────────────────────────────────────────────────
# KEYS[1]  = rl key  (e.g. "rl:abc123:mfa")
# ARGV[1]  = current unix timestamp (float, as string)
# ARGV[2]  = bucket capacity        (int)
# ARGV[3]  = refill_rate            (tokens/sec, float)
# ARGV[4]  = window_seconds TTL     (int)
#
# Returns:
#   "1"  → allowed  (token consumed)
#   "0"  → rejected (bucket empty)

_TOKEN_BUCKET_LUA = """
local key         = KEYS[1]
local now         = tonumber(ARGV[1])
local capacity    = tonumber(ARGV[2])
local refill_rate = tonumber(ARGV[3])
local window      = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens      = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
    tokens      = capacity
    last_refill = now
end

local elapsed = now - last_refill
if elapsed < 0 then elapsed = 0 end

local refilled = elapsed * refill_rate
tokens = tokens + refilled
if tokens > capacity then tokens = capacity end

if tokens < 1 then
    redis.call('EXPIRE', key, window)
    return "0"
end

tokens = tokens - 1
redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, window)
return "1"
"""

_bucket_script_sha: str | None = None


def _load_bucket_script() -> str:
    global _bucket_script_sha
    if _bucket_script_sha is None:
        client = get_redis_client()
        _bucket_script_sha = client.script_load(_TOKEN_BUCKET_LUA)
    return _bucket_script_sha


# ── Public API ─────────────────────────────────────────────────────────────────

def check_rate_limit(session_id: str, bucket: str = "general") -> bool:
    """
    Consume one token from the named bucket for this session.

    Returns
    -------
    True  → request is allowed
    False → rate limit exceeded; caller should return HTTP 429

    On Redis ConnectionError the function returns True (fail-open) so that
    a Redis outage does not also block all traffic.  The Redis Lua gate in
    redis_lua.py handles the authoritative replay / integrity check; rate
    limiting is a best-effort defence layer.
    """
    cfg = BUCKETS.get(bucket)
    if cfg is None:
        raise ValueError(f"Unknown rate-limit bucket: {bucket!r}")

    try:
        client = get_redis_client()
        sha    = _load_bucket_script()
        key    = f"rl:{session_id}:{bucket}"
        now    = time.time()

        result = client.evalsha(
            sha,
            1,            # number of KEYS
            key,          # KEYS[1]
            now,          # ARGV[1]
            cfg.capacity, # ARGV[2]
            cfg.refill_rate,   # ARGV[3]
            cfg.window_seconds # ARGV[4]
        )
        return result == "1"

    except ConnectionError:
        # Fail-open: Redis down → let request through.
        # The gate Lua script will still enforce integrity.
        return True
    except Exception:
        # Any other unexpected error → fail-open for same reason.
        return True
