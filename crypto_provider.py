"""
crypto_provider.py — Signing and key-exchange primitives for QSRAC.

PQC availability is detected once at import time.  All original public
function signatures are preserved.  Two new KEM helpers are added for the
correct ML-KEM handshake:

    kem_encapsulate(peer_public_key) -> (ciphertext: bytes, shared_secret: bytes)
    kem_decapsulate(private_key, ciphertext) -> shared_secret: bytes

The login endpoint MUST use kem_encapsulate and return the ciphertext to the
client so the client can call kem_decapsulate and arrive at the same key.
derive_shared_key() is kept for the classical ECDH path and for any caller
that has not yet been updated.

PQC path  : ML-DSA (Dilithium2) for signing, ML-KEM (Kyber512) for KEM.
Classical : Ed25519 for signing, ECDH/SECP256R1 + HKDF for key derivation.
"""

import logging

log = logging.getLogger(__name__)

# ── PQC availability probe ─────────────────────────────────────────────────────

try:
    from pqcrypto.sign import dilithium2
    from pqcrypto.kem import kyber512
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False

if PQC_AVAILABLE:
    log.info("Using PQC (Dilithium + Kyber)")
else:
    log.info("Using classical crypto fallback")

# ── Classical imports (always available) ──────────────────────────────────────

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ec import (
    SECP256R1,
    generate_private_key,
    ECDH,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend


# ── Signing keypair ────────────────────────────────────────────────────────────

def generate_signing_keypair():
    """
    PQC : Dilithium2 keypair — returns (private_key, public_key) as bytes.
    Classical : Ed25519 keypair — returns cryptography key objects.
    """
    if PQC_AVAILABLE:
        public_key, private_key = dilithium2.generate_keypair()
        return private_key, public_key
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


# ── Internal sign / verify ─────────────────────────────────────────────────────

def sign(private_key, data: bytes) -> bytes:
    if PQC_AVAILABLE:
        return dilithium2.sign(data, private_key)
    return private_key.sign(data)


def verify(signature: bytes, data: bytes, public_key) -> bool:
    if PQC_AVAILABLE:
        try:
            dilithium2.verify(data, signature, public_key)
            return True
        except Exception:
            return False
    try:
        public_key.verify(signature, data)
        return True
    except Exception:
        return False


# ── Exchange / KEM keypair ────────────────────────────────────────────────────

def generate_exchange_keypair():
    """
    PQC : Kyber512 keypair — returns (private_key, public_key) as bytes.
    Classical : ECDH/SECP256R1 keypair — returns cryptography key objects.
    """
    if PQC_AVAILABLE:
        public_key, private_key = kyber512.generate_keypair()
        return private_key, public_key
    private_key = generate_private_key(SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key


def derive_shared_key(private_key, peer_public_key) -> bytes:
    """
    Classical ECDH path only — signature preserved for backward compatibility.

    PQC callers MUST use kem_encapsulate / kem_decapsulate instead so the
    ciphertext is exchanged and both sides derive the same shared secret.
    Calling this function when PQC_AVAILABLE raises RuntimeError to prevent
    silent key mismatch.
    """
    if PQC_AVAILABLE:
        raise RuntimeError(
            "derive_shared_key() is not valid for ML-KEM. "
            "Use kem_encapsulate() on the server and kem_decapsulate() on the client."
        )
    shared_key = private_key.exchange(ECDH(), peer_public_key)
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"qsrac-session-key",
        backend=default_backend(),
    ).derive(shared_key)
    return derived


# ── PQC KEM helpers (correct encapsulate / decapsulate flow) ──────────────────

def kem_encapsulate(peer_public_key: bytes) -> tuple[bytes, bytes]:
    """
    Server-side KEM step.

    Encapsulates against the client's Kyber512 public key.
    Returns (ciphertext, shared_secret).

    The caller MUST transmit `ciphertext` to the client (e.g. in the login
    response) so the client can call kem_decapsulate and arrive at the same
    shared_secret.
    """
    if PQC_AVAILABLE:
        ciphertext, shared_secret = kyber512.encrypt(peer_public_key)
        return ciphertext, shared_secret

    raise NotImplementedError(
        "kem_encapsulate() is only valid in PQC mode. "
        "Server must use derive_shared_key() in classical mode."
    )


def kem_decapsulate(private_key: bytes, ciphertext: bytes) -> bytes:
    """
    Client-side KEM step.

    Decapsulates `ciphertext` with the client's Kyber512 private key and
    returns the shared_secret — which must equal the value returned by the
    corresponding kem_encapsulate call on the server.

    In classical mode ciphertext is ignored and the function raises
    NotImplementedError because the ECDH path has no decapsulation concept;
    clients using classical crypto derive the key via ECDH directly.
    """
    if PQC_AVAILABLE:
        shared_secret = kyber512.decrypt(private_key, ciphertext)
        return shared_secret

    raise NotImplementedError(
        "kem_decapsulate() is only valid in PQC mode. "
        "Classical clients derive the shared key via ECDH."
    )


# ── Serialization helpers ──────────────────────────────────────────────────────

def serialize_public_key(public_key) -> bytes:
    """
    PQC: Dilithium public key is already bytes.
    Classical: Ed25519 raw serialization.
    """
    if PQC_AVAILABLE:
        return public_key
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def deserialize_ed25519_public_key(raw_bytes: bytes):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    return Ed25519PublicKey.from_public_bytes(raw_bytes)


def serialize_exchange_public_key(public_key) -> bytes:
    """
    PQC : Kyber512 public key is already bytes — returned as-is.
    Classical : X9.62 uncompressed point encoding.
    """
    if PQC_AVAILABLE:
        return public_key
    return public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )


def deserialize_exchange_public_key(raw_bytes: bytes):
    from cryptography.hazmat.primitives.asymmetric.ec import (
        EllipticCurvePublicKey,
        SECP256R1,
    )
    public_key = EllipticCurvePublicKey.from_encoded_point(SECP256R1(), raw_bytes)
    return public_key


# ── Token sign / verify (public API) ──────────────────────────────────────────

def sign_token(private_key, data: bytes) -> str:
    raw_signature = sign(private_key, data)
    return raw_signature.hex()


def verify_token(signature_hex: str, data: bytes, public_key) -> bool:
    try:
        raw_signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    return verify(raw_signature, data, public_key)