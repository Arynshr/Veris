"""Deterministic content-derived identifiers (no uuid, so reruns reproduce artifacts)."""

import hashlib


def stable_id(*parts: object, length: int = 12) -> str:
    payload = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :length
    ]
