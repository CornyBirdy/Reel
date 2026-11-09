"""Read the metadata header of a GGUF file (layer count, context length, ...) without loading any weights.

Used to pick sensible load settings and to estimate how much memory a model needs.
Only the key/value section at the start of the file is read; tensors are never touched.
"""

from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"GGUF"
# value type -> (struct format, size)
_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
            6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
STRING, ARRAY = 8, 9
SMALL_ARRAY = 1024  # numeric arrays up to this length are returned (per-layer head counts); longer ones are skipped


class GGUFError(ValueError):
    pass


def _read(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError("The file ended unexpectedly: it is not a complete GGUF file.")
    return b


def _u32(f) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _u64(f) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _string(f) -> str:
    n = _u64(f)
    if n > 1 << 24:
        raise GGUFError("Corrupt GGUF header (string too long).")
    return _read(f, n).decode("utf-8", "replace")


def _value(f, vtype: int):
    if vtype in _SCALARS:
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, _read(f, size))[0]
    if vtype == STRING:
        return _string(f)
    if vtype == ARRAY:
        etype, count = _u32(f), _u64(f)
        if etype in _SCALARS:
            size = _SCALARS[etype][1]
            if count <= SMALL_ARRAY:
                fmt = _SCALARS[etype][0][1]
                return list(struct.unpack(f"<{count}{fmt}", _read(f, size * count)))
            f.seek(size * count, 1)
            return None
        if etype == STRING:
            for _ in range(count):       # the token list is long, but this only reads each length and hops over the text
                n = _u64(f)
                f.seek(n, 1)
            return None
        raise GGUFError("Unsupported GGUF array type.")
    raise GGUFError(f"Unknown GGUF value type {vtype}.")


def read_header(path: Path) -> dict:
    """All plain key/value pairs up to the tokenizer section (which comes after everything we need)."""
    out: dict = {}
    with open(path, "rb") as f:
        if _read(f, 4) != MAGIC:
            raise GGUFError("This is not a GGUF file.")
        version = _u32(f)
        if version < 2:
            raise GGUFError(f"GGUF version {version} is too old.")
        _u64(f)                       # tensor count
        count = _u64(f)
        for _ in range(count):
            key = _string(f)
            vtype = _u32(f)
            if key.startswith("tokenizer.") and any(k.endswith(".block_count") for k in out):
                break                 # everything we care about is written before the tokenizer
            out[key] = _value(f, vtype)
    return out


def _num(v):
    if isinstance(v, list):
        v = [x for x in v if x]
        return max(v) if v else None
    return v


def describe(path: Path) -> dict:
    """The facts Reel needs: architecture, layers, context length, KV-cache geometry."""
    kv = read_header(path)
    arch = kv.get("general.architecture")
    if not arch:
        raise GGUFError("The GGUF file has no architecture entry.")

    def get(name):
        return kv.get(f"{arch}.{name}")

    heads = _num(get("attention.head_count"))
    emb = get("embedding_length")
    kv_heads = _num(get("attention.head_count_kv")) or heads
    key_len = get("attention.key_length") or (emb // heads if emb and heads else None)
    val_len = get("attention.value_length") or key_len
    return {
        "architecture": arch,
        "name": kv.get("general.name"),
        "size_label": kv.get("general.size_label"),
        "layers": get("block_count"),
        "context_length": get("context_length"),
        "embedding": emb,
        "heads": heads,
        "kv_heads": kv_heads,
        "key_length": key_len,
        "value_length": val_len,
        "experts": get("expert_count"),
        "file_type": kv.get("general.file_type"),
    }


def kv_cache_bytes(meta: dict, context: int, bytes_per_value: int = 2) -> int:
    """Memory the attention cache takes at this context length (f16 by default). An estimate: models with
    sliding-window or recurrent layers need less than this."""
    layers, kv_heads = meta.get("layers"), meta.get("kv_heads")
    if not (layers and kv_heads and meta.get("key_length")):
        return int(context * 0.0005 * 1e9 / 1000)   # unknown geometry: ~0.5 MB per token
    per_token = layers * kv_heads * (meta["key_length"] + (meta.get("value_length") or meta["key_length"])) * bytes_per_value
    return int(per_token * context)
