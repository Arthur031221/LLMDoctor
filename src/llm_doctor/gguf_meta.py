"""Read GGUF key/value metadata without touching tensor data.

The gguf package from llama.cpp builds numpy views over every field, including the
150k-entry tokenizer arrays, which takes about two seconds per file. A doctor that
reads dozens of headers needs something faster, so this reader walks the key/value
section directly and skips large arrays by length. It works on a local file (mmap)
or on the first N bytes of a remote file fetched with an HTTP range request.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# GGUF value types (see GGUFValueType in gguf-py).
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = (
    range(13)
)

_SCALAR = {
    UINT8: "<B",
    INT8: "<b",
    UINT16: "<H",
    INT16: "<h",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    BOOL: "<?",
    UINT64: "<Q",
    INT64: "<q",
    FLOAT64: "<d",
}
_SIZE = {k: struct.calcsize(v) for k, v in _SCALAR.items()}

# Arrays longer than this are skipped and recorded as ArrayInfo.
MAX_KEPT_ARRAY = 1024


class GGUFError(Exception):
    pass


class Truncated(GGUFError):
    """The buffer ended before the metadata section did."""


@dataclass
class ArrayInfo:
    elem_type: int
    count: int

    def __repr__(self) -> str:
        return f"<array type={self.elem_type} n={self.count}>"


@dataclass
class GGUFHeader:
    version: int
    tensor_count: int
    metadata: dict = field(default_factory=dict)
    header_bytes: int = 0

    @property
    def arch(self) -> str | None:
        return self.metadata.get("general.architecture")

    @property
    def chat_template(self) -> str | None:
        v = self.metadata.get("tokenizer.chat_template")
        return v if isinstance(v, str) else None

    @property
    def is_projector(self) -> bool:
        md = self.metadata
        return md.get("general.type") == "mmproj" or self.arch == "clip"

    def get(self, key: str, default=None):
        return self.metadata.get(key, default)

    def arch_key(self, suffix: str, default=None):
        a = self.arch
        if not a:
            return default
        return self.metadata.get(f"{a}.{suffix}", default)


class _Cursor:
    __slots__ = ("buf", "end", "pos")

    def __init__(self, buf) -> None:
        self.buf = buf
        self.pos = 0
        self.end = len(buf)

    def need(self, n: int) -> None:
        if self.pos + n > self.end:
            raise Truncated(f"need {self.pos + n} bytes, have {self.end}")

    def unpack(self, fmt: str, size: int):
        self.need(size)
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def string(self) -> str:
        n = self.unpack("<Q", 8)
        self.need(n)
        raw = bytes(self.buf[self.pos : self.pos + n])
        self.pos += n
        return raw.decode("utf-8", errors="replace")

    def skip_strings(self, count: int) -> None:
        buf, pos, end = self.buf, self.pos, self.end
        unpack_from = struct.unpack_from
        for _ in range(count):
            if pos + 8 > end:
                raise Truncated("string array")
            (n,) = unpack_from("<Q", buf, pos)
            pos += 8 + n
        if pos > end:
            raise Truncated("string array")
        self.pos = pos


def _read_value(cur: _Cursor, vtype: int):
    if vtype in _SCALAR:
        return cur.unpack(_SCALAR[vtype], _SIZE[vtype])
    if vtype == STRING:
        return cur.string()
    if vtype == ARRAY:
        etype = cur.unpack("<I", 4)
        count = cur.unpack("<Q", 8)
        if count > MAX_KEPT_ARRAY:
            if etype == STRING:
                cur.skip_strings(count)
            elif etype in _SIZE:
                n = _SIZE[etype] * count
                cur.need(n)
                cur.pos += n
            else:
                for _ in range(count):
                    _read_value(cur, etype)
            return ArrayInfo(etype, count)
        return [_read_value(cur, etype) for _ in range(count)]
    raise GGUFError(f"unknown GGUF value type {vtype}")


def parse_buffer(buf) -> GGUFHeader:
    """Parse GGUF metadata from a bytes-like object. Raises Truncated if it is too short."""
    cur = _Cursor(buf)
    cur.need(4)
    if bytes(buf[0:4]) != GGUF_MAGIC:
        raise GGUFError("not a GGUF file")
    cur.pos = 4
    version = cur.unpack("<I", 4)
    if version > 0xFFFF:
        raise GGUFError("big-endian GGUF files are not supported")
    if version < 2:
        raise GGUFError(f"GGUF version {version} is too old")
    tensor_count = cur.unpack("<Q", 8)
    kv_count = cur.unpack("<Q", 8)
    md: dict = {}
    for _ in range(kv_count):
        key = cur.string()
        vtype = cur.unpack("<I", 4)
        md[key] = _read_value(cur, vtype)
    return GGUFHeader(version=version, tensor_count=tensor_count, metadata=md, header_bytes=cur.pos)


def is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def read_header(path: Path | str) -> GGUFHeader:
    """Read metadata from a local GGUF file via mmap."""
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as e:  # empty file
            raise GGUFError("empty file") from e
        try:
            return parse_buffer(mm)
        except Truncated as e:
            raise GGUFError("file ends inside the metadata section (incomplete download?)") from e
        finally:
            mm.close()


def scalar(value, default=None):
    """GGUF allows some per-layer values to be arrays. Return a representative scalar."""
    if isinstance(value, list):
        nums = [v for v in value if isinstance(v, (int, float))]
        return max(nums) if nums else default
    if isinstance(value, ArrayInfo):
        return default
    return value if value is not None else default
