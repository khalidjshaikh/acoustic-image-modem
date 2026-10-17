"""Image framing, packet headers and forward error correction."""

from __future__ import annotations

import io
import secrets
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

from .config import (
    DEFAULT_BAND_ROWS,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_JPEG_QUALITY,
    HEADER_MAGIC,
    HEADER_PARITY,
    PAYLOAD_DATA_PER_BLOCK,
    PAYLOAD_PARITY,
    PROTOCOL_VERSION,
    RS_BLOCK,
)

HEADER_PREFIX = struct.Struct(">4sBBQHHHHHHIII")
HEADER_SIZE = HEADER_PREFIX.size + 4  # prefix + CRC32
HEADER_CODED_SIZE = HEADER_SIZE + HEADER_PARITY


@lru_cache(maxsize=2)
def _payload_codec(parity: int = PAYLOAD_PARITY) -> RSCodec:
    return RSCodec(parity)


@lru_cache(maxsize=1)
def _header_codec() -> RSCodec:
    return RSCodec(HEADER_PARITY)


@dataclass(frozen=True)
class PacketHeader:
    session: int
    seq: int
    band_count: int
    width: int
    height: int
    band_y: int
    band_h: int
    raw_len: int
    coded_len: int
    payload_crc: int

    def pack(self) -> bytes:
        prefix = HEADER_PREFIX.pack(
            HEADER_MAGIC,
            PROTOCOL_VERSION,
            0,
            self.session,
            self.seq,
            self.band_count,
            self.width,
            self.height,
            self.band_y,
            self.band_h,
            self.raw_len,
            self.coded_len,
            self.payload_crc,
        )
        return prefix + struct.pack(">I", zlib.crc32(prefix) & 0xFFFFFFFF)

    @classmethod
    def unpack(cls, data: bytes) -> "PacketHeader":
        if len(data) != HEADER_SIZE:
            raise ValueError(f"header must be {HEADER_SIZE} bytes")
        prefix = data[:-4]
        (crc,) = struct.unpack(">I", data[-4:])
        if (zlib.crc32(prefix) & 0xFFFFFFFF) != crc:
            raise ValueError("header CRC mismatch")
        (
            magic,
            version,
            _flags,
            session,
            seq,
            band_count,
            width,
            height,
            band_y,
            band_h,
            raw_len,
            coded_len,
            payload_crc,
        ) = HEADER_PREFIX.unpack(prefix)
        if magic != HEADER_MAGIC:
            raise ValueError("not an acoustic image packet")
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {version}")
        if not (0 < width <= 4096 and 0 < height <= 4096):
            raise ValueError("invalid image dimensions")
        if band_count == 0 or seq >= band_count:
            raise ValueError("invalid band numbering")
        if band_y + band_h > height:
            raise ValueError("band extends past image")
        if coded_len == 0 or coded_len % RS_BLOCK:
            raise ValueError("invalid coded payload length")
        return cls(
            session=session,
            seq=seq,
            band_count=band_count,
            width=width,
            height=height,
            band_y=band_y,
            band_h=band_h,
            raw_len=raw_len,
            coded_len=coded_len,
            payload_crc=payload_crc,
        )

    @property
    def key(self) -> tuple[int, int]:
        return self.session, self.seq


@dataclass(frozen=True)
class BandPacket:
    header: PacketHeader
    payload: bytes  # JPEG bytes for this horizontal band


def encode_header(header: PacketHeader) -> bytes:
    encoded = bytes(_header_codec().encode(header.pack()))
    if len(encoded) != HEADER_CODED_SIZE:
        raise AssertionError("unexpected header FEC size")
    return encoded


def decode_header(encoded: bytes) -> PacketHeader:
    if len(encoded) != HEADER_CODED_SIZE:
        raise ValueError("unexpected coded header size")
    try:
        decoded, _, _ = _header_codec().decode(encoded)
    except ReedSolomonError as exc:
        raise ValueError("header FEC failed") from exc
    return PacketHeader.unpack(bytes(decoded))


def fec_encode(payload: bytes) -> bytes:
    if not payload:
        raise ValueError("empty payload")
    pad = (-len(payload)) % PAYLOAD_DATA_PER_BLOCK
    padded = payload + b"\x00" * pad
    codec = _payload_codec()
    return b"".join(bytes(codec.encode(padded[i : i + PAYLOAD_DATA_PER_BLOCK]))
                    for i in range(0, len(padded), PAYLOAD_DATA_PER_BLOCK))


def fec_decode(coded: bytes, raw_len: int) -> bytes:
    if not coded or len(coded) % RS_BLOCK:
        raise ValueError("invalid coded payload length")
    codec = _payload_codec()
    chunks: list[bytes] = []
    try:
        for i in range(0, len(coded), RS_BLOCK):
            decoded, _, _ = codec.decode(coded[i : i + RS_BLOCK])
            chunks.append(bytes(decoded))
    except ReedSolomonError as exc:
        raise ValueError("payload FEC failed") from exc
    payload = b"".join(chunks)[:raw_len]
    if len(payload) != raw_len:
        raise ValueError("payload shorter than declared")
    return payload


def prepare_bands(
    image: Image.Image,
    *,
    width: int = DEFAULT_IMAGE_WIDTH,
    band_rows: int = DEFAULT_BAND_ROWS,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[Image.Image, list[bytearray]]:
    """Resize the picture and compress it as a stack of independent JPEG bands."""
    if width < 16:
        raise ValueError("image width must be at least 16 pixels")
    if band_rows < 1:
        raise ValueError("band_rows must be positive")

    image = image.convert("RGB")
    height = max(1, round(width * image.height / image.width))
    preview = image.resize((width, height), Image.Resampling.LANCZOS)

    bands: list[bytearray] = []
    for y in range(0, height, band_rows):
        strip = preview.crop((0, y, width, min(height, y + band_rows)))
        out = io.BytesIO()
        strip.save(
            out,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling=2,
            progressive=False,
        )
        bands.append(bytearray(out.getvalue()))
    return preview, bands


def build_packets(
    bands: Iterable[bytes],
    width: int,
    height: int,
    *,
    band_rows: int = DEFAULT_BAND_ROWS,
    session: int | None = None,
    band_count: int | None = None,
) -> list[BandPacket]:
    band_list = list(bands)
    total = band_count if band_count is not None else len(band_list)
    if total != len(band_list):
        raise ValueError("band_count does not match number of bands")
    session_id = secrets.randbits(64) if session is None else session

    packets: list[BandPacket] = []
    for seq, payload in enumerate(band_list):
        if len(payload) > 65535:
            raise ValueError("band payload too large")
        coded_len = len(fec_encode(bytes(payload)))
        y = seq * band_rows
        header = PacketHeader(
            session=session_id,
            seq=seq,
            band_count=total,
            width=width,
            height=height,
            band_y=y,
            band_h=min(band_rows, height - y),
            raw_len=len(payload),
            coded_len=coded_len,
            payload_crc=zlib.crc32(payload) & 0xFFFFFFFF,
        )
        packets.append(BandPacket(header, bytes(payload)))
    return packets


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.copy()


def assemble_image(bands: Iterable[tuple[PacketHeader, bytes]]) -> Image.Image:
    """Turn received JPEG bands back into one image (used for saving a copy)."""
    entries = list(bands)
    if not entries:
        raise ValueError("no bands received")
    width = entries[0][0].width
    height = entries[0][0].height
    canvas = Image.new("RGB", (width, height), (7, 10, 14))
    for header, payload in sorted(entries, key=lambda entry: entry[0].seq):
        with Image.open(io.BytesIO(payload)) as strip:
            canvas.paste(strip.convert("RGB"), (0, header.band_y))
    return canvas


def packet_duration_seconds(header: PacketHeader) -> float:
    """Samples from chirp start through the last data symbol, in seconds."""
    from .config import (
        HEADER_BITS_PER_SYMBOL,
        HEADER_COPIES,
        LTF_SYMBOLS,
        PAYLOAD_BITS_PER_SYMBOL,
        SYMBOL_SECONDS,
        SYNC_GAP_SECONDS,
        SYNC_SECONDS,
    )

    header_symbols = (HEADER_CODED_SIZE * 8 + HEADER_BITS_PER_SYMBOL - 1) // HEADER_BITS_PER_SYMBOL
    data_symbols = (header.coded_len * 8 + PAYLOAD_BITS_PER_SYMBOL - 1) // PAYLOAD_BITS_PER_SYMBOL
    return (
        SYNC_SECONDS
        + SYNC_GAP_SECONDS
        + (LTF_SYMBOLS + HEADER_COPIES * header_symbols + data_symbols)
        * SYMBOL_SECONDS
    )
