from __future__ import annotations

import io
import zlib

import pytest
from PIL import Image, ImageDraw

from acoustic_modem.config import RS_BLOCK
from acoustic_modem.protocol import (
    HEADER_CODED_SIZE,
    HEADER_SIZE,
    PacketHeader,
    build_packets,
    decode_header,
    encode_header,
    fec_decode,
    fec_encode,
    prepare_bands,
)


def make_image(width: int = 320, height: int = 400) -> Image.Image:
    image = Image.new("RGB", (width, height), (20, 60, 90))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        draw.line((0, y, width, y), fill=(20 + y % 180, 50 + (y * 2) % 180, 110))
    draw.rectangle((50, 60, 250, 220), fill=(240, 200, 30))
    draw.ellipse((80, 250, 250, 380), fill=(230, 100, 90))
    return image


def test_header_round_trip_and_crc():
    header = PacketHeader(
        session=0xDEADBEEFCAFEBABE,
        seq=3,
        band_count=9,
        width=80,
        height=104,
        band_y=24,
        band_h=8,
        raw_len=500,
        coded_len=765,
        payload_crc=0x12345678,
    )
    assert len(header.pack()) == HEADER_SIZE
    coded = encode_header(header)
    assert len(coded) == HEADER_CODED_SIZE
    assert decode_header(coded) == header

    broken = bytearray(coded)
    broken[12] ^= 0xFF
    # One damaged byte is within the RS correction budget.
    assert decode_header(bytes(broken)) == header


def test_fec_round_trip_and_correction():
    payload = bytes(range(256)) * 2 + b"acoustic image" * 7
    coded = fec_encode(payload)
    assert len(coded) % RS_BLOCK == 0
    assert len(coded) >= len(payload) + 48
    assert fec_decode(coded, len(payload)) == payload

    broken = bytearray(coded)
    for offset in (5, 77, 180):
        broken[offset] ^= 0xFF
    assert fec_decode(bytes(broken), len(payload)) == payload


def test_prepare_bands_has_expected_geometry():
    image = make_image()
    preview, bands = prepare_bands(image, width=80, band_rows=8, quality=60)
    assert preview.width == 80
    assert preview.height == 100
    packets = build_packets(bands, preview.width, preview.height, band_rows=8, session=7)
    assert len(packets) == 13
    assert [p.header.seq for p in packets] == list(range(13))
    assert sum(p.header.band_h for p in packets) == preview.height
    for packet in packets:
        assert zlib.crc32(packet.payload) & 0xFFFFFFFF == packet.header.payload_crc
        assert packet.header.coded_len % RS_BLOCK == 0
        assert packet.header.raw_len == len(packet.payload)
        assert packet.header.raw_len <= packet.header.coded_len


def test_header_rejects_corruption_beyond_fec():
    image = make_image()
    preview, bands = prepare_bands(image, width=32, band_rows=8, quality=40)
    packet = build_packets(bands, preview.width, preview.height, session=1)[0]
    coded = encode_header(packet.header)
    corrupted = bytes([0]) * len(coded)
    with pytest.raises(ValueError):
        decode_header(corrupted)
