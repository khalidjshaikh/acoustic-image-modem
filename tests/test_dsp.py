from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

from acoustic_modem.config import SAMPLE_RATE
from acoustic_modem.dsp import (
    build_waveform,
    decode_audio,
    decode_packet,
    find_sync_peaks,
    wav_to_float,
    write_wav,
)
from acoustic_modem.protocol import build_packets, prepare_bands


def make_image(width: int = 240, height: int = 300) -> Image.Image:
    image = Image.new("RGB", (width, height), (10, 30, 70))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        draw.line((0, y, width, y), fill=(30 + y % 150, 110, 160 - y % 100))
    draw.rectangle((30, 40, 200, 150), fill=(255, 210, 30))
    draw.ellipse((60, 180, 200, 280), fill=(220, 80, 100))
    return image


@pytest.fixture(scope="module")
def transmission():
    image = make_image()
    preview, bands = prepare_bands(image, width=48, band_rows=8, quality=50)
    packets = build_packets(bands, preview.width, preview.height, band_rows=8, session=0x1234)
    wave = build_waveform(packets, amplitude=0.72)
    return packets, wave


def test_clean_decode_every_packet(transmission):
    packets, wave = transmission
    peaks = find_sync_peaks(wave)
    assert len(peaks) == len(packets)
    decoded = [decode_packet(wave, int(peak)) for peak in peaks]
    assert [packet.header.seq for packet in decoded] == list(range(len(packets)))
    for original, recovered in zip(packets, decoded):
        assert recovered.payload == original.payload
        assert recovered.header == original.header


def test_decode_audio_with_leading_silence(transmission):
    packets, wave = transmission
    audio = np.concatenate((np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32), wave))
    decoded = decode_audio(audio)
    assert len(decoded) == len(packets)


def test_noise_and_clock_skew(transmission):
    packets, wave = transmission
    rng = np.random.default_rng(42)
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    received = wave.astype(np.float64) + rng.normal(0, rms * 0.18, len(wave))
    received = resample_poly(received, 1_000_080, 1_000_000)
    peaks = find_sync_peaks(received)
    assert len(peaks) == len(packets)
    for peak in peaks:
        assert decode_packet(received, int(peak)).payload


def test_short_room_reflections(transmission):
    packets, wave = transmission
    rng = np.random.default_rng(7)
    impulse = np.zeros(220)
    impulse[0] = 1.0
    impulse[95] = 0.35
    impulse[190] = 0.18
    received = fftconvolve(wave, impulse, mode="full")[: len(wave)]
    sos = butter(4, [300, 10_000], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    received = sosfilt(sos, received)
    rms = float(np.sqrt(np.mean(received * received)))
    received += rng.normal(0, rms * 0.10, len(received))
    peaks = find_sync_peaks(received)
    assert len(peaks) == len(packets)
    for peak in peaks:
        assert decode_packet(received, int(peak)).payload


def test_wav_round_trip(tmp_path, transmission):
    _, wave = transmission
    path = tmp_path / "modem.wav"
    write_wav(path, wave)
    loaded = wav_to_float(path)
    assert len(loaded) == len(wave)
    assert np.max(np.abs(loaded - wave)) < 0.001
