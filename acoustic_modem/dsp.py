"""OFDM transmitter and receiver for the acoustic image modem.

The physical layer is intentionally simple, but it has all the pieces needed to
survive a real speaker -> air -> microphone path:

* a linear chirp for packet detection,
* four known OFDM training symbols for channel estimation,
* 48 Reed-Solomon parity bytes per 255-byte block,
* a bit interleaver so a frequency notch does not destroy one RS block,
* per-symbol pilot phase tracking for clock/oscillator drift.

A packet carries one horizontal image band.  Bands are independent, so the
receiver can paint each one as soon as it has been recovered.
"""

from __future__ import annotations

import math
import wave
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import find_peaks, fftconvolve

from .config import (
    ALL_BINS,
    CP_SAMPLES,
    DATA_BINS,
    FFT_SIZE,
    FREQ_REPETITION,
    HEADER_BITS_PER_SYMBOL,
    HEADER_COPIES,
    LTF_SYMBOLS,
    PACKET_GAP_SECONDS,
    PAYLOAD_BITS_PER_SYMBOL,
    PAYLOAD_GROUPS,
    PILOT_BINS,
    PILOTS_IN_ALL,
    SAMPLE_RATE,
    SYMBOL_SAMPLES,
    SYNC_F0,
    SYNC_F1,
    SYNC_GAP_SAMPLES,
    SYNC_SAMPLES,
)
from .protocol import (
    HEADER_CODED_SIZE,
    BandPacket,
    PacketHeader,
    decode_header,
    encode_header,
    fec_decode,
    fec_encode,
)

# PILOTS_IN_ALL is imported from config (indices into ALL_BINS).
try:
    PILOTS_IN_ALL = np.asarray(PILOTS_IN_ALL, dtype=np.intp)
except NameError:  # pragma: no cover - compatibility with older config layout
    PILOTS_IN_ALL = np.searchsorted(ALL_BINS, PILOT_BINS)

_DATA_MASK = ~np.isin(ALL_BINS, PILOT_BINS)
_DATA_BIN_POSITIONS = np.flatnonzero(_DATA_MASK)
_PILOT_BIN_POSITIONS = np.flatnonzero(~_DATA_MASK)
_DATA_BIN_INDEX = DATA_BINS.astype(np.float64)

_SYNC_FADE = 0.008
_OFDM_RMS = 0.075
_OFDM_PEAK = 0.72
_DEFAULT_TIMING_OFFSETS = tuple(sorted(range(-256, 257, 8), key=abs))


class DecodeError(RuntimeError):
    """Raised when a packet cannot be recovered."""


@dataclass(frozen=True)
class DecodedPacket:
    header: PacketHeader
    payload: bytes
    sync_start: int

    @property
    def duration_samples(self) -> int:
        return packet_duration_samples(self.header)


@dataclass(frozen=True)
class HeaderProbe:
    """Everything learned from the preamble and header of one packet."""

    header: PacketHeader
    channel: np.ndarray
    header_start: int
    sync_start: int
    timing_offset: int

    @property
    def duration_samples(self) -> int:
        return packet_duration_samples(self.header)


def packet_duration_samples(header: PacketHeader) -> int:
    header_syms = header_symbols()
    payload_syms = payload_symbols(header.coded_len)
    return (
        SYNC_SAMPLES
        + SYNC_GAP_SAMPLES
        + (LTF_SYMBOLS + HEADER_COPIES * header_syms + payload_syms)
        * SYMBOL_SAMPLES
    )


# ---------------------------------------------------------------------------
# Waveform construction


def _raised_cosine(n: int) -> np.ndarray:
    if n <= 0:
        return np.ones(0)
    return np.hanning(2 * n)[:n]


def make_chirp() -> np.ndarray:
    """Linear 0.12 s chirp, faded at both ends to avoid clicks."""
    n = SYNC_SAMPLES
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    duration = n / SAMPLE_RATE
    slope = (SYNC_F1 - SYNC_F0) / duration
    phase = 2.0 * np.pi * (SYNC_F0 * t + 0.5 * slope * t * t)
    chirp = 0.55 * np.sin(phase)

    fade_n = int(_SYNC_FADE * SAMPLE_RATE)
    fade = np.ones(n)
    ramp = _raised_cosine(fade_n)
    fade[:fade_n] = ramp
    fade[-fade_n:] = ramp[::-1]
    chirp *= fade
    return chirp.astype(np.float64)


CHIRP = make_chirp()
_CHIRP_ENERGY = float(np.sum(CHIRP * CHIRP))


def _ltf_spectrum() -> np.ndarray:
    rng = np.random.default_rng(0x5EED_2026)
    bits = rng.integers(0, 2, size=(len(ALL_BINS), 2), dtype=np.int8)
    values = (1.0 - 2.0 * bits[:, 0]) + 1j * (1.0 - 2.0 * bits[:, 1])
    values /= np.sqrt(2.0)
    spec = np.zeros(FFT_SIZE, dtype=np.complex128)
    spec[ALL_BINS] = values
    spec[(-ALL_BINS) % FFT_SIZE] = np.conj(values)
    return spec


LTF_SPECTRUM = _ltf_spectrum()
LTF_VALUES = LTF_SPECTRUM[ALL_BINS].copy()


def _symbol_spectrum(values: np.ndarray) -> np.ndarray:
    spec = np.zeros(FFT_SIZE, dtype=np.complex128)
    spec[ALL_BINS] = values
    spec[(-ALL_BINS) % FFT_SIZE] = np.conj(values)
    return spec


def _ofdm_samples(spec: np.ndarray) -> np.ndarray:
    body = np.fft.ifft(spec).real
    # Equalize the RMS of every OFDM symbol.  Without this the high-PAPR chirp
    # dominates a packet-level normalizer and leaves the data symbols tens of
    # dB too quiet at the microphone.
    rms = float(np.sqrt(np.mean(body * body)))
    peak = float(np.max(np.abs(body)))
    if rms > 0 and peak > 0:
        scale = min(_OFDM_RMS / rms, _OFDM_PEAK / peak)
        body = body * scale
    return np.concatenate((body[-CP_SAMPLES:], body))


def _ltf_samples() -> np.ndarray:
    symbol = _ofdm_samples(LTF_SPECTRUM)
    return np.tile(symbol, LTF_SYMBOLS)


def _qpsk_values(bits: np.ndarray) -> np.ndarray:
    pairs = bits.reshape(-1, 2)
    values = (1.0 - 2.0 * pairs[:, 0]) + 1j * (1.0 - 2.0 * pairs[:, 1])
    return values / np.sqrt(2.0)


def _qpsk_bits(values: np.ndarray) -> np.ndarray:
    bits = np.empty(values.size * 2, dtype=np.uint8)
    bits[0::2] = (values.real < 0).astype(np.uint8)
    bits[1::2] = (values.imag < 0).astype(np.uint8)
    return bits


def _bits_from_bytes(data: bytes) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes_from_bits(bits: np.ndarray) -> bytes:
    usable = (len(bits) // 8) * 8
    if usable == 0:
        return b""
    return np.packbits(bits[:usable].astype(np.uint8)).tobytes()


def header_symbols() -> int:
    return math.ceil(HEADER_CODED_SIZE * 8 / HEADER_BITS_PER_SYMBOL)


def payload_symbols(coded_len: int) -> int:
    return math.ceil(coded_len * 8 / PAYLOAD_BITS_PER_SYMBOL)


def _packet_seed(header: PacketHeader) -> int:
    return (
        header.session
        ^ ((header.seq + 1) * 0x9E3779B97F4A7C15)
        ^ (header.coded_len << 17)
    ) & 0xFFFFFFFFFFFFFFFF


def _interleaver(header: PacketHeader) -> np.ndarray:
    n_bits = payload_symbols(header.coded_len) * PAYLOAD_BITS_PER_SYMBOL
    rng = np.random.Generator(np.random.PCG64(_packet_seed(header)))
    return rng.permutation(n_bits)


def _pad_bits(data: bytes, total_bits: int) -> np.ndarray:
    bits = _bits_from_bytes(data)
    if len(bits) > total_bits:
        raise ValueError("data does not fit in symbol allocation")
    if len(bits) < total_bits:
        bits = np.concatenate((bits, np.zeros(total_bits - len(bits), dtype=np.uint8)))
    return bits


def _make_data_symbols(header: PacketHeader, coded_payload: bytes) -> np.ndarray:
    n_symbols = payload_symbols(len(coded_payload))
    total_bits = n_symbols * PAYLOAD_BITS_PER_SYMBOL
    bits = _pad_bits(coded_payload, total_bits)
    bits = bits[_interleaver(header)]
    values = _qpsk_values(bits).reshape(n_symbols, PAYLOAD_GROUPS)
    out = np.empty((n_symbols, SYMBOL_SAMPLES), dtype=np.float64)
    for i in range(n_symbols):
        used = np.zeros(len(ALL_BINS), dtype=np.complex128)
        for repeat in range(FREQ_REPETITION):
            start = repeat * PAYLOAD_GROUPS
            used[_DATA_BIN_POSITIONS[start : start + PAYLOAD_GROUPS]] = values[i]
        used[_PILOT_BIN_POSITIONS] = 1.0 + 0.0j
        spec = np.zeros(FFT_SIZE, dtype=np.complex128)
        spec[ALL_BINS] = used
        spec[(-ALL_BINS) % FFT_SIZE] = np.conj(used)
        out[i] = _ofdm_samples(spec)
    return out


def _make_header_symbols(header: PacketHeader) -> np.ndarray:
    coded = encode_header(header)
    n_symbols = header_symbols()
    total_bits = n_symbols * HEADER_BITS_PER_SYMBOL
    bits = _pad_bits(coded, total_bits)
    values = _qpsk_values(bits).reshape(n_symbols, -1)
    out = np.empty((n_symbols, SYMBOL_SAMPLES), dtype=np.float64)
    for i in range(n_symbols):
        used = np.empty(len(ALL_BINS), dtype=np.complex128)
        used[_DATA_BIN_POSITIONS] = values[i]
        used[_PILOT_BIN_POSITIONS] = 1.0 + 0.0j
        spec = np.zeros(FFT_SIZE, dtype=np.complex128)
        spec[ALL_BINS] = used
        spec[(-ALL_BINS) % FFT_SIZE] = np.conj(used)
        out[i] = _ofdm_samples(spec)
    return out


def packet_waveform(packet: BandPacket, *, amplitude: float = 0.72) -> np.ndarray:
    header = packet.header
    coded = fec_encode(packet.payload)
    if len(coded) != header.coded_len:
        raise AssertionError("header/payload FEC length mismatch")

    header_block = _make_header_symbols(header)
    data_block = _make_data_symbols(header, coded)

    gap = np.zeros(SYNC_GAP_SAMPLES, dtype=np.float64)
    parts = [
        CHIRP,
        gap,
        _ltf_samples(),
    ]
    for _ in range(HEADER_COPIES):
        parts.append(header_block.ravel())
    parts.append(data_block.ravel())

    body = np.concatenate(parts)
    body *= amplitude
    body = np.clip(body, -1.0, 1.0)

    fade_n = int(0.004 * SAMPLE_RATE)
    if len(body) > 2 * fade_n:
        body[:fade_n] *= np.linspace(0.0, 1.0, fade_n)
        body[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    return body.astype(np.float32)


def build_waveform(
    packets: Sequence[BandPacket],
    *,
    amplitude: float = 0.72,
    gap_seconds: float = PACKET_GAP_SECONDS,
) -> np.ndarray:
    gap = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for packet in packets:
        pieces.append(packet_waveform(packet, amplitude=amplitude))
        pieces.append(gap)
    if pieces:
        pieces.append(np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Receiver


def _local_norm_correlation(signal: np.ndarray) -> np.ndarray:
    if len(signal) < len(CHIRP):
        return np.empty(0, dtype=np.float64)
    corr = fftconvolve(signal, CHIRP[::-1], mode="valid")
    energy = np.convolve(signal * signal, np.ones(len(CHIRP)), mode="valid")
    denom = np.sqrt(np.maximum(energy * _CHIRP_ENERGY, 1e-12))
    return corr / denom


def find_sync_peaks(
    audio: np.ndarray,
    *,
    threshold: float = 0.19,
    min_gap_seconds: float = 0.35,
) -> np.ndarray:
    """Find likely chirp starts, ordered by time."""
    signal = np.asarray(audio, dtype=np.float64)
    if len(signal) < len(CHIRP):
        return np.empty(0, dtype=np.int64)
    # Pad the edges so a transmission that starts at sample 0 is not missed by
    # scipy's local-maximum detector.  Correlator indices are shifted back by
    # the amount of left padding.
    pad = len(CHIRP)
    padded = np.concatenate((np.zeros(pad), signal, np.zeros(pad)))
    normalized = np.abs(_local_norm_correlation(padded))
    if normalized.size == 0:
        return np.empty(0, dtype=np.int64)
    distance = max(1, int(min_gap_seconds * SAMPLE_RATE))
    peaks, _ = find_peaks(normalized, height=threshold, distance=distance)
    peaks = peaks - pad
    peaks = peaks[(peaks >= 0) & (peaks < len(signal))]
    if peaks.size:
        # Keep peaks inside a sane dynamic range so a loud echo does not
        # suppress the actual packet.
        values = normalized[peaks + pad]
        strong = peaks[values >= max(threshold, 0.65 * values.max())]
        if strong.size:
            peaks = strong
    return peaks.astype(np.int64)


def _fft_symbol(audio: np.ndarray, symbol_start: int) -> np.ndarray:
    begin = symbol_start + CP_SAMPLES
    end = begin + FFT_SIZE
    if begin < 0 or end > len(audio):
        raise DecodeError("capture ended inside a symbol")
    return np.fft.fft(audio[begin:end])[ALL_BINS]


def _estimate_channel(audio: np.ndarray, ltf_start: int) -> np.ndarray:
    spectra = []
    for i in range(LTF_SYMBOLS):
        spectra.append(_fft_symbol(audio, ltf_start + i * SYMBOL_SAMPLES))
    reference = spectra[0]
    aligned = [reference]
    for spec in spectra[1:]:
        rotation = np.angle(np.vdot(reference, spec))
        aligned.append(spec * np.exp(-1j * rotation))
    average = np.mean(aligned, axis=0)
    return average / LTF_VALUES


def _phase_fit(equalized: np.ndarray, channel: np.ndarray) -> tuple[float, float]:
    """Robust common-phase + delay fit from the pilot subcarriers.

    Adjacent pilot phase *differences* are wrapped to (-pi, pi) individually.
    This avoids the classic unwrap failure where one noisy pilot shifts every
    later phase by 2*pi and sends the delay estimate off the rails.
    """
    zp = equalized[PILOTS_IN_ALL]
    hp = channel[PILOTS_IN_ALL]
    good = np.abs(hp) > max(1e-9, 0.18 * float(np.median(np.abs(hp))))
    if np.count_nonzero(good) < 6:
        good = np.ones_like(good, dtype=bool)
    z = zp[good]
    k = PILOT_BINS[good].astype(np.float64)
    if len(z) < 2:
        return float(np.angle(np.sum(zp * np.conj(hp)))), 0.0

    dz = np.angle(z[1:] * np.conj(z[:-1]))
    dk = np.diff(k)
    raw_slope = dz / dk
    weights = np.maximum(np.abs(z[1:]) * np.abs(z[:-1]), 1e-9)
    # Reject isolated bad pilots, then use a weighted mean of the robust
    # differences.  The natural slope is small (few radians per 1 kHz).
    median = float(np.median(raw_slope))
    mad = float(np.median(np.abs(raw_slope - median))) + 1e-9
    keep = np.abs(raw_slope - median) < max(0.08, 5.0 * mad)
    if np.count_nonzero(keep) >= 2:
        raw_slope = raw_slope[keep]
        weights = weights[keep]
    slope = float(np.sum(raw_slope * weights) / np.sum(weights))

    centred = np.angle(z * np.exp(-1j * slope * k))
    intercept = float(np.angle(np.sum(np.exp(1j * centred) * np.maximum(np.abs(hp[good]), 1e-9))))
    return intercept, slope


def _soft_equalize(
    spectrum: np.ndarray, channel: np.ndarray
) -> tuple[np.ndarray, float, float]:
    raw = spectrum * np.conj(channel)
    intercept, slope = _phase_fit(raw, channel)
    reg = 0.003 * float(np.mean(np.abs(channel) ** 2)) + 1e-12
    values = raw / (np.abs(channel) ** 2 + reg)
    values = values * np.exp(-1j * (intercept + slope * ALL_BINS))
    return values, intercept, slope


def _demod_symbols(
    audio: np.ndarray,
    data_start: int,
    n_symbols: int,
    channel: np.ndarray,
) -> np.ndarray:
    """Demodulate header symbols (no frequency repetition)."""
    bits: list[np.ndarray] = []
    for i in range(n_symbols):
        spectrum = _fft_symbol(audio, data_start + i * SYMBOL_SAMPLES)
        values, _, _ = _soft_equalize(spectrum, channel)
        bits.append(_qpsk_bits(values[_DATA_BIN_POSITIONS]))
    if not bits:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate(bits)


def _demod_payload_symbols(
    audio: np.ndarray,
    data_start: int,
    n_symbols: int,
    channel: np.ndarray,
) -> np.ndarray:
    """Demodulate payload symbols and combine the frequency-diversity copies.

    Maximal-ratio combining in the frequency domain means a notch in one copy
    contributes almost nothing, while the other copy still decides the symbol.
    """
    bits: list[np.ndarray] = []
    for i in range(n_symbols):
        spectrum = _fft_symbol(audio, data_start + i * SYMBOL_SAMPLES)
        raw = spectrum * np.conj(channel)
        intercept, slope = _phase_fit(raw, channel)
        raw = raw * np.exp(-1j * (intercept + slope * ALL_BINS))
        values = raw[_DATA_BIN_POSITIONS]
        usable = PAYLOAD_GROUPS * FREQ_REPETITION
        combined = values[:usable].reshape(FREQ_REPETITION, PAYLOAD_GROUPS).sum(axis=0)
        bits.append(_qpsk_bits(combined))
    if not bits:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate(bits)


def _decode_header_at(
    audio: np.ndarray, start: int, channel: np.ndarray
) -> PacketHeader | None:
    symbols_per_header = header_symbols()
    for copy in range(HEADER_COPIES):
        copy_start = start + copy * symbols_per_header * SYMBOL_SAMPLES
        try:
            bits = _demod_symbols(audio, copy_start, symbols_per_header, channel)
        except DecodeError:
            continue
        coded = _bytes_from_bits(bits[: HEADER_CODED_SIZE * 8])
        if len(coded) != HEADER_CODED_SIZE:
            continue
        try:
            return decode_header(coded)
        except ValueError:
            continue
    return None


def _decode_payload(
    audio: np.ndarray, start: int, header: PacketHeader, channel: np.ndarray
) -> bytes:
    n_symbols = payload_symbols(header.coded_len)
    start = start + HEADER_COPIES * header_symbols() * SYMBOL_SAMPLES
    interleaved = _demod_payload_symbols(audio, start, n_symbols, channel)
    total_bits = n_symbols * PAYLOAD_BITS_PER_SYMBOL
    if len(interleaved) < total_bits:
        raise DecodeError("capture ended inside payload")
    orig = np.empty_like(interleaved[:total_bits])
    orig[_interleaver(header)] = interleaved[:total_bits]
    coded = _bytes_from_bits(orig[: header.coded_len * 8])
    if len(coded) != header.coded_len:
        raise DecodeError("short coded payload")
    return fec_decode(coded, header.raw_len)


def iter_header_probes(
    audio: np.ndarray,
    sync_start: int,
    *,
    timing_offsets: Iterable[int] | None = None,
) -> Iterable[HeaderProbe]:
    """Yield every timing hypothesis whose header passes CRC/FEC."""
    audio = np.asarray(audio, dtype=np.float64)
    if sync_start < 0 or sync_start + SYNC_SAMPLES > len(audio):
        raise DecodeError("no chirp at requested offset")

    if timing_offsets is None:
        timing_offsets = _DEFAULT_TIMING_OFFSETS

    ltf_start = sync_start + SYNC_SAMPLES + SYNC_GAP_SAMPLES
    for offset in timing_offsets:
        try:
            channel = _estimate_channel(audio, ltf_start + offset)
            header_start = ltf_start + offset + LTF_SYMBOLS * SYMBOL_SAMPLES
            header = _decode_header_at(audio, header_start, channel)
        except (DecodeError, ValueError):
            continue
        if header is None:
            continue
        yield HeaderProbe(
            header=header,
            channel=channel,
            header_start=header_start,
            sync_start=sync_start,
            timing_offset=offset,
        )


def probe_header(
    audio: np.ndarray,
    sync_start: int,
    *,
    timing_offsets: Iterable[int] | None = None,
) -> HeaderProbe:
    """Recover just the header, which is enough to know the packet length."""
    try:
        return next(iter_header_probes(audio, sync_start, timing_offsets=timing_offsets))
    except StopIteration as exc:
        raise DecodeError("header did not decode") from exc


def decode_probe(audio: np.ndarray, probe: HeaderProbe) -> DecodedPacket:
    """Decode the payload after a successful :func:`probe_header` call."""
    audio = np.asarray(audio, dtype=np.float64)
    payload = _decode_payload(audio, probe.header_start, probe.header, probe.channel)
    if (zlib.crc32(payload) & 0xFFFFFFFF) != probe.header.payload_crc:
        raise DecodeError("payload CRC mismatch")
    return DecodedPacket(
        header=probe.header,
        payload=payload,
        sync_start=probe.sync_start,
    )


def decode_packet(
    audio: np.ndarray,
    sync_start: int,
    *,
    timing_offsets: Iterable[int] | None = None,
) -> DecodedPacket:
    """Decode the packet whose chirp begins at ``sync_start``.

    The matched-filter peak can land on different parts of a room's impulse
    response.  Rather than trusting the first header that survives FEC, try
    each plausible FFT alignment and accept only a payload whose CRC passes.
    """
    audio = np.asarray(audio, dtype=np.float64)
    last_error: Exception | None = None
    for probe in iter_header_probes(audio, sync_start, timing_offsets=timing_offsets):
        try:
            return decode_probe(audio, probe)
        except (DecodeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise DecodeError(f"payload did not decode ({last_error})")
    raise DecodeError("header did not decode")


def decode_audio(
    audio: np.ndarray,
    *,
    threshold: float = 0.19,
    progress: callable | None = None,
) -> list[DecodedPacket]:
    """Decode every packet in a complete recording."""
    audio = np.asarray(audio, dtype=np.float64)
    peaks = find_sync_peaks(audio, threshold=threshold)
    packets: list[DecodedPacket] = []
    consumed = -1
    for index, peak in enumerate(peaks):
        if peak <= consumed:
            continue
        try:
            packet = decode_packet(audio, int(peak))
        except DecodeError:
            continue
        packets.append(packet)
        consumed = packet.sync_start + packet.duration_samples
        if progress is not None:
            progress(index + 1, len(peaks), packet)
    return packets


def wav_to_float(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("only 16-bit WAV files are supported")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        # The modem is designed for 48 kHz.  Linear interpolation is adequate
        # for a one-off imported recording; live capture is always 48 kHz.
        duration = len(data) / rate
        new_n = int(round(duration * SAMPLE_RATE))
        old_t = np.linspace(0.0, duration, num=len(data), endpoint=False)
        new_t = np.linspace(0.0, duration, num=new_n, endpoint=False)
        data = np.interp(new_t, old_t, data).astype(np.float32)
    return data
