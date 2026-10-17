"""Speaker -> microphone loopback self-test.

The modem is only as good as the audio path.  This module plays a few known
chirps, records them through the microphone, and reports whether the signal
came back.  It then sends one small modem packet so the complete modem path
(chirp sync, channel estimate, FEC, CRC) is exercised as well.

The analysis functions are deliberately separate from playback so they can be
tested offline with synthetic captures.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .audio_io import device_name, resolve_device, sounddevice_module
from .config import SAMPLE_RATE, SYNC_SAMPLES
from .dsp import (
    CHIRP,
    DecodeError,
    decode_audio,
    decode_packet,
    find_sync_peaks,
    normalized_chirp_correlation,
    packet_waveform,
    write_wav,
)
from .protocol import BandPacket, build_packets, prepare_bands

PRE_ROLL_SECONDS = 0.30
CHIRP_GAP_SECONDS = 0.30
# Duplex streams can delay the speaker by several hundred milliseconds relative
# to the recording.  The tail must outlast that offset or the packet is cut off.
POST_ROLL_SECONDS = 1.20
# A laptop microphone sitting next to its own speaker clips easily.  Start
# quietly; the normalized chirp detector does not need a loud signal.
DEFAULT_LOOPBACK_VOLUME = 0.05
_PACKET_SESSION = 0xA11CE


@dataclass(frozen=True)
class LoopbackPlan:
    """Where each test signal should appear in a perfectly timed capture."""

    expected_chirps: int
    chirp_starts: tuple[int, ...]
    pre_roll_samples: int
    packet_start: int | None
    packet_session: int | None
    packet_size: int | None

    @property
    def expected_syncs(self) -> int:
        """Probe chirps plus the packet's own sync chirp."""
        return self.expected_chirps + (1 if self.packet_start is not None else 0)


@dataclass(frozen=True)
class LoopbackResult:
    """Outcome of analysing one loopback recording."""

    plan: LoopbackPlan
    capture: np.ndarray
    detected_chirps: int
    peak_correlation: float
    first_chirp_offset_seconds: float | None
    capture_rms: float
    capture_peak: float
    snr_db: float | None
    packet_decoded: bool | None
    packet_bytes: int | None
    packet_error: str | None

    @property
    def chirps_ok(self) -> bool:
        # Extra peaks are treated as harmless echoes, not a failure.
        return self.detected_chirps >= self.plan.expected_chirps

    @property
    def packet_ok(self) -> bool:
        return self.packet_decoded is not False

    @property
    def clipped(self) -> bool:
        return self.capture_peak >= 0.99

    @property
    def ok(self) -> bool:
        return self.chirps_ok and self.packet_ok


def _probe_packet(amplitude: float) -> tuple[BandPacket, np.ndarray]:
    """Build one small, self-contained image band for the packet test."""
    pixels = np.zeros((24, 16, 3), dtype=np.uint8)
    pixels[..., 0] = np.linspace(24, 232, 24, dtype=np.uint8)[:, None]
    pixels[..., 1] = 96
    pixels[..., 2] = 200
    image = Image.fromarray(pixels, mode="RGB")
    preview, bands = prepare_bands(image, width=24, band_rows=16, quality=55)
    packets = build_packets(
        bands,
        preview.width,
        preview.height,
        band_rows=16,
        session=_PACKET_SESSION,
    )
    packet = packets[0]
    return packet, packet_waveform(packet, amplitude=amplitude)


def build_loopback_waveform(
    *,
    chirps: int = 3,
    amplitude: float = DEFAULT_LOOPBACK_VOLUME,
    include_packet: bool = True,
) -> tuple[np.ndarray, LoopbackPlan]:
    """Create the signal that will be played through the speaker."""
    if chirps < 1:
        raise ValueError("chirps must be at least 1")
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude must be between 0 and 1")

    pre_roll = np.zeros(int(PRE_ROLL_SECONDS * SAMPLE_RATE), dtype=np.float32)
    gap = np.zeros(int(CHIRP_GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    chirp = np.clip(CHIRP * amplitude, -1.0, 1.0).astype(np.float32)

    pieces: list[np.ndarray] = [pre_roll]
    starts: list[int] = []
    cursor = len(pre_roll)
    for _ in range(chirps):
        starts.append(cursor)
        pieces.extend((chirp, gap))
        cursor += len(chirp) + len(gap)

    packet_start: int | None = None
    packet_session: int | None = None
    packet_size: int | None = None
    if include_packet:
        packet, packet_audio = _probe_packet(amplitude)
        packet_start = cursor
        packet_session = packet.header.session
        packet_size = len(packet.payload)
        pieces.append(packet_audio)
        cursor += len(packet_audio)

    pieces.append(np.zeros(int(POST_ROLL_SECONDS * SAMPLE_RATE), dtype=np.float32))
    waveform = np.concatenate(pieces).astype(np.float32)
    plan = LoopbackPlan(
        expected_chirps=chirps,
        chirp_starts=tuple(starts),
        pre_roll_samples=len(pre_roll),
        packet_start=packet_start,
        packet_session=packet_session,
        packet_size=packet_size,
    )
    return waveform, plan


def record_loopback(
    waveform: np.ndarray,
    *,
    input_device: int | None = None,
    output_device: int | None = None,
) -> np.ndarray:
    """Play ``waveform`` and record the microphone at the same time."""
    sd = sounddevice_module()
    device = None
    if input_device is not None or output_device is not None:
        device = (input_device, output_device)
    capture = sd.playrec(
        np.asarray(waveform, dtype=np.float32),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        latency=0.05,
        blocking=True,
    )
    return np.asarray(capture, dtype=np.float64).reshape(-1)


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def _snr_db(
    signal: np.ndarray, plan: LoopbackPlan, peaks: np.ndarray
) -> float | None:
    if peaks.size == 0 or plan.pre_roll_samples <= 0:
        return None
    noise = _rms(signal[: plan.pre_roll_samples])
    start = int(peaks[0])
    level = _rms(signal[start : start + SYNC_SAMPLES])
    if noise <= 1e-7:
        return math.inf if level > 1e-6 else None
    return 20.0 * math.log10(max(level, 1e-12) / noise)


def _packet_failure(
    signal: np.ndarray, plan: LoopbackPlan, peaks: np.ndarray
) -> str:
    if peaks.size <= plan.expected_chirps:
        return "packet chirp was not detected"
    try:
        decode_packet(signal, int(peaks[plan.expected_chirps]))
    except (DecodeError, ValueError) as exc:
        return str(exc)
    return "packet decoded but did not match the loopback session"


def analyze_capture(
    capture: np.ndarray,
    plan: LoopbackPlan,
    *,
    threshold: float = 0.19,
) -> LoopbackResult:
    """Check a recorded loopback capture for chirps and the modem packet."""
    signal = np.asarray(capture, dtype=np.float64).reshape(-1)
    peaks = find_sync_peaks(signal, threshold=threshold)
    correlation = normalized_chirp_correlation(signal)
    peak_correlation = float(np.max(correlation)) if correlation.size else 0.0

    first_offset: float | None = None
    if peaks.size and plan.chirp_starts:
        first_offset = (int(peaks[0]) - plan.chirp_starts[0]) / SAMPLE_RATE

    packet_decoded: bool | None = None
    packet_bytes: int | None = None
    packet_error: str | None = None
    if plan.packet_session is not None:
        decoded = decode_audio(signal, threshold=threshold)
        matches = [
            packet
            for packet in decoded
            if packet.header.session == plan.packet_session
        ]
        packet_decoded = bool(matches)
        if matches:
            packet_bytes = len(matches[0].payload)
        else:
            packet_error = _packet_failure(signal, plan, peaks)

    return LoopbackResult(
        plan=plan,
        capture=signal,
        detected_chirps=int(peaks.size),
        peak_correlation=peak_correlation,
        first_chirp_offset_seconds=first_offset,
        capture_rms=_rms(signal),
        capture_peak=float(np.max(np.abs(signal))) if signal.size else 0.0,
        snr_db=_snr_db(signal, plan, peaks),
        packet_decoded=packet_decoded,
        packet_bytes=packet_bytes,
        packet_error=packet_error,
    )


def run_loopback(
    *,
    input_device: int | None = None,
    output_device: int | None = None,
    amplitude: float = DEFAULT_LOOPBACK_VOLUME,
    threshold: float = 0.19,
    chirps: int = 3,
    include_packet: bool = True,
    save_wav: str | Path | None = None,
) -> LoopbackResult:
    """Play and record the loopback signal, saving the capture if requested."""
    waveform, plan = build_loopback_waveform(
        chirps=chirps,
        amplitude=amplitude,
        include_packet=include_packet,
    )
    capture = record_loopback(
        waveform,
        input_device=input_device,
        output_device=output_device,
    )
    if save_wav is not None:
        write_wav(save_wav, capture)
    return analyze_capture(capture, plan, threshold=threshold)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play test chirps through the speaker and verify the microphone hears them."
    )
    parser.add_argument("--device", help="use this device for both input and output")
    parser.add_argument("--input-device", help="input device index or part of its name")
    parser.add_argument("--output-device", help="output device index or part of its name")
    parser.add_argument("--volume", type=float, default=DEFAULT_LOOPBACK_VOLUME, help="playback amplitude, 0..1")
    parser.add_argument("--threshold", type=float, default=0.19, help="chirp detector threshold")
    parser.add_argument("--chirps", type=int, default=3, help="number of test chirps (default: 3)")
    parser.add_argument("--no-packet", action="store_true", help="skip the full modem packet test")
    parser.add_argument("--wav", type=Path, help="save the microphone capture as WAV")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    return parser


def _db(value: float) -> str:
    if value <= 0.0:
        return "-inf dBFS"
    return f"{20.0 * math.log10(value):+.1f} dBFS"


def _print_banner(
    args,
    input_index: int | None,
    output_index: int | None,
    duration: float,
) -> None:
    print()
    print("  ACOUSTIC LOOPBACK  /  speaker -> microphone")
    print("  " + "-" * 56)
    print(f"  output      {device_name(output_index, kind='output')}")
    print(f"  input       {device_name(input_index, kind='input')}")
    probe = f"{args.chirps} chirps" + ("" if args.no_packet else " + 1 modem packet")
    print(f"  probe       {probe}  ({duration:.1f} s)")
    print(f"  level       {args.volume:.2f} · threshold {args.threshold:.2f}")
    if args.wav:
        print(f"  capture     {args.wav}")
    print("  " + "-" * 56)
    print("  Playing now; keep the room quiet and do not mute the speaker.")
    print()


def _print_result(result: LoopbackResult) -> None:
    print(
        f"  chirps      {result.detected_chirps}/{result.plan.expected_syncs} heard · "
        f"peak correlation {result.peak_correlation:.2f}"
    )
    if result.first_chirp_offset_seconds is not None:
        offset_ms = result.first_chirp_offset_seconds * 1000.0
        print(f"  timing      first chirp at {offset_ms:+.0f} ms (device + air delay)")
    level = (
        f"  level       peak {_db(result.capture_peak)} · RMS {_db(result.capture_rms)}"
    )
    if result.snr_db is not None:
        snr = "clean" if math.isinf(result.snr_db) else f"{result.snr_db:.0f} dB"
        level += f" · SNR {snr}"
    print(level)
    if result.clipped:
        print("  warning     the input is clipping; lower --volume")
    if result.packet_decoded is not None:
        if result.packet_decoded:
            print(f"  packet      decoded {result.packet_bytes} B · header and CRC ok")
        else:
            print(f"  packet      failed · {result.packet_error}")
    print()


def _print_failure(result: LoopbackResult) -> None:
    if not result.chirps_ok:
        print("  FAIL  the microphone did not hear the test chirps.", file=sys.stderr)
        print("  Check that the output is unmuted and the system volume is up,", file=sys.stderr)
        print("  then select the right devices with --list-devices.", file=sys.stderr)
        print("  In a very quiet room try lowering --threshold to 0.12.", file=sys.stderr)
    else:
        print("  FAIL  chirps were heard, but the modem packet did not decode.", file=sys.stderr)
        if result.clipped:
            print("  The input is clipping: lower --volume until the warning goes away.", file=sys.stderr)
        else:
            print("  Try moving the speaker and microphone closer together, lowering", file=sys.stderr)
            print("  --volume if the audio is distorted, or raising --volume if it is weak.", file=sys.stderr)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        from .audio_io import list_devices

        print(list_devices())
        return 0
    if not 0.0 < args.volume <= 1.0:
        parser.error("--volume must be between 0 and 1")
    if args.chirps < 1:
        parser.error("--chirps must be at least 1")

    input_spec = args.input_device or args.device
    output_spec = args.output_device or args.device
    try:
        input_index = resolve_device(input_spec, kind="input")
        output_index = resolve_device(output_spec, kind="output")
    except RuntimeError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 1

    waveform, plan = build_loopback_waveform(
        chirps=args.chirps,
        amplitude=args.volume,
        include_packet=not args.no_packet,
    )
    _print_banner(args, input_index, output_index, len(waveform) / SAMPLE_RATE)

    try:
        capture = record_loopback(
            waveform,
            input_device=input_index,
            output_device=output_index,
        )
    except Exception as exc:
        print(f"  audio loopback failed: {exc}", file=sys.stderr)
        return 1

    if args.wav is not None:
        write_wav(args.wav, capture)
        print(f"  capture written to {args.wav}")

    result = analyze_capture(capture, plan, threshold=args.threshold)
    _print_result(result)
    if not result.ok:
        _print_failure(result)
        return 1

    if result.packet_decoded is None:
        print("  PASS  the microphone hears the speaker.")
    else:
        print("  PASS  the microphone hears the speaker, and the modem packet decoded.")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
