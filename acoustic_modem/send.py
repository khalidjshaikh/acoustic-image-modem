"""Command-line sender: picture -> JPEG bands -> OFDM -> speaker."""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import numpy as np

from .audio_io import list_devices, resolve_device
from .config import (
    DEFAULT_BAND_ROWS,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_VOLUME,
    PACKET_GAP_SECONDS,
    SAMPLE_RATE,
)
from .dsp import packet_waveform, write_wav
from .protocol import build_packets, load_image, prepare_bands


def _human_bytes(n: int) -> str:
    if n < 1000:
        return f"{n} B"
    if n < 1_000_000:
        return f"{n / 1000:.1f} kB"
    return f"{n / 1_000_000:.2f} MB"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send an image over sound with a tiny acoustic OFDM modem."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="source_image.jpg",
        help="image to send (default: source_image.jpg)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH, help="transmitted pixel width")
    parser.add_argument("--band-rows", type=int, default=DEFAULT_BAND_ROWS, help="pixel rows per packet")
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality per band")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="send the complete image this many times (use 2 to fill gaps in a noisy room)",
    )
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME, help="output amplitude, 0..1")
    parser.add_argument("--device", help="output device index or part of its name")
    parser.add_argument("--wav", type=Path, help="also write the complete transmission to a WAV file")
    parser.add_argument("--no-play", action="store_true", help="build the WAV but do not play it")
    parser.add_argument("--session", help="optional 16-hex-digit session id")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    return parser


def _render(packets, args) -> tuple[np.ndarray, list[tuple[float, int, int, int]], float]:
    gap_n = int(PACKET_GAP_SECONDS * SAMPLE_RATE)
    gap = np.zeros(gap_n, dtype=np.float32)
    pieces: list[np.ndarray] = []
    markers: list[tuple[float, int, int, int]] = []
    cursor = 0
    reps = max(1, args.repeat)

    for rep in range(reps):
        for packet in packets:
            waveform = packet_waveform(packet, amplitude=args.volume)
            markers.append((cursor / SAMPLE_RATE, packet.header.seq, rep, len(waveform)))
            pieces.append(waveform)
            pieces.append(gap)
            cursor += len(waveform) + len(gap)

    pieces.append(np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float32))
    waveform = np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, dtype=np.float32)
    return waveform, markers, cursor / SAMPLE_RATE


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        print(list_devices())
        return 0
    if args.volume <= 0 or args.volume > 1:
        parser.error("--volume must be between 0 and 1")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    source = Path(args.image)
    if not source.exists():
        parser.error(f"image not found: {source}")

    try:
        image = load_image(source)
        preview, bands = prepare_bands(
            image,
            width=args.width,
            band_rows=args.band_rows,
            quality=args.quality,
        )
    except Exception as exc:
        parser.error(str(exc))

    session = int(args.session, 16) if args.session else secrets.randbits(64)
    try:
        packets = build_packets(
            bands,
            preview.width,
            preview.height,
            band_rows=args.band_rows,
            session=session,
        )
    except Exception as exc:
        parser.error(str(exc))

    waveform, markers, duration = _render(packets, args)
    on_air = sum(packet.header.coded_len for packet in packets) * max(1, args.repeat)
    jpeg_bytes = sum(len(packet.payload) for packet in packets)

    print()
    print("  ACOUSTIC IMAGE MODEM  /  transmit")
    print("  " + "-" * 56)
    print(f"  image       {source}  ({image.width}x{image.height})")
    print(f"  canvas      {preview.width}x{preview.height} pixels")
    print(f"  bands       {len(packets)} packets x {preview.width}x{args.band_rows} rows"
          + (f"  (repeated {args.repeat}x)" if args.repeat > 1 else ""))
    print(f"  JPEG data   {_human_bytes(jpeg_bytes)} -> {_human_bytes(on_air)} on air, FEC included")
    print(f"  air time    {duration:.1f} s at 48 kHz / OFDM / QPSK")
    print(f"  session     {session:016x}")
    print("  " + "-" * 56)

    if args.wav:
        write_wav(args.wav, waveform)
        print(f"  WAV written {args.wav}")

    if args.no_play:
        if not args.wav:
            default = Path("transmission.wav")
            write_wav(default, waveform)
            print(f"  --no-play selected; wrote {default}")
        return 0

    try:
        import sounddevice as sd

        out_device = resolve_device(args.device, kind="output")
        started = time.monotonic()
        sd.play(waveform, samplerate=SAMPLE_RATE, device=out_device, blocking=False)
        stream = sd.get_stream()
        next_marker = 0
        last_render = -1.0
        while stream.active:
            elapsed = max(0.0, time.monotonic() - started)
            while next_marker < len(markers) and elapsed >= markers[next_marker][0]:
                _, seq, rep, _ = markers[next_marker]
                packet = packets[seq]
                suffix = f"  pass {rep + 1}/{args.repeat}" if args.repeat > 1 else ""
                print(
                    f"  [{elapsed:06.2f}s] band {seq + 1:02d}/{len(packets)}  "
                    f"rows {packet.header.band_y:03d}-{packet.header.band_y + packet.header.band_h - 1:03d}  "
                    f"{len(packet.payload):4d} B{suffix}"
                )
                next_marker += 1
            if elapsed - last_render > 1.0:
                last_render = elapsed
                pct = 100.0 * min(1.0, elapsed / max(duration, 1e-6))
                print(f"  progress {pct:5.1f}%", end="\r", flush=True)
            time.sleep(0.02)
        sd.stop()
        print(f"\n  transmission complete in {time.monotonic() - started:.1f} s.")
        print("  The listener should be showing the picture by now.\n")
        return 0
    except KeyboardInterrupt:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        print("\n  stopped.")
        return 130
    except Exception as exc:
        print(f"\n  audio output failed: {exc}", file=sys.stderr)
        if not args.wav:
            fallback = Path("transmission.wav")
            write_wav(fallback, waveform)
            print(f"  wrote the transmission to {fallback} instead.", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
