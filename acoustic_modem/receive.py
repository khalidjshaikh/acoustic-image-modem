"""Command-line receiver: microphone -> OFDM -> live picture."""

from __future__ import annotations

import argparse
import queue
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .audio_io import default_input_name, list_devices, resolve_device, rms_level
from .config import (
    HEADER_COPIES,
    INPUT_BLOCK_SAMPLES,
    INPUT_LATENCY_SECONDS,
    LTF_SYMBOLS,
    SAMPLE_RATE,
    SYMBOL_SAMPLES,
    SYNC_GAP_SAMPLES,
    SYNC_SAMPLES,
)
from .dsp import (
    DecodeError,
    HeaderProbe,
    decode_audio,
    decode_packet,
    find_sync_peaks,
    header_symbols,
    probe_header,
    wav_to_float,
)
from .webui import ReceiverState, start_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Listen for an acoustic image modem and paint the picture live."
    )
    parser.add_argument("--device", help="input device index or part of its name")
    parser.add_argument("--wav", type=Path, help="decode a previously recorded WAV instead of the microphone")
    parser.add_argument("--host", default="127.0.0.1", help="UI bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="UI port (default: 8765)")
    parser.add_argument("--save-dir", default="received", help="where completed pictures are written")
    parser.add_argument("--threshold", type=float, default=0.19, help="chirp detector threshold (default: 0.19)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--replay-delay", type=float, default=0.45, help="delay between bands in --wav mode")
    parser.add_argument("--record", type=Path, help="also save the microphone capture as WAV")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    return parser


def _print_banner(args, state: ReceiverState) -> None:
    print()
    print("  ACOUSTIC IMAGE MODEM  /  receive")
    print("  " + "-" * 56)
    print("  OFDM 48 kHz · QPSK · RS(255,207) · 152 carriers")
    if args.wav:
        print(f"  source      {args.wav}")
    else:
        print(f"  input       {default_input_name()}")
    print(f"  dashboard   http://{args.host}:{args.port}/")
    print(f"  saving to   {Path(args.save_dir).resolve()}")
    print("  " + "-" * 56)
    print("  Listening for a modem chirp.  Press Ctrl+C to stop.")
    print()


def _log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"  [{stamp}] {message}", flush=True)


def _probe_min_samples() -> int:
    header_syms = header_symbols()
    return (
        SYNC_SAMPLES
        + SYNC_GAP_SAMPLES
        + (LTF_SYMBOLS + HEADER_COPIES * header_syms + 1) * SYMBOL_SAMPLES
        + SAMPLE_RATE // 50
    )


def _run_wav(path: Path, state: ReceiverState, args) -> int:
    state.set_phase("waiting")
    audio = wav_to_float(path)
    _log(f"loaded {len(audio) / SAMPLE_RATE:.1f}s of audio; searching for packets")
    state.set_phase("decoding")
    packets = sorted(
        decode_audio(audio, threshold=args.threshold),
        key=lambda packet: packet.sync_start,
    )
    if not packets:
        _log("no packets decoded. Check the recording level or lower --threshold.")
        return 1
    _log(f"decoded {len(packets)} packet(s)")
    state.set_phase("receiving")
    for index, packet in enumerate(packets, 1):
        state.add_band(packet.header, packet.payload)
        _log(
            f"band {packet.header.seq + 1:02d}/{packet.header.band_count:02d}  "
            f"{len(packet.payload):4d} B  session {packet.header.session:016x}"
        )
        if index < len(packets):
            time.sleep(max(0.0, args.replay_delay))
    with state.lock:
        if state.save_path:
            _log(f"wrote {state.save_path}")
        else:
            _log(f"received {len(state.bands)}/{state.band_count} bands; waiting for the rest")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def _run_live(state: ReceiverState, args) -> int:
    try:
        import sounddevice as sd
    except Exception as exc:
        _log(f"sounddevice unavailable: {exc}")
        return 1

    blocks: queue.Queue[np.ndarray] = queue.Queue()
    overflow_count = 0
    record_handle = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record_handle = wave.open(str(args.record), "wb")
        record_handle.setnchannels(1)
        record_handle.setsampwidth(2)
        record_handle.setframerate(SAMPLE_RATE)

    def callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        nonlocal overflow_count
        if status:
            overflow_count += 1
        blocks.put(indata[:, 0].copy())

    try:
        device = resolve_device(args.device, kind="input")
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
            device=device,
            blocksize=INPUT_BLOCK_SAMPLES,
            latency=INPUT_LATENCY_SECONDS,
        )
        stream.start()
    except Exception as exc:
        _log(f"could not open the microphone: {exc}")
        return 1

    state.set_phase("waiting")
    buffer = np.empty(0, dtype=np.float32)
    search_from = 0
    smoothed_level = 0.0
    packets_seen = 0
    last_packet_at = 0.0

    def drain(timeout: float) -> bool:
        nonlocal buffer, smoothed_level
        try:
            block = blocks.get(timeout=timeout)
        except queue.Empty:
            return False
        buffer = np.concatenate((buffer, block))
        if record_handle is not None:
            record_handle.writeframes(
                (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            )
        level = rms_level(block)
        smoothed_level = 0.78 * smoothed_level + 0.22 * min(1.0, level * 8.0)
        state.set_level(smoothed_level)
        return True

    _log("microphone open; waiting for carrier")
    if record_handle is not None:
        _log(f"recording microphone to {args.record}")
    last_overflow_log = 0
    try:
        while True:
            if overflow_count and overflow_count != last_overflow_log:
                _log(f"warning: {overflow_count} input overflow event(s)")
                last_overflow_log = overflow_count
            drain(0.15)
            if time.monotonic() - last_packet_at > 1.5 and state.phase == "receiving":
                state.set_phase("listening")

            # Search a sliding window.  Keeping one chirp of overlap means a
            # packet that straddles the window boundary is not missed.
            if len(buffer) - search_from >= int(0.30 * SAMPLE_RATE):
                window_end = min(len(buffer), search_from + int(2.0 * SAMPLE_RATE))
                window = buffer[search_from:window_end]
                peaks = find_sync_peaks(window, threshold=args.threshold)
                if peaks.size == 0:
                    overlap = SYNC_SAMPLES + SAMPLE_RATE // 20
                    search_from = max(search_from, window_end - overlap)
                    if search_from > 2 * SAMPLE_RATE:
                        buffer = buffer[search_from:].copy()
                        search_from = 0
                    continue

                sync_start = search_from + int(peaks[0])
                while len(buffer) < sync_start + _probe_min_samples():
                    drain(0.25)

                try:
                    probe: HeaderProbe = probe_header(buffer, sync_start)
                except DecodeError:
                    # A false chirp, or a header we could not recover.  Skip
                    # past this chirp and keep listening.
                    cut = sync_start + SYNC_SAMPLES + SYNC_GAP_SAMPLES
                    buffer = buffer[cut:].copy()
                    search_from = 0
                    state.set_phase("listening")
                    continue

                _log(
                    f"header lock · session {probe.header.session:016x} · "
                    f"band {probe.header.seq + 1:02d}/{probe.header.band_count:02d} · "
                    f"{probe.header.raw_len} B payload"
                )
                state.set_phase("locked")
                # Pull the whole packet into a local array and then try every
                # plausible FFT alignment.  This is deliberately done after
                # capture: the 1 s input buffer keeps the audio device from
                # dropping samples while the demodulator is busy.
                needed = sync_start + probe.duration_samples + int(0.04 * SAMPLE_RATE)
                while len(buffer) < needed:
                    drain(0.25)
                packet_audio = buffer[sync_start:needed].copy()

                try:
                    packet = decode_packet(packet_audio, 0)
                except (DecodeError, ValueError) as exc:
                    _log(f"payload failed: {exc}")
                    cut = sync_start + probe.duration_samples
                    buffer = buffer[cut:].copy()
                    search_from = 0
                    state.set_phase("listening")
                    continue

                state.add_band(packet.header, packet.payload)
                state.set_phase("receiving")
                packets_seen += 1
                last_packet_at = time.monotonic()
                with state.lock:
                    done = len(state.bands)
                    total = state.band_count
                    saved = state.save_path
                _log(
                    f"band {packet.header.seq + 1:02d}/{packet.header.band_count:02d} painted · "
                    f"{len(packet.payload)} B · {done}/{total} received"
                )
                if saved:
                    _log(f"complete image written to {saved}")

                cut = sync_start + packet.duration_samples + int(0.02 * SAMPLE_RATE)
                buffer = buffer[cut:].copy()
                search_from = 0
    except KeyboardInterrupt:
        _log("stopping")
        stream.stop()
        stream.close()
        if record_handle is not None:
            record_handle.close()
        _log(f"received {packets_seen} packet(s); input overflows: {overflow_count}")
        return 0
    except Exception as exc:
        _log(f"receiver error: {exc}")
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        if record_handle is not None:
            record_handle.close()
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        print(list_devices())
        return 0

    state = ReceiverState(args.save_dir)
    start_server(state, host=args.host, port=args.port, open_browser=not args.no_browser)
    _print_banner(args, state)

    if args.wav:
        return _run_wav(args.wav, state, args)
    return _run_live(state, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
