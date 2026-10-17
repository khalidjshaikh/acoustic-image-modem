"""Small helpers around sounddevice device selection."""

from __future__ import annotations

import sys

import numpy as np


def sounddevice_module():
    """Import sounddevice lazily so the rest of the package works without audio."""
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        raise RuntimeError(
            "sounddevice/PortAudio is not available. Install the Python "
            "dependencies and make sure PortAudio is installed on this machine."
        ) from exc
    return sd


def list_devices() -> str:
    sd = sounddevice_module()
    devices = sd.query_devices()
    lines = []
    for index, device in enumerate(devices):
        lines.append(
            f"{index:3d}  in={device['max_input_channels']:2d}  "
            f"out={device['max_output_channels']:2d}  "
            f"{device['default_samplerate']:7.0f} Hz  {device['name']}"
        )
    return "\n".join(lines)


def resolve_device(spec: str | int | None, *, kind: str) -> int | None:
    """Resolve an index or a case-insensitive substring of a device name."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    sd = sounddevice_module()
    wanted = str(spec).lower()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        index
        for index, device in enumerate(sd.query_devices())
        if device[key] > 0 and wanted in device["name"].lower()
    ]
    if not matches:
        raise RuntimeError(f"no {kind} device matches {spec!r}\n\n{list_devices()}")
    if len(matches) > 1:
        names = "\n".join(f"  {i}: {sd.query_devices(i)['name']}" for i in matches)
        raise RuntimeError(
            f"{spec!r} matches more than one {kind} device; use an index:\n{names}"
        )
    device = matches[0]
    print(f"Using {kind} device {device}: {sd.query_devices(device)['name']}", file=sys.stderr)
    return device


def device_name(index: int | None, *, kind: str) -> str:
    """Human-readable name for a resolved device, or the system default."""
    sd = sounddevice_module()
    try:
        if index is None:
            index = sd.default.device[0 if kind == "input" else 1]
        if index is None or index < 0:
            return "system default"
        return sd.query_devices(index)["name"]
    except Exception:
        return "system default"


def default_input_name() -> str:
    return device_name(None, kind="input")


def rms_level(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
