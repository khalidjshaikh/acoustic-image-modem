from __future__ import annotations

import os

import numpy as np
import pytest

from acoustic_modem.config import SAMPLE_RATE
from acoustic_modem.loopback import (
    analyze_capture,
    build_loopback_waveform,
    run_loopback,
)


def test_loopback_waveform_geometry():
    waveform, plan = build_loopback_waveform(chirps=3, amplitude=0.5, include_packet=True)
    assert plan.expected_chirps == 3
    assert plan.expected_syncs == 4
    assert plan.packet_start is not None and plan.packet_start > plan.chirp_starts[-1]
    assert plan.packet_size and plan.packet_size > 0
    assert len(waveform) > plan.packet_start
    assert np.max(np.abs(waveform)) <= 1.0

    gaps = np.diff(plan.chirp_starts)
    assert np.all(gaps > 0)
    assert np.allclose(gaps, gaps[0])
    assert len(waveform) / SAMPLE_RATE > 2.0


def test_perfect_loopback_hears_chirps_and_packet():
    waveform, plan = build_loopback_waveform(chirps=3, amplitude=0.6)
    result = analyze_capture(waveform, plan)
    assert result.ok
    assert result.chirps_ok
    assert result.detected_chirps >= plan.expected_syncs
    assert result.peak_correlation > 0.9
    assert abs(result.first_chirp_offset_seconds or 0.0) < 0.01
    assert result.packet_decoded is True
    assert result.packet_bytes == plan.packet_size


def test_chirp_only_loopback_skips_packet():
    waveform, plan = build_loopback_waveform(chirps=2, include_packet=False)
    result = analyze_capture(waveform, plan)
    assert result.ok
    assert result.packet_decoded is None
    assert result.packet_error is None
    assert plan.expected_syncs == 2


def test_attenuated_noisy_loopback_still_decodes():
    waveform, plan = build_loopback_waveform(chirps=2, amplitude=0.6)
    rng = np.random.default_rng(11)
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    capture = 0.2 * waveform + rng.normal(0.0, rms * 0.05, len(waveform))
    result = analyze_capture(capture, plan)
    assert result.chirps_ok
    assert result.packet_decoded is True
    assert result.ok


def test_silence_fails_loopback():
    waveform, plan = build_loopback_waveform(chirps=2, include_packet=True)
    result = analyze_capture(np.zeros_like(waveform), plan)
    assert result.detected_chirps == 0
    assert not result.chirps_ok
    assert result.packet_decoded is False
    assert not result.ok


@pytest.mark.hardware
def test_live_speaker_microphone_loopback():
    if os.environ.get("ACOUSTIC_LOOPBACK") != "1":
        pytest.skip("set ACOUSTIC_LOOPBACK=1 to play audio through the real speaker")
    result = run_loopback()
    assert result.chirps_ok, "the microphone did not hear the test chirps"
    assert result.packet_decoded, f"the modem packet did not decode: {result.packet_error}"
