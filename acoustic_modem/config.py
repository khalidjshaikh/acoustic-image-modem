"""Shared physical-layer settings.

Everything in this file is part of the on-air protocol.  Sender and receiver
must agree on these values, but they are deliberately kept in one place so the
modem is easy to experiment with.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48_000
FFT_SIZE = 1024
CP_SAMPLES = 256
SYMBOL_SAMPLES = FFT_SIZE + CP_SAMPLES
SYMBOL_SECONDS = SYMBOL_SAMPLES / SAMPLE_RATE

# Occupied spectrum: 750 Hz .. 7.83 kHz.  The bottom edge avoids mains hum and
# laptop fan rumble; the top edge leaves room for typical speaker/mic roll-off.
BIN_LO = 16
BIN_HI = 168  # exclusive
ALL_BINS = np.arange(BIN_LO, BIN_HI, dtype=np.int32)

# A pilot every 8th bin is used for residual carrier/timing phase tracking.
PILOT_BINS = np.arange(20, BIN_HI, 8, dtype=np.int32)
DATA_BINS = np.setdiff1d(ALL_BINS, PILOT_BINS)
PILOTS_IN_ALL = np.searchsorted(ALL_BINS, PILOT_BINS)
# Headers use every data carrier once.  Payload QPSK symbols are repeated on
# carriers spread across the band: a deep notch in the speaker/microphone
# response then costs diversity instead of a fixed set of corrupted bits.
HEADER_BITS_PER_SYMBOL = 2 * len(DATA_BINS)
FREQ_REPETITION = 2
PAYLOAD_GROUPS = len(DATA_BINS) // FREQ_REPETITION
PAYLOAD_BITS_PER_SYMBOL = 2 * PAYLOAD_GROUPS
BITS_PER_SYMBOL = HEADER_BITS_PER_SYMBOL  # backward-compatible name for headers

# Sync: a linear chirp is easy to hear, has a sharp matched-filter peak and is
# tolerant of the frequency response of cheap speakers.
SYNC_SECONDS = 0.12
SYNC_SAMPLES = int(SYNC_SECONDS * SAMPLE_RATE)
SYNC_GAP_SECONDS = 0.012
SYNC_GAP_SAMPLES = int(SYNC_GAP_SECONDS * SAMPLE_RATE)
SYNC_F0 = 500.0
SYNC_F1 = 8_000.0

LTF_SYMBOLS = 4
HEADER_COPIES = 2

# Reed-Solomon parameters.  The 48 parity bytes per 255-byte block correct up
# to 24 corrupted bytes in each block.  A pseudo-random bit interleaver spreads
# frequency-selective fades across the whole block.
HEADER_PARITY = 48
PAYLOAD_PARITY = 48
RS_BLOCK = 255
PAYLOAD_DATA_PER_BLOCK = RS_BLOCK - PAYLOAD_PARITY

DEFAULT_IMAGE_WIDTH = 100
DEFAULT_BAND_ROWS = 8
DEFAULT_JPEG_QUALITY = 68
DEFAULT_VOLUME = 0.50
PACKET_GAP_SECONDS = 0.11
# A generous input buffer keeps the audio device from dropping samples while
# the Python demodulator is busy with a header or a Reed-Solomon block.
INPUT_LATENCY_SECONDS = 1.0
INPUT_BLOCK_SAMPLES = 1024

HEADER_MAGIC = b"AIM1"
PROTOCOL_VERSION = 1
