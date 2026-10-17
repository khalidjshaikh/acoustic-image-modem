"""Live browser dashboard for the receiver.

The audio decoder runs in Python; this module is deliberately only the eyes of
the receiver.  Bands are streamed to the page with Server-Sent Events, and the
browser reveals them pixel by pixel.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, send_file

from .protocol import PacketHeader, assemble_image


@dataclass(frozen=True)
class BandEvent:
    session: int
    seq: int
    band_count: int
    width: int
    height: int
    band_y: int
    band_h: int
    size: int
    jpeg_b64: str


class ReceiverState:
    def __init__(self, save_dir: str | Path = "received") -> None:
        self.lock = threading.RLock()
        self.save_dir = Path(save_dir)
        self.session: int | None = None
        self.width = 0
        self.height = 0
        self.band_count = 0
        self.bands: dict[int, BandEvent] = {}
        self.packet_data: dict[int, tuple[PacketHeader, bytes]] = {}
        self.level = 0.0
        self.phase = "waiting"
        self.started = time.monotonic()
        self.completed = False
        self.save_path: Path | None = None
        self.image_png: bytes | None = None
        self.subscribers: list[queue.Queue[dict[str, Any]]] = []

    # -- state changes -----------------------------------------------------
    def set_level(self, level: float) -> None:
        with self.lock:
            self.level = max(0.0, min(1.0, level))

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def _ensure_session(self, header: PacketHeader) -> bool:
        """Return True when this starts a new picture."""
        if self.session == header.session:
            return False
        self.session = header.session
        self.width = header.width
        self.height = header.height
        self.band_count = header.band_count
        self.bands.clear()
        self.packet_data.clear()
        self.started = time.monotonic()
        self.completed = False
        self.save_path = None
        self.image_png = None
        return True

    def add_band(self, header: PacketHeader, payload: bytes) -> bool:
        """Add a decoded band; return True if it was new."""
        new_session = False
        with self.lock:
            new_session = self._ensure_session(header)
            if header.seq in self.bands:
                return False
            event = BandEvent(
                session=header.session,
                seq=header.seq,
                band_count=header.band_count,
                width=header.width,
                height=header.height,
                band_y=header.band_y,
                band_h=header.band_h,
                size=len(payload),
                jpeg_b64=base64.b64encode(payload).decode("ascii"),
            )
            self.bands[header.seq] = event
            self.packet_data[header.seq] = (header, payload)
            self.phase = "receiving"
            if new_session:
                self._broadcast_locked(
                    {
                        "type": "reset",
                        "session": f"{header.session:016x}",
                        "width": header.width,
                        "height": header.height,
                        "bands": header.band_count,
                    }
                )
            self._broadcast_locked(self._band_payload(event))
            complete_now = len(self.bands) == self.band_count and not self.completed
            if complete_now:
                self.completed = True
                self.phase = "complete"
                try:
                    image = assemble_image(self.packet_data.values())
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    self.save_path = self.save_dir / f"received-{stamp}-{header.session:016x}.png"
                    image.save(self.save_path)
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    self.image_png = buffer.getvalue()
                except Exception as exc:  # keep decoding even if the disk is read-only
                    self._broadcast_locked({"type": "warning", "message": str(exc)})
                self._broadcast_locked(
                    {
                        "type": "complete",
                        "session": f"{header.session:016x}",
                        "width": self.width,
                        "height": self.height,
                        "bands": self.band_count,
                        "missing": self.missing_bands(),
                        "saved": str(self.save_path) if self.save_path else None,
                    }
                )
            return True

    def missing_bands(self) -> list[int]:
        if not self.band_count:
            return []
        return [i for i in range(self.band_count) if i not in self.bands]

    def _band_payload(self, event: BandEvent) -> dict[str, Any]:
        return {
            "type": "band",
            "session": f"{event.session:016x}",
            "seq": event.seq,
            "bands": event.band_count,
            "width": event.width,
            "height": event.height,
            "y": event.band_y,
            "h": event.band_h,
            "size": event.size,
            "jpeg": event.jpeg_b64,
        }

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.session is None:
                return []
            events: list[dict[str, Any]] = [
                {
                    "type": "reset",
                    "session": f"{self.session:016x}",
                    "width": self.width,
                    "height": self.height,
                    "bands": self.band_count,
                }
            ]
            for seq in sorted(self.bands):
                payload = self._band_payload(self.bands[seq])
                payload["snapshot"] = True
                events.append(payload)
            if self.completed:
                events.append(
                    {
                        "type": "complete",
                        "session": f"{self.session:016x}",
                        "width": self.width,
                        "height": self.height,
                        "bands": self.band_count,
                        "missing": [],
                        "saved": str(self.save_path) if self.save_path else None,
                    }
                )
            return events

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "phase": self.phase,
                "session": f"{self.session:016x}" if self.session is not None else None,
                "width": self.width,
                "height": self.height,
                "bands": self.band_count,
                "received": len(self.bands),
                "missing": self.missing_bands(),
                "level": self.level,
                "completed": self.completed,
                "saved": str(self.save_path) if self.save_path else None,
            }

    # -- SSE plumbing ------------------------------------------------------
    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self.lock:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast_locked(self, event: dict[str, Any]) -> None:
        dead: list[queue.Queue[dict[str, Any]]] = []
        for subscriber in self.subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                dead.append(subscriber)
        for subscriber in dead:
            try:
                self.subscribers.remove(subscriber)
            except ValueError:
                pass


def create_app(state: ReceiverState) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.logger.setLevel("ERROR")

    @app.get("/")
    def index():
        return render_template("receiver.html")

    @app.get("/events")
    def events():
        def stream():
            q = state.subscribe()
            try:
                yield "retry: 1500\n\n"
                for event in state.snapshot():
                    yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                while True:
                    try:
                        event = q.get(timeout=12.0)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            finally:
                state.unsubscribe(q)

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/status")
    def status():
        return jsonify(state.status())

    @app.get("/image/<session>.png")
    def image(session: str):
        with state.lock:
            if state.image_png is None or state.session is None:
                return ("not ready", 404)
            if f"{state.session:016x}" != session:
                return ("not found", 404)
            return send_file(io.BytesIO(state.image_png), mimetype="image/png")

    return app


def start_server(
    state: ReceiverState,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> threading.Thread:
    app = create_app(state)

    def run() -> None:
        app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)

    thread = threading.Thread(target=run, name="image-modem-ui", daemon=True)
    thread.start()

    if open_browser:
        url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"

        def open_later() -> None:
            time.sleep(0.8)
            webbrowser.open(url)

        threading.Thread(target=open_later, name="open-browser", daemon=True).start()
    return thread
