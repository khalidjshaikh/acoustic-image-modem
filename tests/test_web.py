from __future__ import annotations

from PIL import Image, ImageDraw

from acoustic_modem.protocol import build_packets, prepare_bands
from acoustic_modem.webui import ReceiverState, create_app


def make_packets():
    image = Image.new("RGB", (160, 200), (16, 24, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 140, 120), fill=(255, 210, 30))
    draw.ellipse((35, 120, 130, 190), fill=(60, 190, 210))
    preview, bands = prepare_bands(image, width=50, band_rows=8, quality=55)
    return preview, build_packets(bands, preview.width, preview.height, session=0xABC)


def test_state_streams_and_saves_complete_image(tmp_path):
    preview, packets = make_packets()
    state = ReceiverState(save_dir=tmp_path)
    q = state.subscribe()

    for packet in packets:
        assert state.add_band(packet.header, packet.payload)

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    band_events = [event for event in events if event["type"] == "band"]
    assert len(band_events) == len(packets)
    assert events[-1]["type"] == "complete"
    assert events[0]["type"] == "reset"
    assert state.completed
    assert state.save_path is not None and state.save_path.exists()
    with Image.open(state.save_path) as image:
        assert image.size == preview.size

    # Duplicate bands are ignored.
    assert not state.add_band(packets[0].header, packets[0].payload)
    assert len(state.bands) == len(packets)


def test_flask_routes_and_snapshot(tmp_path):
    _, packets = make_packets()
    state = ReceiverState(save_dir=tmp_path)
    state.add_band(packets[0].header, packets[0].payload)
    app = create_app(state)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"ACOUSTIC" in page.data

    status = client.get("/status").get_json()
    assert status["session"] == f"{packets[0].header.session:016x}"
    assert status["received"] == 1
    assert status["bands"] == len(packets)

    snapshot = state.snapshot()
    assert snapshot[0]["type"] == "reset"
    assert snapshot[1]["type"] == "band"
    assert snapshot[1]["snapshot"] is True
