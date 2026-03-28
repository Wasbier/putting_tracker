"""
WebSocket server: read a Tapo (or other) RTSP stream on the PC, run the same
tracking logic as track_putts.py, and push live counts to the mobile app.

Run (example):
  python putting_ws_server.py --stream "rtsp://user:pass@192.168.1.50:554/stream1" --profile night
  python putting_ws_server.py ... --record-to out/session.mp4

Then in the app Practice tab, set the server URL to ws://<PC_LAN_IP>:8765

Requires: pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import websockets

from capture_utils import open_stream
from track_putts import LiveTrackerRuntime, _infer_profile_tag, load_config

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

stop_event = threading.Event()
cmd_queue: queue.Queue[str] = queue.Queue()
state_lock = threading.Lock()
latest_state: dict[str, Any] = {}
clients: set[Any] = set()
clients_lock = threading.Lock()


def resolve_config_path(
    args: argparse.Namespace,
    frame0,
    profiles_dir: Path,
) -> Path:
    if args.config is not None:
        p = Path(args.config)
        if p.is_file():
            return p
    if args.profile == "auto":
        tag, med = _infer_profile_tag(frame0)
        print(f"Auto profile from stream: median gray {med:.1f} -> {tag}")
        return profiles_dir / f"{args.camera_id}_{tag}.json"
    return profiles_dir / f"{args.camera_id}_{args.profile}.json"


def capture_loop(args: argparse.Namespace) -> None:
    cap = open_stream(args.stream)
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        print("Could not read first frame from stream.", file=sys.stderr)
        stop_event.set()
        return
    fh, fw = frame0.shape[:2]
    config_path = resolve_config_path(args, frame0, args.profiles_dir)
    if not config_path.is_file():
        print(
            f"Missing config {config_path}; calibrate with track_putts first.",
            file=sys.stderr,
        )
        stop_event.set()
        cap.release()
        return
    print(f"Using config: {config_path}")
    cfg = load_config(config_path, fw, fh)
    rt = LiveTrackerRuntime.from_loaded_config(cfg, fw, fh)

    writer: cv2.VideoWriter | None = None
    if args.record_to is not None:
        out_path = Path(args.record_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(args.record_fps) if args.record_fps > 0 else 0.0
        if fps <= 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
        if not writer.isOpened():
            print(
                f"Could not open VideoWriter for {out_path}; continuing without --record-to.",
                file=sys.stderr,
            )
            writer = None
        else:
            print(f"Recording stream to {out_path.resolve()} @ {fps:.2f} fps")

    def drain_commands() -> None:
        while not cmd_queue.empty():
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "reset_session":
                rt.reset_tracking_state()
                print("Session reset (tracker counters cleared).")

    drain_commands()
    rt.step(frame0)
    if writer is not None:
        writer.write(frame0)
    with state_lock:
        latest_state.clear()
        latest_state.update(rt.public_snapshot())

    while not stop_event.is_set():
        drain_commands()
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        rt.step(frame)
        if writer is not None:
            writer.write(frame)
        snap = rt.public_snapshot()
        with state_lock:
            latest_state.clear()
            latest_state.update(snap)

    cap.release()
    if writer is not None:
        writer.release()
    print("Capture thread stopped.")


async def ws_handler(websocket: Any) -> None:
    with clients_lock:
        clients.add(websocket)
    try:
        with state_lock:
            if latest_state:
                await websocket.send(json.dumps(latest_state))
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            t = data.get("type")
            if t == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            elif t == "reset_session":
                cmd_queue.put("reset_session")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with clients_lock:
            clients.discard(websocket)


async def broadcast_loop() -> None:
    last_sent: str | None = None
    while not stop_event.is_set():
        await asyncio.sleep(0.2)
        with state_lock:
            if not latest_state:
                continue
            payload = json.dumps(latest_state)
        if payload == last_sent:
            continue
        last_sent = payload
        with clients_lock:
            dead: list[Any] = []
            for ws in clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)


async def amain(args: argparse.Namespace) -> None:
    asyncio.create_task(broadcast_loop())
    async with websockets.serve(ws_handler, args.host, args.port):
        lan = "YOUR_PC_IP"
        print(f"WebSocket listening on ws://{args.host}:{args.port}")
        print(f"On your phone use e.g. ws://{lan}:{args.port} (replace {lan} with this PC's LAN IP).")
        await asyncio.Future()


def main() -> None:
    p = argparse.ArgumentParser(description="Putting tracker WebSocket bridge for Tapo/stream")
    p.add_argument(
        "--stream",
        required=True,
        help='RTSP or stream URL (e.g. Tapo rtsp://...:554/stream1")',
    )
    p.add_argument(
        "--profile",
        choices=["day", "night", "auto"],
        default="night",
        help="Profile JSON under --profiles-dir (default: night)",
    )
    p.add_argument("--camera-id", type=str, default="camera1")
    p.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("config/profiles"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit calibration JSON (overrides --profile)",
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--record-to",
        type=Path,
        default=None,
        help="Save a copy of the stream to this MP4 while tracking (same frames as the tracker).",
    )
    p.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="FPS for --record-to when stream FPS is unknown (0 = use stream or 25).",
    )
    args = p.parse_args()

    cap_thread = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    cap_thread.start()
    time.sleep(0.8)
    if stop_event.is_set():
        raise SystemExit(1)

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
