"""
scripts/simulate_cv.py — OccupAI Computer Vision Simulator
Simulates dual camera feeds ('cars' and 'motorcycles') pushing live occupancy
and slot states to the OccupAI backend without physical cameras or YOLO inference.

Usage:
    python scripts/simulate_cv.py
    python scripts/simulate_cv.py --url http://127.0.0.1:8000 --token occupai_cam_2027
    python scripts/simulate_cv.py --auto   # runs with dynamic arrival/departure simulation
"""

import argparse
import json
import random
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

PH_TZ = ZoneInfo("Asia/Manila")

TOTAL_CARS = 10
TOTAL_MOTOS = 20
CAR_PREFIX = "C"
MOTO_PREFIX = "M"

# Initialize slot states
car_slots = {f"{CAR_PREFIX}{i:02d}": False for i in range(1, TOTAL_CARS + 1)}
moto_slots = {f"{MOTO_PREFIX}{i:02d}": False for i in range(1, TOTAL_MOTOS + 1)}

# Default initial demo state (partially occupied)
for c in ["C01", "C02", "C04"]:
    car_slots[c] = True
for m in ["M01", "M03", "M05", "M08", "M12"]:
    moto_slots[m] = True

slots_lock = threading.Lock()
running = True
auto_traffic = False
last_push_status = {"cars": "waiting", "motorcycles": "waiting"}


def parse_args():
    parser = argparse.ArgumentParser(description="OccupAI Computer Vision Simulator")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="OccupAI backend base URL")
    parser.add_argument("--token", default="occupai_cam_2027", help="Camera security token (CAM_TOKEN)")
    parser.add_argument("--interval", type=float, default=2.5, help="Heartbeat interval in seconds (default: 2.5s)")
    parser.add_argument("--auto", action="store_true", help="Start with auto-traffic simulation enabled")
    return parser.parse_args()


def build_camera_payload(camera_id: str, camera_role: str, slots_dict: dict, total: int):
    occupied = sum(1 for v in slots_dict.values() if v)
    free = max(0, total - occupied)
    pct = round((occupied / total) * 100, 1) if total > 0 else 0.0

    return {
        "occupied": occupied,
        "free": free,
        "total": total,
        "occupancy_pct": pct,
        "lot_full": (free == 0),
        "fps": 15.0,
        "yolo_count": occupied,
        "car_count": occupied if camera_role == "car" else 0,
        "motorcycle_count": occupied if camera_role == "motorcycle" else 0,
        "bicycle_count": 0,
        "camera_id": camera_id,
        "camera_role": camera_role,
        "reporting": True,
        "camera_status": "online",
        "calibration_required": False,
        "timestamp": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_b64": "",
        "yolo_boxes": [],
        "slots": [],
        "zones": dict(slots_dict),
    }


def push_heartbeat(base_url: str, token: str):
    headers = {
        "Content-Type": "application/json",
        "X-Cam-Token": token,
    }
    update_url = f"{base_url.rstrip('/')}/yolo/update"

    with slots_lock:
        cars_payload = build_camera_payload("cars", "car", car_slots, TOTAL_CARS)
        motos_payload = build_camera_payload("motorcycles", "motorcycle", moto_slots, TOTAL_MOTOS)

    for cid, payload in [("cars", cars_payload), ("motorcycles", motos_payload)]:
        try:
            resp = requests.post(update_url, json=payload, headers=headers, timeout=3.0)
            if resp.status_code == 200:
                last_push_status[cid] = "OK (200)"
            else:
                last_push_status[cid] = f"HTTP {resp.status_code}: {resp.text[:60]}"
        except requests.exceptions.ConnectionError:
            last_push_status[cid] = f"Connection failed (is backend running at {base_url}?)"
        except Exception as e:
            last_push_status[cid] = f"Error: {e}"


def sender_loop(base_url: str, token: str, interval: float):
    global running
    while running:
        push_heartbeat(base_url, token)
        time.sleep(interval)


def auto_traffic_loop():
    global auto_traffic, running
    while running:
        time.sleep(random.uniform(5.0, 10.0))
        if not auto_traffic:
            continue
        with slots_lock:
            # 50% chance car event, 50% chance moto event
            if random.random() < 0.5:
                slot = random.choice(list(car_slots.keys()))
                car_slots[slot] = not car_slots[slot]
                action = "PARKED in" if car_slots[slot] else "LEFT"
                print(f"\n[AUTO-TRAFFIC] Car {action} {slot}")
            else:
                slot = random.choice(list(moto_slots.keys()))
                moto_slots[slot] = not moto_slots[slot]
                action = "PARKED in" if moto_slots[slot] else "LEFT"
                print(f"\n[AUTO-TRAFFIC] Motorcycle {action} {slot}")
            print("occupai-sim> ", end="", flush=True)


def print_status():
    with slots_lock:
        car_occ = sum(1 for v in car_slots.values() if v)
        moto_occ = sum(1 for v in moto_slots.values() if v)

        print("\n" + "=" * 60)
        print(f"  OCCUPAI SIMULATED LOT STATUS — {datetime.now(PH_TZ).strftime('%I:%M:%S %p')}")
        print("=" * 60)
        print(f"  Push Status: Cars: {last_push_status['cars']} | Motorcycles: {last_push_status['motorcycles']}")
        print(f"  Auto-Traffic: {'[ENABLED]' if auto_traffic else '[DISABLED]'}")
        print("-" * 60)

        print(f"  CARS ({car_occ}/{TOTAL_CARS} Occupied):")
        car_line = "  "
        for i, (k, v) in enumerate(car_slots.items()):
            icon = "[X]" if v else "[ ]"
            car_line += f"{k}:{icon}  "
            if (i + 1) % 5 == 0:
                print(car_line)
                car_line = "  "

        print(f"\n  MOTORCYCLES ({moto_occ}/{TOTAL_MOTOS} Occupied):")
        moto_line = "  "
        for i, (k, v) in enumerate(moto_slots.items()):
            icon = "[X]" if v else "[ ]"
            moto_line += f"{k}:{icon}  "
            if (i + 1) % 5 == 0:
                print(moto_line)
                moto_line = "  "
        print("=" * 60)
        print("  Legend: [X] = Occupied,  [ ] = Available/Free\n")


def print_help():
    print("""
Available Commands:
  occupy <slot>   (e.g. occupy C01, occ M05)  -> Set slot to Occupied
  free <slot>     (e.g. free C01, free M05)   -> Set slot to Free / Available
  toggle <slot>   (e.g. toggle C03, t C03)    -> Toggle slot state
  auto on|off     (e.g. auto on)              -> Enable/disable dynamic random traffic
  preset <name>   (presets: empty, busy, full)-> Load a preset parking layout
  status / s / map                            -> Display current lot map
  help / ?                                    -> Show this menu
  exit / quit / q                             -> Exit simulator
""")


def set_preset(name: str):
    name = name.lower().strip()
    with slots_lock:
        if name in ("empty", "clear"):
            for k in car_slots:
                car_slots[k] = False
            for k in moto_slots:
                moto_slots[k] = False
            print("[SIM] Lot cleared: 0 occupied.")
        elif name == "full":
            for k in car_slots:
                car_slots[k] = True
            for k in moto_slots:
                moto_slots[k] = True
            print("[SIM] Lot filled: 30/30 occupied.")
        elif name == "busy":
            for k in car_slots:
                car_slots[k] = False
            for k in moto_slots:
                moto_slots[k] = False
            for c in ["C01", "C02", "C03", "C05", "C07", "C08", "C09"]:
                car_slots[c] = True
            for m in ["M01", "M02", "M04", "M06", "M07", "M09", "M11", "M13", "M15", "M16", "M18", "M20"]:
                moto_slots[m] = True
            print("[SIM] Busy preset applied (7 cars, 12 motorcycles).")
        else:
            print(f"[SIM] Unknown preset '{name}'. Choose: empty, busy, full")


def main():
    global running, auto_traffic
    args = parse_args()
    auto_traffic = args.auto

    print("\n" + "#" * 60)
    print("  OccupAI Computer Vision Simulator")
    print(f"  Target: {args.url}")
    print(f"  Token:  {args.token[:4]}****{args.token[-4:] if len(args.token) > 8 else ''}")
    print(f"  Heartbeat: Every {args.interval}s")
    print("#" * 60)

    # Start heartbeat sender thread
    t_sender = threading.Thread(
        target=sender_loop,
        args=(args.url, args.token, args.interval),
        daemon=True,
        name="cv-heartbeat-sender",
    )
    t_sender.start()

    # Start auto-traffic thread
    t_auto = threading.Thread(target=auto_traffic_loop, daemon=True, name="cv-auto-traffic")
    t_auto.start()

    # Give the first heartbeat a moment to complete
    time.sleep(0.5)
    print_status()
    print("Type 'help' for commands, 's' for map, or 'exit' to quit.\n")

    try:
        while running:
            try:
                line = input("occupai-sim> ").strip()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("exit", "quit", "q"):
                print("Stopping simulator...")
                running = False
                break
            elif cmd in ("help", "?"):
                print_help()
            elif cmd in ("status", "s", "map"):
                print_status()
            elif cmd == "auto":
                if len(parts) > 1 and parts[1].lower() in ("on", "1", "true"):
                    auto_traffic = True
                    print("[SIM] Auto-traffic turned ON.")
                elif len(parts) > 1 and parts[1].lower() in ("off", "0", "false"):
                    auto_traffic = False
                    print("[SIM] Auto-traffic turned OFF.")
                else:
                    auto_traffic = not auto_traffic
                    print(f"[SIM] Auto-traffic is now {'ON' if auto_traffic else 'OFF'}.")
            elif cmd == "preset":
                if len(parts) > 1:
                    set_preset(parts[1])
                else:
                    print("Usage: preset <empty|busy|full>")
            elif cmd in ("occupy", "occ"):
                if len(parts) > 1:
                    slot = parts[1].upper()
                    with slots_lock:
                        if slot in car_slots:
                            car_slots[slot] = True
                            print(f"[SIM] Slot {slot} is now OCCUPIED.")
                        elif slot in moto_slots:
                            moto_slots[slot] = True
                            print(f"[SIM] Slot {slot} is now OCCUPIED.")
                        else:
                            print(f"[SIM] Error: Slot '{slot}' not found. Use C01-C10 or M01-M20.")
                else:
                    print("Usage: occupy <slot_id> (e.g. occupy C01)")
            elif cmd in ("free", "clear_slot"):
                if len(parts) > 1:
                    slot = parts[1].upper()
                    with slots_lock:
                        if slot in car_slots:
                            car_slots[slot] = False
                            print(f"[SIM] Slot {slot} is now FREE.")
                        elif slot in moto_slots:
                            moto_slots[slot] = False
                            print(f"[SIM] Slot {slot} is now FREE.")
                        else:
                            print(f"[SIM] Error: Slot '{slot}' not found. Use C01-C10 or M01-M20.")
                else:
                    print("Usage: free <slot_id> (e.g. free C01)")
            elif cmd in ("toggle", "t"):
                if len(parts) > 1:
                    slot = parts[1].upper()
                    with slots_lock:
                        if slot in car_slots:
                            car_slots[slot] = not car_slots[slot]
                            print(f"[SIM] Slot {slot} toggled -> {'OCCUPIED' if car_slots[slot] else 'FREE'}.")
                        elif slot in moto_slots:
                            moto_slots[slot] = not moto_slots[slot]
                            print(f"[SIM] Slot {slot} toggled -> {'OCCUPIED' if moto_slots[slot] else 'FREE'}.")
                        else:
                            print(f"[SIM] Error: Slot '{slot}' not found. Use C01-C10 or M01-M20.")
                else:
                    print("Usage: toggle <slot_id> (e.g. toggle C01)")
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for list of commands.")
    except KeyboardInterrupt:
        print("\nExiting simulator...")
        running = False


if __name__ == "__main__":
    main()
