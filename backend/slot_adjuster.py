"""Fixed, role-specific parking layout support for OccupAI camera workers.

Parking geometry is loaded from normalized coordinates in JSON. Demand is a
status/pricing signal only: it can never add, remove, resize, or move a box.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PH_TZ = ZoneInfo("Asia/Manila")
OFFICIAL_CAMERA_CAPACITIES = {"car": 10, "motorcycle": 20}
OFFICIAL_LOT_CAPACITY = sum(OFFICIAL_CAMERA_CAPACITIES.values())
DEFAULT_LAYOUT_PATH = (
    Path(__file__).resolve().parent.parent / "yolo_service" / "camera_layouts.json"
)


class LayoutConfigError(ValueError):
    """Raised when a saved layout could produce unsafe or incorrect boxes."""


class DemandLevel:
    LOW = "LOW"
    NORMAL = "NORMAL"
    BUSY = "BUSY"
    HIGH = "HIGH"

    @classmethod
    def all(cls):
        return {cls.LOW, cls.NORMAL, cls.BUSY, cls.HIGH}


def _normalized_rect(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise LayoutConfigError(f"{label} must contain [x1, y1, x2, y2]")
    try:
        x1, y1, x2, y2 = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise LayoutConfigError(f"{label} contains a non-numeric coordinate") from exc
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise LayoutConfigError(
            f"{label} must be a non-empty normalized rectangle inside 0..1"
        )
    return x1, y1, x2, y2


def _contains(outer, inner):
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _intersects(left, right):
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _layout_path(config_path=None):
    raw = config_path or os.getenv("CAMERA_LAYOUT_FILE") or DEFAULT_LAYOUT_PATH
    return Path(raw).expanduser().resolve()


def load_normalized_layout(camera_role, config_path=None):
    """Load and validate one camera's saved normalized layout."""
    role = str(camera_role or "").strip().lower()
    if role not in OFFICIAL_CAMERA_CAPACITIES:
        raise LayoutConfigError("CAMERA_ROLE must be 'car' or 'motorcycle'")

    path = _layout_path(config_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LayoutConfigError(f"Camera layout file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LayoutConfigError(f"Camera layout file is not valid JSON: {path}") from exc

    layout = (document.get("layouts") or {}).get(role)
    if not isinstance(layout, dict):
        raise LayoutConfigError(f"No saved layout exists for camera role {role!r}")

    expected = OFFICIAL_CAMERA_CAPACITIES[role]
    if int(layout.get("configured_capacity", -1)) != expected:
        raise LayoutConfigError(
            f"{role} configured_capacity must be exactly {expected}"
        )

    parking_area = _normalized_rect(layout.get("parking_area"), f"{role}.parking_area")
    exclusions = []
    for index, item in enumerate(layout.get("exclusion_areas") or []):
        value = item.get("rect") if isinstance(item, dict) else item
        exclusions.append(
            _normalized_rect(value, f"{role}.exclusion_areas[{index}]")
        )

    slots = layout.get("slots")
    if not isinstance(slots, list) or len(slots) != expected:
        actual = len(slots) if isinstance(slots, list) else 0
        raise LayoutConfigError(
            f"{role} layout must contain exactly {expected} slots; found {actual}"
        )

    normalized = {}
    for index, item in enumerate(slots):
        if not isinstance(item, dict):
            raise LayoutConfigError(f"{role}.slots[{index}] must be an object")
        name = str(item.get("id") or "").strip()
        if not name or name in normalized:
            raise LayoutConfigError(f"{role}.slots[{index}] has a missing or duplicate id")
        rect = _normalized_rect(item.get("rect"), f"{role}.slots[{index}].rect")
        if not _contains(parking_area, rect):
            raise LayoutConfigError(f"Slot {name} is outside {role}.parking_area")
        if any(_intersects(rect, exclusion) for exclusion in exclusions):
            raise LayoutConfigError(f"Slot {name} overlaps a configured exclusion area")
        normalized[name] = rect

    return {
        "role": role,
        "configured_capacity": expected,
        "parking_area": parking_area,
        "exclusion_areas": exclusions,
        "slots": normalized,
        "source": str(path),
    }


def _scale_rect(rect, frame_width, frame_height):
    width = int(frame_width)
    height = int(frame_height)
    if width < 1 or height < 1:
        raise LayoutConfigError("Camera frame dimensions must be positive")
    x1 = max(0, min(width - 1, round(rect[0] * width)))
    y1 = max(0, min(height - 1, round(rect[1] * height)))
    x2 = max(x1 + 1, min(width, round(rect[2] * width)))
    y2 = max(y1 + 1, min(height, round(rect[3] * height)))
    return x1, y1, x2, y2


def build_layout(frame_width, frame_height, camera_role=None, config_path=None, **_ignored):
    """Scale a saved role layout to the current camera resolution.

    Extra keyword arguments are ignored for compatibility with older callers.
    They can no longer alter the fixed layout.
    """
    role = camera_role or os.getenv("CAMERA_ROLE")
    layout = load_normalized_layout(role, config_path)
    return {
        name: _scale_rect(rect, frame_width, frame_height)
        for name, rect in layout["slots"].items()
    }


class GridAdjuster:
    """Compatibility facade whose demand layouts are deliberately identical."""

    def __init__(self, frame_width, frame_height, camera_role=None, config_path=None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_role = camera_role or os.getenv("CAMERA_ROLE")
        self.config_path = config_path
        self._fixed_slots = build_layout(
            frame_width,
            frame_height,
            camera_role=self.camera_role,
            config_path=config_path,
        )

    def slots_for_demand(self, demand):
        if str(demand or "").upper() not in DemandLevel.all():
            raise LayoutConfigError(f"Unknown demand level: {demand!r}")
        return dict(self._fixed_slots)


class SlotState:
    """Thread-safe immutable-layout state used by the detector loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._base_slots = {}
        self.demand = DemandLevel.NORMAL
        self.forecast_veh = None
        self.current_occ = None
        self.last_adjusted = None
        self.adjustment_reason = "Fixed saved camera layout"

    def set_base_slots(self, slots):
        with self._lock:
            self._base_slots = dict(slots)

    def update_slots(self, _new_slots, demand, forecast_veh, current_occ, reason):
        """Legacy API: update demand metadata without touching geometry."""
        with self._lock:
            self.demand = demand if demand in DemandLevel.all() else DemandLevel.NORMAL
            self.forecast_veh = forecast_veh
            self.current_occ = current_occ
            self.last_adjusted = datetime.now(PH_TZ)
            self.adjustment_reason = reason or "Demand is informational only"

    @property
    def active_slots(self):
        with self._lock:
            return dict(self._base_slots)

    @property
    def base_slots(self):
        return self.active_slots

    def check_and_clear_bg_reset(self):
        return False

    def summary(self):
        with self._lock:
            return {
                "demand": self.demand,
                "forecast_veh": self.forecast_veh,
                "current_occ": self.current_occ,
                "n_slots": len(self._base_slots),
                "last_adjusted": (
                    self.last_adjusted.strftime("%H:%M:%S")
                    if self.last_adjusted
                    else None
                ),
                "reason": self.adjustment_reason,
                "layout_mode": "fixed",
            }
