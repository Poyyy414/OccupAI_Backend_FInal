import pytest

from backend.slot_adjuster import (
    DemandLevel,
    GridAdjuster,
    LayoutConfigError,
    build_layout,
    load_normalized_layout,
)


@pytest.mark.parametrize("role,expected", [("car", 10), ("motorcycle", 20)])
def test_role_layout_has_exact_official_count(role, expected):
    assert len(build_layout(640, 480, camera_role=role)) == expected


@pytest.mark.parametrize("role", ["car", "motorcycle"])
def test_demand_levels_cannot_change_fixed_layout(role):
    adjuster = GridAdjuster(640, 480, camera_role=role)
    layouts = [
        adjuster.slots_for_demand(level)
        for level in (DemandLevel.NORMAL, DemandLevel.BUSY, DemandLevel.HIGH)
    ]

    assert layouts[0] == layouts[1] == layouts[2]


@pytest.mark.parametrize("role", ["car", "motorcycle"])
def test_scaled_boxes_stay_inside_frame_and_configured_parking_area(role):
    width, height = 1280, 720
    normalized = load_normalized_layout(role)
    scaled = build_layout(width, height, camera_role=role)
    area = normalized["parking_area"]

    for rect in scaled.values():
        x1, y1, x2, y2 = rect
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert x1 >= round(area[0] * width)
        assert y1 >= round(area[1] * height)
        assert x2 <= round(area[2] * width)
        assert y2 <= round(area[3] * height)


def test_layout_validation_rejects_slot_in_exclusion_area(tmp_path):
    config = tmp_path / "bad-layout.json"
    config.write_text(
        '{"layouts":{"car":{"configured_capacity":10,'
        '"parking_area":[0,0,1,1],"exclusion_areas":[[0,0,1,1]],'
        '"slots":['
        + ",".join(
            f'{{"id":"C{i:02d}","rect":[0.1,0.1,0.2,0.2]}}'
            for i in range(1, 11)
        )
        + "]}}}",
        encoding="utf-8",
    )

    with pytest.raises(LayoutConfigError, match="overlaps"):
        load_normalized_layout("car", config)
