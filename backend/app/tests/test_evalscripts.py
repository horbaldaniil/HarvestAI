"""Sanity checks for evalscripts — verify they reference the bands we expect."""
from __future__ import annotations

from app.integrations.sentinel_hub.evalscripts import (
    HEATMAP_EVALSCRIPTS,
    INDICES_EVALSCRIPT,
    SUPPORTED_INDICES,
)


def test_indices_evalscript_declares_required_bands():
    for band in ("B02", "B03", "B04", "B08", "dataMask"):
        assert band in INDICES_EVALSCRIPT, f"Missing band {band}"


def test_indices_evalscript_outputs_all_indices():
    for out in ("ndvi", "evi", "ndwi", "savi", "dataMask"):
        assert f'id: "{out}"' in INDICES_EVALSCRIPT, f"Missing output {out}"


def test_indices_evalscript_masks_no_data():
    # dataMask check filters out pixels outside the scene footprint
    # and any cloud-affected scenes already removed by maxCloudCoverage.
    assert "dataMask !== 1" in INDICES_EVALSCRIPT


def test_supported_indices_match_heatmap_scripts():
    assert set(SUPPORTED_INDICES) == set(HEATMAP_EVALSCRIPTS.keys())


def test_heatmap_evalscripts_return_rgba_image():
    for name, script in HEATMAP_EVALSCRIPTS.items():
        assert "bands: 4" in script, f"{name} should output RGBA"
        assert "image/png" not in script  # format is set in request, not evalscript
        assert "ramp(" in script, f"{name} should apply colour ramp"
