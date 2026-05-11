"""Unit tests for the BuilderRequest Pydantic schema validation rules.

The model has cross-field invariants that aren't trivial to verify by
endpoint tests alone — those would need PostGIS + a real DB.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.routers.reports import BuilderRequest


def test_field_kind_requires_field_id():
    with pytest.raises(ValidationError, match="field_id"):
        BuilderRequest(kind="field")


def test_field_kind_accepts_field_id():
    req = BuilderRequest(kind="field", field_id=5)
    assert req.field_id == 5


def test_portfolio_kind_needs_no_extras():
    req = BuilderRequest(kind="portfolio")
    assert req.kind == "portfolio"
    assert req.field_id is None


def test_compare_kind_requires_field_ids():
    with pytest.raises(ValidationError, match="field_ids"):
        BuilderRequest(kind="compare")


def test_compare_kind_rejects_empty_field_ids():
    with pytest.raises(ValidationError, match="field_ids"):
        BuilderRequest(kind="compare", field_ids=[])


def test_compare_kind_accepts_multiple_field_ids():
    req = BuilderRequest(kind="compare", field_ids=[1, 2, 3])
    assert req.field_ids == [1, 2, 3]


def test_unknown_section_rejected():
    with pytest.raises(ValidationError, match="unknown sections"):
        BuilderRequest(kind="field", field_id=1, sections=["typo"])


def test_known_sections_accepted():
    req = BuilderRequest(
        kind="field",
        field_id=1,
        sections=["summary", "indices", "prediction"],
    )
    assert set(req.sections or []) == {"summary", "indices", "prediction"}
