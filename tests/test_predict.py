"""Tests for the feature preparation and the web service.

Run with:  uv run pytest

The tests use FastAPI's in-process test client, so no server has to be running.
They need `model.bin` (shipped in the repository, or re-created by `python train.py`).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from features import COLUMN_RENAMES, FEATURES, prepare_record
from predict import EXAMPLE_SESSION, THRESHOLD, app, pipeline, predict_single

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "online_shoppers_intention.csv"

client = TestClient(app)

# A visitor who looked at many products, reached pages with a high page value
# (cart / checkout) and rarely bounced.
HIGH_INTENT_SESSION = {
    "administrative": 4,
    "administrative_duration": 120.0,
    "informational": 1,
    "informational_duration": 30.0,
    "product_related": 60,
    "product_related_duration": 2500.0,
    "bounce_rates": 0.0,
    "exit_rates": 0.01,
    "page_values": 60.0,
    "special_day": 0.0,
    "month": "Nov",
    "operating_systems": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 2,
    "visitor_type": "New_Visitor",
    "weekend": False,
}

# A visitor who opened a single product page and left immediately (a "bounce").
BOUNCE_SESSION = {
    "administrative": 0,
    "administrative_duration": 0.0,
    "informational": 0,
    "informational_duration": 0.0,
    "product_related": 1,
    "product_related_duration": 0.0,
    "bounce_rates": 0.2,
    "exit_rates": 0.2,
    "page_values": 0.0,
    "special_day": 0.0,
    "month": "Feb",
    "operating_systems": 1,
    "browser": 1,
    "region": 1,
    "traffic_type": 1,
    "visitor_type": "Returning_Visitor",
    "weekend": False,
}


def test_prepare_record_types_and_keys():
    record = prepare_record(EXAMPLE_SESSION)
    assert set(record) == set(FEATURES)
    assert record["weekend"] in (0, 1)
    assert isinstance(record["browser"], str)
    assert isinstance(record["page_values"], float)


def test_prepare_record_normalises_june():
    session = dict(EXAMPLE_SESSION, month="June")
    assert prepare_record(session)["month"] == "Jun"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert 0.0 < body["threshold"] < 1.0


def test_predict_returns_valid_response():
    response = client.post("/predict", json=EXAMPLE_SESSION)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"purchase_probability", "purchase", "threshold"}
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert isinstance(body["purchase"], bool)
    assert body["purchase"] == (body["purchase_probability"] >= body["threshold"])


def test_predict_is_deterministic():
    first = client.post("/predict", json=EXAMPLE_SESSION).json()
    second = client.post("/predict", json=EXAMPLE_SESSION).json()
    assert first == second


def test_high_intent_scores_far_above_bounce():
    high = client.post("/predict", json=HIGH_INTENT_SESSION).json()
    low = client.post("/predict", json=BOUNCE_SESSION).json()
    assert high["purchase_probability"] > 0.5
    assert low["purchase_probability"] < 0.05
    assert high["purchase"] is True
    assert low["purchase"] is False


def test_unseen_category_code_does_not_break_the_service():
    # Browser code 99 does not exist in the training data: it must be ignored, not crash.
    response = client.post("/predict", json=dict(EXAMPLE_SESSION, browser=99))
    assert response.status_code == 200


@pytest.mark.parametrize(
    "bad_update",
    [
        {"page_values": -1.0},  # negative value
        {"bounce_rates": 1.5},  # rate above 1
        {"month": "Sept"},  # unknown month label
        {"visitor_type": "Robot"},  # unknown visitor type
        {"unexpected_field": 1},  # extra field
    ],
)
def test_invalid_payload_is_rejected(bad_update):
    response = client.post("/predict", json=dict(EXAMPLE_SESSION, **bad_update))
    assert response.status_code == 422


def test_missing_field_is_rejected():
    payload = dict(EXAMPLE_SESSION)
    del payload["page_values"]
    assert client.post("/predict", json=payload).status_code == 422


def test_api_matches_training_path_on_real_rows():
    """Serving must reproduce the training-time features exactly (no training/serving skew).

    Each raw CSV row is sent through the API exactly as a client would send it, and the
    result is compared with scoring the same row the way train.py builds its features.
    """
    with DATA_PATH.open(newline="") as f_in:
        rows = list(csv.DictReader(f_in))[::411]  # 30 rows spread over the whole file
    assert len(rows) == 30

    for row in rows:
        raw = {COLUMN_RENAMES[name]: value for name, value in row.items()}
        raw.pop("revenue")

        # Training path: raw CSV values -> prepare_record -> pipeline
        expected = float(pipeline.predict_proba([prepare_record(dict(raw, weekend=raw["weekend"] == "TRUE"))])[0, 1])

        # Serving path: typed JSON payload -> API
        payload = {
            "administrative": int(raw["administrative"]),
            "administrative_duration": float(raw["administrative_duration"]),
            "informational": int(raw["informational"]),
            "informational_duration": float(raw["informational_duration"]),
            "product_related": int(raw["product_related"]),
            "product_related_duration": float(raw["product_related_duration"]),
            "bounce_rates": float(raw["bounce_rates"]),
            "exit_rates": float(raw["exit_rates"]),
            "page_values": float(raw["page_values"]),
            "special_day": float(raw["special_day"]),
            "month": "Jun" if raw["month"] == "June" else raw["month"],
            "operating_systems": int(raw["operating_systems"]),
            "browser": int(raw["browser"]),
            "region": int(raw["region"]),
            "traffic_type": int(raw["traffic_type"]),
            "visitor_type": raw["visitor_type"],
            "weekend": raw["weekend"] == "TRUE",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["purchase_probability"] == pytest.approx(expected, abs=1e-6)


def test_predict_single_matches_threshold():
    probability, decision = predict_single(EXAMPLE_SESSION)
    assert decision == (probability >= THRESHOLD)
