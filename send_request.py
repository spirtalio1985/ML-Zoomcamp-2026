"""Send example sessions to a running prediction service and print the answers.

Usage:

    python send_request.py                         # service on http://localhost:9696
    python send_request.py https://my-app.fly.dev  # service deployed somewhere else
"""

from __future__ import annotations

import sys

import requests

SESSIONS = {
    "engaged visitor (many product pages, high page value)": {
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
    },
    "bounce (one product page, left immediately)": {
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
    },
}


def main() -> None:
    base_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:9696"

    health = requests.get(f"{base_url}/health", timeout=30)
    health.raise_for_status()
    print("GET /health ->", health.json())

    for name, session in SESSIONS.items():
        response = requests.post(f"{base_url}/predict", json=session, timeout=30)
        response.raise_for_status()
        print(f"POST /predict [{name}] ->", response.json())


if __name__ == "__main__":
    main()
