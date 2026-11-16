"""Feature definitions shared by training (train.py, notebook.ipynb) and serving (predict.py).

Keeping this logic in one place guarantees that a session is turned into model
features in exactly the same way when the model is trained and when it is served.

This module deliberately has no third-party imports, so it can be used inside the
slim Docker image.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Counts, durations (seconds) and Google Analytics metrics of one browsing session.
NUMERICAL_FEATURES = [
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "product_related",
    "product_related_duration",
    "bounce_rates",
    "exit_rates",
    "page_values",
    "special_day",
]

# Stored as integers in the raw data, but they are arbitrary codes (labels), not
# quantities: browser 4 is not "twice" browser 2. They are therefore one-hot encoded.
CODE_FEATURES = [
    "operating_systems",
    "browser",
    "region",
    "traffic_type",
]

CATEGORICAL_FEATURES = ["month", "visitor_type"] + CODE_FEATURES

# True/False flag, used as a single 0/1 column.
BINARY_FEATURES = ["weekend"]

FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "revenue"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

# The raw file spells June in full while every other month is a 3-letter abbreviation.
MONTH_ALIASES = {"June": "Jun"}

# Original column names in the CSV -> snake_case names used everywhere in this project.
COLUMN_RENAMES = {
    "Administrative": "administrative",
    "Administrative_Duration": "administrative_duration",
    "Informational": "informational",
    "Informational_Duration": "informational_duration",
    "ProductRelated": "product_related",
    "ProductRelated_Duration": "product_related_duration",
    "BounceRates": "bounce_rates",
    "ExitRates": "exit_rates",
    "PageValues": "page_values",
    "SpecialDay": "special_day",
    "Month": "month",
    "OperatingSystems": "operating_systems",
    "Browser": "browser",
    "Region": "region",
    "TrafficType": "traffic_type",
    "VisitorType": "visitor_type",
    "Weekend": "weekend",
    "Revenue": "revenue",
}


def _as_bool(value: Any) -> bool:
    """Interpret True/False given as a bool, a number or text such as "TRUE" / "false"."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        raise ValueError(f"cannot interpret {value!r} as a boolean")
    return bool(value)


def prepare_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one raw session into the feature dictionary expected by the model pipeline.

    `raw` uses the snake_case names above and raw values, e.g. a row of the CSV
    (after renaming the columns) or the JSON body of a `/predict` request.

    * numerical features become floats;
    * `weekend` becomes 0/1;
    * `month` is normalised to its 3-letter abbreviation;
    * code features become strings, so that `DictVectorizer` one-hot encodes them.
    """
    record: dict[str, Any] = {}
    for feature in NUMERICAL_FEATURES:
        record[feature] = float(raw[feature])
    record["weekend"] = int(_as_bool(raw["weekend"]))
    month = str(raw["month"])
    record["month"] = MONTH_ALIASES.get(month, month)
    record["visitor_type"] = str(raw["visitor_type"])
    for feature in CODE_FEATURES:
        record[feature] = str(int(raw[feature]))
    return record
