#!/usr/bin/env python3
"""Download official CBR FX histories and prepare a first EDA dataset.

The CBR publishes RUB per stated nominal of a foreign currency. This script
normalises every observation to RUB per one unit of the recipient currency.
Only dates explicitly published by CBR are kept: calendar forward-filling is
deliberately left to later feature engineering so weekends cannot become
synthetic market observations.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


CBR_DYNAMIC_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
CURRENCIES = {
    "AMD": {"cbr_id": "R01060", "name": "Armenian dram", "series_type": "target"},
    "KGS": {"cbr_id": "R01370", "name": "Kyrgyz som", "series_type": "target"},
    "KZT": {"cbr_id": "R01335", "name": "Kazakh tenge", "series_type": "target"},
    "TJS": {"cbr_id": "R01670", "name": "Tajik somoni", "series_type": "target"},
    "UZS": {"cbr_id": "R01717", "name": "Uzbek sum", "series_type": "target"},
    "USD": {"cbr_id": "R01235", "name": "US dollar", "series_type": "context"},
    "EUR": {"cbr_id": "R01239", "name": "Euro", "series_type": "context"},
    "CNY": {"cbr_id": "R01375", "name": "Chinese yuan", "series_type": "context"},
}


def fetch_history(cbr_id: str, start: date, end: date) -> bytes:
    params = {
        "date_req1": start.strftime("%d/%m/%Y"),
        "date_req2": end.strftime("%d/%m/%Y"),
        "VAL_NM_RQ": cbr_id,
    }
    url = f"{CBR_DYNAMIC_URL}?{urllib.parse.urlencode(params)}"
    # The macOS Python bundled in the execution environment has no configured
    # CA bundle, whereas the system curl does. curl still verifies the TLS
    # certificate by default; do not use an unverified SSL context here.
    result = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location", url],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return result.stdout


def parse_history(payload: bytes, currency: str, series_type: str) -> list[dict[str, object]]:
    root = ET.fromstring(payload)
    rows: list[dict[str, object]] = []
    for record in root.findall("Record"):
        obs_date = datetime.strptime(record.attrib["Date"], "%d.%m.%Y").date()
        nominal = int(record.findtext("Nominal", ""))
        value = float(record.findtext("Value", "").replace(",", "."))
        rows.append(
            {
                "date": obs_date.isoformat(),
                "currency": currency,
                "series_type": series_type,
                "cbr_nominal": nominal,
                "rub_per_nominal": value,
                "rub_per_unit": value / nominal,
            }
        )
    return rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * q
    lower, upper = int(index), min(int(index) + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["currency"])].append(row)

    summary: list[dict[str, object]] = []
    for currency, items in sorted(grouped.items()):
        items.sort(key=lambda x: str(x["date"]))
        prices = [float(x["rub_per_unit"]) for x in items]
        returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        zero_returns = sum(abs(x) < 1e-12 for x in returns)
        summary.append(
            {
                "currency": currency,
                "observations": len(items),
                "start": items[0]["date"],
                "end": items[-1]["date"],
                "nominal_values_seen": ",".join(map(str, sorted({int(x["cbr_nominal"]) for x in items}))),
                "first_rub_per_unit": prices[0],
                "last_rub_per_unit": prices[-1],
                "total_change_pct": (prices[-1] / prices[0] - 1) * 100,
                "min_rub_per_unit": min(prices),
                "max_rub_per_unit": max(prices),
                "daily_return_std_pct": (sum((x - sum(returns) / len(returns)) ** 2 for x in returns) / (len(returns) - 1)) ** 0.5 * 100,
                "abs_daily_return_p95_pct": percentile([abs(x) for x in returns], 0.95) * 100,
                "zero_return_share_pct": zero_returns / len(returns) * 100,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output_dir)

    rows: list[dict[str, object]] = []
    for currency, metadata in CURRENCIES.items():
        payload = fetch_history(str(metadata["cbr_id"]), start, end)
        raw_path = output_dir / "raw" / f"cbr_{currency}_{start}_{end}.xml"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
        rows.extend(parse_history(payload, currency, str(metadata["series_type"])))

    rows.sort(key=lambda x: (str(x["date"]), str(x["currency"])))
    cleaned_path = output_dir / "processed" / f"cbr_fx_daily_{start}_{end}.csv"
    write_csv(cleaned_path, rows)
    summary = build_summary(rows)
    write_csv(output_dir / "processed" / f"cbr_fx_eda_summary_{start}_{end}.csv", summary)
    print(f"saved {len(rows)} observations to {cleaned_path}")


if __name__ == "__main__":
    main()
