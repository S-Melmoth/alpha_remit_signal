"""Strict, bounded access to the official CBR historical-rate XML endpoint."""

from __future__ import annotations

import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from numbers import Real
from types import MappingProxyType

TARGET_CURRENCIES = ("TJS", "UZS", "KGS", "AMD", "KZT")
CONTROL_CURRENCIES = ("USD", "EUR", "CNY")
ALL_CURRENCIES = TARGET_CURRENCIES + CONTROL_CURRENCIES

CURRENCY_IDS = MappingProxyType(
    {
        "TJS": "R01670",
        "UZS": "R01717",
        "KGS": "R01370",
        "AMD": "R01060",
        "KZT": "R01335",
        "USD": "R01235",
        "EUR": "R01239",
        "CNY": "R01375",
    }
)

CBR_ENDPOINT = "https://www.cbr.ru/scripts/XML_dynamic.asp"
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
MAX_RETRY_DELAY_SECONDS = 30.0
_DATE_RE = re.compile(r"\A\d{2}\.\d{2}\.\d{4}\Z")
_DECIMAL_RE = re.compile(r"\A\d+(?:[.,]\d+)?\Z")


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True)
class RateRecord:
    """One CBR quotation normalized to RUB per one foreign-currency unit."""

    currency: str
    effective_date: date
    nominal: float
    value_rub: float
    rub_per_unit: float

    def __post_init__(self) -> None:
        if self.currency not in CURRENCY_IDS:
            raise ValueError(f"Unsupported CBR currency: {self.currency!r}")
        if type(self.effective_date) is not date:
            raise ValueError("effective_date must be a date, not a datetime")
        nominal = _finite_positive(self.nominal, "nominal")
        value_rub = _finite_positive(self.value_rub, "value_rub")
        rub_per_unit = _finite_positive(self.rub_per_unit, "rub_per_unit")
        if not nominal.is_integer():
            raise ValueError("nominal must be a positive whole number of units")
        calculated = value_rub / nominal
        if not math.isclose(calculated, rub_per_unit, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("value_rub / nominal does not match rub_per_unit")
        object.__setattr__(self, "nominal", nominal)
        object.__setattr__(self, "value_rub", value_rub)
        # The normalized field has one canonical binary-float representation:
        # it is always derived from the preserved CBR Value and Nominal fields.
        object.__setattr__(self, "rub_per_unit", calculated)


class CBRUnavailableError(RuntimeError):
    """The bounded set of attempts could not obtain a CBR response."""


def _parse_date(text: str | None, name: str) -> date:
    if text is None or not _DATE_RE.fullmatch(text):
        raise ValueError(f"Invalid CBR {name}")
    try:
        return datetime.strptime(text, "%d.%m.%Y").date()
    except ValueError as exc:
        raise ValueError(f"Invalid CBR {name}") from exc


def _parse_decimal(text: str | None, name: str) -> Decimal:
    if text is None:
        raise ValueError(f"Missing CBR {name}")
    compact = text.replace(" ", "")
    if not _DECIMAL_RE.fullmatch(compact):
        raise ValueError(f"Invalid CBR {name}")
    try:
        value = Decimal(compact.replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid CBR {name}") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"Invalid CBR {name}")
    return value


def _one_child(record: ET.Element, name: str) -> str | None:
    matches = [child for child in record if child.tag == name]
    if len(matches) != 1 or list(matches[0]):
        raise ValueError(f"CBR Record must contain exactly one scalar {name}")
    return matches[0].text


def _parse_document(
    payload: bytes,
    currency: str,
    *,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> tuple[list[RateRecord], tuple[date, date] | None]:
    if currency not in CURRENCY_IDS:
        raise ValueError(f"Unsupported CBR currency: {currency!r}")
    if not isinstance(payload, bytes):
        raise ValueError("CBR payload must be bytes")
    if len(payload) > max_bytes:
        raise ValueError("CBR payload exceeds the parsing size limit")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError("CBR payload must not contain a DTD or entities")
    try:
        root = ET.fromstring(payload)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("Malformed CBR XML payload") from exc
    expected_id = CURRENCY_IDS[currency]
    if root.tag != "ValCurs" or root.attrib.get("ID") != expected_id:
        raise ValueError("Unexpected CBR document type or currency ID")

    range_start_text = root.attrib.get("DateRange1")
    range_end_text = root.attrib.get("DateRange2")
    if (range_start_text is None) != (range_end_text is None):
        raise ValueError("Incomplete CBR date range")
    declared_range: tuple[date, date] | None = None
    if range_start_text is not None:
        range_start = _parse_date(range_start_text, "DateRange1")
        range_end = _parse_date(range_end_text, "DateRange2")
        if range_start > range_end:
            raise ValueError("CBR DateRange1 must not exceed DateRange2")
        declared_range = (range_start, range_end)

    records: list[RateRecord] = []
    seen_dates: set[date] = set()
    for element in root:
        if element.tag != "Record":
            raise ValueError(f"Unexpected CBR XML element: {element.tag!r}")
        if element.attrib.get("Id") != expected_id:
            raise ValueError("Unexpected CBR record currency ID")
        unexpected = {child.tag for child in element} - {"Nominal", "Value", "VunitRate"}
        if unexpected:
            raise ValueError(f"Unexpected CBR Record fields: {sorted(unexpected)}")
        effective_date = _parse_date(element.attrib.get("Date"), "Record Date")
        if effective_date in seen_dates:
            raise ValueError(f"Duplicate CBR record for {currency} on {effective_date}")
        if declared_range is not None and not (
            declared_range[0] <= effective_date <= declared_range[1]
        ):
            raise ValueError("CBR record lies outside the declared date range")

        nominal_decimal = _parse_decimal(_one_child(element, "Nominal"), "Nominal")
        value_decimal = _parse_decimal(_one_child(element, "Value"), "Value")
        unit_decimal = _parse_decimal(_one_child(element, "VunitRate"), "VunitRate")
        if nominal_decimal != nominal_decimal.to_integral_value():
            raise ValueError("CBR Nominal must be a positive whole number")
        if value_decimal / nominal_decimal != unit_decimal:
            raise ValueError("CBR Value / Nominal does not exactly match VunitRate")

        nominal = float(nominal_decimal)
        value_rub = float(value_decimal)
        rub_per_unit = value_rub / nominal
        record = RateRecord(
            currency=currency,
            effective_date=effective_date,
            nominal=nominal,
            value_rub=value_rub,
            rub_per_unit=rub_per_unit,
        )
        records.append(record)
        seen_dates.add(effective_date)

    records.sort(key=lambda item: item.effective_date)
    return records, declared_range


def parse_rates(payload: bytes, currency: str) -> list[RateRecord]:
    """Parse one CBR response; an empty valid ``ValCurs`` is a successful result."""

    records, _ = _parse_document(payload, currency)
    return records


Transport = Callable[[str, float, int], bytes]
Sleeper = Callable[[float], object]


def _stdlib_transport(url: str, timeout_seconds: float, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "alfa-transfer-signals/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise urllib.error.HTTPError(url, status, "unexpected HTTP status", None, None)
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("CBR response exceeds max_response_bytes")
    return payload


class CBRClient:
    """Small official-endpoint client with bounded I/O and retry behavior.

    An injected transport receives ``(url, timeout_seconds, max_response_bytes)``
    and returns response bytes. Only transport failures are retried; malformed or
    semantically inconsistent XML fails immediately.
    """

    def __init__(
        self,
        timeout_seconds: float = 10,
        attempts: int = 3,
        *,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        backoff_seconds: float = 0.25,
        transport: Transport | None = None,
        sleep: Sleeper | None = None,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
            raise ValueError("timeout_seconds must be a positive finite number")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0 or timeout > 300:
            raise ValueError("timeout_seconds must be in (0, 300]")
        if type(attempts) is not int or not 1 <= attempts <= 10:
            raise ValueError("attempts must be an integer in [1, 10]")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 10_000_000:
            raise ValueError("max_response_bytes must be an integer in [1, 10000000]")
        if isinstance(backoff_seconds, bool) or not isinstance(backoff_seconds, Real):
            raise ValueError("backoff_seconds must be a nonnegative finite number")
        backoff = float(backoff_seconds)
        if not math.isfinite(backoff) or not 0 <= backoff <= 60:
            raise ValueError("backoff_seconds must be in [0, 60]")
        if transport is not None and not callable(transport):
            raise ValueError("transport must be callable")
        if sleep is not None and not callable(sleep):
            raise ValueError("sleep must be callable")
        self.timeout_seconds = timeout
        self.attempts = attempts
        self.max_response_bytes = max_response_bytes
        self.backoff_seconds = backoff
        self._transport = transport or _stdlib_transport
        self._sleep = sleep or time.sleep

    @staticmethod
    def _url(currency: str, start: date, end: date) -> str:
        query = urllib.parse.urlencode(
            {
                "date_req1": start.strftime("%d/%m/%Y"),
                "date_req2": end.strftime("%d/%m/%Y"),
                "VAL_NM_RQ": CURRENCY_IDS[currency],
            }
        )
        return f"{CBR_ENDPOINT}?{query}"

    def fetch(self, currency: str, start: date, end: date) -> list[RateRecord]:
        if currency not in CURRENCY_IDS:
            raise ValueError(f"Unsupported CBR currency: {currency!r}")
        if type(start) is not date or type(end) is not date:
            raise ValueError("start and end must be dates, not datetimes")
        if start > end:
            raise ValueError("start must not exceed end")
        url = self._url(currency, start, end)
        payload: bytes | None = None
        last_error: BaseException | None = None
        for attempt in range(self.attempts):
            try:
                payload = self._transport(url, self.timeout_seconds, self.max_response_bytes)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and not 500 <= exc.code <= 599:
                    raise CBRUnavailableError(
                        f"CBR returned non-retryable HTTP {exc.code} for {currency}"
                    ) from exc
                last_error = exc
                if attempt + 1 < self.attempts:
                    delay = min(self.backoff_seconds * (2**attempt), MAX_RETRY_DELAY_SECONDS)
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after is not None and retry_after.strip().isdigit():
                        delay = min(float(retry_after), MAX_RETRY_DELAY_SECONDS)
                    self._sleep(delay)
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    self._sleep(
                        min(
                            self.backoff_seconds * (2**attempt),
                            MAX_RETRY_DELAY_SECONDS,
                        )
                    )
        if payload is None:
            raise CBRUnavailableError(
                f"CBR request failed after {self.attempts} attempts for {currency}"
            ) from last_error
        if not isinstance(payload, bytes):
            raise ValueError("CBR transport must return bytes")
        if len(payload) > self.max_response_bytes:
            raise ValueError("CBR response exceeds max_response_bytes")

        records, declared_range = _parse_document(
            payload, currency, max_bytes=self.max_response_bytes
        )
        if declared_range is not None and declared_range != (start, end):
            raise ValueError("CBR response date range does not match the request")
        if any(not start <= record.effective_date <= end for record in records):
            raise ValueError("CBR response contains a record outside the requested range")
        return records


__all__ = [
    "ALL_CURRENCIES",
    "CBRClient",
    "CBRUnavailableError",
    "CBR_ENDPOINT",
    "CONTROL_CURRENCIES",
    "CURRENCY_IDS",
    "RateRecord",
    "MAX_RETRY_DELAY_SECONDS",
    "TARGET_CURRENCIES",
    "parse_rates",
]
