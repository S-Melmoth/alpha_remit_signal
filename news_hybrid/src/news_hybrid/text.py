"""Neutral Russian notification text for a model-selected moment."""

from decimal import ROUND_HALF_UP, Decimal, localcontext

DESTINATIONS = {
    "AMD": "Армению",
    "KGS": "Кыргызстан",
    "KZT": "Казахстан",
    "TJS": "Таджикистан",
    "UZS": "Узбекистан",
}


def format_rate(rate: float) -> str:
    value = Decimal(str(rate))
    places = max(4, 3 - value.adjusted())
    with localcontext() as context:
        context.prec = max(32, value.adjusted() + places + 4)
        rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return format(rounded, "f").rstrip("0").rstrip(".").replace(".", ",")


def render_push(currency: str, rate: float, effective_date) -> tuple[str, str]:
    title = f"Проверьте курс для перевода в {DESTINATIONS[currency]}"
    body = (
        f"Курс ЦБ на {effective_date:%d.%m.%Y}: "
        f"1 {currency} = {format_rate(rate)} ₽. "
        "Сравните итоговый курс перевода в приложении."
    )
    if len(title) > 64 or len(body) > 200:
        raise ValueError("rendered push text exceeds the delivery contract")
    return title, body
