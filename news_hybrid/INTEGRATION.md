# Подключение к сервису команды

Главный интерфейс:

```python
decide(payload: Mapping, *, state_path: str | Path) -> dict
```

## Вход

```json
{
  "event_id": "cbr:AMD:2026-09-08",
  "decision_at": "2026-09-07T15:00:00Z",
  "features_available_at": "2026-09-07T14:58:00Z",
  "currency": "AMD",
  "rub_per_unit": 0.2194,
  "effective_date": "2026-09-08",
  "price_features": {
    "corridor_move_1_bps": 4.2,
    "corridor_move_5_bps": 11.7,
    "corridor_move_20_bps": 32.1,
    "corridor_level_percentile_20": 0.55,
    "corridor_level_percentile_120": 0.63,
    "corridor_volatility_20_bps": 71.4,
    "corridor_reversal_score_3_bps": -1.8,
    "rub_common_move_1_bps": 2.1,
    "rub_common_move_5_bps": 8.4,
    "rub_common_move_20_bps": 25.3,
    "days_since_prior_publication": 1.0,
    "publication_is_friday": 0.0,
    "publication_month_sin": -0.8660254038,
    "publication_month_cos": -0.5
  },
  "news_events": [
    {
      "key": "cbr-rate-2026-09",
      "available_at": "2026-09-06T12:00:00Z",
      "event_type": "CENTRAL_BANK_DECISION",
      "headline": "Банк России сообщил решение по ключевой ставке",
      "source_urls": ["https://example.org/source"],
      "rub_applies": true,
      "rub_action": "hold",
      "recipient_currencies": [],
      "recipient_actions": {},
      "global_applies": false
    }
  ]
}
```

Допустимые `event_type`:

- `CENTRAL_BANK_DECISION`
- `RATE_EXPECTATION`
- `SANCTIONS`
- `OIL_SUPPLY`
- `FX_FLOW_ANNOUNCEMENT`

Действие ставки: `raise`, `hold`, `cut` или `null`. Действие передаётся только
когда источник уже сообщил факт. Ожидание рынка нельзя записывать как факт.
Все timestamps содержат timezone. Будущие новости отклоняются.

`event_id` идентифицирует одно решение по валюте и публикации курса. Повтор с
теми же данными возвращает прежний результат. Повтор с изменёнными данными
отклоняется. Это защищает от дублирования после сбоя worker.

## Выход

При готовом сигнале:

```json
{
  "status": "READY",
  "reason": "ready",
  "currency": "AMD",
  "score": 0.42,
  "threshold": 0.31,
  "news_gate": true,
  "notification": {
    "notification_id": "...",
    "title": "Проверьте курс для перевода в Армению",
    "body": "Курс ЦБ на 08.09.2026: 1 AMD = 0,2194 ₽. Сравните итоговый курс перевода в приложении."
  }
}
```

В остальных случаях `status=ABSTAIN`, `notification=null`, а `reason` объясняет
причину. Score не является подтверждённой вероятностью и не должен показываться
клиенту.

## Ответственность компонентов

| Компонент | Ответственность |
|---|---|
| Источник курсов команды | Получить и нормализовать котировки ЦБ |
| Новостной адаптер команды | Дедуплицировать события и заполнить проверенные поля новости |
| `news_hybrid` | Признаки, score, эпизод, лимит и готовый текст |
| Сервис отправки | Получатели, согласия, очередь и фактическая доставка |

Не удаляйте SQLite-состояние при каждом запуске. Перед горизонтальным
масштабированием разместите адаптер вокруг одного общего state backend либо
оставьте один worker принятия решений. Включение реальных отправок требует
интеграционного теста с очередью команды.
