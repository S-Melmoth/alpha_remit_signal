# Alpha Remit Signal

Проект для поиска подходящего момента для уведомления клиента о переводе из рублей в TJS, UZS, KGS, AMD или KZT.

Сейчас выполнен первичный EDA дневных официальных курсов Банка России с 2010 года. Курсы нормированы до рублей за одну единицу иностранной валюты; выходные и праздники не добавляются как отдельные наблюдения.

## Структура

- `docs/case.md` — исходное условие кейса;
- `docs/experiment_spec.md` — текущее состояние и рамки эксперимента;
- `docs/decisions.md` — принятые решения;
- `scripts/download_and_eda.py` — загрузка и первичная обработка данных ЦБ;
- `notebooks/cbr_fx_eda.ipynb` — визуальный разведывательный анализ;
- `data/raw/` и `data/processed/` — воспроизводимый снимок данных.

## Запуск

Проект использует Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_and_eda.py --start 2010-01-01
```

Для запуска анализа откройте `notebooks/cbr_fx_eda.ipynb` в VS Code и выберите ядро из `.venv`: **Select Kernel → Python Environments → `.venv`**.

Критическое ограничение дальнейшей работы: все признаки на дату T должны использовать только информацию, доступную на T, а проверка моделей должна выполняться walk-forward.
