# Alpha Remit Signal

Проект для поиска подходящего момента для уведомления клиента о переводе из рублей в TJS, UZS, KGS, AMD или KZT.

Сейчас выполнен первичный EDA дневных официальных курсов Банка России с 2010 года. Курсы нормированы до рублей за одну единицу иностранной валюты; выходные и праздники не добавляются как отдельные наблюдения.

## Структура

- `docs/case.md` — исходное условие кейса;
- `docs/experiment_spec.md` — текущее состояние и рамки эксперимента;
- `docs/decisions.md` — принятые решения;
- `scripts/download_and_eda.py` — загрузка и первичная обработка данных ЦБ;
- `scripts/backtest_hard_signals.py` — единый backtest spike, level и stable-corridor сигналов;
- `scripts/backtest_ml_signals.py` — all-days boosting с nested yearly walk-forward;
- `notebooks/cbr_fx_eda.ipynb` — визуальный разведывательный анализ;
- `notebooks/hard_signals_backtest.ipynb` — единая визуализация spike, level и stable-corridor бэктестов;
- `notebooks/ml_signals_backtest.ipynb` — OOT-метрики и threshold-диагностика ML-сигналов;
- `data/processed/cbr_fx_daily_2010-01-01_2026-09-02.csv` — зафиксированный входной снимок данных;
- `data/processed/ml_cross_h_11_to_10/` — сигналы, параметры и метрики финальной ML-policy.

## Rule-based backtest

```bash
python3 scripts/backtest_hard_signals.py
```

Все выходы находятся в `data/processed/hard_signals/`:

- `walk_forward_metrics.csv` — главный out-of-time результат в целом и по годовым fold;
- `walk_forward_selections.csv` — параметры, выбранные только по доступному прошлому;
- `walk_forward_signals.csv.gz` — фактически выпущенные OOT-сигналы;
- `metrics.csv` и `signals.csv.gz` — исследовательский перебор всей заранее заданной сетки, не финальная оценка.
- `policy_coverage.csv` — число событий и причина исключения для каждой policy × corridor комбинации, включая нулевые.

Один запуск считает три независимых семейства: momentum из N=2/3/4/5 последовательных снижений, низкий уровень и первый выход из стабильного trailing-коридора. N=1 исключён как слишком быстрый и малоинформативный для текущего эксперимента; N=10/20 исключены как отсутствующие или единичные и остаются только в coverage-аудите. Суммарное изменение между T−N и T не подменяет последовательность. Календарных пушей и межсемейных комбинаций нет.

Текущий эксперимент публикует метрики только за последние три года и использует h=2/3/4/5. Более ранняя история доступна только причинным признакам и walk-forward обучению. Для каждого h политика отдельно выбирается на созревшем прошлом и проверяется только на соответствующем будущем горизонте.

`hit_rate` проверяет утверждение сообщения, а `hit_lift` — отношение hit rate сигналов к hit rate случайных дат. Симметричный `local_min_hit_rate` остаётся только диагностикой. `benefit_bps` — положение дня сигнала относительно среднего курса ±h; его 95% интервал и односторонний `p-value` против нуля считаются календарно-месячным block bootstrap.

Для `level`, momentum и выхода вниз hit означает «в следующие h публикаций не стало выгоднее»; для выхода вверх — «к T+h курс действительно вырос». Экономическая выгода у всех семейств одна: курс T против среднего ±h и проверка, что её среднее статистически значимо больше нуля.

## ML backtest

```bash
python3 scripts/backtest_ml_signals.py
```

Актуальный boosting скорит все опубликованные даты без hard-trigger gate. На официальных горизонтах `1/3/5/10/20` совместно моделируются symmetric `local_min`, forward-only `truth_now` и экономическая выгода. Модели и отдельные валютные score-thresholds выбираются nested walk-forward: внутренний прошлый год → следующий outer OOT год. Частота и cooldown в отборе не участвуют. Результаты сохраняются в `data/processed/ml_signals/`.

Воспроизведение замороженной лучшей гипотезы `h_train=11 → h_eval=10`:

```bash
.venv/bin/python scripts/backtest_ml_signals.py \
  --experiment cross-h-reproduction \
  --outer-start-year 2023
.venv/bin/python scripts/build_ml_notebook.py
```

Параметры policy берутся только из прежнего inner-OOF выбора на `h=11`; официальный
`h=10` используется исключительно для пересчёта итоговых hit/lift/benefit. Артефакты
сохраняются в `data/processed/ml_cross_h_11_to_10/`.
