#!/usr/bin/env python3
"""Build the canonical notebook for the frozen h=11 -> h=10 experiment."""

from pathlib import Path

import nbformat as nbf


root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        r"""# ML-сигналы: frozen policy $h_{train}=11 \rightarrow h_{eval}=10$

Модель обучается на

$$good\_push_{T,11}=truth\_now_{T,11}\land[benefit_{T,11}>0].$$

Параметры policy заморожены по прежнему inner-OOF выбору на $h=11$. Горизонт
$h=10$ не участвует в обучении или выборе порога и используется только для
официальной оценки:

$$truth\_now_{T,10}=\mathbf{1}\left[P_T\leq\min(P_{T+1:T+10})\right],$$

$$lift_{10}=\frac{hit\_rate_{signal,10}}{hit\_rate_{random,10}},$$

$$benefit_{T,10}=\left(\frac{mean(P_{T-10:T+10})}{P_T}-1\right)10\,000.$$

USD/EUR/CNY используются как причинные контекстные признаки. Частота не входит
в selection. Период 2023–2026 уже использовался при разработке и не является
новым pristine holdout."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RESULT_DIR = ROOT / 'data' / 'processed' / 'ml_cross_h_11_to_10'
HARD_RESULT_DIR = ROOT / 'data' / 'processed' / 'hard_signals'
metrics = pd.read_csv(RESULT_DIR/'fold_metrics.csv', parse_dates=['fold_start','fold_end','effective_evaluation_end'])
rolling = pd.read_csv(RESULT_DIR/'rolling_window_metrics.csv', parse_dates=['fold_start','fold_end'])
final_h = pd.read_csv(RESULT_DIR/'final_2025_2026_by_horizon.csv', parse_dates=['fold_start','fold_end'])
final_h_summary = pd.read_csv(RESULT_DIR/'final_2025_2026_summary_by_horizon.csv', parse_dates=['evaluation_end_exclusive'])
# Единое требуемое правило: односторонний H1 mean(benefit)>0, alpha=5%.
# CI показывается отдельно и не участвует в булевом вердикте.
for frame in (metrics, rolling, final_h):
    frame['benefit_significant_5pct'] = frame['benefit_p_value_vs_zero'].lt(.05)
choices = pd.read_csv(RESULT_DIR/'outer_choices.csv', parse_dates=['outer_start','effective_evaluation_end'])
hard_report = pd.read_csv(HARD_RESULT_DIR/'hard_indicator_corridor_report.csv')
with (RESULT_DIR/'metadata.json').open(encoding='utf-8') as f: metadata=json.load(f)
currencies=['AMD','KGS','KZT','TJS','UZS']
plt.rcParams.update({'figure.dpi':120, 'axes.titlesize':11, 'figure.titlesize':14})
print(f\"Данные по {metadata['data_as_of']}; context features: {len(metadata['context_feature_columns'])}\")"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## Итоговый отчёт по hard-индикаторам

Матрица ниже построена по честным walk-forward сигналам последних трёх лет.
Для каждой пары `индикатор × коридор` показан худший результат среди
$h=2/3/4/5$. Самостоятельный hard-trigger допускается только если на всех этих
горизонтах одновременно выполнены три условия: $lift\geq1.3$, средняя
экономическая выгода положительна и её односторонний тест значим на уровне 5%.

Ни один hard-индикатор этот фильтр устойчиво не прошёл. Поэтому hard-правила
**не определяют даты отправки** и не входят в финансовые метрики финального
потока. Momentum, низкий уровень и выход вниз при этом показывают положительную
среднюю выгоду на всех проверенных горизонтах. Они сохраняются как причинные
факты для текста уже независимо выбранного ML-пуша, только если соответствующее
условие действительно наблюдается на дату $T$.

Это не использует будущий `benefit` конкретного события: право семейства быть
текстовым reason фиксируется по прошлому walk-forward отчёту, а сам текст на
дате $T$ выбирается исключительно из доступных на $T$ hard-признаков. Выход
вверх исключён и из этой роли: его выгода для сценария «сейчас выгодно» не
положительна."""
    ),
    nbf.v4.new_code_cell(
        """hard_matrix=(hard_report[[
    'indicator','corridor','worst_horizon',
    'hit_rate_signal_at_worst_lift','hit_rate_random_at_worst_lift',
    'min_hit_lift','median_hit_lift','min_benefit_bps','median_benefit_bps',
    'benefit_significant_all_h','standalone_trigger','ml_message_fact_eligible',
    'reason',
]].rename(columns={
    'indicator':'индикатор','corridor':'коридор','worst_horizon':'худший h',
    'hit_rate_signal_at_worst_lift':'hit rate сигнала',
    'hit_rate_random_at_worst_lift':'hit rate случайного дня',
    'min_hit_lift':'минимальный lift','median_hit_lift':'медианный lift',
    'min_benefit_bps':'минимальная выгода, bp',
    'median_benefit_bps':'медианная выгода, bp',
    'benefit_significant_all_h':'benefit значим на всех h',
    'standalone_trigger':'самостоятельный trigger',
    'ml_message_fact_eligible':'можно использовать как текст ML',
    'reason':'причина исключения',
}))
for column in ['benefit значим на всех h','самостоятельный trigger','можно использовать как текст ML']:
    hard_matrix[column]=hard_matrix[column].map({True:'Да',False:'Нет'})
display(hard_matrix.round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        r"""**Прямой вывод.** Momentum, низкий уровень и выход вниз исключаются
как самостоятельные триггеры из-за недостаточного hit-rate lift, но остаются
разрешёнными объясняющими фактами для ML-пуша. Выход вверх исключается полностью
из сценария «сейчас выгодно»: lift местами высокий, однако экономическая выгода
отрицательна. Финальным финансовым индикатором остаётся только ML-policy;
hard-факт влияет на формулировку, но не на решение об отправке."""
    ),
    nbf.v4.new_markdown_cell("## 1. Замороженные параметры и границы оценки"),
    nbf.v4.new_code_cell(
        "display(choices[['outer_start','effective_evaluation_end','model_config','score_lookback','joint_quantile']])"
    ),
    nbf.v4.new_markdown_cell(
        r"""## 2. Финальная оценка 2025–2026 на всех обязательных горизонтах

Здесь оценивается **один и тот же замороженный поток OOT-пушей** модели,
обученной с target $h_{train}=11$. Горизонт $h$ меняет только последующую
проверку исхода, но не даты пушей. Поэтому частота одинакова во всех столбцах.

Чтобы сравнение было честным, справа используется единая граница созревания:
во все столбцы входят только даты, для которых уже известны следующие 20
публикаций. Для каждого $h$ отдельно пересчитываются

$$lift_h=\frac{hit\_rate_{signal,h}}{hit\_rate_{random,h}},$$

$$benefit_{T,h}=\left(\frac{mean(P_{T-h:T+h})}{P_T}-1\right)\cdot10\,000\;bp.$$

Это профиль устойчивости выбранной policy по клиентскому горизонту, а не пять
заново настроенных моделей."""
    ),
    nbf.v4.new_code_cell(
        """evaluation_end=(final_h_summary.evaluation_end_exclusive.min()-pd.Timedelta(days=1)).date()
print(f'Финальное окно: 2025-01-01 — {evaluation_end} (единая зрелость h=20)')

fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for ax,column,title in [
    (axes[0],'truth_now_hit_rate','числитель: hit rate сигнала'),
    (axes[1],'truth_now_random','знаменатель: hit rate случайного дня'),
]:
    matrix=(final_h.pivot(index='currency',columns='horizon',values=column)
                   .reindex(index=currencies,columns=[1,3,5,10,20]))
    image=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='Blues',vmin=0,vmax=.65)
    ax.set_xticks(range(len(matrix.columns)),[f'h={h}' for h in matrix.columns])
    ax.set_yticks(range(len(currencies)),currencies); ax.set_title(title)
    for i in range(len(currencies)):
        for j in range(len(matrix.columns)):
            value=matrix.iloc[i,j]
            ax.text(j,i,f'{value:.3f}' if pd.notna(value) else '—',ha='center',va='center',fontsize=9)
    fig.colorbar(image,ax=ax,shrink=.8)
plt.show()

fig,axes=plt.subplots(1,3,figsize=(17,4.8),constrained_layout=True)
for ax,column,title,vmin,vmax,fmt in [
    (axes[0],'truth_now_lift','truth_now lift к случайному дню',0.8,1.6,'.2f'),
    (axes[1],'local_min_lift','local-min lift в окне ±h',0.8,2.5,'.2f'),
    (axes[2],'benefit_bps','экономическая выгода, bp',0,120,'.1f'),
]:
    matrix=(final_h.pivot(index='currency',columns='horizon',values=column)
                   .reindex(index=currencies,columns=[1,3,5,10,20]))
    image=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdYlGn',vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)),[f'h={h}' for h in matrix.columns])
    ax.set_yticks(range(len(currencies)),currencies); ax.set_title(title)
    for i in range(len(currencies)):
        for j in range(len(matrix.columns)):
            value=matrix.iloc[i,j]
            ax.text(j,i,format(value,fmt) if pd.notna(value) else '—',ha='center',va='center',fontsize=9)
    fig.colorbar(image,ax=ax,shrink=.8)
plt.show()

summary_table=(final_h_summary[[
    'horizon','median_truth_hit_rate','median_truth_random',
    'mean_truth_lift','min_truth_lift','mean_benefit_bps',
    'all_benefits_significant',
]].rename(columns={
    'horizon':'h',
    'median_truth_hit_rate':'Hit rate сигнала',
    'median_truth_random':'Hit rate случайного дня',
    'mean_truth_lift':'Средний lift',
    'min_truth_lift':'Минимальный lift',
    'mean_benefit_bps':'Средняя выгода, bp',
    'all_benefits_significant':'Значима для всех валют',
}))
summary_table['Значима для всех валют'] = summary_table['Значима для всех валют'].map(
    {True:'Да', False:'Нет'}
)
display(summary_table.round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        """В следующей таблице lift рассчитывается **внутри каждой строки** как
`hit_rate_signal / hit_rate_random`. Поэтому медианный lift по пяти валютам
может отличаться от отношения двух медианных hit rate."""
    ),
    nbf.v4.new_code_cell(
        """detail=(final_h[['horizon','currency','truth_now_hit_rate','truth_now_random',
                         'truth_now_lift','benefit_bps','benefit_ci_low_95',
                         'benefit_ci_high_95','benefit_significant_5pct']]
        .sort_values(['horizon','currency']))
display(detail.round(3))"""
    ),
    nbf.v4.new_markdown_cell("## 3. Честные годовые outer-метрики"),
    nbf.v4.new_code_cell(
        """annual=metrics.assign(year=metrics.fold_start.dt.year)
fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
specs=[('truth_now_lift','lift',0.8,1.6),('benefit_bps','benefit, bp',0,100),('signals_per_week','пушей в неделю',0,2)]
for ax,(column,title,vmin,vmax) in zip(axes,specs):
    matrix=annual.pivot(index='currency',columns='year',values=column).reindex(currencies)
    image=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdYlGn',vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)),matrix.columns); ax.set_yticks(range(len(currencies)),currencies); ax.set_title(title)
    for i in range(len(currencies)):
        for j in range(len(matrix.columns)):
            ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(image,ax=ax,shrink=.8)
plt.show()"""
    ),
    nbf.v4.new_markdown_cell("Зелёный годовой lift должен сохраняться по валютам, а не только после объединения нескольких лет."),
    nbf.v4.new_markdown_cell("## 4. Объединённые trailing OOT-окна"),
    nbf.v4.new_code_cell(
        """windows=list(dict.fromkeys(rolling.window))
fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
for ax,(column,title,vmin,vmax) in zip(axes,specs):
    matrix=rolling.pivot(index='currency',columns='window',values=column).reindex(index=currencies,columns=windows)
    image=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdYlGn',vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(windows)),windows,rotation=30,ha='right'); ax.set_yticks(range(len(currencies)),currencies); ax.set_title(title)
    for i in range(len(currencies)):
        for j in range(len(windows)):
            ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(image,ax=ax,shrink=.8)
plt.show()"""
    ),
    nbf.v4.new_markdown_cell("## 5. Полные обязательные метрики"),
    nbf.v4.new_code_cell(
        """columns=['window','currency','signal_count','signals_per_week','truth_now_hit_rate','truth_now_random','truth_now_lift','benefit_bps','benefit_ci_low_95','benefit_ci_high_95','benefit_significant_5pct','clustered_share_within_5_observations']
display(rolling[columns].round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        r"""# 6. Кучность ML и обязательные напоминания 5/20

Здесь разделены два разных продукта.

1. **ML-пуш «сейчас выгодно».** Только для него считаются hit rate, lift и
   экономическая выгода — это результаты раздела 2 без каких-либо календарных
   сообщений.
2. **CRM-напоминание.** Оно сообщает о плановой дате и предзаполненном переводе,
   не утверждает, что курс выгоден, поэтому lift и benefit для него не имеют
   смысла. После объединения считаются только частота и равномерность коммуникаций.

На валютный коридор разрешено не более двух **ML-пушей** в календарную неделю:
после отправленного ML-пуша сигнал следующего календарного дня подавляется, а
из остальных причинно сохраняются не более двух с понедельника по воскресенье.
Отклонённый соседний сигнал не занимает недельную квоту, поэтому более поздний
сигнал той же недели ещё может быть отправлен. После этого ровно 5-го и 20-го
всегда отправляется отдельный CRM-reminder — даже если
рядом или в тот же день есть ML. Для него недельный лимит не применяется,
поэтому общий поток иногда содержит три сообщения за неделю: эта календарная
скученность разрешена постановкой эксперимента.

Reminder не участвует в финансовой оценке. Lift и benefit ниже пересчитаны
только на ML-сигналах, оставшихся после ML-лимита."""
    ),
    nbf.v4.new_code_cell(
        """reminder_distribution = pd.read_csv(RESULT_DIR/'reminder_distribution.csv')
reminder_coverage = pd.read_csv(RESULT_DIR/'reminder_coverage.csv')
reminders = pd.read_csv(RESULT_DIR/'calendar_reminders.csv', parse_dates=['date','calendar_anchor'])
combined = pd.read_csv(RESULT_DIR/'ml_plus_reminders.csv.gz', parse_dates=['date','calendar_anchor'])
communication_metrics = pd.read_csv(RESULT_DIR/'communication_ml_by_horizon.csv')
communication_summary = pd.read_csv(RESULT_DIR/'communication_summary_by_horizon.csv')
display(reminder_coverage.round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## 6.1 Финансовые метрики относятся только к ML

Reminder-даты не входят ни в числитель, ни в случайный baseline, ни в benefit.
Но недельный лимит меняет набор реально отправленных ML-сигналов, поэтому
финансовые метрики пересчитаны по оставшимся ML-дням. Значимость определяется
односторонним тестом $H_1:\mathbb{E}[benefit]>0$ при $\alpha=0.05$; двухсторонний
95% CI остаётся диагностикой и не входит в флаг `Да/Нет`."""
    ),
    nbf.v4.new_code_cell(
        """ml_quality=communication_summary[[
    'horizon','median_truth_hit_rate','median_truth_random',
    'mean_truth_lift','min_truth_lift','mean_benefit_bps',
    'min_benefit_bps','all_benefits_significant',
    'mean_ml_pushes_per_week','mean_total_pushes_per_week',
]].rename(columns={
    'horizon':'h','median_truth_hit_rate':'hit rate ML','median_truth_random':'random hit rate',
    'mean_truth_lift':'средний lift','min_truth_lift':'минимальный lift',
    'mean_benefit_bps':'средняя выгода, bp','min_benefit_bps':'минимальная выгода, bp',
    'all_benefits_significant':'benefit значим для всех валют',
    'mean_ml_pushes_per_week':'средняя частота ML','mean_total_pushes_per_week':'средняя частота всего',
})
display(ml_quality.round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6.1.1 Детализация по валютам на основном горизонте h=10

Это основной горизонт замороженной policy `h_train=11 → h_eval=10`.
Доверительный интервал и p-value относятся только к средней выгоде ML-пушей."""
    ),
    nbf.v4.new_code_cell(
        """total_frequency=(reminder_distribution.loc[
    reminder_distribution.stream.eq('ML cap + mandatory 5/20'),
    ['currency','signals_per_week'],
].rename(columns={'signals_per_week':'все пуши/нед.'}))
h10_currency=(communication_metrics.loc[communication_metrics.horizon.eq(10),[
    'currency','truth_now_hit_rate','truth_now_random','truth_now_lift',
    'benefit_bps','benefit_ci_low_95','benefit_ci_high_95',
    'benefit_p_value_vs_zero','benefit_significant_5pct','signals_per_week',
]].merge(total_frequency,on='currency').rename(columns={
    'currency':'валюта','truth_now_hit_rate':'hit rate ML',
    'truth_now_random':'random hit rate','truth_now_lift':'lift',
    'benefit_bps':'benefit, bp','benefit_ci_low_95':'CI 95% low',
    'benefit_ci_high_95':'CI 95% high','benefit_p_value_vs_zero':'p-value',
    'benefit_significant_5pct':'benefit значим','signals_per_week':'ML/нед.',
}))
display(h10_currency.round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## 6.2 Кучность отправленных ML-пушей

Здесь анализируются только ML-пуши после ограничения «не более двух в неделю»
и запрета отправки два календарных дня подряд. Напоминания 5/20 исключены.

Для последовательных ML-пушей считаем календарный интервал
$\Delta_i=(T_i-T_{i-1})$ в днях. Например, `q90 = 30` означает, что 90% пауз
между соседними ML-пушами не превышают 30 дней. Число и доля активных недель
показывают, насколько равномерно ML-пуши покрывают тестовый период."""
    ),
    nbf.v4.new_code_cell(
        """ml_clustering=(reminder_distribution.loc[
    reminder_distribution.stream.eq('ML weekly cap + day gap'),[
        'currency','active_week_count','total_week_count','active_week_share',
        'gap_calendar_days_q90','gap_calendar_days_q95','gap_calendar_days_q99',
        'max_gap_calendar_days',
    ]].rename(columns={
        'currency':'валюта','active_week_count':'недель с ML-пушем',
        'total_week_count':'всего недель','active_week_share':'доля активных недель',
        'gap_calendar_days_q90':'q90 паузы, дней',
        'gap_calendar_days_q95':'q95 паузы, дней',
        'gap_calendar_days_q99':'q99 паузы, дней',
        'max_gap_calendar_days':'максимальная пауза, дней',
    }))
display(ml_clustering.round(3))"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## 6.3 Интерпретация

ML-лимит выполнен: ML-сообщений не больше двух в неделю и они не приходят два
календарных дня подряд. После двух безусловных reminders в месяц средняя частота
равна примерно 0.95, диапазон по валютам 0.88–1.01. Полные месяцы без сообщений
исчезли, максимальная пауза сократилась
до 16 дней.

Новая policy даёт больше коммуникаций, чем строгий общий лимит (0.83), сохраняя
все разрешённые ML-сигналы. Цена — до трёх сообщений в отдельную неделю и
короткие интервалы около календарных дат. Это допустимо только потому, что
reminder является сервисным сообщением, а не повторным заявлением «сейчас
выгодно»."""
    ),
    nbf.v4.new_code_cell(
        """examples=reminders[[
    'date','currency','calendar_anchor','days_from_anchor','reminder_fallback',
    'level_percentile_20','message_fact',
]].sort_values(['date','currency']).head(20)
display(examples)"""
    ),
]

output = root / "notebooks/ml_signals_backtest.ipynb"
nbf.write(nb, output)
print(f"Saved {output}")
