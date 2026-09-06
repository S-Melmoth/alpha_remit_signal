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
metrics = pd.read_csv(RESULT_DIR/'fold_metrics.csv', parse_dates=['fold_start','fold_end','effective_evaluation_end'])
rolling = pd.read_csv(RESULT_DIR/'rolling_window_metrics.csv', parse_dates=['fold_start','fold_end'])
final_h = pd.read_csv(RESULT_DIR/'final_2025_2026_by_horizon.csv', parse_dates=['fold_start','fold_end'])
final_h_summary = pd.read_csv(RESULT_DIR/'final_2025_2026_summary_by_horizon.csv', parse_dates=['evaluation_end_exclusive'])
choices = pd.read_csv(RESULT_DIR/'outer_choices.csv', parse_dates=['outer_start','effective_evaluation_end'])
with (RESULT_DIR/'metadata.json').open(encoding='utf-8') as f: metadata=json.load(f)
currencies=['AMD','KGS','KZT','TJS','UZS']
plt.rcParams.update({'figure.dpi':120, 'axes.titlesize':11, 'figure.titlesize':14})
print(f\"Данные по {metadata['data_as_of']}; context features: {len(metadata['context_feature_columns'])}\")"""
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
]

output = root / "notebooks/ml_signals_backtest.ipynb"
nbf.write(nb, output)
print(f"Saved {output}")
