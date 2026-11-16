"""生成可离线打开的研报和独立收益图。"""

from html import escape
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import scipy

from backtest import (
    ASSETS, BASE_COST, BOOTSTRAP_REPS, BOOTSTRAP_SEED, CODES, COSTS, PER_YEAR,
    STRATEGIES, TEST_YEAR, VOL_TARGET, matched_forecasts,
)
from risk import EVENT_QUANTILE, HORIZON, LONG_HALF_LIFE, RIDGE, RISK_NAMES, SHORT_HALF_LIFE


PAPER = "#f7f5ef"
INK = "#172f3c"
MUTED = "#526674"
GRID = "#dee4e0"
TEAL = "#137b70"
ORANGE = "#aa573c"
BLUE = "#46658f"
PURPLE = "#766791"
GRAY = "#6a7981"
PALETTE = (TEAL, ORANGE, BLUE, PURPLE, GRAY)

CSS = """
:root { color-scheme:light; --paper:#f7f5ef; --sheet:#fffefb; --ink:#172f3c; --muted:#526674;
  --teal:#096d60; --rule:#d9e1de; --mono:"IBM Plex Mono",Consolas,monospace; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:15px/1.75 "Inter","Microsoft YaHei",
  "PingFang SC","Noto Sans CJK SC",sans-serif; }
a { color:var(--teal); text-underline-offset:.23em; }
a:hover { color:#09413b; }
h1,h2,h3,p { margin-top:0; }
.wrap { max-width:1216px; margin:0 auto; padding:0 40px; }
header { background:#172f3c; color:#fffdf6; border-top:5px solid #4bac96; padding:34px 0 38px; }
.brand { font:600 14px/1 var(--mono); color:#a9d8cd; letter-spacing:.12em; }
h1 { font-size:clamp(30px,5.4vw,57px); line-height:1.2; letter-spacing:-.055em; margin:16px 0; }
.meta { color:#c7d8da; font-size:13px; margin:0; }
nav { border-bottom:1px solid var(--rule); }
nav .wrap { display:flex; flex-wrap:wrap; gap:5px 24px; padding-top:11px; padding-bottom:11px; }
nav a { font-size:13px; font-weight:700; text-decoration:none; }
main { padding-bottom:55px !important; }
section { padding:29px 0; border-bottom:1px solid var(--rule); scroll-margin-top:18px; }
h2 { font-size:25px; line-height:1.35; margin:0 0 16px; letter-spacing:-.035em; }
h3 { font-size:17px; line-height:1.4; margin:24px 0 10px; }
.summary { border-left:4px solid var(--teal); padding:3px 0 3px 20px; max-width:1030px; }
.summary p { margin:0 0 12px; }
.summary p:last-child { margin:0; }
p.note,figcaption { color:var(--muted); font-size:13px; }
sup { padding-left:.15em; font-size:.71em; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
.chart { margin:0; padding:11px; background:var(--sheet); border:1px solid var(--rule); border-radius:7px; }
img { display:block; width:100%; height:auto; }
figcaption { padding:6px 8px 1px; }
.table-wrap { overflow:auto; border:1px solid var(--rule); border-radius:7px; margin:9px 0 12px; }
table { border-collapse:collapse; width:100%; background:var(--sheet); font-variant-numeric:tabular-nums;
  font-size:13px; line-height:1.5; }
caption { text-align:left; font-size:14px; font-weight:700; padding:10px 13px; background:#ecf0ed; }
td,th { padding:8px 10px; text-align:right; white-space:nowrap; }
thead th { background:#f3f5f1; color:#435864; border-bottom:1px solid var(--rule); font-weight:650; }
th:first-child,td:first-child { text-align:left; }
tbody th { font-weight:550; }
tbody tr+tr { border-top:1px solid #ebeeeb; }
tbody tr.focus { background:#e5f2ed; font-weight:680; }
tbody tr.benchmark { background:#f4f1e8; }
details { border:1px solid var(--rule); border-radius:7px; background:var(--sheet); padding:10px 14px; margin:12px 0; }
summary { cursor:pointer; font-weight:700; }
details[open] summary { margin-bottom:12px; }
dl { display:grid; grid-template-columns:105px minmax(0,1fr); gap:5px 16px; margin:0 0 14px; }
dt { font-weight:700; } dd { margin:0; }
code { font:12px/1.6 var(--mono); padding:2px 4px; background:#eaf1ed; overflow-wrap:anywhere; }
.sources { padding-left:23px; }
.sources li { margin-bottom:7px; word-break:break-word; }
footer { padding:14px 0; color:var(--muted); font-size:12px; }
@media(max-width:690px) { .wrap { padding-left:17px; padding-right:17px; } header { padding:25px 0; }
  h2 { font-size:21px; } .grid { grid-template-columns:minmax(0,1fr); } nav .wrap { gap:3px 13px; }
  .summary { padding-left:13px; } }
@media print { body { background:white; } .wrap { padding:0 9mm; } nav { display:none; }
  header { background:white; color:var(--ink); padding:12px 0; border-top:3px solid var(--teal); }
  .brand,.meta { color:var(--muted); } section { padding:16px 0; }
  .chart,tr { break-inside:avoid; } .table-wrap { overflow:visible; } h2,h3 { break-after:avoid; }
  th,td { padding:4px 5px; } table { font-size:9px; } }
"""


def percent(value, digits=2):
    return f"{value:.{digits}%}" if np.isfinite(value) else "—"


def number(value, digits=3):
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def table(caption, headings, rows, focus=None, benchmark=None):
    head = "".join(f'<th scope="col">{escape(str(x))}</th>' for x in headings)
    body = []
    for row in rows:
        name, *values = row
        mark = ' class="focus"' if name == focus else ' class="benchmark"' if name == benchmark else ""
        cells = "".join(f"<td>{escape(str(item))}</td>" for item in values)
        body.append(f'<tr{mark}><th scope="row">{escape(str(name))}</th>{cells}</tr>')
    return (
        '<div class="table-wrap"><table>'
        f"<caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def figure(file, alt, caption, link=True):
    image = f'<img src="{file}" alt="{escape(alt)}">'
    if link:
        image = f'<a href="{file}">{image}</a>'
    return f'<figure class="chart">{image}<figcaption>{caption}</figcaption></figure>'


def fonts():
    names = {entry.name for entry in fontManager.ttflist}
    common = ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "WenQuanYi Micro Hei", "SimHei")
    family = next((name for name in common if name in names), None)
    if family is None and Path("C:/Windows/Fonts/msyh.ttc").exists():
        fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
        family = fontManager.ttflist[-1].name
    if family is None:
        raise RuntimeError("生成中文收益图需要可用的中文字体")
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.unicode_minus": False, "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    })


def returns_plot(folder, paths, keys, filename, title, subtitle, labels=STRATEGIES, period="test"):
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(11.8, 6.1), sharex=True, gridspec_kw={"height_ratios": [2.05, 1], "hspace": .18},
    )
    fig.subplots_adjust(left=.095, right=.965, bottom=.22 if len(keys) > 1 else .155, top=.78)
    fig.text(.095, .932, title, fontsize=21, fontweight="bold")
    fig.text(.095, .872, subtitle, fontsize=10, color=MUTED)
    for axis in (upper, lower):
        axis.grid(axis="y", color=GRID, linewidth=.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="both", which="both", length=0, pad=7)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.axhline(0, color="#9aaea8", lw=.75, zorder=1)
    for code, color in zip(keys, PALETTE):
        frame = paths[(period, code)]
        base = frame.index[0] - pd.Timedelta(days=1)
        dates = frame.index.insert(0, base)
        cumulative = np.r_[0, frame["nav"].to_numpy() - 1]
        dd = np.r_[0, frame["drawdown"].to_numpy()]
        width = 2.3 if code == "chrono" or len(keys) == 1 else 1.45
        upper.plot(dates, cumulative, label=labels[code], color=color, lw=width)
        upper.scatter([dates[-1]], [cumulative[-1]], s=15, color=color, zorder=5)
        lower.plot(dates, dd, color=color, lw=width * .76)
        if len(keys) == 1:
            lower.fill_between(dates, dd, 0, color=color, alpha=.14)
    upper.set_ylabel("累计净收益")
    lower.set_ylabel("回撤")
    lower.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    lower.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    if len(keys) > 1:
        fig.legend(loc="lower left", bbox_to_anchor=(.095, .071), frameon=False, ncols=3, borderaxespad=0)
    fig.text(.095, .033, "数据：腾讯财经后复权日线  |  起始净值 1  |  单边成本 5 bp", fontsize=8, color=MUTED)
    fig.savefig(folder / filename, dpi=170)
    plt.close(fig)


def year_plot(folder, annual):
    years = sorted(annual["period"].unique())
    data = annual.pivot(index="strategy", columns="period", values="total_return").reindex(STRATEGIES)
    fig, axis = plt.subplots(figsize=(9.6, 5.7))
    fig.subplots_adjust(left=.245, right=.965, top=.83, bottom=.12)
    fig.text(.06, .925, "逐年净收益", fontsize=20, fontweight="bold")
    limit = max(.15, np.abs(data.to_numpy()).max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("returns", ["#c67a64", PAPER, "#5aada1"])
    image = axis.imshow(data, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(years)), [f"{year}年" for year in years])
    axis.set_yticks(np.arange(len(data)), [STRATEGIES[key] for key in data.index])
    axis.tick_params(length=0, pad=8)
    axis.spines[:].set_visible(False)
    for i, row in enumerate(data.to_numpy()):
        for j, value in enumerate(row):
            axis.text(j, i, percent(value, 1), ha="center", va="center", fontsize=10, color=INK)
    colorbar = fig.colorbar(image, ax=axis, pad=.04, fraction=.047, format=PercentFormatter(1, decimals=0))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    fig.text(.06, .03, f"{years[-1]} 年为截至样本末日的期间收益。数据：腾讯财经后复权日线。", fontsize=8, color=MUTED)
    fig.savefig(folder / "annual_returns.png", dpi=170)
    plt.close(fig)


def calibration_plot(folder, risk):
    actual, predicted = matched_forecasts(risk, "test")
    fig, axis = plt.subplots(figsize=(9.6, 5.55))
    fig.subplots_adjust(left=.14, right=.945, top=.80, bottom=.18)
    fig.text(.12, .92, "五日方差预测校准", fontsize=20, fontweight="bold")
    fig.text(.12, .86, "每条曲线按各自的预测值等频分成十组", fontsize=10, color=MUTED)
    limits = []
    for key, color in (("ewma", ORANGE), ("chrono", TEAL)):
        frame = pd.DataFrame({"p": predicted[key].stack(), "a": actual.stack()})
        bins = pd.qcut(frame["p"], 10, duplicates="drop")
        grouped = frame.groupby(bins, observed=True).mean()
        display = np.sqrt(grouped * PER_YEAR)
        axis.plot(display["p"], display["a"], "-o", lw=1.6, markersize=4, color=color, label=RISK_NAMES[key])
        limits.extend(display.to_numpy().ravel())
    lo, hi = min(limits) * .9, max(limits) * 1.1
    axis.plot([lo, hi], [lo, hi], "--", lw=1, color=GRAY, label="完全校准")
    axis.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="预测年化波动率", ylabel="实现年化波动率")
    ticks = [value for value in (.01, .02, .04, .08, .16, .32, .64, 1.28) if lo <= value <= hi]
    axis.set_xticks(ticks, [percent(x, 0) for x in ticks])
    axis.set_yticks(ticks, [percent(x, 0) for x in ticks])
    axis.minorticks_off()
    axis.grid(color=GRID, linewidth=.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    fig.text(.12, .045, "共同样本外标签。先按组平均日方差，再以 252 个交易日换算成年化波动率。", fontsize=8, color=MUTED)
    fig.savefig(folder / "risk_calibration.png", dpi=170)
    plt.close(fig)


def event_turnover(paths, risk):
    signal = paths[("test", "chrono")]
    traded = signal.loc[signal["scheduled"]]
    anchors = pd.DatetimeIndex(traded["signal_date"])
    counts = risk.shock_counts.loc[anchors].to_numpy()
    targets = traded[[f"target_{code}" for code in CODES]].to_numpy()
    max_count = np.where(targets > 0, counts, 0).max(axis=1)
    rows = []
    single = None
    for label, mask in (("无冲击", max_count == 0), ("单次冲击", max_count == 1), ("重复冲击", max_count >= 2)):
        dates = traded.index[mask]
        values = [paths[("test", code)].loc[dates, "turnover"].mean() for code in ("mom_ewma", "mom_fixed", "chrono")]
        rows.append([label, len(dates), *(percent(value) for value in values)])
        if label == "单次冲击":
            single = values
    result = table("周度调仓日的平均双边换手", ["调仓前五日冲击", "次数", "EWMA", "固定写入", "ChronoETF"], rows)
    return (
        f"{result}<p>单次冲击组的周均双边换手为 ChronoETF {percent(single[2])}、EWMA {percent(single[0])}。"
        f"两者差值为 {(single[2] - single[0]) * 100:+.2f} 个百分点。</p>"
    )


def performance_table(frame, title, benchmark):
    rows = [
        (STRATEGIES[code], percent(row.total_return), percent(row.cagr), percent(row.annual_vol),
         percent(row.max_drawdown), number(row.sharpe, 2), number(row.sortino, 2), number(row.calmar, 2),
         f"{row.annual_turnover_two_way:.2f}×", percent(row.fee_over_period_start_nav), int(row.drawdown_days))
        for code, row in frame.iterrows()
    ]
    return table(
        title, ["策略", "累计", "年化", "年化波动", "最大回撤", "夏普", "Sortino", "Calmar", "年化换手", "累计费用¹", "水下最长天数"],
        rows, focus=STRATEGIES["chrono"], benchmark=STRATEGIES[benchmark],
    )


def audit_section(all_metrics, main_test, main_risk, risk_metrics, audit):
    if audit is None:
        return "<p>本次运行使用离线模式。表内主结果以腾讯后复权行情计算。</p>"
    differences, actions = audit
    base = all_metrics.query(
        "price_basis == 'action_audit' and period == 'test' and one_way_bps == @BASE_COST",
    ).set_index("strategy").reindex(STRATEGIES)
    audit_risk = risk_metrics.query("price_basis == 'action_audit' and period == 'test'").set_index("model")
    spot = actions.loc[actions["event"].eq("no_action")].iloc[0]
    top = base["sharpe"].idxmax()
    best = main_test["sharpe"].idxmax()
    rank_text = (
        f"两种价格口径的夏普首位均为 {STRATEGIES[best]}"
        if top == best else f"后复权夏普首位为 {STRATEGIES[best]}，公告复核首位为 {STRATEGIES[top]}"
    )
    rank_corr = main_test["sharpe"].rank().corr(base["sharpe"].rank())
    pt = base.at["chrono", "sharpe"] - base.at["mom_ewma", "sharpe"]
    variant = base.at["chrono", "sharpe"]
    overview = table(
        "样本外策略重跑 · 单边 5 bp",
        ["策略", "后复权累计", "公告复核累计", "后复权夏普", "公告复核夏普", "后复权回撤", "公告复核回撤"],
        [
            (STRATEGIES[code], percent(main_test.at[code, "total_return"]), percent(row.total_return),
             number(main_test.at[code, "sharpe"]), number(row.sharpe),
             percent(main_test.at[code, "max_drawdown"]), percent(row.max_drawdown))
            for code, row in base.iterrows()
        ], focus=STRATEGIES["chrono"],
    )
    deviations = table(
        "2015–2026 收益偏差 · 绝对值",
        ["ETF", "中位数（bp）", "95 分位（bp）", "最大（bp）", "最大偏差日"],
        [(r.symbol, f"{r.median_bps:.2f}", f"{r.p95_bps:.2f}", f"{r.max_bps:.2f}", r.max_date)
         for r in differences.itertuples()],
    )
    sample = table(
        "公告窗口及抽查日 · 对上一有行情日",
        ["ETF", "生效日期", "事件", "腾讯后复权", "公告复核"],
        [(r.symbol, r.date, {"split": "份额变更", "dividend": "现金分红", "no_action": "普通交易日"}[r.event],
          percent(r.hfq_return, 4), percent(r.audit_return, 4)) for r in actions.itertuples()],
    )
    return f"""
      <p>{spot.symbol} 在 {spot.date} 的两套收盘收益分别为 {percent(spot.hfq_return)} 和 {percent(spot.audit_return)}。
      {rank_text}，两套排序的 Spearman 相关系数为
      {rank_corr:.2f}。ChronoETF 的夏普为 {variant:.3f}，相对 EWMA 的差为 {pt:+.3f}。</p>
      <p>风险预测方面，ChronoETF / EWMA 的 QLIKE 从
      {main_risk.at["chrono", "qlike"]:.4f} / {main_risk.at["ewma", "qlike"]:.4f}
      变为 {audit_risk.at["chrono", "qlike"]:.4f} / {audit_risk.at["ewma", "qlike"]:.4f}。</p>
      {overview}
      <details><summary>逐标的偏差与公告窗口</summary>
      {deviations}{sample}</details>
      <p class="note">复核时逐日使用不复权报价，现金红利按除息日开盘再投入，按公告比例处理份额变更，
      日内保留不复权收盘与开盘之比。以同一参数和交易规则重训、重跑；原始价格仅在内存中处理。该序列对应税前再投资假设。<sup><a href="#source-2">[2]</a></sup></p>
    """


def write_report(folder, now, market, hashes, all_metrics, paths, risk, risk_metrics, events,
                 intervals, best_baseline, audit):
    fonts()
    hfq = all_metrics.query("price_basis == 'hfq' and one_way_bps == @BASE_COST")
    test = hfq.loc[hfq["period"].eq("test")].set_index("strategy").reindex(STRATEGIES)
    validation = hfq.loc[hfq["period"].eq("validation")].set_index("strategy").reindex(STRATEGIES)
    annual = hfq.loc[hfq["period"].isin([str(year) for year in range(TEST_YEAR, market.close.index[-1].year + 1)])]
    risk_hfq = risk_metrics.query("price_basis == 'hfq'")
    risk_test = risk_hfq.loc[risk_hfq["period"].eq("test")].set_index("model").reindex(RISK_NAMES)
    risk_val = risk_hfq.loc[risk_hfq["period"].eq("validation")].set_index("model").reindex(RISK_NAMES)
    n = int(test.iloc[0]["trading_days"])
    cutoff = test.iloc[0].last_date
    sample = f'{test.iloc[0].first_date} 至 {cutoff}'
    compared = tuple(dict.fromkeys(("chrono", "mom_ewma", best_baseline, "equal_5", "hold_300")))
    returns_plot(folder, paths, compared, "returns_comparison.png", "样本外净收益与回撤",
                 f"{sample}  |  {n} 个交易日")
    year_plot(folder, annual)
    calibration_plot(folder, risk)
    gallery = []
    for code, name in STRATEGIES.items():
        r = test.loc[code]
        filename = f"returns_{code}.png"
        subtitle = (f"{sample}  |  净收益 {percent(r.total_return)}"
                    f"  ·  夏普 {r.sharpe:.2f}  ·  最大回撤 {percent(r.max_drawdown)}")
        returns_plot(folder, paths, (code,), filename, name, subtitle)
        gallery.append(figure(filename, f"{name}的净收益和回撤", f"{name} · 点击查看独立图片"))

    chrono = test.loc["chrono"]
    ewma = test.loc["mom_ewma"]
    leader = test["sharpe"].idxmax()
    leader_path = paths[("test", leader)]
    avg_bond = leader_path["w_511010"].mean()
    risk_diff = 1 - risk_test.at["chrono", "qlike"] / risk_test.at["ewma", "qlike"]
    paired = intervals.query("measure == 'sharpe' and reference == 'mom_ewma' and block == 20").iloc[0]
    risk_ci = intervals.query("measure == 'qlike' and block == 20").iloc[0]
    ci_text = "区间横跨 0" if paired.lower <= 0 <= paired.upper else "区间全部为正" if paired.lower > 0 else "区间全部为负"
    risk_ci_text = (
        "区间横跨 0" if risk_ci.lower <= 0 <= risk_ci.upper
        else "区间全部为负" if risk_ci.upper < 0 else "区间全部为正"
    )
    loss_phrase = f"下降 {percent(risk_diff, 1)}" if risk_diff >= 0 else f"上升 {percent(-risk_diff, 1)}"
    model_performance = performance_table(test, "最终样本外 · 后复权 · 单边 5 bp", best_baseline)
    validation_performance = performance_table(validation, "2021–2023 向前验证 · 后复权 · 单边 5 bp", best_baseline)

    annual_table = table(
        "年度表现 · 样本外",
        ["阶段 / 策略", "交易日", "区间收益", "年化", "最大回撤", "夏普"],
        [(f"{r.period} · {STRATEGIES[r.strategy]}", r.trading_days, percent(r.total_return), percent(r.cagr),
          percent(r.max_drawdown), number(r.sharpe, 2))
         for r in annual.loc[annual["strategy"].isin(dict.fromkeys(("chrono", "mom_ewma", best_baseline)))]
         .sort_values("period", kind="stable").itertuples()],
    )
    exposure = table(
        "平均收盘权重",
        ["策略", *ASSETS, "现金"],
        [
            (name, *[percent(x, 1) for x in paths[("test", key)][
                [f"w_{code}" for code in CODES] + ["cash_weight"]
            ].mean()])
            for key, name in STRATEGIES.items()
        ], focus=STRATEGIES["chrono"], benchmark=STRATEGIES[best_baseline],
    )
    forecast_table = table(
        "五日方差预测 · 所有方法使用同一批资产日",
        ["方法", "验证 QLIKE", "样本外 QLIKE", "样本外 MAE × 10⁴", "预测均值 / 实现均值", "样本外资产日"],
        [
            (RISK_NAMES[key], number(risk_val.at[key, "qlike"], 4), number(row.qlike, 4),
             number(row.mae_variance * 10000, 4), number(row.predicted_over_realized, 3), f"{row.observations:,}")
            for key, row in risk_test.iterrows()
        ], focus="ChronoETF",
    )
    event_table = table(
        "冲击分组 · 样本外五日方差预测",
        ["近五日触发", "资产日", "EWMA QLIKE", "固定写入 QLIKE", "ChronoETF QLIKE", "短期门均值", "长期门均值"],
        [(r.event, f"{r.observations:,}", number(r.ewma, 4), number(r.fixed, 4), number(r.chrono, 4),
          percent(r.short_gate, 1), percent(r.long_gate, 1)) for r in events.itertuples()],
    )
    repeat = events.loc[events["event"].eq("重复冲击")].iloc[0]
    cost = all_metrics.query("price_basis == 'hfq' and period == 'test'")
    cost_return = cost.pivot(index="strategy", columns="one_way_bps", values="total_return").reindex(STRATEGIES)
    cost_sharpe = cost.pivot(index="strategy", columns="one_way_bps", values="sharpe").reindex(STRATEGIES)
    costs = table(
        "成本敏感性 · 各单边费率下的累计净收益与夏普",
        ["策略", *(f"{bps} bp · 收益 / 夏普" for bps in COSTS)],
        [(STRATEGIES[key], *[f"{percent(row[bps])} / {cost_sharpe.at[key, bps]:.2f}" for bps in COSTS])
         for key, row in cost_return.iterrows()],
        focus=STRATEGIES["chrono"], benchmark=STRATEGIES[best_baseline],
    )
    inference = table(
        "ChronoETF 配对差 · 95% 区块自助区间",
        ["指标", "对照", "区块长度（日）", "差值", "95% 区间"],
        [
            ("净夏普" if r.measure == "sharpe" else "日均 QLIKE",
             STRATEGIES[r.reference] if r.measure == "sharpe" else RISK_NAMES[r.reference], r.block,
             f"{r.difference:+.4f}", f"[{r.lower:+.4f}, {r.upper:+.4f}]") for r in intervals.itertuples()
        ],
    )
    fold_table = table(
        "扩展窗口 · 每年仅使用之前已结算的标签",
        ["年份", "最晚训练信号", "最晚训练标签", "该折预测首日", "资产日", "冲击阈值", "固定短门", "固定长门"],
        [(r.year, r.train_signal_max, r.train_label_max, r.oos_first, f"{r.training_asset_days:,}",
          number(r.shock_threshold, 3), percent(r.fixed_short_gate, 1), percent(r.fixed_long_gate, 1))
         for r in risk.folds.itertuples()],
    )
    asset_counts = table(
        "数据覆盖",
        ["ETF", "代码", "有效交易日", "停牌日"],
        [(name, code, f"{market.close[code].count():,}", f"{market.close[code].isna().sum():,}")
         for code, name in zip(CODES, ASSETS)],
    )
    hashes_html = "".join(f"<li><code>{escape(name)} {digest}</code></li>" for name, digest in hashes.items())
    audit_command = " --skip-price-audit" if audit is None else ""
    boot_caption = (
        f"每次重复在配对后的同一组交易日序列中同步抽样。以 {BOOTSTRAP_REPS:,} 次循环区块重抽样、"
        f"种子 {BOOTSTRAP_SEED} 得到百分位区间。方差损失先在日内按 ETF 求均值，保留横截面相关和五日标签重叠。"
    )
    phase_1_end = risk.folds.iloc[0].train_label_max
    validation_from = validation.iloc[0].first_date
    validation_through = validation.iloc[0].last_date
    verdict = (
        "配对结果支持 ChronoETF 相对 EWMA 的夏普改善。"
        if paired.lower > 0 else "这一轮组合实验没有建立 ChronoETF 相对 EWMA 的夏普优势。"
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · 双记忆风险预测与 ETF 轮动</title><style>{CSS}</style></head>
<body>
<header><div class="wrap">
<div class="brand">CHRONOETF / QUANTITATIVE RESEARCH</div>
<h1>双记忆风险预测与 ETF 轮动</h1>
<p class="meta">固定五只境内 ETF　·　样本外 {sample}　·　{now:%Y-%m-%d %H:%M:%S} 北京时间</p>
</div></header>
<nav aria-label="研报目录"><div class="wrap">
<a href="#summary">摘要</a><a href="#returns">组合回测</a><a href="#risk">预测与消融</a>
<a href="#robust">稳健性</a><a href="#methods">方法</a><a href="#sources">证据与复现</a>
</div></nav>
<main class="wrap">
<section id="summary">
<h2>摘要</h2>
<div class="summary">
<p><strong>ChronoETF 的五日方差预测 QLIKE 较 EWMA {loss_phrase}。</strong>
样本外损失为 {risk_test.at["chrono", "qlike"]:.4f}，EWMA 为 {risk_test.at["ewma", "qlike"]:.4f}。
按日配对的损失差 95% 区间为 [{risk_ci.lower:+.3f}, {risk_ci.upper:+.3f}]，{risk_ci_text}。<sup><a href="#source-1">[1]</a></sup></p>
<p><strong>ChronoETF 成本后累计收益 {percent(chrono.total_return)}，净夏普 {chrono.sharpe:.2f}，最大回撤 {percent(chrono.max_drawdown)}。</strong>
同一动量组合接 EWMA 的三项指标依次为 {percent(ewma.total_return)}、{ewma.sharpe:.2f}、{percent(ewma.max_drawdown)}。
配对夏普差为 {paired.difference:+.3f}，95% 区间为 [{paired.lower:+.3f}, {paired.upper:+.3f}]。{verdict}</p>
<p>六条传统策略中，验证期夏普最高的是 {STRATEGIES[best_baseline]}。最终样本外夏普最高的是 {STRATEGIES[leader]}，
夏普 {test.at[leader, "sharpe"]:.2f}，其平均 5 年国债 ETF 仓位为 {percent(avg_bond, 1)}。比较时同时查看资产敞口和名义风险预算。</p>
</div></section>

<section id="returns"><h2>组合回测</h2>
{figure("returns_comparison.png", "ChronoETF、EWMA 与主要传统基线的累计净收益和回撤", "图 1　样本外连续净值。累计收益由每日扣费后净值计算。<sup><a href='#source-1'>[1]</a></sup>")}
{model_performance}
<p class="note">¹ 累计费用为样本区间支付的费用总额除以期初净值。双边换手计入买入与卖出。2026 年数据截至 {cutoff}。</p>
<details><summary>2021–2023 验证期的全部策略</summary>{validation_performance}</details>
{figure("annual_returns.png", "十条策略在各样本外年份的累计净收益", "图 2　历年收益使用连续仓位。年度区间以本年前一日的净值为基准。")}
<details><summary>年度指标和平均资产敞口</summary>{annual_table}{exposure}</details>
<details><summary>十条策略的独立收益图</summary><div class="grid">{''.join(gallery)}</div></details>
</section>

<section id="risk"><h2>预测与模型消融</h2>
<p>同一批量价特征、同一动量篮子和配置规则贯穿各模型。单记忆与双记忆固定写入比较第二条时间尺度的增量。
固定写入的强度取训练期自适应门控的均值，再与事件触发的写入比较。自然日版本只替换衰减时钟。</p>
{forecast_table}
{figure("risk_calibration.png", "EWMA 与 ChronoETF 预测年化波动率的校准图", "图 3　样本外预测值与实现值逐组对照。QLIKE 按五日平均日方差计算。<sup><a href='#source-3'>[3]</a></sup>")}
{event_table}
<p>在 {repeat.observations:,} 个重复冲击资产日上，ChronoETF 的 QLIKE 为 {repeat.chrono:.4f}，
EWMA 为 {repeat.ewma:.4f}，双记忆固定写入为 {repeat.fixed:.4f}。</p>
{event_turnover(paths, risk)}
<p class="note">冲击分组只看信号日及其前四个交易日。周度换手分组使用当期动量篮子里触发次数最多的标的。
分组样本相互依赖，表内差异用于诊断机制。</p>
</section>

<section id="robust"><h2>成本、统计区间与价格口径</h2>
{costs}
{inference}
<p class="note">{boot_caption}<sup><a href="#source-5">[5]</a></sup></p>
<h3>权益事件复核</h3>
{audit_section(all_metrics, test, risk_test, risk_metrics, audit)}
</section>

<section id="methods"><h2>回测与训练方法</h2>
{asset_counts}
<dl>
<dt>样本切分</dt><dd>{market.close.index[0]:%Y-%m-%d} 至 {phase_1_end} 初始训练，
{validation_from} 至 {validation_through} 为扩展窗口验证，{sample} 为最终向前评估。
每年初用历史数据重新估计读出，训练目标窗口的末日早于该年首个预测日。验证期最后五日的跨期标签从选择指标中移除。
验证期用于选择 EWMA 参数及传统基线。回测组合在验证期与最终评估期各从现金建仓，此后组合在各年之间连续运行。</dd>
<dt>价格与成交</dt><dd>后复权价用于权重法计价。周末前最后一个交易日收盘生成目标，下周第一个交易日开盘模拟调仓。
开盘前的隔夜涨跌归旧仓位，先卖后买，计费后以可用现金限制买入。
停牌持仓按最后收盘价计价。如有待执行的目标调整，在复牌开盘按最近有效的目标重新配平。
所有组合只做多，仓位不超过 100%，现金收益和夏普无风险利率均为 0。
每边 5 bp 由 3 bp 佣金和 2 bp 滑点组成，另测 0/10/25 bp。</dd>
<dt>传统组合</dt><dd>沪深 300 ETF 买入持有。五 ETF 每周等权。股债组合配置 60% 沪深 300 ETF、40% 5 年国债 ETF。
逆波动策略按过去 60 日样本波动率的倒数分配权重。动量按 60 日价格涨幅选前两只并等权。
有风控的组合共享这两只标的，以 60 日滚动相关系数连接各自的方差预测，将名义持仓缩放至 {VOL_TARGET:.0%} 年化波动目标，
仓位缩放上限为 1。</dd>
<dt>预测与门控</dt><dd>目标为未来五个交易日对数收益平方的均值。停牌造成的非连续收益和样本末尾缺少未来五日的信号不参与评分。
输入为 20 日与 60 日方差之比、当日收益平方相对 60 日方差、成交量相对过去 20 日中位数及 ETF 固定项。
EWMA 在验证期的衰减系数为 {risk.ewma_lambda:.2f}（候选值 {", ".join(f"{x:.2f}" for x in risk.ewma_validation)}）。
冲击用当日绝对对数收益除以前一日估计的 60 日波动率，阈值取训练期 {EVENT_QUANTILE:.0%} 分位。
每个状态先按半衰期衰减，再向当日收益冲击写入，短期半衰期 {SHORT_HALF_LIFE} 日、长期 {LONG_HALF_LIFE} 日。
单次冲击增大短期写入，量能确认与五日内重复冲击支持长期写入。读出按共享 QLIKE 目标拟合对数方差，
岭惩罚 {RIDGE:.2f}，输出在同一单位下比较。自然日消融以自然日间隔 × 5/7 换算衰减步长。<sup><a href="#source-4">[4]</a></sup></dd>
<dt>指标定义</dt><dd>年化收益为期末净值因子按 252 个交易日折算的几何收益。
夏普为日净收益均值除以日收益样本标准差，再乘 √252。Sortino 以零为目标，用全样本的下行均方根作分母。
最大回撤按净值相对历史最高值的跌幅计算，水下天数连续计数。Calmar 为年化收益除以最大回撤绝对值。
QLIKE = 实现方差 ÷ 预测方差 − log(实现方差 ÷ 预测方差) − 1。</dd>
</dl>
<details><summary>各年度训练边界与写入强度</summary>{fold_table}</details>
<h3>解释边界</h3>
<p>本研究覆盖事后选定的五只 ETF，样本外观察支持此资产池和所列时期的结论。
权重回测按后复权开盘价模拟并使用比例成本。实际账户的逐笔成交还需考虑最小申报单位、最低佣金和资金规模。
逆波动权重容易向低波动资产集中，夏普的差异同时反映资产敞口、仓位风险与择时。风险预测改善与组合收益改善分别取证。</p>
</section>

<section id="sources"><h2>证据与复现</h2>
<ol class="sources">
<li id="source-1">腾讯财经 ETF 后复权日线与本地交易日历。可下载<a href="portfolio_metrics.csv">组合指标全表</a>、
<a href="daily_backtest.csv">每日组合净值、收益与持仓</a>和<a href="risk_forecasts.csv">逐 ETF 风险预测及标签</a>。
每日两张明细表对应主价格口径，组合指标表包含所有成本与价格口径。</li>
<li id="source-2">腾讯财经不复权行情与基金公告。样本包含
<a href="https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf">510500 2015 年折算结果</a>和
<a href="https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-08-23/510500_20220823_1_aMMa3ZNB.pdf">2022 年份额拆分安排</a>。</li>
<li id="source-3">Patton (2011)，<a href="https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf">Volatility Forecast Comparison Using Imperfect Volatility Proxies</a>。</li>
<li id="source-4"><a href="https://www.msci.com/www/research-report/return-to-riskmetrics-the/019088036">RiskMetrics 方法回顾</a>，EWMA 风险预测参考。</li>
<li id="source-5">Künsch (1989)，<a href="https://doi.org/10.1214/aos/1176347265">The Jackknife and the Bootstrap for General Stationary Observations</a>。</li>
</ol>
<p>从项目根目录运行 <code>python -B -X utf8 chronoetf/backtest.py{audit_command}</code>。
环境 Python {platform.python_version()} · NumPy {np.__version__} · pandas {pd.__version__} ·
SciPy {scipy.__version__} · Matplotlib {matplotlib.__version__}。</p>
<details><summary>五份输入文件的 SHA-256</summary><ul class="sources">{hashes_html}</ul></details>
</section>
<footer>ChronoETF 量化研究　·　运行编号 {escape(folder.name)}</footer>
</main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def volatility_plot(folder, paths, labels):
    from volatility_policy import rolling_volatility

    fig, ax = plt.subplots(figsize=(11.8, 4.7))
    fig.subplots_adjust(left=.10, right=.965, top=.76, bottom=.18)
    fig.text(.10, .91, "缩仓与动态控波的实测差异", fontsize=21, fontweight="bold")
    fig.text(.10, .85, "20 日滚动年化波动 · 开盘执行 · 单边 5 bp", color=MUTED, fontsize=10)
    arms = ("budget_60", "matched_75", "matched_ewma_75", "full_75")
    for name, color in zip(arms, (GRAY, ORANGE, BLUE, TEAL)):
        value = rolling_volatility(paths[("replay", name)])
        ax.plot(value.index, value, label=labels[name], color=color, linewidth=1.75 if name == "full_75" else 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=4, frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(folder / "volatility_comparison.png", dpi=155)
    plt.close(fig)


def write_volatility_report(folder, now, metrics, paths, forecasts, tests, goals, fresh, hashes):
    from tmave import FEES, MIN_FRESH_SESSIONS
    from volatility_policy import LABELS, MATCHED_DYNAMIC, MODEL_NAMES, PRIMARY

    focus, moderate, fixed = "full_75", "full_90", "budget_60"
    replay = metrics.query("phase=='replay'").set_index(["year", "one_way_bps", "strategy"]).sort_index()
    full = replay.loc[("all", BASE_COST)].loc[list(LABELS)]
    base, model = full.loc[fixed], full.loc[focus]
    scope = f"{base.first_date} 至 {base.last_date}"
    outcome = tests.query("phase=='replay' and block==20").set_index(["strategy", "metric", "benchmark"]).sort_index()
    hits = outcome.loc[(focus, "annual_vol")]
    v = hits.loc[fixed]
    match = hits.loc["matched_75"]
    annual_cost_pp = 100 * (base.cagr - model.cagr)
    meets = v.difference >= .15 and match.difference >= .05 and annual_cost_pp <= .50

    def lineup(title, frame):
        return table(title, ["策略", "净收益", "净年化", "年化波动", "滚动波动 P90", "净夏普", "最大回撤", "CDaR", "股 / 金 / 债"],
                     [(LABELS[key], percent(r.total_return), percent(r.cagr), percent(r.annual_vol), percent(r.vol20_p90),
                       number(r.sharpe), percent(r.max_drawdown), percent(r.cdar95),
                       " / ".join(percent(w, 2) for w in (r.mean_equity_weight, r.mean_gold_weight, r.mean_bond_weight)))
                      for key, r in frame.iterrows()], LABELS[focus], LABELS[fixed])

    def p_table(phase, selected, names, caption=None):
        frame = tests.query("phase==@phase and strategy==@selected and block==20").set_index(["metric", "benchmark"])

        def effect(metric, key):
            row = frame.loc[(metric, key)]
            return f"{100*row.difference:+.2f} [{100*row.lower:+.2f}, {100*row.upper:+.2f}]"

        return table(caption or f"{LABELS[selected]} · {phase_names[phase]}",
                     ["对照", "年化波动降幅 %", "校正 p", "尾部波动降幅 %", "净年化差 · 百分点"],
                     [(LABELS[key], effect("annual_vol", key), number(frame.at[("annual_vol", key), "p_holm"], 4),
                       effect("vol20_p90", key), effect("cagr", key)) for key in names])

    phase_names = {"development": "开发期", "replay": "历史回放", "replay_2026": "2026 截至 10 月 30 日", "independent": "独立评估"}
    other = "".join(p_table(phase, focus, PRIMARY) for phase in ("development", "replay_2026", "independent") if phase in set(tests.phase))
    fees = table("成本压力 · 年化波动 / 净年化 / 净夏普", ["策略", *(f"{cost} bp" for cost in FEES)],
                 [(LABELS[key], *[f"{percent(r.annual_vol)} / {percent(r.cagr)} / {r.sharpe:.3f}"
                                  for r in (replay.loc[("all", bps, key)] for bps in FEES)]) for key in LABELS], LABELS[focus])
    years = "".join(lineup(f"{year} · 5 bp", replay.loc[(year, BASE_COST)].loc[list(LABELS)]) for year in ("2024", "2025", "2026"))
    dev = metrics.query("phase=='development' and year=='all' and one_way_bps==@BASE_COST").set_index("strategy")
    score = table("股金合成持仓的五日方差预测", ["区间 / 模型", "同标签调仓数", "QLIKE ↓", "预测 / 实现均值", "波动排序相关"],
                  [(f"{phase_names[r.phase]} · {MODEL_NAMES[r.forecast]}", r.weeks, number(r.qlike, 4),
                    number(r.mean_forecast_over_realized), number(r.risk_rank_corr)) for r in forecasts.itertuples()])
    sig = tests.query("analysis=='forecast' and block==4")
    score_test = table("完整 T-MAVE 的五日预测损失改善", ["区间 / 对照", "共同调仓数", "QLIKE 改善及区间", "校正 p"],
                       [(f"{phase_names[r.phase]} · {MODEL_NAMES[r.benchmark]}", r.observations,
                         f"{r.difference:+.4f} [{r.lower:+.4f}, {r.upper:+.4f}]", number(r.p_holm, 5)) for r in sig.itertuples()])
    calibrations = table("开发期固定控制力度", ["档位 / 模型", "年化波动阈值", "平均股金仓位倍率"],
                         [(f"{r.level}% · {MODEL_NAMES[r.model]}", percent(r.goal_annualized), percent(r.development_weighted_scale))
                          for r in goals.itertuples()])
    independent = (f"冻结后当前有 {fresh} 个可用交易日，独立结果单列。" if "independent" in set(tests.phase)
                   else f"冻结后当前有 {fresh} 个交易日，独立评估沿用 {MIN_FRESH_SESSIONS} 日的项目口径。")
    fonts()
    volatility_plot(folder, paths, LABELS)
    returns_plot(folder, paths, (focus, "matched_75", fixed, "ewma_75", moderate), "returns_comparison.png",
                 "常态波动控制下的收益和回撤", f"{scope}  |  单边 5 bp", LABELS, "replay")
    images = []
    for key, r in full.iterrows():
        file = f"returns_{key}.png"
        returns_plot(folder, paths, (key,), file, LABELS[key],
                     f"净年化 {percent(r.cagr)}　年化波动 {percent(r.annual_vol)}　最大回撤 {percent(r.max_drawdown)}", LABELS, "replay")
        images.append(figure(file, f"{LABELS[key]}累计收益和回撤", LABELS[key]))
    outputs = {"portfolio_metrics": "绩效与仓位", "daily_backtest": "日度净值", "weekly_forecasts": "周度预测与缩仓倍率",
               "forecast_scores": "风险预测评分", "paired_tests": "全部配对检验", "calibration": "控制参数", "train_folds": "模型时间边界"}
    links = " · ".join(f'<a href="{name}.csv">{label}</a>' for name, label in outputs.items())
    source_hashes = "".join(f"<li><code>{escape(key)} {value}</code></li>" for key, value in hashes.items())
    status = "同时达到了" if meets else "尚未同时达到"
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · T-MAVE 股金目标波动</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF</div><h1>同样的平均仓位，T-MAVE 能否降低波动</h1>
<p class="meta">历史回放 {scope}　·　生成于 {now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#controls">波动及归因</a>
<a href="#returns">费后绩效</a><a href="#forecast">预测</a><a href="#method">复现</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>75% 档完整 T-MAVE 的年化波动为 {percent(model.annual_vol)}，固定 60 日为 {percent(base.annual_vol)}。
相对原主干下降 {percent(v.difference,1)}，相对同仓位静态下降 {percent(match.difference,1)}。</strong>
两项校正 p 分别为 {number(v.p_holm,4)} 和 {number(match.p_holm,4)}。</p>
<p>净年化从 {percent(base.cagr)} 到 {percent(model.cagr)}，差值为 {100*(model.cagr-base.cagr):+.2f} 个百分点。
点估计{status}原主干降波动 15%、同仓位降波动 5%、年化损失至多 0.50 个百分点三项参考量级。{independent}</p></div></section>
<section id="controls"><h2>波动下降有多少来自仓位时点</h2>
{figure('volatility_comparison.png','相同阶段 20 日滚动年化波动曲线','同仓位对照分别匹配股票、黄金平均目标敞口。')}
{p_table('replay',focus,PRIMARY)}{p_table('replay',moderate,('budget_60','matched_90','ewma_90'))}
<p class="note">波动和尾部波动的降幅为相对比例。括号为逐项 95% 区间。每期 75% 档的四项年化波动比较进行 Holm 校正。
尾部指标为已对齐的 20 日滚动年化波动序列 P90，按日期配对抽样。</p>
{p_table('replay',focus,tuple(MATCHED_DYNAMIC),'历史回放 · 同仓位动态补充对照')}
<p class="note">三个同仓位动态对照各自匹配期内平均股票、黄金目标敞口，波动检验为补充归因，校正独立于主检验。
2026 子期沿用整段回放的匹配阈值。</p>
<details><summary>开发期、2026 子期和可用独立期检验</summary>{other}</details></section>
<section id="returns"><h2>费后收益、回撤与敞口</h2>
{lineup('历史回放 · 5 bp',full)}
{figure('returns_comparison.png','核心控波策略的收益和回撤','所有策略使用同一买卖成本及成交时间。')}
<details><summary>逐年表现及开发期</summary>{years}{lineup('开发期 · 5 bp',dev)}</details>
<details><summary>交易成本 5、10 与 25 bp</summary>{fees}</details>
<details><summary>各策略的独立收益与回撤图</summary><div class="grid">{''.join(images)}</div></details></section>
<section id="forecast"><h2>预测质量与控波强度</h2>{score}{score_test}{calibrations}</section>
<section id="method"><h2>口径与复现</h2>
<dl><dt>信号</dt><dd>在周度信号日固定目标股金权重，预测下一开盘起五个开盘到开盘收益的日均二阶矩。
训练标签在首次对应年度信号前结束。</dd>
<dt>决策</dt><dd>完整 T-MAVE 合并组别快慢状态和交易日、自然日、事件时钟。波动越高，股金总目标仓位越低。
5/10/25 bp 比例费用。仓位倍率 0.35 到 1，国债目标固定，缩出的仓位留现金。</dd>
<dt>归因</dt><dd>同仓位静态权重由各阶段的事后平均目标敞口算出，用于量化择时贡献。动态模型的阈值固定在开发期。
配对区块长度 10/20/40 日，主口径 20 日，两千次抽样，种子 {BOOTSTRAP_SEED}。</dd></dl>
<p><a href="https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf">波动管理组合</a> ·
<a href="https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf">风险预测的 QLIKE</a></p>
<p>运行 <code>python -B -X utf8 chronoetf/backtest.py --vol-target</code>。</p><p>{links}</p>
<details><summary>五份输入文件的 SHA-256</summary><ul class="sources">{source_hashes}</ul></details></section>
<footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def event_window_plot(folder, windows, keys, labels):
    paired = windows.pivot(index=["signal_date", "group", "rebounded"], columns="strategy", values="net_return")
    differences = paired[list(keys)].sub(paired["budget_60"], axis=0) * 10_000
    results = differences.groupby(level="rebounded").mean().reindex([False, True])
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    fig.subplots_adjust(left=.11, right=.96, top=.75, bottom=.19)
    fig.text(.11, .90, "确认事件后五日的组合收益差", fontsize=20, fontweight="bold")
    fig.text(.11, .84, "相对固定 60 日风险预算 · 单边 5 bp · 以基点计", fontsize=10, color=MUTED)
    positions = np.arange(2)
    for k, (key, color) in enumerate(zip(keys, PALETTE)):
        ax.bar(positions + (k - (len(keys) - 1) / 2) * .21, results[key], width=.18, color=color, label=labels[key])
    ax.set_xticks(positions, ["受影响组继续走弱", "受影响组反弹"])
    ax.axhline(0, color=MUTED, lw=1)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False, ncol=len(keys), fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(folder / "event_window_effect.png", dpi=155)
    plt.close(fig)


def write_event_policy_report(folder, now, performance, paths, risk, tests, selection, selected, qualifies, episodes,
                              threshold, fresh, hashes):
    from event_policy import CANDIDATES, LABELS, MODELS, WEEKLY_CONTROLS
    from strategy import covers_objectives
    from tmave import FEES, MIN_FRESH_SESSIONS

    route = selected.split("_")[1]
    fixed, weekly = f"fixed_{route}", f"weekly_{route}"
    focus = LABELS[selected]
    replay = performance.query("phase=='replay'").set_index(["year", "one_way_bps", "strategy"]).sort_index()
    full = replay.loc[("all", BASE_COST)].loc[list(LABELS)]
    winning = full.loc[selected]
    scope = f"{winning.first_date} 至 {winning.last_date}"
    peers = ("budget_60", "budget_120", fixed)
    jointly_better = all(covers_objectives(replay.loc[("all", bps, selected)], replay.loc[("all", bps, peer)])
                         for bps in (5, 10) for peer in peers)
    replay_tests = tests.query("phase=='replay' and block==20 and analysis=='prespecified'")
    primary = replay_tests.loc[replay_tests.primary]
    significant = primary.p_holm.lt(.05) & primary.difference.gt(0.)

    def lineup(title, frame):
        return table(title, ["策略", "净收益", "净年化", "净夏普", "最大回撤", "CDaR", "年化波动", "平均现金", "双边年换手"],
                     [(LABELS[key], percent(r.total_return), percent(r.cagr), number(r.sharpe), percent(r.max_drawdown),
                       percent(r.cdar95), percent(r.annual_vol), percent(r.mean_cash_weight), f"{r.annual_turnover_two_way:.2f}×")
                      for key, r in frame.iterrows()], focus, LABELS["budget_60"])

    def p_table(phase, label):
        show = tests.query("phase==@phase and primary and block==20")
        rows = []
        for r in show.itertuples():
            risk_test = r.metric == "event_qlike"
            metric = "重复冲击 QLIKE" if risk_test else "净夏普"
            peer = (MODELS if risk_test else LABELS)[r.benchmark]
            rows.append((f"{metric} · {peer}", r.observations, f"{r.difference:+.4f}",
                         f"[{r.lower:+.4f}, {r.upper:+.4f}]", number(r.p_holm, 4)))
        return table(label, ["指标 / 对照", "观测数", "改善", "95% 区间", "Holm p"], rows)

    raw_risk = risk.query("phase=='replay'").pivot(index="forecast", columns="case", values="qlike")
    counts = risk.query("phase=='replay' and forecast=='full'").set_index("case")
    risk_table = table("共同风险标签 · 预测损失越低越好", ["模型", "全部股金资产日", "重复负冲击", "无冲击"],
                       [(label, *(number(raw_risk.at[key, case], 4) for case in ("all", "confirmed", "quiet")))
                        for key, label in MODELS.items()], MODELS["full"])
    comparison = replay_tests.set_index(["metric", "benchmark"])
    effect = table("组合净效果 · 点估计及区块区间", ["对照", "净夏普", "净年化 · 百分点", "最大回撤 · 百分点", "CDaR · 百分点"],
                   [(LABELS[key], *[f"{f * r.difference:+.3f} [{f * r.lower:+.3f}, {f * r.upper:+.3f}]"
                                     for metric, f in (("net_sharpe", 1), ("cagr", 100), ("max_drawdown", 100), ("cdar95", 100))
                                     for r in [comparison.loc[(metric, key)]]])
                    for key in ("budget_120", "budget_60", "event_rebalance", f"ewma_{route}", fixed, f"trade_{route}", weekly)])
    m = performance.query("phase=='development' and year=='all' and one_way_bps==@BASE_COST").set_index("strategy")
    choice = selection.groupby("candidate").joint_improvement.all()
    dev = table("开发期两条 T-MAVE 候选 · 5 bp", ["候选", "净年化", "净夏普", "最大回撤", "5 与 10 bp 联合改善"],
                [(LABELS[key], percent(m.at[key, "cagr"]), number(m.at[key, "sharpe"]), percent(m.at[key, "max_drawdown"]),
                  "达到" if choice[key] else "未达到") for key in CANDIDATES], focus)
    fees = table("成本压力 · 净年化 / 夏普 / 最大回撤", ["策略", *(f"{bps} bp" for bps in FEES)],
                 [(LABELS[key], *[f"{percent(r.cagr)} / {r.sharpe:.3f} / {percent(r.max_drawdown)}"
                                  for r in (replay.loc[("all", bps, key)] for bps in FEES)]) for key in LABELS], focus)
    windows = episodes.query("phase == 'replay'")
    core_windows = windows.query("strategy=='budget_60'")
    timings = core_windows.weekly_delay_sessions
    event = table("警戒起点后五日 · 扣费组合损益", ["市场路径 / 策略", "组事件数", "净收益均值", "最大回撤均值"],
                  [(f"{case} · {LABELS[key]}", len(g), percent(g.net_return.mean()), percent(g.drawdown.mean()))
                   for reboot, case in ((True, "反弹"), (False, "继续走弱"))
                   for key in ("budget_60", selected, weekly, fixed)
                   for g in [windows.loc[windows.strategy.eq(key) & windows.rebounded.eq(reboot)]]])
    years = "".join(lineup(f"{year} · 5 bp", replay.loc[(year, BASE_COST)].loc[list(LABELS)]) for year in ("2024", "2025", "2026"))
    weekly_arms = ["budget_60", "weekly_cash", *WEEKLY_CONTROLS]
    weekly_scores = lineup("周频留现金 · 同样的调仓时点 · 5 bp", full.loc[weekly_arms])
    weekly_checks = tests.query("phase=='replay' and block==20 and metric=='net_sharpe' and analysis=='weekly_diagnostic'")
    weekly_table = table("补充净夏普对照", ["策略对", "夏普差", "区块 95% 区间", "Holm p"],
                         [(f"{LABELS[r.strategy]} 对 {LABELS[r.benchmark]}", f"{r.difference:+.5f}",
                           f"[{r.lower:+.5f}, {r.upper:+.5f}]", number(r.p_holm, 4)) for r in weekly_checks.itertuples()])
    fonts()
    returns_plot(folder, paths, (selected, "budget_60", "budget_120", fixed, weekly), "returns_comparison.png",
                 "T-MAVE 事件风控与风险预算", f"{scope}  |  单边 5 bp", LABELS, "replay")
    event_window_plot(folder, windows, (selected, weekly, fixed), LABELS)
    plots = []
    for key, row in full.iterrows():
        filename = f"returns_{key}.png"
        returns_plot(folder, paths, (key,), filename, LABELS[key],
                     f"净年化 {percent(row.cagr)}   夏普 {row.sharpe:.3f}   最大回撤 {percent(row.max_drawdown)}", LABELS, "replay")
        plots.append(figure(filename, f"{LABELS[key]}净值和回撤", LABELS[key]))
    comparison_phases = {"development": "开发期", "replay_2026": "2026 截至 10 月 30 日", "independent": "冻结后独立评估"}
    other = "".join(p_table(phase, label) for phase, label in comparison_phases.items() if phase in set(tests.phase))
    files = {"portfolio_metrics": "组合指标", "daily_backtest": "逐日净值与仓位", "forecast_detail": "方差预测和快慢状态",
             "forecast_scores": "风险评分", "event_windows": "反弹和下跌窗口", "paired_tests": "配对区间及 p 值",
             "selection": "选型", "decisions": "周度与事件触发信号", "train_folds": "训练边界"}
    links = " · ".join(f'<a href="{name}.csv">{label}</a>' for name, label in files.items())
    inputs = "".join(f"<li><code>{escape(key)} {value}</code></li>" for key, value in hashes.items())
    independent = (f"冻结后有 {fresh} 个交易日，独立结果已列出。" if "independent" in set(tests.phase)
                   else f"冻结后有 {fresh} 个交易日。按现有协议，满 {MIN_FRESH_SESSIONS} 日才单列独立评估。")
    status = "达到" if qualifies else "未达到"
    replay_status = "达到" if jointly_better else "未达到"
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · 重复冲击与事件风控</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF</div><h1>重复冲击后的风险预测能否改善组合</h1>
<p class="meta">历史回放 {scope}　·　生成于 {now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#risk">风险预测</a>
<a href="#performance">组合结果</a><a href="#timing">执行时点</a><a href="#method">复现</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>{focus}取得净年化 {percent(winning.cagr)}、净夏普 {winning.sharpe:.3f}、最大回撤 {percent(winning.max_drawdown)}。</strong>
开发期{status}联合改善条件，历史回放{replay_status}。五项主检验中 {int(significant.sum())} 项在校正后达到显著提升。</p>
<p>单边 5 bp。强交易基准固定 60 日的净年化 {percent(full.at['budget_60','cagr'])}、净夏普 {full.at['budget_60','sharpe']:.3f}、
最大回撤 {percent(full.at['budget_60','max_drawdown'])}。{independent}</p></div></section>
<section id="risk"><h2>五日方差预测和配对检验</h2>{risk_table}
<p>重复负冲击样本为 {int(counts.at['confirmed','asset_days'])} 个股金资产日，覆盖 {int(counts.at['confirmed','sessions'])} 个信号日。
预测目标自下一可成交开盘起算五日，所有模型使用相同的开盘价风险标签。</p>
{p_table('replay','历史回放 · 20 日主区块')}
<p class="note">正值有利于完整 T-MAVE。各期两项事件预测与三项净夏普主检验共同作 Holm 校正。区间为逐项未校正 95% 区间。</p>
<details><summary>开发期、2026 子期及可用的独立结果</summary>{other}</details></section>
<section id="performance"><h2>组合绩效、消融与成本</h2>{dev}
{figure('returns_comparison.png','T-MAVE 与固定风险预算的累计净收益和回撤','所有曲线由同一撮合器计费。')}
{lineup('整段回放 · 5 bp',full)}{effect}
<details><summary>逐年表现</summary>{years}</details><details><summary>交易成本 5、10 与 25 bp</summary>{fees}</details>
<details><summary>各策略的独立收益和回撤图</summary><div class="grid">{''.join(plots)}</div></details></section>
<section id="timing"><h2>下一开盘行动的收益和代价</h2>
<p>可完整回放的组事件为 {len(timings)} 个，等到例行周度开盘相对事件次日开盘的中位延迟为 {timings.median():.1f} 个交易日。
同一时点的固定 60 日再平衡用于核对执行频率。窗口反弹由受冲击组五日开盘到开盘收益是否为正定义。</p>
{figure('event_window_effect.png','反弹与继续走弱时的组合收益差','各窗口相对固定 60 日的净收益差。')}{event}</section>
<section><h2>周频等频补充对照</h2>{weekly_scores}{weekly_table}
<p class="note">首轮结果后补充同频归因。保留固定机会和相同风控函数。交易日消融仍有事件门控和快慢记忆。
补充区间不参与主检验或开发选型。</p></section>
<section id="method"><h2>模型、交易规则和复现</h2>
<dl><dt>事件</dt><dd>2015–2020 年标准化绝对冲击的合并阈值是 {threshold:.3f}。近五日两次负冲击且当日为负冲击确认事件，
五个交易信号日内保留警戒。各事件版本的确认、回补与周度机会相同。</dd>
<dt>T-MAVE</dt><dd>快慢状态吸收方差冲击，重复冲击与量能证据提高写入，交易日、自然日和滞后事件量共同决定衰减。
同因子固定写入与交易日消融分别对照事件写入和多时钟。确认范围内最多减该组目标的 25%，轮换模式仅向未警戒且趋势为正的组调入。</dd>
<dt>检验</dt><dd>所有读出按年扩展训练，训练标签终点早于预测信号。净夏普用零无风险利率和 252 日年化。
配对循环区块重采样 {BOOTSTRAP_REPS:,} 次，区块长度 10/20/40 个交易日，20 日作为主结果。</dd></dl>
<p><a href="https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf">方差预测的 QLIKE 损失</a> ·
<a href="https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf">有符号冲击与波动持续性</a> ·
<a href="https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf">波动管理组合</a></p>
<p>运行 <code>python -B -X utf8 chronoetf/backtest.py --event-risk</code>。</p><p>{links}</p>
<details><summary>五份输入文件指纹</summary><ul class="sources">{inputs}</ul></details></section>
<footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def horizon_plots(folder, paths, decisions, point, names, primary, controls, scope):
    returns_plot(folder, paths, (primary, "budget_120", "budget_60", "blend_50", "fixed_memory"),
                 "returns_comparison.png", "自适应记忆与固定窗口", f"{scope}  |  单边 5 bp", names, "replay")
    data = decisions.query("phase == 'replay'").pivot(index="execution_date", columns="strategy", values="weight_60")
    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    fig.subplots_adjust(left=.09, right=.965, top=.79, bottom=.19)
    fig.text(.09, .91, "各策略给 60 日协方差的权重", fontsize=19, fontweight="bold")
    ax.axhline(.5, color=GRAY, lw=1.1, ls="--", label=names["blend_50"])
    for key, color, width in (("vol_ratio", BLUE, 1.05), ("fixed_memory", ORANGE, 1.1), (primary, TEAL, 1.8)):
        ax.step(data.index, data[key], where="post", color=color, lw=width, alpha=.87, label=names[key])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylim(-.025, 1.025)
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(-.01, -.16), ncol=4, frameon=False, fontsize=9)
    fig.savefig(folder / "horizon_weights.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    fig.subplots_adjust(left=.22, right=.965, top=.73, bottom=.135, wspace=.20)
    fig.text(.22, .907, "完整 T-MAVE 对六个非 T-MAVE 策略", fontsize=18, fontweight="bold")
    fig.text(.22, .845, "点为配对差，线为未校正 95% 区间，右侧表示 T-MAVE 更好", color=MUTED, fontsize=10)
    y = np.arange(len(controls))
    for ax, (metric, title) in zip(axes, (("covariance_qlike", "协方差损失改善"), ("net_sharpe", "净夏普改善"))):
        data = point.xs(metric, level="metric").loc[list(controls)]
        ax.axvline(0, color=GRAY, lw=1, ls="--")
        for i, row in enumerate(data.itertuples()):
            color = TEAL if row.significant_gain else ORANGE if row.significant_loss else BLUE
            ax.hlines(i, row.lower, row.upper, color=color, lw=2)
            ax.scatter(row.difference, i, color=color, s=27, zorder=3)
        ax.set_yticks(y, [names[key] for key in controls])
        ax.set_title(title, fontsize=11, pad=11)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.savefig(folder / "paired_comparison.png", dpi=165)
    plt.close(fig)


def write_horizon_report(folder, now, performance, paths, risk, decisions, tests, diagnoses, cal, fresh, hashes):
    from horizon import CONTROLS, LABELS, MAIN_METRICS, PRIMARY
    from tmave import FEES, MIN_FRESH_SESSIONS

    label = LABELS[PRIMARY]
    replay = performance.query("phase=='replay'").set_index(["year", "one_way_bps", "strategy"]).sort_index()
    full = replay.loc[("all", BASE_COST)].loc[list(LABELS)]
    risk_all = risk.query("phase=='replay' and year=='all'").set_index("strategy")
    decision = diagnoses.query("phase=='replay'").set_index("strategy")
    check = tests.query("block==20").set_index(["phase", "metric", "benchmark"]).sort_index()
    point = check.loc["replay"]
    period = f"{full.at[PRIMARY, 'first_date']} 至 {full.at[PRIMARY, 'last_date']}"
    top_risk = risk_all.loc[list(CONTROLS), "covariance_qlike"].idxmin()
    top_sharpe = full.loc[list(CONTROLS), "sharpe"].idxmax()
    match = decision.loc[PRIMARY]
    scopes = {"development": "开发期", "replay": "历史回放", "replay_2026": "2026 截至 10 月 30 日",
              "independent": "冻结后独立评估"}

    def performance_table(title, frame):
        return table(title, ["策略", "净收益", "净年化", "净夏普", "最大回撤", "CDaR 95%", "年化波动", "双边年换手"],
                     [(LABELS[key], percent(r.total_return), percent(r.cagr), number(r.sharpe),
                       percent(r.max_drawdown), percent(r.cdar95), percent(r.annual_vol), f"{r.annual_turnover_two_way:.2f}×")
                      for key, r in frame.iterrows()], label)

    def paired_table(phase):
        data = check.loc[phase]
        rows = []
        for peer in CONTROLS:
            cells = []
            for metric in MAIN_METRICS:
                row = data.loc[(metric, peer)]
                verdict = "提升" if row.significant_gain else "下降" if row.significant_loss else "未达显著"
                cells.extend((f"{row.difference:+.4f} [{row.lower:+.4f}, {row.upper:+.4f}]", number(row.p_holm, 4), verdict))
            rows.append((LABELS[peer], *cells))
        return table(scopes[phase], ["对照", "协方差损失改善及区间", "校正 p", "风险结论",
                                     "净夏普改善及区间", "校正 p", "夏普结论"], rows)

    columns = ("cagr", "max_drawdown", "cdar95")
    outcomes = table("回放期净年化与回撤改善 · 百分点", ["对照", "净年化", "最大回撤", "CDaR"],
                     [(LABELS[key], *[f"{100 * r.difference:+.2f} [{100 * r.lower:+.2f}, {100 * r.upper:+.2f}]"
                                       for r in (point.loc[(metric, key)] for metric in columns)]) for key in CONTROLS])
    cost = table("交易成本压力 · 净年化 / 夏普 / 最大回撤", ["策略", *(f"{fee} bp" for fee in FEES)],
                 [(LABELS[key], *[f"{percent(r.cagr)} / {r.sharpe:.3f} / {percent(r.max_drawdown)}"
                                  for r in (replay.loc[("all", fee, key)] for fee in FEES)]) for key in LABELS], label)
    quality = table("使用同一份未来风险标签", ["策略", "有效周数", "协方差损失 ↓", "固定组合方差损失 ↓",
                    "平均 60 日权重", "权重 P10–P90", "权重与短窗优势秩相关"],
                    [(LABELS[key], int(r.scored_weeks), number(r.covariance_qlike, 4), number(r.reference_qlike, 4),
                      percent(r.mean_weight_60, 1), f"{percent(r.p10_weight_60, 0)}–{percent(r.p90_weight_60, 0)}",
                      number(decision.at[key, "weight_gain_rank_corr"])) for key, r in risk_all.iterrows()], label)
    event = table("根据调仓前已有事件切片", ["期间", "确认 / 其他周数", "确认 / 其他的平均 60 日权重",
                  "确认 / 其他相对静态混合的协方差损失改善"],
                  [(scopes[r.phase], f"{r.confirmed_weeks} / {r.calm_weeks}",
                    f"{percent(r.weight_confirmed,1)} / {percent(r.weight_calm,1)}",
                    f"{r.gain_vs_blend_confirmed:+.4f} / {r.gain_vs_blend_calm:+.4f}")
                   for r in diagnoses.loc[diagnoses.strategy.eq(PRIMARY)].itertuples()])
    clocks = table("共用事件写入的时钟消融", ["对照", "完整 T-MAVE 的协方差损失改善及区间", "净夏普改善及区间"],
                   [(LABELS[key], *[f"{r.difference:+.4f} [{r.lower:+.4f}, {r.upper:+.4f}]"
                                     for r in (point.loc[(metric, key)] for metric in MAIN_METRICS)])
                    for key in LABELS if key not in (*CONTROLS, PRIMARY)])
    hits = [int(point.xs(metric, level="metric").significant_gain.sum()) for metric in MAIN_METRICS]
    misses = [int(point.xs(metric, level="metric").significant_loss.sum()) for metric in MAIN_METRICS]
    independent = (f"冻结后数据共 {fresh} 个交易日，独立评估已列入配对结果。" if "independent" in check.index.levels[0]
                   else f"冻结后数据共 {fresh} 个交易日，满 {MIN_FRESH_SESSIONS} 日将生成独立评估。")
    other_tests = "".join(paired_table(phase) for phase in ("development", "replay_2026", "independent")
                          if phase in check.index.levels[0])
    annual = "".join(performance_table(f"{year} · 5 bp", replay.loc[(year, BASE_COST)].loc[list(LABELS)])
                     for year in ("2024", "2025", "2026"))
    development = performance.query("phase=='development' and year=='all' and one_way_bps==@BASE_COST").set_index("strategy")
    focus = full.loc[PRIMARY]
    fonts()
    horizon_plots(folder, paths, decisions, point, LABELS, PRIMARY, CONTROLS, period)
    images = []
    for key, r in full.iterrows():
        filename = f"returns_{key}.png"
        returns_plot(folder, paths, (key,), filename, LABELS[key],
                     f"净年化 {percent(r.cagr)}   夏普 {r.sharpe:.3f}   最大回撤 {percent(r.max_drawdown)}",
                     LABELS, "replay")
        images.append(figure(filename, f"{LABELS[key]}累计净值与回撤", LABELS[key]))
    sources = "".join(f"<li><code>{escape(name)} {value}</code></li>" for name, value in hashes.items())
    evidence = {"portfolio_metrics": "组合指标", "risk_metrics": "风险评分", "weekly_decisions": "周度权重与协方差",
                "decision_diagnostics": "窗口决策诊断", "paired_tests": "配对区间与 p 值", "daily_backtest": "每日净值与仓位"}
    links = " · ".join(f'<a href="{key}.csv">{title}</a>' for key, title in evidence.items())
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · T-MAVE 自适应协方差</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF</div><h1>T-MAVE 能否选对风险记忆长度</h1>
<p class="meta">历史回放 {period}　·　生成于 {now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#inference">显著性</a>
<a href="#decision">窗口决策</a><a href="#performance">组合绩效</a><a href="#method">复现</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>完整 T-MAVE 对六个非 T-MAVE 对照，协方差损失显著改善 {hits[0]} 个，净夏普显著提高 {hits[1]} 个。</strong>
两项指标显著下降的对照数分别为 {misses[0]} 和 {misses[1]}。历史回放包含 {int(match.scored_weeks)} 个有效风险评估周。</p>
<p>单边 5 bp 时，完整 T-MAVE 净年化 {percent(focus.cagr)}、净夏普 {focus.sharpe:.3f}、最大回撤 {percent(focus.max_drawdown)}。
六个非 T-MAVE 对照中，回放的协方差损失最低为{LABELS[top_risk]}，净夏普最高为{LABELS[top_sharpe]}。</p>
<p>{independent}本轮使用已查看的历史区间。</p></div></section>
<section id="inference"><h2>风险预测与净夏普的显著性</h2>
{figure("paired_comparison.png", "协方差损失和净夏普的配对区间", "点为完整 T-MAVE 相对对照的差异。正值有利于 T-MAVE。")}
{paired_table("replay")}{outcomes}
<p class="note">每期对 6 个对照和 2 个主指标共 12 项作 Holm 校正。区间为逐项 95% 区间，p 值来自同日期的区块重采样。
年化和回撤区间为描述性结果。10/40 个交易日、2/8 个调仓周期的敏感性在配对 CSV 中。</p>
<details><summary>开发期、2026 子期与可用的独立评估</summary>{other_tests}</details></section>
<section id="decision"><h2>T-MAVE 如何决定风险窗口</h2>
{figure("horizon_weights.png", "周度 60 日协方差的混合权重", "高权重使风险估计更依赖最近 60 个交易日。")}
{quality}
<p>固定 60 日的协方差损失在 {percent(match.short_advantage_share,1)} 的回放周优于固定 120 日。
完整 T-MAVE 的短窗权重与这项优势的秩相关为 {number(match.weight_gain_rank_corr)}，
以短窗权重是否超过一半判断窗口方向时，其吻合率为 {percent(match.direction_match,1)}。</p>
{event}{clocks}<p class="note">确认事件指信号日前五个交易日（含当日）至少一组写入长期锚点。时钟消融区间用于定位贡献。</p></section>
<section id="performance"><h2>组合绩效与成本</h2>
{figure("returns_comparison.png", "策略累计净收益和回撤", "五个核心对照使用相同 ETF、周度调仓和交易成本。")}
{performance_table("历史回放 · 5 bp", full)}
<details><summary>逐年绩效与开发期</summary>{annual}{performance_table("2021–2023 开发期 · 5 bp", development)}</details>
<details><summary>5、10 与 25 bp 成本压力</summary>{cost}</details>
<details><summary>十个模型的独立收益与回撤图</summary><div class="grid">{''.join(images)}</div></details>
<p class="note">净夏普以日收益和零无风险利率计算。年化使用 {PER_YEAR} 日。CDaR 为最深 5% 日回撤的平均深度。</p></section>
<section id="method"><h2>计算口径与复现</h2>
<dl><dt>风险估计</dt><dd>三组风险预算统一使用 60 日和 120 日的收缩协方差混合。
T-MAVE 让交易日、自然日和已确认事件决定快慢状态的流逝和写入；压力的经验分位取自 {cal['first']:%Y-%m-%d}
至 {cal['last']:%Y-%m-%d} 的 {cal['signals']} 次调仓信号，分位越高，60 日权重越大。</dd>
<dt>公平标签</dt><dd>每个调仓周期在初次成交开盘与下一次成交开盘间计算三组日收益二阶矩，组内权重固定在信号日。
多元 QLIKE 使用同一组二阶矩和各自的预测矩阵。固定组合的标量 QLIKE 对 15% 股票、15% 黄金、70% 国债评分。</dd>
<dt>检验</dt><dd>配对的循环区块保留净收益和预测差的时间依赖。主口径为 20 个交易日或 4 个连续调仓周期，
{BOOTSTRAP_REPS:,} 次重采样，种子 {BOOTSTRAP_SEED}。双侧 p 值对自助分布作中心化。</dd></dl>
<p><a href="https://web.stanford.edu/~boyd/papers/pdf/cov_pred_finance.pdf">协方差预测与 QLIKE</a> ·
<a href="https://doi.org/10.1016/j.jempfin.2008.03.002">夏普差的时间序列推断</a> ·
<a href="https://doi.org/10.2307/4615733">Holm 多重检验</a></p>
<p>运行 <code>python -B -X utf8 chronoetf/backtest.py --horizon</code>。</p>
<p>{links}</p><details><summary>五份输入的 SHA-256</summary><ul class="sources">{sources}</ul></details>
</section><footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def write_alpha_report(folder, now, performance, paths, selection, selected, same, quality,
                       scenarios, candidates, intervals, fresh, hashes):
    from allocation_alpha import LABELS, TILT_LIMIT
    from strategy import covers_objectives
    from tmave import FEES, MIN_FRESH_SESSIONS

    names = LABELS
    slices = {"all": "全回放", "2024": "2024", "2025": "2025", "2026": "2026 截至 10 月 30 日"}
    merged = (performance.query("phase=='replay'").set_index(["year", "one_way_bps", "strategy"])
              .sort_index(level=["year", "one_way_bps"], sort_remaining=False))
    full = merged.loc[("all", BASE_COST)]
    last = merged.loc[("2026", BASE_COST)]
    baseline, tmave = full.loc["core"], full.loc[selected]
    lookup = ("core", "risk60", selected, same, "guard_fixed")
    scope = f"{baseline.first_date} 至 {baseline.last_date}"
    mark = names[selected]
    base_name = names["core"]

    def pp(value):
        return f"{100 * value:+.2f}"

    def lines(title, frame):
        return table(title, ["策略", "区间净收益", "净年化", "净夏普", "最大回撤", "CDaR 95%",
                             "年化波动", "最深回撤恢复日", "双边年换手", "费用 / 期初净值"],
                     [(names[key], percent(r.total_return), percent(r.cagr), number(r.sharpe),
                       percent(r.max_drawdown), percent(r.cdar95), percent(r.annual_vol),
                       int(r.recovery_sessions) if np.isfinite(r.recovery_sessions) else "期末未恢复",
                       f"{r.annual_turnover_two_way:.2f}×", percent(r.fee_over_period_start_nav))
                      for key, r in frame.iterrows()], mark, base_name)

    fonts()
    returns_plot(folder, paths, lookup, "returns_comparison.png", "T-MAVE 收益配置与强对照",
                 f"{scope}  |  历史回放  |  单边 5 bp", names, "replay")
    figures = []
    for key, r in full.iterrows():
        filename = f"returns_{key}.png"
        returns_plot(folder, paths, (key,), filename, names[key],
                     f"净年化 {percent(r.cagr)}   夏普 {r.sharpe:.3f}   最大回撤 {percent(r.max_drawdown)}",
                     names, "replay")
        figures.append(figure(filename, f"{names[key]}累计净收益与回撤", names[key]))

    first_path = paths[("replay", "core")]
    contributions = []
    for year, g in [("all", first_path), *((str(y), p) for y, p in first_path.groupby(first_path.index.year))]:
        denom = g.nav.iloc[0] / (1 + g.net_return.iloc[0])
        values = g[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee"]].sum() / denom
        contributions.append((slices[year], *[percent(x) for x in values], percent(values.sum())))
    decomposition = table("120 日主干的净值归因", ["期段", "股票", "黄金", "国债", "费用", "合计"], contributions)
    dev_choice = selection.loc[selection.strategy.eq(selected)].iloc[0]
    passed = bool(dev_choice.joint_improvement_both_costs)
    test_pass = all(
        covers_objectives(merged.loc[("all", fee, selected)], merged.loc[("all", fee, "core")])
        for fee in (BASE_COST, 10)
    )
    flag = "达到" if passed else "未达到"
    criterion = "达到" if test_pass else "未达到"
    select_rows = selection.query("one_way_bps==@BASE_COST")
    dev = select_rows.set_index("strategy")
    risk60 = full.loc["risk60"]
    dev_table = table(
        "开发期 · 规则锁定前的选择证据",
        ["策略", "净年化", "净夏普", "最大回撤", "CDaR", "5 和 10 bp 均同向改善", "选入主比较"],
        [(names[r.strategy], percent(r.cagr), number(r.sharpe), percent(r.max_drawdown), percent(r.cdar95),
          ("是" if r.joint_improvement_both_costs else "否") if r.in_tmave_selection_pool else "—",
          "是" if r.chosen_for_replay_comparison else "—") for r in select_rows.itertuples()],
        mark, base_name,
    )
    annual = [
        lines(f"{slices[year]} · 5 bp", merged.loc[(year, BASE_COST)])
        for year in ("2024", "2025", "2026")
    ]
    fees = table(
        "压力测试 · 净年化 / 净夏普 / 最大回撤",
        ["策略", *(f"{fee} bp" for fee in FEES)],
        [(names[key], *[f"{percent(r.cagr)} / {r.sharpe:.3f} / {percent(r.max_drawdown)}"
                         for fee in FEES for r in [merged.loc[("all", fee, key)]]])
         for key in full.index], mark, base_name,
    )
    exposure = table(
        "历史回放 · 日均收盘敞口",
        ["策略", "股票", "黄金", "国债", "现金"],
        [(names[key], *(percent(r[col]) for col in (
            "mean_equity_weight", "mean_gold_weight", "mean_bond_weight", "mean_cash_weight",
        ))) for key, r in full.iterrows()], mark, base_name,
    )
    point = intervals.query("block == 20")
    ci_labels = {"net_sharpe": "夏普", "cagr": "净年化", "cdar95": "CDaR", "max_drawdown": "最大回撤"}
    ci_phase = {"replay": "回放", "replay_2026": "2026", "independent": "独立评估"}
    confidence = table(
        "所选方案减对照 · 20 日配对区块 95% 区间",
        ["期间 / 基准", "指标", "点估计", "下界", "上界"],
        [(f"{ci_phase[r.phase]} · {names[r.benchmark]}", ci_labels[r.metric],
          *[f"{v:+.3f}" if r.metric == "net_sharpe" else f"{pp(v)} 个百分点"
            for v in (r.difference, r.lower, r.upper)]) for r in point.itertuples()],
    )
    quality_phase = {"development": "开发期", **ci_phase}
    checks = table(
        "每个调仓周期的相对收益预测",
        ["期间 / 模型", "不重叠样本", "预测与标签秩相关", "方向正确率"],
        [(f"{quality_phase[r.phase]} · {names[r.model]}", r.nonoverlap_weeks,
          number(r.rank_corr), percent(r.direction_accuracy)) for r in quality.itertuples()],
    )
    case_table = table(
        "同一压力窗口的策略对照",
        ["情景 / 策略", "样本", "平均最大回撤", "平均净收益", "随后五日收益", "平均换手"],
        [(f"{regime} · {names[key]}", len(q), percent(q.max_drawdown.mean()), percent(q.net_return.mean()),
          percent(q.rebounds_5d.mean()), percent(q.turnover.mean()))
         for (regime, key), q in scenarios.loc[scenarios.strategy.isin(lookup)]
         .groupby(["scenario", "strategy"], sort=False)],
    )
    independent = performance.query("phase=='independent' and year=='all' and one_way_bps==@BASE_COST")
    independence = (
        f"冻结后共有 {fresh} 日，单列独立评估。" if len(independent) else
        f"冻结后当前输入为 {fresh} 个交易日，本实验规定 {MIN_FRESH_SESSIONS} 日才进行独立评估。"
    )
    indep_table = lines("冻结后评估", independent.set_index("strategy")) if len(independent) else ""
    s = full.loc[same]
    interval = point.query("phase=='replay' and metric=='net_sharpe' and benchmark=='core'").iloc[0]
    digest = "".join(f"<li><code>{escape(f)} {h}</code></li>" for f, h in hashes.items())
    csv = {
        "portfolio_metrics": "完整指标", "daily_backtest": "逐日净值与持仓",
        "relative_signals": "调仓信号、因子与标签", "train_folds": "年度训练边界",
        "forecast_quality": "预测评估", "selection": "选型依据",
        "paired_intervals": "配对区间", "scenarios": "压力情景",
    }
    links = " · ".join(f'<a href="{f}.csv">{title}</a>' for f, title in csv.items())
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · T-MAVE 收益增强</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF / RELATIVE ALLOCATION</div>
<h1>股金相对配置能否超越风险预算</h1>
<p class="meta">历史回放 {scope}　·　{now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#baseline">主干高分的来源</a>
<a href="#results">绩效与逐年</a><a href="#validation">收益预测与对照</a><a href="#method">复现</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>所选的{mark}在历史回放取得净年化 {percent(tmave.cagr)}、夏普 {tmave.sharpe:.3f}、
最大回撤 {percent(tmave.max_drawdown)}。</strong> 120 日主干对应 {percent(baseline.cagr)}、{baseline.sharpe:.3f}、
{percent(baseline.max_drawdown)}。开发期{flag} 5 与 10 bp 下的共同改善门槛，历史回放{criterion}。
全段夏普差的配对 95% 区间为 [{interval.lower:+.3f}, {interval.upper:+.3f}]。</p>
<p>2026 年，所选方案净收益 {percent(last.at[selected, "total_return"])}，主干 {percent(last.at["core", "total_return"])}。
同因子和同防守设置的固定双记忆回放净年化 {percent(s.cagr)}、夏普 {s.sharpe:.3f}、最大回撤 {percent(s.max_drawdown)}。</p>
<p>60 日风险预算的回放净年化 {percent(risk60.cagr)}、夏普 {risk60.sharpe:.3f}、最大回撤 {percent(risk60.max_drawdown)}。
开发期 60 日风险预算和 120 日主干的净夏普分别为 {dev.at['risk60', 'sharpe']:.3f} 和 {dev.at['core', 'sharpe']:.3f}。</p>
<p>{independence}已查看的年份用于检验机制和诊断组合。</p></div></section>
<section id="baseline"><h2>主干为什么能达到约 1.7 的夏普</h2>
<p>2024–2026 回放的主干平均国债权重 {percent(baseline.mean_bond_weight)}，年化波动 {percent(baseline.annual_vol)}。
2024 和 2025 的正收益抬高了整个期间的夏普。开发期 2021–2023 的主干净夏普为
{dev.at['core', 'sharpe']:.3f}，2026 年为 {last.at["core", "sharpe"]:.3f}。</p>
{decomposition}
<p class="note">逐年贡献除以当年期初净值。各资产损益和费用精确加总为区间净收益。</p></section>
<section id="results"><h2>绩效与逐年结果</h2>
{figure("returns_comparison.png", "T-MAVE 与主要对照的收益回撤", "图 1　相同五只 ETF 与同一撮合成本的主要对照。")}
{lines("历史回放 · 5 bp", full)}{indep_table}
<details><summary>查看 2024、2025 和 2026 逐年表</summary>{''.join(annual)}</details>
<details><summary>十个模型的收益图</summary><div class="grid">{''.join(figures)}</div></details>
<p class="note">年化使用 252 个交易日，夏普用日净收益与零无风险利率。组合最大回撤为负值，CDaR 为最深 5% 日回撤的平均深度。</p></section>
<section id="validation"><h2>同因子、同防守、同时间对照</h2>
{dev_table}{checks}{exposure}
<p>因子模型的目标是下一调仓周期之内的股金相对收益。检验标签之间按开盘边界相接，日组合损益采用原回测引擎。
需要预测结果能通过组间划拨产生扣费后的收益，才形成策略贡献。</p>
<details><summary>成本、{scenarios.start.nunique()} 个情景与配对区间</summary>{fees}{case_table}{confidence}
<p class="note">历史权益冲击候选共 {candidates} 个。按冲击后路径分类，实际信号为点时因子。
2,000 次循环区块按双方同日重抽样。10 和 40 日敏感性在 CSV。</p></details></section>
<section id="method"><h2>方法与复现</h2>
<dl><dt>因子</dt><dd>15/60/120 日风险调整动量差、20 日下行波动比和有符号冲击差。
同一训练数据用固定岭系数 0.10 对比静态因子、固定双记忆、交易日版与完整 T-MAVE。</dd>
<dt>T-MAVE</dt><dd>快慢状态经交易日、自然日与已确认事件流衰减，在回放中按既有门控写入。
组间转移至多为股金总目标敞口的 {percent(TILT_LIMIT,0)}，国债维持原目标。防守版本共用固定记忆控仓，未控组维持其配置目标。</dd>
<dt>时间</dt><dd>每年只使用结束开盘早于该年首次信号的标签，任何选择均基于 2021–2023。
用同一研究流程尝试期限和状态结构，避免把单条回放的高分当成独立验证。</dd></dl>
<p><a href="https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf">多资产时序动量原始研究</a> ·
<a href="https://www.nber.org/papers/w20592">多重因子检验原始研究</a></p>
<p>运行 <code>python -B -X utf8 chronoetf/backtest.py --alpha</code>。</p>
<p>{links}</p><details><summary>输入指纹</summary><ul class="sources">{digest}</ul></details>
</section><footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def write_sleeve_report(folder, now, metrics, paths, selection, winner, factors, diagnostics, calibration,
                         scenarios, candidates, joint, episodes, intervals, fresh, hashes):
    from sleeves import EXPOSURE_TOL, FREEZE_DATE, GUARD_SCALE, GROUP_ARMS, LABELS, SLEEVES
    from tmave import FEES, MIN_FRESH_SESSIONS

    model_names = LABELS
    main = ("core", *GROUP_ARMS)
    methods = ("core", winner, "rolling", "fixed", "static")
    clock_modes = ("trade", "natural", "event", winner)
    base = metrics.loc[metrics.one_way_bps.eq(BASE_COST)]
    full = base.query("phase == 'replay' and year == 'all'").set_index("strategy")
    recent = base.query("phase == 'replay' and year == '2026'").set_index("strategy")
    dev = base.query("phase == 'development' and year == 'all'").set_index("strategy")
    start, end = full.at["core", "first_date"], full.at["core", "last_date"]
    focus, old = model_names[winner], model_names["core"]
    phase_names = {"development": "开发期", "replay": "历史回放", "replay_2026": "2026 回放", "independent": "冻结后评估"}
    vols = ("mean_equity_weight", "mean_gold_weight", "mean_bond_weight", "mean_cash_weight")
    names = SLEEVES

    def pp(value):
        return f"{value * 100:+.2f}"

    def recovery(r):
        return int(r.recovery_sessions) if np.isfinite(r.recovery_sessions) else "截至期末未恢复"

    def performance_table(caption, frame):
        return table(
            caption, ["策略", "区间净收益", "净年化", "净夏普", "最大回撤", "CDaR 95%", "年化波动",
                      "最深回撤恢复日", "最长水下日", "双边年换手", "费用 / 期初净值"],
            [(model_names[key], percent(r.total_return), percent(r.cagr), number(r.sharpe),
              percent(r.max_drawdown), percent(r.cdar95), percent(r.annual_vol), recovery(r),
              int(r.drawdown_days), f"{r.annual_turnover_two_way:.2f}×", percent(r.fee_over_period_start_nav))
             for key, r in frame.iterrows()],
            focus, old,
        )

    fonts()
    returns_plot(folder, paths, main, "returns_groups.png", "分组风控的四条主路径",
                 f"{start} 至 {end}  |  单边 5 bp  |  历史回放", model_names, "replay")
    returns_plot(folder, paths, methods, "returns_mechanism.png", "同因子、同主干的机制比较",
                 "T-MAVE 与滚动阈值、固定双记忆和静态敞口", model_names, "replay")
    images = []
    for key, r in full.iterrows():
        img = f"returns_{key}.png"
        returns_plot(
            folder, paths, (key,), img, model_names[key],
            f"{start} 至 {end}  |  净年化 {percent(r.cagr)}  ·  净夏普 {r.sharpe:.2f}  ·  回撤 {percent(r.max_drawdown)}",
            model_names, "replay",
        )
        images.append(figure(img, f"{model_names[key]}的收益与回撤", model_names[key]))

    by_year = []
    for year, frame in base.query("phase == 'replay' and year != 'all'").groupby("year", sort=True):
        years = frame.set_index("strategy")
        first = years.iloc[0]
        by_year.append(performance_table(
            f"{year} · {first.first_date} 至 {first.last_date} · {first.trading_days} 个交易日 · 5 bp",
            years.loc[list(dict.fromkeys([*main, "rolling", "fixed", "static"]))],
        ))
    choices = table(
        "开发期选型 · 单边 5 bp", ["组合", "净年化", "净夏普", "最大回撤", "CDaR 95%",
                               "四项共同改善", "回撤双改善且收益损失不超过 0.5 个百分点", "进入机制实验"],
        [(model_names[r.strategy], percent(r.cagr), number(r.sharpe), percent(r.max_drawdown), percent(r.cdar95),
          "是" if r.joint_improvement else "—" if r.strategy == "core" else "否",
          "是" if r.bounded_improvement else "—" if r.strategy == "core" else "否",
          "是" if r.selected_for_ablation else "—") for r in selection.itertuples()],
        focus, old,
    )
    exposure_rows = []
    for phase, stats in (("开发期", dev), ("历史回放", full), ("2026 回放", recent)):
        for key in methods:
            exposure_rows.append((f"{phase} · {model_names[key]}", *[percent(stats.at[key, col]) for col in vols]))
    exposures = table("日均收盘敞口", ["期间 / 策略", "股票", "黄金", "国债", "现金"], exposure_rows)
    exposure_gap = full.loc[winner, list(vols)].astype(float) - full.loc["static", list(vols)].astype(float)
    recent_exposure_gap = recent.loc[winner, list(vols)].astype(float) - recent.loc["static", list(vols)].astype(float)
    aligned = "满足" if exposure_gap.abs().max() <= EXPOSURE_TOL else "超出"
    aligned_2026 = "满足" if recent_exposure_gap.abs().max() <= EXPOSURE_TOL else "超出"
    cost_rows = []
    cost_data = metrics.query("phase == 'replay' and year in ['all','2026']").set_index(["strategy", "year", "one_way_bps"])
    for phase in ("all", "2026"):
        for key in dict.fromkeys([*main, "rolling", "fixed", "static"]):
            label = "全回放" if phase == "all" else "2026"
            cost_rows.append((f"{label} · {model_names[key]}", *[
                f"{percent(r.cagr if phase == 'all' else r.total_return)} / {r.sharpe:.3f} / {percent(r.max_drawdown)}"
                for bps in FEES for r in [cost_data.loc[(key, phase, bps)]]
            ]))
    costs = table("成本压力 · 净收益 / 净夏普 / 最大回撤", ["区间 / 策略", *(f"{bps} bp" for bps in FEES)], cost_rows)

    factor_names = {
        "momentum_15": "15 日风险调整动量", "momentum_60": "60 日风险调整动量",
        "momentum_120": "120 日风险调整动量", "momentum_blend": "多周期综合动量",
        "downside_20": "20 日下行波动", "shock_z": "有符号标准化冲击", "pressure_candidate": "合成压力",
    }
    overview_factors = diagnostics.loc[diagnostics.phase.eq("replay") & diagnostics.factor.isin(
        ("momentum_blend", "downside_20", "shock_z", "pressure_candidate"),
    )]
    factor_result = table(
        "每五个交易日抽样 · 未来五日标签",
        ["资产组 / 因子", "覆盖率", "有效配对", "收益秩相关", "下行波动秩相关"],
        [(f"{names[r.sleeve]} · {factor_names[r.factor]}", percent(r.factor_coverage, 1), r.paired_samples,
          number(r.rank_corr_return), number(r.rank_corr_downside)) for r in overview_factors.itertuples()],
    )
    gold_diag = overview_factors.query("sleeve=='gold' and factor=='downside_20'").iloc[0]
    rules = table(
        "2015–2020 校准与开发期静态倍率",
        ["组别", "冲击阈值", "预期事件单位", "三时钟进入 / 回补", "固定写入进入 / 回补",
         "滚动进入 / 回补", "静态倍率"],
        [(names[r.sleeve], number(r.shock_quantile), number(r.event_mean),
          f"{r.full_entry:.3f} / {r.full_exit:.3f}", f"{r.fixed_entry:.3f} / {r.fixed_exit:.3f}",
          f"{r.rolling_entry:.3f} / {r.rolling_exit:.3f}", number(r.static_multiplier)) for r in calibration.itertuples()],
    )

    ref = paths[("replay", winner)]
    order = ref["scheduled"]
    target = ["target_equity", "target_gold"]
    state_rows = []
    for key in clock_modes:
        path = paths[("replay", key)]
        diff = np.abs(path[target].to_numpy() - ref[target].to_numpy()).max(axis=1) > 1e-12
        history = episodes.loc[episodes.phase.eq("replay") & episodes.strategy.eq(key)]
        resolved = history.loc[history.closed, "held_sessions"]
        state_rows.append((
            model_names[key], int((diff & order).sum()), int((diff & order & (ref.index.year == 2026)).sum()),
            int(history.shape[0]), int(history.closed.sum()), number(resolved.median() if len(resolved) else np.nan, 1),
            number(full.at[key, "sharpe"]), percent(full.at[key, "cagr"]), percent(full.at[key, "max_drawdown"]),
        ))
    clocks = table(
        "同阈值时钟比较 · 相对完整三时钟的指令差",
        ["模型", "全回放不同调仓日", "2026 不同调仓日", "降仓指令", "完成回补", "回补中位日数",
         "全回放净夏普", "净年化", "最大回撤"], state_rows, focus,
    )
    clock_finding = (
        "2026 年，交易日基准、加自然日、加事件时间和三类时间合用生成同一套目标仓位。"
        "因此本段收益差无法归因到时钟。"
        if all(row[2] == 0 for row in state_rows) else
        "2026 年的目标权重差出现在表中记录的调仓日。每条权重变化都可追溯到当日已知压力。"
    )
    traces = []
    for sleeve in GROUP_ARMS[winner]:
        act = ref.loc[(ref.index.year == 2026) & ref[f"action_{sleeve}"].isin(("reduce", "restore"))]
        for day, row in act.iterrows():
            z = factors.loc[(row["signal_date"], sleeve)]
            traces.append((
                names[sleeve], row["signal_date"].date().isoformat(), day.date().isoformat(),
                "减仓" if row[f"action_{sleeve}"] == "reduce" else "回补", number(z.momentum_blend),
                number(z.downside_20, 5), number(z.shock_z), number(z.short_full), number(z.anchor_full),
                number(z.pressure_full), z.event_class, percent(row[f"target_{sleeve}"]),
            ))
    signals = table(
        "2026 可追溯的减仓与回补指令",
        ["组别", "信号收盘", "执行开盘", "动作", "动量", "下行波动", "冲击", "短状态", "长期锚点", "读出压力", "事件", "组目标仓位"],
        sorted(traces, key=lambda row: (row[1], row[0])),
    )
    cases = scenarios.loc[scenarios.strategy.isin(methods)]
    case_table = table(
        "急跌反弹、持续下跌与震荡反复 · 事后分类",
        ["类型 / 策略", "窗口", "平均区间净收益", "平均回撤", "冲击后五日净收益", "窗口内恢复", "平均换手"],
        [(f"{phase} · {model_names[key]}", len(q), percent(q.net_return.mean()), percent(q.max_drawdown.mean()),
          percent(q.rebounds_5d.mean()), f"{q.recovered.sum()}/{len(q)}", percent(q.turnover.mean()))
         for (phase, key), q in cases.groupby(["scenario", "strategy"], sort=False)],
    )
    condition = table(
        "2026 同步走弱的日损益 · 以前一收盘的股金 20 日收益定义",
        ["状态 / 策略", "交易日", "股票贡献", "黄金贡献", "国债贡献", "费用", "净贡献", "现金"],
        [(f"{'股金同弱' if r.condition=='joint_weak' else '其他日'} · {model_names[r.strategy]}",
          r.sessions, percent(r.pnl_equity), percent(r.pnl_gold), percent(r.pnl_bond), percent(r.pnl_fee),
          percent(r.return_contribution), percent(r.mean_cash))
         for r in joint.loc[joint.strategy.isin(methods)].sort_values("condition", kind="stable").itertuples()],
    )
    point_20 = intervals.loc[intervals.block.eq(20)]
    ci_names = {"net_sharpe": "净夏普", "cagr": "净年化", "cdar95": "CDaR 95%", "max_drawdown": "最大回撤"}
    confidence = table(
        "所选 T-MAVE 减对照 · 同日配对 · 20 日区块",
        ["期间 / 对照", "度量", "差值", "95% 下界", "95% 上界"],
        [(f"{phase_names[r.phase]} · {model_names[r.benchmark]}", ci_names[r.metric],
          *[f"{x:+.3f}" if r.metric == "net_sharpe" else f"{pp(x)} 个百分点"
            for x in (r.difference, r.lower, r.upper)]) for r in point_20.itertuples()],
    )
    inference = point_20.query("phase=='replay_2026' and benchmark=='core' and metric=='max_drawdown'").iloc[0]
    unlocked = base.query("phase=='independent' and year=='all'").set_index("strategy")
    unlocked_note = (
        f"冻结后有 {fresh} 个可用交易日，另表展示独立评估。"
        if len(unlocked) else f"冻结后当前输入有 {fresh} 个交易日，至少积累 {MIN_FRESH_SESSIONS} 日再做独立评估。"
    )
    unlocked_table = performance_table("冻结后独立评估 · 5 bp", unlocked) if len(unlocked) else ""
    shared = joint.query("condition=='joint_weak'").set_index("strategy")
    flags = selection.loc[selection.strategy.eq(winner)].iloc[0]
    outcome = (
        "同时改善净年化、净夏普、最大回撤和 CDaR" if flags.joint_improvement else
        "符合预设回撤及年化收益限制" if flags.bounded_improvement else "从 Pareto 前沿保留一条实验路径"
    )
    attachments = {
        "portfolio_metrics": "完整指标", "selection": "开发期选型", "daily_backtest": "净值、敞口与订单",
        "factor_states": "因子、时钟与状态", "factor_validation": "因子诊断", "factor_calibration": "阈值与倍率",
        "scenarios": "压力情景", "year_2026_attribution": "2026 同弱归因", "guard_episodes": "回补指令",
        "paired_intervals": "配对区间",
    }
    links = " · ".join(f'<a href="{key}.csv">{label}</a>' for key, label in attachments.items())
    fingerprints = "".join(f"<li><code>{escape(name)} {digest}</code></li>" for name, digest in hashes.items())
    replay_joint = int(shared.at[winner, "sessions"])
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · 因子与 T-MAVE 分组风控</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF / T-MAVE</div><h1>哪些资产组需要减仓</h1>
<p class="meta">五只境内 ETF，三类时间与资产组风控　·　{now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#comparison">分组与逐年表现</a>
<a href="#factors">因子</a><a href="#clock">时钟与指令</a><a href="#robust">情景与稳健性</a>
<a href="#reproduce">复现与简历</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>开发期锁定{focus}。2026 年净收益为 {percent(recent.at[winner, "total_return"])}，主干为
{percent(recent.at["core", "total_return"])}。最大回撤由 {percent(recent.at["core", "max_drawdown"])}
变为 {percent(recent.at[winner, "max_drawdown"])}。</strong> 最大回撤差的 20 日配对区块 95% 区间为
[{pp(inference.lower)}, {pp(inference.upper)}] 个百分点。</p>
<p>2026 年固定双记忆的净收益为 {percent(recent.at["fixed", "total_return"])}，
静态敞口对照最大回撤为 {percent(recent.at["static", "max_drawdown"])}。
在 {replay_joint} 个股金同弱日，主干和所选模型的净损益贡献为
{percent(shared.at["core", "return_contribution"])} 与 {percent(shared.at[winner, "return_contribution"])}。</p>
<p>全回放净年化为 {percent(full.at[winner, "cagr"])}，主干为 {percent(full.at["core", "cagr"])}。
净夏普分别为 {number(full.at[winner, "sharpe"])} 和 {number(full.at["core", "sharpe"])}。</p>
<p>历史回放期为 {start} 至 {end}，已用于之前的研究诊断。{unlocked_note}</p></div></section>
<section id="comparison"><h2>分组与逐年表现</h2>
{figure("returns_groups.png", "四条分组主策略的累计净收益与回撤", "图 1　国债目标权重沿用主干，减出的股票或黄金组权重留现金。")}
<p>2015–2020 固定参数。2021–2023 开发期所选组合{outcome}，静态对照也仅用这一段匹配敞口。</p>
{choices}{performance_table("同一历史区间 · 5 bp", full)}{unlocked_table}
{''.join(by_year)}
<p class="note">年度结果从连续净值按年切片。2026 年截至 10 月 30 日。年化按 252 日换算，夏普使用日净收益及零无风险利率。
最大回撤为负值。CDaR 95% 为最深 5% 的日回撤平均深度。</p>
<details><summary>十种策略的独立收益图</summary><div class="grid">{''.join(images)}</div></details></section>
<section id="factors"><h2>因子构建与诊断</h2>
<p>输入为 15/60/120 日风险调整动量、20 日下行波动、当日有符号冲击及同向重复或权益共振。
股票组的信号篮子按上一收盘可知的 60 日逆波动权重构成。三组共用候选压力的构造方式，阈值分组估计。</p>
{factor_result}
<p>黄金 20 日下行波动与未来五日下行波动的回放秩相关为 {gold_diag.rank_corr_downside:.3f}，
基于 {gold_diag.paired_samples} 个有标签观测。每隔五个交易日抽样，诊断标签在各阶段结尾留空。
覆盖率和相关用于描述本次样本。正式组合收益按次日开盘模拟成交。</p>
{rules}</section>
<section id="clock"><h2>三类时间与仓位指令</h2>
{figure("returns_mechanism.png", "T-MAVE、固定记忆、滚动阈值与静态敞口的净值", "图 2　相同因子与风控幅度下的机制对照。")}
<p>T-MAVE 将交易日步数、实际休市间隔和已确认事件分别计时。先用三类时间衰减，再把新证据写入短期记忆。
长期锚点由同向重复及权益共振更新。组压力满足进入或回补阈值后，下一可交易日开盘调整该组。</p>
{clocks}
<p>{clock_finding}</p>
<p class="note">时钟表的“不同调仓日”比较股票和黄金目标权重，同一日只记一次。回补天数按实际指令之间的交易日计量，
中位数使用已经回补的警戒区间。各组完整长短状态及未结束区间见 CSV。</p>
<details><summary>查看 2026 年每笔减仓与回补的证据</summary>{signals}</details></section>
<section id="robust"><h2>情景、资产敞口与不确定性</h2>
{exposures}
<p>历史回放所选模型减静态对照的股、金、债、现金敞口差依次为
{', '.join(pp(v) for v in exposure_gap)} 个百分点，{aligned}逐项 1 个百分点的近似匹配范围。
2026 单年差值依次为 {', '.join(pp(v) for v in recent_exposure_gap)} 个百分点，{aligned_2026}该范围。</p>
{condition}{case_table}
<p class="note">同弱日按上一收盘的两组过去 20 日对数收益都低于零定义，分组日净损益贡献以 2026 期初净值为分母。
另有 {candidates} 个权益冲击候选，形成 {scenarios.start.nunique()} 个不重叠窗口；三类窗口按事后路径分类。</p>
<details><summary>三档交易成本</summary>{costs}</details>
<details open><summary>配对区块区间</summary>{confidence}
<p class="note">2,000 次配对循环区块重抽样，先重建两条净值再计算每个指标。10 和 40 日结果附在 CSV。
回撤差为正代表最深回撤收窄，CDaR 差为负代表改善。</p></details></section>
<section id="reproduce"><h2>复现与实习项目表述</h2>
<p>基于五只境内 ETF 构建多周期动量和下行风险因子；将 T-MAVE 的双记忆、三类时钟及有界证据更新迁移到分组风控。在统一交易成本下回测 {len(full)} 条策略，以同因子和平均敞口对照检验各机制对净收益与回撤的影响。</p>
<p>面试时可从一笔可追溯的周度订单说明：因子描述当时的强弱，快慢记忆决定压力是否持续，组合只修改被警戒组。将 2026 年收益、回撤与同因子固定双记忆和静态敞口同行展示。</p>
<dl>
<dt>时钟</dt><dd>自然日单位为真实间隔乘以 5/7，和交易日等权混合；有效事件使用上一交易日的活动量，
按历史均值归一，占混合时钟 25%。快、慢半衰期分别为 5 和 30。</dd>
<dt>下单</dt><dd>每组警戒时将目标仓位乘以 {GUARD_SCALE:.2f}。国债及未警戒组维持主干目标仓位。实际敞口由价格漂移与交易费用决定。
五只 ETF 的后复权日线用于权重回测，冻结日期为 {FREEZE_DATE:%Y-%m-%d}。</dd>
</dl>
<p>从项目根目录运行 <code>python -B -X utf8 chronoetf/backtest.py --sleeves</code>。</p>
<p>{links}</p><details><summary>五份输入文件 SHA-256</summary><ul class="sources">{fingerprints}</ul></details>
</section><footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def write_tmave_report(folder, now, performance, paths, scenarios, candidates, calibration,
                       forecasts, intervals, folds, fresh_sessions, hashes):
    from tmave import (
        CDAR_GAIN, FEES, FREEZE_DATE, MAX_CAGR_LOSS, MIN_FRESH_SESSIONS, REGIMES,
        REPLAY_EXPOSURE_TOL, STRATEGIES as names,
    )

    def fmt_pp(value):
        return f"{value * 100:+.2f}"

    def abs_table(frame, caption):
        return table(
            caption, ["策略", "累计净收益", "净年化", "最大回撤", "CDaR 95%", "年化波动", "净夏普",
                      "最长水下日数", "平均权益", "平均现金", "双边年换手", "累计费用"],
            [(names[key], percent(r.total_return), percent(r.cagr), percent(r.max_drawdown), percent(r.cdar95),
              percent(r.annual_vol), number(r.sharpe), int(r.drawdown_days), percent(r.mean_equity_weight),
              percent(r.mean_cash_weight), f"{r.annual_turnover_two_way:.2f}×", percent(r.fee_over_period_start_nav))
             for key, r in frame.iterrows()],
            names["new_new"], names["old_old"],
        )

    full = performance.loc[performance["year"].eq("all")]
    base = full.loc[full["one_way_bps"].eq(BASE_COST)]
    replay = base.loc[base["phase"].eq("replay")].set_index("strategy")
    develop = base.loc[base["phase"].eq("development")].set_index("strategy")
    old, new = replay.loc["old_old"], replay.loc["new_new"]
    gain = 1 - new.cdar95 / old.cdar95
    gain_text = f"相对下降 {percent(gain)}" if gain >= 0 else f"相对上升 {percent(-gain)}"
    return_gap = new.cagr - old.cagr
    gate = gain >= CDAR_GAIN and return_gap >= -MAX_CAGR_LOSS
    equity_gap = new.mean_equity_weight - old.mean_equity_weight
    cash_gap = new.mean_cash_weight - old.mean_cash_weight
    matched = max(abs(equity_gap), abs(cash_gap)) <= REPLAY_EXPOSURE_TOL
    interval = intervals.query("phase=='replay' and comparison=='old_old' and block==20 and measure=='relative_cdar_gain'").iloc[0]
    main_keys = ("old_old", "old_new", "new_old", "new_new")
    sample = f"{old.first_date} 至 {old.last_date}"
    bounds = f"[{percent(interval.lower)}, {percent(interval.upper)}]"
    comparison_status = "达到" if gate else "未达到"
    exposure_status = "在" if matched else "超出"
    outcome = f"{comparison_status}预设的 CDaR 与净年化双门槛"
    interval_note = (
        "区间跨过零" if interval.lower <= 0 <= interval.upper
        else "区间低于零" if interval.upper < 0 else "区间高于零"
    )
    independent_text = (
        f"冻结日后共有 {fresh_sessions} 个可用交易日，单列独立评估。"
        if "independent" in base["phase"].values else
        f"输入只覆盖至 {old.last_date}，冻结日后共 {fresh_sessions} 个可用交易日。独立评估需至少 "
        f"{MIN_FRESH_SESSIONS} 日，本次证据限于历史回放。"
    )
    fonts()
    returns_plot(folder, paths, main_keys, "returns_comparison.png", "T-MAVE 的四组因子对照",
                 f"{sample}  |  {old.trading_days} 个交易日  |  历史回放", names, "replay")
    gallery = []
    for key, name in names.items():
        row = replay.loc[key]
        filename = f"returns_{key}.png"
        returns_plot(
            folder, paths, (key,), filename, name,
            f"{sample}  |  累计 {percent(row.total_return)}  ·  CDaR {percent(row.cdar95)}  ·  夏普 {row.sharpe:.2f}",
            names, "replay",
        )
        gallery.append(figure(filename, f"{name}收益与回撤", name))

    factorial = table(
        "2×2 主实验 · 5 bp",
        ["记忆 / 仓位", "CDaR 95%", "相对现有组合改善", "净年化变化（百分点）", "平均权益", "平均现金"],
        [(names[key], percent(replay.at[key, "cdar95"]), percent(1 - replay.at[key, "cdar95"] / old.cdar95),
          fmt_pp(replay.at[key, "cagr"] - old.cagr), percent(replay.at[key, "mean_equity_weight"]),
          percent(replay.at[key, "mean_cash_weight"])) for key in main_keys], names["new_new"], names["old_old"],
    )
    c = replay["cdar95"]
    change = {
        "memory": fmt_pp(c["new_old"] - c["old_old"]),
        "reading": fmt_pp(c["old_new"] - c["old_old"]),
        "interaction": fmt_pp(c["new_new"] - c["new_old"] - c["old_new"] + c["old_old"]),
        "evidence": fmt_pp(c["new_new"] - c["impulse"]),
    }
    focused = replay.loc["old_new"]
    focused_gain = 1 - focused.cdar95 / old.cdar95
    focused_gap = focused.cagr - old.cagr
    costs = full.loc[full["phase"].eq("replay")]
    fee_lookup = costs.set_index(["strategy", "one_way_bps"])
    cost_table = table(
        "成本压力 · 净年化 / CDaR 95%",
        ["策略", *(f"{bps} bp" for bps in FEES)],
        [(names[key], *[f"{percent(fee_lookup.at[(key, bps), 'cagr'])} / "
                         f"{percent(fee_lookup.at[(key, bps), 'cdar95'])}" for bps in FEES])
         for key in names], names["new_new"], names["old_old"],
    )
    forecasts_table = table(
        "同一资产日样本的风险预测",
        ["阶段 / 模型", "目标", "评价指标", "损失", "资产日"],
        [(f"{r.phase} / {r.model}", "总方差" if r.target == "variance" else "下行半方差",
          "QLIKE" if r.target == "variance" else "MAE × 10⁴",
          number(r.loss if r.target == "variance" else r.loss * 10000, 4), f"{r.asset_days:,}")
         for r in forecasts.itertuples()],
    )
    interval_table = table(
        "配对循环区块自助 · 95% 区间",
        ["阶段 / 对照", "区块（日）", "度量", "点估计", "区间"],
        [(f"{r.phase} / {names[r.comparison]}", r.block,
          "CDaR 相对改善" if r.measure == "relative_cdar_gain" else "净年化差",
          percent(r.estimate), f"[{percent(r.lower)}, {percent(r.upper)}]") for r in intervals.itertuples()],
    )
    calibration_table = table(
        "2021–2023 暴露校准",
        ["策略", "含权益篮子倍数", "纯非权益倍数", "权益误差（百分点）", "现金误差（百分点）", "减仓阈值", "回补阈值"],
        [(names[r.strategy], number(r.equity_basket_scale), number(r.other_basket_scale),
          fmt_pp(r.development_equity_diff), fmt_pp(r.development_cash_diff),
          number(r.entry_threshold), number(r.exit_threshold)) for r in calibration.itertuples()],
    )
    seasonal = performance.loc[
        performance["phase"].eq("replay") & performance["year"].ne("all")
        & performance["strategy"].isin(main_keys)
    ]
    annual_table = table(
        "历史回放 · 年度截面",
        ["年份 / 策略", "交易日", "累计净收益", "净年化", "CDaR 95%", "最大回撤", "净夏普"],
        [(f"{r.year} / {names[r.strategy]}", r.trading_days, percent(r.total_return), percent(r.cagr),
          percent(r.cdar95), percent(r.max_drawdown), number(r.sharpe))
         for r in seasonal.sort_values("year", kind="stable").itertuples()],
    )

    scenarios_rows = []
    narrative = []
    main_scenarios = scenarios.loc[scenarios["strategy"].isin(("old_old", "new_new"))]
    for regime in REGIMES:
        subset = main_scenarios.loc[main_scenarios["scenario"].eq(regime)]
        if subset.empty:
            narrative.append(f"{regime}组没有符合冻结规则的窗口。")
            continue
        paired = subset.groupby("strategy", sort=False)
        n = int(subset["start"].nunique())
        grouping = paired[["max_drawdown", "rebounds_5d", "net_return", "turnover"]].mean()
        narrative.append(
            f"{regime}组有 {n} 个窗口，新组合相对现有组合的平均最大回撤差为 "
            f"{fmt_pp(grouping.at['new_new', 'max_drawdown'] - grouping.at['old_old', 'max_drawdown'])} 个百分点，"
            f"后五日净收益差为 {fmt_pp(grouping.at['new_new', 'rebounds_5d'] - grouping.at['old_old', 'rebounds_5d'])} 个百分点。"
        )
        for key, g in paired:
            recovered = g["days_to_recover_pre_shock_nav"].dropna()
            scenarios_rows.append((
                f"{regime} / {names[key]}", n, percent(g["max_drawdown"].mean()),
                percent(g["rebounds_5d"].mean()), percent(g["net_return"].mean()), percent(g["turnover"].mean()),
                f"{len(recovered)}/{n}", number(recovered.median() if len(recovered) else np.nan, 1),
            ))
    scenarios_table = table(
        "情景配对窗口 · 5 bp · 每窗从首日盘前净值重新计量",
        ["情景 / 策略", "窗口数", "平均最大回撤", "冲击后 5 日净收益", "11 日净收益", "平均换手", "恢复比例", "成功恢复中位天数"],
        scenarios_rows,
    )
    trade_rows = []
    for key in names:
        p = paths[("replay", key)]
        actions = p["controller_action"]
        trade_rows.append((
            names[key], int((actions == "reduce").sum()), int((actions == "restore").sum()),
            int(p["guarded"].sum()), percent(p.loc[p["scheduled"], "turnover"].mean()),
        ))
    trade_table = table(
        "创新读取的执行记录",
        ["策略", "触发减仓", "触发回补", "警戒状态交易日", "每周双边换手"], trade_rows,
    )
    fold_table = table(
        "年度扩展窗口",
        ["年度", "最后训练信号", "最后训练标签", "首个日度预测", "训练资产日", "冲击阈值", "幅度组长期门控"],
        [(r.year, r.last_training_signal, r.last_training_label, r.first_prediction, f"{r.training_rows:,}",
          number(r.downside_threshold), percent(r.impulse_long_write)) for r in folds.itertuples()],
    )
    independent = base.loc[base["phase"].eq("independent")].set_index("strategy")
    independent_table = abs_table(independent, "冻结后的独立评估") if not independent.empty else ""
    fingerprints = "".join(f"<li><code>{escape(name)} {digest}</code></li>" for name, digest in hashes.items())
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · T-MAVE 回撤研究</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF / T-MAVE</div><h1>双记忆如何影响组合回撤</h1>
<p class="meta">研究回放 {sample}　·　{now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#comparison">主实验</a>
<a href="#scenarios">情景回放</a><a href="#checks">稳健性</a><a href="#evidence">方法与复现</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>T-MAVE 组合的 CDaR 为 {percent(new.cdar95)}，原组合为 {percent(old.cdar95)}。</strong>
{gain_text}，净年化变化 {fmt_pp(return_gap)} 个百分点，{outcome}。
主比较的 95% 区间为 {bounds}，{interval_note}。</p>
<p>组合最大回撤由 {percent(old.max_drawdown)} 变为 {percent(new.max_drawdown)}，净夏普由 {number(old.sharpe)}
变为 {number(new.sharpe)}。回放中权益和现金的平均敞口差分别为 {fmt_pp(equity_gap)} 和
{fmt_pp(cash_gap)} 个百分点，两者最大绝对差{exposure_status} 2 个百分点的近似匹配范围。</p>
<p>{independent_text}</p></div></section>
<section id="comparison"><h2>主实验</h2>
{figure("returns_comparison.png", "四个 T-MAVE 消融策略的累计净收益与回撤", "图 1　已查看历史区间的连续净值，单边 5 bp。")}
{factorial}
<p>单换写入和单换读取分别改变 CDaR {change["memory"]} 和 {change["reading"]} 个百分点。
两项同时启用的交互项为 {change["interaction"]} 个百分点。证据门控相对只看冲击幅度的版本，
CDaR 变化 {change["evidence"]} 个百分点。上述差值为新策略减对应对照，负值对应回撤收窄。</p>
<p>现有记忆加创新读取的 CDaR 为 {percent(focused.cdar95)}，相对变化折算为改善 {percent(focused_gain)}，
净年化变化 {fmt_pp(focused_gap)} 个百分点。这是 2×2 消融观察，主比较仍是两项均启用的版本。</p>
{abs_table(replay, "全部策略 · 历史回放 · 5 bp")}
{independent_table}
<p class="note">最长水下交易日含未修复段。累计费用按期初净值计，年换手含买入及卖出。</p>
<details><summary>逐年结果与开发期表现</summary>{annual_table}{abs_table(develop,"开发期 · 用于校准")}</details>
<details><summary>八种策略的独立收益图</summary><div class="grid">{''.join(gallery)}</div></details></section>
<section id="scenarios"><h2>情景回放与换仓</h2>
<p>共有 {candidates} 个达到初始冲击门槛并留有完整观察窗口的日期。互不重叠且符合三类规则的窗口为
{scenarios["start"].nunique()} 个。分类使用事后路径，真实交易只在信号发生后按周执行。</p>
<p>{''.join(narrative)}</p>
{scenarios_table}{trade_table}
<p class="note">恢复指重新达到窗口首日前的组合净值。恢复天数从冲击日计为 0，十一日内未恢复的窗口计入恢复比例，
数值中位数只用已恢复窗口。实际警戒区间与调仓日可按明细中的 signal_date 追溯。</p></section>
<section id="checks"><h2>预测与稳健性</h2>{forecasts_table}{cost_table}
<details open><summary>配对不确定性与敞口校准</summary>{interval_table}{calibration_table}</details>
<p class="note">以 {BOOTSTRAP_REPS:,} 次交易日循环区块同时重抽样两条日收益序列，再重建净值路径和 CDaR。
抽样区间描述本段历史风险差的不确定性。开发期用平均收盘权益与现金敞口确定倍率。</p></section>
<section id="evidence"><h2>方法与复现</h2>
<dl>
<dt>写入</dt><dd>短期状态先按 5 日半衰期向前一日长期状态衰减，再在标准化下行冲击时有界写入。
长期状态按 30 日半衰期衰减，确认重复冲击或权益同步触发后提高长期写入，黄金和国债使用自身重复证据。
幅度对照用相同短期状态更新，在训练期匹配长期写入总量。</dd>
<dt>读取</dt><dd>动量篮子中的短期状态减去长期状态后取加权平均。
仅在周度信号上判定进入和退出警戒，使用校准期固定的两个阈值。
警戒时原波动目标仓位乘 0.75，所有策略持仓上限为 100%，次一可交易时点模拟开盘成交。</dd>
<dt>指标</dt><dd>CDaR 取最深 5% 日回撤的平均深度。预设经济意义门槛是 CDaR 相对改善 {percent(CDAR_GAIN,0)}，
年化净收益损失不超过 {MAX_CAGR_LOSS * 100:.0f} 个百分点。风险预测保留五日总方差，补测负对数收益平方均值，
后者用 MAE，仓位侧以半方差的两倍作方差近似。</dd>
<dt>情景</dt><dd>权益等权日收益低于前 60 日波动的 1.5 倍负值时起算。
后 5 日收复首日损失为急跌反弹。否则，后 10 日跌幅至少 1 倍波动且至少 6 日下跌为持续下跌。
其余窗口中，后 10 日至少 4 次方向反转且累计绝对变化至多 1.5 倍波动为震荡反复。</dd>
<dt>边界</dt><dd>历史回放日期曾用于首版结果解读。协议日期为 {FREEZE_DATE:%Y-%m-%d}。
分组样本数量有限。后复权权重法采用比例成本，权益敞口相近是组合口径，不能替代行业与因子风险匹配。</dd>
</dl>
<details><summary>训练边界与输入指纹</summary>{fold_table}<ul class="sources">{fingerprints}</ul></details>
<p>运行命令 <code>python -B -X utf8 chronoetf/backtest.py --tmave</code></p>
<p><a href="portfolio_metrics.csv">组合指标</a> · <a href="daily_backtest.csv">每日净值与订单状态</a> ·
<a href="risk_states.csv">逐 ETF 预测、证据与门控</a> · <a href="scenarios.csv">情景逐窗口明细</a> ·
<a href="https://www.math.columbia.edu/~chekhlov/ChekhlovUryasevZabarankin--2003.pdf">CDaR 方法出处</a></p>
</section><footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")


def strategy_year_review(performance, paths, core, policy, execution_audit=None):
    """从同一连续回放分年计量，并检查高夏普通常依赖的口径。"""
    keys = (core, policy, "guard_only", "vol_match")
    labels = {
        core: "120 日风险预算主干" if core == "budget_120" else "锁定风险预算主干",
        policy: "主干 + T-MAVE 三时钟",
        "guard_only": "主干 + T-MAVE 仅控仓",
        "vol_match": "开发期同波动对照",
    }
    annual = performance.loc[
        performance.phase.eq("replay") & performance.year.ne("all")
        & performance.one_way_bps.eq(BASE_COST) & performance.strategy.isin(keys)
    ]
    years = sorted(annual.year.unique())
    current = str(years[-1])
    tables = []
    for year in years:
        part = annual.loc[annual.year.eq(year)].set_index("strategy").loc[list(keys)]
        first = part.iloc[0]
        rows = [
            (labels[key], percent(r.total_return), percent(r.cagr), percent(r.max_drawdown),
             number(r.sharpe), percent(r.annual_vol), percent(r.cdar95))
            for key, r in part.iterrows()
        ]
        tables.append(table(
            f"{year} · {first.first_date} 至 {first.last_date} · {first.trading_days} 个交易日",
            ["模型", "区间净收益", "净年化", "最大回撤", "净夏普", "年化波动", "CDaR 95%"],
            rows, labels[policy], labels[core],
        ))

    sharpe_rows, contributions = [], []
    daily_hurdle = np.expm1(np.log1p(.02) / PER_YEAR)
    for key in keys:
        path = paths[("replay", key)]
        returns = path["net_return"].to_numpy()
        mean = returns.mean()
        centered = returns - mean
        ordinary = returns.std(ddof=1) / np.sqrt(PER_YEAR)
        lrv = centered @ centered / len(centered)
        for lag in range(1, 21):
            lrv += 2 * (1 - lag / 21) * (centered[lag:] @ centered[:-lag]) / len(centered)
        serial = mean * np.sqrt(PER_YEAR / lrv) if lrv > 0 else np.nan
        sharpe_rows.append((labels[key], number(mean / ordinary),
                            number((mean - daily_hurdle) / ordinary), number(serial)))

        year = path.loc[path.index.year == int(current)]
        nav_start = year["nav"].iloc[0] / (1 + year["net_return"].iloc[0])
        pnl = year[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee"]].sum() / nav_start
        contributions.append((labels[key], *(percent(v) for v in pnl), percent(pnl.sum())))

    sensitivity = table(
        "整个历史回放的夏普口径敏感性",
        ["模型", "原口径（无风险利率 0%）", "假设年无风险利率 2%", "20 日序列相关修正（无风险利率 0%）"],
        sharpe_rows,
    )
    attribution = table(
        f"{current} 年区间净收益归因 · 相对当年期初净值",
        ["模型", "股票贡献", "黄金贡献", "国债贡献", "交易费用", "合计"], contributions,
    )
    valuation_check = ""
    if execution_audit is not None:
        checks = table(
            "相同目标仓位 · 用公告复核价重新计价",
            ["模型", "全区间净夏普", f"{current} 净收益", f"{current} 最大回撤", f"{current} 净夏普"],
            [(labels[key], number(execution_audit[key]["all"]["sharpe"]),
              percent(execution_audit[key][current]["total_return"]),
              percent(execution_audit[key][current]["max_drawdown"]),
              number(execution_audit[key][current]["sharpe"])) for key in keys],
        )
        valuation_check = f"""{checks}
<p class="note">由腾讯原始报价和基金公告重建复核序列。在同一调仓目标和成本下重新撮合，检验价格口径对损益的影响。</p>"""
    return f"""<section id="annual"><h2>四条模型的逐年表现</h2>
<p>单边成本 5 bp。年度按同一条净值路径切片，跨年继续持仓。区间净收益和回撤从当年期初重新计量。
{current} 年截至 {annual.last_date.max()}。净年化采用 252 日换算，区间净收益为实际回放结果。</p>
{''.join(tables)}
{figure("returns_annual.png", "四条模型的分年累计净收益", "各年净值分别从零收益起算，三张图使用相同纵轴。")}
<h3>夏普 1.7 的复核</h3>
{sensitivity}
<p class="note">原口径按日净收益均值除以日样本标准差，再乘 √252。2% 是敏感性假设。
序列相关修正以 20 日 Bartlett 加权的长程方差计量，展示年化对日收益相关性的敏感程度。
<a href="https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios">Lo（2002）</a></p>
{valuation_check}
<h3>{current} 年收益来自哪里</h3>
{attribution}
<p class="note">股票、黄金与国债包含隔夜和日内损益。费用单列，贡献与当年净收益对账。</p></section>"""


def strategy_year_plot(folder, paths, core, policy):
    keys = (core, policy, "guard_only", "vol_match")
    labels = ("风险预算主干", "T-MAVE 三时钟", "T-MAVE 仅控仓", "同波动对照")
    years = sorted(paths[("replay", core)].index.year.unique())
    fonts()
    fig, panel = plt.subplots(1, len(years), figsize=(12, 4.6), sharey=True, squeeze=False)
    fig.subplots_adjust(left=.085, right=.98, bottom=.25, top=.83, wspace=.16)
    fig.text(.085, .93, "四条模型逐年累计净收益", fontsize=19, fontweight="bold")
    for year, ax in zip(years, panel[0]):
        for key, label, color, style in zip(keys, labels, PALETTE, ("-", "--", ":", "-.")):
            path = paths[("replay", key)]
            part = path.loc[path.index.year == year, "net_return"]
            linked = (1 + part).cumprod() - 1
            ax.plot(linked.index.insert(0, linked.index[0] - pd.Timedelta(days=1)),
                    np.r_[0, linked], label=label, color=color, linestyle=style, lw=1.5)
        ax.set_title(year, fontsize=13, fontweight="bold")
        ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0, color=GRAY, lw=.7)
        ax.grid(axis="y", color=GRID, lw=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0, pad=5)
    fig.legend(*panel[0, 0].get_legend_handles_labels(), ncols=4, loc="lower left",
               bbox_to_anchor=(.085, .1), frameon=False)
    cutoff = paths[("replay", core)].index[-1].date().isoformat()
    fig.text(.085, .045, f"单边成本 5 bp  |  数据截至 {cutoff}  |  各年以当年期初净值为基准",
             color=MUTED, fontsize=9)
    fig.savefig(folder / "returns_annual.png", dpi=170)
    plt.close(fig)


def write_strategy_report(folder, now, performance, paths, confidence, selection, states, scenarios,
                          candidates, choice, fresh, hashes):
    from strategy import LABELS
    from tmave import FEES, FREEZE_DATE, MIN_FRESH_SESSIONS

    phase_names = {"development": "开发期", "replay": "历史回放", "independent": "冻结后独立评估"}
    metric_names = {"net_sharpe": "净夏普", "cagr": "净年化", "cdar95": "CDaR 95%"}
    core, policy = choice["core"], choice["policy"]
    full = performance.loc[performance["year"].eq("all")]
    base = full.loc[full["one_way_bps"].eq(BASE_COST)]
    replay = base.loc[base.phase.eq("replay")].set_index("strategy")
    before, after = replay.loc[core], replay.loc[policy]
    scope = f"{before.first_date} 至 {before.last_date}"
    ranking = replay["sharpe"].idxmax()
    classic = replay.loc["inverse"]
    floor = replay.loc["bond"]
    match = replay.loc["vol_match"]
    plain_tilt, plain_guard = replay.loc["tilt_only"], replay.loc["guard_only"]

    def pp(value):
        return f"{value * 100:+.2f}"

    def recovery(row):
        return str(int(row.recovery_sessions)) if np.isfinite(row.recovery_sessions) else "期末未恢复"

    def portfolio_table(caption, values):
        return table(
            caption,
            ["策略", "累计收益", "净年化", "净夏普", "最大回撤", "CDaR 95%", "年化波动",
             "恢复交易日", "最长水下日", "双边年换手", "累计费用"],
            [(LABELS[key], percent(r.total_return), percent(r.cagr), number(r.sharpe), percent(r.max_drawdown),
              percent(r.cdar95), percent(r.annual_vol), recovery(r), r.drawdown_days,
              f"{r.annual_turnover_two_way:.2f}×", percent(r.fee_over_period_start_nav))
             for key, r in values.iterrows()],
            LABELS[policy], LABELS[core],
        )

    fonts()
    returns_plot(
        folder, paths, (core, policy, "inverse", "bond", "vol_match"), "returns_comparison.png",
        "成熟策略增强与归因对照", f"{scope}  |  {before.trading_days} 个交易日  |  历史回放", LABELS, "replay",
    )
    clock_keys = ("trade", "natural", "event", policy, "fixed")
    returns_plot(
        folder, paths, clock_keys, "returns_clocks.png", "时间分类对照",
        f"{scope}  |  同一主干、仓位参数与调仓时点", LABELS, "replay",
    )
    pictures = []
    for key, r in replay.iterrows():
        file = f"returns_{key}.png"
        returns_plot(
            folder, paths, (key,), file, LABELS[key],
            f"{scope}  |  净年化 {percent(r.cagr)}  ·  夏普 {r.sharpe:.2f}  ·  最大回撤 {percent(r.max_drawdown)}",
            LABELS, "replay",
        )
        pictures.append(figure(file, f"{LABELS[key]}累计收益与回撤", LABELS[key]))

    selected = table(
        "开发期选型 · 2021–2023 · 单边 5 bp",
        ["候选", "净夏普", "净年化", "最大回撤", "CDaR 95%", "Pareto 前沿", "共同改善", "进入主比较"],
        [(LABELS[r.strategy], number(r.sharpe), percent(r.cagr), percent(r.max_drawdown), percent(r.cdar95),
          "是" if r.pareto else "否", ("是" if r.joint_improvement else "否") if r.family == "enhancement" else "—",
          "是" if r.locked else "否") for r in selection.itertuples()],
        LABELS[policy], LABELS[core],
    )
    exposures = table(
        "历史回放 · 收盘平均权重与期初净值损益归因",
        ["策略", "股票", "黄金", "债券", "现金", "股票损益", "黄金损益", "债券损益", "费用"],
        [(LABELS[key], percent(r.mean_equity_weight), percent(r.mean_gold_weight), percent(r.mean_bond_weight),
          percent(r.mean_cash_weight), *[percent(paths[("replay", key)][f"pnl_{group}"].sum())
                                          for group in ("equity", "gold", "bond", "fee")])
         for key, r in replay.iterrows()], LABELS[policy], LABELS[core],
    )
    clocks_table = table(
        "时钟与模块消融 · 5 bp",
        ["策略", "净夏普", "净年化", "最大回撤", "CDaR 95%", "相对交易日版夏普差", "相对交易日版 CDaR 差（百分点）",
         "触发减仓", "触发回补", "警戒日"],
        [(LABELS[key], number(replay.at[key, "sharpe"]), percent(replay.at[key, "cagr"]),
          percent(replay.at[key, "max_drawdown"]), percent(replay.at[key, "cdar95"]),
          f"{replay.at[key, 'sharpe'] - replay.at['trade', 'sharpe']:+.3f}",
          pp(replay.at[key, "cdar95"] - replay.at["trade", "cdar95"]),
          int((paths[("replay", key)]["controller_action"] == "reduce").sum()),
          int((paths[("replay", key)]["controller_action"] == "restore").sum()),
          int(paths[("replay", key)]["guarded"].sum()))
         for key in (*clock_keys, "tilt_only", "guard_only", core)], LABELS[policy], LABELS[core],
    )
    clock_notes = " ".join(
        f"{label}的净夏普变化为 {replay.at[key, 'sharpe'] - replay.at['trade', 'sharpe']:+.3f}，"
        f"净年化变化为 {pp(replay.at[key, 'cagr'] - replay.at['trade', 'cagr'])} 个百分点，"
        f"CDaR 变化为 {pp(replay.at[key, 'cdar95'] - replay.at['trade', 'cdar95'])} 个百分点。"
        for key, label in (("natural", "追加自然日"), ("event", "追加事件时钟"), (policy, "三类时钟合用"))
    )
    tags = states.loc[states.phase.eq("replay")].copy()
    tags["abs_innovation"] = (tags["fast_full"] - tags["anchor_full"]).abs()
    activity = table(
        "时间状态与证据 · 资产日",
        ["分类", "样本数", "平均事件单位", "短期写入", "长期写入", "平均快慢差绝对值"],
        [(name, f"{len(rows):,}", number(rows.event_increment.mean()), percent(rows.fast_gate.mean()),
          percent(rows.slow_gate.mean()), number(rows.abs_innovation.mean()))
         for name, sub in (("信号静默", "quiet"), ("仅其他资产组活跃", "other_active"), ("单次冲击", "single"),
                           ("同向重复", "repeat"), ("权益共振", "co_move"))
         for rows in [tags.loc[tags.event_class.eq(sub)]]]
        + [("跨休市观测", f"{tags.cross_recess.sum():,}",
            number(tags.loc[tags.cross_recess.eq(1), "event_increment"].mean()),
            percent(tags.loc[tags.cross_recess.eq(1), "fast_gate"].mean()),
            percent(tags.loc[tags.cross_recess.eq(1), "slow_gate"].mean()),
            number(tags.loc[tags.cross_recess.eq(1), "abs_innovation"].mean()))],
    )
    fee_lookup = full.loc[full.phase.eq("replay")].set_index(["strategy", "one_way_bps"])
    fees = table(
        "成本敏感性 · 净年化 / 净夏普 / 最大回撤",
        ["策略", *(f"单边 {fee} bp" for fee in FEES)],
        [(LABELS[key], *[f"{percent(r.cagr)} / {r.sharpe:.3f} / {percent(r.max_drawdown)}" for fee in FEES
                         for r in [fee_lookup.loc[(key, fee)]]]) for key in replay.index], LABELS[policy], LABELS[core],
    )
    rows_20 = confidence.loc[confidence.block.eq(20)]

    def ci_value(value, measure):
        return f"{value:+.3f}" if measure == "net_sharpe" else f"{pp(value)} 个百分点"

    intervals = table(
        "增强减对照 · 20 日配对区块 · 95% 区间",
        ["期间 / 对照", "度量", "点估计", "下界", "上界"],
        [(f"{phase_names[r.phase]} / {LABELS[r.benchmark]}", metric_names[r.metric],
          *[ci_value(x, r.metric) for x in (r.difference, r.lower, r.upper)]) for r in rows_20.itertuples()],
    )
    strategy_year_plot(folder, paths, core, policy)
    annual = strategy_year_review(performance, paths, core, policy)
    cases = table(
        f"{scenarios['start'].nunique()} 个历史窗口的主要对照",
        ["情景 / 策略", "窗口", "平均最大回撤", "冲击后五日净收益", "11 日净收益", "窗口内恢复", "平均换手"],
        [(f"{scenario} / {LABELS[key]}", len(g), percent(g.max_drawdown.mean()), percent(g.rebounds_5d.mean()),
          percent(g.net_return.mean()), f"{g.recovered.sum()}/{len(g)}", percent(g.turnover.mean()))
         for (scenario, key), g in scenarios.loc[scenarios.strategy.isin([core, policy, "inverse", "vol_match"])]
         .groupby(["scenario", "strategy"], sort=False)],
    )
    sd = rows_20.query("phase == 'replay' and benchmark == @core and metric == 'net_sharpe'").iloc[0]
    ci_phrase = "区间跨零" if sd.lower <= 0 <= sd.upper else "区间未跨零"
    joint_status = "未出现三项同时改善的增强配置" if not choice["qualifiers"] else "至少一档配置满足开发期共同改善条件"
    available = base.loc[base.phase.eq("independent")].set_index("strategy")
    independent_view = portfolio_table("规则冻结后的独立评估", available) if len(available) else ""
    fwd = (
        f"冻结后可用交易日 {fresh}，在独立表中报告。" if len(available) else
        f"冻结后当前输入只有 {fresh} 个交易日，独立评估需至少 {MIN_FRESH_SESSIONS} 日。本轮结论来自历史回放。"
    )
    fingerprints = "".join(f"<li><code>{escape(name)} {digest}</code></li>" for name, digest in hashes.items())
    csv_list = {
        "portfolio_metrics": "组合指标", "selection": "候选与选型", "daily_backtest": "每日净值、仓位及费用",
        "time_states": "三类时间与事件状态", "scenarios": "情景明细",
        "paired_intervals": "配对区间", "clock_folds": "年度事件拟合",
    }
    links = " · ".join(f'<a href="{key}.csv">{title}</a>' for key, title in csv_list.items())
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF · 成熟策略上的 T-MAVE</title><style>{CSS}</style></head>
<body><header><div class="wrap"><div class="brand">CHRONOETF / T-MAVE</div><h1>成熟策略上的三类时间</h1>
<p class="meta">历史回放 {scope}　·　{now:%Y-%m-%d %H:%M:%S} 北京时间</p></div></header>
<nav aria-label="目录"><div class="wrap"><a href="#summary">摘要</a><a href="#portfolio">策略与选型</a><a href="#annual">逐年表现</a>
<a href="#clocks">时间贡献</a><a href="#robust">稳健性</a><a href="#methods">方法与明细</a></div></nav>
<main class="wrap"><section id="summary"><h2>摘要</h2><div class="summary">
<p><strong>{LABELS[core]}加上锁定的 T-MAVE 模块后，历史回放净夏普由 {before.sharpe:.3f} 变为 {after.sharpe:.3f}，
净年化由 {percent(before.cagr)} 变为 {percent(after.cagr)}，最大回撤由 {percent(before.max_drawdown)}
变为 {percent(after.max_drawdown)}。</strong> CDaR 由 {percent(before.cdar95)} 变为 {percent(after.cdar95)}。
配对净夏普差的 95% 区间为 [{sd.lower:+.3f}, {sd.upper:+.3f}]，{ci_phrase}。</p>
<p>同波动对照的净夏普为 {match.sharpe:.3f}，年化净收益为 {percent(match.cagr)}，最大回撤为 {percent(match.max_drawdown)}。
原逆波动净夏普 {classic.sharpe:.3f}，平均国债权重 {percent(classic.mean_bond_weight)}。国债单持净夏普 {floor.sharpe:.3f}。
全表最高净夏普来自{LABELS[ranking]}（{replay.at[ranking, "sharpe"]:.3f}）。</p>
<p>{fwd}</p></div></section>
<section id="portfolio"><h2>策略与选型</h2>
{figure("returns_comparison.png", "成熟策略增强组及传统策略累计收益和回撤", "图 1　历史回放的净累计收益与回撤，单边 5 bp。")}
<p>主干选型和增强强度只使用 2021–2023 年数据。开发期{joint_status}。
根据冻结的 Pareto 和排序规则锁定{LABELS[policy]}，下文保留这档配置及原主干。</p>
{selected}{portfolio_table("全部策略 · 历史回放 · 5 bp", replay)}
{independent_view}
<p class="note">最大回撤以负值表示，CDaR 以正深度表示。恢复日数从最深谷底到恢复峰值，起点采用阶段期初净值。</p>
<details><summary>全区间资产归因</summary>{exposures}
<p class="note">各资产损益以区间期初净值为基准，逐日隔夜、日内损益和费用相加，与期末组合收益对账。</p></details>
<details><summary>十九条策略的独立收益图</summary><div class="grid">{''.join(pictures)}</div></details>
</section>{annual}<section id="clocks"><h2>三类时间的增量</h2>
{figure("returns_clocks.png", "固定容量下五个时间配置的收益与回撤", "图 2　时钟变换使用相同主干、预算强度、交易规则与警戒阈值。")}
<p>{clock_notes}</p>
<p>仅倾斜风险预算时，净夏普为 {plain_tilt.sharpe:.3f}，最大回撤为 {percent(plain_tilt.max_drawdown)}。
仅保留风险仓位控制时，净夏普为 {plain_guard.sharpe:.3f}，最大回撤为 {percent(plain_guard.max_drawdown)}。
对照同波动组合的历史回放年化波动为 {percent(match.annual_vol)}，完整版为 {percent(after.annual_vol)}。</p>
{clocks_table}{activity}
<p class="note">同一资产日可以同时跨休市和触发事件。“跨休市观测”与事件分类分别记录。仅倾斜配置组保留观察信号，减仓乘数为 1。</p>
</section><section id="robust"><h2>成本、情景与区间估计</h2>
{fees}{intervals}{cases}
<p class="note">{candidates} 个历史候选冲击日，得到 {scenarios["start"].nunique()} 个互不重叠的 11 日窗口。
急跌反弹、持续下跌、震荡反复按事后权益路径分类；组合信号仅使用当时价格和已锁定参数。
采用 {BOOTSTRAP_REPS:,} 次配对循环区块重抽样，10 日和 40 日的敏感性结果可下载。</p>
</section><section id="methods"><h2>方法与复现</h2>
<dl>
<dt>策略来源</dt><dd>华安证券以协方差及风险贡献为基础研究放宽风险平价，引入预期收益改善收益风险权衡。
本实验以股票、黄金、国债三组求风险预算，并施加 65%/45%/70% 组目标权重上限。
华宝证券比较多段动量周期并使用下一日开盘成交。本实验使用 15/60/120 日风险调整动量调节预算。
原报告的样本资产、持仓期限、算法和业绩对应各自研究。</dd>
<dt>时间定义</dt><dd>每个交易日推进一单位，自然日按间隔 × 5/7 换算后与交易日等权混合。
有效事件为收盘后按事前波动标准化的收益冲击，同向重复和权益共振使当次事件单位增加，
其他资产组活动另计。事件流动只使用上一交易日已确认的事件，按折内历史均值归一，占混合时钟 25%。</dd>
<dt>状态与决策</dt><dd>快慢半衰期为 5/30 日，先衰减再写入。短期状态响应单次冲击，长期锚点响应确认的证据。
预算倾斜用 40% 快状态和 60% 锚点，组合警戒用风险贡献加权的下行创新，减仓和回补阈值分别为
{choice["entry"]:.4f} 和 {choice["exit"]:.4f}。波动匹配对照仅用开发期校准仓位倍数至 {choice["vol_scale"]:.4f}。</dd>
<dt>数据与假设</dt><dd>五只 ETF 的后复权开盘和收盘用于权重法，周度信号下个交易日开盘模拟成交。主干和增强共用组上限，原始逆波动与国债单持保留各自的敞口用于归因，国债单持只在期初买入。
CDaR 95% 为最深 5% 日回撤的平均深度。年化以 252 日为基准，现金和无风险利率设为零。
压力窗口、金额费用与独立检验边界以项目实验协议为准。冻结日为 {FREEZE_DATE:%Y-%m-%d}。</dd>
</dl>
<p>原始方法来源 <a href="https://pdf.dfcfw.com/pdf/H301_AP202509101741050066_1.pdf">华安证券研报</a> ·
<a href="https://pdf.dfcfw.com/pdf/H301_AP202412241641416114_1.pdf">华宝证券研报</a></p>
<p>运行命令 <code>python -B -X utf8 chronoetf/backtest.py --strategy</code></p>
<p>{links}</p>
<details><summary>五份数据的输入指纹</summary><ul class="sources">{fingerprints}</ul></details></section>
<footer>ChronoETF 研究　·　{escape(folder.name)}</footer></main></body></html>
"""
    (folder / "report.html").write_text(html, encoding="utf-8")
