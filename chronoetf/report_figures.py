"""Build the compact report's figures from archived evidence, without fitting models."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from report import fonts

INK = "#142D4E"
PRIMARY = "#0072B2"
ORANGE = "#D55E00"
BLUE = "#142D4E"
MUTED = "#526174"
GRID = "#E2E8F0"

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_assets"
OLD = ROOT / "results/20261116-163605"
RESIDUAL = ROOT / "results/20261116-190622"
TRANSFER = ROOT / "results/20261116-195751"
GRAY = "#8291A5"
LIGHT = "#F1F5FA"
MODELS = ["ewma", "calibrated", "joint_static_paired", "joint_fixed_paired",
          "joint_trade_paired", "price_paired", "joint_full_paired"]
NAMES = ["原始 EWMA", "校准 EWMA", "同输入静态", "固定双记忆", "仅交易日", "单价格记忆", "完整 T-MAVE"]
CHECKED_TEXT = []


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, text in CHECKED_TEXT:
        if patch.figure is fig:
            outer = patch.get_window_extent(renderer)
            inner = text.get_window_extent(renderer)
            assert outer.contains(inner.x0, inner.y0) and outer.contains(inner.x1, inner.y1), name
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)


def diagram(height):
    fig, ax = plt.subplots(figsize=(11.8, height))
    fig.subplots_adjust(left=.015, right=.985, top=.97, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, detail="", color=PRIMARY, size=12):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.013",
                          facecolor=LIGHT, edgecolor=color, linewidth=1.2)
    ax.add_patch(patch)
    text = ax.text(x+w/2, y+h/2, title + ("\n"+detail if detail else ""),
                   ha="center", va="center", fontsize=size, linespacing=1.7, color=INK)
    CHECKED_TEXT.append((patch, text))


def arrow(ax, a, b, color=GRAY):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.5, color=color, zorder=0))


def clean(ax, title="", grid="x"):
    ax.set_title(title, loc="left", fontsize=13, pad=12, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis=grid, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def architecture():
    fig, ax = diagram(4.5)
    box(ax,.015,.72,.20,.22,"行情与交易日历","OHLC · 缺失状态\n股票 / 黄金 / 国债")
    box(ax,.285,.72,.28,.22,"基础组合主干","60 日风险预算 · 组内逆波动\n资本上限：65% / 45% / 70%")
    box(ax,.645,.72,.33,.22,"仓位与账户","股金倍率 m · 国债目标 · 现金\n次日开盘交易 → 扣费 → 净值")
    arrow(ax,(.22,.83),(.277,.83)); arrow(ax,(.573,.83),(.637,.83))
    box(ax,.045,.065,.40,.42,"冻结配置路径 · full_75 / full_90",
        "价格压力 → 三时钟快慢状态\n60 日方差尺度读出 → 波动预测\n周度控制股金仓位",BLUE)
    box(ax,.535,.065,.43,.42,"当前残差研究路径",
        "价格压力 + 成熟误差 → 完整 T-MAVE\nEWMA 残差读出 → 历史收缩\n风险预警 / 开发期风险匹配配置",PRIMARY)
    arrow(ax,(.335,.71),(.245,.495),BLUE)
    arrow(ax,(.515,.71),(.74,.495),PRIMARY)
    ax.plot([.43,.43,.81,.81],[.495,.64,.64,.70],color=BLUE,lw=1.4)
    arrow(ax,(.81,.66),(.81,.71),BLUE)
    arrow(ax,(.91,.495),(.91,.71),PRIMARY)
    save(fig,"model_pipeline.png")


def research_timeline():
    fig, ax = diagram(2.05)
    stages=[("2015–2017","阈值与时间单位"),("2018–2020","向前预测积累"),
            ("2021–2023","预算与阈值开发"),("2024–2026","历史回放与迁移")]
    for i,(title,detail) in enumerate(stages):
        x=.015+i*.25
        box(ax,x,.30,.215,.50,title,detail,size=12)
        if i<3:arrow(ax,(x+.223,.55),(x+.242,.55))
    ax.text(.5,.08,"残差与迁移路径  |  逐年扩展训练  |  每组评价：145 周预测 / 683 日账户",ha="center",fontsize=11,color=MUTED)
    save(fig,"research_timeline.png")


def memory_diagram():
    fig, ax = diagram(4.7)
    for x,title,detail in [(.025,"价格压力","动量 · 下行风险 · 有符号冲击"),(.535,"成熟误差","已兑现五日风险 / 旧 EWMA 预测")]:
        box(ax,x,.77,.44,.18,title,detail,size=12)
        box(ax,x,.45,.44,.22,"每路：快状态 f + 慢锚点 a","先流动 → 后写入  |  半衰期 5 / 30",size=12)
        arrow(ax,(x+.22,.76),(x+.22,.68))
    box(ax,.025,.10,.95,.24,"两路分别执行：三时钟流动 + 事件确认写入",
        "时间：交易日 + 自然日 + 滞后一日事件时间\n门控：事件 → 快状态 0.50；同向重复 / 股票共振 → 慢锚点 0.18",BLUE,11)
    arrow(ax,(.245,.35),(.245,.44),BLUE);arrow(ax,(.755,.35),(.755,.44),BLUE)
    ax.text(.5,.025,"两路 × 快慢 = 4 个状态  →  加入 7 个静态特征  →  岭 QLIKE 读出 + 历史收缩",ha="center",fontsize=12,color=INK)
    save(fig,"tmave_memory.png")


def execution_timeline():
    fig, ax = diagram(2.4)
    stages=[("t 日收盘","更新状态\n生成预测与目标"),("t+1 日开盘","调仓成交\n按交易金额计费"),
            ("t+2 … t+5","持仓与账户\n逐日记录损益"),("t+6 日开盘","五段收益齐备\n标签成熟")]
    for i,(title,detail) in enumerate(stages):
        x=.015+i*.25
        box(ax,x,.37,.215,.55,title,detail,size=12)
        if i<3:arrow(ax,(x+.223,.65),(x+.242,.65))
    arrow(ax,(.275,.19),(.97,.19),BLUE)
    ax.text(.62,.035,"标签：t+1 → t+6 的五个开盘区间；年度训练读取此前已成熟标签",ha="center",fontsize=10.5,color=MUTED)
    save(fig,"execution_timeline.png")


def original_forecasts():
    data=pd.read_csv(RESIDUAL/"forecast_scores.csv")
    all_rows=data[(data.phase=="replay")&(data.subset=="all")].set_index("model")
    fig,(left,right)=plt.subplots(1,2,figsize=(11.8,4.2),gridspec_kw={"width_ratios":[1.2,1]})
    fig.subplots_adjust(left=.15,right=.97,top=.86,bottom=.19,wspace=.30)
    values=all_rows.loc[MODELS,"qlike"].to_numpy(); ys=np.arange(len(MODELS))
    left.barh(ys,values,color=[PRIMARY if i==6 else GRAY for i in ys],height=.62)
    left.set(yticks=ys,yticklabels=NAMES,xlim=(0,1.01),xlabel="QLIKE（越低越好）")
    left.invert_yaxis();clean(left,"全期预测损失 · 145 周")
    for y,v in zip(ys,values):left.text(v+.014,y,f"{v:.4f}",va="center",fontsize=10)
    years=["2024","2025","2026"]
    for m,label,color,marker in [("calibrated","校准 EWMA",ORANGE,"s"),("joint_full_paired","完整 T-MAVE",PRIMARY,"o")]:
        rows=data[(data.phase=="replay")&(data.model==m)].set_index("subset")
        right.plot(years,rows.loc[years,"qlike"],marker=marker,color=color,label=label,lw=2)
    right.set(ylim=(0,1),ylabel="年度 QLIKE");clean(right,"按年观察改善方向",grid="y")
    right.legend(frameon=False,loc="lower left",fontsize=10)
    fig.text(.15,.025,"原组合  |  2024-01-02 至 2026-10-30  |  2026 年为截至 10 月 30 日的子期",fontsize=9,color=MUTED)
    save(fig,"original_forecasts.png")


def original_errors():
    data=pd.read_csv(RESIDUAL/"forecast_scores.csv")
    data=data[(data.phase=="replay")&(data.subset=="all")].set_index("model")
    fig,axes=plt.subplots(1,2,figsize=(11.8,2.65))
    fig.subplots_adjust(left=.16,right=.95,top=.79,bottom=.18,wspace=.40)
    models=["ewma","calibrated","joint_full_paired"]; labels=["原始 EWMA","校准 EWMA","完整 T-MAVE"]
    for ax,col,title in zip(axes,["underestimate_twice","overestimate_twice"],["两倍低估次数 / 145 周","两倍高估次数 / 145 周"]):
        vals=np.rint(data.loc[models,col]*data.loc[models,"weeks"]).astype(int)
        ax.barh(labels,vals,color=[GRAY,ORANGE,PRIMARY],height=.55);ax.invert_yaxis()
        ax.set_xlim(0,65);clean(ax,title)
        for i,v in enumerate(vals):ax.text(v+1,i,str(v),va="center",fontsize=12)
    save(fig,"original_errors.png")


def frozen_metrics():
    d=pd.read_csv(OLD/"portfolio_metrics.csv")
    d=d[(d.phase=="replay")&(d.year=="all")&(d.one_way_bps==5)].set_index("strategy")
    keys=["budget_60","full_75","full_90"]; labels=["固定 60 日","强控波","温和控波"]
    fig,axes=plt.subplots(1,3,figsize=(11.8,2.6));fig.subplots_adjust(left=.10,right=.98,top=.78,bottom=.17,wspace=.42)
    for ax,col,title in zip(axes,["cagr","annual_vol","max_drawdown"],["净年化收益","年化波动","最大回撤深度"]):
        vals=d.loc[keys,col].abs()*100
        ax.barh(labels,vals,color=[BLUE,PRIMARY,ORANGE],height=.54);ax.invert_yaxis();clean(ax,title)
        ax.set_xlim(0,max(vals)*1.30);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
        for i,v in enumerate(vals):ax.text(v+max(vals)*.03,i,f"{v:.2f}%",va="center",fontsize=10)
    save(fig,"frozen_metrics.png")


def alerts():
    d=pd.read_csv(TRANSFER/"alert_scores.csv")
    d=d[(d.phase=="replay")&(d.year=="all")&(d.lane=="same_dev_budget")]
    fig,axes=plt.subplots(1,2,figsize=(11.8,3.25));fig.subplots_adjust(left=.07,right=.97,top=.79,bottom=.26,wspace=.20)
    for ax,basket,title in zip(axes,["style","sector"],["宽基 / 红利 · 高风险事件 28 次","行业 · 高风险事件 34 次"]):
        rows=d[d.basket==basket].set_index("model")
        for offset,m,label,color in [(-.18,"calibrated","校准 EWMA",ORANGE),(.18,"joint_full_paired","完整 T-MAVE",PRIMARY)]:
            vals=rows.loc[m,["tp","fp","fn"]].astype(int).to_numpy()
            bars=ax.bar(np.arange(3)+offset,vals,width=.32,color=color,label=label)
            ax.bar_label(bars,padding=3,fontsize=11)
        ax.set(xticks=range(3),xticklabels=["命中","误报","漏报"],ylim=(0,49),ylabel="次数")
        clean(ax,title,grid="y")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.015),ncols=2,frameon=False)
    save(fig,"transfer_alerts.png")


def economics():
    d=pd.read_csv(TRANSFER/"portfolio_metrics.csv")
    d=d[(d.phase=="replay")&(d.year=="all")&(d.one_way_bps==5)].set_index(["basket","strategy"])
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.3));fig.subplots_adjust(left=.11,right=.98,top=.82,bottom=.23,wspace=.27)
    for ax,col,title in zip(axes,["cagr","annual_vol","max_drawdown"],["净年化收益","年化波动","最大回撤深度"]):
        maximum=0
        for offset,strategy,label,color in [(-.18,"matched_calibrated_75","校准 EWMA",ORANGE),(.18,"matched_joint_full_paired_75","完整 T-MAVE",PRIMARY)]:
            vals=np.array([abs(d.loc[(b,strategy),col])*100 for b in ["original","style","sector"]])
            maximum=max(maximum,max(vals)); ys=np.arange(3)+offset
            ax.barh(ys,vals,height=.29,label=label,color=color)
            for y,v in zip(ys,vals):ax.text(v+.06,y,f"{v:.2f}%",va="center",fontsize=10)
        ax.set(yticks=range(3),yticklabels=["原组合","宽基 / 红利","行业"],xlim=(0,maximum*1.30))
        ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));clean(ax,title)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.035),ncols=2,frameon=False)
    save(fig,"matched_economics.png")


def uncertainty():
    orig=pd.read_csv(RESIDUAL/"forecast_tests.csv")
    orig=orig[(orig.phase=="replay")&(orig.benchmark=="calibrated")&(orig.block_weeks==4)].iloc[0]
    new=pd.read_csv(TRANSFER/"forecast_tests.csv")
    new=new[(new.scope=="new_pooled")&(new.benchmark=="calibrated")&(new.block_weeks==4)].iloc[0]
    fig,ax=plt.subplots(figsize=(11.8,2.5));fig.subplots_adjust(left=.17,right=.96,top=.73,bottom=.28)
    for y,row,color in [(1,orig,PRIMARY),(0,new,BLUE)]:
        ax.errorbar(row.difference,y,xerr=[[row.difference-row.lower],[row.upper-row.difference]],fmt="o",color=color,capsize=5,lw=2)
        ax.text(.20,y,f"Holm p={row.p_holm:.4f}",va="center",fontsize=11)
    ax.axvline(0,color=GRAY,ls="--",lw=1)
    ax.set(yticks=[1,0],yticklabels=["原组合","迁移两组同期均值"],ylim=(-.6,1.6),xlim=(-.08,.29),xlabel="校准 EWMA − 完整 T-MAVE 的 QLIKE（正值表示改善）")
    clean(ax,"预测增量与 95% 区间 · 4 周配对区块 / 2,000 次重采样")
    save(fig,"forecast_intervals.png")


def labeled_returns():
    """Direct labels on both panels; the CSV contains only plotted frozen paths."""
    data = pd.read_csv(OUT / "frozen_paths.csv", parse_dates=["date"])
    styles = {
        "budget_60": ("固定 60 日", BLUE, "-"),
        "full_75": ("T-MAVE 强控波", PRIMARY, "-"),
        "full_90": ("T-MAVE 温和控波", ORANGE, "-"),
        "ewma_75": ("EWMA · 75%", "#7A3E9D", "--"),
        "matched_75": ("同仓位静态 · 75%", "#52525B", "-.")}
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 5.7), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1], "hspace": .27})
    fig.subplots_adjust(left=.09, right=.76, top=.87, bottom=.13)
    fig.suptitle("冻结配置：净收益与回撤的连续路径", x=.09, ha="left", fontsize=17, fontweight="bold")
    end = data.date.max()
    for ax, col, title in zip(axes, ["nav", "drawdown"], ["累计净收益", "回撤"]):
        endpoints = []
        for key, (label, color, linestyle) in styles.items():
            rows = data[data.strategy == key].sort_values("date")
            values = rows[col].to_numpy() - (1 if col == "nav" else 0)
            dates = rows.date.to_list()
            ax.plot([dates[0]-pd.Timedelta(days=1), *dates], np.r_[0, values],
                    color=color, ls=linestyle, lw=1.65)
            ax.plot(end, values[-1], "o", ms=3.5, color=color)
            endpoints.append((values[-1], label, color))
        clean(ax, grid="y")
        ax.set_ylabel(title)
        ax.set_xlim(data.date.min()-pd.Timedelta(days=8), end+pd.Timedelta(days=10))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axhline(0, color=GRAY, lw=.6)
        # Spread label positions in axis coordinates, preserving endpoint order.
        lo, hi = ax.get_ylim()
        points = sorted(endpoints)
        target = np.array([(v-lo)/(hi-lo) for v, _, _ in points])
        gap = .12 if col == "nav" else .18
        positions = np.clip(target, .06, .94)
        for i in range(1, len(positions)):
            positions[i] = max(positions[i], positions[i-1]+gap)
        if positions[-1] > .94:
            positions -= positions[-1]-.94
        for (value, label, color), y in zip(points, positions):
            ax.annotate(label, xy=(end, value), xycoords="data", xytext=(1.055,y),
                        textcoords="axes fraction", ha="left", va="center", fontsize=10,
                        color=color, annotation_clip=False,
                        arrowprops={"arrowstyle":"-", "color":color, "lw":.9,
                                    "connectionstyle":"angle,angleA=0,angleB=90,rad=0"})
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.text(.09,.025,"2024-01-02 至 2026-10-30  |  683 个交易日  |  单边 5 bp  |  标签对应各曲线末端",fontsize=9,color=MUTED)
    save(fig,"frozen_returns_labeled.png")


def transfer_forecasts():
    data = pd.read_csv(TRANSFER / "forecast_scores.csv")
    models = ["calibrated","lag_paired","rolling_paired","exponential_paired",
              "price_paired","joint_trade_paired","joint_full_paired"]
    labels = ["校准 EWMA","固定滞后","滚动平均","指数平均","单价格记忆","仅交易日","完整 T-MAVE"]
    fig,axes = plt.subplots(1,2,figsize=(11.8,4.6))
    fig.subplots_adjust(left=.14,right=.97,top=.86,bottom=.17,wspace=.42)
    for ax,basket,title in zip(axes,["style","sector"],["宽基 / 红利组","消费 / 医药 / 能源组"]):
        rows=data[(data.phase=="replay")&(data.subset=="all")&(data.basket==basket)].set_index("model")
        vals=rows.loc[models,"qlike"].to_numpy()
        ax.barh(labels,vals,height=.62,color=[ORANGE,*([GRAY]*5),PRIMARY])
        ax.set(xlim=(0,1.02),xlabel="QLIKE（越低越好）");ax.invert_yaxis();clean(ax,title)
        for i,v in enumerate(vals):ax.text(v+.013,i,f"{v:.4f}",va="center",fontsize=10)
    save(fig,"transfer_forecasts.png")


def palette_reference():
    fig,ax=diagram(3.5)
    palettes=[("金融研报 · 已采用",["#142D4E","#0072B2","#D55E00","#FFFFFF"]),
              ("学术简洁",["#222222","#2459A6","#B33040","#FFFFFF"]),
              ("现代清爽",["#38245E","#7156A5","#E58B30","#FFFFFF"])]
    for row,(title,colors) in enumerate(palettes):
        y=.72-row*.29
        ax.text(.025,y+.08,title,fontsize=13,va="center",fontweight="bold")
        for j,color in enumerate(colors):
            x=.32+j*.17
            ax.add_patch(FancyBboxPatch((x,y),.13,.17,boxstyle="round,pad=0.005",facecolor=color,edgecolor=GRID))
            ax.text(x+.065,y-.065,color,ha="center",fontsize=10,color=INK)
    save(fig,"palette_reference.png")


def main():
    fonts()
    plt.rcParams.update({"axes.facecolor":"white","figure.facecolor":"white","font.size":11,"text.color":INK,"axes.labelcolor":INK,
                         "xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":GRID})
    OUT.mkdir(exist_ok=True)
    for draw in [architecture,research_timeline,memory_diagram,execution_timeline,original_forecasts,
                 original_errors,frozen_metrics,alerts,economics,uncertainty,
                 labeled_returns,transfer_forecasts,palette_reference]:
        draw()
    print(f"Rendered 12 report figures and a palette reference to {OUT}")


if __name__=="__main__":
    main()
