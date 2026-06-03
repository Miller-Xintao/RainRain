# -*- coding: utf-8 -*-
"""
美股下跌概率与涨跌关联分析 (US equity decline-probability & up/down dependence analysis)

数据来源 (committed under data/):
  1) data/sp500_daily_1927_2019.csv         —— 标普500日线 OHLC, 1927-12-30 ~ 2019-12-23
     (fja05680/dow-sp500-100-years, 源自 Yahoo Finance)
  2) data/shiller_sp500_monthly_1871_2026.csv —— Robert Shiller 月度数据, 1871-01 ~ 2026-05
     (datasets/s-and-p-500)

输出:
  - 控制台打印全部统计量
  - figures/*.png 图表
  - results.json 机器可读结果 (供 report 引用)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
RES = {}


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_daily():
    df = pd.read_csv("data/sp500_daily_1927_2019.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret"] = df["Close"].pct_change()
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    # 剔除 0 成交量的极早期(1927-1962 收盘=开盘的合成值不影响收益率, 保留)
    return df


def load_monthly():
    df = pd.read_csv("data/shiller_sp500_monthly_1871_2026.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"SP500": "price"})
    df = df[df["price"] > 0].copy()
    df["ret"] = df["price"].pct_change()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    return df.dropna(subset=["ret"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. 日线: 基础下跌概率
# --------------------------------------------------------------------------- #
def daily_base(df):
    r = df["ret"].values
    n = len(r)
    up = (r > 0).sum()
    down = (r < 0).sum()
    flat = (r == 0).sum()
    out = {
        "n_days": int(n),
        "start": str(df["Date"].iloc[0].date()),
        "end": str(df["Date"].iloc[-1].date()),
        "p_up": up / n,
        "p_down": down / n,
        "p_flat": flat / n,
        "mean_ret": float(r.mean()),
        "median_ret": float(np.median(r)),
        "std_ret": float(r.std(ddof=1)),
        "mean_up": float(r[r > 0].mean()),
        "mean_down": float(r[r < 0].mean()),
        "skew": float(pd.Series(r).skew()),
        "kurtosis": float(pd.Series(r).kurtosis()),  # 超额峰度
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
        "worst_date": str(df.loc[df["ret"].idxmin(), "Date"].date()),
        "best_date": str(df.loc[df["ret"].idxmax(), "Date"].date()),
    }
    # 各跌幅阈值的发生频率
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07]
    out["p_decline_ge"] = {f"{t*100:.1f}%": float((r <= -t).mean()) for t in thresholds}
    out["p_gain_ge"] = {f"{t*100:.1f}%": float((r >= t).mean()) for t in thresholds}
    # 跌幅 >= x% 时, 平均多久发生一次(交易日)
    out["one_in_N_days_decline"] = {
        f"{t*100:.1f}%": (None if (r <= -t).sum() == 0 else n / int((r <= -t).sum()))
        for t in thresholds
    }
    RES["daily_base"] = out
    return out


# --------------------------------------------------------------------------- #
# 2. 涨跌关联: 自相关 / 条件概率 / 连涨连跌
# --------------------------------------------------------------------------- #
def daily_dependence(df):
    r = df["ret"].values
    sign = np.sign(r)
    n = len(r)

    # lag-1 ~ lag-5 自相关
    s = pd.Series(r)
    autocorr = {f"lag{k}": float(s.autocorr(lag=k)) for k in range(1, 6)}

    # 符号自相关 (方向的持续性)
    ssign = pd.Series(sign)
    sign_autocorr = {f"lag{k}": float(ssign.autocorr(lag=k)) for k in range(1, 6)}

    # 条件概率: 今天涨跌 | 昨天涨跌
    prev = sign[:-1]
    cur = sign[1:]
    def cond(prev_state, cur_state):
        mask = prev == prev_state
        if mask.sum() == 0:
            return None
        return float((cur[mask] == cur_state).mean())
    cond_tbl = {
        "P(down | prev down)": cond(-1, -1),
        "P(up   | prev down)": cond(-1, 1),
        "P(down | prev up)":   cond(1, -1),
        "P(up   | prev up)":   cond(1, 1),
    }
    p_down_uncond = (r < 0).mean()
    p_up_uncond = (r > 0).mean()

    # 波动率聚集: |今日收益| 与 |昨日收益| 的相关
    abs_autocorr = pd.Series(np.abs(r)).autocorr(lag=1)
    # 大跌(<=-2%)次日的平均收益 与 大涨(>=2%)次日的平均收益
    big_down_idx = np.where(r <= -0.02)[0]
    big_down_idx = big_down_idx[big_down_idx < n - 1]
    big_up_idx = np.where(r >= 0.02)[0]
    big_up_idx = big_up_idx[big_up_idx < n - 1]
    after_big_down = r[big_down_idx + 1]
    after_big_up = r[big_up_idx + 1]

    # 连涨连跌 (streak / runs)
    streaks_down = []
    streaks_up = []
    cur_len = 1
    for i in range(1, n):
        if sign[i] == sign[i - 1] and sign[i] != 0:
            cur_len += 1
        else:
            if sign[i - 1] < 0:
                streaks_down.append(cur_len)
            elif sign[i - 1] > 0:
                streaks_up.append(cur_len)
            cur_len = 1
    streaks_down = np.array(streaks_down)
    streaks_up = np.array(streaks_up)

    # 给定已连跌 k 天, 第 k+1 天继续跌的概率
    def continue_prob(streaks, k):
        # P(streak length > k | streak length >= k)
        ge_k = (streaks >= k).sum()
        gt_k = (streaks > k).sum()
        return None if ge_k == 0 else float(gt_k / ge_k)

    out = {
        "return_autocorr": autocorr,
        "sign_autocorr": sign_autocorr,
        "abs_return_autocorr_lag1": float(abs_autocorr),
        "p_down_unconditional": float(p_down_uncond),
        "p_up_unconditional": float(p_up_uncond),
        "conditional": cond_tbl,
        "lift_down_after_down": float(cond(-1, -1) / p_down_uncond),
        "mean_ret_after_big_down(<=-2%)": float(after_big_down.mean()),
        "mean_ret_after_big_up(>=2%)": float(after_big_up.mean()),
        "p_down_after_big_down": float((after_big_down < 0).mean()),
        "streak_down_mean": float(streaks_down.mean()),
        "streak_up_mean": float(streaks_up.mean()),
        "streak_down_max": int(streaks_down.max()),
        "streak_up_max": int(streaks_up.max()),
        "P(continue down | already down k days)": {
            f"k={k}": continue_prob(streaks_down, k) for k in [1, 2, 3, 4, 5]
        },
        "P(continue up | already up k days)": {
            f"k={k}": continue_prob(streaks_up, k) for k in [1, 2, 3, 4, 5]
        },
    }
    RES["daily_dependence"] = out
    return out, streaks_down, streaks_up, r


# --------------------------------------------------------------------------- #
# 3. 多周期持有: 下跌概率随持有期变化
# --------------------------------------------------------------------------- #
def horizon_decline(df):
    close = df["Close"].values
    out = {}
    horizons = {"1D": 1, "1W(5d)": 5, "1M(21d)": 21,
                "1Q(63d)": 63, "6M(126d)": 126, "1Y(252d)": 252}
    for name, h in horizons.items():
        if h >= len(close):
            continue
        fwd = close[h:] / close[:-h] - 1.0
        out[name] = {
            "p_decline": float((fwd < 0).mean()),
            "p_gain": float((fwd > 0).mean()),
            "mean": float(fwd.mean()),
            "median": float(np.median(fwd)),
            "p5": float(np.percentile(fwd, 5)),
            "p95": float(np.percentile(fwd, 95)),
        }
    RES["horizon_decline"] = out
    return out


# --------------------------------------------------------------------------- #
# 4. 回撤 / 修正 / 熊市占比
# --------------------------------------------------------------------------- #
def drawdown(df):
    close = df["Close"].values
    running_max = np.maximum.accumulate(close)
    dd = close / running_max - 1.0
    out = {
        "max_drawdown": float(dd.min()),
        "max_drawdown_date": str(df.loc[int(np.argmin(dd)), "Date"].date()),
        "pct_time_in_correction(<=-10%)": float((dd <= -0.10).mean()),
        "pct_time_in_bear(<=-20%)": float((dd <= -0.20).mean()),
        "pct_time_at_within_5pct_of_high": float((dd >= -0.05).mean()),
        "pct_time_below_high": float((dd < 0).mean()),
    }
    RES["drawdown"] = out
    return dd, out


# --------------------------------------------------------------------------- #
# 5. 月度 & 年度下跌概率 (Shiller 1871-2026)
# --------------------------------------------------------------------------- #
def monthly_annual(dm):
    rm = dm["ret"].values
    out = {
        "n_months": int(len(rm)),
        "start": str(dm["Date"].iloc[0].date()),
        "end": str(dm["Date"].iloc[-1].date()),
        "p_down_month": float((rm < 0).mean()),
        "p_up_month": float((rm > 0).mean()),
        "mean_month": float(rm.mean()),
    }
    # 月度下跌概率的日历季节性
    by_month = dm.groupby("month")["ret"].apply(lambda x: (x < 0).mean())
    out["p_down_by_calendar_month"] = {int(k): float(v) for k, v in by_month.items()}
    mean_by_month = dm.groupby("month")["ret"].mean()
    out["mean_ret_by_calendar_month"] = {int(k): float(v) for k, v in mean_by_month.items()}

    # 年度: 用每年最后一个观测的价格算年收益
    yearly = dm.groupby("year")["price"].last()
    yret = yearly.pct_change().dropna()
    out["n_years"] = int(len(yret))
    out["p_down_year"] = float((yret < 0).mean())
    out["p_up_year"] = float((yret > 0).mean())
    out["mean_year"] = float(yret.mean())
    out["worst_year"] = {"year": int(yret.idxmin()), "ret": float(yret.min())}
    out["best_year"] = {"year": int(yret.idxmax()), "ret": float(yret.max())}

    # 月度方向自相关 (动量/反转)
    out["month_return_autocorr_lag1"] = float(pd.Series(rm).autocorr(lag=1))
    sm = np.sign(rm)
    prev, cur = sm[:-1], sm[1:]
    mask = prev < 0
    out["P(down month | prev down month)"] = float((cur[mask] < 0).mean())
    RES["monthly_annual"] = out
    return out, yret


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def make_figures(df, r, streaks_down, dd, dm, yret, ma):
    # Fig1: 日收益分布
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(r * 100, bins=200, color="#4C72B0", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_title(f"S&P500 Daily Return Distribution ({RES['daily_base']['start']} ~ {RES['daily_base']['end']})")
    ax.set_xlabel("Daily return (%)"); ax.set_ylabel("Frequency (days)")
    ax.text(0.02, 0.95, f"P(down)={RES['daily_base']['p_down']*100:.1f}%\n"
                        f"P(up)={RES['daily_base']['p_up']*100:.1f}%\n"
                        f"skew={RES['daily_base']['skew']:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout(); fig.savefig("figures/fig1_daily_return_dist.png", dpi=110); plt.close(fig)

    # Fig2: 跌幅阈值发生频率
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(RES["daily_base"]["p_decline_ge"].keys())
    dvals = [RES["daily_base"]["p_decline_ge"][k] * 100 for k in keys]
    gvals = [RES["daily_base"]["p_gain_ge"][k] * 100 for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, dvals, 0.4, label="Decline ≥ x", color="#C44E52")
    ax.bar(x + 0.2, gvals, 0.4, label="Gain ≥ x", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylabel("% of trading days"); ax.set_title("Daily move magnitude: decline vs gain frequency")
    for i, v in enumerate(dvals): ax.text(i - 0.2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(gvals): ax.text(i + 0.2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.legend(); fig.tight_layout(); fig.savefig("figures/fig2_threshold_freq.png", dpi=110); plt.close(fig)

    # Fig3: 条件概率 (涨跌关联)
    fig, ax = plt.subplots(figsize=(8, 5))
    c = RES["daily_dependence"]["conditional"]
    labels = list(c.keys()); vals = [c[k] * 100 for k in labels]
    base = RES["daily_dependence"]["p_down_unconditional"] * 100
    colors = ["#C44E52" if "down" in l.split("|")[0] else "#55A868" for l in labels]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(base, color="gray", ls="--", label=f"Uncond. P(down)={base:.1f}%")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Probability (%)"); ax.set_title("Conditional probability of today's direction given yesterday")
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/fig3_conditional.png", dpi=110); plt.close(fig)

    # Fig4: 持有期下跌概率
    fig, ax = plt.subplots(figsize=(9, 5))
    h = RES["horizon_decline"]
    names = list(h.keys()); pdec = [h[n]["p_decline"] * 100 for n in names]
    ax.plot(range(len(names)), pdec, "o-", color="#C44E52", lw=2)
    for i, v in enumerate(pdec): ax.text(i, v + 0.6, f"{v:.1f}%", ha="center")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("P(decline) %"); ax.set_ylim(0, 55)
    ax.set_title("Probability of a negative return vs holding horizon")
    fig.tight_layout(); fig.savefig("figures/fig4_horizon.png", dpi=110); plt.close(fig)

    # Fig5: 回撤序列
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(df["Date"], dd * 100, 0, color="#C44E52", alpha=0.7)
    ax.set_ylabel("Drawdown from peak (%)")
    ax.set_title("S&P500 drawdown from running peak (1927–2019)")
    fig.tight_layout(); fig.savefig("figures/fig5_drawdown.png", dpi=110); plt.close(fig)

    # Fig6: 月度日历季节性下跌概率
    fig, ax = plt.subplots(figsize=(9, 5))
    pm = ma["p_down_by_calendar_month"]
    months = list(range(1, 13)); vals = [pm[m] * 100 for m in months]
    colors = ["#C44E52" if v > 50 else "#55A868" for v in vals]
    ax.bar(months, vals, color=colors)
    ax.axhline(ma["p_down_month"] * 100, color="gray", ls="--",
               label=f"Avg P(down month)={ma['p_down_month']*100:.1f}%")
    ax.set_xticks(months); ax.set_xlabel("Calendar month"); ax.set_ylabel("P(down) %")
    ax.set_title("Monthly decline probability by calendar month (Shiller 1871–2026)")
    for i, v in zip(months, vals): ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend(); fig.tight_layout(); fig.savefig("figures/fig6_seasonality.png", dpi=110); plt.close(fig)

    # Fig7: 连跌天数分布 & 续跌概率衰减
    fig, ax = plt.subplots(figsize=(9, 5))
    cont = RES["daily_dependence"]["P(continue down | already down k days)"]
    ks = list(cont.keys()); cv = [cont[k] * 100 for k in ks]
    ax.plot(range(1, len(ks) + 1), cv, "s-", color="#C44E52", lw=2)
    ax.axhline(RES["daily_base"]["p_down"] * 100, color="gray", ls="--",
               label=f"Base P(down)={RES['daily_base']['p_down']*100:.1f}%")
    ax.set_xticks(range(1, len(ks) + 1)); ax.set_xticklabels([k.replace('k=', 'after ') + 'd' for k in ks])
    ax.set_ylabel("P(next day also down) %")
    ax.set_title("Does a down-streak predict another down day?")
    for i, v in enumerate(cv): ax.text(i + 1, v, f"{v:.1f}", ha="center", va="bottom")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/fig7_streak_continue.png", dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
def p(title): print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)

def main():
    df = load_daily()
    dm = load_monthly()

    p("1. 日线基础下跌概率 (S&P500, %s ~ %s)" % (str(df['Date'].iloc[0].date()), str(df['Date'].iloc[-1].date())))
    b = daily_base(df)
    print(f"交易日数: {b['n_days']:,}")
    print(f"P(下跌)={b['p_down']*100:.2f}%  P(上涨)={b['p_up']*100:.2f}%  P(平)={b['p_flat']*100:.3f}%")
    print(f"日均收益={b['mean_ret']*100:.4f}%  中位数={b['median_ret']*100:.4f}%  日波动={b['std_ret']*100:.3f}%")
    print(f"上涨日均涨幅={b['mean_up']*100:.3f}%  下跌日均跌幅={b['mean_down']*100:.3f}% (跌时一次跌更多 -> 负偏)")
    print(f"偏度={b['skew']:.3f}  超额峰度={b['kurtosis']:.2f} (肥尾)")
    print(f"最差单日={b['worst_day']*100:.2f}% ({b['worst_date']})  最好单日={b['best_day']*100:.2f}% ({b['best_date']})")
    print("跌幅 >= 阈值 的发生概率 / 平均N个交易日一次:")
    for k in b["p_decline_ge"]:
        oneN = b["one_in_N_days_decline"][k]
        print(f"  跌 >= {k:>5}: {b['p_decline_ge'][k]*100:6.3f}%   约每 {oneN:6.1f} 个交易日一次")

    p("2. 涨跌关联 (自相关 / 条件概率 / 连涨连跌)")
    d, sd, su, r = daily_dependence(df)
    print("收益率自相关:", {k: round(v, 4) for k, v in d["return_autocorr"].items()})
    print("方向(符号)自相关:", {k: round(v, 4) for k, v in d["sign_autocorr"].items()})
    print(f"|收益|的lag1自相关={d['abs_return_autocorr_lag1']:.4f}  <- 波动率聚集 (大波动后接大波动)")
    print(f"无条件 P(跌)={d['p_down_unconditional']*100:.2f}%")
    for k, v in d["conditional"].items():
        print(f"  {k} = {v*100:.2f}%")
    print(f"昨跌->今跌 相对无条件的 lift = {d['lift_down_after_down']:.3f} (≈1 表示几乎独立)")
    print(f"大跌(<=-2%)次日均值={d['mean_ret_after_big_down(<=-2%)']*100:.3f}% , "
          f"大涨(>=2%)次日均值={d['mean_ret_after_big_up(>=2%)']*100:.3f}%")
    print(f"连跌均值={d['streak_down_mean']:.2f}天(最长{d['streak_down_max']})  "
          f"连涨均值={d['streak_up_mean']:.2f}天(最长{d['streak_up_max']})")
    print("已连跌k天后, 次日继续跌概率:", {k: round(v*100, 1) for k, v in d["P(continue down | already down k days)"].items()})

    p("3. 持有期 vs 下跌概率")
    h = horizon_decline(df)
    for name, v in h.items():
        print(f"  {name:>10}: P(亏损)={v['p_decline']*100:5.1f}%  均值={v['mean']*100:7.2f}%  "
              f"5分位={v['p5']*100:7.2f}%  95分位={v['p95']*100:7.2f}%")

    p("4. 回撤 / 修正 / 熊市 时间占比")
    dd, dout = drawdown(df)
    print(f"史上最大回撤={dout['max_drawdown']*100:.1f}% (谷底 {dout['max_drawdown_date']})")
    print(f"处于修正(>=10%回撤)的时间占比={dout['pct_time_in_correction(<=-10%)']*100:.1f}%")
    print(f"处于熊市(>=20%回撤)的时间占比={dout['pct_time_in_bear(<=-20%)']*100:.1f}%")
    print(f"距离前高5%以内的时间占比={dout['pct_time_at_within_5pct_of_high']*100:.1f}%  低于前高的时间占比={dout['pct_time_below_high']*100:.1f}%")

    p("5. 月度 & 年度下跌概率 (Shiller 1871-2026)")
    ma, yret = monthly_annual(dm)
    print(f"月观测={ma['n_months']:,} ({ma['start']}~{ma['end']})  P(月跌)={ma['p_down_month']*100:.2f}%  月均={ma['mean_month']*100:.3f}%")
    print(f"年观测={ma['n_years']}  P(年跌)={ma['p_down_year']*100:.2f}%  年均={ma['mean_year']*100:.2f}%")
    print(f"最差年={ma['worst_year']['year']}({ma['worst_year']['ret']*100:.1f}%)  最好年={ma['best_year']['year']}({ma['best_year']['ret']*100:.1f}%)")
    print("各日历月 P(下跌)%:", {m: round(v*100, 1) for m, v in ma["p_down_by_calendar_month"].items()})
    print(f"月度方向自相关lag1={ma['month_return_autocorr_lag1']:.4f}  P(本月跌|上月跌)={ma['P(down month | prev down month)']*100:.1f}%")

    make_figures(df, r, sd, dd, dm, yret, ma)
    with open("results.json", "w") as f:
        json.dump(RES, f, indent=2, ensure_ascii=False)
    print("\n[OK] 结果已写入 results.json, 图表已写入 figures/")


if __name__ == "__main__":
    main()
