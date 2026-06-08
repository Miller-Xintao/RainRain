# -*- coding: utf-8 -*-
"""验证'现在加仓 vs 等暴跌再加仓'的结论稳健性。
检查三类潜在缺陷:
  1) 等待期间现金按0%计 -> 改为现金计息(0/2/4/6%年化), 看结论是否翻转
  2) 结论是否被大萧条(1929-39)单一极端期主导 -> 分子区间复算
  3) 独立复核基础数字(P跌/年化漂移), 确认数据没读错
回测B采用同一日历终点的'终值'口径, 公平比较'全程在市' vs '先持币(计息)后入市'。
"""
import numpy as np, pandas as pd
df = pd.read_csv("data/sp500_daily_1927_2019.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
c = df["Close"].values; dts = df["Date"].values

def base_checks(c):
    r = c[1:]/c[:-1]-1.0
    yrs = (len(c))/252
    cagr = (c[-1]/c[0])**(1/yrs)-1
    print(f"  交易日={len(c):,}  P(跌)={(r<0).mean()*100:.2f}%  日均={r.mean()*100:.4f}%  年化CAGR={cagr*100:.2f}%")

def run_window_test(c, X, r_cash_ann, T=252*3, step=21):
    """返回 (立即均值, 等待均值, 等待跑赢比例, 踏空次数, n)。等待期现金按 r_cash_ann 年化计息。"""
    daily_cash = (1+r_cash_ann)**(1/252)-1
    n=len(c); imm=[]; wait=[]; never=0
    for s in range(0, n-T, step):
        seg=c[s:s+T+1]
        imm.append(seg[-1]/seg[0]-1.0)
        seg_dd = seg/np.maximum.accumulate(seg)-1.0
        hit=np.where(seg_dd<=-X)[0]
        if len(hit)==0:
            # 全程持币计息
            wait.append((1+daily_cash)**T - 1.0); never+=1
        else:
            b=hit[0]
            cash_part=(1+daily_cash)**b          # 等待b天的现金增值
            mkt_part=seg[-1]/seg[b]              # 入市后市场增值
            wait.append(cash_part*mkt_part - 1.0)
    imm=np.array(imm); wait=np.array(wait)
    return imm.mean(), wait.mean(), (wait>imm).mean(), never, len(imm)

periods = {
    "全样本 1928-2019": (None, None),
    "战后 1950-2019": ("1950-01-01", None),
    "现代 1990-2019": ("1990-01-01", None),
    "剔除大萧条 1940-2019": ("1940-01-01", None),
}
print("="*78)
print("基础数字独立复核:")
for name,(lo,hi) in periods.items():
    m = np.ones(len(df),bool)
    if lo: m &= (df["Date"]>=lo).values
    sub=c[m]; print(f"  {name:<22}", end=""); base_checks(sub)

print("\n"+"="*78)
print("结论稳健性: 【立即投资】 vs 【等回调X%再买(现金计息)】  —— 3年窗口, 终值口径")
print("数值=平均终期收益; 看'等待'是否能反超'立即'")
print("="*78)
for name,(lo,hi) in periods.items():
    m=np.ones(len(df),bool)
    if lo: m&=(df["Date"]>=lo).values
    sub=c[m]
    print(f"\n--- {name} (n_days={len(sub):,}) ---")
    for X in [0.10, 0.20]:
        print(f"  等回调{int(X*100)}%:")
        for rc in [0.00, 0.02, 0.04, 0.06]:
            im,wa,win,nev,N = run_window_test(sub, X, rc)
            flag = "←等待反超!" if wa>im else ""
            print(f"    现金年息{int(rc*100)}%: 立即{im*100:+6.2f}% vs 等待{wa*100:+6.2f}%  "
                  f"等待跑赢比例{win*100:4.1f}%  踏空{nev}/{N} {flag}")
