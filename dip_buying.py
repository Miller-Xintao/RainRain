# -*- coding: utf-8 -*-
"""现在加仓 vs 等暴跌再加仓 —— 用标普500日线(1928-2019)回测"""
import numpy as np, pandas as pd
df = pd.read_csv("data/sp500_daily_1927_2019.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
c = df["Close"].values; n = len(c)
H = 252  # 持有1年

# 当前相对前高的回撤
peak = np.maximum.accumulate(c); dd = c/peak - 1.0

print("="*68)
print("A) 按【当前回撤深度】分桶, 看未来1年的收益 (能不能靠跌得多来提高胜率?)")
print("="*68)
buckets = [(-0.001,0.001,"在前高附近(回撤<5%)"),(-0.05,-0.001,"回撤5-10%"),
           (-0.10,-0.05,"回撤10-20%"),(-0.20,-0.10,"回撤20-30%"),(-1.0,-0.20,"回撤>30% (深熊)")]
# 重新定义清晰的桶边界
edges = [0,-0.05,-0.10,-0.20,-0.30,-1.01]
labels = ["前高附近 (回撤0-5%)","回撤 5-10%","回撤 10-20%","回撤 20-30%","回撤 >30% (深熊)"]
fwd = np.full(n, np.nan); fwd[:n-H] = c[H:]/c[:n-H]-1.0
for i in range(len(labels)):
    lo, hi = edges[i+1], edges[i]
    mask = (dd<=hi)&(dd>lo)&(~np.isnan(fwd))
    f = fwd[mask]
    if len(f)==0: continue
    print(f"{labels[i]:<20} 样本{len(f):5d}天  1年后: 平均{f.mean()*100:+6.2f}%  中位{np.median(f)*100:+6.2f}%  胜率{(f>0).mean()*100:4.1f}%  最差{f.min()*100:+6.1f}%")

print("\n" + "="*68)
print("B) 【现在就买】 vs 【持币等回调X%再买】 —— 含等待期间踏空成本")
print("="*68)
# 策略: 从每个起点出发, 投资周期固定 T 天.
#  - 立即投资: 第0天买入, 持有到第T天.
#  - 等回调: 持币(0收益), 直到出现"自起点以来回撤>=X%"才买入, 买入后持有到第T天; 若T天内从未回调, 则一直空仓(期末收益0).
T = 252*3  # 3年窗口
for X in [0.05, 0.10, 0.20]:
    imm, wait, never = [], [], 0
    starts = range(0, n-T, 21)  # 每月一个起点
    for s in starts:
        seg = c[s:s+T+1]
        # 立即
        imm.append(seg[-1]/seg[0]-1.0)
        # 等回调: 段内运行高点回撤
        seg_peak = np.maximum.accumulate(seg)
        seg_dd = seg/seg_peak - 1.0
        hit = np.where(seg_dd <= -X)[0]
        if len(hit)==0:
            wait.append(0.0); never += 1   # 一直没等到, 全程空仓
        else:
            buy = hit[0]
            wait.append(seg[-1]/seg[buy]-1.0)
    imm, wait = np.array(imm), np.array(wait)
    print(f"\n等待阈值 = 回调 {int(X*100)}%  (3年窗口, {len(imm)}个起点)")
    print(f"  立即投资 : 平均{imm.mean()*100:+6.2f}%  中位{np.median(imm)*100:+6.2f}%  胜率{(imm>0).mean()*100:4.1f}%")
    print(f"  等回调买 : 平均{wait.mean()*100:+6.2f}%  中位{np.median(wait)*100:+6.2f}%  胜率{(wait>0).mean()*100:4.1f}%  (其中{never}个起点3年内从未触发, 全程踏空)")
    better = (wait>imm).mean()
    print(f"  -> 等待跑赢立即的比例: {better*100:.1f}%   平均超额: {(wait-imm).mean()*100:+.2f}%")
