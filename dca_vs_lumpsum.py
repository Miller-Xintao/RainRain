# -*- coding: utf-8 -*-
"""一次性投入 (Lump Sum) vs 定投 (DCA) —— 标普500日线(1928-2019)回测。
量化两个权衡:
  1) 期望收益: 一次性 vs 定投, 谁的终值更高、胜率多少
  2) 路径风险: 累积期内的最大回撤, 看定投是否更平滑
口径: 总资金=1, 在 K 个月内每月投 1/K (定投), 或第0天一把投入(一次性)。
未投入现金按 rc 年化计息(默认0%, 对定投略保守)。两策略在同一终点比较终值。
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/sp500_daily_1927_2019.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
c = df["Close"].values; n = len(c); STEP = 21  # 1月≈21交易日

def backtest(c, K, hold_days, rc_ann=0.0, start_step=21):
    """K=定投月数; hold_days=从起点到比较终点的总交易日数(>=K*21)。"""
    rc_d = (1+rc_ann)**(1/252)-1
    share_unit = 1.0/K
    ls_ret, dca_ret, ls_mdd, dca_mdd = [], [], [], []
    deploy_offsets = np.array([i*STEP for i in range(K)])
    for s in range(0, n-hold_days, start_step):
        end = s+hold_days
        # ---- 终值 ----
        ls_ret.append(c[end]/c[s]-1.0)
        dval = share_unit*np.sum(c[end]/c[s+deploy_offsets]
                                 * (1+rc_d)**deploy_offsets)  # 各份现金等到投入日的计息
        dca_ret.append(dval-1.0)
        # ---- 累积期内的路径与最大回撤 ----
        rel = np.arange(hold_days+1)
        seg = c[s:end+1]
        # 一次性: 全程满仓
        ls_path = seg/c[s]
        ls_mdd.append((ls_path/np.maximum.accumulate(ls_path)-1).min())
        # 定投: 每月投1/K, 未投部分为计息现金
        n_dep = np.minimum(K, rel//STEP + 1)           # 截至该日已投份数
        cum_shares = np.concatenate([[0.0], np.cumsum(share_unit/c[s+deploy_offsets])])
        eq = cum_shares[n_dep]*seg                      # 已投部分的权益市值
        cash = (K-n_dep)*share_unit                     # 未投现金(此处不计息, 对定投保守)
        dca_path = eq+cash
        dca_mdd.append((dca_path/np.maximum.accumulate(dca_path)-1).min())
    return (np.array(ls_ret), np.array(dca_ret), np.array(ls_mdd), np.array(dca_mdd))

def summarize(tag, ls, dca, lsdd, dcadd):
    print(f"\n--- {tag}  (n={len(ls)}个起点) ---")
    print(f"  终值收益  一次性: 均值{ls.mean()*100:+6.2f}%  中位{np.median(ls)*100:+6.2f}%  胜率{(ls>0).mean()*100:4.1f}%")
    print(f"           定投  : 均值{dca.mean()*100:+6.2f}%  中位{np.median(dca)*100:+6.2f}%  胜率{(dca>0).mean()*100:4.1f}%")
    print(f"  -> 一次性跑赢定投比例: {(ls>dca).mean()*100:.1f}%   平均收益差(一次性-定投): {(ls-dca).mean()*100:+.2f}%")
    print(f"  累积期最大回撤(越小越好)  一次性: 均值{lsdd.mean()*100:6.1f}%  最差{lsdd.min()*100:6.1f}%")
    print(f"                          定投  : 均值{dcadd.mean()*100:6.1f}%  最差{dcadd.min()*100:6.1f}%")
    print(f"  -> 定投平均把回撤从 {lsdd.mean()*100:.1f}% 降到 {dcadd.mean()*100:.1f}% (减少 {(lsdd.mean()-dcadd.mean())*100:.1f}个百分点)")
    return dict(ls_mean=ls.mean(), dca_mean=dca.mean(), ls_mdd=lsdd.mean(), dca_mdd=dcadd.mean(),
                ls_win=(ls>0).mean(), dca_win=(dca>0).mean(), ls_beats=(ls>dca).mean())

print("="*78)
print("一次性投入 vs 定投(DCA)  —— 总资金=1, 同一终点比较, 现金0%计息(对定投保守)")
print("="*78)
configs = [
    ("定投6个月, 比较点=第6月末",   6,  6*STEP),
    ("定投12个月, 比较点=第12月末", 12, 12*STEP),
    ("定投12个月, 比较点=第3年末(投完再持有2年)", 12, 3*252),
]
res = {}
for tag, K, hold in configs:
    out = backtest(c, K, hold)
    res[tag] = summarize(tag, *out)

# 现金计息敏感性(对定投更公平)
print("\n" + "="*78)
print("敏感性: 给定投未投入现金计息后, 收益差如何变化 (定投12个月, 比较点第12月末)")
print("="*78)
for rc in [0.0, 0.02, 0.04]:
    ls, dca, _, _ = backtest(c, 12, 12*STEP, rc_ann=rc)
    print(f"  现金年息{int(rc*100)}%: 一次性均值{ls.mean()*100:+.2f}%  定投均值{dca.mean()*100:+.2f}%  差{(ls-dca).mean()*100:+.2f}%  一次性跑赢比例{(ls>dca).mean()*100:.1f}%")

# ---- 图: 收益 vs 回撤 的权衡 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
tags = ["DCA 6m", "DCA 12m", "DCA 12m\n(+hold to 3y)"]
ls_m = [res[t]["ls_mean"]*100 for t in res]; dca_m = [res[t]["dca_mean"]*100 for t in res]
lsdd = [res[t]["ls_mdd"]*100 for t in res]; dcadd = [res[t]["dca_mdd"]*100 for t in res]
x = np.arange(len(tags))
ax1.bar(x-0.2, ls_m, 0.4, label="Lump Sum", color="#4C72B0")
ax1.bar(x+0.2, dca_m, 0.4, label="DCA", color="#DD8452")
ax1.set_xticks(x); ax1.set_xticklabels(tags); ax1.set_ylabel("Mean terminal return (%)")
ax1.set_title("Return: Lump Sum usually wins"); ax1.legend()
for i,v in enumerate(ls_m): ax1.text(i-0.2,v,f"{v:.1f}",ha="center",va="bottom",fontsize=8)
for i,v in enumerate(dca_m): ax1.text(i+0.2,v,f"{v:.1f}",ha="center",va="bottom",fontsize=8)
ax2.bar(x-0.2, lsdd, 0.4, label="Lump Sum", color="#4C72B0")
ax2.bar(x+0.2, dcadd, 0.4, label="DCA", color="#DD8452")
ax2.set_xticks(x); ax2.set_xticklabels(tags); ax2.set_ylabel("Mean max drawdown (%)")
ax2.set_title("Risk: DCA has a smoother ride (smaller drawdown)"); ax2.legend()
for i,v in enumerate(lsdd): ax2.text(i-0.2,v,f"{v:.1f}",ha="center",va="top",fontsize=8)
for i,v in enumerate(dcadd): ax2.text(i+0.2,v,f"{v:.1f}",ha="center",va="top",fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig8_dca_vs_lumpsum.png", dpi=110); plt.close(fig)
print("\n[OK] 图已保存 figures/fig8_dca_vs_lumpsum.png")
