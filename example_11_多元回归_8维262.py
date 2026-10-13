"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（262份）

【量表变量】
  - 8个维度 ABCDEFGH，每个维度3题，共24题
  - 题目取值：1-5正整数，每题必须含1-5全部选项

【信效度要求】
  - 各维度 Cronbach's α > 0.80
  - AVE > 0.50，CR > 0.70

【EFA要求】
  - 8个维度符合探索性因子分析（EFA）
  - KMO > 0.60（适合因子分析）
  - Bartlett 球形检验显著（p < 0.05）
  - 提取8个因子后，每题在对应因子上有高载荷（>0.50）
  - 旋转方法：主成分分析 + Varimax正交旋转

【回归假设】
  - A、B、C、D、E、F、G 分别对 H 有显著影响（OLS多元回归）
  - 各预测变量 β 均显著（p < 0.05）

【生成方法总结（供后续参考）】
  1. 独立因子法：生成8个相对独立的潜变量fA~fH
     - fA~fG 由公共因子 fCommon（弱相关基底）+ 独立噪声构成
     - fH = Σ(β_iX_i) + noise（由fA~fG共同决定，保证回归显著）
  2. 题目生成：item = λ*factor + ε，λ=0.82确保α>0.80
  3. EFA结构：因子间相关较低（通过控制公共因子权重），保证Varimax旋转后结构清晰
  4. 均值控制：分位数截断迭代收敛，所有维度均值≈3.0（对称分布）
  5. 选项完整性：ensure_all_levels强制补全
  6. SEED搜索：网格搜索最优SEED确保所有α>0.80
==============================================================================
"""

import numpy as np
import pandas as pd

# ── 参数 ──────────────────────────────────────────────────────────────────────
N        = 262
LOADING  = 0.82      # 因子载荷，越高α越高（3题维度需≥0.80）
SEED     = 7         # 经网格搜索确定：所有维度α均>0.80的最优种子

DIMS     = ['A','B','C','D','E','F','G','H']
N_ITEMS  = 3         # 每维度3题
N_LEVELS = 5

# 公共因子权重（控制维度间相关强度，值越小EFA结构越清晰）
COMMON_WEIGHT = 0.25
# 预测H的路径系数（均为正向）
COEF_H = {'A':0.18,'B':0.18,'C':0.16,'D':0.16,'E':0.15,'F':0.15,'G':0.15}
NOISE_H = 0.55
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

def std_norm(x):
    return (x - x.mean()) / x.std()

def to_likert_target(col, n_levels=5, target_mean=3.0, tol=0.08, max_iter=40):
    center = (n_levels + 1) / 2
    shift  = (target_mean - center) * 2.0
    col_s  = col + shift
    pct    = np.linspace(0, 100, n_levels + 1)
    cuts   = np.percentile(col_s, pct)
    cuts[0] -= 1e-9; cuts[-1] += 1e-9
    result = np.digitize(col_s, cuts[1:-1]) + 1
    result = np.clip(result, 1, n_levels).astype(int)
    for _ in range(max_iter):
        cur = result.mean()
        if abs(cur - target_mean) <= tol:
            break
        if cur > target_mean:
            n_adj = max(1, int((cur - target_mean) * len(result) * 0.6))
            cands = np.where(result > 1)[0]
            if len(cands) == 0: break
            chosen = cands[np.argsort(-result[cands])[:n_adj]]
            result[chosen] -= 1
        else:
            n_adj = max(1, int((target_mean - cur) * len(result) * 0.6))
            cands = np.where(result < n_levels)[0]
            if len(cands) == 0: break
            chosen = cands[np.argsort(result[cands])[:n_adj]]
            result[chosen] += 1
    return result

def ensure_all_levels(col_vals, n_levels=5):
    result = col_vals.copy()
    for v in range(1, n_levels + 1):
        if v not in result:
            counts = np.bincount(result, minlength=n_levels+1)[1:]
            for src in np.argsort(-counts):
                src_v = src + 1
                if src_v == v: continue
                cands = np.where(result == src_v)[0]
                if len(cands) == 0: continue
                median_idx = cands[np.argsort(
                    np.abs(np.arange(len(result))[cands] - len(result)//2))[0]]
                result[median_idx] = v
                break
    return result

# ── Step1: 生成潜变量 ──────────────────────────────────────────────────────
# 公共因子（微弱，保持维度间有一定相关，但不破坏EFA结构）
fCommon = std_norm(np.random.normal(0, 1, N))

# A~G：主要由独立噪声构成，带少量公共因子
factors = {}
for d in ['A','B','C','D','E','F','G']:
    raw = COMMON_WEIGHT * fCommon + np.random.normal(0, 1, N)
    factors[d] = std_norm(raw)

# H：由A~G共同决定
fH_raw = sum(COEF_H[d] * factors[d] for d in COEF_H) + np.random.normal(0, NOISE_H, N)
factors['H'] = std_norm(fH_raw)

# ── Step2: 生成题目 ──────────────────────────────────────────────────────────
noise_std = np.sqrt(1 - LOADING**2)
data = {}
for dim in DIMS:
    factor = factors[dim]
    for item in range(1, N_ITEMS + 1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        col = to_likert_target(raw, N_LEVELS, 3.0, tol=0.08)
        col = ensure_all_levels(col, N_LEVELS)
        data[f"{dim}{item}"] = col

df = pd.DataFrame(data)

# ── 主程序（仅生成数据，不含检验）────────────────────────────────────────────
if __name__ == '__main__':
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据生成完成，共 {len(df)} 行，{len(df.columns)} 列")
    print(f"列名：{list(df.columns)}")
