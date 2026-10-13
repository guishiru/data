"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（270份）

【量表变量 ABCD】
  - 4个维度，题数：A=4, B=7, C=6, D=6，共23题
  - 题目取值：1-5正整数，每题5个选项都必须出现（无缺失选项）
  - 各维度及整体 Cronbach's α > 0.80
  - 效度：AVE > 0.50，CR > 0.70
  - CFA：四因子模型拟合良好（CFI>0.90，RMSEA<0.08，GFI>0.90）

【均值约束】
  - A维度均值：2.5 ~ 3.0
  - B维度均值：3.0 ~ 3.5
  - C维度均值：2.5 ~ 3.0
  - D维度均值：2.0 ~ 2.5

【结构假设（OLS回归验证）】
  - B → A：显著负向（β<0）
  - C → A：显著正向（β>0）
  - D → A：显著正向（β>0）
  - B → C：显著负向（β<0）
  - B → D：显著负向（β<0）
  - C 正向中介 B→A：C→A正，B→C负 → 间接效应<0
  - D 正向中介 B→A：D→A正，B→D负 → 间接效应<0

【生成方法总结（供后续参考）】
  1. 结构路径驱动法 + 均值偏移：
     - fB 独立正态，其他因子由路径方程生成
     - fA = β_BA*fB + β_CA*fC + β_DA*fD + noise  (B负,C/D正)
     - fC = β_BC*fB + noise_C   (B负→C)
     - fD = β_BD*fB + noise_D   (B负→D)
  2. 题目 = λ*factor + ε，先映射连续值，再用分位数截断法控制均值
  3. 精确均值控制：先生成偏向目标均值的正态均值，
     再用迭代微调确保每题均值落在目标区间内，同时保证选项1-5全出现
  4. 选项完整性：替换策略优先选择频率最高的相邻值
==============================================================================
"""

import numpy as np
import pandas as pd

# ── 参数 ──────────────────────────────────────────────────────────────────────
N        = 270
LOADING  = 0.80     # 提高至0.80确保A维度4题也>0.80
SEED     = 4        # 经网格搜索确定最优种子

DIMS     = ['A','B','C','D']
N_ITEMS  = {'A':4, 'B':7, 'C':6, 'D':6}
N_LEVELS = 5

# 各维度目标均值区间
MEAN_RANGE = {'A':(2.5, 3.0), 'B':(3.0, 3.5), 'C':(2.5, 3.0), 'D':(2.0, 2.5)}
MEAN_TARGET = {'A': 2.75, 'B': 3.20, 'C': 2.75, 'D': 2.30}

# 路径系数（B负向，C/D正向影响A）
COEF_BA  = -0.38
COEF_CA  =  0.32
COEF_DA  =  0.28
COEF_BC  = -0.42
COEF_BD  = -0.35

NOISE_A  =  0.62
NOISE_C  =  0.62
NOISE_D  =  0.65
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

def std_norm(x):
    return (x - x.mean()) / x.std()

def to_likert_target(col, n_levels=5, target_mean=3.0, tol=0.08, max_iter=30):
    """
    将连续变量映射到 1~n_levels 整数，精确控制均值到 target_mean±tol。
    算法：先用分位数截断映射，然后迭代调整直到均值收敛。
    """
    center = (n_levels + 1) / 2
    shift = (target_mean - center) * 2.0
    col_s = col + shift

    pct = np.linspace(0, 100, n_levels + 1)
    cuts = np.percentile(col_s, pct)
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
    """确保1~n_levels每个值都出现，替换频率最高的相邻值中的少数几个"""
    result = col_vals.copy()
    for v in range(1, n_levels + 1):
        if v not in result:
            counts = np.bincount(result, minlength=n_levels+1)[1:]
            for src in np.argsort(-counts):
                src_v = src + 1
                if src_v == v: continue
                cands = np.where(result == src_v)[0]
                if len(cands) == 0: continue
                median_idx = cands[np.argsort(np.abs(np.arange(len(result))[cands] - len(result)//2))[0]]
                result[median_idx] = v
                break
    return result

# ── Step1: 生成潜变量 ──────────────────────────────────────────────────────
fB = std_norm(np.random.normal(0, 1, N))

fC_raw = COEF_BC * fB + np.random.normal(0, NOISE_C, N)
fD_raw = COEF_BD * fB + np.random.normal(0, NOISE_D, N)
fC = std_norm(fC_raw)
fD = std_norm(fD_raw)

fA_raw = COEF_BA*fB + COEF_CA*fC + COEF_DA*fD + np.random.normal(0, NOISE_A, N)
fA = std_norm(fA_raw)

factor_map = {'A': fA, 'B': fB, 'C': fC, 'D': fD}

# ── Step2: 生成题目 ──────────────────────────────────────────────────────────
noise_std = np.sqrt(1 - LOADING**2)
data = {}
for dim in DIMS:
    factor = factor_map[dim]
    tm = MEAN_TARGET[dim]
    for item in range(1, N_ITEMS[dim]+1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        col = to_likert_target(raw, N_LEVELS, tm, tol=0.08)
        col = ensure_all_levels(col, N_LEVELS)
        data[f"{dim}{item}"] = col

df = pd.DataFrame(data)
all_cols = [f"{d}{i}" for d in DIMS for i in range(1, N_ITEMS[d]+1)]

# ── 主程序（仅生成数据，不含检验）────────────────────────────────────────────
if __name__ == '__main__':
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据生成完成，共 {len(df)} 行，{len(df.columns)} 列")
    print(f"列名：{all_cols}")
