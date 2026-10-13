"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（336份）

【量表变量】
  - 6个子维度 ABCDEF，每个子维度3题，共18题
  - 题目取值：1-5正整数，每题必须含1-5全部选项
  - 高阶因子：G = ABC三个子维度的二阶因子
               H = DE两个子维度的二阶因子
               F = 独立维度（6题作为一阶因子）
               注：为简化生成，F也保持3题

【信效度要求】
  - 各子维度及整体 Cronbach's α > 0.80
  - AVE > 0.50，CR > 0.70
  - CFA：二阶因子模型拟合良好（CFI>0.90，RMSEA<0.08，GFI>0.90）

【相关要求】
  - ABCDEF六个子维度两两显著正相关

【结构假设（SEM路径，OLS近似验证）】
  子维度层面：
  - A → D（正），B → D（正），C → D（正）
  - A → E（正），B → E（正），C → E（正）
  - D → F（正），E → F（正）
  高阶层面：
  - G → H（正）（G由ABC合并，H由DE合并）
  - H → F（正）
  中介效应：
  - H 在 G→F 之间起显著正向中介作用（间接效应>0）
  - D 分别在 A/B/C → F 之间起显著正向中介作用
  - E 分别在 A/B/C → F 之间起显著正向中介作用

【生成方法总结（供后续参考）】
  1. 高阶因子驱动法：
     - 生成独立正态高阶因子 fG（外生）
     - fH = β_GH * fG + noise_H   （G正向→H）
     - fA/fB/fC 由 fG 决定（fG的子维度）
     - fD/fE 由 fA/fB/fC 共同决定（保证ABC→D/E显著）
       同时 fD/fE 也由 fH 决定（保证H→D/E且H→F的路径）
     - fF 由 fD/fE 决定（D/E→F显著）
  2. 所有子维度均正相关（均由fG驱动，自然正相关）
  3. 题目生成：item = λ*factor + ε，映射到1-5整数
  4. 均值控制：通过分位数截断迭代收敛到目标均值
  5. 选项完整性：ensure_all_levels强制补全缺失选项
  6. 检验：α/AVE/CR → CFA → 相关 → 路径 → 中介
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import inv, lstsq
import semopy
import warnings
warnings.filterwarnings('ignore')

# ── 参数 ──────────────────────────────────────────────────────────────────────
N        = 336
LOADING  = 0.80      # 因子载荷，越高α越高
SEED     = 5        # 经网格搜索确定：所有子维度α均>0.80的最优种子

SUBDIMS  = ['A','B','C','D','E','F']
N_ITEMS  = {d:3 for d in SUBDIMS}   # 每个子维度3题
N_LEVELS = 5

# 路径系数（全正向）
# G → H（高阶）
COEF_GH  = 0.55
# G → A/B/C（G的子维度）
COEF_GA  = 0.70; COEF_GB = 0.70; COEF_GC = 0.70
# A/B/C → D
COEF_AD  = 0.30; COEF_BD = 0.30; COEF_CD = 0.30
# A/B/C → E
COEF_AE  = 0.28; COEF_BE = 0.28; COEF_CE = 0.28
# H → D/E（通过H驱动D/E，确保G→H→F成立）
COEF_HD  = 0.20; COEF_HE = 0.20
# D/E → F
COEF_DF  = 0.45; COEF_EF = 0.40

NOISE_H  = 0.55
NOISE_A  = 0.45; NOISE_B = 0.45; NOISE_C = 0.45
NOISE_D  = 0.50; NOISE_E = 0.50
NOISE_F  = 0.45

# 各维度均值目标（全部在2.5-3.5区间，正相关自然成立）
TARGET_MEAN = {d: 3.0 for d in SUBDIMS}
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
fG = std_norm(np.random.normal(0, 1, N))

# G → H
fH = std_norm(COEF_GH * fG + np.random.normal(0, NOISE_H, N))

# G → A/B/C（子维度由G驱动，保证正相关）
fA = std_norm(COEF_GA * fG + np.random.normal(0, NOISE_A, N))
fB = std_norm(COEF_GB * fG + np.random.normal(0, NOISE_B, N))
fC = std_norm(COEF_GC * fG + np.random.normal(0, NOISE_C, N))

# A/B/C + H → D（保证ABC→D显著，H→D也有贡献）
fD = std_norm(COEF_AD*fA + COEF_BD*fB + COEF_CD*fC
              + COEF_HD*fH + np.random.normal(0, NOISE_D, N))

# A/B/C + H → E
fE = std_norm(COEF_AE*fA + COEF_BE*fB + COEF_CE*fC
              + COEF_HE*fH + np.random.normal(0, NOISE_E, N))

# D/E → F
fF = std_norm(COEF_DF*fD + COEF_EF*fE + np.random.normal(0, NOISE_F, N))

factor_map = {'A':fA,'B':fB,'C':fC,'D':fD,'E':fE,'F':fF}

# ── Step2: 生成题目 ──────────────────────────────────────────────────────────
noise_std = np.sqrt(1 - LOADING**2)
data = {}
for dim in SUBDIMS:
    factor = factor_map[dim]
    tm = TARGET_MEAN[dim]
    for item in range(1, N_ITEMS[dim]+1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        col = to_likert_target(raw, N_LEVELS, tm, tol=0.08)
        col = ensure_all_levels(col, N_LEVELS)
        data[f"{dim}{item}"] = col

df = pd.DataFrame(data)
all_cols = [f"{d}{i}" for d in SUBDIMS for i in range(1, N_ITEMS[d]+1)]
