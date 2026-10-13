"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（350份）

【量表变量 ABCDE】
  - 5个维度，题数：A=4, B=4, C=4, D=4, E=6，共22题
  - 题目取值：1-5正整数，每题5个选项都必须出现（无缺失选项）
  - 信度：每个维度及整体 Cronbach's α > 0.70
  - 效度：AVE > 0.50，CR > 0.70
  - EFA：KMO > 0.60，Bartlett p<0.05，5因子结构清晰
  - CFA：五因子模型拟合良好（CFI>0.90，RMSEA<0.08）
  - ABCDE 两两正相关（r>0，p<0.05）

【结构假设（调节效应用交互项回归验证）】
  - A 显著正向影响 C（β_AC > 0）
  - A 显著正向影响 D（β_AD > 0）
  - B 显著正向影响 C（β_BC > 0）
  - B 显著正向影响 D（β_BD > 0）
  - C 显著正向影响 D（β_CD > 0）
  - C 在 A→D 之间起正向中介（间接效应 β_AC × β_CD > 0）
  - C 在 B→D 之间起正向中介（间接效应 β_BC × β_CD > 0）
  - E 正向调节 A→C（A×E 交互项系数 > 0）
  - E 正向调节 B→C（B×E 交互项系数 > 0）

【生成方法总结（供后续参考）】
  1. 潜变量驱动法：先生成5个弱相关潜变量（ρ=0.30，保证两两正相关）。
  2. 按结构路径生成 fC 和 fD：
     fC = β_AC*fA + β_BC*fB + β_AE_C*(fA×fE) + β_BE_C*(fB×fE) + β_EC*fE + noise
          → A/B正向→C，E正向调节A→C和B→C（交互项系数>0）
     fD = β_AD*fA + β_BD*fB + β_CD*fC + noise
          → A/B/C均正向→D，C正向中介
  3. 题目 = λ*factor + ε，λ=0.72，等频映射到1-5，强制无缺失。
  4. 检验：α → AVE/CR → EFA → CFA → 相关 → 路径假设
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import inv, lstsq
from scipy.stats import chi2 as chi2_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import semopy
import warnings
warnings.filterwarnings('ignore')

# ── 参数 ──────────────────────────────────────────────────────────────────────
N        = 350
LOADING  = 0.72
RHO      = 0.30    # 维度间基础相关（保证ABCDE两两正相关）
SEED     = 2025

DIMS     = ['A','B','C','D','E']
N_ITEMS  = {'A':4, 'B':4, 'C':4, 'D':4, 'E':6}
N_LEVELS = 5

# 结构路径系数（均为正，保证正向影响）
COEF_AC   =  0.40   # A → C
COEF_BC   =  0.35   # B → C
COEF_EC   =  0.10   # E → C（直接）
COEF_AE_C =  0.20   # A×E → C（E正向调节A→C）
COEF_BE_C =  0.18   # B×E → C（E正向调节B→C）
COEF_AD   =  0.25   # A → D（直接）
COEF_BD   =  0.25   # B → D（直接）
COEF_CD   =  0.45   # C → D（C正向中介）
NOISE_C   =  0.55
NOISE_D   =  0.55
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

def std_norm(x):
    return (x - x.mean()) / x.std()

# Step1: 生成5个弱相关潜变量（全部正相关）
cov = np.full((5, 5), RHO)
np.fill_diagonal(cov, 1.0)
base = np.random.multivariate_normal([0]*5, cov, N)
fA = std_norm(base[:,0])
fB = std_norm(base[:,1])
fE = std_norm(base[:,2])
# fC 和 fD 由路径生成，fE种子用base[:,3]保留独立性

# Step2: 生成 fC（含调节项）
AE_int = std_norm(fA * fE)  # 中心化交互项
BE_int = std_norm(fB * fE)
fC_raw = (COEF_AC*fA + COEF_BC*fB + COEF_EC*fE
          + COEF_AE_C*AE_int + COEF_BE_C*BE_int
          + np.random.normal(0, NOISE_C, N))
fC = std_norm(fC_raw)

# Step3: 生成 fD（C正向中介）
fD_raw = (COEF_AD*fA + COEF_BD*fB + COEF_CD*fC
          + np.random.normal(0, NOISE_D, N))
fD = std_norm(fD_raw)

# Step4: 生成题目
noise_std = np.sqrt(1 - LOADING**2)

def to_likert_forced(col, n_levels=5):
    quantiles = np.percentile(col, np.linspace(0, 100, n_levels+1))
    quantiles[0] -= 1e-9; quantiles[-1] += 1e-9
    result = np.digitize(col, quantiles[1:-1]) + 1
    result = np.clip(result, 1, n_levels)
    for v in range(1, n_levels+1):
        if v not in result:
            target = np.percentile(col, (v-0.5)/n_levels*100)
            diffs = np.abs(col - target)
            for idx in np.argsort(diffs):
                if result[idx] != v:
                    result[idx] = v
                    break
    return result.astype(int)

factor_map = {'A': fA, 'B': fB, 'C': fC, 'D': fD, 'E': fE}
data = {}
for dim in DIMS:
    factor = factor_map[dim]
    for item in range(1, N_ITEMS[dim]+1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        data[f"{dim}{item}"] = to_likert_forced(raw, N_LEVELS)

df = pd.DataFrame(data)
all_cols = [f"{d}{i}" for d in DIMS for i in range(1, N_ITEMS[d]+1)]
