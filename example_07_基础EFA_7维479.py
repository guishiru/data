"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（479份）

【量表变量 ABCDEFG】
  - 7个维度，题数分别为：A=6, B=5, C=6, D=3, E=3, F=3, G=3，共29题
  - 题目取值：1-7正整数，每题7个选项都必须出现（无缺失选项）
  - 信度：每个维度 Cronbach's α > 0.70，整体（29题）α > 0.70
  - 效度：AVE > 0.50，CR > 0.70
  - EFA：KMO > 0.60，Bartlett p<0.05，7因子结构清晰（每题主载荷所在因子与维度对应）

【生成方法总结（供后续参考）】
  1. 潜变量驱动法：先生成7个弱相关的连续潜变量（ρ=0.25），
     保证维度间有一定相关（整体α不过低）但不过高（EFA因子可分）。
  2. 每道题 = λ*factor + ε，λ=0.72，ε~N(0, 1-λ²)
     → 理论 α ≈ 0.82~0.87（各维度），整体 α 靠维度间相关拉高至>0.70
  3. 连续值通过等频分位数映射到1-7整数，强制每个选项至少出现一次（无缺失）。
  4. 检验流程：α（各维度+整体）→ AVE/CR → EFA（KMO/Bartlett/7因子归属）
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import inv
from scipy.stats import chi2 as chi2_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 参数 ──────────────────────────────────────────────────────────────────────
N       = 479
LOADING = 0.72      # 因子载荷（控制各维度α）
RHO     = 0.28      # 维度间相关（拉动整体α>0.70）
SEED    = 2025
N_LEVELS = 7        # Likert 量表级数

# 各维度题数：A=6, B=5, C=6, D=3, E=3, F=3, G=3
DIMS = ['A','B','C','D','E','F','G']
N_ITEMS = {'A':6, 'B':5, 'C':6, 'D':3, 'E':3, 'F':3, 'G':3}
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

# Step1: 生成7个弱相关潜变量
n_dims = len(DIMS)
cov = np.full((n_dims, n_dims), RHO)
np.fill_diagonal(cov, 1.0)
factors = np.random.multivariate_normal([0]*n_dims, cov, N)
# 标准化
factors = (factors - factors.mean(axis=0)) / factors.std(axis=0)

# Step2: 生成题目数据
noise_std = np.sqrt(1 - LOADING**2)

def to_likert_forced(col, n_levels=7):
    """等频分位数映射到1-n_levels整数，强制每个选项至少出现一次"""
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

data = {}
for dim_idx, dim in enumerate(DIMS):
    factor = factors[:, dim_idx]
    for item in range(1, N_ITEMS[dim]+1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        data[f"{dim}{item}"] = to_likert_forced(raw, N_LEVELS)

df = pd.DataFrame(data)
all_cols = [f"{dim}{i}" for dim in DIMS for i in range(1, N_ITEMS[dim]+1)]
