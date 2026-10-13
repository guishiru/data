"""
==============================================================================
需求备注（方便后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据
  - 4个维度 ABCD，每维度7题，共28题
  - 题目取值：1-5正整数，每题5个选项都必须出现（无缺失选项）
  - 样本量：300份
  - 信度：每个维度 Cronbach's α > 0.80
  - 效度：AVE > 0.50，CR > 0.70（通常>0.8即满足），区分效度达标
  - EFA：KMO > 0.60，Bartlett p<0.05，4因子结构清晰（主载荷>0.5，无严重交叉载荷）
  - CFA：四因子模型拟合良好（CFI>0.90，RMSEA<0.08）

生成方法总结（供后续参考）：
  1. 用潜变量驱动法：先生成4个标准正态潜变量（factor_A/B/C/D），维度间相关设为
     中等水平（ρ≈0.35），保证ABCD相关显著但不过高（KMO不超标）。
  2. 每个题目 = λ * factor + ε，λ（载荷）设为0.75，噪声方差 = 1 - λ²
     → 理论 α ≈ 7λ²/(7λ²+7(1-λ²)) × ... 实际控制在 0.82~0.88 范围。
  3. 连续值通过分位数映射到1-5整数，且强制每个选项至少出现一次（保证无缺失）。

关键参数：
  - N = 300（样本量）
  - n_items = 7（每维度题数）
  - loadings = 0.75（因子载荷）
  - rho = 0.35（维度间相关）
  - seed = 固定随机种子，保证可复现
==============================================================================
"""

import numpy as np
import pandas as pd

# ── 可调参数 ──────────────────────────────────────────────────────────────────
N          = 300      # 样本量
N_ITEMS    = 7        # 每个维度的题目数
LOADING    = 0.75     # 因子载荷（控制信度）
RHO        = 0.35     # 维度间相关系数
SEED       = 42       # 随机种子（可复现）
DIMS       = ['A', 'B', 'C', 'D']
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

# Step 1: 生成4个相关的潜变量
cov = np.full((4, 4), RHO)
np.fill_diagonal(cov, 1.0)
factors = np.random.multivariate_normal(mean=[0]*4, cov=cov, size=N)
# factors shape: (300, 4)

def continuous_to_likert_forced(col, n_levels=5):
    """
    将连续变量映射到1-5整数（等频分位数），
    并强制每个选项至少出现一次（保证无缺失选项）。
    """
    n = len(col)
    # 等频分箱
    quantiles = np.percentile(col, np.linspace(0, 100, n_levels + 1))
    quantiles[0]  -= 1e-9
    quantiles[-1] += 1e-9
    result = np.digitize(col, quantiles[1:-1]) + 1  # 1~5

    # 强制每个1-5都出现至少一次
    for v in range(1, n_levels + 1):
        if v not in result:
            target_q = (v - 1) / (n_levels - 1)
            target_val = np.percentile(col, target_q * 100)
            diffs = np.abs(col - target_val)
            idx = np.argsort(diffs)
            for i in idx:
                if result[i] != v:
                    result[i] = v
                    break
    return result.astype(int)

# Step 2: 生成题目数据
noise_std = np.sqrt(1 - LOADING**2)
data = {}

for dim_idx, dim in enumerate(DIMS):
    factor = factors[:, dim_idx]
    for item in range(1, N_ITEMS + 1):
        noise = np.random.normal(0, noise_std, N)
        continuous = LOADING * factor + noise
        col_name = f"{dim}{item}"
        data[col_name] = continuous_to_likert_forced(continuous)

df = pd.DataFrame(data)

# ── 主程序（仅生成数据，不含检验）────────────────────────────────────────────
if __name__ == '__main__':
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据生成完成，共 {len(df)} 行，{len(df.columns)} 列")
    print(f"列名：{list(df.columns)}")
