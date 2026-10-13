"""
╔══════════════════════════════════════════════════════════════════╗
║                     问卷数据生成代码                              ║
╠══════════════════════════════════════════════════════════════════╣
║  【需求】                                                         ║
║  - 500份问卷，20题，7级量表（选项1~7）                            ║
║  - ABCDE共5个维度，每个维度4题                                    ║
║    A: Q1~Q4 / B: Q5~Q8 / C: Q9~Q12 / D: Q13~Q16 / E: Q17~Q20  ║
║  - 信度：每个维度 Cronbach's α > 0.8                             ║
║  - 效度：整体 KMO > 0.8，Bartlett p < 0.05                      ║
║  - 结构方程路径（均显著正向，p < 0.05，β > 0）：                  ║
║      A → B, C, D, E                                              ║
║      B → C                                                       ║
║      E → B, C, D                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  【生成思路总结】                                                  ║
║                                                                   ║
║  1. 确定潜变量的有向无环图（DAG）拓扑顺序                         ║
║     A（纯外生）→ E → B/D → C                                     ║
║     严格按此顺序生成，是 SEM 路径显著的核心前提。                  ║
║     坑：若把有因果关系的变量设为"独立外生"，该路径在SEM中永远      ║
║     不显著（v2 曾犯此错，142次迭代全部 A→E 失败）。               ║
║                                                                   ║
║  2. 潜变量生成公式                                                 ║
║     因变量 = Σ(路径系数 × 父节点潜变量) + noise × randn          ║
║     路径系数0.55、噪声0.45是经验甜蜜点：                          ║
║     - 路径系数太小(<0.3) → SEM 不显著                            ║
║     - 路径系数太大(>0.85) → 多重共线性，SEM 估计不稳定            ║
║                                                                   ║
║  3. 测量模型（潜变量 → 观测题项）                                  ║
║     题项 = loading × std(latent) + sqrt(1-loading²) × error     ║
║     loading = sqrt(rho_within)，rho_within=0.70 对应 α≈0.88     ║
║     信度与 SEM 路径解耦：rho_within 只控制 α，不影响路径          ║
║                                                                   ║
║  4. 离散化：等频分7段（非等距），避免量表两端堆积拉低信度           ║
║                                                                   ║
║  5. 最终参数（第1次即通过验证）：                                  ║
║     rho_within=0.70, coef_A=0.55, coef_E=0.55,                  ║
║     coef_B=0.40, noise=0.45, seed=42                             ║
║                                                                   ║
║  【依赖】 pip install pandas numpy openpyxl                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd

# ── 参数配置 ──────────────────────────────────────────────────────
N          = 500    # 样本量
N_LEVELS   = 7      # 量表级数
RHO_WITHIN = 0.70   # 题项间相关（决定 Cronbach's α，越高α越大）
COEF_A     = 0.55   # A 对各因变量的路径系数
COEF_E     = 0.55   # E 对 B/C/D 的路径系数
COEF_B     = 0.40   # B 对 C 的路径系数
NOISE      = 0.45   # 潜变量残差噪声
SEED       = 42     # 随机种子（固定保证可复现）

# ── 因子-题项映射 ─────────────────────────────────────────────────
FACTORS = {
    'A': ['Q1',  'Q2',  'Q3',  'Q4'],
    'B': ['Q5',  'Q6',  'Q7',  'Q8'],
    'C': ['Q9',  'Q10', 'Q11', 'Q12'],
    'D': ['Q13', 'Q14', 'Q15', 'Q16'],
    'E': ['Q17', 'Q18', 'Q19', 'Q20'],
}
ALL_ITEMS = [f'Q{i+1}' for i in range(20)]


def generate_questionnaire(
    n=N, n_levels=N_LEVELS,
    rho_within=RHO_WITHIN,
    coef_A=COEF_A, coef_E=COEF_E, coef_B=COEF_B,
    noise=NOISE, seed=SEED
):
    """
    基于结构方程因果链生成问卷数据

    潜变量拓扑顺序（严格遵守 DAG，确保 SEM 路径显著）：
      A（纯外生）
      ↓ coef_A
      E = coef_A·A + noise
      ↓ coef_A / coef_E
      B = coef_A·A + coef_E·E + noise
      D = coef_A·A + coef_E·E + noise
      ↓ coef_A / coef_B / coef_E
      C = coef_A·A + coef_B·B + coef_E·E + noise
    """
    np.random.seed(seed)

    # ── Step 1: 按拓扑顺序生成潜变量 ────────────────────────────
    A = np.random.randn(n)
    E = coef_A * A + noise * np.random.randn(n)
    B = coef_A * A + coef_E * E + noise * np.random.randn(n)
    D = coef_A * A + coef_E * E + noise * np.random.randn(n)
    C = coef_A * A + coef_B * B + coef_E * E + noise * np.random.randn(n)

    latents = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}

    # ── Step 2: 测量模型（潜变量 → 4个观测题项）────────────────
    # 题项 = loading × std(latent) + sqrt(1-loading²) × error
    # loading = sqrt(rho_within) 决定题项与潜变量的相关强度
    loading = np.sqrt(rho_within)
    obs = {}
    for fname, items in FACTORS.items():
        lat = latents[fname]
        lat_std = (lat - lat.mean()) / (lat.std() + 1e-8)   # 标准化
        for item in items:
            raw = loading * lat_std + np.sqrt(1 - loading**2) * np.random.randn(n)
            obs[item] = raw

    # ── Step 3: 等频分段映射到 1~N_LEVELS 离散值 ────────────────
    df = pd.DataFrame()
    df_raw = pd.DataFrame(obs)[ALL_ITEMS]
    for col in ALL_ITEMS:
        ranks  = df_raw[col].rank(method='first')
        groups = pd.cut(ranks, bins=n_levels, labels=range(1, n_levels + 1))
        df[col] = groups.astype(int)

    return df
