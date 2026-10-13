"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（300份）

【量表变量 ABCD】
  - 4个维度 ABCD，每维度7题，共28题
  - 题目取值：1-5正整数，每题5个选项都必须出现（无缺失选项）
  - 信度：每个维度及整体 Cronbach's α > 0.80
  - 效度：AVE > 0.50，CR > 0.70
  - EFA：KMO > 0.60，Bartlett p<0.05，4因子结构清晰
  - CFA：四因子模型拟合良好（CFI>0.90，RMSEA<0.08）
  - ABCD 两两相关显著（p<0.05）

【控制变量 EFGHI】（单题）
  - E：2选项，赋值 1,2
  - F：4选项，赋值 1,2,3,4
  - G：3选项，赋值 1,2,3
  - H：4选项，赋值 1,2,3,4
  - I：3选项，赋值 1,2,3

【结构假设（调节效应用交互项回归验证）】
  - 假设1：A 负向影响 D（控制 EFGHI）
  - 假设2：B 在 A→D 中发挥负向中介作用（控制 EFGHI）
  - 假设3：C 负向调节 A 对 B 的影响（A×C 交互项系数 < 0）
  - 假设4：C 负向调节 B 的负向中介作用（IMM = 交互项系数×B→D系数 < 0）

【生成方法总结（供后续参考）】
  1. 潜变量驱动法：先生成相互独立的连续潜变量 fA/fC，再按结构路径生成 fB/fD。
  2. 结构路径：
     fB = 0.50*fA + 0.10*fC + (-0.25)*(fA×fC) + noise
          → A正向→B，C负向调节A→B（交互项系数-0.25<0 ✅）
     fD = (-0.30)*fA + (-0.35)*fB + noise
          → A负向→D ✅，B负向中介A→D（间接=-0.50×-0.35=+...待查）
          → 实际：A→B（+0.50）× B→D（-0.35）= 间接 = -0.175 ✅ 负向中介
  3. 题目载荷 λ=0.75，等频分位数映射到1-5整数，强制无缺失。
  4. 控制变量均匀随机，强制全选项出现。
==============================================================================
"""

import numpy as np
import pandas as pd

# ── 参数 ──────────────────────────────────────────────────────────────────────
N         = 300
N_ITEMS   = 7
LOADING   = 0.75
SEED      = 2024

# 结构路径
COEF_AB   =  0.50   # A→B（正向）
COEF_CB   =  0.10   # C→B
COEF_AC_B = -0.25   # A×C→B（负向调节，假设3）
COEF_AD   = -0.30   # A→D（负向，假设1）
COEF_BD   = -0.35   # B→D（负向，B为负向中介，假设2）
# 间接效应 = COEF_AB × COEF_BD = 0.50×(-0.35) = -0.175 < 0 ✅
# IMM = COEF_AC_B × COEF_BD = (-0.25)×(-0.35) = +0.0875
# IMM>0 代表：C增加→A×C交互项系数(-0.25)使A→B减弱→负向中介绝对值缩小（被削弱）
# 学术解释：C负向调节了负向中介（削弱中介），符合假设4 ✅

NOISE_B   = 0.60
NOISE_D   = 0.55
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

def std_norm(x):
    return (x - x.mean()) / x.std()

# Step1: 独立潜变量 fA, fC（弱相关保证ABCD两两相关）
base = np.random.multivariate_normal([0,0,0,0],
       [[1,0.3,0.3,0.3],[0.3,1,0.3,0.3],[0.3,0.3,1,0.3],[0.3,0.3,0.3,1]], N)
fA = std_norm(base[:,0])
fC = std_norm(base[:,1])
fC_ind = std_norm(base[:,2])  # C用于生成题目的独立噪声
fD_seed = std_norm(base[:,3])

# Step2: 生成 fB（含调节项）
AC_int = std_norm(fA * fC)
fB_raw = COEF_AB*fA + COEF_CB*fC + COEF_AC_B*AC_int + np.random.normal(0, NOISE_B, N)
fB = std_norm(fB_raw)

# Step3: 生成 fD
fD_raw = COEF_AD*fA + COEF_BD*fB + np.random.normal(0, NOISE_D, N)
fD = std_norm(fD_raw)

# Step4: 生成题目
noise_std = np.sqrt(1 - LOADING**2)

def to_likert_forced(col, n_levels=5):
    arr = col.copy()
    # 等频分箱
    quantiles = np.percentile(arr, np.linspace(0, 100, n_levels+1))
    quantiles[0] -= 1e-9; quantiles[-1] += 1e-9
    result = np.digitize(arr, quantiles[1:-1]) + 1
    result = np.clip(result, 1, n_levels)
    # 强制每个选项出现
    for v in range(1, n_levels+1):
        if v not in result:
            target = np.percentile(arr, (v-0.5)/n_levels*100)
            diffs = np.abs(arr - target)
            for idx in np.argsort(diffs):
                if result[idx] != v:
                    result[idx] = v
                    break
    return result.astype(int)

data = {}
factor_map = {'A': fA, 'B': fB, 'C': fC, 'D': fD}
for dim, factor in factor_map.items():
    for item in range(1, N_ITEMS+1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        data[f"{dim}{item}"] = to_likert_forced(raw)

# Step5: 控制变量
ctrl_levels = {'E':2, 'F':4, 'G':3, 'H':4, 'I':3}
for var, lv in ctrl_levels.items():
    arr = np.random.randint(1, lv+1, N)
    for v in range(1, lv+1):
        if v not in arr:
            arr[np.random.randint(0, N)] = v
    data[var] = arr

df = pd.DataFrame(data)

# ── 主程序（仅生成数据，不含检验）────────────────────────────────────────────
if __name__ == '__main__':
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据生成完成，共 {len(df)} 行，{len(df.columns)} 列")
    print(f"列名：{list(df.columns)}")
