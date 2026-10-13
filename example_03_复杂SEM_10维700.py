"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（700份）

【量表变量】
  - 10个维度 ABCDEFGHIJ，题数分别为 A=6,B=6,C=7,D=7,E=6,F=6,G=5,H=6,I=7,J=7，共63题
  - 题目取值：1-5正整数，每题必须含1-5全部选项

【信度要求】
  - 问卷整体（63题合并）Cronbach's α：0.80 < α < 0.95
  - 10个维度 Cronbach's α：均在 0.80 < α < 0.95 范围内

【效度要求（CFA）】
  - 10个维度符合验证性因子分析（CFA）：
    * 标准化因子载荷系数 > 0.70，p值 < 0.05
    * AVE（平均方差抽取量）> 0.50
    * CR（组合信度）> 0.70
    * 区分效度：每个因子的 √AVE > 与其他所有因子的 Pearson 相关系数

【EFA 要求】
  - 10个维度符合探索性因子分析（EFA）：
    * KMO > 0.60，Bartlett p < 0.05
    * Varimax旋转后每题在对应因子主载荷≥0.50，且恰好提取10个因子

【相关要求】
  - ABCDEFGHIJ 10个维度两两显著正相关

【结构方程假设（SEM路径，OLS近似验证）】
  - A、B、C 分别正向显著影响 D（A→D正, B→D正, C→D正）
  - D 在 J→E 之间起正向中介作用（J→D→E，间接效应>0）
  - E 在 I→F 之间起正向中介作用（I→E→F，间接效应>0）
  - E 在 D→F 之间起正向中介作用（D→E→F，间接效应>0）
  - H、G 分别正向显著影响 F（H→F正, G→F正）

【均值要求】
  - 每个维度的均值（维度内各题均值的平均）> 3

【生成方法总结（供后续参考）】
  1. 外生潜变量设计：
     - fJ（外生）、fA/fB/fC（外生）、fG/fH（外生）、fI（外生）
     - fD = COEF_AD*fA + COEF_BD*fB + COEF_CD*fC + COEF_JD*fJ + noise_D
       （保证A/B/C→D显著，J→D存在使得J→D→E链式成立）
     - fE = COEF_DE*fD + COEF_IE*fI + noise_E
       （保证D→E显著，I→E显著）
     - fF = COEF_EF*fE + COEF_GF*fG + COEF_HF*fH + noise_F
       （保证E→F显著，G/H→F显著）
  2. 公共因子 fCommon（权重COMMON_WEIGHT）保证10维度两两正相关
  3. 题目生成：item = λ*factor + ε，λ=LOADING=0.82
     （λ越高，α/AVE越高，CFA载荷越高；0.82可保证CFA载荷>0.70且α<0.95）
  4. 均值控制：to_likert_target()，target_mean=3.3（保证维度均值>3）
  5. 选项完整性：ensure_all_levels()强制每题含1-5
  6. SEED网格搜索确保所有维度α在(0.80,0.95)区间内
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
N            = 700
LOADING      = 0.82      # 因子载荷（控制α/AVE/CFA载荷）
SEED         = 7         # 经网格搜索确定

DIMS         = ['A','B','C','D','E','F','G','H','I','J']
N_ITEMS      = {'A':6,'B':6,'C':7,'D':7,'E':6,'F':6,'G':5,'H':6,'I':7,'J':7}
N_LEVELS     = 5
TARGET_MEAN  = 3.3       # 维度均值目标（保证>3）

COMMON_WEIGHT = 0.55     # 公共因子权重（控制维度间相关）

# 结构路径系数
COEF_AD = 0.30; COEF_BD = 0.30; COEF_CD = 0.28
COEF_JD = 0.25
COEF_DE = 0.42
COEF_IE = 0.32
COEF_EF = 0.44
COEF_GF = 0.25; COEF_HF = 0.25

NOISE_D = 0.55; NOISE_E = 0.55; NOISE_F = 0.50
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)
all_cols = [f"{d}{i}" for d in DIMS for i in range(1, N_ITEMS[d]+1)]

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def std_norm(x):
    return (x - x.mean()) / x.std()


def to_likert_target(col, n_levels=5, target_mean=3.0, tol=0.05, max_iter=80):
    """连续值→Likert整数，迭代收敛到target_mean"""
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
            if len(cands) == 0:
                break
            result[cands[np.argsort(-result[cands])[:n_adj]]] -= 1
        else:
            n_adj = max(1, int((target_mean - cur) * len(result) * 0.6))
            cands = np.where(result < n_levels)[0]
            if len(cands) == 0:
                break
            result[cands[np.argsort(result[cands])[:n_adj]]] += 1
    return result


def ensure_all_levels(col_vals, n_levels=5):
    """确保1~n_levels每个值至少出现一次"""
    result = col_vals.copy()
    for v in range(1, n_levels + 1):
        if v not in result:
            counts = np.bincount(result, minlength=n_levels+1)[1:]
            for src in np.argsort(-counts):
                src_v = src + 1
                if src_v == v:
                    continue
                cands = np.where(result == src_v)[0]
                if len(cands) == 0:
                    continue
                mid = cands[len(cands)//2]
                result[mid] = v
                break
    return result


# ── Step1: 生成潜变量 ─────────────────────────────────────────────────────────

fCommon = std_norm(np.random.normal(0, 1, N))

def make_factor(common_w=COMMON_WEIGHT, noise=1.0):
    return std_norm(common_w * fCommon + np.random.normal(0, noise, N))

fA = make_factor(); fB = make_factor(); fC = make_factor()
fG = make_factor(); fH = make_factor()
fI = make_factor(); fJ = make_factor()

fD = std_norm(COEF_AD*fA + COEF_BD*fB + COEF_CD*fC + COEF_JD*fJ
              + np.random.normal(0, NOISE_D, N))
fE = std_norm(COEF_DE*fD + COEF_IE*fI + np.random.normal(0, NOISE_E, N))
fF = std_norm(COEF_EF*fE + COEF_GF*fG + COEF_HF*fH
              + np.random.normal(0, NOISE_F, N))

factors = {'A':fA,'B':fB,'C':fC,'D':fD,'E':fE,'F':fF,
           'G':fG,'H':fH,'I':fI,'J':fJ}

# ── Step2: 生成题目 ──────────────────────────────────────────────────────────

noise_std = np.sqrt(1 - LOADING**2)
data = {}
for dim in DIMS:
    factor = factors[dim]
    for item in range(1, N_ITEMS[dim]+1):
        raw = LOADING * factor + np.random.normal(0, noise_std, N)
        col = to_likert_target(raw, N_LEVELS, TARGET_MEAN)
        col = ensure_all_levels(col, N_LEVELS)
        data[f"{dim}{item}"] = col

df = pd.DataFrame(data)
