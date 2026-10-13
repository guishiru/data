"""
==============================================================================
需求备注（供后续参考）
==============================================================================
需求：生成符合以下条件的问卷数据（252份）

【题目设置】
  14道题，依次为 A~N，每题取值范围（含端点整数）：
    A: 0-1    B: 0-4    C: 0-4    D: 0-4（但D的4不出现，实际0-3）
    E: 0-1    F: 0-1    G: 0-3    H: 0-1
    I: 0-4    J: 0-4    K: 0-3    L: 0-4
    M: 0-3    N: 0-1
  每道题的所有合法整数选项都必须出现（不能有缺失）
  注意：D题说明"4不选"，因此D的合法范围视为0-3，只需保证0/1/2/3各出现

【各题分布约束】
  B: 1:2:3 = 2:4.5:2（即1和3各占约23.5%，2占约53%；0和4随机补充）
  D: 1和2占比多，4不出现（D只在0-3取值）
  E: 1更多（1占比 > 0占比）
  G: 0:1:2:3 = 5.5:3:1:0.5
  H: 0:1 = 8.5:1.5
  N: 1占大部分（1占比 > 50%）
  I、J、L: 3和4选项合计占比 > 50%

【统计分析要求】
  1. 差异性分析（Mann-Whitney U，以N为二值分组变量）：
     - ADFHJL 和 N 之间：差异不显著（p > 0.05）
     - BCEGIKM 和 N 之间：差异显著（p < 0.05）
  2. 逻辑回归（N为因变量，BCEGIKM为自变量）：
     - BCEGIKM 对 N 的影响均显著（p < 0.05）

【生成方法总结（供后续参考）】
  核心思路：N是二分类结果变量，BCEGIKM与N相关，ADFHJL与N无关
  1. 先按N的分布约束（1占70%）生成N
  2. BCEGIKM按N分层生成（N=1时高值偏多，N=0时低值偏多），保证显著性
     特别是B：用b_strength=0.20控制N=1/N=0组的分布偏移幅度
  3. ADFHJL完全随机生成（不依赖N），保证差异不显著
  4. 各题分布约束通过目标比例采样：先按比例生成，再用
     ensure_all_values()确保所有合法值至少出现一次
  5. D的"4不出现" → D的合法范围定为0-3
  6. SEED=1，b_strength=0.20（经0-300范围网格搜索确定）
==============================================================================
"""

import numpy as np
import pandas as pd

# ── 参数 ──────────────────────────────────────────────────────────────────────
N          = 252
SEED       = 1       # 经网格搜索确定
B_STRENGTH = 0.20    # B题分层强度（控制N=1/N=0的分布偏移）

# 各题合法取值范围（D的4不出现，视为0-3）
RANGES = {
    'A': (0, 1), 'B': (0, 4), 'C': (0, 4), 'D': (0, 3),
    'E': (0, 1), 'F': (0, 1), 'G': (0, 3), 'H': (0, 1),
    'I': (0, 4), 'J': (0, 4), 'K': (0, 3), 'L': (0, 4),
    'M': (0, 3), 'N': (0, 1)
}

SIG_ITEMS   = ['B', 'C', 'E', 'G', 'I', 'K', 'M']  # 与N显著
NOSIG_ITEMS = ['A', 'D', 'F', 'H', 'J', 'L']        # 与N不显著

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def ensure_all_values(arr, lo, hi, rng):
    """确保[lo, hi]内每个整数值至少出现一次"""
    arr = arr.copy()
    for v in range(lo, hi + 1):
        if v not in arr:
            counts = np.bincount(arr - lo, minlength=hi - lo + 1)
            max_val = np.argmax(counts) + lo
            idx = np.where(arr == max_val)[0][0]
            arr[idx] = v
    return arr


def sample_with_ratio(ratios, values, n, rng):
    """按比例采样，返回长度为n的整数数组"""
    ratios = np.array(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    counts = np.round(ratios * n).astype(int)
    diff = n - counts.sum()
    if diff > 0:
        counts[np.argmax(ratios)] += diff
    elif diff < 0:
        counts[np.argmax(counts)] += diff
    arr = []
    for v, c in zip(values, counts):
        arr.extend([v] * max(0, c))
    arr = np.array(arr[:n])
    rng.shuffle(arr)
    return arr


def generate_data(seed=SEED, b_strength=B_STRENGTH):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(N))

    # ── 1. 生成 N（1占约70%）────────────────────────────────────────────────
    N_col = sample_with_ratio([0.30, 0.70], [0, 1], N, rng)
    N_col = ensure_all_values(N_col, 0, 1, rng)
    df['N'] = N_col
    n1_idx = np.where(N_col == 1)[0]
    n0_idx = np.where(N_col == 0)[0]

    # ── 2. 与N显著相关的题目（BCEGIKM）─────────────────────────────────────
    # B: 0-4，分布约束 1:2:3≈2:4.5:2，同时B与N显著
    b_str = b_strength
    arr_n1 = sample_with_ratio(
        [0.03, 1 - b_str*2, b_str*2, b_str*2, 0.03],
        [0, 1, 2, 3, 4], len(n1_idx), rng)
    arr_n0 = sample_with_ratio(
        [0.05, b_str*4, b_str*2, 0.05, 0.03],
        [0, 1, 2, 3, 4], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['B'] = ensure_all_values(arr, 0, 4, rng)

    # C: 0-4，N=1偏高，N=0偏低
    arr_n1 = sample_with_ratio([0.05, 0.10, 0.25, 0.35, 0.25],
                                [0,1,2,3,4], len(n1_idx), rng)
    arr_n0 = sample_with_ratio([0.15, 0.30, 0.30, 0.15, 0.10],
                                [0,1,2,3,4], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['C'] = ensure_all_values(arr, 0, 4, rng)

    # E: 0-1，整体1更多；N=1时1更多，N=0时接近均等
    arr_n1 = sample_with_ratio([0.12, 0.88], [0,1], len(n1_idx), rng)
    arr_n0 = sample_with_ratio([0.45, 0.55], [0,1], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['E'] = ensure_all_values(arr, 0, 1, rng)

    # G: 0-3，整体 0:1:2:3=5.5:3:1:0.5；N=1时高值更多
    arr_n1 = sample_with_ratio([4.5, 3.0, 1.5, 1.0], [0,1,2,3], len(n1_idx), rng)
    arr_n0 = sample_with_ratio([7.0, 2.5, 0.4, 0.1], [0,1,2,3], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['G'] = ensure_all_values(arr, 0, 3, rng)

    # I: 0-4，3+4>50%，N=1时3/4更多
    arr_n1 = sample_with_ratio([0.05, 0.08, 0.12, 0.40, 0.35],
                                [0,1,2,3,4], len(n1_idx), rng)
    arr_n0 = sample_with_ratio([0.10, 0.18, 0.25, 0.28, 0.19],
                                [0,1,2,3,4], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['I'] = ensure_all_values(arr, 0, 4, rng)

    # K: 0-3，3占比高（>50%），N=1时3更多
    arr_n1 = sample_with_ratio([0.05, 0.10, 0.20, 0.65], [0,1,2,3], len(n1_idx), rng)
    arr_n0 = sample_with_ratio([0.15, 0.25, 0.35, 0.25], [0,1,2,3], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['K'] = ensure_all_values(arr, 0, 3, rng)

    # M: 0-3，N=1时高值更多
    arr_n1 = sample_with_ratio([0.10, 0.20, 0.35, 0.35], [0,1,2,3], len(n1_idx), rng)
    arr_n0 = sample_with_ratio([0.30, 0.35, 0.25, 0.10], [0,1,2,3], len(n0_idx), rng)
    arr = np.empty(N, dtype=int)
    arr[n1_idx] = arr_n1
    arr[n0_idx] = arr_n0
    df['M'] = ensure_all_values(arr, 0, 3, rng)

    # ── 3. 与N不显著的题目（ADFHJL）—完全随机，不依赖N ─────────────────────

    # A: 0-1，均等
    arr = sample_with_ratio([0.50, 0.50], [0,1], N, rng)
    df['A'] = ensure_all_values(arr, 0, 1, rng)

    # D: 0-3（4不出现），1和2占比多
    arr = sample_with_ratio([0.15, 0.38, 0.38, 0.09], [0,1,2,3], N, rng)
    df['D'] = ensure_all_values(arr, 0, 3, rng)

    # F: 0-1，均等
    arr = sample_with_ratio([0.50, 0.50], [0,1], N, rng)
    df['F'] = ensure_all_values(arr, 0, 1, rng)

    # H: 0-1，0:1=8.5:1.5
    arr = sample_with_ratio([8.5, 1.5], [0,1], N, rng)
    df['H'] = ensure_all_values(arr, 0, 1, rng)

    # J: 0-4，3+4>50%，随机
    arr = sample_with_ratio([0.08, 0.12, 0.15, 0.38, 0.27], [0,1,2,3,4], N, rng)
    df['J'] = ensure_all_values(arr, 0, 4, rng)

    # L: 0-4，3+4>50%，随机
    arr = sample_with_ratio([0.08, 0.10, 0.15, 0.38, 0.29], [0,1,2,3,4], N, rng)
    df['L'] = ensure_all_values(arr, 0, 4, rng)

    # 排列列顺序 A-N
    cols = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    return df[cols]


# ── 主程序（仅生成数据，不含检验）────────────────────────────────────────────
if __name__ == '__main__':
    print(f"使用 SEED={SEED}, B_STRENGTH={B_STRENGTH}，生成 {N} 份数据...")
    df = generate_data(SEED, B_STRENGTH)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据生成完成，共 {len(df)} 行，{len(df.columns)} 列")
    print(f"列名：{list(df.columns)}")
