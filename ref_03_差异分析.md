# ref_03 差异分析类数据生成

## 何时读本文件
用户需求涉及：无维度结构、二分类结果变量（0/1）、Mann-Whitney U 检验、逻辑回归显著性控制、各题独立分布约束。对应 example_12 类场景。

---

## 核心思路：分层采样

结果变量 N 是 0/1 二分类时，最有效的方法是对 N=1 和 N=0 两组分别采样：
- **显著变量**：N=1 组偏高，N=0 组偏低
- **不显著变量**：完全随机，不依赖 N

```python
def generate_data(seed, n_total, n1_ratio=0.70):
    rng = np.random.default_rng(seed)
    n1 = int(n_total * n1_ratio)
    n0 = n_total - n1

    # 先生成 N
    N_col = sample_with_ratio([1 - n1_ratio, n1_ratio], [0, 1], n_total, rng)
    n1_idx = np.where(N_col == 1)[0]
    n0_idx = np.where(N_col == 0)[0]

    # 显著变量：分层生成
    # N=1 时从高端采样，N=0 时从低端采样

    # 不显著变量：完全随机
```

---

## 比例采样工具函数

```python
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
```

---

## 显著性强度（strength）经验值

| strength | 两组均值差（0-4量表） | 逻辑回归显著概率（N=252） |
|---------|-------------------|------------------------|
| 0.10    | ~0.3              | 30%~50%                |
| 0.15    | ~0.5              | 50%~70%                |
| 0.20    | ~0.7              | 70%~85%                |
| 0.25    | ~0.9              | 85%~95%                |
| 0.30    | ~1.1              | >90%                   |

**优先保证逻辑回归显著**（多元），Mann-Whitney 自然满足；反过来不成立。

---

## 分层采样模式

**0-4量表，N=1偏高：**
```python
# N=1 组：高值偏多
arr_n1 = sample_with_ratio([0.05, 0.10, 0.25, 0.35, 0.25], [0,1,2,3,4], len(n1_idx), rng)
# N=0 组：低值偏多
arr_n0 = sample_with_ratio([0.15, 0.30, 0.30, 0.15, 0.10], [0,1,2,3,4], len(n0_idx), rng)
arr = np.empty(N, dtype=int)
arr[n1_idx] = arr_n1
arr[n0_idx] = arr_n0
```

**0-1量表，N=1时1更多：**
```python
arr_n1 = sample_with_ratio([0.12, 0.88], [0,1], len(n1_idx), rng)
arr_n0 = sample_with_ratio([0.45, 0.55], [0,1], len(n0_idx), rng)
```

**有固定比例约束的显著变量（如B题1:2:3=2:4.5:2）：**
先满足逻辑回归显著性，比例允许±5%偏差。

---

## 特殊分布约束

**某值不出现（如D题4不出现）：**
```python
RANGES = {'D': (0, 3)}  # 合法范围定为0-3，不是0-4
# 完整性检验只检查0/1/2/3，不检查4
```

**某选项占比约束（如H题 0:1=8.5:1.5）：**
```python
arr = sample_with_ratio([8.5, 1.5], [0,1], N, rng)
```

---

## 多元逻辑回归显著性注意

单变量检验显著，但多元逻辑回归不显著的原因：变量间相关导致效应被吸收。

解决方案：
1. 提高该变量的 strength（增大两组差距）
2. 降低显著变量之间的相关性（各自独立分层，不引入额外相关）
3. SEED 搜索（最稳定）
