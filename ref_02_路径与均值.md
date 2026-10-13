# ref_02 结构路径与均值控制

## 何时读本文件
用户需求涉及：维度间因果路径（中介/调节/SEM/回归）、均值约束（某维度均值>3 或落在某区间）。

---

## 一、结构路径注入

**核心原则：路径关系必须在潜变量层面注入，严格按 DAG 拓扑顺序生成。**
先生成外生变量，再按路径方程生成内生变量，绝不能把有因果关系的变量设为独立外生。

### 直接路径（A → B）
```python
fA = np.random.randn(N)
path_coef = 0.30   # 建议 0.25~0.40
fB = path_coef * fA + np.sqrt(1 - path_coef**2) * np.random.randn(N)
```

路径系数选择（N=300时）：

| 路径系数 | 预期标准化β | 显著性 |
|---------|-----------|--------|
| 0.20    | 0.18~0.22 | 通常显著 |
| 0.25    | 0.23~0.28 | 稳定显著 |
| 0.30    | 0.27~0.33 | 稳定显著 |
| 0.40    | 0.36~0.44 | 高度显著 |

### 多预测变量（A + B → C）
```python
coef_A, coef_B = 0.28, 0.28
residual_var = np.sqrt(1 - coef_A**2 - coef_B**2)
fC = coef_A * fA + coef_B * fB + residual_var * np.random.randn(N)
```
**约束**：各路径系数的平方和 < 1，否则 residual_var 变虚数。
安全上限：3个预测变量各0.30（0.09×3=0.27，√0.73=0.85 ✓）

### 单中介（A → M → B）
```python
fA = np.random.randn(N)
fM = 0.35 * fA + np.sqrt(1 - 0.35**2) * np.random.randn(N)
coef_AM_direct, coef_MB = 0.20, 0.35   # 0=完全中介
residual = np.sqrt(1 - coef_AM_direct**2 - coef_MB**2)
fB = coef_AM_direct * fA + coef_MB * fM + residual * np.random.randn(N)
```

### 调节效应（A × W → B）
```python
fA = np.random.randn(N)
fW = np.random.randn(N)
fA_std = (fA - fA.mean()) / fA.std()
fW_std = (fW - fW.mean()) / fW.std()
interaction = fA_std * fW_std   # 标准化后再相乘
coef_A, coef_W, coef_int = 0.30, 0.20, 0.20
res = np.sqrt(1 - coef_A**2 - coef_W**2 - coef_int**2)
fB = coef_A*fA + coef_W*fW + coef_int*interaction + res * np.random.randn(N)
```
注意：交互项系数 < 0.20 在 N=300 时不稳定，建议 ≥ 0.20。

### 高阶因子（二阶SEM）
```python
fG = np.random.randn(N)   # 二阶因子
fA = 0.80 * fG + np.sqrt(1 - 0.80**2) * np.random.randn(N)
fB = 0.78 * fG + np.sqrt(1 - 0.78**2) * np.random.randn(N)
fC = 0.82 * fG + np.sqrt(1 - 0.82**2) * np.random.randn(N)
# 再从一阶维度生成题目
```

### 负向路径
路径系数取负值即可：
```python
fB = -0.40 * fA + np.sqrt(1 - 0.40**2) * np.random.randn(N)
```
注意：有负向路径时不报告整体α，只报告各维度α。

---

## 二、均值控制

**核心问题**：直接等频切割均值固定在中间值（1-5量表约3.0），无法满足均值约束。

### 标准实现：带均值控制的 Likert 转换
```python
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
        if abs(cur - target_mean) <= tol: break
        if cur > target_mean:
            n_adj = max(1, int((cur - target_mean) * len(result) * 0.6))
            cands = np.where(result > 1)[0]
            if len(cands) == 0: break
            result[cands[np.argsort(-result[cands])[:n_adj]]] -= 1
        else:
            n_adj = max(1, int((target_mean - cur) * len(result) * 0.6))
            cands = np.where(result < n_levels)[0]
            if len(cands) == 0: break
            result[cands[np.argsort(result[cands])[:n_adj]]] += 1
    return result
```

### 强制补全缺失选项
```python
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
                mid = cands[len(cands)//2]
                result[mid] = v
                break
    return result
```

### 均值目标经验值（1-5量表）

| 目标均值 | 说明 |
|---------|------|
| 2.0~2.5 | 明显偏低 |
| 2.5~3.0 | 偏低 |
| 3.0     | 居中（默认） |
| 3.3     | 略偏高，推荐默认 |
| 3.5~4.0 | 中偏高 |
| >4.0    | 允许1分不出现（放宽完整性要求） |

### 特殊量表处理
```python
# 0-3取值（4个选项）
def to_scale_0_3(z_scores, target_mean=1.5):
    result = to_likert_target(z_scores, target_mean + 1, scale_points=4)
    return result - 1

# 0-4取值（5个选项）
def to_scale_0_4(z_scores, target_mean=2.0):
    result = to_likert_target(z_scores, target_mean + 1, scale_points=5)
    return result - 1

# 二值（0-1），控制1的比例
def to_binary(z_scores, prob_1=0.7):
    threshold = np.percentile(z_scores, (1 - prob_1) * 100)
    return (z_scores >= threshold).astype(int)
```

---

## 三、最小样本量参考

路径系数β在 power=0.80 时所需最小N：

| β    | 最小N |
|------|-------|
| 0.15 | ~350  |
| 0.20 | ~197  |
| 0.25 | ~126  |
| 0.30 | ~88   |

N不足时：增大路径系数，或增大样本量。
