# ref_04 SEED 搜索与约束矛盾排障

## 何时读本文件
生成逻辑正确但约束偶尔不满足时（α超范围、路径不显著、选项缺失），或遇到"做不到"的约束组合时。

---

## 一、标准 SEED 搜索框架

```python
def generate_data(seed):
    np.random.seed(seed)
    # ... 生成逻辑 ...
    return df

def validate_all(df):
    """返回 (是否全部通过, 失败原因列表)"""
    failures = []

    # 检验α范围
    for dim, items in dim_items.items():
        alpha = cronbach_alpha(df[items])
        if not (alpha_min <= alpha <= alpha_max):
            failures.append(f"{dim} α={alpha:.3f} 不在[{alpha_min},{alpha_max}]")

    # 检验均值约束
    for dim, (lo, hi) in mean_constraints.items():
        m = df[[c for c in df.columns if c.startswith(dim)]].mean().mean()
        if not (lo <= m <= hi):
            failures.append(f"{dim} 均值={m:.2f} 不在[{lo},{hi}]")

    # 检验选项完整性
    for col, required in completeness_checks.items():
        for v in required:
            if v not in df[col].values:
                failures.append(f"{col} 缺少选项{v}")

    return len(failures) == 0, failures

# 搜索
for seed in range(300):
    df = generate_data(seed)
    passed, failures = validate_all(df)
    if passed:
        print(f"✅ SEED={seed} 满足所有约束")
        break
    if seed % 50 == 0:
        print(f"seed={seed}: {failures[:2]}")
else:
    print("❌ 0~299内未找到，请调整参数")
```

---

## 二、搜索范围建议

| 约束难度 | 建议范围 | 预期成功率 |
|---------|---------|-----------|
| 简单（α范围+均值） | 0~50 | >95% |
| 中等（+路径显著+区分效度） | 0~100 | >90% |
| 复杂（+上限约束+逻辑回归） | 0~300 | >80% |
| 极复杂（有矛盾约束） | 0~500 | <50%，需调参 |

**失败率>50% 说明参数与逻辑不兼容，需调整参数，不是继续搜索。**

---

## 三、常见约束矛盾与解法

### 矛盾1：α下限 vs α上限（α>0.80 且 α<0.95）
根本原因：高λ+多题数容易超上限。
解法：λ=0.82，题数≤6，SEED搜索。
λ=0.82 的安全范围：3题α≈0.83，5题α≈0.88，6题α≈0.89，均不超0.95。

### 矛盾2：严格比例约束 vs 显著性
场景：B题1:2:3必须严格等于2:4.5:2，同时B必须在逻辑回归中显著。
解法：优先保证逻辑回归显著，比例允许±5%偏差。

### 矛盾3：CFA载荷>0.70 vs 区分效度（√AVE>r）
解法：提高λ，降低COMMON_WEIGHT（0.55→0.45）。
验证：√AVE=λ≈0.82，维度间相关≈COMMON_W²≈0.20，0.82>0.20 ✅

### 矛盾4：整体α偏低 vs 各维度α达标
原因：维度间存在负向路径，负相关题目混在一起计算整体α时互相抵消。
解法：不报告整体α，只报告各维度α。

### 矛盾5：均值>3 vs 选项完整性
均值>4.0时，强制出现1分会拉低均值。
解法：均值>4.0时，required_levels = [2,3,4,5]，允许1分不出现。

### 矛盾6：某值不出现 vs 完整性检验
场景：D题合法取值0-4，但业务约束要求4不出现。
解法：修改完整性检验逻辑，RANGES['D'] = (0, 3)，只检验0/1/2/3。

### 矛盾7：N不足 vs 路径显著
场景：N=100时，路径系数0.25不显著。
解法：增大路径系数（0.20→0.30），或增大样本量。
最小N公式：N_min ≈ (1.96 + 0.84)² / β²

---

## 四、调试技巧：找最难满足的约束

```python
failure_counts = {}
for seed in range(100):
    df = generate_data(seed)
    _, failures = validate_all(df)
    for f in failures:
        key = f.split('=')[0]
        failure_counts[key] = failure_counts.get(key, 0) + 1

for k, v in sorted(failure_counts.items(), key=lambda x: -x[1]):
    print(f"{k}: 失败{v}/100次")
```

失败率最高的约束就是需要重点调参的地方。
