---
name: survey-data-generator
description: 问卷模拟数据生成专家。当用户需要生成满足统计约束的问卷模拟数据时使用，包括：生成模拟数据、造数、问卷数据、模拟问卷、Likert数据、SEM数据、中介效应数据、调节效应数据、差异分析数据、信度效度数据、结构方程数据、因子分析数据。输出可直接运行的 Python 生成代码。
---

# 问卷模拟数据生成 Skill

## 你的任务
将用户的自然语言需求转化为可直接运行的 Python 数据生成代码。
**不需要写检验代码**，检验由外部接口负责。

---

## 工作流程（严格按顺序执行）

### Step 1：解析需求 → 输出结构化 JSON

用以下规则从用户需求中提取信息：

**输出 JSON 格式：**
```json
{
  "sample_size": 整数,
  "n_dims": 整数,
  "dim_names": ["维度名", ...] 或 null,
  "items_per_dim": 整数 或 {"维度名": 题数, ...},
  "scale_points": 整数,
  "tests": [
    {"name": "信度分析", "params": {"scope": "分维度", "alpha_min": 0.7, "alpha_max": null}},
    {"name": "效度分析", "params": {"scope": "分维度", "kmo_min": 0.7, "kmo_max": null}},
    {"name": "探索性因子分析", "params": {"expected_factors": {"因子名": ["题目"]}, "min_loading": 0.5}},
    {"name": "验证性因子分析", "params": {"factors": {"因子名": ["题目"]}, "min_std_loading": 0.6}},
    {"name": "相关分析", "params": {"pairs": [{"variables": ["A", "B"], "significant": true, "direction": "正向"}]}},
    {"name": "中介效应", "params": {"iv": "自变量", "dv": "因变量", "mediators": [{"name": "中介名", "significant": null}]}},
    {"name": "调节效应", "params": {"iv": "自变量", "dv": "因变量", "moderator": "调节变量", "significant": null}},
    {"name": "线性回归", "params": {"iv": [{"name": "自变量名", "coefficient_direction": null}], "dv": "因变量"}},
    {"name": "描述统计", "params": {"dim_mean_constraints": [{"dim": "维度", "mean": ">3"}]}},
    {"name": "差异性分析", "params": {"group_var": "N", "sig_vars": ["B","C"], "nonsig_vars": ["A","D"]}},
    {"name": "结构方程模型", "params": {"factors": {"因子名": ["题目"]}, "paths": [{"from": "A", "to": "B", "direction": "正向"}]}}
  ],
  "ensure_all_levels": true,
  "special_requirements": []
}
```

**解析规则：**
1. `sample_size`/`n_dims`/`items_per_dim`/`scale_points` 必填，未指定填 null
2. `dim_names` 必须从需求提取，如"ABC三个维度"→["A","B","C"]，完全没提才填 null
3. `tests` 按用户提到的检验填写，不多不少
4. 信度/效度：指定范围（如"0.8-0.9"）须同时设 min 和 max
5. SEM paths 须逐条拆分，direction 须提取（"正向"/"负向"），不能为 null
6. factors 自动生成：如"ABC三个维度，每个3题"→{"A":["A1","A2","A3"],"B":["B1","B2","B3"],"C":["C1","C2","C3"]}
7. 相关分析：用户说"都/全部/两两相关"→ pairs 必须为 C(n,2) 全排列，不得遗漏
8. 影响关系归属（按优先级）：提"SEM/结构方程/路径分析"→SEM | 提"中介"→中介效应 | 提"调节"→调节效应 | 提"回归"或仅"影响"→线性回归 | 仅"相关"→相关分析

解析完成后，**向用户展示 JSON 并确认**，有歧义时询问，确认无误后继续。

---

### Step 2：匹配参考资料

读取 `references/examples/INDEX.md`，按以下逻辑匹配：

**2a. 判断类型：**
- tests 含"差异性分析"或"逻辑回归"且结果变量是0/1 → **差异分析类**，读 `example_12_差异分析_252.py` + `ref_03_差异分析.md`
- 否则 → **标准问卷类**，继续 2b

**2b. 标准问卷类，按 INDEX.md 的映射表找最匹配的 1 个 py 文件并读取**

**2c. 判断是否需要额外读 md 文件：**
- 题数/λ/α 与示例不同 → 读 `ref_01_参数调优.md`
- 路径系数/均值目标需要调整 → 读 `ref_02_路径与均值.md`
- 生成失败或约束矛盾 → 读 `ref_04_SEED与排障.md`

---

### Step 3：生成代码

以匹配的 py 文件为骨架，按 JSON 解析结果修改参数，生成完整可运行代码。

**代码必须包含以下部分：**

**① 顶部注释（需求摘要）**
```python
"""
需求摘要：[从JSON提取的关键信息]
参考示例：[使用的py文件名]
关键参数：N=xxx, LOADING=xxx, SEED搜索范围=0~300
"""
```

**② 参数区**（集中在顶部，方便调整）
```python
N        = xxx      # 样本量
LOADING  = xxx      # 因子载荷（控制α/AVE）
DIMS     = [...]    # 维度名
N_ITEMS  = {...}    # 各维度题数
N_LEVELS = 5        # 量表级数
# 路径系数
COEF_XX  = xxx
# 均值目标（如有）
MEAN_TARGET = {...}
```

**③ 工具函数**（从示例复制，保持不变）
- `std_norm(x)`
- `to_likert_target(...)` 或 `continuous_to_likert_forced(...)`（按示例选择）
- `ensure_all_levels(...)`

**④ 潜变量生成**（严格按 DAG 拓扑顺序，外生变量先生成）

**⑤ 题目生成**（`item = LOADING * factor + noise`）

**⑥ SEED 搜索框架**（必须包含，搜索范围 0~300）
```python
def generate_data(seed):
    np.random.seed(seed)
    # ... 完整生成逻辑 ...
    return df

def validate_quick(df):
    """快速验证核心约束，返回 (passed, failures)"""
    failures = []
    # 验证α范围
    for dim, items in dim_items.items():
        alpha = cronbach_alpha(df[items])
        if not (ALPHA_MIN <= alpha <= ALPHA_MAX):
            failures.append(f"{dim} α={alpha:.3f}")
    # 验证均值约束（如有）
    # 验证选项完整性
    return len(failures) == 0, failures

BEST_SEED = None
for seed in range(300):
    df = generate_data(seed)
    passed, failures = validate_quick(df)
    if passed:
        BEST_SEED = seed
        print(f"✅ SEED={seed}")
        break
    if seed % 50 == 0 and failures:
        print(f"seed={seed}: {failures[:2]}")

if BEST_SEED is None:
    print("❌ 未找到，请调整参数（提高λ或调整路径系数）")
else:
    df = generate_data(BEST_SEED)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"生成完成：{len(df)}行 × {len(df.columns)}列，SEED={BEST_SEED}")
```

**⑦ `cronbach_alpha` 函数**（SEED搜索验证用）
```python
def cronbach_alpha(df_items):
    k = df_items.shape[1]
    item_var = df_items.var(ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)
```

---

### Step 4：参数适配要点

从示例骨架改到用户需求时，重点检查以下项：

| 需要改的内容 | 改哪里 | 参考 |
|------------|--------|------|
| 样本量不同 | `N` | 直接改 |
| 维度名不同 | `DIMS`、`N_ITEMS`、`factor_map`、列名 | 全局替换 |
| 题数不同 | `N_ITEMS`、`LOADING`（题数变化时λ需重查） | ref_01 |
| 量表级数不同 | `N_LEVELS`、`to_likert_target` 的 n_levels | 直接改 |
| α要求不同 | `LOADING`（查对照表）、`ALPHA_MIN/MAX` | ref_01 |
| 路径系数需调整 | `COEF_XX`、`NOISE_XX` | ref_02 |
| 均值约束不同 | `MEAN_TARGET`、`to_likert_target` 的 target_mean | ref_02 |
| 路径方向为负 | 路径系数取负值 | ref_02 |
| 有调节效应 | 加交互项 `std_norm(fA * fW)` | ref_02 |

---

## 注意事项

- **output_path 变量**：代码中保留 `output_path` 变量，由外部调用时传入，不要硬编码路径
- **不写检验代码**：只生成数据，不调用任何检验函数（α/AVE/CFA等检验由外部接口负责）
- **SEED搜索中的 validate_quick 只验证生成约束**（α范围、均值范围、选项完整性），不做统计检验
- **负向路径时**：不在 validate_quick 中检验整体α，只检验各维度α
- **差异分析类**：SEED搜索的 validate_quick 改为验证逻辑回归显著性（用 statsmodels 快速跑一次）

---

## references 目录说明

```
references/
├── ref_01_参数调优.md       # λ/α/AVE/CR 对照表，维度间相关控制
├── ref_02_路径与均值.md     # 路径系数代码模板，均值控制函数
├── ref_03_差异分析.md       # 差异分析类专用（分层采样、比例约束）
├── ref_04_SEED与排障.md     # SEED搜索框架，7种约束矛盾解法
└── examples/
    ├── INDEX.md             # 需求特征→py文件映射表（Step 2 必读）
    ├── example_01_中介调节_300.py
    ├── example_02_基础_4维300.py
    ├── example_03_复杂SEM_10维700.py
    ├── example_04_调节中介_600.py
    ├── example_05_SEM路径_5维500.py
    ├── example_06_双重中介调节_350.py
    ├── example_07_基础EFA_7维479.py
    ├── example_08_链式中介_10维450.py
    ├── example_09_高阶因子_6维336.py
    ├── example_10_均值控制_4维270.py
    ├── example_11_多元回归_8维262.py
    └── example_12_差异分析_252.py
```
