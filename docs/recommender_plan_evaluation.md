# 香水推荐系统 — 方案评估

**日期**: 2026-05-07

---

## 用户计划

```
用户输入 accords 或需求
  ↓
系统预测配方 notes（现有 RAG 模型）
  ↓
从数据库检索类似真实香水
  ↓
按 Rating Value 降序排列
  ↓
推荐 Top-5："你可能喜欢这些高评分香水"

附加：提供原料配方 + 制作流程分析
```

---

## 整体评价：方向正确，值得做

当前项目最大的问题是**可展示性不足**。现在跑 baseline 出来的是 JSON 和三张表格——面试官看了没感觉。推荐系统天然适合 Demo：输入几个标签，弹出 5 款高评分香水，画面感完全不同。

---

## 逐环节分析

### 步骤 1：用户输入 accords

没问题。Streamlit 做下拉多选或自由文本输入都可以。84 个 accords，做 multiselect 下拉体验最好。

### 步骤 2：RAG 预测 notes

现有管线直接用，零改动。预测出来的 notes 有两个用途：一是展示"AI 为你的偏好设计了什么配方"，二是作为检索真实香水的桥梁。

### 步骤 3：检索相似真实香水

这里有一个架构选择问题：

| 方案 | 做法 | 优缺点 |
|------|------|--------|
| A：用预测的 notes 检索 | 拿 RAG 预测出来的 15 个 notes，编码后在向量库里找最相似的香水 | 引入了模型的误差，预测错了检索也偏 |
| B：用原始 accords 直接检索 | 用户输入的 accords 直接查向量库，但结果按评分排序 | 更稳定，不依赖模型预测质量 |
| C：混合检索 | accords 检索 + notes 交集过滤 + 评分加权 | 最全面，但逻辑复杂 |

**建议用方案 B**。accords 检索的结果本来就是向量库的核心能力。区别只是原来取 top-5 按相似度排序给 LLM 参考，现在取 top-20 按 Rating Value 排序给用户展示。不需要预测 notes 作为中间步骤——推荐和配方预测是两个独立的产品功能，可以分开。

### 步骤 4：按 Rating Value 排序

有一个陷阱：**Rating Count 低的香水评分不可靠**。一款 5.0 分但只有 3 人投的香水，不如一款 4.2 分有 5000 人投的香水有说服力。

需要用**贝叶斯加权**（类似 IMDb 的评分排名）：

```
weighted_score = (v / (v + m)) * R + (m / (v + m)) * C

其中:
  v = 该香水的 Rating Count
  m = 最低投票阈值（建议 50）
  R = 该香水的 Rating Value
  C = 全局平均评分
```

这样一款 R=5.0, v=3 的香水得分会大幅降低，而 R=4.2, v=5000 的香水接近真实 4.2。

### 步骤 5：推荐 Top-5

建议展示时包含：
- 香水名、品牌
- 评分（星级 + 数值）
- 真实的 Top/Mid/Base notes
- 匹配上的 accords（高亮显示哪些 accords 跟用户输入重合）

---

## 关于"原料配方和制作流程"

需要谨慎。LLM 分析"如何制作一款香水"是可行的——调香流程（称量、稀释、陈化、过滤）是公开知识。但具体到"bergamot 加多少克"——LLM 不知道，也没有公开的训练数据，香水配方比例是商业机密。

建议做两个层面的分析：
1. **成分分类**：把预测的 notes 按天然/合成分类，按挥发度（前/中/后调）说明为什么这个排序
2. **通用制作流程**：调香的标准步骤（配比设计 → 原料混合 → 酒精稀释 → 陈化 → 冷冻过滤 → 灌装），这是通识，LLM 可以写得很专业

**避免让 LLM 给出精确克数或百分比，那会是幻觉。**

---

## 架构建议

整个应用的推荐管线：

```python
def recommend(accords, store, top_k=20, final_n=5):
    # Step 1: 检索相似香水（向量库原生能力）
    candidates = store.retrieve(accords, top_k=top_k)
    
    # Step 2: 贝叶斯加权评分
    for c in candidates:
        v = c['rating_count']
        R = c['rating_value']
        c['weighted_score'] = (v / (v + 50)) * R + (50 / (v + 50)) * 4.0
    
    # Step 3: 按加权评分排序
    candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    # Step 4: 去重（同一品牌同一系列）
    # Step 5: 返回 Top-N
    return candidates[:final_n]
```

---

## 工作量估算

| 任务 | 时间 |
|------|------|
| `src/recommender.py` 推荐引擎 | 半天 |
| `src/formulator.py` 配方分析 | 半天 |
| `app/streamlit_app.py` UI | 1 天 |
| 向量库重建（加评分字段） | 1 小时 |
| 调试和优化 | 半天 |
| **合计** | **2-3 天** |

---

## 风险点

1. **向量库需要重建**：旧向量库的元数据没有 `rating_value`，虽然 `src/rag.py` 已经加了字段，但磁盘上存的是旧版。要删掉 `chroma_fragrances/` 重新 `build()`。
2. **评分数据质量**：统计 Rating Count 的分布，确认有多少香水投票数太低不适合推荐。
3. **accords 覆盖度**：84 个 accords 有些很罕见（比如 "conifer"、"ozonic"），这些 accords 对应香水少，推荐结果可能重复或不相关。
