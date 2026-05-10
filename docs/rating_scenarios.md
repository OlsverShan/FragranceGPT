# Fragrantica 评分数据的应用场景

数据集中 Rating Value（1-5 分）和 Rating Count（投票数）目前未使用。以下是可扩展的方向。

---

## 当前任务 vs 评分

| | 当前任务（Accords → Notes） | 评分相关任务 |
|---|---|---|
| 输入 | 5 个 accords | accords + notes |
| 输出 | 配方 notes | 评分 / 推荐 |
| 评分有用吗 | 否（配方跟受欢迎程度无关） | 是（评分是目标变量） |

---

## 场景总览

### 场景 1：配方质量预测（性价比最高）

```
任务：给定 notes，预测这款香水的评分
输入：Top/Mid/Base notes 的完整配方
输出：预测 Rating Value（1-5）
模型：回归模型（XGBoost / LLM fine-tune）
```

**为什么做**：这是 accords→notes 的反向任务。可以跟现有项目形成互补——一个生成配方，一个评估配方质量。面试时展示两个模型形成闭环。

**技术栈**：
- 特征工程：notes 的 TF-IDF / embedding
- 模型：XGBoost / LightGBM（轻量）、或 fine-tune 小 LLM
- 评估：MAE、RMSE、跟真实评分的相关系数

**数据注意**：只用 Rating Count > 50 的香水（高信度样本），约 15,000 条有效数据。

---

### 场景 2：香水推荐系统（Demo 最佳）

```
用户输入喜欢的 accords
  ↓
系统预测配方 notes（现有 RAG 模型）
  ↓
从数据库检索类似真实香水
  ↓
按 Rating Value 降序排列
  ↓
推荐 Top-5："你可能喜欢这些高评分香水"
```

**为什么做**：这是最自然的 Demo 扩展。用户看到的不只是"系统预测了 notes"，而是"系统推荐了 5 款真实高评分香水"。体验感完全不同。

**技术栈**：
- 现有 RAG 管线 + 评分排序
- 可选：collaborative filtering（"喜欢 A 的人也喜欢 B"）

---

### 场景 3：高信度样本筛选

```
用途：所有消融实验的 few-shot 范例 / 训练集只用 Rating Count > 500 的香水
```

**为什么做**：当前 few-shot 范例是从全部数据中按 Rating Count 排序选的。可以进一步只用评分双高的（Rating Value > 4.0 且 Rating Count > 500）作为"基准真值"，提升范例质量。

**预期效果**：Few-shot 和 RAG 检索质量提升（参考样本更可靠）。

---

### 场景 4：市场分析（面试加分项）

```
分析维度：
- 哪些 accord 组合平均评分最高？
- 哪些 notes 在高评分香水中出现频率远高于低评分香水？
- 不同品牌 / 国家 / 年份的评分趋势
```

**为什么做**：这不是 ML 模型，但面试时被问"你的数据还有什么价值"时，可以展示这些分析。证明你不只会调 API，还有数据分析能力。

**输出示例**：
```
高评分配方（>4.5）中最常出现的 notes：
  bergamot (82%), sandalwood (76%), vanilla (71%), musk (68%)
  
最受欢迎 accord 组合（平均评分最高）：
  citrus + aromatic + woody + fresh spicy + amber = 4.31
  floral + woody + musky + powdery + fresh = 4.28
```

---

### 场景 5：加权 RAG 检索

```
当前 RAG：按余弦相似度检索 Top-5
改进：相似度 × Rating Value 加权 → 优先检索"既相似又高评分"的香水
```

**为什么做**：RAG 检索到的参考香水如果本身 notes 不准确（低质量标注），反而会误导 LLM。加权高分香水 = 参考更可靠。

**三个加权方案**：
- 方案 A：只用 Rating Count > 100 的香水建向量库（简单过滤）
- 方案 B：检索时用 `相似度 × normalized_rating` 排序
- 方案 C：消融实验对比：不加权 vs 加权 vs 过滤 → 量化评分的贡献

---

## 建议优先级

| 顺序 | 做什么 | 成本 | 作用 |
|------|--------|------|------|
| 1 | 市场分析（accord/note 评分统计） | 低 | 面试多一个分析角度 |
| 2 | 配方质量预测（notes → rating） | 中 | 与现有项目形成闭环 |
| 3 | Demo 推荐系统（用评分排序） | 低 | 展示最直观 |
| 4 | 加权 RAG 检索 | 低 | 消融实验新维度 |
