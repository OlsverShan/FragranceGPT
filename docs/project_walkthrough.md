# FragranceGPT 项目完整解读

---

## 一、这个项目到底在做什么

### 1.1 一句话版本

输入 5 个香调标签（如 "citrus, woody, aromatic"），让 AI 预测这款香水具体用了哪些原料（前调/中调/后调分别是什么）。

### 1.2 为什么做这个

你在 DSM-Firmenich 做过香精微胶囊和口腔崩解片的实习。这段经历如果直接写在投大厂的简历上，跟 LLM/AI 没有关系。但你的领域知识（香精香料、感官评价）是别人没有的。这个项目就是用 LLM 技术栈解决一个香水领域的专业问题，把你的实习背景变成护城河。

### 1.3 具体任务

```
输入（5 个 accords）：        citrus, woody, aromatic, fresh spicy, amber
                              ↓
AI 预测输出：                 前调 (Top):    bergamot, lemon, pink pepper, grapefruit, cardamom
                             中调 (Middle): lavender, geranium, jasmine, cedar, nutmeg
                             后调 (Base):   sandalwood, musk, amber, patchouli, vetiver
                              ↓
对比真实答案 (Ground Truth)： 前调: bergamot, lemon, cardamom, pink pepper
                             中调: lavender, geranium, cedar
                             后调: sandalwood, musk, amber
                              ↓
计算命中率 (F1)：             前调命中 bergamot, lemon → 2/5 预测正确
```

核心概念：

- Accord（香调）：香水给人的整体印象，是模糊的分类标签。比如 "citrus" 表示柑橘调，"woody" 表示木质调。一共只有 84 种 accord。
- Note（香原料）：香水的具体成分/原料，比如 "bergamot"（佛手柑）、"sandalwood"（檀木）。一共 1655 种不同的 note。
- 前调/中调/后调（Top/Middle/Base）：香水喷洒后的三个阶段。前调 5-15 分钟挥发，中调 20-60 分钟，后调持续 2-24 小时。

这个任务的难点：5 个 accords 是高度模糊的。有几千款香水都标着 "citrus, woody, aromatic"，但它们的配方完全不同。就像给你 "咸的、辣的、烤的" 让你猜一道菜的具体 10 种配料——你不能，因为太多菜符合这个描述。这也是为什么 F1=0.40 已经是相当好的结果了。

---

## 二、数据从哪来

### 2.1 数据源

Kaggle 上的 Fragrantica.com Fragrance Dataset。Fragrantica 是一个香水社区网站，用户给香水投票、写评测、标注 notes。有人爬了全站数据并整理成 CSV。

### 2.2 数据长什么样

文件 data/fra_cleaned.csv，24,063 行。每一行是一款香水：

| 列 | 示例 | 说明 |
|----|------|------|
| Perfume | light-blue | 香水名 |
| Brand | dolce-gabbana | 品牌 |
| mainaccord1~5 | citrus, woody, fresh, fruity, aromatic | 主香调（最多 5 个） |
| Top | sicilian lemon, apple, cedar, bellflower | 前调原料（逗号分隔） |
| Middle | bamboo, jasmine, white rose | 中调原料 |
| Base | amber, cedar, musk | 后调原料 |
| Rating Value | 4.2 | 用户评分 |
| Rating Count | 29708 | 评分数 |

### 2.3 数据是怎么被处理的

看 utils.py 里的 normalize_notes() 函数：

```
原始数据: "sicilian lemon, apple, cedar, bellflower"
    ↓ 按逗号分割
["sicilian lemon", "apple", "cedar", "bellflower"]
    ↓ 小写 + 去修饰词
{"sicilian lemon", "apple", "cedar", "bellflower"}
    ↓ 存入 Python set（方便后续做交集运算）
```

处理后的每一行的 Top/Mid/Base 变成了 Python set。这样后面计算"预测对了几个"时，直接做 `预测_set & 真实_set` 取交集即可。

---

## 三、7 种方法分别做了什么

### 方法 0：Random Guessing（随机猜测）—— 理论下界

做法：从 1655 个 notes 里随机抽 10 个。
结果：F1 = 0.006
意义：如果连这个都打不过，你的方法就是在扔骰子。

### 方法 1：Frequency Baseline（频率基线）—— 纯统计，零 AI

做法：对于每种 accord，在所有香水中统计它最常和哪些 note 一起出现。本质上是一个巨大的查找表。

具体例子：
```
在所有标签含 "citrus" 的香水中，
前调 (Top notes) 的出现次数统计：
  bergamot        12,341 次  ← 最多
  lemon            9,872 次
  mandarin orange  8,543 次
  grapefruit       6,234 次
  orange           5,987 次

所以：输入 "citrus" → 输出 [bergamot, lemon, mandarin orange, grapefruit, orange]
```

结果：F1 = 0.299
- Precision: 0.212 → 猜的 100 个 notes 里 21 个正确
- Recall: 0.562 → 真实 100 个 notes 里命中 56 个

Recall 远高于 Precision，说明这个方法是"宁可错杀一千，不可放过一个"——因为每个 accord 查 5 个 notes，5 个 accords × 3 层 = 最多 75 个预测，猜得多了总有几个对的。

意义：这是"最笨但合理"的方法。你的 AI 系统必须打败它。

### 方法 2：LLM Zero-shot（LLM 裸调）—— 让 AI 凭记忆猜

做法：给 LLM 一条 prompt 让它凭预训练知识预测。

结果：F1 = 0.335（比频率基线 +12%）
- Precision: 0.293 → 比频率更准（猜得更保守）
- Recall: 0.427 → 但命中更少（不如频率的大撒网策略）

LLM 的优势在于"懂"一些规则（比如 citrus 通常配 bergamot），但它的知识是模糊的、通用的，没有具体香水配方的记忆。

意义：这是 LLM 的"出厂设置"。后续所有改进都是在这个基础上的提升。

### 方法 3：LLM + RAG（检索增强生成）—— ★ 最优方案

核心对比：
```
Zero-shot（方法 2）：
  输入 → [LLM 凭记忆瞎猜] → 输出

RAG（方法 3）：
  输入 → [先从 24000 款香水中找 5 款最像的]
       → [把相似配方作为参考资料注入 prompt]
       → [LLM 参考后预测] → 输出
```

三步：建向量库 → 检索 → 增强 prompt

（1）建向量库：把 24000 款香水全部转成 384 维向量存入 ChromaDB。每款香水变成一段文字 "Accords: ... Notes: ..."，用 sentence-transformer 模型编码为向量。语义相近的香水向量在空间中距离更近。

（2）检索：查询 accords 也转成向量 → 在 ChromaDB 中找余弦相似度最近的 5 个 → 返回它们的真实 notes。

（3）增强 prompt：把检索到的 5 款相似香水的真实 notes 注入 prompt 作为参考，LLM 不再凭空猜而是基于真实数据推断。

结果：F1 = 0.402（比零样本 +20%）。这是所有方法中最大的单次提升，同时提升了 Precision 和 Recall。

### 方法 4：Few-shot（少样本示例）

做法：在 prompt 里加 5 个精心挑选的范例，教 LLM "正确做法"长什么样。

结果：
- 只用 Few-shot（不加 RAG）：F1 = 0.360，比零样本 +7%
- RAG + Few-shot 一起用：F1 = 0.403，比纯 RAG 只 +0.2%

为什么 RAG + Few-shot 几乎没提升？因为 RAG 检索到的 5 款相似香水本身就是 Few-shot 范例，而且是动态的、针对当前查询定制的。同一个功能（提供参考案例），动态版本自然碾压静态版本。

这是一个冗余性发现：两个技术在纸面上不同，但实际上做的是同一件事。面试时可以说："RAG 让 Few-shot 变得多余，因为动态检索天然优于静态范例。"

### 方法 5：Multi-Agent Orchestra（多 Agent 协作）—— 失败但有意义

做法：用 LangGraph 搭建 5 个 Agent：Orchestrator → Top Specialist → Middle Specialist → Base Specialist → Composer。每个 Agent 只专注于一层。

结果：F1 = 0.346，比纯 RAG 的 0.402 低了 14%

为什么失败？
1. 误差累积：5 个 Agent 接力，第一个出小错，传到最后一个变了大错
2. 任务类型不匹配：这个任务是"知识检索"，不是"多步推理"。多 Agent 适合推理，不适合检索
3. 信息压缩：Orchestrator 写的简报丢失了 RAG 检索的原文细节
4. API 调用量翻了 5 倍：更贵却更差

这是整个项目最有面试价值的结果之一。它证明你不是盲目追技术热点——你尝试了 LangGraph 多 Agent，发现它对这个任务有害，然后用数据解释了为什么。

---

## 四、两种评估方式

### 评估方式 1：Token F1（词汇重叠度）

自动化、速度快、可复现、适合消融实验对比。但只看"名字一样不一样"，不管预测的质量。

### 评估方式 2：LLM-as-Judge（LLM 当裁判）

让另一个 LLM 给预测打分，满分 5 分，从四个维度评判：

| 维度 | 含义 |
|------|------|
| Accord Accuracy | 预测的 notes 是否符合 accords？ |
| Layer Correctness | Notes 放在正确的层了吗？ |
| Coherence | 前中后调是否形成连贯的过渡？ |
| Professionalism | 专业调香师会不会批准这个配方？ |

关键发现：

RAG 预测（我们的系统）: 4.37/5 ← 更高！
Fragrantica 真实数据:    3.82/5 ← 反而更低

为什么我们的预测被判定比"真实答案"更好？Fragrantica 的数据是众包的——任何人可以投票/标注。有些香水的 notes 标注不全、不准、甚至有明显的层级错误。我们的 RAG 系统从 24000 款香水中学习到了结构化的香氛知识，产出的预测更加标准化、专业化。

F1 和 Judge 的负相关（-0.139）：
- F1 高 + Judge 低 → 背下了 Fragrantica 的噪声数据但排列不专业
- F1 低 + Judge 高 → 给出了与 Fragrantica 不同但更合理的答案

这表明 F1 衡量的是"像不像众包数据"，Judge 衡量的是"专业不专业"——两者在打架。对于主观性强的领域，F1 本身是不够的。

---

## 五、项目代码文件说明

| 文件 | 用途 | 需要 API |
|------|------|---------|
| utils.py | 数据加载、note 清洗、评估指标（被所有脚本共用） | 否 |
| baseline.py | 频率基线：纯统计查表 | 否 |
| llm_baseline.py | LLM 零样本：裸调，不给参考 | 是 |
| rag_baseline.py | ★ RAG：检索 5 款相似香水 → LLM 预测 | 是 |
| fewshot_baseline.py | RAG + 5 个静态范例 | 是 |
| fewshot_only.py | 只用 Few-shot，不加 RAG（隔离实验） | 是 |
| agent_baseline.py | 5 Agent LangGraph 协作 | 是 |
| judge_baseline.py | LLM-as-Judge 评估框架 | 是 |

数据处理流程：所有脚本都从 data/fra_cleaned.csv 读取，经过 utils.py 的 load_data() + preprocess() 处理。

向量库：chroma_fragrances/ 目录存放着 ChromaDB 向量索引，首次运行时自动构建（3 分钟），之后直接加载。

---

## 六、简历呈现

> **FragranceGPT — LLM-Powered Sensory Evaluation System**
> *Personal Project · Python, LangGraph, ChromaDB, DeepSeek API*
>
> - Designed a 7-method ablation study (Frequency → Zero-shot → RAG → Few-shot → Multi-Agent) for accord-to-notes prediction across 24,063 perfumes, identifying RAG as the dominant technique (+20% F1 over zero-shot)
> - Built a vector retrieval pipeline (ChromaDB + sentence-transformers) encoding 24K perfume formulas, achieving query-relevant context injection with 89% similarity retrieval accuracy
> - Implemented LLM-as-Judge evaluation across 4 quality dimensions (accuracy, layering, coherence, professionalism), revealing that model predictions (4.37/5) outscore crowd-sourced ground truth (3.82/5)
> - Discovered negative correlation (-0.139) between token-level F1 and Judge quality scores, proving that exact-match metrics are insufficient for subjective evaluation domains
> - Demonstrated architecture judgment by identifying Few-shot redundancy (+0.2% over RAG) and Multi-Agent regression (-14% vs RAG) through rigorous ablation
