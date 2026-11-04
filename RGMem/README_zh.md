<div align="center">

<h1 align="center">RGMem</h1>

<img src="assets/rgmem-banner.png" width="100%" alt="RGMem——受重整化群启发的记忆演化框架">

**面向长期个性化对话的多尺度、自演化记忆系统**

<p>
  <a href="https://arxiv.org/abs/2510.16392"><strong>论文</strong></a> ·
  <a href="https://arxiv.org/pdf/2510.16392"><strong>PDF</strong></a> ·
  <a href="assets/RGMem-Poster.pdf"><strong>海报</strong></a> ·
  <a href="docs/implementation.md"><strong>实现文档</strong></a> ·
  <a href="README.md"><strong>English</strong></a>
</p>

[![ICML 2026](https://img.shields.io/badge/ICML-2026-4B44CE.svg)](https://icml.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.16392-b31b1b.svg)](https://arxiv.org/abs/2510.16392)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2E8B57.svg)](LICENSE)

</div>

<p align="center">
  <img src="assets/rgmem-teaser.png" width="100%" alt="标准记忆与 RGMem 在多尺度对话场景中的对比">
</p>

<p align="center">
  <sub><em><strong>图 1.</strong> 扁平检索容易受关键词和近因偏置影响；RGMem 联合微观事实、中观模式和宏观用户特征，在当前需求与长期意图之间取得平衡。</em></sub>
</p>

**目录：** [项目简介](#项目简介) · [方法](#方法) · [实验结果](#实验结果) · [分析亮点](#分析亮点) · [快速开始](#快速开始) · [引用](#引用)

## 项目简介

长期对话记忆不只是检索问题。一个有效的记忆系统既要保留具体证据和反复出现的行为模式，也要响应真实的偏好变化，同时避免被短期噪声干扰。

RGMem 将这一过程建模为受重整化群（Renormalization Group, RG）启发的**多尺度记忆演化**：

- **层级粗粒化**：将情景对话逐步压缩为更加稳定的表示；
- **阈值驱动演化**：只有证据充分累积后才更新用户画像；
- **快慢变量分离**：区分长期稳定特征与快速变化的观测；
- **宏观不变性**：在事实变化之上保留跨场景的稳定行为规律。

RGMem 的记忆状态是显式、可审计的，能够同时支持事实回忆和高层个性化推理。

## 方法

<p align="center">
  <img src="assets/rgmem-architecture.png" width="100%" alt="RGMem 方法架构">
</p>

<p align="center"><sub><em><strong>图 2.</strong> RGMem 的状态空间构建、重整化流与多尺度观测。</em></sub></p>

论文方法与代码实现采用一致的三阶段结构：

1. **L0——微观证据层**：将原始对话切分为情景单元，并抽取结构化事实与用户相关结论；L0 记忆存储在 Elasticsearch。
2. **L1——演化知识空间**：使用 Neo4j 构建包含实例事件、一般事件和抽象概念的层级知识图谱，三个算子推动记忆演化：
   - `R_K1`：聚合同一关系上的重复证据；
   - `R_K2`：生成概念级序参量 `Σ` 与修正项 `Δ`；
   - `R_K3`：沿概念层级向上传播稳定结构。
3. **L2——多尺度观测层**：根据查询联合检索情景证据、关系总结和用户画像，并构建长度受控的上下文。

Redis 负责 L0 到 L1 的增量任务协调，FastAPI 提供记忆生成与检索接口。

## 实验结果

RGMem 在 **PersonaMem（128k）** 和 **LOCOMO** 上进行了评测：

- **PersonaMem**：在 GPT-4o-mini 和 GPT-4.1 下分别达到 `63.87` 和 `74.01`，相比第二名提升 **+7.08** 和 **+8.98** 个百分点；
- **LOCOMO**：在 gpt-4o-mini 和 gpt-4.1-mini 下分别达到 `78.92` 和 `86.17`；前者超过 Full-Context（`78.92` 对 `71.41`），后者接近 Full-Context（`86.17` 对 `87.52`）；
- **稳定性—可塑性**：RGMem 同时提升事实保持能力和最新偏好适应能力，而不是仅在传统折中曲线上移动。

<p align="center">
  <img src="assets/personamem-results.png" width="100%" alt="PersonaMem 实验结果">
</p>

<p align="center"><sub><em><strong>表 1.</strong> PersonaMem 准确率（%），括号中为相对第二名的提升。</em></sub></p>

<p align="center">
  <img src="assets/locomo-results.svg" width="80%" alt="LOCOMO 实验结果">
</p>

<p align="center"><sub><em><strong>表 2.</strong> LOCOMO 结果（%），采用 LLM-as-a-judge 评测。</em></sub></p>

<p align="center">
  <img src="assets/stability-plasticity.png" width="70%" alt="PersonaMem 上的稳定性与可塑性对比">
</p>

<p align="center"><sub><em><strong>图 5.</strong> RGMem 在 PersonaMem 的 Recall Facts 与 Latest Preference 上取得最佳联合表现。</em></sub></p>

## 分析亮点

以下实验进一步解释 RGMem 性能提升的来源。所有结果均为三次运行的平均值；LOCOMO 使用 gpt-4.1 作为评测模型，PersonaMem 报告多项选择准确率。

### 有效信息密度

<p align="center">
  <img src="assets/token-efficiency.png" width="72%" alt="检索 token 数量与准确率的关系">
</p>

<p align="center"><sub><em><strong>图 3.</strong> 准确率在约 3.8k 检索 token 附近达到峰值，随后下降，说明结构化粗粒化比无差别扩展上下文更有效。</em></sub></p>

### 临界演化阈值

<p align="center">
  <img src="assets/phase-transition-locomo.png" width="49%" alt="LOCOMO 上的演化阈值分析">
  <img src="assets/phase-transition-personamem.png" width="49%" alt="PersonaMem 上的演化阈值分析">
</p>

<p align="center"><sub><em><strong>图 4.</strong> 两个数据集都在 <code>θ_inf = 3</code> 附近出现最优区域：阈值过低会对噪声过度反应，阈值过高则会使画像过于僵化。</em></sub></p>

### 组件消融

每一个记忆尺度都对最终性能有贡献。即使检索更多上下文，移除 RG 演化算子仍会显著降低性能。

| 数据集 | 移除 L0 | 移除 L1 | 移除 RG | RGMem |
|:--|--:|--:|--:|--:|
| LOCOMO Overall | 56.29 | 79.88 | 82.17 | **86.17** |
| PersonaMem Overall | 47.22 | 54.31 | 57.37 | **63.87** |

在 LOCOMO 上，`移除 RG` 使用 4,354 个上下文 token，而 RGMem 仅使用 3,788 个；在 PersonaMem 上，两者分别为 8,479 和 7,105。因此，完整系统同时具有更高准确率和更高信息密度。

### 端到端效率

PersonaMem 128k 的成本按每个 session 统计，包含记忆构建、检索和回答生成。

| 方法 | 输入 token（k） | API 调用数 | 准确率（%） |
|:--|--:|--:|--:|
| Mem0 | 1,336.24 | 1,029.22 | 56.79 |
| MemoryOS | 3,489.71 | 3,278.42 | 54.23 |
| **RGMem** | **1,421.79** | **978.54** | **63.87** |

RGMem 在输入 token 成本与 Mem0 相近、API 调用更少的情况下，相比最佳基线提升 **+7.08 个百分点**。

<details>
<summary><strong>偏好变化动力学</strong></summary>

<p align="center">
  <img src="assets/preference-shift-slow.png" width="49%" alt="RGMem 在缓慢偏好变化下的动力学">
  <img src="assets/preference-shift-abrupt.png" width="49%" alt="RGMem 在偏好突然反转后的收敛过程">
</p>

在缓慢变化下，用户画像保持稳定，直至累积证据触发快速转变；面对突然且密集的偏好反转，新画像可以在少于 200 个 turn 内完成收敛。

</details>

<details>
<summary><strong>PersonaMem 20% 子集上的补充机制实验</strong></summary>

- 固定语义锚点将 Latest Preference 从 L0+GraphRAG 的 `49.72` 提升至 `74.11`；
- 移除修正项 `Δ` 后，Latest Preference 从 `74.11` 降至 `52.58`；
- 阈值驱动更新达到 `65.32%`，高频更新和周期性全局重写分别为 `53.72%` 和 `48.95%`。

这些子集分数用于机制分析，不应与上方完整 PersonaMem 数据集结果直接比较。

</details>

完整消融、收敛分析、参数敏感性、鲁棒性和计算成本详见[论文](https://arxiv.org/abs/2510.16392)附录 B。

## 快速开始

### 环境要求

- Python `>=3.13`
- [uv](https://docs.astral.sh/uv/)
- Elasticsearch 8.x
- Redis
- Neo4j
- Supabase 或 Elasticsearch，用作原始对话存储
- 所选 LLM 与嵌入模型的 API 凭据

### 安装

```bash
git clone https://github.com/fenhg297/RGMem.git
cd RGMem
uv sync --frozen
cp .env.example .env
```

填写 `.env`，并确保 Elasticsearch、Redis、Neo4j 和历史对话数据库可以访问。默认配置使用：

- OpenAI 兼容的 LLM 与嵌入接口；
- Supabase 原始对话存储；
- Elasticsearch 索引 `evaluation_persona_mem` 存储 L0 记忆；
- `text-embedding-3-large` 生成 3072 维 L0 向量；
- `text-embedding-3-small` 生成 1536 维 L1 向量。

### 启动 API

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

交互式 API 文档位于 `http://127.0.0.1:8000/docs`。

## API 使用

### 触发记忆生成

只有未处理消息达到批处理阈值后才会开始 L0 处理，默认阈值为 16 条。

```bash
curl -X POST "http://127.0.0.1:8000/L0_memory/trigger_memory_generation" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_01",
    "role_id": 10001,
    "created_at": "2025-08-01T12:00:00.000Z"
  }'
```

### 检索多尺度上下文

`retrieval_type` 支持 `relevance`、`recency` 和 `hybrid`。

```bash
curl -X POST "http://127.0.0.1:8000/L0_memory/retrieve_rag_context" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_01",
    "role_id": 10001,
    "query": "What music has the user been interested in recently?",
    "retrieval_type": "hybrid",
    "entity_queries": [
      {"name": "Music", "retrieve": ["summary", "history"]},
      {"name": "User", "retrieve": []}
    ]
  }'
```

接口返回包含 L0 证据和 L1 知识的 `rag_context` 字符串。完整数据流与配置参考[实现文档](docs/implementation.md)。

## 离线流程与评测

仓库提供了无需在线 Redis Listener 的离线处理脚本：

```bash
# 1. 设置 LOCAL_HISTORY_FILE_PATH，生成 L0 记忆。
uv run python scripts/1_produce_l0_memories.py

# 2. 设置 DATA_WORKING_DIRECTORY，将生成的 memory ID 写入 L1。
uv run python scripts/2_consume_and_build_l1.py

# 3. 可选：更新标记为 dirty 的层级总结。
uv run python scripts/trigger_dirty_summary.py

# 4. 设置 INPUT_DATASET_PATH、EVAL_USER_ID 和 EVAL_ROLE_ID 后进行评测。
uv run python scripts/evaluate_l0_plus_l1.py
```

本仓库不包含评测数据。请自行准备对话 JSON/JSONL 和评测集。对话记录至少应包含 `user_id`、`role_id`、`role`、`content` 和 `created_at`；`session_id`、`user_name`、`assistant_name` 为可选字段。

## 项目结构

```text
RGMem/
├── config/                 # 配置与提示词
├── l0_memory/              # 情景记忆生成和 ES 检索
├── l1_memory/              # 知识图谱演化和图检索
├── l2_retrieval/           # 多尺度检索编排
├── scripts/                # 离线构建与评测
├── docs/                   # 详细实现文档
├── assets/                 # README 图片和论文海报
├── main.py                 # FastAPI 入口
├── pyproject.toml
└── uv.lock
```

## 配置与安全

`config/settings.py` 从项目根目录 `.env` 加载配置。请基于 [`.env.example`](.env.example) 创建本地配置，切勿提交真实凭据。

> [!WARNING]
> 早期源码快照中包含疑似真实凭据的 `.env` 文件。如果这些值曾经有效，请在公开或部署仓库前将其从版本控制中删除并轮换相关凭据。

## 引用

如果 RGMem 对你的研究有帮助，请引用：

```bibtex
@article{tian2025rgmem,
  title={Rgmem: Renormalization group-based memory evolution for language agent user profile},
  author={Tian, Ao and Lu, Yunfeng and Fan, Xinxin and Wang, Changhao and Zhou, Lanzhi and Zhang, Yeyao and Liu, Yanfang},
  journal={arXiv preprint arXiv:2510.16392},
  year={2025}
}
```

## 联系方式

论文或代码相关问题请联系 Ao Tian：`tianao@buaa.edu.cn`，也可以在 GitHub 提交 Issue。
