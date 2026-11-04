<div align="center">

<h1 align="center">RGMem</h1>

<img src="assets/rgmem-banner.png" width="100%" alt="RGMem — Renormalization Group-inspired Memory Evolution">

**A multi-scale, self-evolving memory framework for long-term personalized dialogue**

<p>
  <a href="https://arxiv.org/abs/2510.16392"><strong>Paper</strong></a> ·
  <a href="https://arxiv.org/pdf/2510.16392"><strong>PDF</strong></a> ·
  <a href="assets/RGMem-Poster.pdf"><strong>Poster</strong></a> ·
  <a href="docs/implementation.md"><strong>Implementation Guide</strong></a> ·
  <a href="README_zh.md"><strong>中文</strong></a>
</p>

[![ICML 2026](https://img.shields.io/badge/ICML-2026-4B44CE.svg)](https://icml.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.16392-b31b1b.svg)](https://arxiv.org/abs/2510.16392)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2E8B57.svg)](LICENSE)

</div>

<p align="center">
  <img src="assets/rgmem-teaser.png" width="100%" alt="Standard memory compared with RGMem on a multi-scale dialogue scenario">
</p>

<p align="center">
  <sub><em><strong>Figure 1.</strong> Flat retrieval reacts to lexical overlap and recency. RGMem integrates microscopic facts, mesoscopic patterns, and macroscopic traits to balance current needs with long-term intent.</em></sub>
</p>

**Contents:** [Overview](#overview) · [Method](#method) · [Results](#results) · [Analysis Highlights](#analysis-highlights) · [Quick Start](#quick-start) · [Citation](#citation)

## Overview

Long-term dialogue memory is not only a retrieval problem. A useful memory system must preserve concrete evidence, consolidate recurring patterns, adapt to genuine preference changes, and avoid overreacting to transient noise.

RGMem models this process as **multi-scale memory evolution**, inspired by the renormalization group (RG):

- **Hierarchical coarse-graining** compresses episodic dialogue into increasingly stable representations.
- **Thresholded evolution** updates a profile only after sufficient evidence accumulates.
- **Slow and fast variables** separate durable user traits from rapidly changing observations.
- **Macroscopic invariance** preserves stable behavior patterns beyond individual facts.

The result is an explicit and auditable memory state that supports both factual recall and higher-level personalization.

## Method

<p align="center">
  <img src="assets/rgmem-architecture.png" width="100%" alt="RGMem architecture">
</p>

<p align="center"><sub><em><strong>Figure 2.</strong> State-space construction, renormalization flow, and multi-scale observation in RGMem.</em></sub></p>

RGMem follows the same three-stage organization in the paper and the implementation:

1. **L0 — Microscopic evidence.** Raw dialogue is segmented into episodes and transformed into structured facts and user-related conclusions. L0 memories are stored in Elasticsearch.
2. **L1 — Evolving knowledge space.** A hierarchical knowledge graph in Neo4j organizes instances, recurring events, and abstract concepts. Three operators drive its evolution:
   - `R_K1`: consolidates repeated relation-level evidence;
   - `R_K2`: synthesizes concept-level order parameters `Σ` and correction terms `Δ`;
   - `R_K3`: propagates stable structure through the concept hierarchy.
3. **L2 — Multi-scale observation.** Query-aware retrieval combines relevant episodic evidence with relation- and profile-level representations into a bounded context for the language model.

Redis coordinates incremental L0-to-L1 processing, while FastAPI exposes memory generation and retrieval endpoints.

## Results

RGMem is evaluated on **PersonaMem (128k)** and **LOCOMO**. Reported values below follow the benchmark-specific protocols used in the paper.

- **PersonaMem:** RGMem reaches `63.87` with GPT-4o-mini and `74.01` with GPT-4.1, improving over the second-best baseline by **+7.08** and **+8.98** points, respectively.
- **LOCOMO:** RGMem reaches `78.92` with gpt-4o-mini and `86.17` with gpt-4.1-mini. It exceeds Full-Context with gpt-4o-mini (`78.92` vs. `71.41`) and approaches it with gpt-4.1-mini (`86.17` vs. `87.52`).
- **Stability–plasticity:** RGMem improves both factual retention and adaptation to the latest preference instead of merely moving along the baseline trade-off frontier.

<p align="center">
  <img src="assets/personamem-results.png" width="100%" alt="PersonaMem benchmark results">
</p>

<p align="center"><sub><em><strong>Table 1.</strong> PersonaMem accuracy (%). Parentheses report the margin over the second-best baseline.</em></sub></p>

<p align="center">
  <img src="assets/locomo-results.svg" width="80%" alt="LOCOMO benchmark results">
</p>

<p align="center"><sub><em><strong>Table 2.</strong> LOCOMO results (%) under LLM-as-a-judge evaluation.</em></sub></p>

<p align="center">
  <img src="assets/stability-plasticity.png" width="70%" alt="Stability-plasticity comparison on PersonaMem">
</p>

<p align="center"><sub><em><strong>Figure 5.</strong> RGMem achieves the strongest joint balance between Recall Facts and Latest Preference on PersonaMem.</em></sub></p>

## Analysis Highlights

The following analyses explain where RGMem's gains come from. Results are averaged over three runs; LOCOMO uses gpt-4.1 as the judge, while PersonaMem reports multiple-choice accuracy.

### Effective information density

<p align="center">
  <img src="assets/token-efficiency.png" width="72%" alt="Accuracy as a function of retrieved token count">
</p>

<p align="center"><sub><em><strong>Figure 3.</strong> Accuracy peaks near 3.8k retrieved tokens and then declines, showing that structured coarse-graining is more effective than indiscriminate context expansion.</em></sub></p>

### A critical evolution threshold

<p align="center">
  <img src="assets/phase-transition-locomo.png" width="49%" alt="Evolution threshold analysis on LOCOMO">
  <img src="assets/phase-transition-personamem.png" width="49%" alt="Evolution threshold analysis on PersonaMem">
</p>

<p align="center"><sub><em><strong>Figure 4.</strong> Both benchmarks exhibit an optimal regime around <code>θ_inf = 3</code>: lower thresholds overreact to noise, while higher thresholds make the profile too rigid.</em></sub></p>

### Component ablations

Every scale contributes to the final result. Removing the RG evolution operators hurts even when more context is retrieved.

| Benchmark | w/o L0 | w/o L1 | w/o RG | RGMem |
|:--|--:|--:|--:|--:|
| LOCOMO overall | 56.29 | 79.88 | 82.17 | **86.17** |
| PersonaMem overall | 47.22 | 54.31 | 57.37 | **63.87** |

On LOCOMO, `w/o RG` uses 4,354 context tokens versus RGMem's 3,788. On PersonaMem, it uses 8,479 versus 7,105. The complete system is therefore both more accurate and more information-dense.

### End-to-end efficiency

PersonaMem 128k cost is measured per session and includes memory construction, retrieval, and response generation.

| Method | Input tokens (k) | API calls | Accuracy (%) |
|:--|--:|--:|--:|
| Mem0 | 1,336.24 | 1,029.22 | 56.79 |
| MemoryOS | 3,489.71 | 3,278.42 | 54.23 |
| **RGMem** | **1,421.79** | **978.54** | **63.87** |

RGMem improves over the strongest baseline by **+7.08 points** with comparable input-token cost and fewer API calls than Mem0.

<details>
<summary><strong>Preference-shift dynamics</strong></summary>

<p align="center">
  <img src="assets/preference-shift-slow.png" width="49%" alt="RGMem dynamics under a slow preference shift">
  <img src="assets/preference-shift-abrupt.png" width="49%" alt="RGMem convergence after an abrupt preference reversal">
</p>

Under a slow shift, the profile remains stable until accumulated evidence triggers a rapid transition. Under an abrupt, dense reversal, the new profile converges in fewer than 200 turns.

</details>

<details>
<summary><strong>Additional mechanistic studies on the 20% PersonaMem subset</strong></summary>

- Fixed semantic anchors improve Latest Preference from `49.72` with L0+GraphRAG to `74.11`.
- Removing the correction term `Δ` reduces Latest Preference from `74.11` to `52.58`.
- Threshold-driven updates reach `65.32%`, compared with `53.72%` for high-frequency updates and `48.95%` for periodic rewrites.

These subset scores are mechanistic analyses and should not be compared directly with the full PersonaMem benchmark results above.

</details>

See the [paper](https://arxiv.org/abs/2510.16392) and Appendix B for full ablations, convergence analyses, parameter sensitivity, robustness tests, and computational cost.

## Quick Start

### Requirements

- Python `>=3.13`
- [uv](https://docs.astral.sh/uv/)
- Elasticsearch 8.x
- Redis
- Neo4j
- Supabase or Elasticsearch as the raw dialogue store
- API credentials for the selected LLM and embedding providers

### Install

```bash
git clone https://github.com/fenhg297/RGMem.git
cd RGMem
uv sync --frozen
cp .env.example .env
```

Fill in `.env`, then ensure Elasticsearch, Redis, Neo4j, and the configured history database are reachable. The default configuration uses:

- OpenAI-compatible LLM and embedding APIs;
- Supabase for raw dialogue history;
- Elasticsearch index `evaluation_persona_mem` for L0 memory;
- `text-embedding-3-large` with 3072-dimensional L0 vectors;
- `text-embedding-3-small` with 1536-dimensional L1 vectors.

### Start the API

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## API

### Trigger memory generation

L0 processing starts only when enough unprocessed dialogue has accumulated. The default threshold is 16 messages.

```bash
curl -X POST "http://127.0.0.1:8000/L0_memory/trigger_memory_generation" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_01",
    "role_id": 10001,
    "created_at": "2025-08-01T12:00:00.000Z"
  }'
```

### Retrieve multi-scale context

`retrieval_type` accepts `relevance`, `recency`, or `hybrid`.

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

The endpoint returns a `rag_context` string containing the selected L0 evidence and L1 knowledge. See the [implementation guide](docs/implementation.md) for the full data flow and configuration reference.

## Offline Pipeline and Evaluation

The repository includes scripts for running the pipeline without the online Redis listener:

```bash
# 1. Set LOCAL_HISTORY_FILE_PATH, then produce L0 memories.
uv run python scripts/1_produce_l0_memories.py

# 2. Set DATA_WORKING_DIRECTORY, then consume the generated IDs into L1.
uv run python scripts/2_consume_and_build_l1.py

# 3. Optionally update dirty hierarchical summaries.
uv run python scripts/trigger_dirty_summary.py

# 4. Set INPUT_DATASET_PATH, EVAL_USER_ID, and EVAL_ROLE_ID, then evaluate.
uv run python scripts/evaluate_l0_plus_l1.py
```

The release does not include benchmark data. Provide your own dialogue JSON/JSONL and evaluation data. Dialogue records should include `user_id`, `role_id`, `role`, `content`, and `created_at`; `session_id`, `user_name`, and `assistant_name` are optional.

## Project Structure

```text
RGMem/
├── config/                 # Settings and prompts
├── l0_memory/              # Episodic memory generation and ES retrieval
├── l1_memory/              # Knowledge-graph evolution and graph retrieval
├── l2_retrieval/           # Multi-scale retrieval orchestration
├── scripts/                # Offline ingestion and evaluation
├── docs/                   # Detailed implementation documentation
├── assets/                 # README figures and poster
├── main.py                 # FastAPI entry point
├── pyproject.toml
└── uv.lock
```

## Configuration and Security

Configuration is loaded from the project-root `.env` by `config/settings.py`. Start from [`.env.example`](.env.example) and never commit real credentials.

> [!WARNING]
> Earlier source snapshots contained secret-looking `.env` files. If any values in those snapshots were real, remove them from version control and rotate the affected credentials before publishing or deploying this repository.

## Citation

If RGMem is useful in your research, please cite:

```bibtex
@article{tian2025rgmem,
  title={Rgmem: Renormalization group-based memory evolution for language agent user profile},
  author={Tian, Ao and Lu, Yunfeng and Fan, Xinxin and Wang, Changhao and Zhou, Lanzhi and Zhang, Yeyao and Liu, Yanfang},
  journal={arXiv preprint arXiv:2510.16392},
  year={2025}
}
```

## Contact

For questions about the paper or code, please contact Ao Tian at `tianao@buaa.edu.cn` or open a GitHub issue.
