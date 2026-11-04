# PersonaMem 单用户多轮 RGMem 更新机制诊断

该 runner 固定选择 Kanoa Manu 的一个 `shared_context_id`，按原始顺序分成五轮，持续写入同一个隔离测试用户。目标是观察 RGMem 的 L0 分段、L1 `ADD/UPDATE`、阈值触发的即时演化以及 native dirty-summary 整合，不做 PersonaMem 多选题准确率复现。

旧运行 `personamem-kanoa-multiturn-001` 是 pilot 证据；`personamem-kanoa-multiturn-002` 保留为包含 L1 工程空写入的失败证据。协议 v4 使用 `RGMem with strict entity-path schema retry` 和新的 `run-id`，例如 `kanoa-multiturn-003`；不要在同一个用户上重放已经启动的轮次。

## 观测边界

- 原始数据没有 `created_at`，runner 用 `2026-01-01T00:00:00Z + original_index 分钟` 生成确定性排序时间；
- `system` 消息不送入 L0，user/assistant 的原始 `content` 不改写；
- 索引 51 是 assistant-role 记录，但正文内嵌 `User:`，该异常保留并写入诊断；
- 五轮是执行窗口，不声明为原始数据中的真实 session；
- 该设计属于同一用户跨五个顺序窗口的多轮长期记忆更新；由于五轮共用一个 `session_id`，不把它表述为五个数据库 session；
- `graph.add` 内部同时包含 L1 写入和阈值触发的 inference/community-summary，因此这一段按联合步骤记录，不能从外部 runner 伪装成两个独立因果步骤；
- dirty-summary 使用当前测试用户和 role 的原生 `summarize_dirty_nodes`，不会调用扫描全图所有用户的全局 scheduler；
- 没有实际 summary 改写的 dirty scan 记为 `effective_consolidation=false`，不能算作一次整合压力。
- 工程补丁严格执行官方 `{"path": [...]}` tool schema；若 LLM 返回直接的 `[...]` 或其他非法结构，则加入明确的 schema 纠正提示重新请求，最多重试两次（共三次请求），不接收错误结构；
- 每个 L0 调用 `graph.add` 后，协议 v4 同时检查 RGMem 返回值和图谱 diff；若三次请求后仍为空，则记录 `round_failure.json` 与 `.failed` 并立即停止，不写 `.completed`；该情况记为运行无效，不计入语义 UPDATE ERROR。

## A/K 标注

`prepare` 会生成 `diagnostic_labels.json`。其中：

- `A`：本次证据支持应当改变的最小偏好范围；
- `K`：修订后仍应保留的范围外偏好；
- `probe_A` / `probe_K`：分别检查范围内更新和范围外保留。

该文件默认是 `candidate_requires_human_review`，不会传给 L0、L1 或 L2，也不会参与 RGMem 更新。人工审核后才能把它当作判定“正确更新/误伤”的金标准。

## 开始方式

以下两个命令不调用 LLM，也不写 ES/Neo4j：

```powershell
uv run python run_personamem_single_user.py inspect
uv run python run_personamem_single_user.py prepare --run-id kanoa-multiturn-003
```

先打开并审核：

```text
artifacts/personamem-kanoa-multiturn-003/diagnostic_labels.json
```

然后由实验操作者逐步执行。`round` 会调用 LLM/Embedding API，并写入 Elasticsearch 和 Neo4j；`consolidate` 在存在 dirty 节点时会调用 L1 LLM、写入 Neo4j并执行整合后的 L2 读取。协议 v3 的主分析聚焦图谱中的 ADD/UPDATE、A/K 作用域和整合前后快照，不启用尚未校准的 L2 diagnostic probes：

```powershell
uv run python run_personamem_single_user.py round --run-id kanoa-multiturn-003 --round 1 --confirm-api-and-write
uv run python run_personamem_single_user.py consolidate --run-id kanoa-multiturn-003 --after-round 1 --confirm-api-and-write

uv run python run_personamem_single_user.py round --run-id kanoa-multiturn-003 --round 2 --confirm-api-and-write
uv run python run_personamem_single_user.py consolidate --run-id kanoa-multiturn-003 --after-round 2 --confirm-api-and-write

uv run python run_personamem_single_user.py round --run-id kanoa-multiturn-003 --round 3 --confirm-api-and-write
uv run python run_personamem_single_user.py consolidate --run-id kanoa-multiturn-003 --after-round 3 --confirm-api-and-write

uv run python run_personamem_single_user.py round --run-id kanoa-multiturn-003 --round 4 --confirm-api-and-write
uv run python run_personamem_single_user.py consolidate --run-id kanoa-multiturn-003 --after-round 4 --confirm-api-and-write

uv run python run_personamem_single_user.py round --run-id kanoa-multiturn-003 --round 5 --confirm-api-and-write
uv run python run_personamem_single_user.py consolidate --run-id kanoa-multiturn-003 --after-round 5 --confirm-api-and-write
```

新协议要求每轮之后先完成对应的 `consolidate`，才能开始下一轮。这样 round N 的即时变化和其后的整合变化不会混入 round N+1。

`--with-diagnostic-probes` 会增加 L0 relevance embedding/retrieval 次数，但不会更新记忆。若先控制成本，可以省略该参数；L0 provenance、L1 checkpoint 和 dirty-summary diff 仍会完整记录，只是 `l2_diagnostic_probes.json` 的 `enabled` 为 `false`。

## 本地查看与汇总

以下命令只读取本地产物，不调用 LLM，也不连接数据库：

```powershell
uv run python run_personamem_single_user.py status --run-id kanoa-multiturn-003
uv run python run_personamem_single_user.py report --run-id kanoa-multiturn-003
```

每轮主要产物：

```text
round-XX/
  input.jsonl
  input_manifest.json
  l0_memory_ids.txt
  l0_provenance_trace.json
  add_update_audit.jsonl
  checkpoints/
    round-before.json
    l0-01-before-l1.json
    l0-01-after-l1.json
    ...
    round-after.json
  l2_entity_queries.json
  l2_result.txt
  l2_diagnostic_probes.json
  round_summary.json
  consolidation/
    before.json
    after.json
    consolidation_report.json
```

`l0_provenance_trace.json` 记录 topic splitter 每次收到的窗口、返回的分段边界、每个 L0 的原始索引、user evidence 索引、assistant context 索引、虚拟 EOF，以及 LLM 自报的 `metadata.created_at_range`。这可以直接检查“实际切片边界”和“LLM 元数据范围”是否发生偏移。

`checkpoints/` 保存每个 L0 写入 L1 前后的完整用户图快照，包括 history、conclusions、inference、summary、dirty flag 和计数器。`add_update_audit.jsonl` 保存对应 diff、每次 inference/community-summary 阈值检查及是否触发，并明确标记 `graph.add` 是联合的写入/自动演化边界。

若某次 L1 写入为空，该轮不会生成 `.completed`，而会保留 `.started`、`.failed`、`round_failure.json`、已完成的 L0/L1 checkpoint 和审计日志。`status` 会显示 `failed`，`report` 也会统计失败轮次与 `empty_l1_writes`。这类失败是实验结果，不能当作成功轮次继续 consolidation。

`consolidation_report.json` 记录 dirty 节点前后集合、真实 summary 改写节点和有效改写次数。重复空扫描不算整合压力。

运行 `report` 后，运行目录根部生成：

```text
add_update_report.json
add_update_report.md
```

它汇总已经完成轮次的每个 L0 provenance、节点/关系 `ADD/UPDATE` 和 dirty-summary 实际改写，不把 smoke 成功表述为准确率、完整复现或机制正确性证据。
