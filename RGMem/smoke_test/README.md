# RGMem 隔离冒烟测试

该目录只用于验证最小的 `L0 -> L1 -> L2` 数据流，不用于论文全量评测。

## 安全边界

- 冒烟测试不修改 `scripts/` 中的官方离线脚本。
- 每次运行必须使用新的 `run-id`，并生成独立用户 `rgmem_smoke_<run-id>`。
- 运行产物写入 `smoke_test/artifacts/<run-id>/`，该目录已被 Git 忽略。
- `init-es` 只在目标索引不存在时创建索引，绝不覆盖已有索引。
- L0、L1、L2 阶段都要求显式传入 `--confirm-api-and-write`。
- 不提供自动清理命令，避免误删数据库；清理时应先人工核对测试用户和索引。
- 不要使用正式用户 ID，也不要在同一个 `run-id` 上重复执行 L1。

## 测试数据

固定输入位于 `fixtures/minimal_dialogue.jsonl`，共 6 条消息，只描述一个稳定偏好：

- 长期喜欢科幻电影；
- 偏好太空探索和人工智能主题；
- 反复观看《星际穿越》和《银翼杀手》；
- 希望优先获得有思想深度的科幻电影推荐。

运行 L0 时，模板中的用户和 Session ID 会被替换为与 `run-id` 对应的隔离 ID。

## 建议执行顺序

以下命令仅为使用说明，本目录创建时没有执行这些命令。

进入项目根目录：

```powershell
cd "F:\1111111111sufework\zuhui_test_ZS\RGMEM_test\RGMem"
```

### 1. 只读预检

不会调用模型，也不会写数据库：

```powershell
uv run python smoke_test/run_smoke.py preflight
```

### 2. 初始化 L0 索引

仅当 `evaluation_persona_mem` 不存在时执行。该步骤会创建 ES 索引：

```powershell
uv run python smoke_test/run_smoke.py init-es --confirm-write
```

### 3. 运行 L0

选择一个从未使用过的 `run-id`。该步骤会调用 LLM/Embedding API，并写入 Elasticsearch：

```powershell
uv run python smoke_test/run_smoke.py l0 --run-id smoke-001 --confirm-api-and-write
```

预期产物：

```text
smoke_test/artifacts/smoke-001/
  input.jsonl
  manifest.json
  l0_memory_ids.txt
```

### 4. 运行 L1

该步骤会调用 LLM/Embedding API，并写入 Neo4j：

```powershell
uv run python smoke_test/run_smoke.py l1 --run-id smoke-001 --confirm-api-and-write
```

不要对同一个 `run-id` 重复运行 L1，否则关系计数可能重复累计。运行器会创建阶段标记并拒绝正常重复执行。

### 5. 运行 L2

该步骤执行一次受限的 L0+L1 联合检索。L1 实体查询会产生少量 Embedding API 调用：

```powershell
uv run python smoke_test/run_smoke.py l2 --run-id smoke-001 --confirm-api-and-write
```

结果同时显示在终端，并保存到：

```text
smoke_test/artifacts/smoke-001/l2_result.txt
```

## 成功标准

一次最小闭环只有同时满足以下条件才算通过：

1. L0 在 ES 中产生至少一条属于测试用户的记忆；
2. `l0_memory_ids.txt` 非空；
3. L1 在 Neo4j 中产生属于同一测试用户的节点和关系；
4. L2 返回 L0 事实以及 L1 图谱关系；
5. 正式用户数据、正式输入文件和官方离线脚本均未被修改。

单 Session 通常不足以触发 inference/summary 阈值，因此冒烟测试不把 `Σ/Δ` 或长期偏好演化视为成功标准。

