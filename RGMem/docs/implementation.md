# RGMem Implementation Guide

[English README](../README.md) · [中文 README](../README_zh.md)

> [!IMPORTANT]
> This document describes the released L0/L1/L2 implementation. The current snapshot is missing the `l0_memory.persona_generation` package and the related scheduling settings referenced by `main.py`, so the online service does not start without restoring or disabling that integration.

## 1. Overview

This project is a hierarchical long-term memory system built for AI Agents. Its primary functions are:
1.  **Data Ingestion**: Receives and processes raw dialogue history.
2.  **Knowledge Construction**: Transforms unstructured dialogues into a structured Knowledge Graph (KG).
3.  **Unified Retrieval**: Provides a unified API to retrieve context information from different memory layers.

## 2. Core Architecture

The system adopts a layered architecture with clear responsibilities for each layer:

*   **L0: Event Memory Layer**
    *   **Responsibility**: Processes raw dialogue streams into independent, structured **Topical Memories**. This serves as the raw source of truth for all memories.
    *   **Core Technologies**: Elasticsearch (Storage and Retrieval), Redis (Task Scheduling).

*   **L1: Knowledge Graph Layer**
    *   **Responsibility**: Consumes topical memories produced by L0. It uses LLMs to analyze and extract entities and relationships, constructing a structured **Knowledge Graph**. This layer is responsible for forming long-term, associative knowledge.
    *   **Core Technologies**: Neo4j (Graph Storage and Querying), Redis (Task Queues).

*   **L2: Unified Retrieval Layer**
    *   **Responsibility**: Acts as the single entry point for all memory queries. It sends requests **concurrently** to L0 (Keyword/Vector Retrieval) and L1 (Graph Retrieval), then aggregates, filters, and formats the results.
    *   **Core Technologies**: FastAPI, asyncio (Concurrency).

**Data Flow Diagram:**
```text
// Ingestion Flow
[Raw Dialogue] -> [L0: MemoryServer] -> [ES: Store L0 Memory] -> [Redis: Publish Task] -> [L1: Listener] -> [Neo4j: Build Graph]

// Retrieval Flow
[API Query] -> [L2: Retriever] --Concurrent--> [L0: RagServer(ES)] & [L1: MemoryGraph(Neo4j)] -> [Formatted JSON]
```

## 3. Key Workflows

### A. Ingestion Flow (L0)

This is the core pipeline for transforming dialogues into knowledge.

1.  **Trigger**: An external service calls the `/L0_memory/trigger_memory_generation` API (`main.py`), passing `user_id`, `role_id`, and the latest message timestamp.

2.  **Task Scheduling (L0)**:
    *   `l0_memory/memory_server.py` receives the request. It first attempts to acquire a Redis **Distributed Lock** based on `user_id` and `role_id` (`l0_memory/redis_db/state_manager.py`) to ensure dialogues for the same user are processed serially.
    *   It checks if the number of new messages since the last processing exceeds the batch threshold (`TOPIC_ANALYSIS_BATCH_SIZE`).

3.  **Asynchronous Processing (L0)**:
    *   If the threshold is met, `MemoryServer` fetches dialogue data from the configured history database (Supabase or Elasticsearch) and submits it to a **ThreadPoolExecutor**.
    *   A background thread executes the `process_chunk` method in `l0_memory/core/memory_processor.py`.

4.  **Memory Generation (L0)**:
    *   `MemoryProcessor` calls the LLM (via `llm_interface` factory) using `TOPIC_SPLIT_PROMPT` to segment long dialogues into independent, semantically complete topics.
    *   For each topic, it calls the LLM again using `L0_SUMMARY_GENERATION_PROMPT` to generate a structured JSON object containing `core_fact` and `conclusions`.
    *   It generates vectors for `core_fact` and `conclusions` using the client created by the `embedding_interface` factory.

5.  **Storage & Notification (L0)**:
    *   The final L0 memory document (including vectors) is written to Elasticsearch (`l0_memory/utils/elasticsearch_client.py`).
    *   `l0_memory/utils/redis_publisher.py` pushes the newly generated `memory_id` into a user-specific Redis Task Queue (List) and publishes a notification (content: `user_session_key`) to a global Pub/Sub channel (`l1_memory_tasks`). **This is the hand-off point from L0 to L1.**

6.  **State Update (L0)**:
    *   Upon task completion, the `on_task_completed` callback updates the user's `last_processed_created_at` timestamp in Redis and releases the distributed lock.

### B. Retrieval Flow (L0 Part)

1.  **Trigger**: The API receives a retrieval request, handled by `l2_retrieval/retriever.py (L2Retriever)`.
2.  **Scheduling**: `L2Retriever` creates an asynchronous task for L0.
3.  **Adapter**: This task calls the L0 retrieval service via `l2_retrieval/l0_adapter.py`.
4.  **Execution**: `l0_memory/rag_system/rag_server.py` receives the request and executes two types of retrieval:
    *   **Recency Retrieval (`get_recency_context`)**: Queries ES directly for the latest N L0 memories for the user.
    *   **Relevance Retrieval (`get_relevance_context`)**: Calls `l0_memory/rag_system/elasticsearch_retriever.py` to perform keyword, vector, or hybrid search to find `core_fact` and `conclusions` most relevant to the query.

## 4. Modules & Files (Config & L0 Memory)

### `config/`
*   `settings.py`: **Global Configuration Center**. Defines database connections, LLM/Embedding model configs, API keys, thread pool sizes, RAG parameters, etc.
*   `new_prompts.py`: Stores all carefully engineered instruction templates (Prompts) for LLMs.

### `l0_memory/`
*   `memory_server.py`: **Main Service Entry for L0**. Listens for API requests, manages distributed locks and thread pools, and schedules memory generation tasks.
*   `core/memory_processor.py`: **Core L0 Processor**. Encapsulates the pipeline from dialogue chunks to structured L0 memories (Topic Splitting -> Summary Generation -> Vectorization -> Storage -> Task Publishing).
*   `memory_generation/processor.py`: Encapsulates logic for calling LLMs to perform topic splitting and summary generation.
*   `data_processing/loader.py`: Utility for loading dialogue history and formatting it into LLM-friendly text.
*   `history_db/`: History Database **Adapters**.
    *   `base.py`: Abstract base class.
    *   `elasticsearch.py`, `supabase.py`: Concrete implementations for fetching raw dialogues.
*   `embedding_interface/` & `llm_interface/`: **Factory Pattern** implementations.
    *   `base.py`: Abstract base classes for clients.
    *   `factory.py`: Factories to dynamically create client instances (e.g., volc, openai) based on `settings.py`.
    *   `volc_client.py`, etc.: Concrete provider implementations.
*   `rag_system/`: L0 Retrieval System.
    *   `rag_server.py`: Entry point offering "Recency" and "Relevance" retrieval interfaces.
    *   `elasticsearch_retriever.py`: Core logic for Keyword, Vector, and Hybrid search (including RRF fusion) on Elasticsearch.
*   `redis_db/state_manager.py`: Encapsulates Redis operations for state management (session progress) and distributed locks.
*   `utils/`: General utilities.
    *   `elasticsearch_client.py`: Low-level interactions with Elasticsearch.
    *   `redis_publisher.py`: Logic for publishing to Redis queues/channels (L0->L1 bridge).
    *   `logger.py`, `schema.py`: Logging and Pydantic models.

## 5. Configuration (L0 Specifics)

Key configurations in `settings.py` related to L0:

*   **Models**:
    *   `LLM_PROVIDER`, `L0_LLM_CONFIGS`: Controls the LLM used for L0 generation and RAG keyword extraction.
    *   `EMBEDDING_PROVIDER`, `EMBEDDING_CONFIGS`: Controls the Embedding model for L0 vectorization.

*   **Database & Services**:
    *   `ES_CONFIG`, `ES_INDEX_...`: Elasticsearch connection and index names.
    *   `REDIS_CONFIG`, `REDIS_KEY_TEMPLATES`: Redis connection and key templates. Used for **State Management**, **Distributed Locks**, and **L1 Task Publishing**.
    *   `HISTORY_DB_PROVIDER`: Specifies the source database for raw dialogue history.

*   **Workflow Parameters**:
    *   `TOPIC_ANALYSIS_BATCH_SIZE`: Threshold for `memory_server.py` to trigger batch processing.
    *   `MEMORY_SERVER_MAX_WORKERS`: Thread pool size controlling L0 concurrency.
    *   `DEFAULT_PROCESSING_START_TIME`: Default start timestamp for new sessions.

*   **RAG Parameters**:
    *   `RAG_RETRIEVAL_MODE`: Mode for `elasticsearch_retriever.py` (`hybrid`, `keyword_only`, `vector_only`).
    *   `RAG_..._TOP_K`: Number of facts/conclusions to retrieve.
    *   `RAG_..._THRESHOLD`: Score thresholds for filtering.
    *   `RAG_RECENCY_CONTEXT_SIZE`: Number of latest L0 memories to return.

## 6. Key Workflows (Continued)

### A. Ingestion Flow (L1 Part)

This pipeline follows immediately after L0 task publishing.

7.  **Task Listening (L1)**:
    *   `l1_memory/l1_listener.py` is a long-running background service. It subscribes to the `l1_memory_tasks` channel via Redis **Pub/Sub**.
    *   Upon receiving a message (`user_session_key`), `L1TaskListener` submits it to a **thread pool** to avoid blocking the listener loop.

8.  **Task Processing (L1)**:
    *   The background thread acquires an **L1-level Redis Distributed Lock** for the `user_session_key` to ensure serial graph construction for the user.
    *   It then loops through the user-specific Redis **Task Queue** (`l1_queue:{user_session_key}`), popping (`LPOP`) pending `memory_id`s.

9.  **Data Loading & Graph Construction (L1)**:
    *   For each `memory_id`, `l1_memory/data_loader.py` loads and formats the L0 memory data from Elasticsearch.
    *   The `add` method in `l1_memory/graphs/neo4j.py (MemoryGraph)` is called. This is the **Core Logic of L1**:
        a.  **Entity Extraction**: Calls LLM via `_retrieve_nodes_from_data` with `EXTRACT_ENTITIES_PROMPT` to extract structured **Entity Paths** (e.g., `User -> Interest -> Music -> Classical`).
        b.  **Relation Extraction**: Calls LLM via `_establish_nodes_relations_from_data` with `EXTRACT_RELATIONS_PROMPT` to generate context-rich **Relationships**.
        c.  **Conclusion Matching**: Calls `l1_memory/graphs/value_conclusions.py` using `CONCLUSION_MATCHING_PROMPT` to attach high-value L0 conclusions to the most relevant graph relationships.
        d.  **Entity Linking/Merging**: Before writing, `MemoryGraph` computes vectors for new entities and performs vector similarity search in Neo4j (`_search_and_get_node_name`). If similar entities exist, an **LLM Decision Process** (`_llm_merge_decision`) determines whether to merge, ensuring graph quality.
        e.  **Writing to Graph**: Constructs a complex Cypher query to write entities, relationships, history, and conclusions **atomically** into Neo4j.

10. **Knowledge Evolution (L1)**:
    *   During writing, `MemoryGraph` increments `mentions` counts on relationships and `community_influence_score` on abstract nodes.
    *   Thresholds (`inference_threshold`, `summary_threshold`) trigger asynchronous updates in `l1_memory/graphs/inference.py (InferenceProcessor)` and `l1_memory/graphs/summary.py (SummaryProcessor)`.
        *   `InferenceProcessor`: Generates deeper **Inferences** for relationships incrementally.
        *   `SummaryProcessor`: Generates **Community Summaries** for abstract nodes and triggers **Hierarchical Summary** updates for parent nodes (via `is_dirty` flags).

11. **Task Completion (L1)**:
    *   After processing the user's queue, the L1 distributed lock is released. Failed tasks are moved to a **Dead Letter Queue**.

### B. Retrieval Flow (L1 & L2 Part)

1.  **Concurrent Scheduling (L2)**:
    *   `l2_retrieval/retriever.py (L2Retriever)` creates a parallel asynchronous task for L1 alongside the L0 task.

2.  **Adapter & Call (L2 -> L1)**:
    *   The task calls L1 retrieval via `l2_retrieval/l1_adapter.py`.
    *   `L1Adapter` calls `l1_memory/memory/main.py (Memory).retrieve`, which delegates to `l1_memory/graphs/graph_retrieval.py (GraphRetriever)`.

3.  **Graph Retrieval (L1)**:
    *   `extract_knowledge` in `GraphRetriever` is the **Core L1 Retrieval Logic**.
    *   It builds a Cypher query that filters candidate nodes and applies inline `vector.similarity.cosine` scoring for multiple query entities.
    *   The query fetches node properties, 1-hop neighbor relationships, inferences, history, conclusions, and summaries.
    *   `_format_results` aggregates and deduplicates the raw Neo4j output into a clean, entity-centric dictionary.

4.  **Integration & Formatting (L2)**:
    *   `L2Retriever` waits for all L0 and L1 tasks to complete.
    *   It calls `l2_retrieval/formatter.py` to format the Recency/Relevance data from L0 and the Graph Knowledge from L1 into a unified, structured string for upstream consumption.

## 7. Modules & Files (L1 & L2)

### `l1_memory/`
*   `l1_listener.py`: **Main Service Entry for L1**. A background consumer listening to Redis tasks and processing graph construction via a thread pool.
*   `data_loader.py`: **L0 -> L1 Data Adapter**. Loads L0 memories from ES and converts them for L1.
*   `graphs/neo4j.py (MemoryGraph)`: **L1 Core Business Logic**. Handles graph R/W operations, entity extraction, linking, and database writes.
*   `graphs/graph_retrieval.py (GraphRetriever)`: **L1 Core Retrieval Logic**. Builds/executes Cypher queries to extract knowledge efficiently.
*   `graphs/inference.py` & `summary.py`: **Knowledge Evolution**. Handle relationship inference and node summary updates.
*   `graphs/value_conclusions.py`: Matches and attaches L0 high-value conclusions to L1 graph relationships.
*   `graphs/tools.py` & `utils.py`: Schema definitions and Prompt templates for L1 function calling.
*   `memory/main.py (Memory)`: **L1 Top-Level Facade**. Provides a high-level API for external callers (L2), hiding `MemoryGraph` complexity.
*   `dirty_summary.py`: Background scheduler triggering global, periodic **Hierarchical Summary** updates via `SummaryProcessor`.
*   `llms/` & `embeddings/`: Factory implementations for L1's LLM and Embedding clients.

### `l2_retrieval/`
*   `retriever.py (L2Retriever)`: **L2 Orchestrator**. Receives requests, schedules **concurrent** retrieval from L0/L1, and aggregates results.
*   `l0_adapter.py` & `l1_adapter.py`: **Adapters**. Decouple L2 from direct L0/L1 dependencies.
*   `formatter.py`: **Presentation Layer**. Formats diverse data structures from L0 and L1 into a clean string.

## 8. Configuration (L1 & L2 Specifics)

*   **L1 Models & DB**:
    *   `L1_LLM_CONFIG`, `L1_EMBEDDER_CONFIG`: Models used for graph construction/retrieval.
    *   `L1_GRAPH_STORE_CONFIG`: Neo4j connection details.

*   **L1 Workflow Parameters**:
    *   `L1_LISTENER_MAX_WORKERS`: Thread pool size for L1 graph construction.
    *   `inference_threshold`, `summary_threshold`: Thresholds triggering asynchronous evolution tasks.
    *   `node_similar_threshold`: Similarity threshold for Entity Linking.
    *   `retrieval_similar_threshold`, `retrieval_top_k`: Thresholds for graph vector retrieval.

## 9. Run & Deploy

### A. Environment Setup

Initialize backend storage before running the service.

#### 1. Initialize Elasticsearch Index

Execute the following via Kibana Dev Tools. The index name and vector dimensions must match `ES_INDEX_TOPICAL_L0` and the selected embedding provider in `config/settings.py`; the example below follows the default OpenAI configuration.

```http
PUT /evaluation_persona_mem
{
  "mappings": {
    "properties": {
      "memory_id": { "type": "keyword" },
      "metadata": {
        "properties": {
          "user_id": { "type": "keyword" },
          "role_id": { "type": "keyword" },
          "created_at": { "type": "date" }
        }
      },
      "core_fact": { "type": "text" },
      "conclusions": { "type": "object", "enabled": false },
      "keywords": { "type": "keyword" },
      "atomic_units": {
        "type": "nested",
        "properties": {
          "unit_id": { "type": "keyword" },
          "unit_type": { "type": "keyword" },
          "content": { "type": "text" },
          "tag": { "type": "keyword" },
          "content_vector": {
            "type": "dense_vector",
            "dims": 3072
          }
        }
      }
    }
  }
}
```

### B. Start Service

Run the FastAPI service from the project root:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
This initializes all background listeners/schedulers and listens on `http://0.0.0.0:8000`.

## 10. API Usage Guide

### A. Core APIs

#### 1. Memory Generation

*   **Endpoint**: `POST /L0_memory/trigger_memory_generation`
*   **Purpose**: Triggers the L0 memory generation pipeline.
*   **Timing**: May be called after a dialogue turn; processing starts once the configured batch threshold is reached.
*   **Body**:
    ```json
    {
      "user_id": "string",
      "role_id": 10001,
      "created_at": "2025-05-23T10:00:00.000Z"
    }
    ```

#### 2. Unified Retrieval

*   **Endpoint**: `POST /L0_memory/retrieve_rag_context`
*   **Purpose**: Retrieves context from L0 and L1.
*   **Timing**: Called when the main chat model needs context.
*   **Body**:
    ```json
    {
      "user_id": "string",
      "role_id": 10001,
      "query": "What music has the user been interested in recently?",
      "retrieval_type": "hybrid",
      "entity_queries": [
        {
          "name": "Music",
          "retrieve": ["summary", "inference", "history", "conclusions"]
        }
      ],
      "keywords": "music interest"
    }
    ```

### B. Testing Methods

#### Using `curl`

```bash
# Example: Trigger Memory Generation
curl -X POST "http://127.0.0.1:8000/L0_memory/trigger_memory_generation" \
-H "Content-Type: application/json" \
-d '{
      "user_id": "test_user_01",
      "role_id": 10001,
      "created_at": "2025-08-01T12:00:00.000Z"
    }'

# Example: Execute Unified Retrieval
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
