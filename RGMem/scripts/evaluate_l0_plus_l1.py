# file: evaluation/evaluate_l0_plus_l1.py

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import asyncio
from collections import defaultdict
import re
# --- Python 路径设置 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- 项目模块导入 ---
from config import settings
from l0_memory.llm_interface.factory import LlmFactory as L0LlmFactory
from l0_memory.utils.logger import console_logger

from l1_memory.memory.main import Memory
from l1_memory.graphs.neo4j import MemoryGraph
from l1_memory.configs.base import MemoryConfig
from l2_retrieval.l1_adapter import L1Adapter
from l2_retrieval.retriever import L2Retriever
import tiktoken
# ==============================================================================
#  >>>>> 评估任务配置 <<<<<
# ==============================================================================
EVAL_USER_ID = ""
EVAL_ROLE_ID = 101
INPUT_DATASET_PATH = ""

QUERY_STRUCTURING_PROMPT = """
#### Role
You are a highly intelligent **Unified Retrieval Strategy Planner**. Your mission is to deconstruct a user's query (which may include a question and multiple choice options) into a structured set of retrieval instructions. These instructions will drive a hybrid search system, querying both a structured Knowledge Graph (KG) and an unstructured text database. Your output must be a precise, actionable Python list of dictionaries that routes each generated keyword to the correct retrieval layer.

#### Core Task
Analyze the user's input (Question + Options) and generate a JSON list. Each element in the list will be a dictionary containing a keyword (`"name"`) and its designated KG retrieval layers (`"retrieve"`), which can be `summary`, `inference`, `conclusions`, or `history`.

---
### **Mental Workflow: A Three-Stage Strategy**

You must internally follow this strict three-stage thought process to arrive at the final output.

**Stage 1: Comprehensive Keyword Brainstorming**
First, generate a broad list of all potentially relevant keywords from BOTH the question and the provided options. This involves three parallel tracks:
*   **A. Foundational Keyword Extraction & Expansion**:
    *   Extract key common nouns and verbs from the question and all options.
    *   Expand nouns with synonyms and broader concepts (e.g., `charity race` -> `competition event`).
    *   Expand verbs with different forms and related concepts (e.g., `paint` -> `painted painting artwork`).
    *   **Do not expand proper nouns** (e.g., `The Palace Museum`).
*   **B. Text Search Enhancement Keywords**:
    *   Extract or generate keywords for time, location, and other metadata.
    *   **For absolute dates (e.g., 'October 13, 2023'), you must perform hierarchical decomposition**, generating the original string, the `YYYY-MM-DD` format, the `YYYY-MM` format, and the `YYYY` format as separate keywords (e.g., `October 13, 2023`, `2023-10-13`, `2023-10`, `2023`).
    *   Generate semantic intent words based on query tense (`history`, `record` for past; `plan`, `schedule` for future).
    *   **Crucially, low-value time words like `when`, `time`, `date` must be ignored and not included in the brainstorm list.**
*   **C. KG Abstract Entity Generation**:
    *   Act like a KG Architect. Based on the foundational keywords, generate potential high-level `abstract` "folder" entities.
    *   *Example*: From `paint`, `sunrise`, generate `Art Creation`, `Interests & Entertainment`.
    *   *Example*: From `visit`, `museum`, generate `Cultural Activities`, `Leisure Plan`, `Travel & Commute`.

**Stage 2: Core Intent Analysis & KG Candidate Selection**
Second, analyze the user's primary intent to select a handful of "elite" keywords for KG retrieval.
*   **Analyze the Query's Core Question**:
    *   "**Summary/Overall**..." queries point to `summary`.
    *   "**Why/Relationship**..." queries point to `inference`.
    *   "**What/Confirm**..." queries point to `conclusions`.
    *   "**Recall/Specifics of an event**..." queries point to `history`.
*   **Select 10-15 Elite KG Keywords**: From the brainstormed list, select the most potent keywords that directly represent KG entities (`event` or `abstract`). Prioritize nouns and conceptual phrases. These become your KG retrieval targets. All other brainstormed keywords will default to text-only search.

**Stage 3: Parameter Assignment**
Based on the previous stages, construct the full list of query objects.
*  **Assign `retrieve` Parameters**:
    *   For the **elite KG keywords** selected in Stage 2, assign the appropriate `summary`, `inference`, `conclusions`, `history` parameters based on your core intent analysis. A single keyword can have multiple parameters if the query is complex.
    *   For **all other keywords** from the brainstorming list, assign an **empty list** `[]` to the `retrieve` parameter, designating them for text-only search.

---
#### **Final Purification Rule (Highest Priority)**
**Before generating the final JSON**, you MUST perform a final check on your complete list of keywords. You must find and **completely remove any dictionaries** where the `"name"` key corresponds to one of the following user alias keywords: `Caroline`, `Melanie`, `Alex`. These words are considered universal noise and must not appear in the final output under any circumstances.

---
### **Output Structure & Examples**

Your final output must be a **single, valid JSON list**. Do not include any variable assignments, keys like "entity_queries_example", code block markers (like ```json), or any text outside of the JSON list itself. Your entire response must start with `[` and end with `]`.

**Example 1: Query is "When did the user visit the Forbidden City?"**
*   **Analysis**: Intent is to find a specific past event ("When did... visit"). This requires factual history and conclusions.
*   **Elite KG Keywords**: "Forbidden City", "Palace Museum", "Travel Plans", "Sightseeing", "Travel & Commute".
*   **Text Keywords**: "User", "When", "Visited", "visit", "trip", "history", "record", etc.
*   **Formatted Output Content**:
[
    {{"name": "Forbidden City", "retrieve": ["history", "conclusions"]}},
    {{"name": "Palace Museum", "retrieve": ["history", "conclusions"]}},
    {{"name": "Travel Plans", "retrieve": ["summary", "history"]}},
    {{"name": "Sightseeing", "retrieve": ["history"]}},
    {{"name": "Travel & Commute", "retrieve": ["summary"]}},
    {{"name": "User", "retrieve": []}},
    {{"name": "Visited", "retrieve": []}},
    {{"name": "visit", "retrieve": []}},
    {{"name": "trip", "retrieve": []}},
    {{"name": "history", "retrieve": []}},
    {{"name": "record", "retrieve": []}}
]

### User Input (Question + Options)
---
{query}
---

### Output (A pure JSON list of query objects)
"""
ANSWER_GENERATION_PROMPT = """
### Role
You are an intelligent and helpful Q&A assistant with a perfect memory of your past conversations with the user. Your task is to answer the [User Question] by leveraging this memory.

### Understanding the Context Format
The [Contextual Information] provided below is your memory, structured in Markdown. Here is how to interpret it:

*   **`## Recent/Relevant Memories (L0)`**: These are raw, unstructured facts and conclusions from recent or relevant conversations. They provide direct, verbatim evidence.
*   **`## Knowledge Graph Insights (L1)`**: This is structured, hierarchical knowledge derived from all past conversations. It reveals deeper patterns and relationships.
    *   **`### Entity: [Entity Name]`**: Represents a core concept or person. All information below it is related to this entity.
    *   **`Summary`**: A high-level, comprehensive profile of the user regarding a broad topic (e.g., their personality, their interest in "Art Creation"). Use this for general, thematic questions.
    *   **`Inference`**: A synthesized insight explaining the *'why'* behind the relationship between two entities. This is highly valuable for answering questions about motivations or connections.
    *   **`History`**: A specific, raw event that occurred. This is the ground-truth evidence for answering "what happened" or "when" questions.
    *   **`Conclusion`**: A high-value insight directly extracted from one or more `History` events.

#### **Key Interpretation Rule (Crucially Important)**
You must understand that **all** information in the context, including the hierarchical `(classification)` relationships, is **personalized and derived from this specific user's history**.

**For example:** a relationship like `Relationship (classification): (painting) -> (painting_animals)` does **not** mean this is a universal, encyclopedic fact. It means that **based on this user's past conversations**, "painting animals" has been identified as a specific sub-category of their general interest in "painting". This entire structure is a map of **the user's world**, not the real world. You must answer from this personalized perspective.

### Answering Strategy
You must follow the hierarchical strategy below to construct your answer:

1.  **Prioritize Contextual Information**:
    *   First, carefully read and interpret the [Contextual Information] using the guide above.
    *   Your answer **must be primarily based** on this information. Your response should reflect that you "remember" these details.

2.  **Use General Knowledge as a Supplement**:
    *   Only if the information is **completely missing** from the context, you may use your general knowledge.

3.  **Be Concise**:
    *   Your answer should be direct and to the point.

### Contextual Information (from the user's historical memory)
---
{context}
---

### User Question
---
{question}
---

### Your Answer
"""

LLM_JUDGE_PROMPT = """
Your task is to label a generated answer as 'CORRECT' or 'WRONG' based on a gold standard answer.
You will be given the following data:
(1) A question.
(2) A 'gold' (ground truth) answer.
(3) A 'generated' answer from another model.
RULES FOR JUDGEMENT:
- Be generous. If the generated answer contains the core information from the gold answer, it is CORRECT.
- The generated answer can be much longer and more conversational than the gold answer. This is acceptable.
- For time-related questions, different formats (e.g., "May 7th" vs "7 May 2023") are CORRECT if they refer to the same date. Relative time references (like "last Tuesday") are also CORRECT if they align with the gold answer's time period.
DATA TO EVALUATE:
---
Question: {question}
Gold answer: {standard_answer}
Generated answer: {generated_answer}
---
Your output MUST be a single, valid JSON object with one key, "label". The value must be either the string "CORRECT" or the string "WRONG".
Do not include any other text, explanations, or markdown.
Warning: Your response must be and can only be a pure JSON array, without any explanations, comments, or Markdown markup. Any additional characters will cause the program to fail.

Example of your required output:

{{"label": "CORRECT"}}
"""



# scripts/3_evaluate_L0+L1_hybrid.py

import os
import sys
import json
import asyncio
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import re
import tiktoken

# --- Setup Python Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Core Imports ---
from config import settings
from l0_memory.utils.logger import console_logger
from l0_memory.llm_interface.factory import LlmFactory as L0LlmFactory
from l2_retrieval import L2Retriever
from l2_retrieval.l1_adapter import L1Adapter
from l1_memory.memory.main import Memory
from l1_memory.graphs.neo4j import MemoryGraph
from l1_memory.configs.base import MemoryConfig

# ==============================================================================
#  >>>>> Configuration <<<<<
# ==============================================================================
INPUT_DATASET_PATH = "dataset/evaluation_dataset.json" # Please modify this path
EVAL_USER_ID = "123456789" # Test User ID
EVAL_ROLE_ID = 1

# --- Prompts ---
QUERY_STRUCTURING_PROMPT = """
You are an expert at structuring natural language queries into entity-based search instructions.
Your goal is to extract key entities and keywords from the user's question to search a knowledge graph.

Return a JSON list of objects, where each object has:
- "name": The entity or keyword string.
- "retrieve": A list of information types to retrieve (choose from ["history", "conclusions", "summary", "inference"]).

User Question: "{query}"

Output Format:
[
    {{"name": "Entity1", "retrieve": ["history", "summary"]}},
    {{"name": "Keyword2", "retrieve": []}}
]
"""

ANSWER_GENERATION_PROMPT = """
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question.
Question: {question}
Answer:
"""

LLM_JUDGE_PROMPT = """
You are an impartial judge evaluating the correctness of an AI's answer.

Question: {question}
Standard Answer: {standard_answer}
AI Generated Answer: {generated_answer}

Compare the AI Generated Answer with the Standard Answer.
If the AI answer captures the core meaning and facts of the standard answer, label it as "CORRECT".
Otherwise, label it as "WRONG".

Return ONLY a JSON object:
{{
    "label": "CORRECT" or "WRONG",
    "reason": "Brief explanation"
}}
"""

# ==============================================================================
#  >>>>> Helper Functions <<<<<
# ==============================================================================

def sanitize_llm_json_output(response_str: str) -> str:
    """
    A robust function to clean and extract non-standard JSON strings returned by LLMs.
    """
    # 1. Extract content wrapped in ```json ... ``` or ``` ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response_str, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        content = response_str

    # 2. Greedily extract content from the first '[' to the last ']'
    # This removes prefixes like "entity_queries_example = "
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        list_str = match.group(0)
    else:
        list_str = content  # Assume content is the list string if regex fails

    # 3. Remove trailing commas
    # This regex matches a comma followed by whitespace and a closing bracket/brace
    cleaned_str = re.sub(r',\s*([]}])', r'\1', list_str)

    return cleaned_str.strip()

def initialize_output_file(filepath: str, config_data: list):
    """Initializes the evaluation results file."""
    initial_structure = {"config": config_data, "qa": []}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(initial_structure, f, indent=2, ensure_ascii=False)


def append_result_to_json(filepath: str, new_qa_item: dict):
    """Appends a single result to the JSON file."""
    with open(filepath, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data['qa'].append(new_qa_item)
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.truncate()


def write_final_report(filepath: str, total_correct: int, total_questions: int,
                       category_stats: dict, total_retrieved_tokens: int):
    """Calculates final accuracy and category statistics, writing them to the file."""
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    average_tokens_per_query = total_retrieved_tokens / total_questions if total_questions > 0 else 0

    category_accuracies = {}
    sorted_categories = sorted(category_stats.keys())
    for category in sorted_categories:
        stats = category_stats[category]
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        category_accuracies[category] = f"{accuracy:.2f}%"

    final_report = {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy_percent": f"{overall_accuracy:.2f}%",
        "total_retrieved_context_tokens": total_retrieved_tokens,
        "average_tokens_per_query": f"{average_tokens_per_query:.2f}",
        "category_accuracies": category_accuracies,
        "category_raw_stats": {cat: category_stats[cat] for cat in sorted_categories}
    }

    console_logger.info("\n--- L0+L1 Comprehensive Evaluation Final Report ---")
    console_logger.info(json.dumps(final_report, indent=2, ensure_ascii=False))

    with open(filepath, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data['final_summary'] = final_report
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.truncate()


# ==============================================================================
#  >>>>> Core Evaluation Process <<<<<
# ==============================================================================
async def run_l0_l1_evaluation():
    """Executes the complete L0+L1 hybrid memory evaluation process."""
    console_logger.info("--- L0+L1 Hybrid Memory Evaluation Script Started ---")

    # --- Step 1: Initialize L2 and Evaluation Components ---
    console_logger.info("Initializing L2 Retriever and dependencies...")
    try:
        l1_config = MemoryConfig(
            llm=settings.L1_LLM_CONFIG,
            embedder=settings.L1_EMBEDDER_CONFIG,
            graph_store=settings.L1_GRAPH_STORE_CONFIG
        )
        memory_graph_singleton = MemoryGraph(config=l1_config)
        l1_memory_singleton = Memory(memory_graph=memory_graph_singleton)
        l1_adapter_instance = L1Adapter(l1_memory=l1_memory_singleton)
        l2_retriever = L2Retriever(l1_adapter=l1_adapter_instance)
        console_logger.info("L2 Retriever initialized successfully.")
    except Exception as e:
        console_logger.critical(f"Failed to initialize L2 Retriever, script cannot proceed: {e}", exc_info=True)
        return

    console_logger.info(f"Loading L0 LLM clients from settings (Provider: {settings.LLM_PROVIDER})...")
    provider = settings.LLM_PROVIDER
    provider_config = settings.L0_LLM_CONFIGS.get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found in L0_LLM_CONFIGS.")

    query_struct_model_name = provider_config["rag_keyword_model"]
    chat_model_name = provider_config["task_models"]["eval_chat_model"]
    judge_model_name = provider_config["task_models"]["judge_model"]

    factory_base_config = {
        "api_key": provider_config.get("api_key"),
        "base_url": provider_config.get("base_url"),
        **provider_config.get("params", {})
    }

    query_struct_client = L0LlmFactory.create(provider, {"model": query_struct_model_name, **factory_base_config},
                                              "QueryStructClient")
    chat_client = L0LlmFactory.create(provider, {"model": chat_model_name, **factory_base_config}, "EvalChatClient")
    judge_client = L0LlmFactory.create(provider, {"model": judge_model_name, **factory_base_config}, "EvalJudgeClient")
    console_logger.info(
        f"Query Structuring ({query_struct_model_name}), Chat ({chat_model_name}), Judge ({judge_model_name}) clients loaded.")

    # Initialize Tokenizer
    try:
        encoding = tiktoken.encoding_for_model(chat_model_name)
    except KeyError:
        console_logger.warning(f"Specific tokenizer for '{chat_model_name}' not found, using 'cl100k_base' as generic tokenizer.")
        encoding = tiktoken.get_encoding("cl100k_base")
    console_logger.info(f"Tokenizer '{encoding.name}' loaded.")

    # --- Step 2: Prepare Input/Output Files ---
    input_path = Path(INPUT_DATASET_PATH)
    output_filename = f"chat_evaluation_L1_{EVAL_USER_ID}_{input_path.stem}.json"
    output_filepath = str(Path("evaluation") / "results" / "LOCOMO" / "Caroline_Melanie" / "l1" / output_filename)
    os.makedirs(Path(output_filepath).parent, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    config_data = [
        {"evaluation_target": "L0+L1 Hybrid Memory (via L2 Retriever)"},
        {"llm_provider": provider},
        {"user_id": EVAL_USER_ID, "role_id": EVAL_ROLE_ID},
        {"query_struct_model": query_struct_model_name, "chat_model": chat_model_name, "judge_model": judge_model_name},
    ]
    initialize_output_file(output_filepath, config_data)

    # --- Step 3: Process Each QA Pair ---
    total_questions = len(dataset["qa"])
    total_correct_count = 0
    total_retrieved_tokens = 0
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for qa_item in tqdm(dataset["qa"], desc="Evaluating L0+L1 Hybrid Memory"):
        question = qa_item.get("question")
        reference_answer = qa_item.get("answer")
        category = str(qa_item.get("category", "unknown"))

        new_qa_item = qa_item.copy()

        try:
            # Step 3.1: Structure question into entity_queries
            struct_prompt = QUERY_STRUCTURING_PROMPT.format(query=question)
            structured_response_raw = await asyncio.to_thread(
                query_struct_client.get_structured_completion, struct_prompt, model_override=query_struct_model_name
            )

            parsed_list = []
            response_to_parse = ""

            if isinstance(structured_response_raw, list):
                parsed_list = structured_response_raw
            elif isinstance(structured_response_raw, str):
                response_to_parse = structured_response_raw
            elif isinstance(structured_response_raw, dict):
                response_to_parse = json.dumps(structured_response_raw, ensure_ascii=False)

            if response_to_parse:
                cleaned_json_str = sanitize_llm_json_output(response_to_parse)
                try:
                    parsed_list = json.loads(cleaned_json_str)
                except json.JSONDecodeError:
                    console_logger.warning(f"Failed to parse JSON list after sanitization: {cleaned_json_str}")

            entity_queries = []
            if isinstance(parsed_list, list):
                for item in parsed_list:
                    if isinstance(item, dict) and "name" in item:
                        entity_queries.append(item)
                    elif isinstance(item, str):
                        entity_queries.append({"name": item, "retrieve": []})
            else:
                console_logger.warning(f"Parsed entity_queries is not a list, but {type(parsed_list)}. Using empty list.")

            new_qa_item["generated_entity_queries"] = entity_queries

            # Step 3.2: Call L2 Retriever for hybrid retrieval
            l2_context_xml = await l2_retriever.retrieve(
                query=question,
                user_id=str(EVAL_USER_ID),
                role_id=int(EVAL_ROLE_ID),
                retrieval_type="relevance",
                entity_queries=entity_queries
            )
            new_qa_item["retrieved_context"] = l2_context_xml

            # Calculate retrieved tokens
            retrieved_token_count = len(encoding.encode(l2_context_xml))
            new_qa_item["retrieved_context_token_count"] = retrieved_token_count
            total_retrieved_tokens += retrieved_token_count

            # Step 3.3: Generate answer
            prompt = ANSWER_GENERATION_PROMPT.format(context=l2_context_xml, question=question)
            generated_answer = await asyncio.to_thread(
                chat_client.get_completion, prompt, model_override=chat_model_name
            )
            new_qa_item["generated_answer"] = generated_answer

            # Step 3.4: LLM Judge
            judge_prompt = LLM_JUDGE_PROMPT.format(
                question=question,
                standard_answer=reference_answer,
                generated_answer=generated_answer
            )
            judge_response = await asyncio.to_thread(
                judge_client.get_structured_completion, judge_prompt, model_override=judge_model_name
            )
            judge_label = judge_response.get("label", "WRONG")
            new_qa_item["llm_judge"] = judge_label

            # Update statistics
            category_stats[category]["total"] += 1
            if judge_label == "CORRECT":
                total_correct_count += 1
                category_stats[category]["correct"] += 1

        except Exception as e:
            console_logger.error(f"Error processing question '{question}': {e}", exc_info=True)
            new_qa_item["generated_answer"] = f"ERROR: {e}"
            new_qa_item["llm_judge"] = "ERROR"
            category_stats[category]["total"] += 1

        append_result_to_json(output_filepath, new_qa_item)

    # --- Step 4: Write Final Report ---
    write_final_report(output_filepath, total_correct_count, total_questions, category_stats, total_retrieved_tokens)
    console_logger.info(f"\n✅ L0+L1 Comprehensive Evaluation Completed! Results saved to {output_filepath}")


if __name__ == "__main__":
    asyncio.run(run_l0_l1_evaluation())