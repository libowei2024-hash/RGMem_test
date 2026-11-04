# file: l1_memory/graphs/value_conclusions.py

import json
from typing import List, Dict, Any, Optional
import re
from l1_memory.utils import logger
from datetime import datetime
import pytz
from l1_memory.graphs.utils import CONCLUSION_MATCHING_PROMPT_TEMPLATE


class ValueConclusionProcessor:
    """
    Processes high-value conclusions and updates them into the knowledge graph.
    """

    def __init__(self, llm_client: Any):
        """
        Initializes the processor.
        Args:
            llm_client: An initialized LLM client instance.
        """
        self.llm = llm_client
        logger.info("ValueConclusionProcessor initialized (matching mode).")

    def _filter_data(
            self,
            conclusions_data: Dict[str, Any],
            all_triplets: List[Dict[str, str]]
    ) -> (List[Dict[str, Any]], List[Dict[str, str]]):
        """Filters for high-value conclusions and prepares target triplets."""
        high_value_conclusions = [
            item for item in conclusions_data.get("conclusions", [])
            if item.get("tag") == "High-Value"
        ]
        logger.debug(f"Filtered {len(high_value_conclusions)} high-value conclusions.")

        target_triplets = all_triplets
        logger.debug(f"Using all {len(target_triplets)} triplets as matching candidates.")

        return high_value_conclusions, target_triplets

    def _recursive_search_for_list(self, data: Any) -> Optional[List[Dict]]:
        """
        Performs a deep recursive search for the target list within any JSON structure.
        Criteria for target list:
        1. It is a list.
        2. If not empty, its first element is a dictionary containing the 'conclusion_content' key.
        """
        if isinstance(data, list):
            if not data:
                return data  # An empty list could be a valid result
            if isinstance(data[0], dict) and "conclusion_content" in data[0]:
                return data

        if isinstance(data, dict):
            priority_keys = ["output", "result", "results", "response", "data", "matches"]
            for key in priority_keys:
                if key in data:
                    res = self._recursive_search_for_list(data[key])
                    if res is not None and len(res) > 0:
                        return res

            for key, value in data.items():
                if key not in priority_keys:
                    res = self._recursive_search_for_list(value)
                    if res is not None and len(res) > 0:
                        return res
        return None

    def _invoke_llm_for_matching(
            self,
            high_value_conclusions: List[Dict[str, Any]],
            target_triplets: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Builds a prompt and calls the LLM for matching conclusions to triplets."""
        if not high_value_conclusions or not target_triplets:
            logger.info("No high-value conclusions or target triplets to match, skipping LLM call.")
            return []

        conclusions_str = "\n".join([f"- {c['content']}" for c in high_value_conclusions])
        triplets_str = "\n".join(
            [f"- \"({t['source']}) --[{t['relationship']}]--> ({t['destination']})\"" for t in target_triplets])

        prompt = CONCLUSION_MATCHING_PROMPT_TEMPLATE.format(
            conclusions_str=conclusions_str,
            triplets_str=triplets_str
        )
        logger.debug("--- LLM Call Start: Conclusion Matching ---")

        try:
            response_str = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # Stage 1: String cleaning and initial parsing
            parsed_data = None
            if isinstance(response_str, str):
                clean_str = response_str.strip()
                match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', clean_str)
                json_str = match.group(1) if match else clean_str
                try:
                    parsed_data = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning(f"Initial JSON parsing failed, will attempt more robust extraction...")
                    raise
            else:
                parsed_data = response_str

            # Stage 2: Adaptive structure extraction
            match_results = []
            found_list = self._recursive_search_for_list(parsed_data)

            if found_list is not None:
                match_results = found_list
                if isinstance(parsed_data, dict):
                    logger.info(f"Recursively extracted target list from a nested structure (original type: dict).")
            elif isinstance(parsed_data, dict) and "conclusion_content" in parsed_data:
                match_results = [parsed_data]
                logger.info("Detected LLM returned a single object, automatically wrapping it in a list.")
            elif isinstance(parsed_data, list):
                match_results = parsed_data
            else:
                logger.error(
                    f"Critical: Could not identify a valid data structure from the LLM response. Type: {type(parsed_data)}, Preview: {str(parsed_data)[:200]}")
                return []

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse LLM matching results: {e}. Response preview: {str(response_str)[:200]}")
            return []
        except Exception as e:
            logger.error(f"An unknown error occurred during LLM matching call: {e}", exc_info=True)
            return []

        logger.info(f"LLM successfully returned and parsed {len(match_results)} matching results.")
        return match_results

    def match_conclusions_to_triplets(
            self,
            conclusions_data: Dict[str, Any],
            all_triplets: List[Dict[str, str]]
    ) -> Dict[tuple, List[Dict[str, Any]]]:
        """Executes the matching process from high-value conclusions to original triplets."""
        memory_id = conclusions_data.get("memory_id")
        memory_timestamp = conclusions_data.get("created_at")

        logger.info(f"--- Starting high-value conclusion matching for Memory ID: {memory_id} ---")
        high_value_conclusions, target_triplets = self._filter_data(conclusions_data, all_triplets)
        llm_match_results = self._invoke_llm_for_matching(high_value_conclusions, target_triplets)

        # Robustness check: filter out any non-dictionary elements from the LLM's response list.
        cleaned_match_results = [item for item in llm_match_results if isinstance(item, dict)]
        if len(cleaned_match_results) != len(llm_match_results):
            logger.warning(
                f"LLM match results contained non-dictionary elements which were filtered out. "
                f"Original count: {len(llm_match_results)}, Cleaned count: {len(cleaned_match_results)}")

        conclusion_map = {}
        for match in cleaned_match_results:
            conclusion_content = match.get("conclusion_content")
            matched_triplets_str = match.get("matched_triplets", [])

            if not conclusion_content:
                continue

            conclusion_entry = {
                "type": "L1_conclusion", "content": conclusion_content,
                "source_memory_id": memory_id, "timestamp": memory_timestamp
            }

            for triplet_str in matched_triplets_str:
                parsed = re.match(r"\((.*)\) --\[[^\]]+\]--> \((.*)\)", triplet_str.strip())
                if not parsed:
                    logger.warning(f"Could not parse triplet string from LLM: {triplet_str}")
                    continue
                source_name, dest_name = parsed.groups()
                original_triplet = next(
                    (t for t in all_triplets if t['source'] == source_name and t['destination'] == dest_name), None)
                if not original_triplet:
                    continue

                triplet_key = (
                original_triplet['source'], original_triplet['relationship'], original_triplet['destination'])
                if triplet_key not in conclusion_map:
                    conclusion_map[triplet_key] = []
                conclusion_map[triplet_key].append(conclusion_entry)

        logger.info(f"--- High-value conclusion matching completed for Memory ID: {memory_id} ---")
        return conclusion_map

    def process_and_update_graph(
            self,
            conclusions_data: Dict[str, Any],
            all_triplets: List[Dict[str, str]],
            entity_type_map: Dict[str, str],
            filters: Dict[str, str]
    ):
        """Executes the full process of handling high-value conclusions and updating the graph."""
        logger.debug("--- [Diagnostic] Entering ValueConclusionProcessor.process_and_update_graph ---")
        logger.debug(f"Input all_triplets count: {len(all_triplets)}")
        logger.debug("--------------------------------------------------------------------")

        memory_id = conclusions_data.get("memory_id")
        memory_timestamp = conclusions_data.get("created_at", datetime.now(pytz.utc).isoformat())

        logger.info(f"--- Processing high-value conclusions for Memory ID: {memory_id} ---")
        high_value_conclusions, target_triplets = self._filter_data(conclusions_data, all_triplets)
        match_results = self._invoke_llm_for_matching(high_value_conclusions, target_triplets)

        self._update_graph_with_conclusions(match_results, memory_id, memory_timestamp, filters)

        logger.info(f"--- High-value conclusion processing finished for Memory ID: {memory_id} ---")