import unittest
from unittest.mock import patch

from l1_memory.graphs.neo4j import (
    ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS,
    MemoryGraph,
    validate_entity_paths,
)


def tool_response(entity_paths):
    return {
        "tool_calls": [
            {
                "name": "extract_core_entities",
                "arguments": {"entity_paths": entity_paths},
            }
        ]
    }


class StubLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class EntityPathSchemaRetryTest(unittest.TestCase):
    def setUp(self):
        self.path = [
            {"name": "User", "type": "abstract"},
            {"name": "Music Theory", "type": "event"},
        ]
        self.expected = [
            {
                "path": [
                    {"name": "user", "type": "abstract"},
                    {"name": "music_theory", "type": "event"},
                ]
            }
        ]

    def test_documented_path_wrapper_is_accepted(self):
        self.assertEqual(validate_entity_paths([{"path": self.path}]), self.expected)

    def test_bare_list_path_is_rejected(self):
        with self.assertRaises(TypeError):
            validate_entity_paths([self.path])

    @patch("l1_memory.graphs.neo4j.logger")
    def test_invalid_response_is_retried_until_official_schema(self, _logger):
        graph = MemoryGraph.__new__(MemoryGraph)
        graph.llm = StubLLM(
            [
                tool_response([self.path]),
                tool_response([{"path": self.path}]),
            ]
        )

        result = graph._retrieve_nodes_from_data("example", {})

        self.assertEqual(result, self.expected)
        self.assertEqual(len(graph.llm.calls), 2)
        retry_prompt = graph.llm.calls[1]["messages"][0]["content"]
        self.assertIn("Do not return a path as a bare array", retry_prompt)

    @patch("l1_memory.graphs.neo4j.logger")
    def test_invalid_responses_stop_after_retry_budget(self, _logger):
        graph = MemoryGraph.__new__(MemoryGraph)
        graph.llm = StubLLM(
            [tool_response([self.path])] * ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS
        )

        result = graph._retrieve_nodes_from_data("example", {})

        self.assertEqual(result, [])
        self.assertEqual(len(graph.llm.calls), ENTITY_PATH_EXTRACTION_MAX_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
