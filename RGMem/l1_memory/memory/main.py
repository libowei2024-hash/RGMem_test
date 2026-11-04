# l1_memory/memory/main.py

import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from l1_memory.configs.base import MemoryConfig
from l1_memory.memory.base import MemoryBase
from l1_memory.memory.setup import setup_config
from l1_memory.utils.factory import EmbedderFactory, LlmFactory
from l1_memory.graphs.neo4j import MemoryGraph

setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    """
    The main entry point for interacting with the L1 Memory system.
    This class acts as a facade, providing high-level methods to add,
    search, and retrieve information from the underlying knowledge graph.
    """

    def __init__(self, config: MemoryConfig = None, memory_graph: MemoryGraph = None):
        """
        Initializes the Memory facade.

        It can be initialized in two ways:
        1. By passing an existing `memory_graph` instance (dependency injection).
        2. By passing a `config` object, in which case it will create its own MemoryGraph.
        """
        if memory_graph:
            # Option 1: Dependency injection, use the provided instance directly
            self.graph = memory_graph
            self.config = self.graph.config  # Get config from the graph instance
        elif config:
            # Option 2: Standard initialization, create an internal instance
            self.config = config
            if self.config.graph_store and self.config.graph_store.config:
                self.graph = MemoryGraph(self.config)
            else:
                raise ValueError("GraphStore configuration is required when creating a new Memory instance.")
        else:
            raise ValueError("Either a 'config' object or a 'memory_graph' instance must be provided.")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        """Creates a Memory instance from a configuration dictionary."""
        try:
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    def add(
            self,
            data: Dict[str, Any],
            *,
            user_id: str,
            role_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adds a new memory to the knowledge graph."""
        if not all(k in data for k in ["memory_id", "core_fact"]):
            raise ValueError("Input 'data' must be a dict containing at least 'memory_id' and 'core_fact'.")

        filters = {"user_id": user_id}
        if role_id:
            filters["role_id"] = role_id

        return self.graph.add(input_data=data, filters=filters)

    def search(
            self,
            query: str,
            *,
            user_id: str,
            role_id: Optional[str] = None,
            default_retrieve_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a smart search using a natural language query.
        Uses a dedicated LLM prompt to extract search entities.
        """
        logger.info(f"Performing smart search for query: '{query}'")

        # Step 1: Extract search entities using the dedicated method in MemoryGraph
        extracted_entities = self.graph.extract_entities_for_search(query=query)

        if not extracted_entities:
            logger.warning("Could not extract any entities from the search query. Returning empty results.")
            return []

        logger.info(f"Extracted entities for search: {extracted_entities}")

        # Step 2: Build the structured entity queries
        if default_retrieve_fields is None:
            default_retrieve_fields = ["summary", "inference", "conclusions"]

        entity_queries = [
            {"name": entity, "retrieve": default_retrieve_fields}
            for entity in extracted_entities
        ]

        # Step 3: Call the retrieval method
        return self.retrieve(entity_queries=entity_queries, user_id=user_id, role_id=role_id)

    def retrieve(
            self,
            entity_queries: List[Dict],
            *,
            user_id: str,
            role_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Performs a precise, structured retrieval from the knowledge graph.
        This is the primary method for advanced querying.

        Args:
            entity_queries (List[Dict]): A list of structured query objects.
                Each object must contain 'name' (str) and 'retrieve' (List[str]).
            user_id (str): The ID of the user.
            role_id (str, optional): The ID of the role/agent.

        Returns:
            Dict[str, Any]: The structured retrieval results.
        """
        filters = {"user_id": user_id}
        if role_id:
            filters["role_id"] = role_id

        return self.graph.retriever.extract_knowledge(
            entity_queries=entity_queries, filters=filters
        )

    def format_retrieval_results_as_text(self, results: dict, entity_queries: list) -> str:
        """
        Formats the structured retrieval results into a human-readable text block.

        Args:
            results (dict): Raw dictionary returned by the `retrieve` or `search` method.
            entity_queries (list): Original list of entity queries sent to retrieval.

        Returns:
            str: Formatted string suitable for display or logging.
        """
        return self.graph.retriever.format_results_as_text(
            results=results,
            entity_queries=entity_queries
        )

    def delete_all(
            self,
            *,
            user_id: str,
            role_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Deletes all memories associated with a specific user and/or agent."""
        filters = {"user_id": user_id}
        if role_id:
            filters["role_id"] = role_id

        self.graph.delete_all(filters)
        return {"message": "All knowledge graph memories for the specified scope have been deleted."}

    def reset(self):
        """Completely resets the entire graph store. Use with caution."""
        logger.warning("Resetting the entire knowledge graph.")
        self.graph.reset()

    def get_all(
            self,
            *,
            user_id: str,
            role_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all relationship triples for a given user/agent from the graph.
        """
        filters = {"user_id": user_id}
        if role_id is not None:
            filters["role_id"] = role_id

        # Call the existing get_all method in the MemoryGraph instance
        return self.graph.get_all(filters=filters)