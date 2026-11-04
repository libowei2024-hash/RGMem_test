#l1_memory/graphs/tools.py
RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "Establish relationships among entities based on the provided text, including classifying each relationship's logical type as either 'event' or 'classification'.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought_process": {
                    "type": "string",
                    "description": "A step-by-step reasoning of how you built and classified each relationship triplet based on the rules. Explain your decisions, especially the classification logic."
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The source entity of the relationship."
                            },
                            "relationship": {
                                "type": "string",
                                "description": "The descriptive label of the relationship between the source and destination entities."
                            },
                            "destination": {
                                "type": "string",
                                "description": "The destination entity of the relationship."
                            },
                            "relationship_type": {
                                "type": "string",
                                "description": "The logical classification of the relationship.",
                                "enum": ["classification", "event"] # 严格约束分类
                            }
                        },
                        "required": [
                            "source",
                            "relationship",
                            "destination",
                            "relationship_type"
                        ]
                    }
                }
            },
            "required": ["thought_process", "entities"]
        }
    }
}

RELATIONS_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relations",
        "description": "Establish relationships among the entities based on the provided text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_entity": {
                                "type": "string",
                                "description": "The source entity of the relationship.",
                            },
                            "relationship": {
                                "type": "string",
                                "description": "The relationship between the source and destination entities.",
                            },
                            "destination_entity": {
                                "type": "string",
                                "description": "The destination entity of the relationship.",
                            },
                        },
                        "required": [
                            "source_entity",
                            "relationship",
                            "destination_entity",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}
EXTRACT_CORE_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_core_entities",
        "description": "Extracts core entities from the text, structuring them into logical paths from a root concept to a specific event.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_paths": {
                    "type": "array",
                    "description": "A list of logical paths. Each path is a list of entities, ordered from the most general to the most specific.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "The name of the extracted entity."
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "The type of the entity, must be either 'abstract' or 'event'."
                                        }
                                    },
                                    "required": ["name", "type"]
                                }
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            "required": ["entity_paths"]
        }
    }
}
SEARCH_ENTITY_EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_search_entities",
        "description": "Extracts core entities from a user's search query.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A core entity extracted from the user's query."
                    },
                    "description": "A list of core entities identified in the user's query."
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        },
    },
}