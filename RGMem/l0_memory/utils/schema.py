# l0_memory/utils/schema.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class created_atRange(BaseModel):
    start_created_at: str
    end_created_at: str

class MessageIdRange(BaseModel):
    start_message_id: str
    end_message_id: str

class AnalysisDetails(BaseModel):
    engagement_score: float
    confidence_level: str

class Metadata(BaseModel):
    session_id: str
    user_id: str
    role_id: int
    created_at_range: created_atRange
    created_at: str
    time_since_last_topic: str
    analysis_details: AnalysisDetails

class Conclusion(BaseModel):
    content: str
    tag: str


class AtomicUnit(BaseModel):
    unit_id: str
    unit_type: str  # 'core_fact' or 'conclusion'
    content: str
    tag: Optional[str] = None
    content_vector: List[float]

class L0Memory(BaseModel):
    memory_id: str
    metadata: Metadata
    core_fact: str
    conclusions: List[Conclusion]
    keywords: List[str]
    summary_text: Optional[str] = None
    summary_vector: Optional[List[float]] = None
    atomic_units: Optional[List[AtomicUnit]] = None

