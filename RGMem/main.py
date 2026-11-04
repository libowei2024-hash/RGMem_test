# main.py

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Literal
import asyncio
from datetime import datetime, time, timedelta, timezone

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, model_validator

# --- Project Imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from config import settings
from l0_memory.memory_server import MemoryServer
from l0_memory.persona_generation.generator import PersonaGenerator
from l1_memory.l1_listener import L1TaskListener
from l1_memory.dirty_summary import DirtySummaryScheduler
from l2_retrieval import L2Retriever
from l0_memory.utils.logger import console_logger
from l1_memory.memory.main import Memory
from l1_memory.graphs.neo4j import MemoryGraph
from l1_memory.configs.base import MemoryConfig
from l2_retrieval.l1_adapter import L1Adapter

# --- Global Service Instances ---
memory_server_instance: Optional[MemoryServer] = None
persona_generator_instance: Optional[PersonaGenerator] = None
l1_task_listener_instance: Optional[L1TaskListener] = None
dirty_summary_scheduler_instance: Optional[DirtySummaryScheduler] = None
l2_retriever_instance: Optional[L2Retriever] = None

# --- Background Task Handles ---
persona_scheduler_task: Optional[asyncio.Task] = None
l1_listener_task: Optional[asyncio.Task] = None
dirty_summary_task: Optional[asyncio.Task] = None

async def persona_scheduler(generator: PersonaGenerator):
    """Asyncio-based periodic task scheduler running in the background for persona generation."""
    console_logger.info("Background persona scheduler started.")
    while True:
        try:
            schedule_hour = settings.PERSONA_GENERATION_SCHEDULE_HOUR
            schedule_minute = settings.PERSONA_GENERATION_SCHEDULE_MINUTE
            now_utc = datetime.now(timezone.utc)
            # Calculate the next run time based on configuration
            target_time_today = now_utc.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
            if now_utc < target_time_today:
                next_run_time = target_time_today
            else:
                next_run_time = target_time_today + timedelta(days=1)
            sleep_seconds = (next_run_time - now_utc).total_seconds()
            console_logger.info(
                f"Next persona batch processing task scheduled for {next_run_time.isoformat()}, waiting {sleep_seconds:.0f} seconds...")
            await asyncio.sleep(sleep_seconds)
            console_logger.info("Time reached, starting persona batch processing task...")
            await asyncio.to_thread(generator.run_batch_processing)
            console_logger.info("Persona batch processing task completed for this cycle.")
        except asyncio.CancelledError:
            console_logger.info("Persona scheduler task cancelled, shutting down.")
            break
        except Exception as e:
            console_logger.error(f"Error in persona scheduler: {e}", exc_info=True)
            await asyncio.sleep(300)

async def dirty_summary_scheduler(scheduler: DirtySummaryScheduler):
    """Asyncio-based periodic task scheduler for L1 hierarchical summary updates."""
    console_logger.info("Background L1 hierarchical summary scheduler started.")
    while True:
        try:
            # Scheduled 2 hours later than the persona task
            schedule_hour = (settings.PERSONA_GENERATION_SCHEDULE_HOUR + 2) % 24
            schedule_minute = settings.PERSONA_GENERATION_SCHEDULE_MINUTE

            now_utc = datetime.now(timezone.utc)
            target_time_today = now_utc.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
            if now_utc < target_time_today:
                next_run_time = target_time_today
            else:
                next_run_time = target_time_today + timedelta(days=1)
            sleep_seconds = (next_run_time - now_utc).total_seconds()
            console_logger.info(f"Next L1 hierarchical summary task scheduled for {next_run_time.isoformat()}, waiting {sleep_seconds:.0f} seconds...")
            await asyncio.sleep(sleep_seconds)
            console_logger.info("Time reached, starting L1 hierarchical summary task...")
            await asyncio.to_thread(scheduler.run_batch_processing)
            console_logger.info("L1 hierarchical summary task completed for this cycle.")
        except asyncio.CancelledError:
            console_logger.info("L1 hierarchical summary scheduler task cancelled, shutting down.")
            break
        except Exception as e:
            console_logger.error(f"Error in L1 hierarchical summary scheduler: {e}", exc_info=True)
            await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Responsible for creating unique, singleton service instances and injecting them into dependent components.
    """
    console_logger.info("--- FastAPI App Starting Up: Initializing services... ---")
    global memory_server_instance, persona_generator_instance, l1_task_listener_instance, \
        dirty_summary_scheduler_instance, l2_retriever_instance, persona_scheduler_task, \
        l1_listener_task, dirty_summary_task

    # 1. Create unified L1 configuration object
    l1_config = MemoryConfig(
        llm=settings.L1_LLM_CONFIG,
        embedder=settings.L1_EMBEDDER_CONFIG,
        graph_store=settings.L1_GRAPH_STORE_CONFIG
    )

    # 2. Create the unique MemoryGraph instance (Core Singleton)
    memory_graph_singleton = MemoryGraph(config=l1_config)

    # 3. Create L1 Memory Facade dependent on MemoryGraph
    l1_memory_singleton = Memory(memory_graph=memory_graph_singleton)

    # --- Initialize Services with Dependency Injection ---

    # L0 Services (No L1 dependency)
    memory_server_instance = MemoryServer()
    persona_generator_instance = PersonaGenerator()

    # L1 Services (Dependent on MemoryGraph)
    l1_task_listener_instance = L1TaskListener(memory_graph=memory_graph_singleton)
    dirty_summary_scheduler_instance = DirtySummaryScheduler(memory_graph=memory_graph_singleton)

    # L2 Services (Dependent on L1 Adapter)
    l1_adapter_instance = L1Adapter(l1_memory=l1_memory_singleton)
    l2_retriever_instance = L2Retriever(l1_adapter=l1_adapter_instance)

    # Start all background tasks
    persona_scheduler_task = asyncio.create_task(persona_scheduler(persona_generator_instance))
    dirty_summary_task = asyncio.create_task(dirty_summary_scheduler(dirty_summary_scheduler_instance))
    l1_listener_task = asyncio.create_task(asyncio.to_thread(l1_task_listener_instance.start_listening))

    console_logger.info("--- All background services have been started. ---")

    yield

    # --- Shutdown Cleanup Logic ---
    console_logger.info("--- FastAPI App Shutting Down: Cleaning up services... ---")

    # 1. Gracefully cancel asyncio tasks
    tasks_to_cancel = [persona_scheduler_task, dirty_summary_task]
    for task in tasks_to_cancel:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # 2. Gracefully shutdown thread-based services
    # l1_task_listener_instance.shutdown() stops the loop first, then the thread pool
    if l1_task_listener_instance:
        await asyncio.to_thread(l1_task_listener_instance.shutdown)

    if memory_server_instance:
        await asyncio.to_thread(memory_server_instance.shutdown)

    if persona_generator_instance:
        await asyncio.to_thread(persona_generator_instance.shutdown)

    # 3. Wait for l1_listener_task to truly finish
    if l1_listener_task:
        try:
            await l1_listener_task
        except Exception as e:
            console_logger.warning(f"L1 listener task finished with an exception: {e}")

    console_logger.info("--- All services have been cleaned up. ---")


# Initialize FastAPI app with lifecycle manager
app = FastAPI(lifespan=lifespan)

# Create a router with prefix
router = APIRouter(prefix="/L0_memory")


# --- Data Models ---
class MemoryTriggerInput(BaseModel):
    user_id: str
    role_id: int
    created_at: str

class RagQueryInput(BaseModel):
    user_id: str
    role_id: int
    query: str
    entity_queries: Optional[List[Dict]] = None
    keywords: Optional[str] = None
    retrieval_type: Literal["relevance", "recency", "hybrid"] = "relevance"

    @model_validator(mode='after')
    def check_queries_or_keywords_exist(self):
        entity_queries = self.entity_queries
        keywords = self.keywords
        retrieval_type = self.retrieval_type

        if retrieval_type in ["relevance", "hybrid"]:
            if not entity_queries and not keywords:
                raise ValueError(
                    'When retrieval_type is "relevance" or "hybrid", one of "entity_queries" or "keywords" must be provided.')
        return self

class RagResponse(BaseModel):
    rag_context: str


# --- API Endpoints ---
@router.post("/trigger_memory_generation", status_code=202)
def trigger_memory_endpoint(trigger_input: MemoryTriggerInput):
    """
    Receives minimal memory trigger parameters and invokes the memory service.
    """
    if memory_server_instance:
        console_logger.info(f"[*] Received [Memory Generation] request: user_id={trigger_input.user_id}, role_id={trigger_input.role_id}")
        memory_server_instance.trigger_memory_processing(
            user_id=trigger_input.user_id,
            role_id=trigger_input.role_id,
            created_at=trigger_input.created_at
        )
        return {"message": "Memory generation task triggered successfully."}
    else:
        return {"message": "Error: MemoryServer not initialized."}, 503


@router.post("/retrieve_rag_context", response_model=RagResponse)
async def retrieve_rag_endpoint(query_input: RagQueryInput):
    """
    Unified retrieval endpoint. Accepts both new and legacy request formats,
    forwarding all parameters to the L2Retriever for processing.
    """
    if l2_retriever_instance:
        console_logger.info(
            f"[*] Received [Unified Retrieval] request for user: {query_input.user_id} "
            f"(Type: {query_input.retrieval_type})"
        )

        combined_context = await l2_retriever_instance.retrieve(
            query=query_input.query,
            user_id=query_input.user_id,
            role_id=query_input.role_id,
            retrieval_type=query_input.retrieval_type,
            entity_queries=query_input.entity_queries,
            keywords=query_input.keywords
        )

        return {"rag_context": combined_context}

    return {"rag_context": "Error: L2Retriever not initialized."}, 503


app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    console_logger.info("Please use 'uvicorn main:app --reload' to start the server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)