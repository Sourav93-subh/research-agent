from typing import Annotated, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class SubTask(BaseModel):
    id: str
    query: str
    status: str = "pending"  # pending | running | done | failed


class SearchResult(BaseModel):
    subtask_id: str
    query: str
    content: str
    sources: list[str] = Field(default_factory=list)


class CriticFeedback(BaseModel):
    passed: bool
    gaps: list[str] = Field(default_factory=list)
    retry_queries: list[str] = Field(default_factory=list)
    notes: str = ""


class ResearchState(BaseModel):
    # Input
    user_query: str = ""

    # Planner output
    subtasks: list[SubTask] = Field(default_factory=list)

    # Searcher output
    search_results: list[SearchResult] = Field(default_factory=list)

    # Critic output
    critic_feedback: CriticFeedback | None = None
    retry_count: int = 0
    max_retries: int = 2

    # Final output
    final_report: str = ""

    # Internal
    error: str | None = None
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True