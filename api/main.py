import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from graph.workflow import run_research
from models.state import ResearchState
from rich.console import Console

load_dotenv()
console = Console()

# In-memory job store (swap for Redis in production)
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    console.log("[green]Research Agent API started[/green]")
    yield
    console.log("[yellow]Shutting down[/yellow]")


app = FastAPI(
    title="Multi-Agent Research API",
    description="Autonomous research pipeline powered by Claude + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ──────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    max_retries: int = 2


class ResearchResponse(BaseModel):
    job_id: str
    status: str
    query: str


class JobResult(BaseModel):
    job_id: str
    status: str           # pending | running | done | failed
    query: str
    report: str | None = None
    sources: list[str] = []
    subtask_count: int = 0
    retry_count: int = 0
    duration_seconds: float | None = None
    error: str | None = None


# ── Background job runner ────────────────────────────────────────────────────

async def _run_job(job_id: str, query: str, max_retries: int):
    jobs[job_id]["status"] = "running"
    start = time.time()
    try:
        state: ResearchState = await run_research(query)
        jobs[job_id].update({
            "status": "done",
            "report": state.final_report,
            "sources": list({
                s
                for r in state.search_results
                for s in r.sources
            }),
            "subtask_count": len(state.subtasks),
            "retry_count": state.retry_count,
            "duration_seconds": round(time.time() - start, 2),
        })
    except Exception as e:
        console.log(f"[red]Job {job_id} failed:[/red] {e}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "duration_seconds": round(time.time() - start, 2),
        })


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "research-agent"}


@app.post("/research", response_model=ResearchResponse, status_code=202)
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Submit a research query. Returns a job_id immediately.
    Poll /research/{job_id} to get results.
    """
    import uuid
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "query": req.query,
        "report": None,
        "sources": [],
        "subtask_count": 0,
        "retry_count": 0,
        "duration_seconds": None,
        "error": None,
    }

    background_tasks.add_task(_run_job, job_id, req.query, req.max_retries)

    return ResearchResponse(job_id=job_id, status="pending", query=req.query)


@app.get("/research/{job_id}", response_model=JobResult)
async def get_result(job_id: str):
    """Poll this endpoint to check job status and retrieve the report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResult(**jobs[job_id])


@app.get("/research", response_model=list[JobResult])
async def list_jobs():
    """List all jobs (for debugging)."""
    return [JobResult(**j) for j in jobs.values()]


@app.delete("/research/{job_id}", status_code=204)
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del jobs[job_id]