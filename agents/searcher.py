import asyncio
from rich.console import Console
from models.state import ResearchState, SearchResult, SubTask
from tools.search import SearchTool

console = Console()


async def _run_single_search(
    subtask: SubTask,
    tool: SearchTool,
) -> SearchResult:
    result = await tool.search(subtask.query)
    return SearchResult(
        subtask_id=subtask.id,
        query=subtask.query,
        content=result["content"],
        sources=result["sources"],
    )


async def searcher_agent(state: ResearchState) -> ResearchState:
    console.log(f"[bold teal]🔎 Searcher:[/bold teal] Running {len(state.subtasks)} searches in parallel")

    tool = SearchTool()

    tasks = [_run_single_search(subtask, tool) for subtask in state.subtasks]
    results: list[SearchResult] = await asyncio.gather(*tasks)

    # Merge with any existing results (handles retries)
    existing_ids = {r.subtask_id for r in state.search_results}
    merged = list(state.search_results) + [r for r in results if r.subtask_id not in existing_ids]

    console.log(f"[teal]Searcher collected {len(merged)} total results[/teal]")
    return state.model_copy(update={"search_results": merged})