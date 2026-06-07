from langgraph.graph import StateGraph, END
from models.state import ResearchState
from agents.planner import planner_agent
from agents.searcher import searcher_agent
from agents.critic import critic_agent
from agents.synthesizer import synthesizer_agent
from rich.console import Console

console = Console()


def should_retry(state: ResearchState) -> str:
    """
    Routing function after critic runs.
    - If critic passed → go to synthesizer
    - If critic failed AND retries remain → go back to searcher
    - If out of retries → go to synthesizer anyway (best effort)
    """
    feedback = state.critic_feedback

    if feedback is None or feedback.passed:
        return "synthesize"

    if state.retry_count < state.max_retries:
        console.log(f"[yellow]↻ Retrying search (attempt {state.retry_count}/{state.max_retries})[/yellow]")
        return "retry"

    console.log("[yellow]Max retries reached — synthesizing with available data[/yellow]")
    return "synthesize"


def build_graph() -> StateGraph:
    # LangGraph needs a dict-based state; we wrap our Pydantic model
    graph = StateGraph(dict)

    async def planner_node(state: dict) -> dict:
        result = await planner_agent(ResearchState(**state))
        return result.model_dump()

    async def searcher_node(state: dict) -> dict:
        result = await searcher_agent(ResearchState(**state))
        return result.model_dump()

    async def critic_node(state: dict) -> dict:
        result = await critic_agent(ResearchState(**state))
        return result.model_dump()

    async def synthesizer_node(state: dict) -> dict:
        result = await synthesizer_agent(ResearchState(**state))
        return result.model_dump()

    def routing_node(state: dict) -> str:
        return should_retry(ResearchState(**state))

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "critic")

    # Conditional: critic decides to retry or synthesize
    graph.add_conditional_edges(
        "critic",
        routing_node,
        {
            "synthesize": "synthesizer",
            "retry": "searcher",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# Singleton compiled graph
research_graph = build_graph()


async def run_research(query: str) -> ResearchState:
    """Main entry point — runs the full pipeline."""
    console.rule(f"[bold]Starting research:[/bold] {query}")

    initial_state = ResearchState(user_query=query).model_dump()
    final_state = await research_graph.ainvoke(initial_state)

    return ResearchState(**final_state)