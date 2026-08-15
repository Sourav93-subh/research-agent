import os
from dotenv import load_dotenv
load_dotenv()
import json
import uuid
from groq import AsyncGroq
from rich.console import Console
from models.state import ResearchState, CriticFeedback, SubTask

console = Console()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

CRITIC_SYSTEM = """You are a rigorous research quality analyst. You review gathered research results
and determine if they sufficiently answer the original query.

Your job:
1. Check if the results cover the query comprehensively
2. Identify any important gaps or missing angles
3. Decide if more searching is needed
4. If gaps exist, provide specific retry queries to fill them

Return ONLY valid JSON — no markdown, no explanation:
{
  "passed": true or false,
  "gaps": ["gap 1", "gap 2"],
  "retry_queries": ["specific query to fill gap 1"],
  "notes": "brief summary of your assessment"
}

Pass (passed: true) if the research is thorough enough to write a good report.
Fail (passed: false) only if there are critical missing pieces."""


def _format_results_for_critic(state: ResearchState) -> str:
    parts = [f"Original query: {state.user_query}\n"]
    for r in state.search_results:
        parts.append(f"--- Sub-query: {r.query} ---\n{r.content[:800]}\n")
    return "\n".join(parts)


async def critic_agent(state: ResearchState) -> ResearchState:
    console.log(f"[bold coral]🧐 Critic:[/bold coral] Evaluating {len(state.search_results)} results (attempt {state.retry_count + 1})")

    formatted = _format_results_for_critic(state)

    response = await client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        extra_body={"thinking": {"type": "disabled"}},
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"Review this research:\n\n{formatted}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        feedback = CriticFeedback(
            passed=parsed.get("passed", True),
            gaps=parsed.get("gaps", []),
            retry_queries=parsed.get("retry_queries", []),
            notes=parsed.get("notes", ""),
        )
    except (json.JSONDecodeError, KeyError) as e:
        console.log(f"[red]Critic parse error:[/red] {e}")
        feedback = CriticFeedback(passed=True, notes="Parse error — passing through")

    if feedback.passed:
        console.log(f"[green]✓ Critic passed:[/green] {feedback.notes}")
    else:
        console.log(f"[yellow]✗ Critic flagged gaps:[/yellow] {feedback.gaps}")

    new_state = state.model_copy(update={
        "critic_feedback": feedback,
        "retry_count": state.retry_count + 1,
    })

    if not feedback.passed and feedback.retry_queries:
        retry_subtasks = [
            SubTask(id=str(uuid.uuid4()), query=q)
            for q in feedback.retry_queries
        ]
        new_state = new_state.model_copy(update={"subtasks": retry_subtasks})

    return new_state