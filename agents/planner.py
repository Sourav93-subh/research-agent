import os
from dotenv import load_dotenv
load_dotenv()
import json
import uuid
from groq import AsyncGroq
from rich.console import Console
from models.state import ResearchState, SubTask

console = Console()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

PLANNER_SYSTEM = """You are a research planning expert. Your job is to break down a complex research
query into 3-5 focused sub-queries that can be searched independently and in parallel.

Rules:
- Each sub-query should cover a distinct angle of the main topic
- Sub-queries should be specific enough to yield useful search results
- Avoid overlap between sub-queries
- Return ONLY valid JSON — no markdown, no explanation

Output format:
{
  "subtasks": [
    {"id": "1", "query": "specific search query here"},
    {"id": "2", "query": "another specific query"}
  ]
}"""


async def planner_agent(state: ResearchState) -> ResearchState:
    console.log(f"[bold purple]🧠 Planner:[/bold purple] Breaking down query → '{state.user_query}'")

    response = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Research query: {state.user_query}"},
        ],
        temperature=0.3,
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
        subtasks = [
            SubTask(id=str(uuid.uuid4()), query=t["query"])
            for t in parsed["subtasks"]
        ]
    except (json.JSONDecodeError, KeyError) as e:
        console.log(f"[red]Planner parse error:[/red] {e}\nRaw: {raw}")
        subtasks = [SubTask(id=str(uuid.uuid4()), query=state.user_query)]

    console.log(f"[purple]Planner created {len(subtasks)} sub-tasks[/purple]")
    for t in subtasks:
        console.log(f"  [dim]→ {t.query}[/dim]")

    return state.model_copy(update={"subtasks": subtasks})