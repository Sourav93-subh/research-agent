import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq
from rich.console import Console
from models.state import ResearchState

console = Console()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

SYNTHESIZER_SYSTEM = """You are an expert research writer. You receive raw search results and write
a comprehensive, well-structured research report.

Report format:
# [Title based on the query]

## Executive Summary
2-3 sentence overview of key findings.

## Key Findings
Bullet-pointed main findings, organized by theme.

## Detailed Analysis
Prose sections diving deep into each major theme. Cite sources inline as [Source: URL].

## Conclusion
Summary and implications.

## Sources
Numbered list of all referenced URLs.

Rules:
- Be factual and grounded in the provided research
- Do not hallucinate — only include what the sources support
- Be concise but thorough
- Write in a professional, neutral tone"""


def _format_results_for_synthesis(state: ResearchState) -> str:
    parts = [f"Research query: {state.user_query}\n\n## Gathered Research\n"]
    all_sources = []

    for r in state.search_results:
        # Truncate each result to stay within token limits
        truncated = r.content[:600]
        parts.append(f"### {r.query}\n{truncated}\n")
        all_sources.extend(r.sources)

    unique_sources = list(dict.fromkeys(all_sources))
    parts.append(f"\n## Available Sources\n" + "\n".join(unique_sources))
    return "\n".join(parts)


async def synthesizer_agent(state: ResearchState) -> ResearchState:
    console.log(f"[bold purple]✍️  Synthesizer:[/bold purple] Writing final report")

    formatted = _format_results_for_synthesis(state)

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": formatted},
        ],
        temperature=0.4,
        max_tokens=4096,
    )

    report = response.choices[0].message.content.strip()
    console.log(f"[green]✓ Report generated ({len(report)} chars)[/green]")

    return state.model_copy(update={"final_report": report})