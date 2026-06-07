#!/usr/bin/env python
"""
CLI runner — use this to test the pipeline without the API.

Usage:
  python run.py "What is the current state of fusion energy?"
"""
import sys
import asyncio
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()
console = Console()


async def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python run.py 'Your research query'[/red]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    from graph.workflow import run_research
    state = await run_research(query)

    console.rule("[bold green]Research Complete[/bold green]")

    if state.error:
        console.print(f"[red]Error:[/red] {state.error}")
        sys.exit(1)

    console.print(Markdown(state.final_report))
    console.rule()
    console.print(f"[dim]Sub-tasks: {len(state.subtasks)} | Retries: {state.retry_count} | Sources: {sum(len(r.sources) for r in state.search_results)}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())