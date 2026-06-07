import os
import asyncio
from typing import Any
from tavily import AsyncTavilyClient
from rich.console import Console

console = Console()


class SearchTool:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set in environment")
        self.client = AsyncTavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
    ) -> dict[str, Any]:
        """
        Run a single search query. Returns dict with:
          - content: combined text from top results
          - sources: list of URLs
        """
        console.log(f"[cyan]🔍 Searching:[/cyan] {query}")
        try:
            response = await self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_raw_content=False,
            )
            results = response.get("results", [])

            content_parts = []
            sources = []

            for r in results:
                title = r.get("title", "")
                body = r.get("content", "")
                url = r.get("url", "")
                if body:
                    content_parts.append(f"### {title}\n{body}")
                if url:
                    sources.append(url)

            combined = "\n\n".join(content_parts) if content_parts else "No results found."
            return {"content": combined, "sources": sources}

        except Exception as e:
            console.log(f"[red]Search failed:[/red] {e}")
            return {"content": f"Search failed: {e}", "sources": []}

    async def batch_search(
        self,
        queries: list[str],
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Run multiple searches in parallel."""
        tasks = [self.search(q, max_results=max_results) for q in queries]
        return await asyncio.gather(*tasks)