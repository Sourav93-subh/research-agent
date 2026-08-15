<div align="center">

<br/>

```
██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║
██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
█████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
```

**Autonomous multi-agent research pipeline powered by LangGraph + Groq**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://research-agent-9pwhw1mm1-sourav93-subhs-projects.vercel.app)
[![API](https://img.shields.io/badge/API-Railway-blueviolet?style=for-the-badge&logo=railway)](https://research-agent-production-2bab.up.railway.app/docs)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

![Research Agent Demo](https://raw.githubusercontent.com/Sourav93-subh/research-agent/main/ui/preview.png)

<br/>

</div>

---

## ✦ What it does

Given any research query, the system autonomously:

1. **Plans** — breaks the query into 3–5 focused sub-queries
2. **Searches** — runs all sub-queries in parallel via Tavily web search
3. **Critiques** — evaluates coverage, triggers retries if gaps are found
4. **Synthesizes** — writes a structured report with citations and sources

The whole pipeline runs in ~10–15 seconds and produces a professional research report.

---

## ✦ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Planner   │  ← Breaks query into parallel sub-tasks (Groq LLM)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Searcher × N  (runs in parallel)   │  ← Tavily web search per sub-task
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Critic    │  ← Validates quality, triggers retry if gaps found
└──────┬──────┘
       │
    passed? ──── no ──→ back to Searcher (max 2 retries)
       │ yes
       ▼
┌──────────────┐
│ Synthesizer  │  ← Writes final structured report
└──────────────┘
       │
       ▼
  Final Report
```

---

## ✦ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| LLM | Groq — gpt-oss-120b |
| Web search | Tavily |
| API framework | FastAPI + async |
| Data validation | Pydantic v2 |
| Backend hosting | Railway |
| Frontend hosting | Vercel |
| Logging | Rich |

---

## ✦ Project Structure

```
research-agent/
├── agents/
│   ├── planner.py       # breaks query into sub-tasks
│   ├── searcher.py      # parallel web search
│   ├── critic.py        # quality control + retry logic
│   └── synthesizer.py   # final report generation
├── graph/
│   └── workflow.py      # LangGraph state machine + routing
├── tools/
│   └── search.py        # Tavily search wrapper
├── api/
│   └── main.py          # FastAPI REST API (job queue pattern)
├── models/
│   └── state.py         # Pydantic typed state
├── ui/
│   └── index.html       # Frontend UI
├── tests/
│   └── test_workflow.py # Unit + integration tests
├── main.py              # Entry point
└── requirements.txt
```

---

## ✦ Getting Started

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) — free, no credit card
- [Tavily API key](https://tavily.com) — free tier

### Installation

```bash
# Clone the repo
git clone https://github.com/Sourav93-subh/research-agent.git
cd research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GROQ_API_KEY and TAVILY_API_KEY to .env
```

### Run locally

```bash
# CLI — quickest way to test
python run.py "What is the current state of nuclear fusion energy?"

# API server
uvicorn api.main:app --reload --port 8000
# Interactive docs at http://localhost:8000/docs

# Open the UI
open ui/index.html
```

### Run tests

```bash
# Unit tests (no API keys needed)
pytest tests/ -v

# Integration tests (requires real keys)
pytest tests/ -v -m integration
```

---

## ✦ API Reference

### POST `/research`
Submit a research query. Returns immediately with a `job_id`.

```json
{
  "query": "What is the future of quantum computing?",
  "max_retries": 2
}
```

### GET `/research/{job_id}`
Poll for results. Status: `pending` → `running` → `done` / `failed`

```json
{
  "job_id": "uuid",
  "status": "done",
  "report": "# Quantum Computing...",
  "sources": ["https://..."],
  "subtask_count": 4,
  "duration_seconds": 12.4
}
```

---

## ✦ Resume Bullet

> Built production multi-agent research system using LangGraph and Groq — planner, parallel searcher, critic, and synthesizer agents with automatic retry logic; exposed via async FastAPI with job-queue pattern; deployed on Railway + Vercel

---

## ✦ License

MIT © [Sourav Subham](https://github.com/Sourav93-subh)