import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from models.state import ResearchState, SubTask, SearchResult, CriticFeedback


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    return ResearchState(user_query="What is quantum computing?")


@pytest.fixture
def state_with_subtasks():
    return ResearchState(
        user_query="What is quantum computing?",
        subtasks=[
            SubTask(id="1", query="quantum computing basics"),
            SubTask(id="2", query="quantum computing applications"),
            SubTask(id="3", query="quantum computing companies 2024"),
        ],
    )


@pytest.fixture
def state_with_results():
    return ResearchState(
        user_query="What is quantum computing?",
        subtasks=[SubTask(id="1", query="quantum computing basics")],
        search_results=[
            SearchResult(
                subtask_id="1",
                query="quantum computing basics",
                content="Quantum computing uses qubits instead of classical bits...",
                sources=["https://example.com/quantum"],
            )
        ],
    )


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestResearchState:
    def test_default_state(self, base_state):
        assert base_state.user_query == "What is quantum computing?"
        assert base_state.subtasks == []
        assert base_state.search_results == []
        assert base_state.retry_count == 0
        assert base_state.max_retries == 2

    def test_state_copy_preserves_fields(self, base_state):
        updated = base_state.model_copy(update={"retry_count": 1})
        assert updated.retry_count == 1
        assert updated.user_query == base_state.user_query

    def test_critic_feedback_model(self):
        fb = CriticFeedback(
            passed=False,
            gaps=["missing recent data"],
            retry_queries=["quantum computing 2024 news"],
        )
        assert not fb.passed
        assert len(fb.gaps) == 1


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_planner_creates_subtasks(self, base_state):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"subtasks": [{"id": "1", "query": "quantum basics"}, {"id": "2", "query": "quantum applications"}]}')]

        with patch("agents.planner.client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            from agents.planner import planner_agent
            result = await planner_agent(base_state)

        assert len(result.subtasks) == 2
        assert result.subtasks[0].query == "quantum basics"

    @pytest.mark.asyncio
    async def test_planner_handles_json_error(self, base_state):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="this is not json")]

        with patch("agents.planner.client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            from agents.planner import planner_agent
            result = await planner_agent(base_state)

        # Falls back to original query as single subtask
        assert len(result.subtasks) == 1
        assert result.subtasks[0].query == base_state.user_query


class TestSearcherAgent:
    @pytest.mark.asyncio
    async def test_searcher_runs_parallel_searches(self, state_with_subtasks):
        mock_search_result = {"content": "Some content", "sources": ["https://example.com"]}

        with patch("agents.searcher.SearchTool") as MockTool:
            instance = MockTool.return_value
            instance.search = AsyncMock(return_value=mock_search_result)
            from agents.searcher import searcher_agent
            result = await searcher_agent(state_with_subtasks)

        assert len(result.search_results) == 3

    @pytest.mark.asyncio
    async def test_searcher_merges_on_retry(self, state_with_results):
        new_subtask = SubTask(id="99", query="retry query")
        state = state_with_results.model_copy(update={"subtasks": [new_subtask]})
        mock_result = {"content": "New content", "sources": ["https://new.com"]}

        with patch("agents.searcher.SearchTool") as MockTool:
            instance = MockTool.return_value
            instance.search = AsyncMock(return_value=mock_result)
            from agents.searcher import searcher_agent
            result = await searcher_agent(state)

        # Old result still present + new one added
        assert len(result.search_results) == 2


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_critic_passes_good_research(self, state_with_results):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"passed": true, "gaps": [], "retry_queries": [], "notes": "Comprehensive"}')]

        with patch("agents.critic.client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            from agents.critic import critic_agent
            result = await critic_agent(state_with_results)

        assert result.critic_feedback is not None
        assert result.critic_feedback.passed is True

    @pytest.mark.asyncio
    async def test_critic_creates_retry_subtasks(self, state_with_results):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"passed": false, "gaps": ["missing history"], "retry_queries": ["quantum computing history"], "notes": "Needs more depth"}')]

        with patch("agents.critic.client") as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            from agents.critic import critic_agent
            result = await critic_agent(state_with_results)

        assert result.critic_feedback.passed is False
        assert len(result.subtasks) == 1
        assert result.subtasks[0].query == "quantum computing history"


class TestWorkflowRouting:
    def test_routes_to_synthesize_when_passed(self, state_with_results):
        from graph.workflow import should_retry
        state = state_with_results.model_copy(update={
            "critic_feedback": CriticFeedback(passed=True)
        })
        assert should_retry(state) == "synthesize"

    def test_routes_to_retry_when_failed_and_retries_remain(self, state_with_results):
        from graph.workflow import should_retry
        state = state_with_results.model_copy(update={
            "critic_feedback": CriticFeedback(passed=False, retry_queries=["more info"]),
            "retry_count": 1,
            "max_retries": 2,
        })
        assert should_retry(state) == "retry"

    def test_routes_to_synthesize_when_out_of_retries(self, state_with_results):
        from graph.workflow import should_retry
        state = state_with_results.model_copy(update={
            "critic_feedback": CriticFeedback(passed=False),
            "retry_count": 3,
            "max_retries": 2,
        })
        assert should_retry(state) == "synthesize"


# ── Integration test (skipped in CI without real API keys) ───────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """Runs the real pipeline — requires ANTHROPIC_API_KEY and TAVILY_API_KEY."""
    from graph.workflow import run_research
    state = await run_research("What is LangGraph and how does it work?")

    assert state.final_report
    assert len(state.final_report) > 200
    assert len(state.subtasks) >= 2
    assert len(state.search_results) >= 2