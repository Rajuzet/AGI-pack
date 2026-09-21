"""Unit and integration tests for ReAct orchestrator and self-correction loop."""

import json
from unittest.mock import MagicMock
import pytest

from agent.orchestrator import AgentFinalAnswer, ReActOrchestrator, ReActStep
from agent.tools import ToolDefinition, ToolRegistry


def test_react_step_to_dict():
    """Verify ReActStep data model serialization."""
    step = ReActStep(
        step_number=1,
        thought="Need to compute sum",
        action="execute_python_code",
        action_input={"code": "print(1 + 1)"},
        observation="2",
        is_error=False,
    )
    d = step.to_dict()
    assert d["step_number"] == 1
    assert d["action"] == "execute_python_code"
    assert d["observation"] == "2"
    assert d["is_error"] is False


def test_agent_final_answer_markdown_and_json():
    """Verify AgentFinalAnswer parses to valid JSON and Markdown report."""
    steps = [
        ReActStep(
            step_number=1,
            thought="Step 1 thought",
            action="execute_python_code",
            action_input={"code": "print(42)"},
            observation="42",
        ),
        ReActStep(
            step_number=2,
            thought="Step 2 finished",
        ),
    ]
    answer = AgentFinalAnswer(
        query="What is 42?",
        final_answer="42 is the ultimate answer.",
        steps=steps,
        total_steps=2,
        retries_used=0,
        sources=["https://arxiv.org/abs/2401.00001"],
        success=True,
        execution_time_seconds=0.15,
    )

    # Test JSON serialization
    json_str = answer.to_json()
    parsed = json.loads(json_str)
    assert parsed["query"] == "What is 42?"
    assert parsed["total_steps"] == 2
    assert len(parsed["steps"]) == 2
    assert parsed["sources"] == ["https://arxiv.org/abs/2401.00001"]

    # Test Markdown formatting
    md = answer.to_markdown()
    assert "# Agent Reasoning Report: What is 42?" in md
    assert "42 is the ultimate answer." in md
    assert "https://arxiv.org/abs/2401.00001" in md
    assert "Step 1" in md


def test_orchestrator_successful_execution():
    """Verify orchestrator runs complete thought -> action -> observation -> final answer cycle."""
    mock_responses = [
        # Step 1 response: choose action
        "Thought: I will compute the value.\nAction: execute_python_code\nAction Input: {\"code\": \"print(10 * 10)\"}",
        # Step 2 response: conclude
        "Thought: I got the observation.\nFinal Answer: 10 * 10 is equal to 100.",
    ]
    call_idx = 0

    def mock_llm(messages):
        nonlocal call_idx
        resp = mock_responses[call_idx]
        call_idx += 1
        return resp

    orchestrator = ReActOrchestrator(llm_client=mock_llm)
    result = orchestrator.run("What is 10 * 10?")

    assert result.success is True
    assert result.total_steps == 2
    assert "10 * 10 is equal to 100." in result.final_answer
    assert result.steps[0].action == "execute_python_code"
    assert result.steps[0].observation == "100"
    assert result.retries_used == 0


def test_orchestrator_error_self_correction():
    """Verify orchestrator detects tool execution error, feeds trace back, and self-corrects."""
    mock_responses = [
        # Step 1: Intentionally trigger a syntax error
        "Thought: I will test code with a syntax error.\nAction: execute_python_code\nAction Input: {\"code\": \"print(1 +\"}",
        # Step 2: Agent observes error in context, reflects, and executes corrected code!
        "Thought: The previous code failed with a SyntaxError. I will fix the closing parenthesis.\nAction: execute_python_code\nAction Input: {\"code\": \"print(1 + 1)\"}",
        # Step 3: Conclude successfully
        "Thought: The calculation succeeded.\nFinal Answer: The corrected expression evaluates to 2.",
    ]
    call_idx = 0

    def mock_llm(messages):
        nonlocal call_idx
        resp = mock_responses[call_idx]
        call_idx += 1
        return resp

    orchestrator = ReActOrchestrator(llm_client=mock_llm)
    result = orchestrator.run("Compute expression with retry test")

    assert result.success is True
    assert result.total_steps == 3
    assert result.retries_used == 1  # 1 self-correction retry occurred!
    assert result.steps[0].is_error is True
    assert result.steps[0].observation is not None and "ERROR (Validation)" in result.steps[0].observation
    assert result.steps[0].retry_count == 1
    assert "Analyzing error trace" in (result.steps[0].reflection or "")

    # Step 2 was successful correction
    assert result.steps[1].is_error is False
    assert result.steps[1].observation == "2"
    assert "evaluates to 2" in result.final_answer


def test_orchestrator_max_retries_limit():
    """Verify orchestrator caps retries at max_error_retries (3) and prevents infinite error loops."""
    # LLM keeps failing with bad code
    broken_resp = "Thought: Trying code.\nAction: execute_python_code\nAction Input: {\"code\": \"1 / 0\"}"
    final_resp = "Thought: Too many errors.\nFinal Answer: Could not compute due to repeated division by zero."

    call_count = 0

    def stubbornly_failing_llm(messages):
        nonlocal call_count
        call_count += 1
        # Check if system notification informs about max retries reached
        if any("Maximum retry limit reached" in m["content"] for m in messages):
            return final_resp
        return broken_resp

    orchestrator = ReActOrchestrator(llm_client=stubbornly_failing_llm, max_error_retries=3)
    result = orchestrator.run("Stubborn failure test")

    # The agent should record the retries and conclude
    assert result.retries_used >= 3
    assert any(s.retry_count >= 3 for s in result.steps if s.is_error)
