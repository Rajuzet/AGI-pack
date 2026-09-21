"""Unit and integration tests for agent tools and sandboxed execution."""

import json
from unittest.mock import MagicMock, patch
import pytest

from agent.tools import (
    CodeSecurityError,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    execute_python_code,
    fetch_live_web,
    query_vector_memory,
    validate_python_code,
)


def test_python_code_validation_valid():
    """Verify clean Python code passes AST screening."""
    valid_code = """
def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)
result = factorial(5)
print(result)
"""
    # Should not raise
    validate_python_code(valid_code)


def test_python_code_validation_syntax_error():
    """Verify syntax error is intercepted before execution."""
    broken_code = "def broken(x) print(x)"
    with pytest.raises(SyntaxError) as exc_info:
        validate_python_code(broken_code)
    assert "SyntaxError" in str(exc_info.value)


def test_python_code_validation_disallowed_calls():
    """Verify forbidden destructive system calls are caught by AST scanner."""
    malicious_code = """
import os
os.system("echo compromised")
"""
    with pytest.raises(CodeSecurityError) as exc_info:
        validate_python_code(malicious_code)
    assert "restricted function 'os.system'" in str(exc_info.value)

    rmtree_code = """
import shutil
shutil.rmtree("/important/dir")
"""
    with pytest.raises(CodeSecurityError) as exc_info:
        validate_python_code(rmtree_code)
    assert "restricted function 'shutil.rmtree'" in str(exc_info.value)


def test_execute_python_code_success():
    """Verify successful execution returns stdout."""
    code = "print('Hello Autonomous Agent')"
    output = execute_python_code(code)
    assert output.strip() == "Hello Autonomous Agent"


def test_execute_python_code_runtime_error():
    """Verify runtime exception generates structured diagnostic with stderr."""
    code = """
x = 10
y = 0
print(x / y)
"""
    output = execute_python_code(code)
    assert "ERROR (Execution failed" in output
    assert "ZeroDivisionError" in output


def test_execute_python_code_syntax_error():
    """Verify static syntax error returns formatted error message."""
    code = "print(1 +"
    output = execute_python_code(code)
    assert "ERROR (Validation)" in output
    assert "SyntaxError" in output


def test_execute_python_code_timeout():
    """Verify execution exceeding timeout limit is halted."""
    code = """
import time
time.sleep(5)
"""
    output = execute_python_code(code, timeout_seconds=1)
    assert "ERROR (Timeout)" in output
    assert "exceeded 1 seconds limit" in output


def test_query_vector_memory_empty_query():
    """Verify empty query returns error message."""
    result = query_vector_memory("   ")
    assert "ERROR: Query string cannot be empty" in result


def test_query_vector_memory_mocked_results():
    """Verify vector memory retrieval formats top matching documents with scores."""
    mock_results = [
        {
            "id": "doc-001",
            "score": 0.9421,
            "title": "Scaling Law Limits in Cyclic Reasoning Agents",
            "source": "arxiv",
            "text": "Empirical analysis reveals substantial accuracy improvements using ReAct loops.",
            "metadata": {
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "authors": ["A. Vaswani", "D. Silver"],
            },
        }
    ]

    mock_store = MagicMock()
    mock_store.index.ntotal = 10
    mock_store.similarity_search.return_value = mock_results

    with patch("agent.tools.get_agent_vector_store", return_value=mock_store):
        output = query_vector_memory("cyclic reasoning")

        assert "Found 1 relevant passages" in output
        assert "Scaling Law Limits in Cyclic Reasoning Agents" in output
        assert "Score: 0.9421" in output
        assert "https://arxiv.org/pdf/2401.00001" in output


def test_fetch_live_web_empty_query():
    """Verify empty web search query is rejected."""
    result = fetch_live_web("")
    assert "ERROR: Search query cannot be empty" in result


def test_fetch_live_web_mocked():
    """Verify live web retrieval handles HTML response."""
    mock_html = """
    <html>
      <body>
        <a class="result__snippet">Breakthrough in multi-step agent reasoning with AST guards.</a>
        <a class="result__url" href="https://example.com/research">example.com</a>
      </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_html

    with patch("httpx.Client.get", return_value=mock_resp):
        output = fetch_live_web("agent reasoning")
        assert "Web search results for 'agent reasoning':" in output
        assert "Breakthrough in multi-step agent reasoning" in output
        assert "https://example.com/research" in output


def test_tool_registry_schemas():
    """Verify tool definitions produce valid OpenAI and Anthropic schemas."""
    registry = create_default_tool_registry()

    openai_schemas = registry.get_openai_schemas()
    assert len(openai_schemas) == 3

    tool_names = [s["function"]["name"] for s in openai_schemas]
    assert "query_vector_memory" in tool_names
    assert "execute_python_code" in tool_names
    assert "fetch_live_web" in tool_names

    for schema in openai_schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]
        assert "properties" in schema["function"]["parameters"]

    anthropic_schemas = registry.get_anthropic_schemas()
    assert len(anthropic_schemas) == 3
    for schema in anthropic_schemas:
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema


def test_tool_registry_execution_dispatch():
    """Verify ToolRegistry dispatches execution and handles unknown tools."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo_test",
            description="Echoes input",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            func=lambda msg: f"Echo: {msg}",
        )
    )

    result = registry.execute("echo_test", msg="Hello World")
    assert result == "Echo: Hello World"

    missing_result = registry.execute("non_existent_tool")
    assert "ERROR: Tool 'non_existent_tool' does not exist" in missing_result
