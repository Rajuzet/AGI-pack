"""Autonomous agent tool definitions and sandboxed execution engine.

Implements strict function-calling schemas (Anthropic/OpenAI compatible),
sandboxed Python execution with AST screening and timeouts, vector memory lookup,
and live web retrieval.
"""

import ast
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from config import get_settings
from ingestion.stream_sources import TextNormalizer
from memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restricted Python Sandbox Execution
# ---------------------------------------------------------------------------

class CodeSecurityError(Exception):
    """Raised when code violates sandbox safety policies."""
    pass


class ASTSecurityScanner(ast.NodeVisitor):
    """AST visitor that detects dangerous calls and imports before execution."""

    DISALLOWED_MODULES = {
        "ctypes", "winreg", "_winreg", "msvcrt", "pty", "posix",
    }
    DISALLOWED_CALLS = {
        "shutil.rmtree", "os.system", "os.popen", "os.unlink", "os.remove",
        "subprocess.call", "subprocess.Popen", "subprocess.run",
    }

    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in self.DISALLOWED_MODULES:
                self.violations.append(f"Import of restricted module '{alias.name}' is prohibited.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in self.DISALLOWED_MODULES:
                self.violations.append(f"Import from restricted module '{node.module}' is prohibited.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = f"{node.func.value.id}.{node.func.attr}"

        if func_name in self.DISALLOWED_CALLS:
            self.violations.append(f"Calling restricted function '{func_name}' is prohibited in sandbox.")

        self.generic_visit(node)


def validate_python_code(code_str: str) -> None:
    """Validate that Python code does not contain syntax errors or banned operations."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise SyntaxError(f"Python SyntaxError: {exc.msg} at line {exc.lineno}") from exc

    scanner = ASTSecurityScanner()
    scanner.visit(tree)
    if scanner.violations:
        raise CodeSecurityError("; ".join(scanner.violations))


def execute_python_code(code: str, timeout_seconds: int = 15) -> str:
    """Execute Python code in an isolated subprocess with timeout and safety screening.

    Args:
        code: Python source code string to execute.
        timeout_seconds: Hard execution limit in seconds.

    Returns:
        String representing stdout, stderr, or error diagnostic.
    """
    clean_code = code.strip()
    if not clean_code:
        return "ERROR: Empty code provided."

    # 1. Static AST validation
    try:
        validate_python_code(clean_code)
    except (SyntaxError, CodeSecurityError) as exc:
        logger.warning("Code validation failed: %s", exc)
        return f"ERROR (Validation): {exc}"

    # 2. Write to temporary script
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(clean_code)
        temp_path = temp_file.name

    try:
        # Run isolated subprocess with bounded execution window
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode != 0:
            return (
                f"ERROR (Execution failed with exit code {process.returncode}):\n"
                f"Stderr:\n{stderr}\n"
                f"Stdout:\n{stdout}"
            ).strip()

        if stdout:
            return stdout
        elif stderr:
            return f"Process completed with warnings (stderr):\n{stderr}"
        else:
            return "SUCCESS: Code executed successfully with no output to stdout."

    except subprocess.TimeoutExpired:
        return f"ERROR (Timeout): Code execution exceeded {timeout_seconds} seconds limit."
    except Exception as exc:
        return f"ERROR (System): Failed to run sandbox process: {exc}"
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Vector Memory Retrieval Tool
# ---------------------------------------------------------------------------

_GLOBAL_VECTOR_STORE: Optional[VectorStore] = None


def get_agent_vector_store() -> VectorStore:
    """Obtain or initialize the global VectorStore for agent memory retrieval."""
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        settings = get_settings()
        _GLOBAL_VECTOR_STORE = VectorStore(settings=settings)
        # Attempt to load local index
        _GLOBAL_VECTOR_STORE.load_local()
    return _GLOBAL_VECTOR_STORE


def query_vector_memory(query: str, top_k: int = 5) -> str:
    """Semantic vector search across persistent research papers and technical news.

    Args:
        query: Natural language query string.
        top_k: Number of relevant matching passages to retrieve.

    Returns:
        Formatted string of matching documents with cosine similarity scores.
    """
    clean_query = query.strip()
    if not clean_query:
        return "ERROR: Query string cannot be empty."

    store = get_agent_vector_store()
    if store.index.ntotal == 0:
        return f"OBSERVATION: Vector memory index is currently empty. No documents found for query: '{query}'."

    results = store.similarity_search(clean_query, top_k=top_k)
    if not results:
        return f"OBSERVATION: No matching documents found for query: '{query}'."

    lines = [f"Found {len(results)} relevant passages from vector memory for query '{clean_query}':"]
    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        title = res.get("title") or "Untitled Document"
        source = res.get("source") or "unknown"
        text = res.get("text", "")
        meta = res.get("metadata", {})

        url = meta.get("pdf_url") or meta.get("link") or meta.get("abs_url") or "N/A"
        authors = ", ".join(meta.get("authors", [])) or "N/A"

        lines.append(
            f"\n[Result {i}] (Score: {score:.4f})\n"
            f"Title: {title}\n"
            f"Source: {source} | Authors: {authors}\n"
            f"Link: {url}\n"
            f"Passage:\n{text}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live Web Retrieval Tool
# ---------------------------------------------------------------------------

def fetch_live_web(query: str, max_results: int = 5) -> str:
    """Perform targeted live web retrieval and extract clean content snippets.

    Args:
        query: Search keywords or research topic.
        max_results: Maximum results to return.

    Returns:
        Formatted search results with source titles, URLs, and text excerpts.
    """
    clean_query = query.strip()
    if not clean_query:
        return "ERROR: Search query cannot be empty."

    settings = get_settings()
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    # DuckDuckGo HTML search endpoint
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(clean_query)}"

    try:
        with httpx.Client(headers=headers, timeout=10.0, follow_redirects=True) as client:
            response = client.get(search_url)

        if response.status_code != 200:
            # Fallback to ArXiv direct query if search engine is throttled
            arxiv_url = f"{settings.arxiv_api_url}?search_query=all:{quote_plus(clean_query)}&max_results={max_results}"
            with httpx.Client(headers=headers, timeout=10.0) as client:
                arxiv_resp = client.get(arxiv_url)
            if arxiv_resp.status_code == 200:
                import feedparser
                parsed = feedparser.parse(arxiv_resp.text)
                if parsed.entries:
                    results = [f"Web/ArXiv Results for '{clean_query}':"]
                    for idx, e in enumerate(parsed.entries[:max_results], 1):
                        results.append(
                            f"\n[{idx}] {e.get('title', '')}\nURL: {e.get('id', '')}\nSnippet: {TextNormalizer.normalize(e.get('summary', ''), max_tokens=150)}"
                        )
                    return "\n".join(results)

            return f"OBSERVATION: Web search returned status {response.status_code}. No direct results found."

        # Parse results using regex / TextNormalizer
        raw_html = response.text
        # Extract result snippets from DuckDuckGo HTML
        import re
        result_blocks = re.findall(
            r'<a class="result__snippet[^>]*>(.*?)</a>',
            raw_html,
            re.DOTALL | re.IGNORECASE,
        )
        title_blocks = re.findall(
            r'<a class="result__url[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            raw_html,
            re.DOTALL | re.IGNORECASE,
        )

        if not result_blocks:
            clean_page = TextNormalizer.normalize(raw_html, max_tokens=300)
            return f"OBSERVATION: Web search conducted for '{clean_query}'. Summary:\n{clean_page}"

        items = []
        limit = min(len(result_blocks), max_results)
        for i in range(limit):
            snippet = TextNormalizer.strip_html(result_blocks[i]).strip()
            url = title_blocks[i][0] if i < len(title_blocks) else "N/A"
            items.append(f"\n[{i+1}] Snippet: {snippet}\nURL: {url}")

        return f"Web search results for '{clean_query}':\n" + "\n".join(items)

    except Exception as exc:
        logger.warning("Live web search encountered an issue: %s", exc)
        return f"ERROR (Web Search): Failed to retrieve web results for '{clean_query}': {exc}"


# ---------------------------------------------------------------------------
# Standardized Tool Registry & Schemas
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Structured representation of a callable agent tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling standard."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use standard."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    """Registry maintaining tools, schemas, and execution dispatching."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool definition by name."""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> str:
        """Dispatch tool execution by name."""
        tool = self._tools.get(name)
        if not tool:
            return f"ERROR: Tool '{name}' does not exist in registry. Available: {list(self._tools.keys())}"
        try:
            return str(tool.func(**kwargs))
        except Exception as exc:
            return f"ERROR (Tool Execution Failure in '{name}'): {exc}"

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Return list of all registered tools in OpenAI format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_anthropic_schemas(self) -> List[Dict[str, Any]]:
        """Return list of all registered tools in Anthropic format."""
        return [t.to_anthropic_schema() for t in self._tools.values()]


def create_default_tool_registry() -> ToolRegistry:
    """Instantiate standard ToolRegistry populated with core agent tools."""
    registry = ToolRegistry()

    # 1. query_vector_memory
    registry.register(
        ToolDefinition(
            name="query_vector_memory",
            description=(
                "Search the persistent long-term vector memory for peer-reviewed papers, "
                "scientific articles, and technical reports using dense semantic retrieval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or research topic.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top matching documents to retrieve.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            func=query_vector_memory,
        )
    )

    # 2. execute_python_code
    registry.register(
        ToolDefinition(
            name="execute_python_code",
            description=(
                "Execute Python code inside an isolated sandbox environment. "
                "Use this tool for data analysis, mathematical verification, string manipulation, "
                "and scientific computations. Returns standard output (stdout) or error tracebacks (stderr)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python 3 source code to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Execution timeout limit in seconds.",
                        "default": 15,
                    },
                },
                "required": ["code"],
            },
            func=execute_python_code,
        )
    )

    # 3. fetch_live_web
    registry.register(
        ToolDefinition(
            name="fetch_live_web",
            description=(
                "Perform live web searches to find real-time updates, news, documentation, "
                "or recent developments not present in local vector memory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of snippets to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            func=fetch_live_web,
        )
    )

    return registry
