"""Advanced autonomous agent toolset and sandboxed execution environment.

Extends the core tool registry with:
1. execute_shell_command: Sandboxed command runner with strict allowlist and timeout.
2. fetch_arxiv_pdf_text: Deep full-text structure and page extraction from ArXiv PDFs.
3. sql_query_executor: Strict read-only SQL introspection over agi_memory relational database.
"""

from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from pypdf import PdfReader

from agent.tools import ToolDefinition, ToolRegistry, create_default_tool_registry
from config import get_settings
from storage.mysql_manager import MySQLAuditManager, get_mysql_manager

logger = logging.getLogger("agi_agent.tools_advanced")


# ===========================================================================
# 1. Sandboxed Shell Command Execution
# ===========================================================================

ALLOWED_COMMAND_BINARIES = {
    "git", "curl", "jq", "echo", "cat", "head", "tail", "grep",
    "wc", "ls", "dir", "python", "uname", "whoami", "sort", "uniq",
    "date", "find",
}

DISALLOWED_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\bsudo\b", r"\bsu\b", r"\bchmod\b",
    r"\bchown\b", r"\bdd\b", r"\bmkfs\b", r"\bformat\b", r"\bshutdown\b",
    r"\breboot\b", r"\bkill\b", r"\bkillall\b", r"\bpkill\b",
    r":\(\)\s*\{",  # Forkbomb
]


class ShellSecurityError(Exception):
    """Raised when shell command violates sandbox safety policies."""
    pass


def validate_shell_command(command: str) -> str:
    """Screen command against safety allowlist and destructive command patterns."""
    cmd_clean = command.strip()
    if not cmd_clean:
        raise ShellSecurityError("Command string cannot be empty.")

    # Check for explicitly banned commands/patterns
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, cmd_clean, re.IGNORECASE):
            raise ShellSecurityError(f"Security policy violation: Pattern '{pattern}' is prohibited.")

    try:
        tokens = shlex.split(cmd_clean, posix=False)
    except Exception:
        tokens = cmd_clean.split()

    if not tokens:
        raise ShellSecurityError("Command string cannot be empty.")

    # Parse command binaries: a binary appears at index 0 and after any shell operator (&&, ||, ;, |)
    expect_binary = True
    shell_operators = {"&&", "||", ";", "|", "&"}

    for token in tokens:
        cleaned_token = token.strip().strip('"').strip("'")
        if not cleaned_token:
            continue

        if cleaned_token in shell_operators:
            expect_binary = True
            continue

        if expect_binary:
            base_bin = Path(cleaned_token).name.lower()
            if base_bin.endswith(".exe"):
                base_bin = base_bin[:-4]

            if base_bin not in ALLOWED_COMMAND_BINARIES:
                raise ShellSecurityError(
                    f"Binary '{base_bin}' is not in the sandbox allowlist. Allowed: {sorted(list(ALLOWED_COMMAND_BINARIES))}"
                )
            expect_binary = False

    return cmd_clean


def execute_shell_command(command: str, timeout_seconds: int = 15) -> str:
    """Execute allowlisted shell command within resource and timeout boundaries.

    Args:
        command: The shell command to execute.
        timeout_seconds: Hard execution timeout (max 30 seconds).

    Returns:
        Captured standard output or detailed execution error.
    """
    timeout = min(max(1, int(timeout_seconds)), 30)

    try:
        validated_cmd = validate_shell_command(command)
    except ShellSecurityError as err:
        return f"SECURITY ERROR: {err}"

    # If python is referenced as executable, ensure it resolves to active sys.executable
    py_exec = f'"{sys.executable}"'
    run_cmd = re.sub(r"^(python3?(\.exe)?)\b", lambda m: py_exec, validated_cmd)
    run_cmd = re.sub(r"([|;&]\s*)(python3?(\.exe)?)\b", lambda m: f"{m.group(1)}{py_exec}", run_cmd)

    try:
        proc = subprocess.Popen(
            run_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
            else:
                proc.kill()
            try:
                proc.communicate(timeout=1)
            except Exception:
                pass
            return f"ERROR: Shell command timed out after {timeout} seconds."

        output = stdout.strip()
        errors = stderr.strip()

        if proc.returncode == 0:
            return output if output else "[Command completed with exit code 0 and no output]"
        return f"COMMAND FAILED (Exit Code {proc.returncode}):\nSTDOUT: {output}\nSTDERR: {errors}"

    except Exception as exc:
        return f"ERROR (Execution Failure): {exc}"


# ===========================================================================
# 2. ArXiv PDF Full-Text & Structure Extraction
# ===========================================================================

def normalize_arxiv_pdf_url(url: str) -> str:
    """Convert abstract or paper URLs into direct ArXiv PDF endpoints."""
    clean_url = url.strip()
    # Normalize https://arxiv.org/abs/2301.07041 -> https://arxiv.org/pdf/2301.07041.pdf
    match = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+(?:v[0-9]+)?)", clean_url)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    if clean_url.endswith(".pdf"):
        return clean_url
    if "arxiv.org/pdf/" in clean_url:
        return f"{clean_url}.pdf" if not clean_url.endswith(".pdf") else clean_url
    return clean_url


def fetch_arxiv_pdf_text(pdf_url: str, max_pages: int = 5) -> str:
    """Download an ArXiv PDF document and extract clean structured text.

    Args:
        pdf_url: Web URL of the ArXiv paper (supports /abs/ and /pdf/ URLs).
        max_pages: Maximum number of leading pages to extract (default 5).

    Returns:
        Formatted multi-page text extract.
    """
    target_url = normalize_arxiv_pdf_url(pdf_url)
    pages_to_extract = min(max(1, int(max_pages)), 20)

    logger.info("Downloading PDF from %s (extracting up to %d pages)...", target_url, pages_to_extract)

    try:
        with httpx.Client(timeout=25.0, follow_redirects=True) as client:
            resp = client.get(target_url, headers={"User-Agent": "AGIAgent-PDFParser/1.0"})
            resp.raise_for_status()
            pdf_bytes = resp.content

        reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        num_extracted = min(pages_to_extract, total_pages)

        extracted_text_blocks = []
        for page_idx in range(num_extracted):
            page = reader.pages[page_idx]
            page_text = page.extract_text() or ""
            # Strip excessive blank lines and trim
            clean_page = re.sub(r"\n{3,}", "\n\n", page_text).strip()
            extracted_text_blocks.append(f"--- [Page {page_idx + 1} of {total_pages}] ---\n{clean_page}")

        summary_header = f"=== Extracted {num_extracted}/{total_pages} Pages from {target_url} ===\n\n"
        return summary_header + "\n\n".join(extracted_text_blocks)

    except httpx.HTTPError as http_err:
        return f"ERROR (HTTP Request Failed): {http_err}"
    except Exception as exc:
        return f"ERROR (PDF Parsing Failed): {exc}"


# ===========================================================================
# 3. Read-Only SQL Query Executor for Relational Audit Store
# ===========================================================================

BANNED_SQL_KEYWORDS = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bREPLACE\b", r"\bCREATE\b",
    r"\bGRANT\b", r"\bREVOKE\b", r"\bLOCK\b", r"\bSET\b", r"\bEXEC\b",
]


class SQLSecurityError(Exception):
    """Raised when SQL query violates read-only safety policy."""
    pass


def validate_read_only_sql(query: str) -> str:
    """Validate query is strictly read-only and does not contain SQL injection tricks."""
    cleaned = query.strip()
    if not cleaned:
        raise SQLSecurityError("SQL query cannot be empty.")

    # Remove trailing semicolon
    cleaned_no_semi = cleaned.rstrip(";").strip()

    # Reject multi-statement query attempts (e.g. "SELECT 1; DROP TABLE ...")
    if ";" in cleaned_no_semi:
        raise SQLSecurityError("Multi-statement SQL queries are strictly prohibited.")

    # Must start with SELECT, SHOW, DESCRIBE, or EXPLAIN
    if not re.match(r"^(SELECT|SHOW|DESCRIBE|EXPLAIN)\b", cleaned_no_semi, re.IGNORECASE):
        raise SQLSecurityError("Only read-only statements (SELECT, SHOW, DESCRIBE, EXPLAIN) are permitted.")

    # Check for any banned modification keywords
    for pattern in BANNED_SQL_KEYWORDS:
        if re.search(pattern, cleaned_no_semi, re.IGNORECASE):
            raise SQLSecurityError(f"Modification keyword pattern '{pattern}' is strictly prohibited.")

    return cleaned_no_semi


def sql_query_executor(query: str, max_rows: int = 50) -> str:
    """Execute a read-only SQL query against the agi_memory database.

    Args:
        query: Read-only SQL query (SELECT, SHOW, DESCRIBE, EXPLAIN).
        max_rows: Maximum records to return (default 50).

    Returns:
        Markdown table or JSON structure of query results.
    """
    row_limit = min(max(1, int(max_rows)), 200)

    try:
        safe_query = validate_read_only_sql(query)
    except SQLSecurityError as sec_err:
        return f"SECURITY ERROR: {sec_err}"

    # Ensure LIMIT clause is applied if query is a SELECT
    if safe_query.upper().startswith("SELECT") and "LIMIT" not in safe_query.upper():
        safe_query = f"{safe_query} LIMIT {row_limit}"

    mysql_mgr = get_mysql_manager()
    conn = mysql_mgr.get_connection()
    if not conn:
        return f"DATABASE OFFLINE: Relational audit database '{mysql_mgr.database}' is currently unreachable."

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(safe_query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return f"[Query executed successfully. 0 rows returned for: '{safe_query}']"

        # Format into clean Markdown Table
        headers = list(rows[0].keys())
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = []
        for r in rows:
            values = [str(r.get(h, ""))[:100].replace("\n", " ") for h in headers]
            data_rows.append("| " + " | ".join(values) + " |")

        table_str = "\n".join([header_row, sep_row] + data_rows)
        return f"=== SQL Query Results ({len(rows)} rows) ===\n\n{table_str}"

    except Exception as exc:
        if conn and conn.is_connected():
            conn.close()
        return f"SQL EXECUTION ERROR: {exc}"


# ===========================================================================
# 4. Extended Tool Registry Factory
# ===========================================================================

def create_extended_tool_registry(include_core: bool = True) -> ToolRegistry:
    """Create a ToolRegistry containing both core and advanced autonomous tools."""
    registry = create_default_tool_registry() if include_core else ToolRegistry()

    # 1. execute_shell_command
    registry.register(
        ToolDefinition(
            name="execute_shell_command",
            description=(
                "Execute sandboxed shell commands for system inspection, dependency exploration, "
                "or directory parsing. Strict allowlist enforced (git, curl, jq, echo, cat, grep, wc, ls, python)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 15, max 30).",
                        "default": 15,
                    },
                },
                "required": ["command"],
            },
            func=execute_shell_command,
        )
    )

    # 2. fetch_arxiv_pdf_text
    registry.register(
        ToolDefinition(
            name="fetch_arxiv_pdf_text",
            description=(
                "Extract structured full-text directly from an ArXiv paper PDF URL. "
                "Useful for analyzing equations, methodology, and empirical tables beyond abstract summaries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pdf_url": {
                        "type": "string",
                        "description": "Direct PDF or abstract URL from arxiv.org.",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum number of initial pages to extract (default 5).",
                        "default": 5,
                    },
                },
                "required": ["pdf_url"],
            },
            func=fetch_arxiv_pdf_text,
        )
    )

    # 3. sql_query_executor
    registry.register(
        ToolDefinition(
            name="sql_query_executor",
            description=(
                "Query the agi_memory MySQL relational database in read-only mode. "
                "Allows introspecting cataloged papers, past agent sessions, and training runs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Read-only SQL query (SELECT, SHOW, DESCRIBE, EXPLAIN).",
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum rows to return (default 50).",
                        "default": 50,
                    },
                },
                "required": ["query"],
            },
            func=sql_query_executor,
        )
    )

    return registry
