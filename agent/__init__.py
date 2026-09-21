"""Autonomous Reasoning Agent Package.

Exposes ReActOrchestrator, ToolRegistry, standard schemas, and tool execution routines.
"""

from agent.orchestrator import (
    AgentFinalAnswer,
    ReActOrchestrator,
    ReActStep,
    create_fastapi_serving_client,
)
from agent.tools import (
    CodeSecurityError,
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    execute_python_code,
    fetch_live_web,
    query_vector_memory,
)

from agent.tools_advanced import (
    ShellSecurityError,
    SQLSecurityError,
    create_extended_tool_registry,
    execute_shell_command,
    fetch_arxiv_pdf_text,
    sql_query_executor,
)

from agent.multi_agent_consensus import (
    ConsensusResult,
    CriticAgent,
    DebateRound,
    ExecutionPlan,
    ExecutorAgent,
    MultiAgentConsensusOrchestrator,
    PlanStep,
    PlannerAgent,
)

__all__ = [
    "AgentFinalAnswer",
    "CodeSecurityError",
    "ConsensusResult",
    "CriticAgent",
    "DebateRound",
    "ExecutionPlan",
    "ExecutorAgent",
    "MultiAgentConsensusOrchestrator",
    "PlanStep",
    "PlannerAgent",
    "ReActOrchestrator",
    "ReActStep",
    "ShellSecurityError",
    "SQLSecurityError",
    "ToolDefinition",
    "ToolRegistry",
    "create_default_tool_registry",
    "create_extended_tool_registry",
    "execute_python_code",
    "execute_shell_command",
    "fetch_arxiv_pdf_text",
    "fetch_live_web",
    "query_vector_memory",
    "sql_query_executor",
]
