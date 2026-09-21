"""Autonomous ReAct reasoning engine with cyclic error reflection and self-correction.

Implements the standard ReAct loop:
Thought -> Action (Tool Call) -> Observation (Tool Output) -> Reflection/Self-Correction -> Final Answer

Features:
- Strict tool execution via ToolRegistry.
- Automatic error detection (SyntaxError, runtime exceptions, empty search, timeouts).
- Contextual feedback loop: feeds error traces back into the reasoning context for self-correction (up to 3 retries).
- Dual output parsers: Structured JSON (`AgentFinalAnswer`) and clean Markdown synthesis.
- Pluggable LLM backends: OpenAI/Anthropic compatible APIs, custom callables, or offline heuristic reasoner for testing.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agent.tools import ToolRegistry, create_default_tool_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models & ReAct Traces
# ---------------------------------------------------------------------------

@dataclass
class ReActStep:
    """Individual step within the ReAct reasoning trajectory."""

    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_error: bool = False
    reflection: Optional[str] = None
    retry_count: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return asdict(self)


@dataclass
class AgentFinalAnswer:
    """Structured response synthesized by the autonomous reasoning agent."""

    query: str
    final_answer: str
    steps: List[ReActStep] = field(default_factory=list)
    total_steps: int = 0
    retries_used: int = 0
    sources: List[str] = field(default_factory=list)
    success: bool = True
    execution_time_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire answer and execution trace to a dictionary."""
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "retries_used": self.retries_used,
            "sources": self.sources,
            "success": self.success,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "timestamp": self.timestamp,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize structured response as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Synthesize a clean, professional Markdown report."""
        lines = [
            f"# Agent Reasoning Report: {self.query}",
            "",
            f"**Status:** {'Success' if self.success else 'Completed with Warnings'}"
            f" | **Steps Taken:** {self.total_steps}"
            f" | **Self-Correction Retries:** {self.retries_used}"
            f" | **Runtime:** {self.execution_time_seconds:.2f}s",
            "",
            "## Executive Summary",
            "",
            self.final_answer.strip(),
            "",
        ]

        if self.sources:
            lines.extend([
                "## Sources & References",
                "",
            ])
            for src in self.sources:
                lines.append(f"- {src}")
            lines.append("")

        lines.extend([
            "## Reasoning & Action Trajectory",
            "",
        ])
        for step in self.steps:
            lines.append(f"### Step {step.step_number}")
            lines.append(f"**Thought:** {step.thought.strip()}")
            if step.action:
                lines.append(f"**Action:** `{step.action}`")
                lines.append(f"**Action Input:** ```json\n{json.dumps(step.action_input or {}, indent=2)}\n```")
            if step.observation:
                obs_preview = step.observation.strip()
                lines.append(f"**Observation:**\n```text\n{obs_preview}\n```")
            if step.reflection:
                lines.append(f"> [!WARNING]\n> **Self-Correction Reflection (Retry {step.retry_count}):** {step.reflection.strip()}\n")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts & Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Autonomous AI Research and Reasoning Agent.
Your objective is to solve complex analytical, scientific, and technical inquiries by methodically reasoning and utilizing external tools.

You have access to the following tools:
{tool_descriptions}

Follow the ReAct (Reasoning and Acting) protocol strictly:
1. Formulate a 'Thought' describing your step-by-step reasoning and strategy.
2. If an action is required, specify the 'Action' (exact tool name) and 'Action Input' (valid JSON object matching tool parameters).
3. Wait for the 'Observation' (tool output).
4. If an Observation contains an ERROR or unexpected output, engage in 'Reflection' to diagnose the failure, correct parameters or code, and retry the action.
5. When you have sufficient information or have completed the required computation, conclude with 'Final Answer'.

Output format format for tool use:
Thought: <your rationale for taking this action>
Action: <one of the available tool names>
Action Input: {{"param1": "value1"}}

Output format when finished:
Thought: I have gathered sufficient evidence and computed the necessary results.
Final Answer: <comprehensive, well-structured synthesized response in clean Markdown>
"""


# ---------------------------------------------------------------------------
# FastAPIServingClient Helper
# ---------------------------------------------------------------------------

def create_fastapi_serving_client(
    endpoint_url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "default",
    timeout: float = 60.0,
    fallback_client: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Callable[[List[Dict[str, str]]], str]:
    """Create a client callable that queries the local FastAPI model serving microservice.

    Args:
        endpoint_url: URL to the /v1/chat/completions endpoint.
        model: Model identifier parameter for the request.
        timeout: HTTP request timeout in seconds.
        fallback_client: Optional fallback callable if the serving endpoint is unreachable.

    Returns:
        Callable taking message history and returning assistant text response.
    """
    import httpx

    def _call_serving_endpoint(messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(endpoint_url, json=payload)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
                raise ValueError(f"Unexpected response format from serving endpoint: {data}")
        except Exception as exc:
            logger.warning("Error querying serving endpoint '%s': %s", endpoint_url, exc)
            if fallback_client:
                logger.info("Falling back to built-in heuristic reasoner.")
                return fallback_client(messages)
            raise

    return _call_serving_endpoint


# ---------------------------------------------------------------------------
# ReAct Orchestrator Engine
# ---------------------------------------------------------------------------

class ReActOrchestrator:
    """Cyclic ReAct loop orchestrator with self-healing error reflection."""

    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        llm_client: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        max_steps: int = 8,
        max_error_retries: int = 3,
        audit_manager: Optional[Any] = None,
        serving_endpoint: Optional[str] = None,
    ) -> None:
        """Initialize the ReAct orchestrator.

        Args:
            tools: ToolRegistry containing available tools (defaults to default registry).
            llm_client: Callable taking message history list and returning model response string.
            max_steps: Maximum reasoning loop cycles before forcing termination.
            max_error_retries: Maximum consecutive self-correction retries per failed action.
            audit_manager: Optional MySQLAuditManager instance for relational telemetry.
            serving_endpoint: Optional URL to local FastAPI serving endpoint (e.g. http://localhost:8000/v1/chat/completions).
        """
        self.tools = tools or create_default_tool_registry()
        self.max_steps = max_steps
        self.max_error_retries = max_error_retries
        self.audit_manager = audit_manager
        self.serving_endpoint = serving_endpoint

        if llm_client is not None:
            self.llm_client = llm_client
        elif serving_endpoint:
            self.llm_client = create_fastapi_serving_client(
                endpoint_url=serving_endpoint,
                fallback_client=self._default_llm_backend,
            )
        else:
            self.llm_client = self._default_llm_backend

    def _build_tool_descriptions(self) -> str:
        """Format registered tools for injection into the system prompt."""
        descriptions = []
        for name, tool in self.tools._tools.items():
            params_repr = json.dumps(tool.parameters.get("properties", {}), indent=2)
            descriptions.append(
                f"- `{name}`: {tool.description}\n  Parameters: {params_repr}"
            )
        return "\n\n".join(descriptions)

    def _default_llm_backend(self, messages: List[Dict[str, str]]) -> str:
        """Built-in heuristic reasoner for testing and offline execution without remote API keys.

        Analyzes recent user request and conversation history to simulate an intelligent
        ReAct agent: executes python code, queries vector store, searches web, and handles errors.
        """
        last_msg = messages[-1]["content"]

        # 1. Check if we just received an observation
        if "Observation:" in last_msg:
            # Check for error in observation
            if "ERROR" in last_msg or "Traceback" in last_msg or "SyntaxError" in last_msg:
                # If error is a Python SyntaxError or name error, demonstrate self-correction!
                if "SyntaxError" in last_msg or "invalid syntax" in last_msg:
                    return (
                        "Thought: The previous Python script failed due to a syntax error. "
                        "I will analyze the syntax error and correct the code.\n"
                        "Action: execute_python_code\n"
                        'Action Input: {"code": "nums = [x for x in range(1, 11)]\\nprint(f\'Sum: {sum(nums)}\')\\n"}'
                    )
                elif "ZeroDivisionError" in last_msg:
                    return (
                        "Thought: The previous code encountered division by zero. "
                        "I will add a guard condition to prevent dividing by zero.\n"
                        "Action: execute_python_code\n"
                        'Action Input: {"code": "denominator = 1\\nprint(f\'Safe result: {100 / denominator}\')\\n"}'
                    )
                else:
                    return (
                        "Thought: The tool reported an error. I will formulate a corrected query.\n"
                        "Action: query_vector_memory\n"
                        'Action Input: {"query": "deep learning architectures transformer"}'
                    )

            # Observation was successful! Formulate final answer based on observation
            lines = last_msg.strip().split("\n")
            obs_content = "\n".join(lines[:15])
            return (
                "Thought: I have obtained the necessary observation and data to answer the query.\n"
                f"Final Answer: ### Research Analysis\n\nBased on our investigation:\n{obs_content}\n\n"
                "The findings confirm the expected behavior and provide rigorous empirical backing."
            )

        # 2. Initial user query analysis
        user_query = messages[1]["content"] if len(messages) > 1 else last_msg
        user_lower = user_query.lower()

        if any(w in user_lower for w in ["calculate", "compute", "fibonacci", "prime", "python", "code", "math", "sum"]):
            # Trigger Python tool
            if "syntax error test" in user_lower:
                # Deliberate error to test self-correction
                return (
                    "Thought: I will execute Python code to test error handling.\n"
                    "Action: execute_python_code\n"
                    'Action Input: {"code": "def broken(x) print(x)"}'
                )
            elif "fibonacci" in user_lower:
                code = (
                    "def fib(n):\n"
                    "    a, b = 0, 1\n"
                    "    res = []\n"
                    "    for _ in range(n):\n"
                    "        res.append(a)\n"
                    "        a, b = b, a + b\n"
                    "    return res\n\n"
                    "seq = fib(10)\n"
                    "tenth = seq[-1]\n"
                    "is_prime = tenth > 1 and all(tenth % i != 0 for i in range(2, int(tenth**0.5) + 1))\n"
                    "print(f'First 10 Fibonacci: {seq}')\n"
                    "print(f'10th number: {tenth}, Is Prime: {is_prime}')"
                )
                return (
                    "Thought: I need to calculate the first 10 Fibonacci numbers and check if the 10th number is prime. "
                    "I will write a Python script in the sandbox.\n"
                    "Action: execute_python_code\n"
                    f"Action Input: {json.dumps({'code': code})}"
                )
            else:
                code = "import math\nprint(f'Computed value: {math.pi * 42}')"
                return (
                    "Thought: I will execute the calculation using the Python sandbox.\n"
                    "Action: execute_python_code\n"
                    f"Action Input: {json.dumps({'code': code})}"
                )

        elif any(w in user_lower for w in ["vector", "memory", "paper", "stored", "arxiv", "archive", "robotics"]):
            # Trigger Vector memory query
            search_term = "robotics coding agents" if "robotics" in user_lower else "machine learning artificial intelligence"
            return (
                "Thought: The user is inquiring about stored papers or technical topics. "
                "I will query our persistent vector memory index.\n"
                "Action: query_vector_memory\n"
                f'Action Input: {{"query": "{search_term}", "top_k": 3}}'
            )

        elif any(w in user_lower for w in ["web", "live", "current", "latest news", "search"]):
            # Trigger Live web search
            return (
                "Thought: The user is requesting live updates from the web. "
                "I will perform a targeted web retrieval.\n"
                "Action: fetch_live_web\n"
                'Action Input: {"query": "latest breakthroughs in LLM reasoning agents", "max_results": 3}'
            )

        else:
            # Direct response
            return (
                "Thought: The objective can be directly answered based on fundamental principles.\n"
                f"Final Answer: In response to your inquiry regarding '{user_query}':\n\n"
                "Autonomous agentic architectures utilize iterative reasoning loops (ReAct) "
                "interleaved with tool usage and reflection to execute complex multi-step workflows."
            )

    def _parse_model_output(self, output: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """Parse Thought, Action, Action Input, and Final Answer from model response.

        Returns:
            Tuple of (thought, action_name, action_input_dict, final_answer)
        """
        thought = ""
        action = None
        action_input = None
        final_answer = None

        # 1. Check for Final Answer
        final_match = re.search(r"Final Answer\s*:\s*(.*)", output, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_answer = final_match.group(1).strip()

        # 2. Extract Thought
        thought_match = re.search(r"Thought\s*:\s*(.*?)(?=\nAction\s*:|\nFinal Answer\s*:|$)", output, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        elif not final_answer:
            thought = output.strip()

        # 3. Extract Action
        action_match = re.search(r"Action\s*:\s*([a-zA-Z0-9_\-]+)", output)
        if action_match:
            action = action_match.group(1).strip()

        # 4. Extract Action Input
        input_match = re.search(r"Action Input\s*:\s*(.*?)(?=\nObservation\s*:|\nThought\s*:|$)", output, re.DOTALL | re.IGNORECASE)
        if input_match and action:
            raw_input = input_match.group(1).strip()
            # Strip markdown code fences if present (```json ... ```)
            raw_input = re.sub(r"^```(?:json)?\s*", "", raw_input)
            raw_input = re.sub(r"\s*```$", "", raw_input)
            raw_input = raw_input.strip()

            try:
                action_input = json.loads(raw_input)
                if not isinstance(action_input, dict):
                    action_input = {"input": action_input}
            except json.JSONDecodeError:
                # Handle single string parameter or key-value fallback
                action_input = {"raw_input": raw_input}

        return thought, action, action_input, final_answer

    def _is_error_observation(self, obs: str) -> bool:
        """Determine if tool observation represents an error requiring self-correction."""
        if not obs:
            return True
        upper_obs = obs.upper()
        error_indicators = [
            "ERROR:",
            "ERROR (",
            "SYNTAXERROR",
            "ZERODIVISIONERROR",
            "NAMEERROR",
            "TRACEBACK (MOST RECENT CALL LAST)",
            "CODE EXECUTION EXCEEDED",
            "DISALLOWED_CALLS",
        ]
        return any(ind in upper_obs for ind in error_indicators)

    def _extract_sources(self, steps: List[ReActStep]) -> List[str]:
        """Extract links, URLs, and paper references from tool observations."""
        sources = set()
        url_pattern = re.compile(r"https?://[^\s\)]+")

        for step in steps:
            if step.observation:
                found_urls = url_pattern.findall(step.observation)
                for u in found_urls:
                    clean_u = u.rstrip(".,;")
                    sources.add(clean_u)

                # Extract titles from vector memory
                titles = re.findall(r"Title:\s*([^\n]+)", step.observation)
                for t in titles:
                    sources.add(f"Paper: {t.strip()}")

        return sorted(list(sources))

    def _audit_session(self, final_answer_obj: AgentFinalAnswer) -> None:
        """Record session metadata and step traces to MySQL audit table if available."""
        mgr = self.audit_manager
        if mgr is None:
            try:
                from storage.mysql_manager import get_mysql_manager
                mgr = get_mysql_manager()
            except Exception:
                return

        if mgr is not None:
            try:
                session_data = {
                    "session_id": f"sess_{int(time.time() * 1000)}_{os.getpid() % 10000}",
                    "input_objective": final_answer_obj.query,
                    "status": "success" if final_answer_obj.success else "failed",
                    "total_steps": final_answer_obj.total_steps,
                    "retry_count": final_answer_obj.retries_used,
                    "execution_latency": final_answer_obj.execution_time_seconds,
                }
                steps_data = [
                    {
                        "step_index": s.step_number,
                        "thought_trace": s.thought,
                        "action_name": s.action,
                        "action_input": s.action_input,
                        "observation_output": s.observation,
                        "step_latency": s.duration_seconds,
                    }
                    for s in final_answer_obj.steps
                ]
                mgr.record_agent_session(session_data, steps_data)
            except Exception as exc:
                logger.debug("MySQL audit session logging skipped/failed: %s", exc)

    def run(self, query: str) -> AgentFinalAnswer:
        """Execute the cyclic ReAct reasoning loop for the given query.

        Args:
            query: The user objective or research question.

        Returns:
            AgentFinalAnswer structured object containing the answer, steps, and telemetry.
        """
        start_time = time.perf_counter()
        logger.info("Initiating ReAct reasoning cycle for query: '%s'", query)

        system_msg = SYSTEM_PROMPT.format(tool_descriptions=self._build_tool_descriptions())
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]

        steps: List[ReActStep] = []
        consecutive_errors = 0
        total_retries = 0
        step_number = 1

        while step_number <= self.max_steps:
            step_start = time.perf_counter()

            # Call LLM backend
            try:
                response = self.llm_client(messages)
            except Exception as exc:
                logger.error("LLM backend invocation failed: %s", exc)
                err_answer = AgentFinalAnswer(
                    query=query,
                    final_answer=f"ERROR: LLM invocation failure: {exc}",
                    steps=steps,
                    total_steps=len(steps),
                    retries_used=total_retries,
                    success=False,
                    execution_time_seconds=time.perf_counter() - start_time,
                )
                self._audit_session(err_answer)
                return err_answer

            thought, action, action_input, final_answer = self._parse_model_output(response)

            # Check if agent produced a final answer
            if final_answer is not None:
                step_duration = time.perf_counter() - step_start
                final_step = ReActStep(
                    step_number=step_number,
                    thought=thought or "Final synthesis reached.",
                    duration_seconds=step_duration,
                )
                steps.append(final_step)
                logger.info("ReAct cycle concluded successfully at step %d", step_number)

                ans = AgentFinalAnswer(
                    query=query,
                    final_answer=final_answer,
                    steps=steps,
                    total_steps=len(steps),
                    retries_used=total_retries,
                    sources=self._extract_sources(steps),
                    success=True,
                    execution_time_seconds=time.perf_counter() - start_time,
                )
                self._audit_session(ans)
                return ans

            # Check if an action was selected
            if not action:
                logger.warning("Step %d did not specify an Action or Final Answer.", step_number)
                # Force synthesis or clarification
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please proceed by either choosing an Action with valid Action Input, or concluding with 'Final Answer:'."
                })
                step_number += 1
                continue

            # Execute the selected tool
            logger.info("Step %d: Executing action '%s' with input: %s", step_number, action, action_input)
            action_kwargs = action_input or {}
            observation = self.tools.execute(action, **action_kwargs)
            step_duration = time.perf_counter() - step_start

            is_error = self._is_error_observation(observation)
            reflection_text: Optional[str] = None

            if is_error:
                consecutive_errors += 1
                total_retries += 1
                logger.warning(
                    "Action '%s' returned an error (consecutive errors: %d/%d): %s",
                    action,
                    consecutive_errors,
                    self.max_error_retries,
                    observation[:200],
                )

                if consecutive_errors <= self.max_error_retries:
                    reflection_text = (
                        f"The action '{action}' encountered an error. "
                        f"Analyzing error trace and preparing self-correction (Retry {consecutive_errors}/{self.max_error_retries})."
                    )
                else:
                    reflection_text = (
                        f"Exceeded maximum retries ({self.max_error_retries}) for action '{action}'. "
                        "Switching strategy or synthesizing best available response."
                    )
            else:
                # Successful execution resets consecutive error counter
                consecutive_errors = 0

            step = ReActStep(
                step_number=step_number,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                is_error=is_error,
                reflection=reflection_text,
                retry_count=consecutive_errors if is_error else 0,
                duration_seconds=step_duration,
            )
            steps.append(step)

            # Append trajectory to context
            messages.append({
                "role": "assistant",
                "content": f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input or {})}",
            })

            # Formulate user observation feedback with reflection instruction if errored
            obs_message = f"Observation: {observation}"
            if is_error and consecutive_errors <= self.max_error_retries:
                obs_message += (
                    f"\n\n[SYSTEM NOTIFICATION]: The previous action produced an error. "
                    f"Reflect on the root cause and execute a self-corrected Action (Attempt {consecutive_errors}/{self.max_error_retries})."
                )
            elif is_error and consecutive_errors > self.max_error_retries:
                obs_message += (
                    "\n\n[SYSTEM NOTIFICATION]: Maximum retry limit reached for this action. "
                    "Acknowledge the limitation and proceed to 'Final Answer:' with your findings."
                )

            messages.append({"role": "user", "content": obs_message})
            step_number += 1

        # Fallback if loop exceeded max steps
        logger.warning("ReAct reasoning reached maximum step limit (%d). Forcing final synthesis.", self.max_steps)
        final_answer = (
            "Maximum reasoning step budget reached. Summary of observations:\n"
            + "\n".join([f"- Step {s.step_number} ({s.action}): {str(s.observation)[:150]}" for s in steps if s.action])
        )

        fallback_ans = AgentFinalAnswer(
            query=query,
            final_answer=final_answer,
            steps=steps,
            total_steps=len(steps),
            retries_used=total_retries,
            sources=self._extract_sources(steps),
            success=False,
            execution_time_seconds=time.perf_counter() - start_time,
        )
        self._audit_session(fallback_ans)
        return fallback_ans
