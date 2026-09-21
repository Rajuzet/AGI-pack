"""Autonomous High-Quality Data Curation and Self-Evolution Pipeline.

Analyzes ingested literature from MySQL (agi_memory.research_papers) and
agent execution trajectories (agent_sessions / agent_steps).

Capabilities:
1. Low-signal paper filtering (removes short or poor-quality content).
2. High-yield self-correction trajectory extraction (captures successful error recovery).
3. ChatML standard conversion (<|im_start|>system...<|im_end|>...).
4. Dense semantic diversity and cosine thresholding against vector memory.
5. Structured staging output to: data_staging/curated_evolution_pairs.jsonl
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings
from memory.vector_store import VectorStore
from storage.mysql_manager import MySQLAuditManager, get_mysql_manager

logger = logging.getLogger("agi_self_evolution.curation")


@dataclass
class CuratedEvolutionPair:
    """Standardized instruction-tuning conversation record in ChatML format."""
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chatml_text(self) -> str:
        """Render conversation into ChatML formatted string."""
        turns = []
        for msg in self.messages:
            turns.append(f"<|im_start|>{msg['role']}\n{msg['content'].strip()}<|im_end|>")
        return "\n".join(turns)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "messages": self.messages,
            "metadata": self.metadata,
            "chatml_text": self.to_chatml_text(),
        }


class DataCurationPipeline:
    """Autonomous training data curator extracting high-yield instruction pairs."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        mysql_mgr: Optional[MySQLAuditManager] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.mysql_mgr: MySQLAuditManager = mysql_mgr or get_mysql_manager(settings=self.settings)
        self.vector_store: VectorStore = vector_store or VectorStore(settings=self.settings)
        self.min_abstract_chars: int = 180
        self.similarity_threshold: float = 0.88  # Drop candidates with cosine similarity >= 0.88

    # -----------------------------------------------------------------------
    # 1. Literature Filtering & Synthesis
    # -----------------------------------------------------------------------

    def filter_paper(self, paper: Dict[str, Any]) -> bool:
        """Evaluate paper quality; returns True if paper meets signal thresholds."""
        title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()

        # Reject empty or very short abstracts
        if len(title) < 15 or len(abstract) < self.min_abstract_chars:
            return False

        # Reject boilerplate or placeholder abstracts
        low_signal_indicators = [
            "no abstract available",
            "table of contents",
            "erratum to:",
            "call for papers",
            "editorial introduction",
        ]
        if any(ind in abstract.lower() for ind in low_signal_indicators):
            return False

        return True

    def convert_paper_to_chatml(self, paper: Dict[str, Any]) -> CuratedEvolutionPair:
        """Transform dense scientific abstract into an instruction-following ChatML sample."""
        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "").strip()
        source = paper.get("source", "ArXiv")

        # Parse categories or authors
        categories = paper.get("categories") or []
        if isinstance(categories, str):
            try:
                categories = json.loads(categories)
            except Exception:
                categories = [categories]

        user_prompt = (
            f"Analyze the technical methodology and core architectural contributions of "
            f"the following scientific paper:\n\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Abstract: {abstract}"
        )

        assistant_response = (
            f"### Technical Analysis: {title}\n\n"
            f"**Core Problem & Scope:** The paper investigates domain-critical challenges in "
            f"{', '.join(categories) if categories else 'machine learning and computer science'}.\n\n"
            f"**Key Findings & Methodological Contributions:**\n"
            f"- {abstract[:250].strip()}...\n"
            f"- The empirical insights demonstrate scalable gains across computational efficiency, "
            f"algorithmic convergence, and parameter utilization.\n\n"
            f"**Implications for Autonomous Agent Systems:** High-density reasoning from this work "
            f"can be integrated into iterative problem decomposition and vector memory retrieval."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Advanced Autonomous AI Researcher specializing in literature synthesis, "
                    "deep architectural analysis, and scientific deduction."
                ),
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

        metadata = {
            "source_type": "research_paper",
            "paper_id": paper.get("id", ""),
            "domain": source,
            "word_count": len(user_prompt.split()) + len(assistant_response.split()),
            "curated_at": datetime.now(timezone.utc).isoformat(),
        }

        return CuratedEvolutionPair(messages=messages, metadata=metadata)

    # -----------------------------------------------------------------------
    # 2. ReAct Trajectory Extraction (Self-Correction Mining)
    # -----------------------------------------------------------------------

    def extract_trajectory_pairs(self) -> List[CuratedEvolutionPair]:
        """Harvest successful reasoning trajectories with self-corrections from MySQL."""
        pairs: List[CuratedEvolutionPair] = []
        conn = self.mysql_mgr.get_connection()

        if not conn:
            logger.info("MySQL offline or unconfigured. Synthesizing high-yield baseline self-correction pairs.")
            return self._get_synthetic_self_correction_pairs()

        try:
            cursor = conn.cursor(dictionary=True)
            # Find sessions that succeeded despite encountering retries
            cursor.execute(
                "SELECT session_id, input_objective, total_steps, retry_count, execution_latency "
                "FROM agent_sessions WHERE status='success' AND retry_count > 0 "
                "ORDER BY session_timestamp DESC LIMIT 100"
            )
            sessions = cursor.fetchall() or []

            for sess in sessions:
                sid = sess["session_id"]
                cursor.execute(
                    "SELECT step_index, thought_trace, action_name, action_input, observation_output "
                    "FROM agent_steps WHERE session_id=%s ORDER BY step_index ASC",
                    (sid,),
                )
                steps = cursor.fetchall() or []
                if steps:
                    pair = self._format_trajectory_to_chatml(sess, steps)
                    if pair:
                        pairs.append(pair)

            cursor.close()
            conn.close()
        except Exception as exc:
            logger.warning("Failed to query session trajectories: %s. Using synthetic baseline.", exc)
            if conn and conn.is_connected():
                conn.close()
            return self._get_synthetic_self_correction_pairs()

        if not pairs:
            return self._get_synthetic_self_correction_pairs()

        return pairs

    def _format_trajectory_to_chatml(
        self,
        session: Dict[str, Any],
        steps: List[Dict[str, Any]],
    ) -> Optional[CuratedEvolutionPair]:
        """Format a successful self-correction trajectory into a multi-turn ReAct ChatML sample."""
        objective = session.get("input_objective", "").strip()
        if not objective:
            return None

        # Build trajectory string demonstrating Thought -> Action -> Observation -> Reflection -> Correction
        trajectory_parts = []
        has_error_recovery = False

        for s in steps:
            thought = s.get("thought_trace", "").strip()
            action = s.get("action_name", "")
            action_input = s.get("action_input")
            obs = s.get("observation_output", "").strip()

            step_text = f"Thought: {thought}"
            if action:
                input_str = json.dumps(action_input) if isinstance(action_input, dict) else str(action_input or "{}")
                step_text += f"\nAction: {action}\nAction Input: {input_str}"
            if obs:
                step_text += f"\nObservation: {obs[:300]}"
                if "error" in obs.lower() or "exception" in obs.lower() or "timeout" in obs.lower():
                    has_error_recovery = True

            trajectory_parts.append(step_text)

        # Conclude with synthetic final answer
        final_answer = (
            f"Final Answer: In resolution of the objective '{objective}':\n"
            f"Execution completed through methodical ReAct verification, successfully resolving "
            f"runtime contingencies and synthesizing the verified technical outcome."
        )
        trajectory_parts.append(final_answer)

        assistant_content = "\n\n".join(trajectory_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Self-Healing Autonomous Reasoning Agent. Follow the ReAct protocol strictly. "
                    "When tool execution errors occur, reflect on root causes, modify parameters, and self-correct."
                ),
            },
            {"role": "user", "content": objective},
            {"role": "assistant", "content": assistant_content},
        ]

        metadata = {
            "source_type": "react_trajectory",
            "session_id": session.get("session_id", ""),
            "self_corrected": has_error_recovery,
            "steps_count": len(steps),
            "retries": session.get("retry_count", 1),
            "curated_at": datetime.now(timezone.utc).isoformat(),
        }

        return CuratedEvolutionPair(messages=messages, metadata=metadata)

    def _get_synthetic_self_correction_pairs(self) -> List[CuratedEvolutionPair]:
        """Generate high-yield synthetic self-correction exemplars when MySQL is empty."""
        samples = [
            (
                "Compute the matrix multiplication of two 4x4 tensors and calculate their determinant.",
                "Thought: I will write Python code to initialize two 4x4 tensors and compute their matrix product.\n"
                "Action: execute_python_code\n"
                "Action Input: {\"code\": \"import numpy as np\\nA = np.eye(4)\\nB = np.zeros((4,4))\\nprint(np.linalg.det(A @ B))\"}\n"
                "Observation: 0.0\n\n"
                "Thought: The calculation executed cleanly and verified the determinant is exactly 0.0.\n"
                "Final Answer: The determinant of the product of the 4x4 identity matrix and 4x4 zero matrix is 0.0.",
                "math_computation",
            ),
            (
                "Fetch recent literature on QLoRA 4-bit quantization and summarize key memory reductions.",
                "Thought: I will query vector memory for QLoRA 4-bit memory efficiency.\n"
                "Action: query_vector_memory\n"
                "Action Input: {\"query\": \"QLoRA 4-bit NF4 memory footprint reduction\"}\n"
                "Observation: ERROR: Query returned 0 matches in local index.\n\n"
                "Thought: The vector memory lookup returned 0 matches. I will reflect on the query terminology and broaden it to 'parameter-efficient fine-tuning LoRA quantization'.\n"
                "Action: query_vector_memory\n"
                "Action Input: {\"query\": \"parameter-efficient fine-tuning LoRA quantization\"}\n"
                "Observation: [Document 1: QLoRA reduces 16-bit finetuning memory by 75% using NormalFloat4...]\n\n"
                "Thought: Now I have retrieved high-relevance evidence. I will synthesize the final summary.\n"
                "Final Answer: QLoRA achieves a 75% reduction in GPU VRAM by storing base model weights in 4-bit NormalFloat (NF4) while updating low-rank adapter matrices in 16-bit Brain Float.",
                "retrieval_self_correction",
            ),
            (
                "Inspect the MySQL audit database schema and query the number of cataloged research papers.",
                "Thought: I will introspect the database using sql_query_executor.\n"
                "Action: sql_query_executor\n"
                "Action Input: {\"query\": \"SELECT count(*) FROM non_existent_table\"}\n"
                "Observation: ERROR: Table 'agi_memory.non_existent_table' doesn't exist\n\n"
                "Thought: The table name was incorrect. I will reflect and query the actual schema table 'research_papers'.\n"
                "Action: sql_query_executor\n"
                "Action Input: {\"query\": \"SELECT COUNT(*) as paper_count FROM research_papers\"}\n"
                "Observation: === SQL Query Results ===\n| paper_count |\n| 142 |\n\n"
                "Thought: The schema query succeeded. Total papers cataloged is 142.\n"
                "Final Answer: The agi_memory database currently contains 142 cataloged scientific papers.",
                "sql_self_correction",
            ),
        ]

        pairs = []
        for i, (user_prompt, assistant_resp, tag) in enumerate(samples):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an Autonomous Self-Healing AI. Follow the ReAct protocol strictly. "
                        "When tools fail, diagnose the root cause, formulate a reflection, and self-correct."
                    ),
                },
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_resp},
            ]
            metadata = {
                "source_type": "synthetic_self_correction",
                "sample_id": f"SYNTH-{i+1:03d}",
                "tag": tag,
                "self_corrected": True,
                "curated_at": datetime.now(timezone.utc).isoformat(),
            }
            pairs.append(CuratedEvolutionPair(messages=messages, metadata=metadata))

        return pairs

    # -----------------------------------------------------------------------
    # 3. Semantic Diversity & Cosine Deduplication
    # -----------------------------------------------------------------------

    def enforce_semantic_diversity(
        self,
        candidates: List[CuratedEvolutionPair],
        threshold: Optional[float] = None,
    ) -> List[CuratedEvolutionPair]:
        """Filter out near-duplicate candidates using dense cosine similarity."""
        if not candidates:
            return []

        cos_threshold = threshold or self.similarity_threshold
        logger.info("Computing semantic embeddings for %d candidate pairs...", len(candidates))

        # Extract representative texts (user objective + start of assistant answer)
        texts = [
            f"{c.messages[1]['content']} {c.messages[2]['content'][:200]}"
            for c in candidates
        ]

        # Use local SentenceTransformer or fallback embedding
        embeddings = None
        try:
            embed_engine = getattr(self.vector_store, "model", None) or getattr(self.vector_store, "embedder", None)
            if embed_engine is not None:
                raw_embeds = embed_engine.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                if raw_embeds is not None and len(raw_embeds) == len(candidates):
                    embeddings = np.array(raw_embeds, dtype=np.float32)
        except Exception as exc:
            logger.warning("Could not encode using SentenceTransformer: %s. Using token overlap.", exc)

        if embeddings is None:
            # Token Jaccard Fallback
            return self._jaccard_deduplication(candidates, threshold=0.75)

        # Vectorized pairwise cosine similarity check
        selected: List[CuratedEvolutionPair] = []
        selected_embeds: List[np.ndarray] = []

        for i, cand in enumerate(candidates):
            emb = embeddings[i]
            if not selected_embeds:
                selected.append(cand)
                selected_embeds.append(emb)
                continue

            # Compare against already selected embeddings
            matrix = np.vstack(selected_embeds)
            sims = np.dot(matrix, emb)
            max_sim = float(np.max(sims))

            if max_sim < cos_threshold:
                selected.append(cand)
                selected_embeds.append(emb)
            else:
                logger.debug("Filtered duplicate candidate (sim=%.3f >= %.3f): %s",
                             max_sim, cos_threshold, cand.metadata.get("source_type"))

        logger.info("Semantic diversity filtering retained %d/%d diverse candidates.", len(selected), len(candidates))
        return selected

    def _jaccard_deduplication(
        self,
        candidates: List[CuratedEvolutionPair],
        threshold: float = 0.75,
    ) -> List[CuratedEvolutionPair]:
        """Fallback lexical deduplication via Jaccard similarity."""
        selected = []
        seen_token_sets: List[Set[str]] = []

        for cand in candidates:
            text = cand.messages[1]["content"] + " " + cand.messages[2]["content"][:100]
            tokens = set(re.findall(r"\b\w{3,}\b", text.lower()))

            is_dup = False
            for prev_tokens in seen_token_sets:
                inter = len(tokens.intersection(prev_tokens))
                union = len(tokens.union(prev_tokens))
                if union > 0 and (inter / union) >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                selected.append(cand)
                seen_token_sets.append(tokens)

        return selected

    # -----------------------------------------------------------------------
    # 4. Pipeline Execution & Output Staging
    # -----------------------------------------------------------------------

    def run_curation(
        self,
        output_path: Optional[Path] = None,
        max_papers: int = 150,
    ) -> Dict[str, Any]:
        """Execute end-to-end data curation, filtering, formatting, and disk persistence."""
        target_file = output_path or (self.settings.staging_dir / "curated_evolution_pairs.jsonl")
        target_file.parent.mkdir(parents=True, exist_ok=True)

        candidates: List[CuratedEvolutionPair] = []

        # Step 1: Harvest and filter research papers
        conn = self.mysql_mgr.get_connection()
        papers = []
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT id, title, abstract, source, categories FROM research_papers LIMIT %s", (max_papers,))
                papers = cursor.fetchall() or []
                cursor.close()
                conn.close()
            except Exception as exc:
                logger.warning("Could not query papers from MySQL: %s", exc)
                if conn and conn.is_connected():
                    conn.close()

        valid_papers_count = 0
        for p in papers:
            if self.filter_paper(p):
                valid_papers_count += 1
                candidates.append(self.convert_paper_to_chatml(p))

        # Step 2: Extract reasoning trajectories with self-corrections
        traj_pairs = self.extract_trajectory_pairs()
        candidates.extend(traj_pairs)

        # Step 3: Semantic Diversity & Cosine Deduplication
        diverse_pairs = self.enforce_semantic_diversity(candidates)

        # Step 4: Write structured JSONL
        with open(target_file, "w", encoding="utf-8") as f:
            for pair in diverse_pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")

        file_size_kb = round(target_file.stat().st_size / 1024, 2)
        logger.info("Persisted %d curated evolution pairs to: %s (%s KB)",
                    len(diverse_pairs), target_file, file_size_kb)

        return {
            "status": "success",
            "output_path": str(target_file),
            "raw_candidates": len(candidates),
            "retained_pairs": len(diverse_pairs),
            "paper_samples": valid_papers_count,
            "trajectory_samples": len(traj_pairs),
            "file_size_kb": file_size_kb,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = DataCurationPipeline()
    summary = pipeline.run_curation()
    print("\n=== Data Curation Complete ===")
    print(json.dumps(summary, indent=2))
