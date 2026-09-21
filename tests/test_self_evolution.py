"""Automated Test Suite for Continuous Self-Evolution, Auto-Tuning, and Multi-Agent Consensus.

Validates:
1. Curation Pipeline:
   - Paper signal and content length filtering.
   - Trajectory mining for ReAct tool error recovery and self-correction.
   - Standard ChatML formatting (<|im_start|>...<|im_end|>).
   - Semantic diversity checking with cosine distance deduplication.
2. Auto-Tune Trigger:
   - Volume-based and evaluation degradation trigger conditions.
   - Dynamic LoRA hyperparameter optimization based on historical loss curves.
   - Cloud training payload generation and GCS dispatch.
3. Multi-Agent Consensus:
   - Planner DAG decomposition.
   - Executor toolchain dispatch.
   - Critic groundedness inspection and citation audit.
   - Multi-round debate protocol convergence and markdown reporting.
4. DSPy Prompt Optimizer:
   - Parameterized prompt mutation and compilation.
   - Empirical candidate scoring and leaderboard generation.
   - Persistence of winning prompt configurations.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from agent.multi_agent_consensus import (
    ConsensusResult,
    CriticAgent,
    CriticReview,
    DebateRound,
    ExecutionPlan,
    ExecutorAgent,
    MultiAgentConsensusOrchestrator,
    PlannerAgent,
    PlanStep,
)
from agent.tools import ToolRegistry
from config import Settings
from optimization.dspy_prompt_optimizer import (
    BASELINE_PROMPT,
    DSPyPromptOptimizer,
    ParameterizedPrompt,
    PromptEvaluationScore,
)
from self_evolution.auto_tune_trigger import (
    AutoTuneTriggerEngine,
    DynamicLoRAHyperparameters,
    RetrainingDecision,
)
from self_evolution.curation_pipeline import (
    CuratedEvolutionPair,
    DataCurationPipeline,
)


# ===========================================================================
# 1. Curation Pipeline Tests
# ===========================================================================


class TestCurationPipeline:
    """Test suite for autonomous high-quality data curation."""

    @pytest.fixture
    def mock_pipeline(self, tmp_path):
        """Construct DataCurationPipeline with temporary directory and mocked dependencies."""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.staging_dir = tmp_path / "staging"
        mock_settings.staging_dir.mkdir(parents=True, exist_ok=True)
        mock_mysql = MagicMock()
        mock_vstore = MagicMock()
        pipeline = DataCurationPipeline(
            settings=mock_settings,
            mysql_mgr=mock_mysql,
            vector_store=mock_vstore,
        )
        return pipeline, mock_mysql, mock_vstore

    def test_filter_low_signal_papers(self, mock_pipeline):
        pipeline, _, _ = mock_pipeline

        good_paper = {
            "title": "Parameter-Efficient LoRA Adaptation of Large Models",
            "abstract": (
                "Low-Rank Adaptation (LoRA) freezes pretrained model weights and injects "
                "trainable rank decomposition matrices into each layer of the Transformer architecture, "
                "greatly reducing the number of trainable parameters for downstream tasks." * 2
            ),
            "source": "arxiv",
        }
        short_paper = {
            "title": "Short Stub Paper",
            "abstract": "Too short content.",
            "source": "arxiv",
        }
        boilerplate_paper = {
            "title": "Call for Papers - Workshop on AI",
            "abstract": "Call for papers on autonomous agents and systems " * 10,
            "source": "arxiv",
        }

        assert pipeline.filter_paper(good_paper) is True
        assert pipeline.filter_paper(short_paper) is False
        assert pipeline.filter_paper(boilerplate_paper) is False

        pair = pipeline.convert_paper_to_chatml(good_paper)
        assert isinstance(pair, CuratedEvolutionPair)
        assert len(pair.messages) == 3
        assert pair.messages[0]["role"] == "system"
        assert pair.messages[1]["role"] == "user"
        assert pair.messages[2]["role"] == "assistant"
        assert pair.metadata["source_type"] == "research_paper"

    def test_extract_self_correction_trajectories(self, mock_pipeline):
        pipeline, _, _ = mock_pipeline

        session = {
            "session_id": "sess-recovery-001",
            "input_objective": "Compute factorial of 15 using python",
            "status": "success",
            "total_steps": 2,
            "retry_count": 1,
        }
        steps = [
            {
                "step_index": 1,
                "thought_trace": "I will run python to calculate factorial",
                "action_name": "execute_python_code",
                "action_input": {"code": "print(math.factorial(15))"},
                "observation_output": "ERROR: NameError: name 'math' is not defined",
            },
            {
                "step_index": 2,
                "thought_trace": "math was not imported. I must import math first.",
                "action_name": "execute_python_code",
                "action_input": {"code": "import math\nprint(math.factorial(15))"},
                "observation_output": "1307674368000",
            },
        ]

        pair = pipeline._format_trajectory_to_chatml(session, steps)
        assert pair is not None
        assert pair.metadata["source_type"] == "react_trajectory"
        assert pair.metadata["self_corrected"] is True
        assert pair.metadata["session_id"] == "sess-recovery-001"

        assistant_reply = pair.messages[2]["content"]
        assert "NameError" in assistant_reply
        assert "import math" in assistant_reply
        assert "1307674368000" in assistant_reply

    def test_chatml_formatting(self):
        pair = CuratedEvolutionPair(
            messages=[
                {"role": "system", "content": "You are an autonomous AGI system."},
                {"role": "user", "content": "What is LoRA?"},
                {"role": "assistant", "content": "Low-Rank Adaptation freezes base weights."},
            ],
            metadata={"source": "test"},
        )
        chatml_text = pair.to_chatml_text()
        assert "<|im_start|>system\nYou are an autonomous AGI system.<|im_end|>" in chatml_text
        assert "<|im_start|>user\nWhat is LoRA?<|im_end|>" in chatml_text
        assert "<|im_start|>assistant\nLow-Rank Adaptation freezes base weights.<|im_end|>" in chatml_text

    def test_semantic_diversity_filter(self, mock_pipeline):
        pipeline, _, mock_vstore = mock_pipeline

        pair1 = CuratedEvolutionPair(
            messages=[
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "LoRA rank decomposition parameter reduction"},
                {"role": "assistant", "content": "LoRA introduces low rank matrices into layers."},
            ]
        )
        pair2 = CuratedEvolutionPair(
            messages=[
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "LoRA rank decomposition parameter reduction"},
                {"role": "assistant", "content": "LoRA introduces low rank matrices into layers."},
            ]
        )
        pair3 = CuratedEvolutionPair(
            messages=[
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "Postgres index query optimization"},
                {"role": "assistant", "content": "B-tree index enables fast logarithmic search."},
            ]
        )

        # Mock vector_store model
        mock_model = MagicMock()
        mock_vstore.model = mock_model
        # Embeddings: pair 1 & 2 identical, pair 3 orthogonal
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        mock_model.encode.return_value = np.array([v1, v2, v3])

        deduped = pipeline.enforce_semantic_diversity([pair1, pair2, pair3], threshold=0.88)
        assert len(deduped) == 2
        assert deduped[0] == pair1
        assert deduped[1] == pair3


# ===========================================================================
# 2. Auto-Tune Trigger & Hyperparameter Tuning Tests
# ===========================================================================


class TestAutoTuneTrigger:
    """Test suite for autonomous retraining trigger evaluation and dynamic LoRA specs."""

    @pytest.fixture
    def mock_trigger(self, tmp_path):
        staging_dir = tmp_path / "data_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        pairs_file = staging_dir / "curated_evolution_pairs.jsonl"

        mock_settings = MagicMock(spec=Settings)
        mock_settings.staging_dir = staging_dir
        mock_settings.gcs_bucket_name = "agi-agent-ingestion-data"
        mock_settings.serving_model_id = "Qwen/Qwen2.5-7B-Instruct"

        mock_mysql = MagicMock()
        mock_gcs = MagicMock()
        engine = AutoTuneTriggerEngine(
            settings=mock_settings,
            mysql_mgr=mock_mysql,
            gcs_mgr=mock_gcs,
            min_curated_samples=100,
            eval_score_threshold=75.0,
        )
        return engine, mock_mysql, mock_gcs, pairs_file

    def test_trigger_on_sample_volume(self, mock_trigger):
        engine, mock_mysql, _, pairs_file = mock_trigger

        # Write 150 JSONL records to exceed threshold of 100
        with open(pairs_file, "w", encoding="utf-8") as f:
            for i in range(150):
                f.write(json.dumps({"messages": [{"role": "user", "content": f"query {i}"}]}) + "\n")

        with patch.object(engine, "get_latest_benchmark_score", return_value=85.0), \
             patch.object(engine, "get_prior_training_loss", return_value=0.82):
            decision = engine.evaluate_retraining_trigger(dry_run=True)

        assert decision.should_retrain is True
        assert any("reached threshold" in r for r in decision.trigger_reasons)
        assert decision.curated_sample_count == 150
        assert decision.recommended_params is not None

    def test_trigger_on_evaluation_degradation(self, mock_trigger):
        engine, mock_mysql, _, pairs_file = mock_trigger

        # Only 20 samples (under 100 threshold)
        with open(pairs_file, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"messages": [{"role": "user", "content": f"query {i}"}]}) + "\n")

        # Live score dropped to 68.5 (below degradation threshold of 75.0)
        with patch.object(engine, "get_latest_benchmark_score", return_value=68.5), \
             patch.object(engine, "get_prior_training_loss", return_value=1.45):
            decision = engine.evaluate_retraining_trigger(dry_run=True)

        assert decision.should_retrain is True
        assert any("dropped below target" in r for r in decision.trigger_reasons)
        assert decision.latest_eval_score == 68.5

    def test_dynamic_hyperparameter_optimization(self, mock_trigger):
        engine, _, _, _ = mock_trigger

        # Case A: Low loss, small dataset -> conservative rank 16
        spec_low = engine.calculate_hyperparameters(sample_count=50, prior_loss=0.45, eval_score=85.0)
        assert spec_low.lora_r == 16
        assert spec_low.lora_alpha == 32
        assert spec_low.learning_rate == 2e-4

        # Case B: Prior loss > 1.2 -> rank 32, alpha 64, expanded modules
        spec_mid = engine.calculate_hyperparameters(sample_count=50, prior_loss=1.35, eval_score=80.0)
        assert spec_mid.lora_r == 32
        assert spec_mid.lora_alpha == 64
        assert "gate_proj" in spec_mid.target_modules

        # Case C: Degradation eval_score < 70 or loss > 1.8 -> rank 64 with lower lr
        spec_high = engine.calculate_hyperparameters(sample_count=50, prior_loss=1.90, eval_score=65.0)
        assert spec_high.lora_r == 64
        assert spec_high.lora_alpha == 64
        assert spec_high.learning_rate == 1e-4
        assert "down_proj" in spec_high.target_modules

    def test_cloud_payload_dispatch(self, mock_trigger):
        engine, _, mock_gcs, pairs_file = mock_trigger

        # Write 5 samples
        with open(pairs_file, "w", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({"messages": [{"role": "user", "content": f"sample {i}"}]}) + "\n")

        mock_gcs.upload_file.return_value = "gs://agi-agent-ingestion-data/training_jobs/pending/job_test.json"

        with patch.object(engine, "get_latest_benchmark_score", return_value=85.0), \
             patch.object(engine, "get_prior_training_loss", return_value=0.5):
            decision = engine.evaluate_retraining_trigger(force=True, dry_run=False)

        assert decision.should_retrain is True
        assert decision.job_spec_uri is not None
        assert decision.job_spec_uri.startswith("gs://")
        assert mock_gcs.upload_file.called


# ===========================================================================
# 3. Multi-Agent Consensus Tests
# ===========================================================================


class TestMultiAgentConsensus:
    """Test suite for collaborative multi-agent reasoning panel."""

    @pytest.fixture
    def mock_panel(self):
        tool_registry = MagicMock(spec=ToolRegistry)
        tool_registry._tools = ["execute_python_code", "query_vector_memory", "sql_query_executor", "fetch_live_web"]
        orchestrator = MultiAgentConsensusOrchestrator(
            tools=tool_registry,
            max_debate_rounds=3,
        )
        return orchestrator, orchestrator.planner, orchestrator.executor, orchestrator.critic, tool_registry

    def test_planner_dag_decomposition(self, mock_panel):
        _, planner, _, _, _ = mock_panel
        goal = "Calculate memory consumption of 7B QLoRA model and compute factorial of 15"
        plan = planner.plan(goal)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) >= 2
        assert plan.steps[0].step_id == 1
        assert plan.steps[0].target_tool in ["execute_python_code", "query_vector_memory", "sql_query_executor"]
        assert "Empirical verification" in plan.strategy

    def test_executor_toolchain_dispatch(self, mock_panel):
        _, _, executor, _, tool_registry = mock_panel
        tool_registry.execute.return_value = "Memory for 4-bit: 3.5GB; 16-bit: 14.0GB"

        plan = ExecutionPlan(
            goal="Memory calculation",
            strategy="Empirical calculation",
            steps=[
                PlanStep(
                    step_id=1,
                    objective="Compute 4-bit footprint",
                    target_tool="execute_python_code",
                    expected_output="Memory footprint in GB",
                )
            ],
        )

        synthesis, observations = executor.execute(plan)
        assert len(observations) == 1
        assert observations[0]["tool"] == "execute_python_code"
        assert "Verified Findings" in synthesis
        assert "Memory for 4-bit" in synthesis

    def test_critic_revision_requested_when_claims_lack_evidence(self, mock_panel):
        _, _, _, critic, _ = mock_panel
        goal = "Estimate speedup of flash attention"
        # Purely conversational assertions with no tool observations
        review = critic.audit(
            goal=goal,
            synthesis="FlashAttention provides a 4.5x speedup without verification.",
            observations=[],
            round_number=1,
        )

        assert review.approved is False
        assert len(review.objections) > 0
        assert review.groundedness_score < 70.0
        assert any("lacks empirical tool evidence" in obj for obj in review.objections)

    def test_critic_approval_with_grounded_evidence(self, mock_panel):
        _, _, _, critic, _ = mock_panel
        goal = "Compute 15 factorial"
        review = critic.audit(
            goal=goal,
            synthesis=(
                "### Verified Findings for: Compute 15 factorial\n"
                "**Methodological Strategy:** Empirical verification via sandboxed code execution.\n"
                "**Empirical Tool Observations:**\n- [execute_python_code]: 1307674368000\n"
                "**Analytical Conclusion:** Based on verified tool execution, 15! = 1307674368000."
            ),
            observations=[
                {
                    "step_id": 1,
                    "tool": "execute_python_code",
                    "output": "1307674368000",
                }
            ],
            round_number=2,
        )

        assert review.approved is True
        assert review.groundedness_score >= 80.0
        assert len(review.objections) == 0

    def test_multi_agent_consensus_debate_loop(self, mock_panel):
        orchestrator, planner, executor, critic, tool_registry = mock_panel

        goal = "Inspect database for paper count"
        tool_registry.execute.side_effect = [
            "Table research_papers has 1,250 entries",
            "Mean citation count is 42.8",
        ]

        result = orchestrator.run_consensus(goal=goal)
        assert isinstance(result, ConsensusResult)
        assert result.consensus_approved is True
        assert result.total_rounds >= 1
        assert len(result.final_answer) > 0

        # Test Markdown report generation
        md = result.to_markdown()
        assert "# Multi-Agent Collaborative Consensus Report" in md
        assert "[APPROVED]" in md
        assert "Total Debate Rounds" in md


# ===========================================================================
# 4. DSPy Prompt Optimizer Tests
# ===========================================================================


class TestDSPyPromptOptimizer:
    """Test suite for autonomous prompt mutation and compilation."""

    @pytest.fixture
    def optimizer(self, tmp_path):
        out_path = tmp_path / "system_prompts.json"
        return DSPyPromptOptimizer(config_output_path=out_path)

    def test_parameterized_prompt_render(self):
        prompt = ParameterizedPrompt(
            prompt_id="test_prompt",
            persona="You are a test agent.",
            thought_instructions="Think carefully.",
            action_guidelines="Use tools precisely.",
            reflection_rules="Self-correct errors.",
            final_answer_format="Final Answer: <solution>",
        )
        rendered = prompt.render()
        assert "You are a test agent." in rendered
        assert "Think carefully." in rendered
        assert "Use tools precisely." in rendered
        assert "Self-correct errors." in rendered
        assert "{tool_descriptions}" in rendered

    def test_generate_candidate_mutations(self, optimizer):
        candidates = optimizer.generate_candidate_mutations(BASELINE_PROMPT)
        assert len(candidates) == 4
        ids = [c.prompt_id for c in candidates]
        assert "prompt_v1_baseline" in ids
        assert "prompt_v2_strict_schema" in ids
        assert "prompt_v3_causal_reflection" in ids
        assert "prompt_v4_grounded_citations" in ids

    def test_evaluate_and_persist_winning_prompt(self, optimizer, tmp_path):
        out_path = tmp_path / "winning_prompt.json"
        optimizer.config_output_path = out_path

        # Mock evaluation of candidates to test sorting and persistence
        scores = [
            PromptEvaluationScore(
                prompt_id="prompt_v1_baseline",
                composite_score=72.0,
                tool_accuracy=90.0,
                step_efficiency=2.0,
                self_correction_rate=80.0,
                semantic_faithfulness=40.0,
                evaluation_time_seconds=0.1,
            ),
            PromptEvaluationScore(
                prompt_id="prompt_v4_grounded_citations",
                composite_score=84.5,
                tool_accuracy=98.0,
                step_efficiency=1.8,
                self_correction_rate=100.0,
                semantic_faithfulness=65.0,
                evaluation_time_seconds=0.1,
            ),
        ]

        with patch.object(optimizer, "evaluate_candidate", side_effect=scores):
            # Generate only 2 test candidates
            with patch.object(optimizer, "generate_candidate_mutations", return_value=[
                BASELINE_PROMPT,
                ParameterizedPrompt(
                    prompt_id="prompt_v4_grounded_citations",
                    persona="Grounded Agent",
                    thought_instructions=BASELINE_PROMPT.thought_instructions,
                    action_guidelines=BASELINE_PROMPT.action_guidelines,
                    reflection_rules=BASELINE_PROMPT.reflection_rules,
                    final_answer_format=BASELINE_PROMPT.final_answer_format,
                ),
            ]):
                winning_prompt, ranked_scores = optimizer.optimize_system_prompt()

        assert winning_prompt.prompt_id == "prompt_v4_grounded_citations"
        assert ranked_scores[0].composite_score == 84.5
        assert out_path.is_file()

        # Validate saved JSON structure
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["active_prompt_id"] == "prompt_v4_grounded_citations"
        assert data["best_composite_score"] == 84.5
        assert len(data["optimization_leaderboard"]) == 2


# ===========================================================================
# 5. Settings Integration Tests
# ===========================================================================


class TestConfigSelfEvolutionSettings:
    """Verify settings integration for the self-evolution subsystem."""

    def test_settings_load_active_system_prompt(self, tmp_path):
        prompt_file = tmp_path / "system_prompts.json"
        prompt_file.write_text(
            json.dumps({"rendered_system_prompt": "Active Specialized System Prompt"}),
            encoding="utf-8",
        )

        settings = Settings(
            system_prompts_config_path=str(prompt_file),
            auto_retrain_sample_threshold=500,
            auto_retrain_score_threshold=70.0,
        )

        loaded_prompt = settings.load_active_system_prompt()
        assert loaded_prompt == "Active Specialized System Prompt"
        assert settings.auto_retrain_sample_threshold == 500
        assert settings.auto_retrain_score_threshold == 70.0
