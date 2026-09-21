"""Autonomous Retraining Trigger and Dynamic LoRA Hyperparameter Optimization Engine.

Monitors live system signals (curated sample count, Prometheus metrics, benchmark scores)
and dynamically adjusts training hyperparameters based on past loss trajectories.

Capabilities:
1. Multi-factor retraining trigger evaluation (volume thresholds + evaluation regression).
2. Dynamic hyperparameter adaptation (LoRA rank r, alpha, target modules, learning rate).
3. Cloud training payload dispatch to GCS: gs://<bucket>/training_jobs/pending/
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings
from storage.gcs_manager import GCSManager, get_gcs_manager
from storage.mysql_manager import MySQLAuditManager, get_mysql_manager

logger = logging.getLogger("agi_self_evolution.auto_tune")


# ===========================================================================
# Data Models
# ===========================================================================

@dataclass
class DynamicLoRAHyperparameters:
    """Fine-tuning hyperparameters dynamically tuned based on empirical loss trajectories."""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    tuning_rationale: str = "Standard default configuration"


@dataclass
class RetrainingDecision:
    """Decision summary outlining whether retraining was triggered and why."""
    should_retrain: bool
    trigger_reasons: List[str]
    curated_sample_count: int
    latest_eval_score: float
    prior_final_loss: Optional[float]
    recommended_params: DynamicLoRAHyperparameters
    job_spec_uri: Optional[str] = None
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ===========================================================================
# Auto-Tune Trigger Engine
# ===========================================================================

class AutoTuneTriggerEngine:
    """Autonomous evaluator evaluating continuous retraining readiness."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        mysql_mgr: Optional[MySQLAuditManager] = None,
        gcs_mgr: Optional[GCSManager] = None,
        min_curated_samples: int = 500,
        eval_score_threshold: float = 80.0,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.mysql_mgr: MySQLAuditManager = mysql_mgr or get_mysql_manager(settings=self.settings)
        self.gcs_mgr: GCSManager = gcs_mgr or get_gcs_manager(settings=self.settings)
        self.min_curated_samples = min_curated_samples
        self.eval_score_threshold = eval_score_threshold

    # -----------------------------------------------------------------------
    # Signal Inspection
    # -----------------------------------------------------------------------

    def get_curated_sample_count(self) -> int:
        """Count total instruction pairs staged in data_staging/curated_evolution_pairs.jsonl."""
        staging_file = self.settings.staging_dir / "curated_evolution_pairs.jsonl"
        if not staging_file.exists():
            return 0
        try:
            with open(staging_file, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception as exc:
            logger.warning("Failed to count lines in curated staging file: %s", exc)
            return 0

    def get_latest_benchmark_score(self) -> float:
        """Read composite evaluation score from evaluation/evaluation_results.json."""
        eval_file = Path("evaluation/evaluation_results.json")
        if not eval_file.exists():
            # Fallback to local default baseline
            return 85.0
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rankings = data.get("rankings", [])
            if rankings:
                # Top rank composite score
                top_score = rankings[0].get("metrics", {}).get("composite_score", 85.0)
                return float(top_score)
        except Exception as exc:
            logger.warning("Failed to read evaluation results JSON: %s", exc)
        return 85.0

    def get_prior_training_loss(self) -> Optional[float]:
        """Fetch final training loss of the most recent training run from MySQL."""
        conn = self.mysql_mgr.get_connection()
        if not conn:
            return None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT training_loss FROM training_runs "
                "ORDER BY executed_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if row and row.get("training_loss") is not None:
                return float(row["training_loss"])
        except Exception as exc:
            logger.debug("Failed to query prior training loss: %s", exc)
            if conn and conn.is_connected():
                conn.close()
        return None

    # -----------------------------------------------------------------------
    # Dynamic Hyperparameter Adaptation
    # -----------------------------------------------------------------------

    def calculate_hyperparameters(
        self,
        sample_count: int,
        prior_loss: Optional[float],
        eval_score: float,
    ) -> DynamicLoRAHyperparameters:
        """Derive optimal LoRA hyperparameters based on empirical telemetry."""
        # Default baseline
        r = 16
        alpha = 32
        lr = 2e-4
        modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        grad_accum = 8
        rationale = []

        # Case 1: Prior loss was severely high (> 1.8) or evaluation degraded significantly (< 70.0)
        if eval_score < 70.0 or (prior_loss is not None and prior_loss > 1.8):
            r = 64
            alpha = 64
            lr = 1e-4
            modules.extend(["gate_proj", "up_proj", "down_proj"])
            grad_accum = 16
            rationale.append(f"Performance regression (eval={eval_score:.1f}%): Scaled rank r=64 and lowered lr=1e-4.")

        # Case 2: Prior training loss was moderately high or plateaued (> 1.2)
        elif prior_loss is not None and prior_loss > 1.2:
            r = 32
            alpha = 64
            modules.extend(["gate_proj", "up_proj", "down_proj"])
            rationale.append(f"Prior loss high ({prior_loss:.3f} > 1.2): Scaled LoRA rank r=32, alpha=64 and added MLP layers.")

        # Case 3: Prior loss converged well (< 0.8) but large new dataset
        elif sample_count >= 1000:
            r = 32
            alpha = 64
            lr = 1.5e-4
            rationale.append(f"Large curated corpus ({sample_count} samples): Scaled rank r=32 for broader capacity.")

        # Case 4: Stable healthy regime
        else:
            rationale.append(f"Stable regime (prior loss={prior_loss}, eval={eval_score:.1f}%): Maintained efficient r=16 baseline.")

        return DynamicLoRAHyperparameters(
            lora_r=r,
            lora_alpha=alpha,
            lora_dropout=0.05,
            learning_rate=lr,
            target_modules=modules,
            batch_size=2,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=3,
            warmup_ratio=0.03,
            tuning_rationale="; ".join(rationale),
        )

    # -----------------------------------------------------------------------
    # Decision Evaluation & Cloud Dispatch
    # -----------------------------------------------------------------------

    def evaluate_retraining_trigger(
        self,
        force: bool = False,
        dry_run: bool = False,
    ) -> RetrainingDecision:
        """Evaluate signals and conditionally stream cloud job payload to GCS."""
        curated_count = self.get_curated_sample_count()
        eval_score = self.get_latest_benchmark_score()
        prior_loss = self.get_prior_training_loss()

        reasons = []
        if force:
            reasons.append("Retraining explicitly forced by operator.")

        if curated_count >= self.min_curated_samples:
            reasons.append(f"Curated dataset reached threshold ({curated_count} >= {self.min_curated_samples} samples).")

        if eval_score < self.eval_score_threshold:
            reasons.append(f"Model benchmark composite score dropped below target ({eval_score:.1f}% < {self.eval_score_threshold}%).")

        should_retrain = len(reasons) > 0
        recommended_params = self.calculate_hyperparameters(curated_count, prior_loss, eval_score)

        job_spec_uri = None
        if should_retrain and not dry_run:
            job_spec_uri = self._dispatch_cloud_job(curated_count, recommended_params)

        return RetrainingDecision(
            should_retrain=should_retrain,
            trigger_reasons=reasons,
            curated_sample_count=curated_count,
            latest_eval_score=eval_score,
            prior_final_loss=prior_loss,
            recommended_params=recommended_params,
            job_spec_uri=job_spec_uri,
        )

    def _dispatch_cloud_job(
        self,
        sample_count: int,
        params: DynamicLoRAHyperparameters,
    ) -> Optional[str]:
        """Upload training job payload and curated dataset to GCS."""
        run_id = f"auto_train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        staging_file = self.settings.staging_dir / "curated_evolution_pairs.jsonl"

        job_payload = {
            "run_id": run_id,
            "base_model": self.settings.serving_model_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": sample_count,
            "hyperparameters": asdict(params),
            "target_runtime": "kaggle_gpu_p100",
            "gcs_output_checkpoint_uri": f"gs://{self.settings.gcs_bucket_name}/models/lora_checkpoints/{run_id}/",
        }

        # Write local spec
        spec_file = self.settings.staging_dir / f"job_spec_{run_id}.json"
        spec_file.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")

        # Stream spec and dataset to GCS under training_jobs/pending/
        remote_spec_blob = f"training_jobs/pending/{spec_file.name}"
        remote_data_blob = f"training_jobs/pending/dataset_{run_id}.jsonl"

        try:
            # Upload job spec
            spec_uri = self.gcs_mgr.upload_file(spec_file, remote_spec_blob, content_type="application/json")
            # Upload curated dataset if exists
            if staging_file.exists():
                self.gcs_mgr.upload_file(staging_file, remote_data_blob, content_type="application/x-ndjson")
            logger.info("Successfully dispatched cloud training job to GCS: %s", spec_uri)
            return spec_uri
        except Exception as exc:
            logger.warning("GCS dispatch skipped/failed (offline mode): %s", exc)
            return f"gs://{self.settings.gcs_bucket_name}/{remote_spec_blob} (staged locally: {spec_file})"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Auto-Tune Retraining Trigger Evaluator")
    parser.add_argument("--force", action="store_true", help="Force retraining trigger regardless of threshold")
    parser.add_argument("--min-samples", type=int, default=500, help="Minimum sample threshold to trigger")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without dispatching GCS payload")
    args = parser.parse_args()

    engine = AutoTuneTriggerEngine(min_curated_samples=args.min_samples)
    decision = engine.evaluate_retraining_trigger(force=args.force, dry_run=args.dry_run)

    print("\n=== Retraining Trigger Decision ===")
    print(f"Should Retrain: {decision.should_retrain}")
    print(f"Curated Samples: {decision.curated_sample_count}")
    print(f"Latest Eval Score: {decision.latest_eval_score}%")
    print(f"Trigger Reasons: {decision.trigger_reasons}")
    print(f"Tuning Rationale: {decision.recommended_params.tuning_rationale}")
    print(f"Recommended LoRA Rank: r={decision.recommended_params.lora_r}, alpha={decision.recommended_params.lora_alpha}")
    print(f"Job Spec GCS URI: {decision.job_spec_uri}")
