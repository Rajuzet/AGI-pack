"""Unit and Integration Tests for Kubernetes Helm Packaging, GitOps Pipeline & HPA Scaling.

Validates:
1. Helm Chart metadata (Chart.yaml) and default production configuration (values.yaml).
2. All 11 Helm templates in deployments/helm/agi-synapse/templates/.
3. Kubernetes schema correctness: apps/v1, autoscaling/v2, networking.k8s.io/v1.
4. GitHub Actions CI/CD pipeline definition (.github/workflows/ci_cd_pipeline.yml).
5. HPA replica scaling calculation and benchmark report metrics.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HELM_DIR = PROJECT_ROOT / "deployments" / "helm" / "agi-synapse"
TEMPLATES_DIR = HELM_DIR / "templates"
WORKFLOW_FILE = PROJECT_ROOT / ".github" / "workflows" / "ci_cd_pipeline.yml"
OPERATIONS_DOC = PROJECT_ROOT / "deployments" / "K8S_OPERATIONS.md"
LOAD_TEST_SCRIPT = PROJECT_ROOT / "benchmarks" / "run_k8s_load_test.py"


# ---------------------------------------------------------------------------
# 1. Helm Chart & Values Validation
# ---------------------------------------------------------------------------

def test_helm_chart_metadata():
    """Verify Chart.yaml structure, API version, and required metadata."""
    chart_path = HELM_DIR / "Chart.yaml"
    assert chart_path.exists(), f"Chart.yaml not found at {chart_path}"

    with open(chart_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["apiVersion"] == "v2", "Chart apiVersion must be v2"
    assert data["name"] == "agi-synapse"
    assert "version" in data
    assert data["appVersion"] == "1.0.0"
    assert "maintainers" in data
    assert len(data["keywords"]) > 0


def test_helm_values_structure():
    """Verify values.yaml contains all required production configuration blocks."""
    values_path = HELM_DIR / "values.yaml"
    assert values_path.exists(), f"values.yaml not found at {values_path}"

    with open(values_path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)

    # Global & Workload Identity
    assert values["global"]["workloadIdentity"]["enabled"] is True
    assert "agi-agent-workload-identity" in values["global"]["workloadIdentity"]["gcpServiceAccount"]

    # Serving Microservice
    serving = values["serving"]
    assert serving["replicaCount"] >= 1
    assert "ghcr.io/rajuzet" in serving["image"]["repository"]
    assert serving["rollingUpdate"]["maxSurge"] == 1
    assert serving["rollingUpdate"]["maxUnavailable"] == 0
    assert serving["probes"]["liveness"]["path"] == "/health"
    assert serving["probes"]["readiness"]["path"] == "/health"
    assert serving["resources"]["limits"]["nvidia.com/gpu"] == 1

    # Ingestion Daemon
    daemon = values["daemon"]
    assert daemon["replicaCount"] == 1
    assert "ghcr.io/rajuzet" in daemon["image"]["repository"]
    assert daemon["metricsPort"] == 9090
    assert daemon["persistence"]["enabled"] is True
    assert daemon["persistence"]["size"] == "20Gi"

    # MySQL
    mysql = values["mysql"]
    assert mysql["port"] == 3306
    assert mysql["database"] == "agi_memory"

    # Service & Ingress
    assert values["service"]["type"] == "ClusterIP"
    assert values["service"]["ports"]["serving"]["port"] == 8000
    assert values["service"]["ports"]["daemonMetrics"]["port"] == 9090
    assert values["service"]["ports"]["prometheus"]["port"] == 9091

    # HPA v2
    hpa = values["hpa"]
    assert hpa["enabled"] is True
    assert hpa["minReplicas"] == 1
    assert hpa["maxReplicas"] == 8
    assert hpa["targetCPUUtilizationPercentage"] == 75
    assert hpa["customMetrics"]["enabled"] is True


# ---------------------------------------------------------------------------
# 2. Template Syntax and Existence Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "template_name",
    [
        "_helpers.tpl",
        "serviceaccount.yaml",
        "configmap.yaml",
        "secret.yaml",
        "pvc.yaml",
        "deployment-serving.yaml",
        "deployment-daemon.yaml",
        "service.yaml",
        "ingress.yaml",
        "hpa.yaml",
        "servicemonitor.yaml",
    ],
)
def test_helm_template_files_exist_and_non_empty(template_name: str):
    """Verify that every required template exists and contains valid template content."""
    template_path = TEMPLATES_DIR / template_name
    assert template_path.exists(), f"Missing template: {template_name}"
    content = template_path.read_text(encoding="utf-8").strip()
    assert len(content) > 0, f"Template {template_name} is empty"


def test_deployment_serving_manifest_specifications():
    """Verify key enterprise attributes in deployment-serving.yaml."""
    content = (TEMPLATES_DIR / "deployment-serving.yaml").read_text(encoding="utf-8")
    assert "kind: Deployment" in content
    assert "apiVersion: apps/v1" in content
    assert "RollingUpdate" in content
    assert "maxSurge" in content
    assert "maxUnavailable" in content
    assert "livenessProbe" in content
    assert "readinessProbe" in content
    assert "nvidia.com/gpu" in content or ".Values.serving.resources" in content


def test_deployment_daemon_manifest_specifications():
    """Verify key attributes in deployment-daemon.yaml."""
    content = (TEMPLATES_DIR / "deployment-daemon.yaml").read_text(encoding="utf-8")
    assert "kind: Deployment" in content
    assert "apiVersion: apps/v1" in content
    assert "daemon-storage" in content
    assert "metrics-daemon" in content
    assert "continuous_ingest.py" in content


def test_service_manifest_specifications():
    """Verify multi-port routing in service.yaml."""
    content = (TEMPLATES_DIR / "service.yaml").read_text(encoding="utf-8")
    assert "kind: Service" in content
    assert "8000" in content or "serving.port" in content
    assert "9090" in content or "daemonMetrics.port" in content
    assert "9091" in content or "prometheus.port" in content


def test_hpa_manifest_specifications():
    """Verify autoscaling/v2 compliance and metrics in hpa.yaml."""
    content = (TEMPLATES_DIR / "hpa.yaml").read_text(encoding="utf-8")
    assert "apiVersion: autoscaling/v2" in content
    assert "kind: HorizontalPodAutoscaler" in content
    assert "scaleTargetRef" in content
    assert "agi_serving_ttft_seconds" in content
    assert "agi_serving_active_requests" in content
    assert "scaleUp" in content
    assert "scaleDown" in content


# ---------------------------------------------------------------------------
# 3. GitOps CI/CD Workflow Validation
# ---------------------------------------------------------------------------

def test_github_actions_workflow_structure():
    """Verify .github/workflows/ci_cd_pipeline.yml has all 4 requested stages."""
    assert WORKFLOW_FILE.exists(), f"Workflow file not found at {WORKFLOW_FILE}"

    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        wf = yaml.safe_load(f)

    assert "name" in wf
    assert "jobs" in wf
    jobs = wf["jobs"]

    # Stage 1: Lint & Static Typing
    assert "lint-and-typecheck" in jobs
    # Stage 2: Chaos & Regression Gate
    assert "test-regression-chaos" in jobs
    assert jobs["test-regression-chaos"]["needs"] == "lint-and-typecheck"
    # Stage 3: Container Image Build & Push
    assert "build-and-push-images" in jobs
    assert jobs["build-and-push-images"]["needs"] == "test-regression-chaos"
    # Stage 4: Helm Dry-Run & Lint
    assert "helm-lint-and-dryrun" in jobs
    assert jobs["helm-lint-and-dryrun"]["needs"] == "build-and-push-images"


# ---------------------------------------------------------------------------
# 4. HPA Calculation & Load Test Benchmark Verification
# ---------------------------------------------------------------------------

def test_hpa_scaling_formula():
    """Verify Kubernetes HPA v2 desired replica calculation."""
    from benchmarks.run_k8s_load_test import compute_hpa_desired_replicas

    # With target 15 requests per pod:
    assert compute_hpa_desired_replicas(10, target_concurrency_per_pod=15) == 1
    assert compute_hpa_desired_replicas(15, target_concurrency_per_pod=15) == 1
    assert compute_hpa_desired_replicas(20, target_concurrency_per_pod=15) == 2
    assert compute_hpa_desired_replicas(40, target_concurrency_per_pod=15) == 3
    assert compute_hpa_desired_replicas(60, target_concurrency_per_pod=15) == 4
    assert compute_hpa_desired_replicas(80, target_concurrency_per_pod=15) == 6
    assert compute_hpa_desired_replicas(100, target_concurrency_per_pod=15) == 7
    # Max replica clamp at 8:
    assert compute_hpa_desired_replicas(200, target_concurrency_per_pod=15, max_replicas=8) == 8


def test_k8s_load_test_report_metrics():
    """Verify that the generated benchmark report contains valid metrics."""
    json_path = PROJECT_ROOT / "benchmarks" / "k8s_hpa_scaling_report.json"
    assert json_path.exists(), f"Benchmark report JSON missing: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["overall_summary"]["overall_success_rate_pct"] == 100.0
    assert report["overall_summary"]["average_react_retention_pct"] == 100.0
    assert len(report["tiers"]) == 5
    assert report["tiers"][-1]["concurrency"] == 100
    assert report["tiers"][-1]["hpa_scaling"]["calculated_desired_replicas"] == 7


def test_operations_runbook_completeness():
    """Verify K8S_OPERATIONS.md covers GKE, EKS, Bare-Metal, and Workload Identity."""
    assert OPERATIONS_DOC.exists(), f"Operations handbook missing at {OPERATIONS_DOC}"
    content = OPERATIONS_DOC.read_text(encoding="utf-8")
    assert "Google Kubernetes Engine (GKE)" in content
    assert "AWS Elastic Kubernetes Service (EKS)" in content
    assert "NVIDIA GPU Operator" in content
    assert "Workload Identity" in content
    assert "helm upgrade --install" in content
    assert "autoscaling/v2" in content
    assert "Rollback" in content
