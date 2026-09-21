"""Automated DevSecOps Pre-Push Verification Gate for Project Synapse.

Validates repository security and operational readiness prior to git push:
1. Secret & Sensitive File Leak Audit:
   - Ensures .env, GCP private keys (*.json), and binary memory indices are excluded.
   - Verifies template examples (.env.example, secrets/service-account.json.example) are intact.
2. Cloud-Native Kubernetes & Helm Manifest Verification:
   - Executes tests/test_helm_and_k8s_manifests.py
3. Full End-to-End Regression & Resilience Test Gate:
   - Executes the complete test suite (tests/)
4. Emits Git staging status and ready-to-run Git commit & push commands.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Color terminal codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_step(step_name: str, command: list[str]) -> bool:
    """Run a sub-process step and return True if successful."""
    print(f"\n{BOLD}{CYAN}>>> [STEP] {step_name}...{RESET}")
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        elapsed = time.perf_counter() - start_time
        if proc.returncode == 0:
            print(f"{GREEN}[PASS] {step_name} ({elapsed:.2f}s){RESET}")
            if proc.stdout:
                # Print tail of output
                lines = proc.stdout.strip().split("\n")
                tail = "\n  ".join(lines[-4:])
                print(f"  {tail}")
            return True
        else:
            print(f"{RED}[FAIL] {step_name} (Exit code {proc.returncode}){RESET}")
            if proc.stdout:
                print(proc.stdout[-800:])
            if proc.stderr:
                print(proc.stderr[-800:])
            return False
    except Exception as exc:
        print(f"{RED}[ERROR] Execution failed for {step_name}: {exc}{RESET}")
        return False


def audit_secrets_and_staging() -> bool:
    """Audit git untracked and eligible files to ensure no sensitive secrets leak."""
    print(f"\n{BOLD}{CYAN}>>> [STEP] DevSecOps Secret Sanitization & Leak Audit...{RESET}")

    # Check git check-ignore for critical patterns
    sensitive_targets = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "secrets" / "service-account.json",
        PROJECT_ROOT / "data_memory" / "index.faiss",
        PROJECT_ROOT / "data_staging" / "test_harvest.jsonl",
    ]

    all_ignored = True
    for target in sensitive_targets:
        rel_path = target.relative_to(PROJECT_ROOT)
        proc = subprocess.run(
            ["git", "check-ignore", str(rel_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"  {GREEN}[OK] Pattern correctly ignored:{RESET} {rel_path}")
        else:
            print(f"  {RED}[X] Vulnerability: Sensitive path not ignored:{RESET} {rel_path}")
            all_ignored = False

    # Check templates exist
    templates = [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "secrets" / "service-account.json.example",
    ]
    for tmpl in templates:
        if tmpl.exists():
            print(f"  {GREEN}[OK] Template present:{RESET} {tmpl.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  {RED}[X] Missing required template:{RESET} {tmpl.relative_to(PROJECT_ROOT)}")
            all_ignored = False

    return all_ignored


def main() -> int:
    print(f"\n{BOLD}{'=' * 75}{RESET}")
    print(f"{BOLD} Project Synapse: Automated Pre-Push CI/CD Verification Gate{RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")

    python_bin = sys.executable

    # 1. Secret & File Leak Audit
    if not audit_secrets_and_staging():
        print(f"\n{RED}[FATAL] Secret audit failed. Aborting push readiness.{RESET}")
        return 1

    # 2. Helm & Kubernetes Manifest Tests
    manifest_ok = run_step(
        "Helm Packaging & Kubernetes Manifest Tests",
        [python_bin, "-m", "pytest", "tests/test_helm_and_k8s_manifests.py", "-v"],
    )
    if not manifest_ok:
        print(f"\n{RED}[FATAL] Helm & K8s manifest validation failed.{RESET}")
        return 1

    # 3. Full 132-Test Suite
    full_tests_ok = run_step(
        "Full Regression & Chaos Engineering Test Suite (132+ Tests)",
        [python_bin, "-m", "pytest", "tests/", "-q"],
    )
    if not full_tests_ok:
        print(f"\n{RED}[FATAL] Full test suite encountered failures.{RESET}")
        return 1

    # 4. Display Git Staging Status & Push Instructions
    proc = subprocess.run(
        ["git", "status", "-s"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    status_output = proc.stdout.strip()

    print(f"\n{BOLD}{GREEN}==========================================================================={RESET}")
    print(f"{BOLD}{GREEN} PRE-PUSH VERIFICATION GATE PASSED (100% Tests & Security Clear)            {RESET}")
    print(f"{BOLD}{GREEN}==========================================================================={RESET}")
    print(f"\n{BOLD}Sanitized Staging Status:{RESET}")
    if status_output:
        for line in status_output.split("\n")[:15]:
            print(f"  {line}")
        if len(status_output.split("\n")) > 15:
            print(f"  ... and {len(status_output.split('\n')) - 15} more files")
    else:
        print("  Clean working directory.")

    print(f"\n{BOLD}Recommended Push Commands:{RESET}")
    print(f"  {CYAN}git add .{RESET}")
    print(f'  {CYAN}git commit -m "feat(gitops): enterprise helm chart, GHCR release pipeline and HPA scaling"{RESET}')
    print(f"  {CYAN}git push -u origin main{RESET}")
    print(f"  {CYAN}git push -u project-synapse main{RESET}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
