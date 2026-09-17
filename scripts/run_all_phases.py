#!/usr/bin/env python3
"""Unified End-to-End Pipeline Runner — Phases 1 to 7.

Executes the complete evidence-first customer support pipeline from raw CSV
to evaluation tables and final report:
  Phase 1: Ingestion, Graph Reconstruction & Anomaly Auditing
  Phase 2: Golden Dataset Manifest Validation & Three-Way Splitting
  Phase 3: Intent, Escalation & Template Baselines (Train -> Dev/Test)
  Phase 4: Historical Evidence Retrieval Corpus & Dev Evaluation
  Phase 5: Deterministic Policy Engine Evaluation (Safety vs Cost Trade-off)
  Phase 6: Evidence-Grounded Reply Generation with PII & Sentiment Guardrails
  Phase 7: Multi-Tier Reply Evaluation, Blinded Judging & Annotation Export

Usage:
  # Fast offline run (100% free, zero network calls, uses cached outputs):
  python scripts/run_all_phases.py --offline --skip-extraction

  # Full run from raw CSV with live model:
  python scripts/run_all_phases.py --provider gemini
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(step_name: str, cmd: list, cwd: Path = REPO_ROOT):
    """Run a pipeline subprocess command with timing and error handling."""
    print("\n" + "=" * 75)
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] Step '{step_name}' completed in {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run all Tesco Customer Support Assistant phases end-to-end.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "gemini", "openai", "ollama"],
        help="LLM provider for Phases 6 and 7 ('mock' runs 100%% offline).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g., gemini-3.5-flash, gpt-4o-mini, llama3.1:8b).",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip Phase 1 extraction if outputs/conversations.jsonl already exists.",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        default=1,
        choices=range(1, 8),
        help="Starting phase number (1-7).",
    )
    parser.add_argument(
        "--to-phase",
        type=int,
        default=7,
        choices=range(1, 8),
        help="Ending phase number (1-7).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enforce offline execution using mock provider and existing artifacts.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the complete automated pytest suite before executing pipeline.",
    )
    args = parser.parse_args()

    provider = "mock" if args.offline else args.provider
    python_exe = sys.executable

    total_start = time.time()
    print("=" * 75)
    print("TESCO CUSTOMER SUPPORT AI ASSISTANT — UNIFIED PIPELINE RUNNER")
    print(f"Phases: {args.from_phase} to {args.to_phase} | Provider: {provider}")
    print("=" * 75)

    # Pre-flight tests if requested
    if args.run_tests:
        run_step("Pre-Flight Hermetic Tests", [python_exe, "-m", "pytest", "tests/", "-v"])

    # Phase 1: Ingestion & Thread Reconstruction
    if args.from_phase <= 1 <= args.to_phase:
        conv_path = REPO_ROOT / "outputs" / "conversations.jsonl"
        if args.skip_extraction and conv_path.exists():
            print(f"\n[INFO] Skipping Phase 1 extraction: {conv_path} already exists.")
        else:
            run_step(
                "Phase 1: Ingestion & Thread Reconstruction",
                [python_exe, "scripts/run_pipeline.py", "--input-path", "tesco_tweets.csv"],
            )

    # Phase 2: Golden Set Validation & Freeze
    if args.from_phase <= 2 <= args.to_phase:
        run_step(
            "Phase 2: Golden Set Validation & Manifest Freeze",
            [python_exe, "scripts/freeze_golden_set.py"],
        )

    # Phase 3: Baseline Modeling (Intent, Escalation, Reply Templates)
    if args.from_phase <= 3 <= args.to_phase:
        run_step(
            "Phase 3: Intent, Escalation & Template Baselines",
            [python_exe, "scripts/run_baselines.py"],
        )

    # Phase 4: Retrieval Corpus & Dev Evaluation
    if args.from_phase <= 4 <= args.to_phase:
        run_step(
            "Phase 4A: Build Historical Evidence Corpus",
            [python_exe, "scripts/build_retrieval_corpus.py"],
        )
        run_step(
            "Phase 4B: Evaluate Dev Historical Retrieval",
            [python_exe, "scripts/evaluate_dev_retrieval.py"],
        )

    # Phase 5: Deterministic Policy Engine Evaluation
    if args.from_phase <= 5 <= args.to_phase:
        run_step(
            "Phase 5: Evaluate Deterministic Policy Layer",
            [python_exe, "scripts/evaluate_policy.py"],
        )

    # Phase 6: Evidence-Grounded Reply Generation
    if args.from_phase <= 6 <= args.to_phase:
        gen_cmd = [python_exe, "scripts/generate_replies.py", "--provider", provider]
        if args.model:
            gen_cmd.extend(["--model", args.model])
        run_step(
            f"Phase 6: Evidence-Grounded Reply Generation ({provider})",
            gen_cmd,
        )

    # Phase 7: Evaluation & Blinded Judging
    if args.from_phase <= 7 <= args.to_phase:
        eval_cmd = [
            python_exe,
            "scripts/run_phase7_evaluation.py",
            "--stage", "dev",
            "--provider", provider,
        ]
        if args.model:
            eval_cmd.extend(["--model", args.model])
        run_step(
            f"Phase 7: Reply Evaluation & Blinded Judging ({provider})",
            eval_cmd,
        )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"PIPELINE RUN COMPLETE in {total_elapsed:.2f}s")
    print("All requested phases completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
