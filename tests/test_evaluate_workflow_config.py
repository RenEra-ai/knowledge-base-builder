"""Config regression for the Evaluate KB Candidate workflow.

Release-grade candidate evaluation must run on the pinned CI runner (the runbook
forbids treating a locally-run evaluation as release-grade), fetch the corpus
from a prior build run, and evaluate it through the PINNED Track A KbService
checked out from the public boomi-mcp-server repo. Parsed as plain text.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "evaluate-candidate.yml"


def _text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_workflow_exists():
    assert _WORKFLOW.is_file()


def test_declares_required_inputs():
    text = _text()
    inputs = text.split("inputs:", 1)[1].split("permissions:", 1)[0]
    assert "build_run_id:" in inputs
    assert "candidate:" in inputs
    assert "mcp_ref:" in inputs


def test_fetches_corpus_from_the_build_run():
    text = _text()
    assert "name: boomi-knowledge-db" in text
    assert "run-id: ${{ inputs.build_run_id }}" in text


def test_checks_out_track_a_kbservice():
    text = _text()
    assert "repository: RenEra-ai/boomi-mcp-server" in text


def test_runs_eval_harness_against_the_pinned_service():
    text = _text()
    assert "eval_harness.py run" in text
    assert "--server-path boomi-mcp-server" in text


def test_has_actions_read_for_cross_run_download():
    text = _text()
    perms = text.split("permissions:", 1)[1].split("concurrency:", 1)[0]
    assert "actions: read" in perms


def test_uploads_a_named_eval_bundle():
    text = _text()
    assert "name: eval-${{ inputs.candidate }}" in text
