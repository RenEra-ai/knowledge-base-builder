"""Config regression for the Build Knowledge Base GitHub Actions workflow.

The staged kb-25 rollout (docs/kb25-release-runbook.md) needs the build workflow
to do three things the original design could not:

1. Freeze the source snapshot ONCE and reuse it across candidate builds, so
   C0/B5/B56/B567 are built from byte-identical sources (the gates compare
   candidates against each other — a per-dispatch re-scrape makes them
   incomparable).
2. Build-and-evaluate a candidate WITHOUT publishing a release (the original
   workflow published on every manual dispatch).
3. Publish under a chosen tag (e.g. kb-25) instead of kb-<run_number>.

Parsed as plain text (not YAML): PyYAML is not a project dependency, and these
assertions only need to see that the inputs, gates, and artifact wiring exist.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "build-knowledge-base.yml"

_SNAPSHOT_ARTIFACT = "kb25-source-snapshot"


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _dispatch_inputs_block(text: str) -> str:
    """The workflow_dispatch inputs block (up to the top-level permissions:)."""
    assert "workflow_dispatch:" in text
    after = text.split("workflow_dispatch:", 1)[1]
    return after.split("\npermissions:", 1)[0]


# --- new dispatch inputs ------------------------------------------------------

def test_declares_mode_input_with_freeze_and_build():
    block = _dispatch_inputs_block(_workflow_text())
    assert "mode:" in block
    assert "freeze" in block and "build" in block


def test_declares_snapshot_run_id_input():
    block = _dispatch_inputs_block(_workflow_text())
    assert "snapshot_run_id:" in block


def test_declares_publish_boolean_input_defaulting_off():
    block = _dispatch_inputs_block(_workflow_text())
    assert "publish:" in block
    # the publish input is a boolean and defaults off so an eval-only dispatch
    # does not cut a release.
    publish_seg = block.split("publish:", 1)[1].split("release_tag:", 1)[0]
    assert "type: boolean" in publish_seg
    assert "default: false" in publish_seg


def test_declares_release_tag_input():
    block = _dispatch_inputs_block(_workflow_text())
    assert "release_tag:" in block


# --- snapshot freeze / reuse wiring -------------------------------------------

def test_freeze_mode_uploads_snapshot_artifact():
    text = _workflow_text()
    assert _SNAPSHOT_ARTIFACT in text
    # the snapshot upload is gated on freeze mode.
    assert "inputs.mode == 'freeze'" in text


def test_build_can_reuse_a_prior_snapshot_by_run_id():
    text = _workflow_text()
    # a download-artifact step pulls the snapshot from the referenced run.
    assert "run-id: ${{ inputs.snapshot_run_id }}" in text
    # and the scrape is skipped when reusing a snapshot.
    assert "inputs.snapshot_run_id == ''" in text


def test_cross_run_artifact_download_has_actions_read_permission():
    text = _workflow_text()
    perms = text.split("permissions:", 1)[1].split("concurrency:", 1)[0]
    assert "actions: read" in perms
    assert "contents: write" in perms


def test_legacy_chunk_can_consume_the_shared_snapshot():
    text = _workflow_text()
    # so C0 is built from the same frozen bytes as the S5 candidates: the
    # legacy chunk step must pass --snapshot to chunk_docs.py.
    legacy = text.split("Chunk documents (legacy)", 1)[1].split("Chunk documents (S5)", 1)[0]
    assert "chunk_docs.py" in legacy
    assert "--snapshot source_snapshot/" in legacy


# --- publish gating + tag control ---------------------------------------------

def test_release_is_gated_on_the_publish_input():
    text = _workflow_text()
    # the create-release step must require inputs.publish, not merely
    # github.event_name == 'workflow_dispatch'.
    assert "action-gh-release" in text
    assert "inputs.publish" in text
    # tag comes from release_tag (falling back to the run number).
    tail = text.rsplit("Create GitHub release", 1)[1]
    assert "inputs.release_tag" in tail
    assert "github.run_number" in tail


def test_snapshot_freeze_step_still_present():
    text = _workflow_text()
    assert "snapshot.py freeze" in text
    assert "snapshot.py verify" in text
