from pathlib import Path

import numpy as np
import pytest

from warp_as_history.event_aligned import (
    drop_reference_render_frame,
    event_aligned_indices,
    run_scripted_event_timeline,
    scripted_event_segments,
    validate_generated_cleanup_path,
)
from warp_as_history.training.fixed_teacher import fixed_identity_from_row, validate_fixed_identity


def test_event_aligned_indices_and_reference_render_drop():
    indices = event_aligned_indices(12, 100)
    assert indices["reference_frame_index"] == 11
    assert indices["target_indices"] == list(range(12, 45))
    assert indices["render_pose_indices"] == list(range(11, 45))
    assert indices["event_local_frame"] == 0
    rendered = np.arange(34)
    assert drop_reference_render_frame(rendered).tolist() == list(range(1, 34))


def test_scripted_timeline_truncates_and_restarts_from_last_committed_frame():
    calls = []

    def generate(reference, segment, window_frames):
        calls.append((reference, segment["event_frame"] if "event_frame" in segment else segment["target_indices"][0]))
        return [f"event{segment['target_indices'][0]}-frame{index}" for index in range(window_frames)]

    events = [
        {"event_frame": 5, "action_type": "place"},
        {"event_frame": 8, "action_type": "mine_complete"},
        {"event_frame": 8, "action_type": "place"},
    ]
    result = run_scripted_event_timeline("initial", events, 50, generate)
    assert len(result["segments"]) == 2
    assert result["segments"][0]["committed_frame_count"] == 3
    assert calls[0][0] == "initial"
    assert calls[1][0] == "event5-frame2"
    assert result["oracle_reference"] is False


def test_scripted_segments_merge_same_frame_events():
    segments = scripted_event_segments(
        [{"event_frame": 6, "action_type": "place"}, {"event_frame": 6, "action_type": "mine_active"}],
        60,
    )
    assert len(segments) == 1
    assert len(segments[0]["events"]) == 2


def test_fixed_identity_includes_event_reference_indices():
    row = {
        "event_id": "evt",
        "action_type": "place",
        "history_type": "first",
        "target_indices": "12,13",
        "reference_frame_index": "11",
        "history_indices": "",
        "geometry_keyframe_frames": "11,",
        "render_pose_indices": "11,12,13",
        "candidate_config_hash": "cfg",
    }
    identity = fixed_identity_from_row(row)
    assert identity["reference_frame_index"] == 11
    validate_fixed_identity(row, identity, "cfg")
    broken = dict(identity, reference_frame_index=10)
    with pytest.raises(RuntimeError, match="reference_frame_index"):
        validate_fixed_identity(row, broken, "cfg")


def test_cleanup_target_requires_exact_allowlist(tmp_path):
    allowed = tmp_path / "data" / "teacher_preparation"
    assert validate_generated_cleanup_path(allowed, [allowed]) == allowed.resolve()
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_generated_cleanup_path(allowed.parent, [allowed])
    with pytest.raises(ValueError, match="empty"):
        validate_generated_cleanup_path("", [allowed])


def test_transformers_route_reference_context_without_target_conditioning():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "helios/modules/transformer_helios.py",
        "helios/diffusers_version/transformer_helios_diffusers.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert 'interaction_conditioning.get("reference_latents")' in source
        assert "reference_tokens=reference_tokens" in source
        assert 'interaction_conditioning.get("target_latents")' not in source
