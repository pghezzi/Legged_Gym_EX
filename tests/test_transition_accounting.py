"""Frozen-paper reporting only: no training, search, or simulation."""
import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    evaluate_transition_accounting, _mean_transition_delay,
)
from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import evaluate_sequential_predictions
from legged_gym.scripts.depth_data_pipeline import evaluate_paper_offline_experiments_1_2 as paper


@pytest.mark.parametrize("prediction,matched,delay", [
    ([0, 1, 1, 2], 2, 0.),  # immediate matching at both boundaries
    ([0, 0, 1, 2], 2, .5),  # one frame delay, then zero
    ([0, 0, 0, 1], 0, None),  # class 1 arrives only in the next segment
])
def test_segment_bounds_and_misses(prediction, matched, delay):
    result = evaluate_transition_accounting([0, 1, 1, 2], prediction)
    assert result["total_transitions_v2"] == 2
    assert result["matched_transitions_v2"] == matched
    assert result["missed_transitions_v2"] == 2 - matched
    assert result["transition_miss_rate_v2"] == (2 - matched) / 2
    records = result["transition_records_v2"]
    assert len(records) == 2
    assert records[0]["segment_end_frame_exclusive"] == 3
    if delay is None:
        assert math.isnan(result["mean_matched_transition_delay_frames_v2"])
        assert all(r["missed"] and r["matched_frame"] is None for r in records)
    else:
        assert result["mean_matched_transition_delay_frames_v2"] == delay


def test_sequences_do_not_match_across_boundaries_or_reused_ids():
    result = evaluate_transition_accounting([0, 1, 1, 0, 1], [0, 0, 1, 0, 1], [7, 7, 8, 7, 7])
    assert result["total_transitions_v2"] == 2
    records = result["transition_records_v2"]
    assert records[0]["missed"] and records[0]["segment_end_frame_exclusive"] == 2
    assert records[1]["matched"] and records[1]["sequence_start_frame"] == 3
    assert records[1]["sequence_transition_frame"] == 1


@pytest.mark.parametrize("truth,ids", [([], []), ([0], [0]), ([0, 0], [0, 0]), ([0, 1], [0, 1])])
def test_zero_denominators(truth, ids):
    result = evaluate_transition_accounting(truth, truth, ids)
    assert result["total_transitions_v2"] == 0
    assert math.isnan(result["transition_miss_rate_v2"])
    assert math.isnan(result["mean_matched_transition_delay_frames_v2"])
    assert result["transition_records_v2"] == []


def test_length_validation():
    with pytest.raises(ValueError):
        evaluate_transition_accounting([0, 1], [0])
    with pytest.raises(ValueError):
        evaluate_transition_accounting([0, 1], [0, 1], [0])


def test_search_scoring_and_false_transitions_unchanged():
    truth, predictions, ids = [0, 1, 1, 2], [0, 0, 0, 1], [0] * 4
    assert _mean_transition_delay(truth, predictions, ids) == 2.  # legacy explicitly retained
    legacy = evaluate_sequential_predictions(truth, predictions, [0, 1, 2], ids)
    corrected = evaluate_sequential_predictions(truth, predictions, [0, 1, 2], ids,
                                                 transition_accounting_v2=True)
    assert "transition_records_v2" not in legacy
    for key in ("selection_score", "score_v2", "legacy_selection_score", "mean_transition_delay", "false_transition_rate"):
        assert corrected[key] == legacy[key]
    assert corrected["missed_transitions_v2"] == 2


def test_all_six_paper_configs_and_seed_aggregation_exports(tmp_path):
    rows, records = [], []
    logits = torch.tensor([[[8., 0.], [8., 0.], [0., 8.], [0., 8.]]])
    for architecture in ("feature_nn", "raw_depth_nn"):
        for seed in paper.MODEL_SEEDS:
            results = paper._sequential_metrics(SimpleNamespace(class_ids=[0, 1]), logits,
                                                 [0, 1, 1, 1], [0] * 4)
            assert set(results) == {"instantaneous", "ema", "bayes"}
            for method, metrics in results.items():
                assert metrics["total_transitions_v2"] == 1
                assert len(metrics["transition_records_v2"]) == 1
                row = dict(architecture=architecture, temporal_method=method, seed=seed, **metrics)
                rows.append(row)
                records.extend(metrics["transition_records_v2"])
    summaries = paper._aggregate(rows, ("architecture", "temporal_method"), paper.TRANSITION_V2_METRICS, [0, 1])
    assert len(summaries) == 6
    for summary in summaries:
        assert summary["num_seeds"] == 3
        for key in paper.TRANSITION_V2_METRICS:
            values = summary[key + "_raw"]
            assert summary[key + "_mean"] == pytest.approx(np.mean(values))
            assert summary[key + "_std"] == pytest.approx(np.std(values, ddof=1))
    paper._write_rows(tmp_path / "summary.csv", summaries)
    paper.save_results(tmp_path / "summary.json", summaries)
    assert len(list(csv.DictReader((tmp_path / "summary.csv").open()))) == 6
    assert len(json.loads((tmp_path / "summary.json").read_text())) == 6
    paper.save_results(tmp_path / "records.json", records)
    assert len(json.loads((tmp_path / "records.json").read_text())) == 18


def test_aggregate_unavailable_delays():
    row = dict(seed=0, per_class_recall={"0": 1.}, confusion_matrix=[[1]],
               **{key: float("nan") for key in paper.TRANSITION_V2_METRICS})
    summary = paper._aggregate([dict(row, seed=seed) for seed in (0, 1, 2)], (), paper.TRANSITION_V2_METRICS, [0])[0]
    assert math.isnan(summary["mean_matched_transition_delay_frames_v2_mean"])
    assert math.isnan(summary["mean_matched_transition_delay_frames_v2_std"])


def test_empty_transition_csv_has_schema(tmp_path):
    path = tmp_path / "transitions.csv"
    paper._write_rows(path, [], paper.TRANSITION_RECORD_FIELDS)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(paper.TRANSITION_RECORD_FIELDS)
        assert list(reader) == []
