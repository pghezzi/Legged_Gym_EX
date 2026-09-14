"""Small CPU checks for stage reconciliation and provenance-based charging."""
import importlib.util
import ast
import json
import math
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.utils.training_cost import STAGE_FIELDS, stage_accounting, provenance_map


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "cost_summary", ROOT / "legged_gym/scripts/depth_data_pipeline/summarize_training_costs.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def record(kind, path, seconds, **extra):
    result = dict(run_type=kind, dataset_path=str(path), data_env_steps=10,
                  gpu_count=1, peak_gpu_memory_mb=1.,
                  **stage_accounting(seconds, 1, setup_s=seconds / 4,
                                     preprocessing_s=seconds / 4), **extra)
    result["metric_status"] = provenance_map(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_disjoint_stages_and_allocated_gpu_hours():
    result = stage_accounting(21., 2, setup_s=1., data_generation_s=2.,
                              preprocessing_s=3., optimization_s=4., artifact_serialization_s=5.)
    assert sum(result[k] for k in STAGE_FIELDS) == pytest.approx(21.)
    assert result["overhead_s"] == 6.
    assert result["gpu_hours"] == pytest.approx(21. * 2 / 3600.)
    assert result["gpu_active_time_s"] is None
    assert stage_accounting(21., 0)["gpu_hours"] == 0.
    with pytest.raises(ValueError, match="overlap"):
        stage_accounting(1., 1, setup_s=2.)


def test_preparation_deduplicates_sources_and_excludes_ordered_test(tmp_path):
    compiled, raw, calibration = tmp_path / "compiled", tmp_path / "raw.pt", tmp_path / "calibration.pt"
    collection = record("classifier_data_collection", raw, 10.)
    cal_collection = record("classifier_data_collection", calibration, 5.)
    compilation = record("dataset_compilation", compiled, 3., source_files=[str(raw), str(raw), str(calibration)])
    write(Path(str(raw) + ".training_cost.json"), collection)
    write(Path(str(calibration) + ".training_cost.json"), cal_collection)
    write(compiled / "training_cost.json", compilation)
    write(compiled / "split_manifest.json", {"source_files": [str(raw)], "calibration_source_file": str(calibration)})
    write(tmp_path / "ordered" / "training_cost.json", record("dataset_compilation", tmp_path / "ordered", 999.))
    write(tmp_path / "manifest.json", {"structural_dataset": str(compiled), "ordered_dataset": str(tmp_path / "ordered")})
    preparation = summary._preparation_records(tmp_path, [tmp_path])
    assert len(preparation) == 3
    classifier = record("classifier_training", tmp_path / "model.pt", 2., method="Feature Router")
    for method in ("Feature Router", "Raw-Depth Router"):
        classifier["method"] = method
        # Repeated discovery must not charge the same artifact twice.
        composed = summary._compose_router_run(classifier, preparation + preparation)
        assert composed["total_wallclock_s"] == pytest.approx(20.)
        assert composed["compilation_s"] == 3.
        assert sum(composed[k] for k in STAGE_FIELDS) == pytest.approx(20.)
        assert composed["gpu_hours"] == pytest.approx(20. / 3600.)
        assert composed["data_env_steps"] == 20
    Path(str(calibration) + ".training_cost.json").unlink()
    preparation = summary._preparation_records(tmp_path, [])
    missing = summary._compose_router_run(classifier, preparation)
    assert missing["total_wallclock_s"] is None
    assert missing["data_env_steps"] is None


def test_legacy_timings_and_partial_seed_summaries_stay_unavailable(tmp_path):
    old = record("classifier_data_collection", tmp_path / "raw.pt", 5.)
    old.pop("timing_schema_version")
    old["preprocessing_s"] = 0.  # Legacy hard-coded value, not a measurement.
    classifier = record("classifier_training", tmp_path / "model.pt", 2., method="Feature Router")
    compiled = record("dataset_compilation", tmp_path, 3.)
    composed = summary._compose_router_run(classifier, [old, compiled])
    assert composed["total_wallclock_s"] is None
    assert composed["preprocessing_s"] is None
    rows = summary._aggregate([classifier, composed])
    assert rows[0]["total_wallclock_s_mean"] is None


def test_relocated_provenance_and_shared_aliases_are_read_only(tmp_path):
    offline, compiled, sidecars = tmp_path / "offline", tmp_path / "compiled", tmp_path / "sidecars"
    raw = tmp_path / "raw.pt"
    raw_cost = sidecars / "raw.pt.training_cost.json"
    compile_cost = compiled / "training_cost.json"
    write(raw_cost, record("classifier_data_collection", "/archive/raw.pt", 10.))
    write(compile_cost, record("dataset_compilation", "/archive/compiled", 3.,
                              source_files=["/archive/raw.pt", "/alias/raw.pt"]))
    write(offline / "manifest.json", {"structural_dataset": "/archive/compiled",
                                      "structural_split_manifest": "/archive/compiled/split_manifest.json"})
    write(compiled / "split_manifest.json", {"calibration_source_file": "/alias/raw.pt"})
    resolver = summary.PathResolver([("/archive", tmp_path), ("/alias/raw.pt", raw)])
    originals = {p: p.read_bytes() for p in (raw_cost, compile_cost, offline / "manifest.json")}
    preparation = summary._preparation_records(offline, [raw_cost, compile_cost], resolver=resolver)
    assert len(preparation) == 2  # Shared training/calibration source charged once.
    classifier = record("classifier_training", tmp_path / "model.pt", 2., method="Feature Router")
    result = summary._compose_router_run(classifier, preparation)
    assert result["total_wallclock_s"] == pytest.approx(15.)
    assert result["data_env_steps"] == 10
    assert result["collection_run_ids"] == [str(raw)]
    assert all(path.read_bytes() == before for path, before in originals.items())
    raw_cost.unlink()
    missing = summary._compose_router_run(classifier, summary._preparation_records(offline, [compile_cost], resolver=resolver))
    assert missing["total_wallclock_s"] is None and missing["data_env_steps"] is None


def test_relocated_relative_sidecar_and_prefix_boundaries(tmp_path):
    moved = tmp_path / "isolated" / "capture.training_cost.json"
    resolver = summary.PathResolver([("/host/data", tmp_path / "data"),
                                     ("/host/data/capture.training_cost.json", moved)])
    assert resolver.resolve("/host/data-other/a") == Path("/host/data-other/a")
    assert resolver.resolve("/host/data/capture.training_cost.json") == moved
    assert resolver.resolve("capture.pt", resolver.record_dir(moved)) == tmp_path / "data/capture.pt"
    with pytest.raises(ValueError, match="absolute"):
        summary.PathResolver([("relative", tmp_path)])


def test_deployment_files_exclude_training_checkpoints_and_deduplicate(tmp_path):
    offline = tmp_path / "offline"
    records = [{"method": method, "seed": 0, "artifact_size_mb": 999.} for method in summary.METHODS]
    paths = [offline / "manifest.json", offline / "artifacts/feature_nn/extractor.pt",
             offline / "artifacts/feature_nn/standardizer.pt"]
    for architecture in ("feature_nn", "raw_depth_nn"):
        paths += [offline / f"artifacts/{architecture}/seed_0/{name}" for name in ("classifier.pt", "nn_model_args.pt")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)
    export = tmp_path / "student.pt"
    export.write_bytes(b"x" * 2048)
    alias = tmp_path / "student_alias.pt"
    os.link(export, alias)
    summary._deployment_sizes(records, offline, [("distilled:0", export), ("distilled:0", alias)])
    assert [r["deployment_size_mb"] for r in records] == [5/1024, 3/1024, 2/1024]
    assert all(r["artifact_size_mb"] == 999. for r in records)
    export.unlink()
    summary._deployment_sizes(records, offline, [("distilled:0", export)])
    assert records[-1]["deployment_size_mb"] is None
    assert records[-1]["metric_status"]["deployment_size_mb"] == "unavailable"


def test_cost_aggregation_only_never_loads_models_or_evaluates(tmp_path, monkeypatch):
    offline, compiled = tmp_path / "offline", tmp_path / "compiled"
    raw = tmp_path / "raw.pt"
    write(Path(str(raw) + ".training_cost.json"), record("classifier_data_collection", raw, 10.))
    write(compiled / "training_cost.json", record("dataset_compilation", compiled, 3., source_files=[str(raw)]))
    training = []
    for architecture, method in zip(("feature_nn", "raw_depth_nn"), summary.METHODS):
        for seed in range(3):
            training.append(dict(architecture=architecture, seed=seed))
            write(offline / f"artifacts/{architecture}/seed_{seed}/training_cost.json",
                  record("classifier_training", offline, 2., method=method, architecture=architecture, seed=seed))
    write(offline / "manifest.json", dict(structural_dataset=str(compiled), training_runs=training,
                                        evaluation_runtime_seconds=123456789))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Model inference/loading must not be needed"))
    monkeypatch.setattr(summary, "_plots", lambda *args: [])
    output = tmp_path / "costs"
    summary.main(["--paper-offline-dir", str(offline), "--output", str(output), "--no-auto-distillation"])
    audit = json.loads((output / "training_cost_manifest.json").read_text())
    assert len(audit["per_run_records"]) == 6
    assert all(r["total_wallclock_s"] == 15. for r in audit["per_run_records"])
    assert all(r["deployment_size_mb"] is None for r in audit["per_run_records"])
    assert (output / "training_cost_comparison.tex").is_file()


def test_tables_and_stacked_figures_include_all_stages(tmp_path):
    records = []
    for method in summary.METHODS:
        item = record("integration", tmp_path / method, 21., method=method)
        item.update(compilation_s=3., additional_locomotion_policy_env_steps=0,
                    total_post_specialist_env_steps=10)
        records.append(item)
    rows = summary._aggregate(records)
    summary._write_table(tmp_path, rows)
    figures = summary._plots(tmp_path, rows, None)
    table = (tmp_path / "training_cost_comparison.csv").read_text()
    for field in (*STAGE_FIELDS, "compilation_s"):
        assert field in table
    assert any("data_vs_optimization" in path for path in figures)
    for row in rows:
        assert sum(row[f"{key}_mean"] for key in STAGE_FIELDS) == pytest.approx(row["total_wallclock_s_mean"])


def test_compiler_writes_reconciling_sidecar(tmp_path, monkeypatch):
    import legged_gym
    raw = tmp_path / "raw.pt"
    frames, envs = 3, 10
    torch.save({"depth_images": torch.zeros(frames, envs, 2, 2),
                "base_rpy": torch.zeros(frames, envs, 3),
                "base_ang_vel": torch.zeros(frames, envs, 3),
                "terrain_name": [["rough"] * envs for _ in range(frames)]}, raw)
    monkeypatch.setattr(legged_gym, "LEGGED_GYM_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["compile_depth_data", "--files", str(raw), "--frac", "1", "--allow-legacy-provenance"])
    runpy.run_path(str(ROOT / "legged_gym/scripts/depth_data_pipeline/compile_depth_data.py"), run_name="__main__")
    sidecar = next(tmp_path.rglob("training_cost.json"))
    result = json.loads(sidecar.read_text())
    assert result["source_files"] == [str(raw)]
    assert result["gpu_hours"] == 0.
    assert sum(result[k] for k in STAGE_FIELDS) == pytest.approx(result["total_wallclock_s"])
    assert result["data_loading_s"] == result["setup_s"]
    assert result["preprocessing_s"] == pytest.approx(sum(result[k] for k in ("sampling_s", "splitting_s", "merging_s")))
    assert all(result[k] > 0 for k in ("data_loading_s", "sampling_s", "splitting_s", "merging_s", "artifact_serialization_s"))


def test_classifier_accounts_loading_but_excludes_sequence_preparation(tmp_path, monkeypatch):
    """Run the real accounting path with tiny fake fit/extraction and a virtual clock."""
    from rsl_rl.utils import training_cost as cost
    path = ROOT / "legged_gym/scripts/depth_data_pipeline/evaluate_paper_offline_experiments_1_2.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    outer = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                 and n.target.id == "architecture")
    seeds = next(n for n in outer.body if isinstance(n, ast.For))
    # Stop at the cost record; the excluded preparation before fitting still runs.
    end = next(i for i, n in enumerate(seeds.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id == "training_records")
    seeds.body = seeds.body[:end + 1]
    main.body = main.body[:main.body.index(outer) + 1]
    main.body.append(ast.Return(value=ast.Name(id="training_records", ctx=ast.Load())))
    clock = [0.]
    monkeypatch.setattr(cost.time, "perf_counter", lambda: clock[0])
    def tick(value=1.):
        clock[0] += value
    data = dict(depth_images=torch.zeros(4, 4, 4), orientation_rpy=torch.zeros(4, 3),
                angular_velocity=torch.zeros(4, 3), labels=[0] * 4)
    def load(*args, **kwargs):
        tick()
        return data
    monkeypatch.setattr(torch, "load", load)
    real_save = torch.save
    def save(*args, **kwargs):
        tick()
        return real_save(*args, **kwargs)
    monkeypatch.setattr(torch, "save", save)
    class Artifact:
        def save(self, path):
            tick()
            Path(path).write_text("artifact")
        def transform(self, features):
            tick()
            return features
    def features(*args, **kwargs):
        tick()
        return torch.zeros(4, 2)
    def model(*args, **kwargs):
        tick()
        result = torch.nn.Linear(2, 1)
        result.get_args = lambda: {}
        return result
    class Classifier(Artifact):
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            tick(5.)
            self.training_history = dict(epochs_completed=1, optimizer_updates=1,
                best_epoch=0, stopped_early=False, best_weights_restored=True)
    def sequence_ids(data, **kwargs):
        tick(10000.)  # Must not inflate any integration stage or total.
        return []
    args = SimpleNamespace(dataset=tmp_path, classifier_data=tmp_path / "structural",
        ordered_data=tmp_path / "ordered", output=tmp_path / "out", device="cpu",
        batch_size=4, quiet_training=True, plot_only=None, bundle_from_existing=None, allow_legacy_provenance=True)
    namespace = dict(Sequence=list, torch=torch, nn=torch.nn, math=math,
        _parse_args=lambda argv: args, _resolve_data_folder=lambda folder, *args: folder,
        _labels=lambda labels: labels, sequence_ids_for=sequence_ids, validate_partitions=lambda *a, **kw: {},
        make_terrain_extractor=lambda path: Artifact(), extract_dataset_features=features,
        fit_standardizer=lambda *a, **kw: Artifact(), pack_raw_depth_state_inputs=features,
        _set_seed=lambda seed: None, MODEL_SEEDS=(0, 1),
        FIXED_MODEL_CONFIGS={key: dict(dropout_p=0., weight_decay=0.) for key in ("feature_nn", "raw_depth_nn")},
        TerrainDepthFeatureClassifierNN=model, TerrainDepthClassifierNN=model,
        NeuralClassifierAdapter=Classifier, fit_nn=None, MAX_EPOCHS=1,
        EARLY_STOPPING_PATIENCE=1, save_results=lambda path, value: write(path, value))
    namespace.update({name: getattr(cost, name) for name in (
        "WallTimer", "stage_accounting", "reset_peak_memory", "peak_memory_mb",
        "artifact_size_mb", "cuda_device_info", "provenance_map", "write_cost_record")})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(path), "exec"), namespace)
    records = namespace["main"]()
    assert len(records) == 4
    for result in records:
        assert result["data_loading_s"] == 2.
        assert result["setup_s"] == 3.  # train/validation loading plus model construction
        assert result["optimization_s"] == 5.
        assert result["preprocessing_s"] > 0
        assert result["total_wallclock_s"] < 100.
        assert sum(result[k] for k in STAGE_FIELDS) == pytest.approx(result["total_wallclock_s"])
    assert records[0]["total_wallclock_s"] == records[1]["total_wallclock_s"]
    assert records[2]["total_wallclock_s"] == records[3]["total_wallclock_s"]
