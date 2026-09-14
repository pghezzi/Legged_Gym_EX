"""Launcher-only tests. Opt-in real Docker tests never import/run experiments."""
import os
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "legged_gym/scripts/run_paper_experiments_docker.sh"


class PaperDockerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper docker tests ")
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs with spaces"
        self.inputs.mkdir()
        for name in ("train.pt", "val.pt", "test.pt", "calibration.pt", "specialist.pt", "distilled.pt", "figure_data.pt"):
            (self.inputs / name).touch()
        self.offline = self.inputs / "existing offline"
        self.offline.mkdir()
        for name in ("manifest.json", "experiment_1_instantaneous_per_seed.json"):
            (self.offline / name).write_text("{}")
        for architecture in ("feature_nn", "raw_depth_nn"):
            directory = self.offline / "artifacts" / architecture
            for seed in range(3):
                (directory / f"seed_{seed}").mkdir(parents=True)
                for name in ("classifier.pt", "nn_model_args.pt"):
                    (directory / f"seed_{seed}" / name).touch()
            if architecture == "feature_nn":
                for name in ("extractor.pt", "standardizer.pt"):
                    (directory / name).touch()

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, mode, *args):
        return subprocess.run(["bash", str(SCRIPT), mode, "--output-root", str(self.root / "outputs"),
                               "--run-id", "test-run", *map(str, args)], text=True, capture_output=True)

    def common(self):
        return ["--classifier-data", self.inputs, "--ordered-data", self.inputs,
                "--jit", self.inputs / "specialist.pt", "--distilled-jit", self.inputs / "distilled.pt"]

    def test_syntax_help_and_all_dry_run(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        self.assertEqual(subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True).returncode, 0)
        result = self.launch("all", *self.common(), "--dry-run", "--gpu", "1",
                             "--offline-arg", "--batch-size", "--offline-arg", "32",
                             "--locomotion-arg", "--eval-seeds", "--locomotion-arg", "101")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/paper/offline", result.stdout)
        self.assertIn("/paper/locomotion", result.stdout)
        self.assertIn("readonly", result.stdout)
        self.assertIn("device=1", result.stdout)
        self.assertIn("--batch-size 32", result.stdout)
        self.assertIn("--eval-seeds 101", result.stdout)
        self.assertNotIn(" -it ", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_existing_results_and_plot_inputs(self):
        result = self.launch("locomotion", *self.common(), "--paper-offline-dir", self.offline, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/inputs/offline (readonly)", result.stdout)
        result = self.launch("plot-only", "--bundle", self.inputs / "figure_data.pt", "--gpu", "none", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--gpus", result.stdout)

    def test_cost_only_paths_and_no_experiment_commands(self):
        sidecar = self.inputs / "capture.pt.training_cost.json"
        sidecar.write_text("{}")
        result = self.launch("cost-only", "--paper-offline-dir", self.offline,
                             "--collection-cost", sidecar, "--compilation-cost", self.inputs,
                             "--distillation-run", self.inputs,
                             "--path-map", "/old workspace/data", self.inputs,
                             "--deployment-artifact", "distilled:0", self.inputs / "distilled.pt",
                             "--gpu", "none", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("summarize_training_costs", result.stdout)
        self.assertIn("/paper/costs", result.stdout)
        self.assertIn("/inputs/offline", result.stdout)
        self.assertIn("readonly", result.stdout)
        self.assertIn("--no-auto-distillation", result.stdout)
        self.assertNotIn("evaluate_paper_offline_experiments_1_2", result.stdout)
        self.assertNotIn("run_paper_locomotion_evaluation", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_evaluation_failure_still_audits_and_preserves_first_exit(self):
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        docker = fake_bin / "docker"
        docker.write_text("#!/usr/bin/env bash\nset -eu\n"
                          "while (($#)); do\n"
                          "  if [[ $1 == paper ]]; then\n"
                          "    case $2 in offline) exit 0;; locomotion) exit 7;; costs) exit 4;; esac\n"
                          "  fi\n  shift\ndone\nexit 99\n")
        docker.chmod(0o755)
        with mock.patch.dict(os.environ, {"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]}):
            result = self.launch("all", *self.common())
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        run = self.root / "outputs/test-run"
        self.assertEqual((run / "logs/costs.exit_status").read_text().strip(), "4")
        self.assertEqual((run / "exit_status").read_text().strip(), "7")
        self.assertEqual(len((run / "commands.sh").read_text().splitlines()), 3)

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_COSTS") == "1" and shutil.which("docker"), "opt-in real cost aggregation")
    def test_real_cost_only_without_experiments(self):
        training = []
        for architecture, method in (("feature_nn", "Feature Router"), ("raw_depth_nn", "Raw-Depth Router")):
            for seed in range(3):
                training.append(dict(architecture=architecture, seed=seed))
                record = dict(run_type="classifier_training", architecture=architecture, method=method,
                              seed=seed, optimization_s=2., timing_schema_version=2,
                              metric_status={"optimization_s": "measured"})
                (self.offline / f"artifacts/{architecture}/seed_{seed}/training_cost.json").write_text(json.dumps(record))
        manifest = self.offline / "manifest.json"
        manifest.write_text(json.dumps(dict(training_runs=training, structural_dataset="/not-mounted/compiled")))
        before = manifest.read_bytes()
        result = self.launch("cost-only", "--paper-offline-dir", self.offline, "--gpu", "none")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run = self.root / "outputs/test-run"
        self.assertTrue((run / "costs/training_cost_manifest.json").is_file())
        audit = json.loads((run / "costs/training_cost_manifest.json").read_text())
        self.assertTrue(all(r["total_wallclock_s"] is None for r in audit["per_run_records"]))
        self.assertEqual(len(audit["per_run_records"]), 6)
        self.assertIn("summarize_training_costs", (run / "commands.sh").read_text())
        self.assertFalse((run / "logs/offline.log").exists())
        self.assertFalse((run / "logs/locomotion.log").exists())
        self.assertEqual((run / "logs/costs.exit_status").read_text().strip(), "0")
        self.assertEqual(manifest.read_bytes(), before)
        for path in (run / "costs").rglob("*"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o777 if path.is_dir() else 0o666)
        # Relocate the complete preparation graph; duplicate source references
        # must still count once even through two historic path aliases.
        def timing(kind, dataset, seconds, **extra):
            return dict(run_type=kind, dataset_path=dataset, timing_schema_version=2,
                        total_wallclock_s=seconds, gpu_hours=0., gpu_count=0,
                        setup_s=0., data_generation_s=0., preprocessing_s=0.,
                        optimization_s=seconds, artifact_serialization_s=0., overhead_s=0., **extra)
        for row in training:
            path = self.offline / f"artifacts/{row['architecture']}/seed_{row['seed']}/training_cost.json"
            original = json.loads(path.read_text())
            original.update(timing("classifier_training", "unused", 2.))
            path.write_text(json.dumps(original))
        compiled = self.inputs / "compiled"
        compiled.mkdir()
        compilation = compiled / "training_cost.json"
        compilation.write_text(json.dumps(timing("dataset_compilation", "/recorded/compiled", 3.,
                                                source_files=["/recorded/raw.pt", "/alias/raw.pt"])))
        collection = self.inputs / "raw.pt.training_cost.json"
        collection.write_text(json.dumps(timing("classifier_data_collection", "/recorded/raw.pt", 5., data_env_steps=12)))
        manifest.write_text(json.dumps(dict(training_runs=training, structural_dataset="/recorded/compiled")))
        before = {path: path.read_bytes() for path in (manifest, compilation, collection)}
        result = self.launch("cost-only", "--paper-offline-dir", self.offline, "--gpu", "none", "--run-id", "relocated",
                             "--collection-cost", collection, "--compilation-cost", compiled,
                             "--path-map", "/recorded", self.inputs, "--path-map", "/alias", self.inputs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        audit = json.loads((self.root / "outputs/relocated/costs/training_cost_manifest.json").read_text())
        self.assertTrue(all(r["total_wallclock_s"] == 10. for r in audit["per_run_records"]))
        self.assertTrue(all(r["data_env_steps"] == 12 for r in audit["per_run_records"]))
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
        manifest.write_text("invalid JSON")
        result = self.launch("cost-only", "--paper-offline-dir", self.offline, "--gpu", "none", "--run-id", "cost-failure")
        self.assertNotEqual(result.returncode, 0)
        failure = self.root / "outputs/cost-failure"
        self.assertEqual(int((failure / "logs/costs.exit_status").read_text()), result.returncode)
        self.assertEqual(stat.S_IMODE((failure / "logs/costs.log").stat().st_mode), 0o666)

    def test_invalid_paths_overrides_and_existing_run_rejected(self):
        for args in (("--output",), ("--out=/tmp/escape",), ("--classifier-d",)):
            result = self.launch("offline", *self.common(), "--dry-run", "--offline-arg", *args)
            self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(self.launch("offline", "--dry-run").returncode, 0)
        self.assertNotEqual(self.launch("all", *self.common(), "--run-id", "../escape", "--dry-run").returncode, 0)
        old = self.root / "outputs/test-run"
        old.mkdir(parents=True, mode=0o700)
        before = stat.S_IMODE(old.stat().st_mode)
        self.assertNotEqual(self.launch("all", *self.common(), "--dry-run").returncode, 0)
        self.assertEqual(stat.S_IMODE(old.stat().st_mode), before)
        (self.inputs / "calibration.pt").unlink()
        self.assertNotEqual(self.launch("offline", *self.common(), "--run-id", "missing-calibration", "--dry-run").returncode, 0)

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SMOKE") == "1" and shutil.which("docker"), "opt-in Docker test")
    def test_real_container_success_and_failure_permissions(self):
        image = os.environ.get("PAPER_SMOKE_IMAGE", "ubuntu:20.04")
        input_mode = stat.S_IMODE((self.inputs / "figure_data.pt").stat().st_mode)
        for outcome, code in (("success", 0), ("failure", 23)):
            result = self.launch("plot-only", "--bundle", self.inputs / "figure_data.pt", "--image", image,
                                 "--gpu", "none", "--run-id", outcome, "--smoke-test", outcome)
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)
            run = self.root / "outputs" / outcome
            self.assertEqual((run / "exit_status").read_text().strip(), str(code))
            self.assertEqual((run / "logs/offline.exit_status").read_text().strip(), str(code))
            self.assertIn("smoke stdout", (run / "logs/offline.log").read_text())
            self.assertIn("smoke stderr", (run / "logs/offline.log").read_text())
            for path in [run, *run.rglob("*")]:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o777 if path.is_dir() else 0o666, str(path))
            self.assertEqual((run / "offline/smoke/artifact.txt").read_text(), "host-visible artifact\n")
            # A different uid can traverse, read, overwrite and create outputs.
            subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "65534:65534",
                            "--mount", f"type=bind,source={run},target=/paper", "--entrypoint", "bash", image,
                            "-c", "umask 000; test -r /paper/offline/smoke/artifact.txt && "
                            "printf writable >> /paper/offline/smoke/artifact.txt && "
                            "touch /paper/locomotion/other-user.txt"], check=True, capture_output=True)
        self.assertEqual(stat.S_IMODE((self.inputs / "figure_data.pt").stat().st_mode), input_mode)


if __name__ == "__main__":
    unittest.main()
