"""Launcher-only tests. Opt-in real Docker tests never import/run experiments."""
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


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
