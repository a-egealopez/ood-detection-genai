import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.benchmarking.generate_summary import run_summary


def _write_eval_result(logs_dir: Path, method_type: str, experiment_id: str, data: dict) -> Path:
    fp = logs_dir / method_type / experiment_id / "eval_results.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(data))
    return fp


def _make_eval_data(
    method: str = "vae",
    dataset: str = "mnist",
    seed: int = 42,
    lr: str = "1e-3",
    modes: list[str] | None = None,
) -> dict:
    modes = modes or ["recon"]
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "lr": lr,
        "metrics": {
            mode: {
                "auroc": 0.90 + i * 0.01,
                "aupr": 0.88 + i * 0.01,
                "fpr_at_95_tpr": 0.12 - i * 0.01,
            }
            for i, mode in enumerate(modes)
        },
        "thresholds": {
            mode: {"threshold": 0.05 + i * 0.01, "tpr": 0.95, "fpr": 0.05}
            for i, mode in enumerate(modes)
        },
    }


class TestRunSummary:
    """Test run_summary function."""

    def test_run_summary_creates_output_dir(self, temp_checkpoint_dir):
        """Happy path: run_summary creates output directory if it doesn't exist."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "results" / "summary.csv"
        logs_dir.mkdir(parents=True)

        result = run_summary(str(logs_dir), str(out_csv))

        assert out_csv.parent.exists()
        assert result == out_csv

    def test_run_summary_creates_csv_file(self, temp_checkpoint_dir):
        """Happy path: run_summary creates CSV file at output path."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"
        logs_dir.mkdir(parents=True)

        result = run_summary(str(logs_dir), str(out_csv))

        assert out_csv.exists()
        assert result == out_csv

    def test_run_summary_csv_has_headers(self, temp_checkpoint_dir):
        """Happy path: Generated CSV has expected headers."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"
        logs_dir.mkdir(parents=True)

        run_summary(str(logs_dir), str(out_csv))

        with open(out_csv) as f:
            first_line = f.readline().strip()

        expected_headers = [
            "file", "method", "dataset", "seed", "lr",
            "score", "auroc", "aupr", "fpr_at_95_tpr",
            "threshold_at_5_fpr", "tpr_at_5_fpr", "fpr_at_5_fpr",
        ]
        for header in expected_headers:
            assert header in first_line, f"Header '{header}' missing from CSV"

    def test_run_summary_single_log_file(self, temp_checkpoint_dir):
        """Happy path: Process single JSON log file successfully."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        _write_eval_result(
            logs_dir, "vae", "recon_vae_mnist_s42_lr1e-3_ep100",
            _make_eval_data()
        )

        result = run_summary(str(logs_dir), str(out_csv))

        with open(out_csv) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_run_summary_multiple_log_files(self, temp_checkpoint_dir):
        """Happy path: Process multiple JSON log files."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        for seed in [42, 43, 44]:
            _write_eval_result(
                logs_dir, "vae", f"recon_vae_mnist_s{seed}_lr1e-3_ep100",
                _make_eval_data(seed=seed)
            )

        run_summary(str(logs_dir), str(out_csv))

        with open(out_csv) as f:
            lines = f.readlines()

        assert len(lines) == 4

    def test_run_summary_extracts_all_metrics(self, temp_checkpoint_dir):
        """Happy path: auroc, aupr, fpr_at_95_tpr y thresholds se extraen correctamente."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        _write_eval_result(
            logs_dir, "vae", "recon_vae_mnist_s42_lr1e-3_ep100",
            _make_eval_data()
        )

        run_summary(str(logs_dir), str(out_csv))

        import csv
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert float(row["auroc"]) == 0.90
        assert float(row["aupr"]) == 0.88
        assert float(row["fpr_at_95_tpr"]) == 0.12
        assert float(row["threshold_at_5_fpr"]) == 0.05
        assert float(row["tpr_at_5_fpr"]) == 0.95
        assert float(row["fpr_at_5_fpr"]) == 0.05

    def test_run_summary_extracts_metadata_from_json(self, temp_checkpoint_dir):
        """Happy path: método, dataset, seed y lr se leen del JSON."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        _write_eval_result(
            logs_dir, "vae", "recon_vae_mnist_s42_lr1e-3_ep100",
            _make_eval_data(method="vae", dataset="mnist", seed=42, lr="1e-3")
        )

        run_summary(str(logs_dir), str(out_csv))

        import csv
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))

        row = rows[0]
        assert row["method"] == "vae"
        assert row["dataset"] == "mnist"
        assert row["seed"] == "42"
        assert row["lr"] == "1e-3"

    def test_run_summary_extracts_metadata_from_folder(self, temp_checkpoint_dir):
        """Happy path: metadata se extrae de la estructura de carpetas como fallback."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        data = {
            "metrics": {"recon": {"auroc": 0.91, "aupr": 0.89, "fpr_at_95_tpr": 0.11}},
            "thresholds": {},
        }
        _write_eval_result(logs_dir, "vae", "recon_vae_mnist_s99_lr1e-4_ep50", data)

        run_summary(str(logs_dir), str(out_csv))

        import csv
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))

        row = rows[0]
        assert row["method"] == "vae"
        assert row["dataset"] == "mnist"
        assert row["seed"] == "99"
        assert row["lr"] == "1e-4"

    def test_run_summary_handles_invalid_json(self, temp_checkpoint_dir):
        """Happy path: Invalid JSON files are skipped gracefully."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        _write_eval_result(
            logs_dir, "vae", "recon_vae_mnist_s42_lr1e-3_ep100",
            _make_eval_data()
        )

        bad = logs_dir / "vae" / "recon_vae_mnist_s99_lr1e-3_ep100" / "eval_results.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("not valid json {{{")

        run_summary(str(logs_dir), str(out_csv))

        import csv
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1

    def test_run_summary_multiple_score_modes(self, temp_checkpoint_dir):
        """Happy path: Multiple score modes are expanded to separate rows."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        _write_eval_result(
            logs_dir, "vae", "recon_vae_mnist_s42_lr1e-3_ep100",
            _make_eval_data(modes=["recon", "knn", "residual"])
        )

        run_summary(str(logs_dir), str(out_csv))

        with open(out_csv) as f:
            lines = f.readlines()

        assert len(lines) == 4


class TestBenchmarkingIntegration:
    """Integration tests for benchmarking pipeline."""

    def test_full_summary_pipeline(self, temp_checkpoint_dir):
        """Happy path: Complete summary generation from multiple diverse logs."""
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        methods = ["vae", "ddpm"]
        datasets = ["mnist", "sicap"]
        seeds = [42, 123]
        modes = ["recon", "knn"]

        for method in methods:
            for dataset in datasets:
                for seed in seeds:
                    _write_eval_result(
                        logs_dir,
                        method,
                        f"recon_{method}_{dataset}_s{seed}_lr1e-3_ep100",
                        _make_eval_data(method=method, dataset=dataset, seed=seed, modes=modes),
                    )

        run_summary(str(logs_dir), str(out_csv))

        import csv
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 16

        for row in rows:
            assert row["auroc"], f"auroc missing in {row}"
            assert row["aupr"], f"aupr missing in {row}"
            assert row["fpr_at_95_tpr"], f"fpr_at_95_tpr missing in {row}"

    def test_distance_method_structure(self, temp_checkpoint_dir):
        logs_dir = temp_checkpoint_dir / "logs"
        out_csv = temp_checkpoint_dir / "summary.csv"

        data = _make_eval_data(method="dist_pixel", dataset="mnist", seed=42, modes=["residual", "knn"])
        data.pop("lr")  # distance method no tiene lr
        _write_eval_result(logs_dir, "distance", "dist_pixel_mnist_s42", data)

        run_summary(str(logs_dir), str(out_csv))

        import csv
        with open(out_csv) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["method"] == "dist_pixel"
        assert rows[0]["dataset"] == "mnist"