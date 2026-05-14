import sys
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluate import (
    EvalResults,
    _aupr,
    _auroc,
    compute_metrics,
    _fpr_at_tpr,
    _labels,
    _threshold_analysis,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.extract import ScoreBundle

DEFAULT_KNN_K = 1


class TestLabels:
    """Test label generation for ROC curves."""

    def test_labels_shape(self):
        """
        Happy path: _labels produces correct shapes for ID and OOD scores.
        """
        id_scores = np.array([0.1, 0.2, 0.3])
        ood_scores = np.array([0.6, 0.7, 0.8, 0.9])

        y, s = _labels(id_scores, ood_scores)

        # y should have all labels (0 for ID, 1 for OOD)
        assert len(y) == len(id_scores) + len(ood_scores)
        assert len(s) == len(id_scores) + len(ood_scores)

    def test_labels_id_label_zero(self):
        """
        Happy path: ID samples are labeled as 0.
        """
        id_scores = np.array([0.1, 0.2])
        ood_scores = np.array([0.9])

        y, s = _labels(id_scores, ood_scores)

        # First 2 labels should be 0 (ID)
        assert np.array_equal(y[:2], [0, 0])

    def test_labels_ood_label_one(self):
        """
        Happy path: OOD samples are labeled as 1.
        """
        id_scores = np.array([0.1, 0.2])
        ood_scores = np.array([0.8, 0.9])

        y, s = _labels(id_scores, ood_scores)

        # Last 2 labels should be 1 (OOD)
        assert np.array_equal(y[-2:], [1, 1])

    def test_labels_score_concatenation(self):
        """
        Happy path: Scores are concatenated in order.
        """
        id_scores = np.array([1.0, 2.0])
        ood_scores = np.array([3.0, 4.0])

        y, s = _labels(id_scores, ood_scores)

        assert np.array_equal(s, [1.0, 2.0, 3.0, 4.0])


class TestAUROC:
    """Test AUROC computation."""

    def test_auroc_perfect_separation(self):
        """
        Happy path: Perfect separation (ID < OOD) gives AUROC = 1.0.
        """
        id_scores = np.array([0.1, 0.2, 0.3])
        ood_scores = np.array([0.9, 0.95, 1.0])

        auroc = _auroc(id_scores, ood_scores)

        assert auroc == pytest.approx(1.0, abs=0.01)

    def test_auroc_no_separation(self):
        """
        Happy path: Identical distributions give AUROC ≈ 0.5.
        """
        rng = np.random.default_rng(42)
        id_scores = rng.uniform(0, 1, 100)
        ood_scores = rng.uniform(0, 1, 100)

        auroc = _auroc(id_scores, ood_scores)

        # Should be close to 0.5 for identical distributions
        assert 0.4 < auroc < 0.6

    def test_auroc_returns_scalar(self):
        """
        Happy path: AUROC returns single float value in [0, 1].
        """
        id_scores = np.array([0.2, 0.3])
        ood_scores = np.array([0.7, 0.8])

        auroc = _auroc(id_scores, ood_scores)

        assert isinstance(auroc, float)
        assert 0 <= auroc <= 1


class TestAUPR:
    """Test AUPR (Average Precision) computation."""

    def test_aupr_perfect_separation(self):
        """
        Happy path: Perfect separation gives AUPR = 1.0.
        """
        id_scores = np.array([0.1, 0.2])
        ood_scores = np.array([0.9, 1.0])

        aupr = _aupr(id_scores, ood_scores)

        assert aupr == pytest.approx(1.0, abs=0.01)

    def test_aupr_returns_scalar(self):
        """
        Happy path: AUPR returns single float value in [0, 1].
        """
        id_scores = np.array([0.2, 0.3])
        ood_scores = np.array([0.7, 0.8])

        aupr = _aupr(id_scores, ood_scores)

        assert isinstance(aupr, float)
        assert 0 <= aupr <= 1


class TestFPRAtTPR:
    """Test FPR at target TPR computation."""

    def test_fpr_at_tpr_perfect_separation(self):
        """
        Happy path: Perfect separation gives FPR = 0 at any TPR threshold.
        """
        id_scores = np.array([0.1, 0.2])
        ood_scores = np.array([0.9, 1.0])

        fpr = _fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95)

        assert fpr == pytest.approx(0.0, abs=0.01)

    def test_fpr_at_tpr_returns_scalar(self):
        """
        Happy path: FPR@TPR returns single float value in [0, 1].
        """
        id_scores = np.array([0.2, 0.3])
        ood_scores = np.array([0.7, 0.8])

        fpr = _fpr_at_tpr(id_scores, ood_scores)

        assert isinstance(fpr, float)
        assert 0 <= fpr <= 1

    def test_fpr_at_tpr_default_threshold(self):
        """
        Happy path: Default TPR target is 0.95.
        """
        id_scores = np.array([0.2, 0.3])
        ood_scores = np.array([0.7, 0.8])

        # Should not raise error
        fpr = _fpr_at_tpr(id_scores, ood_scores)
        assert isinstance(fpr, float)


class TestComputeMetrics:
    """Test metric computation dictionary."""

    def test_compute_metrics_has_all_keys(self):
        """
        Happy path: Computed metrics has auroc, aupr, fpr_at_95_tpr.
        """
        id_scores = np.array([0.2, 0.3])
        ood_scores = np.array([0.7, 0.8])

        metrics = compute_metrics(id_scores, ood_scores)

        assert "auroc" in metrics
        assert "aupr" in metrics
        assert "fpr_at_95_tpr" in metrics

    def test_compute_metrics_values_valid(self):
        """
        Happy path: All metrics are floats in valid ranges.
        """
        id_scores = np.array([0.2, 0.3])
        ood_scores = np.array([0.7, 0.8])

        metrics = compute_metrics(id_scores, ood_scores)

        for key, val in metrics.items():
            assert isinstance(val, (float, np.floating))
            assert 0 <= val <= 1


class TestScoreBundle:
    """Test ScoreBundle dataclass."""

    def test_score_bundle_creation(self):
        """
        Happy path: ScoreBundle can be created with ID/OOD scores.
        """
        id_recon = np.array([0.1, 0.2])
        ood_recon = np.array([0.8, 0.9])

        bundle = ScoreBundle(id_recon=id_recon, ood_recon=ood_recon)

        assert np.array_equal(bundle.id_recon, id_recon)
        assert np.array_equal(bundle.ood_recon, ood_recon)

    def test_score_bundle_defaults_empty(self):
        """
        Happy path: Optional score arrays default to empty.
        """
        bundle = ScoreBundle(id_recon=np.array([0.1]), ood_recon=np.array([0.9]))

        assert len(bundle.id_residual) == 0
        assert len(bundle.ood_residual) == 0
        assert len(bundle.id_knn) == 0
        assert len(bundle.ood_knn) == 0

    def test_score_bundle_vae_modes(self):
        """
        Happy path: ScoreBundle supports VAE modes (recon, residual, knn, elbo).
        """
        bundle = ScoreBundle(
            id_recon=np.array([0.1]),
            ood_recon=np.array([0.9]),
            id_elbo=np.array([0.05]),
            ood_elbo=np.array([0.95]),
        )

        assert len(bundle.id_recon) > 0
        assert len(bundle.ood_recon) > 0
        assert len(bundle.id_elbo) > 0
        assert len(bundle.ood_elbo) > 0

    def test_score_bundle_ddpm_modes(self):
        """
        Happy path: ScoreBundle supports DDPM modes (recon, residual, knn, noise).
        """
        bundle = ScoreBundle(
            id_recon=np.array([0.1]),
            ood_recon=np.array([0.9]),
            id_noise=np.array([0.05]),
            ood_noise=np.array([0.95]),
        )

        assert len(bundle.id_recon) > 0
        assert len(bundle.ood_recon) > 0
        assert len(bundle.id_noise) > 0
        assert len(bundle.ood_noise) > 0


class TestThresholdAnalysis:
    """Test threshold computation for ROC analysis."""

    def test_threshold_analysis_structure(self):
        """
        Happy path: Threshold analysis returns dict with threshold, fpr, tpr.
        """
        bundle = ScoreBundle(
            id_recon=np.array([0.1, 0.2, 0.3]),
            ood_recon=np.array([0.7, 0.8, 0.9]),
        )
        active = [("recon", "id_recon", "ood_recon")]

        result = _threshold_analysis(bundle, active, fpr_target=0.05)

        assert "recon" in result
        assert "threshold" in result["recon"]
        assert "fpr" in result["recon"]
        assert "tpr" in result["recon"]

    def test_threshold_analysis_fpr_target_respected(self):
        """
        Happy path: FPR at threshold is close to target FPR.
        """
        np.random.seed(42)
        id_scores = np.random.uniform(0.1, 0.4, 100)
        ood_scores = np.random.uniform(0.6, 0.9, 100)

        bundle = ScoreBundle(id_recon=id_scores, ood_recon=ood_scores)
        active = [("recon", "id_recon", "ood_recon")]
        fpr_target = 0.1

        result = _threshold_analysis(bundle, active, fpr_target=fpr_target)

        # FPR should be close to target
        actual_fpr = result["recon"]["fpr"]
        assert 0 <= actual_fpr <= 0.2  # Allow some tolerance


class TestEvalResults:
    """Test EvalResults dataclass."""

    def test_eval_results_creation(self):
        """
        Happy path: EvalResults can be created with metrics and thresholds.
        """
        metrics = {"recon": {"auroc": 0.95, "aupr": 0.92, "fpr_at_95_tpr": 0.08}}
        thresholds = {"recon": {"threshold": 0.5, "fpr": 0.05, "tpr": 0.95}}

        results = EvalResults(metrics=metrics, thresholds=thresholds)

        assert results.metrics["recon"]["auroc"] == 0.95
        assert results.thresholds["recon"]["threshold"] == 0.5

    def test_eval_results_to_dict(self):
        """
        Happy path: EvalResults.to_dict() returns valid dict.
        """
        metrics = {"recon": {"auroc": 0.95}}
        thresholds = {"recon": {"threshold": 0.5}}

        results = EvalResults(metrics=metrics, thresholds=thresholds)
        result_dict = results.to_dict()

        assert "metrics" in result_dict
        assert "thresholds" in result_dict
        assert result_dict["metrics"] == metrics
        assert result_dict["thresholds"] == thresholds


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_evaluate_with_bundle_complete_workflow(self):
        """
        Happy path: Complete evaluation from ScoreBundle to metrics.
        """
        # Create synthetic ID/OOD scores
        np.random.seed(42)
        id_recon = np.random.exponential(scale=0.3, size=50)
        ood_recon = np.random.exponential(scale=0.8, size=50)

        bundle = ScoreBundle(id_recon=id_recon, ood_recon=ood_recon)

        # Compute metrics
        metrics = compute_metrics(bundle.id_recon, bundle.ood_recon)

        # Should have good separation
        assert metrics["auroc"] > 0.7  # At least 70% AUROC
