import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from claimlens.calibration import (
    area_under_precision_coverage,
    bootstrap_interval,
    brier_score,
    expected_calibration_error,
    fit_logistic,
    fit_temperature,
    negative_log_likelihood,
    precision_coverage_curve,
    reliability_bins,
    softmax,
)


def synthetic_overconfident(temperature, n=20000, seed=0):
    """Logits whose true scale is known: labels are sampled at temperature 1."""
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(size=(n, 3)) * 2.0
    probabilities = softmax(true_logits)
    labels = np.array([rng.choice(3, p=p) for p in probabilities])
    return true_logits * temperature, labels


def test_softmax_is_stable_for_large_logits():
    probabilities = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isfinite(probabilities).all()
    assert probabilities[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.5])
def test_fit_temperature_recovers_a_known_scale(true_temperature):
    logits, labels = synthetic_overconfident(true_temperature)
    assert fit_temperature(logits, labels) == pytest.approx(true_temperature, rel=0.05)


def test_temperature_scaling_cannot_change_the_predicted_class():
    logits, labels = synthetic_overconfident(2.0, n=500)
    temperature = fit_temperature(logits, labels)
    assert (logits.argmax(1) == (logits / temperature).argmax(1)).all()


def test_scaling_reduces_ece_and_nll_on_overconfident_logits():
    logits, labels = synthetic_overconfident(3.0)
    temperature = fit_temperature(logits, labels)
    before = softmax(logits)
    after = softmax(logits / temperature)
    correct = before.argmax(1) == labels
    assert expected_calibration_error(after.max(1), correct) < expected_calibration_error(
        before.max(1), correct
    )
    assert negative_log_likelihood(logits / temperature, labels) < negative_log_likelihood(
        logits, labels
    )


@pytest.mark.parametrize("l2", [0.1, 1.0, 10.0])
def test_fit_logistic_matches_scikit_learn(l2):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 3))
    weights = np.array([1.5, -2.0, 0.5])
    targets = (rng.random(300) < 1 / (1 + np.exp(-(features @ weights + 0.4)))).astype(float)
    reference = LogisticRegression(C=1.0 / l2, tol=1e-10, max_iter=10_000).fit(features, targets)
    ours = fit_logistic(features, targets, l2=l2)
    assert ours[:-1] == pytest.approx(reference.coef_[0], abs=1e-4)
    assert ours[-1] == pytest.approx(reference.intercept_[0], abs=1e-4)


def test_fit_logistic_stays_finite_on_separable_data():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    weights = fit_logistic(features, np.array([0.0, 0.0, 1.0, 1.0]))
    assert np.isfinite(weights).all()
    assert weights[0] > 0


def test_ece_hand_computed():
    confidences = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 1, 0], dtype=bool)
    # Two bins with two items each: |1.0 - 0.95| and |0.5 - 0.55|.
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.05)


def test_ece_is_zero_when_confidence_equals_accuracy():
    confidences = np.array([0.75, 0.75, 0.75, 0.75])
    correct = np.array([1, 1, 1, 0], dtype=bool)
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)


def test_reliability_bins_count_every_item_once():
    rng = np.random.default_rng(0)
    confidences = rng.random(500)
    correct = rng.random(500) < confidences
    bins = reliability_bins(confidences, correct)
    assert sum(b["count"] for b in bins) == 500


def test_a_confidence_of_exactly_one_lands_in_the_top_bin():
    bins = reliability_bins(np.array([1.0]), np.array([True]))
    assert bins[0]["upper"] == 1.0


def test_brier_hand_computed():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    assert brier_score(probabilities, np.array([0, 1])) == pytest.approx(0.25)


def test_precision_coverage_hand_computed():
    curve = precision_coverage_curve(
        np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, False, True, True])
    )
    assert [p["coverage"] for p in curve] == [0.25, 0.5, 0.75, 1.0]
    assert [p["precision"] for p in curve] == pytest.approx([1.0, 0.5, 2 / 3, 0.75])


def test_precision_coverage_keeps_tied_confidences_together():
    curve = precision_coverage_curve(np.array([0.9, 0.9, 0.5]), np.array([True, False, True]))
    assert [p["answered"] for p in curve] == [2, 3]


def test_area_of_a_flat_curve_is_its_height_over_the_covered_range():
    curve = [{"coverage": 0.5, "precision": 0.8}, {"coverage": 1.0, "precision": 0.8}]
    assert area_under_precision_coverage(curve) == pytest.approx(0.4)


def test_bootstrap_interval_brackets_the_statistic():
    values = np.random.default_rng(1).normal(size=200)
    low, high = bootstrap_interval(lambda i: float(values[i].mean()), 200, n_resamples=500)
    assert low < values.mean() < high


def test_bootstrap_interval_is_reproducible():
    values = np.arange(50.0)
    first = bootstrap_interval(lambda i: float(values[i].mean()), 50, n_resamples=100)
    second = bootstrap_interval(lambda i: float(values[i].mean()), 50, n_resamples=100)
    assert first == second
