import pytest

from claimlens.fusion import reciprocal_rank_fusion


def test_hand_computed_scores():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 12 + 1 / 11)
    assert fused["c"] == pytest.approx(1 / 12)


def test_agreement_beats_a_single_first_place():
    ranked = [doc for doc, _ in reciprocal_rank_fusion([["x", "both"], ["y", "both"]], k=60)]
    assert ranked[0] == "both"


def test_scores_ignore_list_scale_by_construction():
    assert reciprocal_rank_fusion([[3, 1, 2]], k=1)[0][0] == 3


def test_weights_shift_the_outcome():
    ranked = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[1.0, 2.0])
    assert ranked[0][0] == "b"


def test_ties_keep_first_appearance_order():
    assert [d for d, _ in reciprocal_rank_fusion([["a"], ["b"]], k=60)] == ["a", "b"]


def test_validation():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="weight"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([["a", "a"]])
