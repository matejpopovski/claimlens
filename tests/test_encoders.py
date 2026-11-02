import numpy as np
import pytest

from claimlens.encoders import CrossEncoder, DenseIndex, TextEncoder, length_sorted_batches


def test_embeddings_are_unit_length(tiny_encoder_dir):
    encoder = TextEncoder(tiny_encoder_dir, device="cpu")
    vectors = encoder.encode(["aspirin reduces fever", "vitamin"])
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_encoding_order_survives_length_sorting(tiny_encoder_dir):
    encoder = TextEncoder(tiny_encoder_dir, device="cpu", batch_size=2)
    texts = ["a study of patients", "aspirin", "vitamin cures scurvy"]
    together = encoder.encode(texts)
    one_by_one = np.vstack([encoder.encode([t]) for t in texts])
    assert together == pytest.approx(one_by_one, abs=1e-4)


def test_query_prefix_changes_query_vectors_only(tiny_encoder_dir):
    encoder = TextEncoder(tiny_encoder_dir, query_prefix="the study of ", device="cpu")
    document = encoder.encode(["aspirin"])
    query = encoder.encode(["aspirin"], is_query=True)
    assert not np.allclose(document, query)


def test_cls_and_mean_pooling_differ(tiny_encoder_dir):
    mean = TextEncoder(tiny_encoder_dir, pooling="mean", device="cpu")
    cls = TextEncoder(tiny_encoder_dir, pooling="cls", device="cpu")
    assert not np.allclose(mean.encode(["aspirin fever"]), cls.encode(["aspirin fever"]))


def test_unknown_pooling_is_refused(tiny_encoder_dir):
    with pytest.raises(ValueError, match="pooling"):
        TextEncoder(tiny_encoder_dir, pooling="max", device="cpu")


def test_dense_index_ranks_by_similarity_with_stable_ties():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    hits = DenseIndex(embeddings).top_k(np.array([1.0, 0.0], dtype=np.float32), 2)
    assert [doc for doc, _ in hits] == [0, 2]


def test_cross_encoder_scores_each_passage(tiny_reranker_dir):
    reranker = CrossEncoder.load(tiny_reranker_dir, device="cpu", batch_size=1)
    scores = reranker.score("aspirin", ["aspirin reduces fever", "vitamin", "a study"])
    assert scores.shape == (3,)
    assert reranker.score("aspirin", ["vitamin"])[0] == pytest.approx(scores[1], abs=1e-5)


def test_cross_encoder_needs_a_single_logit_head(tiny_nli_dir):
    with pytest.raises(ValueError, match="one output logit"):
        CrossEncoder.load(tiny_nli_dir, device="cpu")


def test_length_sorted_batches_cover_every_index_once():
    batches = length_sorted_batches([5, 1, 3, 2], 2)
    assert batches == [[1, 3], [2, 0]]
