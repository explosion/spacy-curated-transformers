import pytest
from spacy.attrs import IS_SPACE
from spacy.tokens import Doc
from spacy.vocab import Vocab
from thinc.api import Ragged, get_current_ops, registry

from spacy_curated_transformers._compat import has_hf_transformers
from spacy_curated_transformers.tokenization.hf_loader import (
    build_hf_piece_encoder_loader_v1,
)
from spacy_curated_transformers.tokenization.sentencepiece_encoder import (
    build_sentencepiece_encoder_v1,
)


@pytest.fixture
def toy_encoder(sentencepiece_toy_model_path):
    encoder = build_sentencepiece_encoder_v1()
    encoder.init = registry.model_loaders.get(
        "spacy-curated-transformers.SentencepieceLoader.v1"
    )(path=sentencepiece_toy_model_path)
    encoder.initialize()
    return encoder


def test_sentencepiece_encoder(toy_encoder, sample_docs):
    encoding = toy_encoder.predict(sample_docs)
    _check_toy_encoder(encoding)


@pytest.mark.parametrize(
    "strings",
    [["a", "b", "c"], ["a", " ", "c"], ["cat", "sat", "mat"], [" "], [" ", "hi"]],
)
def test_sp_whitespace_encoding(toy_encoder, strings):
    doc = Doc(
        Vocab(lex_attr_getters={IS_SPACE: lambda w: int(w.isspace())}), words=strings
    )
    encoding = toy_encoder.predict([doc])
    # Strip bos and eos markers
    doc_encoding = encoding[0][1:-1]
    for i, token in enumerate(doc):
        if token.is_space:
            assert doc_encoding.lengths[i] == 0
        else:
            assert doc_encoding.lengths[i] >= 1


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_sentencepiece_encoder_hf_model(sample_docs):
    encoder = build_sentencepiece_encoder_v1()
    encoder.init = build_hf_piece_encoder_loader_v1(name="xlm-roberta-base")
    encoder.initialize()

    encoding = encoder.predict(sample_docs)

    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops = get_current_ops()
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 2, 2, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd, [1, 86, 24123, 9, 23039, 677, 9, 5500, 70819, 5, 4, 2]
    )

    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 1, 1, 1, 1, 2, 1, 2, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd, [1, 38395, 641, 1220, 73202, 159, 7663, 120323, 5, 4, 2]
    )


def test_serialize(toy_encoder, sample_docs):
    encoder_bytes = toy_encoder.to_bytes()
    encoder2 = build_sentencepiece_encoder_v1()
    encoder2.from_bytes(encoder_bytes)
    encoding = encoder2.predict(sample_docs)
    _check_toy_encoder(encoding)


def _check_toy_encoder(encoding):
    assert isinstance(encoding, list)
    assert len(encoding) == 2

    assert isinstance(encoding[0], Ragged)
    ops = get_current_ops()
    ops.xp.testing.assert_array_equal(
        encoding[0].lengths, [1, 1, 1, 1, 1, 1, 1, 6, 2, 1]
    )
    ops.xp.testing.assert_array_equal(
        encoding[0].dataXd,
        [1, 8, 465, 10, 947, 41, 10, 170, 168, 110, 28, 20, 143, 7, 4, 2],
    )

    assert isinstance(encoding[1], Ragged)
    ops.xp.testing.assert_array_equal(encoding[1].lengths, [1, 2, 1, 1, 1, 4, 3, 2, 1])
    ops.xp.testing.assert_array_equal(
        encoding[1].dataXd,
        [1, 483, 546, 112, 171, 567, 62, 20, 45, 0, 84, 115, 27, 7, 4, 2],
    )
