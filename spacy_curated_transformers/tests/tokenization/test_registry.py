import pytest
from spacy.util import registry as spacy_registry
from spacy_curated_transformers.util import registry


@pytest.mark.parametrize(
    "encoder_name",
    [
        "spacy-curated-transformers.BertWordpieceEncoder.v1",
        "spacy-curated-transformers.ByteBpeEncoder.v1",
        "spacy-curated-transformers.CamembertSentencepieceEncoder.v1",
        "spacy-curated-transformers.CharEncoder.v1",
        "spacy-curated-transformers.SentencepieceEncoder.v1",
        "spacy-curated-transformers.WordpieceEncoder.v1",
        "spacy-curated-transformers.XlmrSentencepieceEncoder.v1",
    ],
)
def test_encoder_from_registry(encoder_name):
    spacy_registry.architectures.get(encoder_name)()


@pytest.mark.parametrize(
    "loader_name",
    [
        "spacy-curated-transformers.ByteBpeLoader.v1",
        "spacy-curated-transformers.CharEncoderLoader.v1",
        "spacy-curated-transformers.HFTransformerEncoderLoader.v1",
        "spacy-curated-transformers.HFPieceEncoderLoader.v1",
        "spacy-curated-transformers.PyTorchCheckpointLoader.v1",
        "spacy-curated-transformers.SentencepieceLoader.v1",
        "spacy-curated-transformers.WordpieceLoader.v1",
    ],
)
def test_encoder_loader_from_registry(loader_name):
    # Can't be constructed, since most loaders have mandatory arguments.
    registry.model_loaders.get(loader_name)
