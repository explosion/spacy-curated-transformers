import pytest
from spacy.util import registry as spacy_registry

from spacy_curated_transformers.util import registry


@pytest.mark.parametrize(
    "model_name",
    [
        "spacy-curated-transformers.AlbertTransformer.v1",
        "spacy-curated-transformers.BertTransformer.v1",
        "spacy-curated-transformers.CamembertTransformer.v1",
        "spacy-curated-transformers.RobertaTransformer.v1",
        "spacy-curated-transformers.XlmrTransformer.v1",
        "spacy-curated-transformers.WithStridedSpans.v1",
        "spacy-curated-transformers.ScalarWeight.v1",
        "spacy-curated-transformers.TransformerLayersListener.v1",
        "spacy-curated-transformers.LastTransformerLayerListener.v1",
        "spacy-curated-transformers.ScalarWeightingListener.v1",
    ],
)
def test_model_from_registry(model_name):
    # Can't be constructed, since all models have mandatory arguments.
    spacy_registry.architectures.get(model_name)


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
def test_tokenizer_encoder_from_registry(encoder_name):
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
def test_model_loaders_from_registry(loader_name):
    # Can't be constructed, since most loaders have mandatory arguments.
    registry.model_loaders.get(loader_name)


@pytest.mark.parametrize(
    "callback_name",
    [
        "spacy-curated-transformers.gradual_transformer_unfreezing.v1",
    ],
)
def test_callbacks_from_registry(callback_name):
    # Can't be constructed, since all callbacks have mandatory arguments.
    spacy_registry.callbacks.get(callback_name)
