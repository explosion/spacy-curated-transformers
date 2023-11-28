import re

import pytest
import spacy
from spacy.cli import app
from typer.testing import CliRunner

import spacy_curated_transformers.cli.debug_pieces
import spacy_curated_transformers.cli.fill_config_transformer
from spacy_curated_transformers._compat import has_hf_transformers, has_huggingface_hub

from .util import make_tempdir


def test_debug_pieces():
    result = CliRunner().invoke(app, ["debug", "pieces", "--help"])
    assert result.exit_code == 0


# fmt: off
FILL_TRANSFORMER_CONFIG_STRS_AND_OUTPUTS = [
# TODO: add albert testing model
(
"""
[nlp]
lang = "en"
pipeline = ["transformer"]
[components]
[components.transformer]
factory = "curated_transformer"
[components.transformer.model]
@architectures = "spacy-curated-transformers.BertTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.BertWordpieceEncoder.v1"}
[components.transformer.model.with_spans]
@architectures = "spacy-curated-transformers.WithStridedSpans.v1"
""",

"""
[nlp]
lang = "en"
pipeline = ["transformer"]

[components]

[components.transformer]
factory = "curated_transformer"

[components.transformer.model]
@architectures = "spacy-curated-transformers.BertTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.BertWordpieceEncoder.v1"}
with_spans = {"@architectures":"spacy-curated-transformers.WithStridedSpans.v1"}
attention_probs_dropout_prob = 0.1
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_width = 32
intermediate_width = 37
layer_norm_eps = 0.0
max_position_embeddings = 512
model_max_length = 512
num_attention_heads = 4
num_hidden_layers = 5
padding_idx = 0
type_vocab_size = 16
vocab_size = 1124

[initialize]

[initialize.components]

[initialize.components.transformer]

[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-bert"
revision = "main"

[initialize.components.transformer.piecer_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-bert"
revision = "main"
""",

["--model-name", "hf-internal-testing/tiny-random-bert", "--model-revision", "main"],
),

(
"""
[nlp]
lang = "en"
pipeline = ["transformer"]
[components]
[components.transformer]
factory = "curated_transformer"
[components.transformer.model]
@architectures = "spacy-curated-transformers.CamembertTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.ByteBpeEncoder.v1"}
[components.transformer.model.with_spans]
@architectures = "spacy-curated-transformers.WithStridedSpans.v1"
""",

"""
[nlp]
lang = "en"
pipeline = ["transformer"]

[components]

[components.transformer]
factory = "curated_transformer"

[components.transformer.model]
@architectures = "spacy-curated-transformers.CamembertTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.ByteBpeEncoder.v1"}
with_spans = {"@architectures":"spacy-curated-transformers.WithStridedSpans.v1"}
attention_probs_dropout_prob = 0.1
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_width = 32
intermediate_width = 37
layer_norm_eps = 0.0
max_position_embeddings = 512
model_max_length = 512
num_attention_heads = 4
num_hidden_layers = 5
padding_idx = 1
type_vocab_size = 16
vocab_size = 1000

[initialize]

[initialize.components]

[initialize.components.transformer]

[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-camembert"
revision = "main"

[initialize.components.transformer.piecer_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-camembert"
revision = "main"
""",

["--model-name", "hf-internal-testing/tiny-random-camembert", "--model-revision", "main"],
),

(
"""
[nlp]
lang = "en"
pipeline = ["transformer"]
[components]
[components.transformer]
factory = "curated_transformer"
[components.transformer.model]
@architectures = "spacy-curated-transformers.RobertaTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.ByteBpeEncoder.v1"}
[components.transformer.model.with_spans]
@architectures = "spacy-curated-transformers.WithStridedSpans.v1"
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-roberta"
[initialize.components.transformer.piecer_loader]
@model_loaders = "spacy-curated-transformers.ByteBpeLoader.v1"
vocab_path = "/tmp/1"
merges_path = "/tmp/2"
""",

"""
[nlp]
lang = "en"
pipeline = ["transformer"]

[components]

[components.transformer]
factory = "curated_transformer"

[components.transformer.model]
@architectures = "spacy-curated-transformers.RobertaTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.ByteBpeEncoder.v1"}
with_spans = {"@architectures":"spacy-curated-transformers.WithStridedSpans.v1"}
attention_probs_dropout_prob = 0.1
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_width = 32
intermediate_width = 37
layer_norm_eps = 0.0
max_position_embeddings = 512
model_max_length = 512
num_attention_heads = 4
num_hidden_layers = 5
padding_idx = 1
type_vocab_size = 16
vocab_size = 1000

[initialize]

[initialize.components]

[initialize.components.transformer]

[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-roberta"
revision = "main"

[initialize.components.transformer.piecer_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "hf-internal-testing/tiny-random-roberta"
revision = "main"
""",

["--model-name", "hf-internal-testing/tiny-random-roberta", "--model-revision", "main"],
),

(
"""
[nlp]
lang = "en"
pipeline = ["transformer"]
[components]
[components.transformer]
factory = "curated_transformer"
[components.transformer.model]
@architectures = "spacy-curated-transformers.XlmrTransformer.v1"
piece_encoder = {"@architectures":"spacy-curated-transformers.XlmrSentencepieceEncoder.v1"}
[components.transformer.model.with_spans]
@architectures = "spacy-curated-transformers.WithStridedSpans.v1"
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
revision = "main"
[initialize.components.transformer.piecer_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
revision = "main"
""",

"""
[nlp]
lang = "en"
pipeline = ["transformer"]

[components]

[components.transformer]
factory = "curated_transformer"

[components.transformer.model]
@architectures = "spacy-curated-transformers.XlmrTransformer.v1"
attention_probs_dropout_prob = 0.1
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_width = 32
intermediate_width = 37
layer_norm_eps = 0.00001
max_position_embeddings = 512
model_max_length = 2147483647
num_attention_heads = 4
num_hidden_layers = 5
padding_idx = 1
type_vocab_size = 16
vocab_size = 1024
piece_encoder = {"@architectures":"spacy-curated-transformers.XlmrSentencepieceEncoder.v1"}
with_spans = {"@architectures":"spacy-curated-transformers.WithStridedSpans.v1"}

[initialize]

[initialize.components]

[initialize.components.transformer]

[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
revision = "main"

[initialize.components.transformer.piecer_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
revision = "main"
""",

[],
),
]
# fmt: on


@pytest.mark.slow
@pytest.mark.skipif(not has_huggingface_hub, reason="requires Hugging Face Hub")
@pytest.mark.skipif(
    not has_hf_transformers, reason="requires Hugging Face transformers"
)
@pytest.mark.parametrize(
    "config, output, extra_args", FILL_TRANSFORMER_CONFIG_STRS_AND_OUTPUTS
)
def test_fill_config_transformer(config, output, extra_args):
    with make_tempdir() as d:
        file_path = d / "test_conf"
        output_path = d / "output_conf"
        with open(file_path, "w", encoding="utf8") as f:
            f.writelines([config])

        result = CliRunner().invoke(
            app,
            [
                "init",
                "fill-curated-transformer",
                str(file_path),
                str(output_path),
            ]
            + extra_args,
        )
        try:
            assert result.exit_code == 0
        except AssertionError:
            if result.exception is not None:
                raise result.exception
            else:
                raise ValueError(
                    f"Curated Transformer fill config failed! Stderr: \n{result.stderr}"
                )
        filled_config = spacy.util.load_config(output_path)
        expected_config = spacy.util.load_config_from_str(output)
        assert filled_config == expected_config


@pytest.mark.slow
@pytest.mark.skipif(not has_huggingface_hub, reason="requires Hugging Face Hub")
@pytest.mark.skipif(
    not has_hf_transformers, reason="requires Hugging Face transformers"
)
@pytest.mark.parametrize(
    "config", [output for _, output, _ in FILL_TRANSFORMER_CONFIG_STRS_AND_OUTPUTS]
)
def test_validate_test_filled_configs(config):
    config = spacy.util.load_config_from_str(config, interpolate=True)
    nlp = spacy.util.load_model_from_config(config, validate=True, auto_fill=True)
    nlp.initialize()
