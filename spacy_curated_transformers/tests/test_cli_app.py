import re

import pytest
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
(
"""
[nlp]
lang = "en"
pipeline = ["transformer"]
[components]
[components.transformer]
factory = "curated_transformer"
[components.transformer.model]
@architectures = "spacy-curated-transformers.AlbertTransformer.v1"
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.piece_loader]
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
@architectures = "spacy-curated-transformers.AlbertTransformer.v1"
attention_probs_dropout_prob = 0
hidden_act = "gelu_new"
hidden_dropout_prob = 0
hidden_width = 32
intermediate_width = 37
layer_norm_eps = 0.0
max_position_embeddings = 512
model_max_length = 2147483647
num_attention_heads = 4
num_hidden_layers = 5
padding_idx = 0
type_vocab_size = 16
vocab_size = 1024
embedding_width = 128
num_hidden_groups = 1
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/albert-test"
revision = "main"
[initialize.components.transformer.piece_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/albert-test"
revision = "main"
""",

["--model-name", "explosion-testing/albert-test", "--model-revision", "main"],
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
@architectures = "spacy-curated-transformers.BertTransformer.v1"
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
attention_probs_dropout_prob = 0.1
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_width = 32
intermediate_width = 37
layer_norm_eps = 0.0
max_position_embeddings = 512
model_max_length = 2147483647
num_attention_heads = 4
num_hidden_layers = 5
padding_idx = 0
type_vocab_size = 16
vocab_size = 1024
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/bert-test"
revision = "main"
[initialize.components.transformer.piece_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/bert-test"
revision = "main"
""",

["--model-name", "explosion-testing/bert-test", "--model-revision", "main"],
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
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/camembert-test"
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
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/camembert-test"
[initialize.components.transformer.piece_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/camembert-test"
revision = "main"
""",

[],
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
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/roberta-test"
[initialize.components.transformer.piece_loader]
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
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/roberta-test"
revision = "main"
[initialize.components.transformer.piece_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/roberta-test"
revision = "main"
""",

["--model-name", "explosion-testing/roberta-test", "--model-revision", "main"],
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
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
[initialize.components.transformer.piece_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
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
[initialize]
[initialize.components]
[initialize.components.transformer]
[initialize.components.transformer.encoder_loader]
@model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
[initialize.components.transformer.piece_loader]
@model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
name = "explosion-testing/xlm-roberta-test"
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

        with open(output_path, "r", encoding="utf8") as f:
            all_lines = f.readlines()
            # Remove all whitespace and compare.
            output_str = "".join(all_lines)
            output_str = re.sub(r"\s*", "", output_str)
            expected_str = re.sub(r"\s*", "", output)
            assert output_str == expected_str
