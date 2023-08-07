import pytest
import spacy_curated_transformers.cli.debug_pieces
import spacy_curated_transformers.cli.fill_config_transformer
from spacy.cli import app
from typer.testing import CliRunner

from .util import make_tempdir


def test_debug_pieces():
    result = CliRunner().invoke(app, ["debug", "pieces", "--help"])
    assert result.exit_code == 0


FILL_TRANSFORMER_CONFIG_STRS = [
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
    [initialize]
    [initialize.components]
    [initialize.components.transformer]
    [initialize.components.transformer.encoder_loader]
    @model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
    name = "explosion-testing/albert-test"
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
    [initialize]
    [initialize.components]
    [initialize.components.transformer]
    [initialize.components.transformer.encoder_loader]
    @model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
    name = "explosion-testing/bert-test"
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
    @architectures = "spacy-curated-transformers.RobertaTransformer.v1"
    [initialize]
    [initialize.components]
    [initialize.components.transformer]
    [initialize.components.transformer.encoder_loader]
    @model_loaders = "spacy-curated-transformers.HFTransformerEncoderLoader.v1"
    name = "explosion-testing/roberta-test"
    """,
]


@pytest.mark.slow
@pytest.mark.parametrize("config", FILL_TRANSFORMER_CONFIG_STRS)
def test_fill_config_transformer(config):
    with make_tempdir() as d:
        file_path = d / "test_conf"
        with open(file_path, "w", encoding="utf8") as f:
            f.writelines([config])

        result = CliRunner().invoke(
            app,
            [
                "init",
                "fill-config-transformer",
                str(file_path),
                "-",
            ],
        )
        assert result.exit_code == 0
