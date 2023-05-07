from spacy.cli import app
from typer.testing import CliRunner

import spacy_curated_transformers.cli.quantize
import spacy_curated_transformers.cli.debug_pieces


def test_quantize():
    result = CliRunner().invoke(app, ["quantize-transformer", "--help"])
    assert result.exit_code == 0


def test_debug_pieces():
    result = CliRunner().invoke(app, ["debug", "pieces", "--help"])
    assert result.exit_code == 0
