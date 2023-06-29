import spacy_curated_transformers.cli.debug_pieces
from spacy.cli import app
from typer.testing import CliRunner


def test_debug_pieces():
    result = CliRunner().invoke(app, ["debug", "pieces", "--help"])
    assert result.exit_code == 0
