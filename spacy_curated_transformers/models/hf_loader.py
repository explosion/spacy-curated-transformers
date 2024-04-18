from typing import Callable, List, Optional

from curated_transformers.models import FromHF
from spacy.tokens import Doc

from .types import TorchTransformerModelT


def build_hf_transformer_encoder_loader_v1(
    *,
    name: str,
    revision: str = "main",
) -> Callable[
    [TorchTransformerModelT, Optional[List[Doc]], Optional[List[Doc]]],
    TorchTransformerModelT,
]:
    """Construct a callback that initializes a supported transformer
    model with weights from a corresponding HuggingFace model.

    name (str):
        Name of the HuggingFace model.
    revision (str):
        Name of the model revision/branch.
    """

    def load(model, X=None, Y=None):
        encoder = model.shims[0]._model.curated_encoder
        assert isinstance(encoder, FromHF)
        device = model.shims[0].device
        encoder.from_hf_hub_(name=name, revision=revision, device=device)
        return model

    return load
