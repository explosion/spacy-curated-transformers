from typing import Callable, List, Optional

from curated_transformers.models import FromHFHub
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
        encoder = model.shims[0]._model
        assert isinstance(encoder, FromHFHub)
        device = model.shims[0].device
        from_hf_hub = encoder.from_hf_hub

        # We can discard the previously initialized model entirely
        # and use the Curated Transformers API to load it from the
        # hub.
        model.shims[0]._model = None
        del encoder
        encoder = from_hf_hub(name=name, revision=revision, device=device)
        model.shims[0]._model = encoder
        return model

    return load
