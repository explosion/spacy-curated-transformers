from typing import List, Optional, TypeVar, Generic
from dataclasses import dataclass
import torch
from torch import Tensor

from thinc.types import Floats2d, Ragged

from ..errors import Errors

TrfOutputT = TypeVar("TrfOutputT", Floats2d, Ragged)


@dataclass
class TransformerModelOutput(Generic[TrfOutputT]):
    """Wrapper for PyTorchTransformerOutput consumed by downstream non-PyTorch components.
    Also acts as the accumulator for the outputs of subsequent models in the Transformer pipeline.
    """

    # Non-padded, un-stacked versions of the outputs.
    # The outer list tracks Docs and the inner list
    # tracks the embedding + hidden layer outputs of each Doc.
    #
    # The inner-most element is a Floats2d when returned by
    # the PyTorchWrapper transformer model, which are subsequently
    # converted to Ragged by the models that follow.
    all_outputs: List[List[TrfOutputT]]

    # Set to True if only the last hidden layer's outputs are preserved.
    last_layer_only: bool

    def __init__(
        self, *, outputs: List[List[TrfOutputT]], last_layer_only: bool
    ) -> None:
        self.all_outputs = outputs
        self.last_layer_only = last_layer_only

    @property
    def embedding_layers(self) -> List[TrfOutputT]:
        if self.last_layer_only:
            return []
        else:
            return [y[0] for y in self.all_outputs]

    @property
    def last_hidden_layer_states(self) -> List[TrfOutputT]:
        return [y[-1] for y in self.all_outputs]

    @property
    def all_hidden_layer_states(self) -> List[List[TrfOutputT]]:
        return [y[1:] for y in self.all_outputs]

    @property
    def num_outputs(self) -> int:
        return len(self.all_outputs[0])


@dataclass
class DocTransformerOutput:
    """Stored on Doc instances. Each Ragged element corresponds to a layer in
    original TransformerModelOutput, containing representations of piece identifiers."""

    all_outputs: List[Ragged]

    # Set to True if only the last hidden layer's outputs are preserved.
    last_layer_only: bool

    def __init__(self, *, all_outputs: List[Ragged], last_layer_only: bool) -> None:
        self.all_outputs = all_outputs
        self.last_layer_only = last_layer_only

    @property
    def embedding_layer(self) -> Optional[Ragged]:
        if self.last_layer_only:
            return None
        else:
            return self.all_outputs[0]

    @property
    def last_hidden_layer_state(self) -> Ragged:
        return self.all_outputs[-1]

    @property
    def all_hidden_layer_states(self) -> List[Ragged]:
        return self.all_outputs[1:]

    @property
    def num_outputs(self) -> int:
        return len(self.all_outputs)
