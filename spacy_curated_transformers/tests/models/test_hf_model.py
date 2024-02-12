from dataclasses import dataclass
from typing import Callable

import pytest
from curated_transformers.layers import AttentionMask
from curated_transformers.models import (
    ALBERTConfig,
    ALBERTEncoder,
    BERTConfig,
    BERTEncoder,
    RoBERTaConfig,
    RoBERTaEncoder,
)
from thinc.api import get_torch_default_device
from torch.nn import Module

from spacy_curated_transformers._compat import has_hf_transformers, transformers
from spacy_curated_transformers.models.architectures import _pytorch_encoder
from spacy_curated_transformers.models.hf_loader import (
    build_hf_transformer_encoder_loader_v1,
)

from ..util import torch_assertclose


@dataclass
class ModelConfig:
    config: BERTConfig
    encoder: Callable[[BERTConfig], Module]
    hf_model_name: str
    hidden_width: int
    max_seq_len: int
    padding_idx: int


TEST_MODELS = [
    ModelConfig(
        ALBERTConfig(n_pieces=30000),
        ALBERTEncoder,
        "albert-base-v2",
        hidden_width=768,
        max_seq_len=512,
        padding_idx=0,
    ),
    ModelConfig(
        BERTConfig(n_pieces=28996),
        BERTEncoder,
        "bert-base-cased",
        hidden_width=768,
        max_seq_len=512,
        padding_idx=0,
    ),
    ModelConfig(
        RoBERTaConfig(),
        RoBERTaEncoder,
        "roberta-base",
        hidden_width=768,
        max_seq_len=512,
        padding_idx=1,
    ),
    ModelConfig(
        RoBERTaConfig(n_pieces=250002),
        RoBERTaEncoder,
        "xlm-roberta-base",
        hidden_width=768,
        max_seq_len=512,
        padding_idx=1,
    ),
]


def encoder_from_config(config: ModelConfig):
    encoder = config.encoder(config.config)
    model = _pytorch_encoder(
        encoder, config.hidden_width, config.padding_idx, config.max_seq_len
    )
    model.init = build_hf_transformer_encoder_loader_v1(name=config.hf_model_name)
    return model


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model_config", TEST_MODELS)
def test_hf_load_weights(model_config):
    model = encoder_from_config(model_config)
    assert model


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("model_config", TEST_MODELS)
def test_model_against_hf_transformers(model_config):
    torch_device = get_torch_default_device()

    model = encoder_from_config(model_config)
    model.initialize()
    encoder = model.shims[0]._model
    encoder.eval()
    hf_encoder = transformers.AutoModel.from_pretrained(model_config.hf_model_name).to(
        torch_device
    )
    hf_encoder.eval()

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.hf_model_name
    )
    tokenization = hf_tokenizer(
        ["This is a test.", "Let's match outputs"], padding=True, return_tensors="pt"
    ).to(torch_device)
    X = tokenization["input_ids"]
    attention_mask = tokenization["attention_mask"]

    # Test with the tokenizer's attention mask
    Y_encoder = encoder(
        X, attention_mask=AttentionMask(bool_mask=attention_mask.bool())
    )
    Y_hf_encoder = hf_encoder(X, attention_mask=attention_mask)

    torch_assertclose(
        Y_encoder.last_hidden_layer_state,
        Y_hf_encoder.last_hidden_state,
    )
