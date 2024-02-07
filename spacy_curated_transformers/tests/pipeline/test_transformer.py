import multiprocessing
from functools import partial
from typing import Any, Dict

import numpy
import pytest
import spacy
import torch
from spacy import Config, util
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.util import registry as spacy_registry
from thinc.api import CupyOps, get_current_ops
from thinc.backends import get_array_ops
from thinc.model import Model

from spacy_curated_transformers._compat import has_hf_transformers, transformers
from spacy_curated_transformers.models.architectures import (
    build_bert_transformer_model_v1,
    build_camembert_transformer_model_v1,
    build_roberta_transformer_model_v1,
    build_xlmr_transformer_model_v1,
)
from spacy_curated_transformers.models.hf_loader import (
    build_hf_transformer_encoder_loader_v1,
)
from spacy_curated_transformers.models.listeners import (
    ListenerStateUtils,
    WrappedTransformerAndListener,
)
from spacy_curated_transformers.models.with_strided_spans import (
    build_with_strided_spans_v1,
)
from spacy_curated_transformers.pipeline.transformer import (
    DEFAULT_CONFIG,
    CuratedTransformer,
    make_transformer,
)
from spacy_curated_transformers.tokenization import (
    build_bert_wordpiece_encoder_v1,
    build_byte_bpe_encoder_v1,
    build_camembert_sentencepiece_encoder_v1,
    build_hf_piece_encoder_loader_v1,
    build_xlmr_sentencepiece_encoder_v1,
)
from spacy_curated_transformers.tokenization.sentencepiece_encoder import (
    build_sentencepiece_encoder_loader_v1,
)
from spacy_curated_transformers.util import create_gradual_transformer_unfreezing

from ..util import make_tempdir, torch_assertclose, xp_assert_array_equal

# Torch currently interacts badly with the fork method:
# https://github.com/pytorch/pytorch/issues/17199
multiprocessing.set_start_method("spawn")

cfg_string_last_layer_listener = """
    # LastTransformerLayerListener

    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v2"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-curated-transformers.LastTransformerLayerListener.v1"
    width = ${components.transformer.model.hidden_width}
    pooling = {"@layers":"reduce_mean.v1"}

    [components.transformer]
    factory = "curated_transformer"
    all_layer_outputs = False

    [components.transformer.model]
    @architectures = "spacy-curated-transformers.BertTransformer.v1"
    vocab_size = 28996
    num_hidden_layers = 1
    hidden_width = 60
    piece_encoder = {"@architectures":"spacy-curated-transformers.BertWordpieceEncoder.v1"}
    with_spans = {"@architectures":"spacy-curated-transformers.WithStridedSpans.v1"}

    [initialize]

    [initialize.components]

    [initialize.components.transformer]

    [initialize.components.transformer.piecer_loader]
    @model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
    name = "bert-base-cased"
"""

cfg_string_scalar_weighting_layer_listener = """
    # ScalarWeightingListener

    [nlp]
    lang = "en"
    pipeline = ["transformer","tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v2"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-curated-transformers.ScalarWeightingListener.v1"
    width = ${components.transformer.model.hidden_width}
    pooling = {"@layers":"reduce_mean.v1"}

    [components.tagger.model.tok2vec.weighting]
    @architectures = "spacy-curated-transformers.ScalarWeight.v1"
    num_layers = ${components.transformer.model.num_hidden_layers}

    [components.transformer]
    factory = "curated_transformer"
    all_layer_outputs = True

    [components.transformer.model]
    @architectures = "spacy-curated-transformers.BertTransformer.v1"
    vocab_size = 28996
    num_hidden_layers = 1
    hidden_width = 60
    piece_encoder = {"@architectures":"spacy-curated-transformers.BertWordpieceEncoder.v1"}
    with_spans = {"@architectures":"spacy-curated-transformers.WithStridedSpans.v1"}

    [initialize]

    [initialize.components]

    [initialize.components.transformer]

    [initialize.components.transformer.piecer_loader]
    @model_loaders = "spacy-curated-transformers.HFPieceEncoderLoader.v1"
    name = "bert-base-cased"
"""

TRAIN_DATA = [
    (
        "I like green eggs",
        {"tags": ["N", "V", "J", "N"], "cats": {"preference": 1.0, "imperative": 0.0}},
    ),
    (
        "Eat blue ham",
        {"tags": ["V", "J", "N"], "cats": {"preference": 0.0, "imperative": 1.0}},
    ),
]


def create_tagger(cfg_string):
    config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)

    tagger = nlp.get_pipe("tagger")

    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    nlp.initialize(lambda: train_examples)

    return nlp


def create_and_train_tagger(cfg_string):
    nlp = create_tagger(cfg_string)

    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.create_optimizer()

    for _ in range(10):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    return nlp


def evaluate_tagger_on_train_data(model):
    docs = list(model.pipe(["Eat blue ham", "I like green eggs"]))
    assert [t.tag_ for t in docs[0]] == ["V", "J", "N"]
    assert [t.tag_ for t in docs[1]] == ["N", "V", "J", "N"]


def test_default_pipe_config_can_be_constructed():
    nlp = spacy.blank("en")
    nlp.add_pipe("curated_transformer", config={"model": {"vocab_size": 32}})


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize(
    "cfg_string",
    [cfg_string_last_layer_listener, cfg_string_scalar_weighting_layer_listener],
)
def test_tagger(cfg_string):
    model = create_and_train_tagger(cfg_string)
    evaluate_tagger_on_train_data(model)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize(
    "cfg_string",
    [cfg_string_last_layer_listener, cfg_string_scalar_weighting_layer_listener],
)
@pytest.mark.skipif(
    isinstance(get_current_ops(), CupyOps),
    reason="multiprocessing and GPU support are incompatible",
)
def test_tagger_multiprocessing(cfg_string):
    model = create_tagger(cfg_string)
    for _ in model.pipe(["This is a test..."] * 100, n_process=2):
        pass


def _hf_tokenize_per_token(tokenizer, docs, *, roberta=False):
    if roberta:
        hf_encoding = [
            tokenizer(
                [
                    doc[idx - 1].whitespace_ + token.text if idx > 0 else token.text
                    for idx, token in enumerate(doc)
                ]
            )
            for doc in docs
        ]
    else:
        hf_encoding = [tokenizer([token.text for token in doc]) for doc in docs]
    ids = []
    lens = []
    bos_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.cls_token_id
    )
    eos_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.sep_token_id
    )
    for i in range(len(hf_encoding)):
        doc_ids = [id for e in hf_encoding[i]["input_ids"] for id in e[1:-1]]
        ids.append([bos_id] + doc_ids + [eos_id])
        lens.append(len(ids[-1]))

    torch_ids = torch.full(
        (len(ids), max(lens)), tokenizer.pad_token_id, dtype=torch.int
    )
    for i in range(len(ids)):
        torch_ids[i][: len(ids[i])] = torch.tensor(ids[i])

    attention_mask = torch_ids.ne(tokenizer.pad_token_id)

    return torch_ids, attention_mask, lens


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_bert_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_bert_transformer_model_v1(
        piece_encoder=build_bert_wordpiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=28996,
    )
    model.get_ref("transformer").init = build_hf_transformer_encoder_loader_v1(
        name="bert-base-cased"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="bert-base-cased"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    hf_model = transformers.AutoModel.from_pretrained("bert-base-cased")

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(hf_tokenizer, docs)
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch_assertclose(
            hf_doc_encoding[:encoding_len][1:-1],
            torch.tensor(doc._.trf_data.last_hidden_layer_state.dataXd),
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_camembert_transformer_pipe_against_hf():
    nlp = spacy.blank("fr")
    model = build_camembert_transformer_model_v1(
        piece_encoder=build_camembert_sentencepiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=32005,
    )
    model.get_ref("transformer").init = build_hf_transformer_encoder_loader_v1(
        name="camembert-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="camembert-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("camembert-base")
    hf_model = transformers.AutoModel.from_pretrained("camembert-base")

    docs = [
        nlp.make_doc("J'ai vu une fille avec un télescope."),
        nlp.make_doc("Aujourd'hui, nous allons manger un poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(hf_tokenizer, docs)
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch_assertclose(
            hf_doc_encoding[:encoding_len][1:-1],
            torch.tensor(doc._.trf_data.last_hidden_layer_state.dataXd),
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_roberta_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_roberta_transformer_model_v1(
        piece_encoder=build_byte_bpe_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=50265,
    )
    model.get_ref("transformer").init = build_hf_transformer_encoder_loader_v1(
        name="roberta-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="roberta-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    hf_model = transformers.AutoModel.from_pretrained("roberta-base")

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(
        hf_tokenizer, docs, roberta=True
    )
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch_assertclose(
            hf_doc_encoding[:encoding_len][1:-1],
            torch.tensor(doc._.trf_data.last_hidden_layer_state.dataXd),
        )


def test_empty_input(sentencepiece_toy_model_path):
    nlp = spacy.blank("en")
    model = build_xlmr_transformer_model_v1(
        piece_encoder=build_xlmr_sentencepiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        num_hidden_layers=2,
        vocab_size=1000,
        hidden_width=12,
    )

    model.get_ref("piece_encoder").init = build_sentencepiece_encoder_loader_v1(
        path=sentencepiece_toy_model_path
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)
    pipe(nlp.make_doc(""))


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_xlmr_transformer_pipe_against_hf():
    nlp = spacy.blank("en")
    model = build_xlmr_transformer_model_v1(
        piece_encoder=build_xlmr_sentencepiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=250002,
    )
    model.get_ref("transformer").init = build_hf_transformer_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model)

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
    hf_model = transformers.AutoModel.from_pretrained("xlm-roberta-base")

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]

    hf_ids, attention_mask, lens = _hf_tokenize_per_token(hf_tokenizer, docs)
    hf_encoding = hf_model(hf_ids, attention_mask=attention_mask)
    docs = list(pipe.pipe(docs))

    for doc, hf_doc_encoding, encoding_len in zip(
        docs, hf_encoding.last_hidden_state, lens
    ):
        torch_assertclose(
            hf_doc_encoding[:encoding_len][1:-1],
            torch.tensor(doc._.trf_data.last_hidden_layer_state.dataXd),
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_frozen_transformer_pipe():
    config = Config().from_str(cfg_string_scalar_weighting_layer_listener)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    tagger = nlp.get_pipe("tagger")
    transformer = nlp.get_pipe("transformer")
    transformer.frozen = True

    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["tags"]:
            tagger.add_label(tag)

    optimizer = nlp.initialize(lambda: train_examples)

    def get_transformer_params_sorted():
        params = transformer.model.get_ref("transformer").shims[0]._model.state_dict()
        return list(sorted(params.items()))

    transformer_init_params = [
        (k, v.clone()) for k, v in get_transformer_params_sorted()
    ]

    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    transformer_trained_params = get_transformer_params_sorted()
    for (old_param, old_vec), (new_param, new_vec) in zip(
        transformer_init_params, transformer_trained_params
    ):
        assert old_param == new_param
        torch_assertclose(
            old_vec,
            new_vec,
        )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_transformer_pipe_outputs():
    nlp = spacy.blank("en")
    model = build_xlmr_transformer_model_v1(
        piece_encoder=build_xlmr_sentencepiece_encoder_v1(),
        with_spans=build_with_strided_spans_v1(),
        vocab_size=250002,
    )
    model.get_ref("transformer").init = build_hf_transformer_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.get_ref("piece_encoder").init = build_hf_piece_encoder_loader_v1(
        name="xlm-roberta-base"
    )
    model.initialize()
    pipe = make_transformer(nlp, "transformer", model, all_layer_outputs=False)

    docs = [
        nlp.make_doc("I saw a girl with a telescope."),
        nlp.make_doc("Today we will eat poké bowl."),
    ]
    docs = list(pipe.pipe(docs))
    assert all([doc._.trf_data.last_layer_only for doc in docs]) == True
    assert all([len(doc._.trf_data.all_outputs) == 1 for doc in docs]) == True

    serialized = [doc.to_bytes() for doc in docs]
    deserialized = [Doc(nlp.vocab).from_bytes(doc_bytes) for doc_bytes in serialized]
    for doc, doc_deserialized in zip(docs, deserialized):
        _assert_doc_model_output_equal(doc, doc_deserialized)

    pipe = make_transformer(nlp, "transformer", model, all_layer_outputs=True)
    docs = list(pipe.pipe(docs))
    assert all([not doc._.trf_data.last_layer_only for doc in docs]) == True
    assert all([len(doc._.trf_data.all_outputs) == 12 + 1 for doc in docs]) == True

    serialized = [doc.to_bytes() for doc in docs]
    deserialized = [Doc(nlp.vocab).from_bytes(doc_bytes) for doc_bytes in serialized]
    for doc, doc_deserialized in zip(docs, deserialized):
        _assert_doc_model_output_equal(doc, doc_deserialized)


cfg_string_gradual_unfreezing = (
    cfg_string_last_layer_listener
    + """
    [corpora]

    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = "toy-en-corpus.spacy"
    max_length = 500
    gold_preproc = false
    limit = 0

    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = "toy-en-corpus.spacy"
    max_length = 500
    gold_preproc = false
    limit = 0

    [training]
    train_corpus = "corpora.train"
    dev_corpus = "corpora.dev"
    seed = 1
    gpu_allocator = "pytorch"
    dropout = 0.1
    accumulate_gradient = 3
    patience = 5000
    max_epochs = 1
    max_steps = 6
    eval_frequency = 10

    [training.batcher]
    @batchers = "spacy.batch_by_padded.v1"
    discard_oversize = False
    get_length = null
    size = 1
    buffer = 256
"""
)


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
def test_gradual_transformer_unfreezing(test_dir):
    def wrapped_callback():
        def inner(
            nlp: Language,
            args: Dict[str, Any],
            gradual_unfreezing_callback,
            unfreeze_step,
        ):
            current_step = args["step"]
            gradual_unfreezing_callback(nlp, args)

            transformer = nlp.get_pipe("transformer")
            if current_step < unfreeze_step:
                assert transformer.frozen == True
            else:
                assert transformer.frozen == False

        return partial(
            inner,
            gradual_unfreezing_callback=create_gradual_transformer_unfreezing({"*": 3}),
            unfreeze_step=3,
        )

    spacy_registry.callbacks.register(
        "test_gradual_unfreezing_callback",
        func=wrapped_callback,
    )

    config = Config().from_str(cfg_string_gradual_unfreezing, interpolate=False)
    config["corpora"]["train"]["path"] = str(test_dir / "toy-en-corpus.spacy")
    config["corpora"]["dev"]["path"] = str(test_dir / "toy-en-corpus.spacy")
    config["training"]["before_update"] = {
        "@callbacks": "test_gradual_unfreezing_callback"
    }
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)

    nlp = init_nlp(config)
    train(nlp)
    assert nlp.get_pipe("transformer").frozen == False

    with pytest.raises(ValueError):
        create_gradual_transformer_unfreezing({"transformer": 5, "*": 4})


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize(
    ["cfg_string", "listener_name", "listener_entrypoint"],
    [
        (
            cfg_string_last_layer_listener,
            "last_transformer_layer_listener",
            "spacy-curated-transformers.LastTransformerLayerListener.v1",
        ),
        (
            cfg_string_scalar_weighting_layer_listener,
            "scalar_weighting_listener",
            "spacy-curated-transformers.ScalarWeightingListener.v1",
        ),
    ],
)
def test_replace_listeners(cfg_string, listener_name, listener_entrypoint):
    orig_config = Config().from_str(cfg_string)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    text = "This is awesome"
    examples = [Example.from_dict(nlp.make_doc(text), {"tags": ["A", "B", "C"]})]
    optimizer = nlp.initialize(lambda: examples)

    # verify correct configuration with transformer listener
    transformer = nlp.get_pipe("transformer")
    tagger = nlp.get_pipe("tagger")

    tagger_tok2vec = tagger.model.get_ref("tok2vec")

    assert ListenerStateUtils.is_listener(tagger_tok2vec)
    assert transformer.listener_map["tagger"][0] == tagger_tok2vec
    assert isinstance(transformer.model, Model)
    assert transformer.model.name == "transformer_model"
    assert (
        nlp.config["components"]["transformer"]["model"]["@architectures"]
        == "spacy-curated-transformers.BertTransformer.v1"
    )
    assert (
        nlp.config["components"]["tagger"]["model"]["tok2vec"]["@architectures"]
        == listener_entrypoint
    )

    # train pipe before replacing listeners
    for i in range(5):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        doc = nlp(text)

    preds = [t.tag_ for t in doc]
    doc_tensor = tagger_tok2vec.predict([doc])

    # replace listener and verify predictions are still the same
    transformer.frozen = True
    nlp.replace_listeners("transformer", "tagger", ["model.tok2vec"])
    tagger = nlp.get_pipe("tagger")
    tagger_tok2vec = tagger.model.get_ref("tok2vec")
    assert isinstance(tagger_tok2vec, WrappedTransformerAndListener)
    assert tagger_tok2vec.frozen_transformer
    assert tagger_tok2vec.layers[0].name == "transformer_model"
    assert tagger_tok2vec.layers[1].name == listener_name
    assert (
        nlp.config["components"]["tagger"]["model"]["tok2vec"]["@architectures"]
        == "spacy-curated-transformers.BertTransformer.v1"
    )
    assert (
        nlp.config["components"]["tagger"]["model"]["tok2vec"]["wrapped_listener"][
            "@architectures"
        ]
        == listener_entrypoint
    )
    doc2 = nlp(text)
    assert preds == [t.tag_ for t in doc2]
    pred_tensor = tagger_tok2vec.predict([doc2])
    xp_assert_array_equal(doc_tensor, pred_tensor)

    optimizer = nlp.resume_training()
    trf_output_frozen = tagger_tok2vec.layers[0].predict([doc2])
    for i in range(5):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        assert losses["tagger"] > 0.0
    trf_output_frozen_after_update = tagger_tok2vec.layers[0].predict([doc2])

    for x, y in zip(
        trf_output_frozen.all_outputs, trf_output_frozen_after_update.all_outputs
    ):
        for x1, y1 in zip(x, y):
            xp_assert_array_equal(x1.dataXd, y1.dataXd)

    tagger_tok2vec.frozen_transformer = False
    trf_output = tagger_tok2vec.layers[0].predict([doc2])
    # attempt training with the new pipeline
    for i in range(5):
        losses = {}
        nlp.update(examples, sgd=optimizer, losses=losses)
        assert losses["tagger"] > 0.0
    trained_trf_output = tagger_tok2vec.layers[0].predict([doc2])

    for x, y in zip(trf_output.all_outputs, trained_trf_output.all_outputs):
        for x1, y1 in zip(x, y):
            assert not numpy.array_equal(x1.dataXd, y1.dataXd)

    # ensure IO goes OK
    doc_tensor_trained = tagger_tok2vec.predict([doc])

    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)
        nlp2 = util.load_model_from_path(file_path)
        doc3 = nlp2(text)
        tagger2 = nlp2.get_pipe("tagger")
        tagger_tok2vec2 = tagger2.model.get_ref("tok2vec")
        pred_tensor = tagger_tok2vec2.predict([doc3])
        xp_assert_array_equal(doc_tensor_trained, pred_tensor)


def test_transformer_add_pipe():
    config = Config().from_str(cfg_string_last_layer_listener)
    nlp = util.load_model_from_config(config, auto_fill=True, validate=True)
    nlp.remove_pipe("transformer")

    nlp = util.load_model_from_config(
        Config().from_str(nlp.config.to_str()), auto_fill=True, validate=True
    )
    transformer = nlp.add_pipe("curated_transformer")
    assert isinstance(transformer, CuratedTransformer)
    assert (
        nlp.config["components"]["curated_transformer"]["model"]["vocab_size"]
        == DEFAULT_CONFIG["transformer"]["model"]["vocab_size"]
    )
    assert (
        nlp.config["components"]["curated_transformer"]["model"]["@architectures"]
        == DEFAULT_CONFIG["transformer"]["model"]["@architectures"]
    )
    assert (
        nlp.config["components"]["curated_transformer"]["model"]["piece_encoder"][
            "@architectures"
        ]
        == DEFAULT_CONFIG["transformer"]["model"]["piece_encoder"]["@architectures"]
    )
    assert (
        nlp.config["components"]["curated_transformer"]["model"]["with_spans"][
            "@architectures"
        ]
        == DEFAULT_CONFIG["transformer"]["model"]["with_spans"]["@architectures"]
    )


def _assert_doc_model_output_equal(doc1: Doc, doc2: Doc):
    output1 = doc1._.trf_data
    output2 = doc2._.trf_data

    assert output1.last_layer_only == output2.last_layer_only
    assert len(output1.all_outputs) == len(output2.all_outputs)

    for layer1, layer2 in zip(output1.all_outputs, output2.all_outputs):
        ops = get_array_ops(layer1.dataXd)
        ops.xp.testing.assert_allclose(layer1.dataXd, layer2.dataXd)
        ops.xp.testing.assert_array_equal(layer1.lengths, layer2.lengths)
