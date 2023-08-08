import json
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional

from spacy import util
from spacy.cli._util import import_code, init_cli, parse_config_overrides
from spacy.cli.init_config import save_config
from typer import Argument as Arg
from typer import Context
from typer import Option as Opt
from wasabi import Printer

from .._compat import has_hf_transformers, has_huggingface_hub


@init_cli.command(
    "fill-config-transformer",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def init_fill_config_transformer_cli(
    # fmt: off
    ctx: Context,  # This is only used to read additional arguments
    base_path: Path = Arg(..., help="Path to the base config file to fill", exists=True, allow_dash=True, dir_okay=False),
    output_path: Path = Arg(..., help="Path to output .cfg file (or - for stdout)", exists=False, allow_dash=True),
    code_path: Optional[Path] = Opt(None, "--code-path", "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    transformer_name: Optional[str] = Opt(None, "--name", "-n", help="Name of the transformer pipe whose config is to be filled (default: first transformer pipe)."),
    # fmt: on
):
    """
    Fetches the hyperparameters of a Curated Transformer model
    from the Hugging Face Hub and fills in the config with them.

    DOCS: https://spacy.io/api/cli#fill-config-transformer
    """

    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    init_fill_config_transformer(
        base_path,
        output_path,
        config_overrides=overrides,
        transformer_name=transformer_name,
    )


CURATED_TRANSFORMER_TO_HF_MODEL_TYPE: Dict[str, str] = {
    "AlbertTransformer": "albert",
    "BertTransformer": "bert",
    "CamembertTransformer": "camembert",
    "RobertaTransformer": "roberta",
    "XlmrTransformer": "xlm-roberta",
}


class HfParamSource(IntEnum):
    MODEL_CONFIG = (1,)
    TOKENIZER_CONFIG = (2,)


# Entrypoint parameters that are common to all curated transformer models.
COMMON_ENTRYPOINT_PARAMS: Dict[str, HfParamSource] = {
    "attention_probs_dropout_prob": HfParamSource.MODEL_CONFIG,
    "hidden_act": HfParamSource.MODEL_CONFIG,
    "hidden_dropout_prob": HfParamSource.MODEL_CONFIG,
    "hidden_width": HfParamSource.MODEL_CONFIG,
    "intermediate_width": HfParamSource.MODEL_CONFIG,
    "layer_norm_eps": HfParamSource.MODEL_CONFIG,
    "max_position_embeddings": HfParamSource.MODEL_CONFIG,
    "model_max_length": HfParamSource.MODEL_CONFIG,
    "num_attention_heads": HfParamSource.MODEL_CONFIG,
    "num_hidden_layers": HfParamSource.MODEL_CONFIG,
    "padding_idx": HfParamSource.MODEL_CONFIG,
    "type_vocab_size": HfParamSource.MODEL_CONFIG,
    "vocab_size": HfParamSource.MODEL_CONFIG,
    "model_max_length": HfParamSource.TOKENIZER_CONFIG,
}

MODEL_SPECIFIC_ENTRYPOINT_PARAMS: Dict[str, Dict[str, HfParamSource]] = {
    "albert": {
        "embedding_width": HfParamSource.MODEL_CONFIG,
        "num_hidden_groups": HfParamSource.MODEL_CONFIG,
    }
}

ENTRYPOINT_PARAMS_TO_HF_CONFIG_KEYS: Dict[str, str] = {
    "hidden_width": "hidden_size",
    "intermediate_width": "intermediate_size",
    "padding_idx": "pad_token_id",
    "embedding_width": "embedding_size",
}


def init_fill_config_transformer(
    config_path: Path,
    output_path: Path,
    *,
    config_overrides: Dict[str, Any] = {},
    transformer_name: Optional[str] = None,
):
    msg = Printer()
    if not has_huggingface_hub or not has_hf_transformers:
        msg.fail(
            "This command requires the `huggingface-hub` and `transformers` packages to be installed",
            exits=1,
        )
    config = util.load_config(config_path, overrides=config_overrides)
    if transformer_name is None:
        components = config["components"]
        transformers = [
            name
            for name, config in components.items()
            if config["factory"] == "curated_transformer"
        ]
        if not transformers:
            msg.fail(
                "Pipeline config does not contain a Curated Transformer component",
                exits=1,
            )
        transformer_name = transformers[0]
    msg.info(f"Updating config for Curated Transformer pipe '{transformer_name}'")

    loader_config = _get_encoder_loader_config(config, transformer_name)
    if loader_config is None or loader_config.get("name") is None:  # type: ignore
        msg.fail(
            "Pipeline config does not have a valid model loader configuration "
            f"for the '{transformer_name}' Curated Transformer component",
            exits=1,
        )
    assert loader_config is not None

    model_name = loader_config.get("name")
    model_revision = loader_config.get("revision", "main")
    assert model_name is not None and model_revision is not None

    msg.info(f"Hugging Face model name: '{model_name}' | Revision: '{model_revision}'")

    trf_arch = _get_curated_transformer_arch(config, transformer_name)
    if trf_arch is None:
        msg.fail(
            f"Pipeline config does not have a valid model architecture for the '{transformer_name}' Curated Transformer component",
            exits=1,
        )
    assert trf_arch is not None

    hf_model_type = CURATED_TRANSFORMER_TO_HF_MODEL_TYPE.get(trf_arch)
    if hf_model_type is None:
        msg.fail(
            f"Missing Hugging Face model type for Curated Transformer model architecture '{trf_arch}'",
            exits=1,
        )
    assert hf_model_type is not None

    hf_config = _get_hf_model_config(model_name, model_revision)
    incoming_hf_model_type = hf_config["model_type"]
    if incoming_hf_model_type != hf_model_type:
        msg.fail(
            f"Hugging Face model of type '{incoming_hf_model_type}' cannot be loaded into "
            f"Curated Transformer pipe '{transformer_name}' due to mismatching architectures",
            exits=1,
        )

    params_to_fill = COMMON_ENTRYPOINT_PARAMS.copy()
    params_to_fill.update(MODEL_SPECIFIC_ENTRYPOINT_PARAMS.get(hf_model_type, {}))
    hf_tokenizer = _load_hf_tokenizer(model_name, model_revision, msg)

    filled_params = {}
    for param_name, source in params_to_fill.items():
        hf_key = ENTRYPOINT_PARAMS_TO_HF_CONFIG_KEYS.get(param_name, param_name)
        if source == HfParamSource.MODEL_CONFIG:
            value = hf_config.get(hf_key)
            if value is None:
                msg.fail(
                    f"Hugging Face model config has a missing key '{hf_key}'", exits=1
                )
        elif source == HfParamSource.TOKENIZER_CONFIG:
            value = getattr(hf_tokenizer, hf_key, None)
            if value is None:
                msg.fail(
                    f"Hugging Face tokenizer config has a missing key '{hf_key}'",
                    exits=1,
                )
        filled_params[param_name] = value

    msg.info(title="Filled-in model parameters")
    msg.table(filled_params)

    trf_config = config["components"][transformer_name]["model"]
    trf_config.update(filled_params)

    is_stdout = str(output_path) == "-"
    save_config(config, output_path, is_stdout=is_stdout, silent=is_stdout)


def _get_hf_model_config(name: str, revision: str) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=name, filename="config.json", revision=revision
    )
    return _parse_json_file(config_path)


def _parse_json_file(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_encoder_loader_config(
    config: Dict[str, Any], transformer_name: str
) -> Optional[Dict[str, Any]]:
    try:
        loader = config["initialize"]["components"][transformer_name]["encoder_loader"]
        assert "HFTransformerEncoderLoader" in loader["@model_loaders"]
        return loader
    except (KeyError, AssertionError):
        return None


def _get_curated_transformer_arch(
    config: Dict[str, Any], transformer_name: str
) -> Optional[str]:
    try:
        transformer = config["components"][transformer_name]["model"]
        trf_arch_splits = transformer["@architectures"].split(".")
        assert len(trf_arch_splits) == 3 and trf_arch_splits[-2].endswith("Transformer")
        return trf_arch_splits[-2]
    except (KeyError, AssertionError):
        return None


def _load_hf_tokenizer(model_name: str, model_revision: str, msg: Printer) -> Any:
    # The entry point params that we need to fetch from the upstream HF tokenizer
    # are not consistently serialized across all available models. So, we'll just
    # load the tokenizer using the `transformers` library and directly read the params
    # from the instance.
    from transformers import AutoTokenizer

    try:
        hf_tokenzier = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )

        # The `model_max_length` value can sometimes be set to a very large integer
        # if the model doesn't have an upperbound on its input length. Saving this
        # as-is to the config will raise an error during serialization. So, we'll
        # try to look up the correct value manually. If we don't find one, fallback
        # to a smaller (large) value that can be serialized correctly.
        # cf: https://github.com/huggingface/transformers/blob/224da5df6956340a5c680e5b57b4914d4d7298b6/src/transformers/tokenization_utils_base.py#L104
        if hf_tokenzier.model_max_length == int(1e30):
            hf_tokenzier.model_max_length = 9999
            if getattr(hf_tokenzier, "max_model_input_sizes", None) is not None:
                model_name_splits = model_name.split("/")
                if len(model_name_splits) != 0:
                    model_name = model_name_splits[-1]
                model_max_length = hf_tokenzier.max_model_input_sizes.get(model_name)
                if model_max_length is not None:
                    hf_tokenzier.model_max_length = model_max_length
    except BaseException as e:
        msg.fail(
            f"Couldn't load Hugging Face tokenizer '{model_name}' ('{model_revision}') - Error:\n{e}",
            exits=1,
        )

    return hf_tokenzier
