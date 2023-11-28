import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from confection import Config
from spacy import util
from spacy.cli._util import import_code, init_cli, parse_config_overrides
from spacy.cli.init_config import save_config
from typer import Argument as Arg
from typer import Context
from typer import Option as Opt
from wasabi import Printer

from .._compat import has_hf_transformers, has_huggingface_hub


@init_cli.command(
    "fill-curated-transformer",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": False},
)
def init_fill_curated_transformer_cli(
    # fmt: off
    ctx: Context,  # This is only used to read additional arguments
    base_path: Path = Arg(..., help="Path to the base config file to fill", exists=True, allow_dash=True, dir_okay=False),
    output_path: Path = Arg("-", help="Path to output .cfg file or '-' for stdout (default: stdout)", exists=False, allow_dash=True),
    model_name: Optional[str] = Opt(None, "--model-name", "-m", help="Name of the Hugging Face model. If not provided, the model name will be read in from the encoder loader config"),
    model_revision: Optional[str] = Opt(None, "--model-revision", "-r", help="Revision of the Hugging Face model (default: 'main')."),
    transformer_name: Optional[str] = Opt(None, "--pipe-name", "-n", help="Name of the transformer pipe whose config is to be filled (default: first transformer pipe)."),
    code_path: Optional[Path] = Opt(None, "--code-path", "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    # fmt: on
):
    """
    Fill Curated Transformer parameters from HF Hub.

    DOCS: https://spacy.io/api/cli#init-fill-curated-transformer
    """
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    init_fill_config_curated_transformer(
        base_path,
        output_path,
        cli_model_name=model_name,
        cli_model_revision=model_revision,
        cli_transformer_name=transformer_name,
        config_overrides=overrides,
    )


CURATED_TRANSFORMER_TO_HF_MODEL_TYPE: Dict[str, str] = {
    "AlbertTransformer": "albert",
    "BertTransformer": "bert",
    "CamembertTransformer": "camembert",
    "RobertaTransformer": "roberta",
    "XlmrTransformer": "xlm-roberta",
}

HF_MODEL_TYPE_TO_EXAMPLE_MODELS: Dict[str, str] = {
    "albert": "albert-base-v2",
    "bert": "bert-base-cased",
    "camembert": "camembert/camembert-base",
    "roberta": "roberta-base",
    "xlm-roberta": "xlm-roberta-base",
}


class HfParamSource(Enum):
    MODEL_CONFIG = 1
    TOKENIZER_CONFIG = 2


# Entrypoint parameters that are common to all curated transformer models.
COMMON_ENTRYPOINT_PARAMS: Dict[str, HfParamSource] = {
    "attention_probs_dropout_prob": HfParamSource.MODEL_CONFIG,
    "hidden_act": HfParamSource.MODEL_CONFIG,
    "hidden_dropout_prob": HfParamSource.MODEL_CONFIG,
    "hidden_width": HfParamSource.MODEL_CONFIG,
    "intermediate_width": HfParamSource.MODEL_CONFIG,
    "layer_norm_eps": HfParamSource.MODEL_CONFIG,
    "max_position_embeddings": HfParamSource.MODEL_CONFIG,
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


class ModelSource(Enum):
    CliArgument = 1
    Loader = 2


def init_fill_config_curated_transformer(
    config_path: Path,
    output_path: Path,
    cli_model_name: Optional[str],
    cli_model_revision: Optional[str],
    cli_transformer_name: Optional[str],
    config_overrides: Dict[str, Any],
):
    msg = Printer()
    _validate_hf_packages(msg)

    config = util.load_config(config_path, overrides=config_overrides)

    transformer_name = _resolve_curated_trf_pipe_name(msg, config, cli_transformer_name)
    model_src = _resolve_model_source(msg, cli_model_name, cli_model_revision)
    model_name, model_revision = _resolve_model_name_and_revision(
        msg, config, model_src, cli_model_name, cli_model_revision, transformer_name
    )

    hf_model_type = _lookup_hf_model_type_for_curated_architecture(
        msg, config, transformer_name
    )
    hf_config = _get_hf_model_config(msg, model_name, model_revision)
    _validate_hf_model_type(msg, hf_config, hf_model_type, transformer_name)

    params_to_fill = COMMON_ENTRYPOINT_PARAMS.copy()
    params_to_fill.update(MODEL_SPECIFIC_ENTRYPOINT_PARAMS.get(hf_model_type, {}))
    hf_tokenizer = _load_hf_tokenizer(model_name, model_revision, msg)

    filled_params = _fill_parameters(msg, params_to_fill, hf_config, hf_tokenizer)

    # Update transformer model config with the filled params.
    trf_config = config["components"][transformer_name]["model"]
    trf_config.update(filled_params)

    if model_src == ModelSource.CliArgument:
        # Overwrite the encoder and piecer loader configs.
        _save_encoder_loader_config(
            msg, config, transformer_name, model_name, model_revision
        )
        _save_piecer_loader_config(
            msg, config, transformer_name, model_name, model_revision, overwrite=True
        )
    else:
        # Only fill in the piecer loader config if it's not present.
        _save_piecer_loader_config(
            msg, config, transformer_name, model_name, model_revision, overwrite=False
        )

    is_stdout = str(output_path) == "-"
    if is_stdout:
        msg.info(title="\n\nOutput:")
    save_config(config, output_path, is_stdout=is_stdout, silent=is_stdout)


def _validate_hf_packages(msg: Printer):
    if not has_huggingface_hub or not has_hf_transformers:
        msg.fail(
            "This command requires the `huggingface-hub` and `transformers` packages to be installed",
            exits=1,
        )


def _resolve_model_source(
    msg: Printer,
    cli_model_name: Optional[str],
    cli_model_revision: Optional[str],
) -> ModelSource:
    if cli_model_name is None and cli_model_revision is not None:
        msg.fail(
            "The model revision command-line argument must be accompanied by the model name argument",
            exits=1,
        )

    if cli_model_name is None:
        return ModelSource.Loader
    else:
        return ModelSource.CliArgument


def _resolve_curated_trf_pipe_name(
    msg: Printer, config: Config, cli_transformer_name: Optional[str]
) -> str:
    if cli_transformer_name is None:
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
    else:
        transformer_name = cli_transformer_name
    msg.info(f"Updating config for Curated Transformer pipe '{transformer_name}'")
    return transformer_name


def _resolve_model_name_and_revision(
    msg: Printer,
    config: Config,
    model_src: ModelSource,
    cli_model_name: Optional[str],
    cli_model_revision: Optional[str],
    transformer_name: str,
) -> Tuple[str, str]:
    try:
        loader_config = config["initialize"]["components"][transformer_name][
            "encoder_loader"
        ]
        assert "HFTransformerEncoderLoader" in loader_config["@model_loaders"]
    except (KeyError, AssertionError):
        loader_config = None

    if model_src == ModelSource.CliArgument:
        assert cli_model_name is not None

        model_name = cli_model_name
        model_revision = (
            cli_model_revision if cli_model_revision is not None else "main"
        )
        if cli_model_revision is None:
            msg.warn("Using default model revision")
        msg.info(
            title=f"Using provided Hugging Face model name and {'default ' if cli_model_revision is None else ''}revision:",
            text=f"{model_name} ({model_revision})",
        )
    else:
        msg.info(f"Looking up Hugging Face model name and revision in loader config...")
        if loader_config is None or loader_config.get("name") is None:  # type: ignore
            msg.fail(
                "Pipeline config does not have a valid model loader configuration "
                f"for the '{transformer_name}' Curated Transformer component. You can "
                "provide the '--model-name' and '--model-revision' command-line "
                "arguments to automatically fill in the loader configuration.",
                exits=1,
            )
        assert loader_config is not None

        model_name = loader_config["name"]
        model_revision = loader_config.get("revision", "main")
        msg.info(
            title="Using loader Hugging Face model name and revision:",
            text=f"{model_name} ({model_revision})",
        )

    return model_name, model_revision


def _lookup_hf_model_type_for_curated_architecture(
    msg: Printer, config: Config, transformer_name: str
) -> str:
    try:
        transformer = config["components"][transformer_name]["model"]
        trf_arch_splits = transformer["@architectures"].split(".")
        assert len(trf_arch_splits) == 3 and trf_arch_splits[-2].endswith("Transformer")
        curated_arch = trf_arch_splits[-2]
    except (KeyError, AssertionError):
        msg.fail(
            "Pipeline config does not have a valid model architecture for "
            f"the '{transformer_name}' Curated Transformer component",
            exits=1,
        )

    hf_model_type = CURATED_TRANSFORMER_TO_HF_MODEL_TYPE.get(curated_arch)
    if hf_model_type is None:
        msg.fail(
            "Missing Hugging Face model type for Curated Transformer "
            f"model architecture '{curated_arch}'",
            exits=1,
        )
    assert hf_model_type is not None
    return hf_model_type


def _validate_hf_model_type(
    msg: Printer,
    hf_config: Dict[str, Any],
    expected_model_type: str,
    transformer_name: str,
):
    incoming_hf_model_type = hf_config["model_type"]
    if incoming_hf_model_type != expected_model_type:
        hf_model_type_to_curated_arch = {
            v: k for k, v in CURATED_TRANSFORMER_TO_HF_MODEL_TYPE.items()
        }

        if incoming_hf_model_type not in CURATED_TRANSFORMER_TO_HF_MODEL_TYPE.values():
            msg.fail(
                f"Hugging Face model of type '{incoming_hf_model_type}' cannot be loaded into "
                f"Curated Transformer pipe '{transformer_name}' of type '{expected_model_type}' - "
                "It is not supported by `spacy-curated-transformers`. The "
                f"`{hf_model_type_to_curated_arch[expected_model_type]}` architecture "
                f"expects a Hugging Face model of type '{expected_model_type}'.",
                exits=1,
            )
        else:
            expected_arch = hf_model_type_to_curated_arch[incoming_hf_model_type]
            msg.fail(
                f"Hugging Face model of type '{incoming_hf_model_type}' cannot be loaded into "
                f"Curated Transformer pipe '{transformer_name}' of type '{expected_model_type}' - "
                f"Change the 'components.{transformer_name}.model.@architectures' entrypoint "
                f"to use the '{expected_arch}' architecture. Alternatively, use a different "
                f"model with the current architecuture, e.g: '{HF_MODEL_TYPE_TO_EXAMPLE_MODELS[expected_model_type]}'.",
                exits=1,
            )


def _fill_parameters(
    msg: Printer,
    params_to_fill: Dict[str, HfParamSource],
    hf_config: Dict[str, Any],
    hf_tokenizer: Any,
) -> Dict[str, Any]:
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
        assert value is not None
        filled_params[param_name] = value

    msg.info(title="Filled-in model parameters:")
    msg.table(filled_params)

    return dict(sorted(filled_params.items(), key=lambda item: item[0]))


def _create_intermediate_configs(config: Config, dot_path: str):
    splits = dot_path.split(".")
    current = config
    while splits:
        name = splits[0]
        if name not in current:
            current[name] = {}
        current = current[name]
        splits = splits[1:]


def _save_encoder_loader_config(
    msg: Printer,
    config: Config,
    transformer_name: str,
    model_name: str,
    model_revision: str,
):
    _create_intermediate_configs(
        config, f"initialize.components.{transformer_name}.encoder_loader"
    )
    inner = config["initialize"]["components"][transformer_name]["encoder_loader"]
    if inner:
        msg.warn(f"Overwriting transformer encoder loader config")
        inner.clear()

    inner.update(
        {
            "@model_loaders": "spacy-curated-transformers.HFTransformerEncoderLoader.v1",
            "name": model_name,
            "revision": model_revision,
        }
    )


def _save_piecer_loader_config(
    msg: Printer,
    config: Config,
    transformer_name: str,
    model_name: str,
    model_revision: str,
    *,
    overwrite: bool,
):
    _create_intermediate_configs(
        config, f"initialize.components.{transformer_name}.piecer_loader"
    )
    inner = config["initialize"]["components"][transformer_name]["piecer_loader"]
    if inner:
        if overwrite:
            msg.warn(f"Overwriting piece encoder loader config")
            inner.clear()
        else:
            arch = inner.get("@model_loaders")
            if (
                arch == "spacy-curated-transformers.HFPieceEncoderLoader.v1"
                and inner["name"] == model_name
                and inner.get("revision", "main") == model_revision
            ):
                return

            msg.warn(
                f"Existing piece encoder loader might not be compatible with model '{model_name}'"
            )
            return

    inner.update(
        {
            "@model_loaders": "spacy-curated-transformers.HFPieceEncoderLoader.v1",
            "name": model_name,
            "revision": model_revision,
        }
    )


def _get_hf_model_config(msg: Printer, name: str, revision: str) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    try:
        config_path = hf_hub_download(
            repo_id=name, filename="config.json", revision=revision
        )
    except BaseException as e:
        msg.fail(
            f"Couldn't fetch the model configuration for model name '{name}' "
            f"(revision: {revision}). Ensure that the model name and revision "
            "are correct."
        )
        msg.fail(f"Exception: {e}", exits=1)

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _load_hf_tokenizer(model_name: str, model_revision: str, msg: Printer) -> Any:
    # The entry point params that we need to fetch from the upstream HF tokenizer
    # are not consistently serialized across all available models. So, we'll just
    # load the tokenizer using the `transformers` library and directly read the params
    # from the instance.
    from transformers import AutoTokenizer

    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )

        # The `model_max_length` value can sometimes be set to a very large integer
        # if the model doesn't have an upperbound on its input length. Saving this
        # as-is to the config will raise an error during serialization. So, we'll
        # try to look up the correct value manually. If we don't find one, fallback
        # to a smaller (large) value that can be serialized correctly.
        # cf: https://github.com/huggingface/transformers/blob/224da5df6956340a5c680e5b57b4914d4d7298b6/src/transformers/tokenization_utils_base.py#L104
        if hf_tokenizer.model_max_length == int(1e30):
            hf_tokenizer.model_max_length = 2147483647
            if getattr(hf_tokenizer, "max_model_input_sizes", None) is not None:
                model_name_splits = model_name.split("/")
                if len(model_name_splits) != 0:
                    model_name = model_name_splits[-1]
                model_max_length = hf_tokenizer.max_model_input_sizes.get(model_name)
                if model_max_length is not None:
                    hf_tokenizer.model_max_length = model_max_length
    except BaseException as e:
        msg.fail(
            f"Couldn't load Hugging Face tokenizer '{model_name}' ('{model_revision}') - Error:\n{e}",
            exits=1,
        )

    return hf_tokenizer
