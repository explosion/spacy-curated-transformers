import itertools
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable

import thinc
from spacy.language import Language

if TYPE_CHECKING:
    from .pipeline.transformer import CuratedTransformer

thinc.registry.create("model_loaders", entry_points=True)
registry = thinc.registry


def all_equal(iterable: Iterable[Any]) -> bool:
    """Return True if all the elements are equal to each other
    (or if the input is an empty sequence), False otherwise."""
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)  # type: ignore


def gradual_transformer_unfreezing_per_pipe(
    nlp: Language,
    callback_args: Dict[str, Any],
    freeze_params: Dict[str, int],
):
    current_step = callback_args["step"]

    # Scoped import to avoid import cycles.
    from .pipeline.transformer import CuratedTransformer

    for name, pipe in nlp.components:
        unfreeze_step = freeze_params.get(name)
        if unfreeze_step is None:
            continue
        elif not isinstance(pipe, CuratedTransformer):
            raise TypeError(
                "Attempting to perform gradual unfreezing of a non-transformer pipe "
                f"('{name}'. Only transformer pipes support this feature"
            )

        pipe.frozen = current_step < unfreeze_step


def gradual_transformer_unfreezing_all_pipes(
    nlp: Language, callback_args: Dict[str, Any], unfreeze_step: int
):
    current_step = callback_args["step"]

    # Scoped import to avoid import cycles.
    from .pipeline.transformer import CuratedTransformer

    for _, pipe in nlp.components:
        if not isinstance(pipe, CuratedTransformer):
            continue

        pipe.frozen = current_step < unfreeze_step


def create_gradual_transformer_unfreezing(
    target_pipes: Dict[str, int]
) -> Callable[[Language, Dict[str, Any]], None]:
    """Construct a callback that can be used to gradually unfreeze the
    weights of one or more Transformer components during training. This
    can be used to prevent catastrophic forgetting during fine-tuning.

    target_pipes (Dict[str, int]):
        A dictionary whose keys and values correspond to the names of Transformer
        components and the training step at which they should be unfrozen respectively.
    """
    unfreeze_step_all_pipes = target_pipes.get("*")
    if unfreeze_step_all_pipes is not None and len(target_pipes) > 1:
        raise ValueError(
            "The target pipe names for gradual transformer unfreezing contain both the "
            "wild-card operator ('*') and individual names. Use either of the two but not both"
        )

    if unfreeze_step_all_pipes is not None:
        return partial(
            gradual_transformer_unfreezing_all_pipes,
            unfreeze_step=unfreeze_step_all_pipes,
        )
    else:
        return partial(
            gradual_transformer_unfreezing_per_pipe,
            freeze_params=target_pipes,
            last_unfreeze_step=max(target_pipes.values()),
        )
