from typing import Optional, Tuple

from thinc.api import Schedule

# This is the parameter prefix for curated encoders.
_CURATED_ENCODER_PREFIX = "curated_encoder."


def transformer_discriminative(
    default_schedule: Schedule,
    transformer_schedule: Schedule,
) -> Schedule:
    """Discriminative learning rate schedule for transformer encoders.

    This schedule uses `transformer_schedule` for all transformer encoder
    parameters and `default_schedule` for other parameters.

    default_schedule (Schedule): default schedule.
    transformer_schedule (Schedule): schedule for transformer parameters.
    """
    return Schedule(
        "transfomer",
        _transformer_discriminative_schedule,
        attrs={
            "default_schedule": default_schedule,
            "transformer_schedule": transformer_schedule,
        },
    )


def _transformer_discriminative_schedule(
    schedule: Schedule, step: int, *, key: Optional[Tuple[int, str]] = None, **kwargs
) -> float:
    default_schedule: Schedule = schedule.attrs["default_schedule"]
    transformer_schedule: Schedule = schedule.attrs["transformer_schedule"]

    if key is None:
        return default_schedule(step=step, key=key, **kwargs)

    key_str = key[1]
    # We don't do a strict prefix check, since we want to support
    # an encoder wrapped into another model as well. In the latter
    # case, the prefix becomes an infix.
    if _CURATED_ENCODER_PREFIX in key_str:
        return transformer_schedule(step=step, key=key, **kwargs)

    return default_schedule(step=step, key=key, **kwargs)
