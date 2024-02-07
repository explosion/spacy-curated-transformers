from thinc.api import constant

from spacy_curated_transformers.schedules import transformer_discriminative


def test_schedules():
    default_schedule = constant(1e-3)
    transformer_schedule = constant(1e-5)
    schedule = transformer_discriminative(default_schedule, transformer_schedule)
    assert schedule(0, key=(0, "some_key")) == 1e-3
    assert schedule(0, key=(1, "curated_encoder.embeddings")) == 1e-5
    assert schedule(0, key=(2, "wrapping_model.curated_encoder.embeddings")) == 1e-5
