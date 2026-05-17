"""Sampling module regression tests."""

from sequenzo.hierarchical import sampling_scheme_description


def test_sampling_scheme_description_static_sequence():
    text = sampling_scheme_description("sequence")
    assert "sequence subsampling" in text
    assert "NOT direct sampling" in text or "not direct sampling" in text.lower()


def test_sampling_scheme_description_static_pair():
    text = sampling_scheme_description("pair")
    assert "direct pair sampling" in text
