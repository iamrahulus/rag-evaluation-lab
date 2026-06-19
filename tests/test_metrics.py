from src.evaluation.metrics import _normalise


def test_normalise_already_normalised():
    assert _normalise(0.8) == 0.8


def test_normalise_out_of_ten():
    assert _normalise(8.0) == 0.8


def test_normalise_clamps_above_one():
    assert _normalise(1.5) == 1.0


def test_normalise_clamps_below_zero():
    assert _normalise(-0.1) == 0.0


def test_normalise_exactly_one():
    assert _normalise(1.0) == 1.0


def test_normalise_exactly_ten():
    assert _normalise(10.0) == 1.0
