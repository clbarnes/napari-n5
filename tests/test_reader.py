from itertools import pairwise
from typing import Any

from napari_n5 import napari_get_reader


def test_reader_multiscale(data_dir):
    p = str(data_dir / "n5viewer.n5")

    reader = napari_get_reader(p)
    assert reader is not None
    layers_data = reader(p)

    assert len(layers_data) == 1
    arrs, *kwargs_type = layers_data[0]

    assert isinstance(arrs, list)
    assert len(arrs) == 3
    for a, b in pairwise(arrs):
        assert a.ndim == b.ndim
        for a_s, b_s in zip(a.shape, b.shape, strict=True):
            assert b_s < a_s

    if kwargs_type:
        kwargs: dict[str, Any]
        kwargs, *type_tup = kwargs_type
        assert kwargs["multiscale"]
    if type_tup:
        assert len(type_tup) == 1
        layer_type = type_tup[0]
        assert layer_type == "image"
