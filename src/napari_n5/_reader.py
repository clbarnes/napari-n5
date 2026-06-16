"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

import zarr
from pydantic import ValidationError
from zarr.abc.store import Store
from zarr.storage import FsspecStore
from zarr_n5.storage import ImplicitGroupWrapperStore, N5WrapperStore

from .types import N5ViewerArrayMetadata, N5ViewerGroupMetadata, PixelResolution

if TYPE_CHECKING:
    from npe2.types import (
        LayerData,
        PathLike,
        PathOrPaths,
        ReaderFunction,
        ReaderGetter,
    )

logger = logging.getLogger(__name__)
T = TypeVar("T")


def n5_array_to_layerdata(arr: zarr.Array) -> LayerData:
    kwargs: dict[str, Any] = {"multiscale": False}
    try:
        meta = N5ViewerArrayMetadata.model_validate(arr.attrs)
    except ValidationError:
        return (arr, kwargs, "image")  # type: ignore

    ndim = arr.ndim
    if unit := meta.resolution_unit:
        kwargs["units"] = (unit,) * ndim
    if res := meta.resolution:
        kwargs["scale"] = res

    return (arr, kwargs, "image")  # type: ignore


def n5_array_reader(path: PathOrPaths) -> list[LayerData]:
    out = []
    for node in paths_to_nodes(path):
        if isinstance(node, zarr.Group):
            logger.warning("Ignoring N5 group")
            continue
        out.append(n5_array_to_layerdata(node))
    return out


def scale_datasets(parent: zarr.Group) -> Iterable[zarr.Array]:
    idx = 0
    while True:
        name = f"s{idx}"
        node = parent.get(name)
        if node is None:
            break
        if not isinstance(node, zarr.Array):
            break
        yield node
        idx += 1


def n5viewer_multiscale_to_layerdata(group: zarr.Group) -> LayerData:
    # infallible
    grp_meta = N5ViewerGroupMetadata.model_validate(group.attrs)

    base_pxres: None | PixelResolution = None
    if pr := grp_meta.pixel_resolution:
        base_pxres = pr
    elif r := grp_meta.resolution:
        base_pxres = PixelResolution.from_dimensions(r)

    arrays = []
    scales = []
    translate = []
    unit = None

    is_base = True
    for arr, scale in zip(
        scale_datasets(group), grp_meta.scales or itertools.repeat(None), strict=False
    ):
        pxres = None
        if is_base:
            pxres = base_pxres
            if scale is None:
                scale = (1.0,) * arr.ndim

        try:
            arr_meta = N5ViewerArrayMetadata.model_validate(arr.attrs)
            if scale is None:
                scale = same_or(scale, arr_meta.downsampling_factors)
            if pxres is None:
                pxres = same_or(pxres, arr_meta.get_pixel_resolution())
            if is_base:
                base_pxres = same_or(base_pxres, pxres)
        except ValidationError:
            pass

        if scale is None:
            raise ValueError("Missing downscale factors")

        base_px_offsets = tuple(0.5 * (s - 1) for s in scale)
        if base_pxres is None:
            raise ValueError("Unknown base units")
        translation = tuple(
            bpo * d
            for bpo, d in zip(base_px_offsets, base_pxres.dimensions, strict=True)
        )

        if pxres is not None:
            res = pxres.dimensions
            unit = same_or(unit, pxres.unit)
        elif base_pxres is not None:
            res = tuple(
                d * s for s, d in zip(scale, base_pxres.dimensions, strict=True)
            )
            unit = same_or(unit, base_pxres.unit)
        else:
            res = scale

        translate.append(translation)
        arrays.append(arr)
        scales.append(res)
        is_base = False

    kwargs = {
        # "translate": translate,
        "scale": scales,
        "units": (unit,) * len(translate[0]),
        "multiscale": True,
    }

    return (arrays, kwargs, "image")


def same_or(a: T | None, *args: T | None) -> T | None:
    for arg in args:
        if a is None:
            a = arg
        elif arg is not None and a != arg:
            raise ValueError(f"{a} != {arg}")
    return a


def n5viewer_multiscale_reader(path: PathOrPaths) -> list[LayerData]:
    out = []
    for node in paths_to_nodes(path):
        if isinstance(node, zarr.Array):
            logger.warning("Ignoring N5 array")
            continue
        out.append(n5viewer_multiscale_to_layerdata(node))
    return out


def path_to_node(path: PathLike) -> None | zarr.Array | zarr.Group:
    try:
        inner = FsspecStore.from_url(path, read_only=True)
    except Exception:  # noqa: BLE001
        logger.debug("Could not open FsspecStore")
        return None

    store: Store = N5WrapperStore(inner)
    if path.endswith(".n5"):
        store = ImplicitGroupWrapperStore(store)

    try:
        node = zarr.open(store=store, mode="r")
    except Exception:  # noqa: BLE001
        logger.debug("Could not open Zarr node with N5WrapperStore")
        return None

    return node


def paths_to_nodes(path: PathOrPaths) -> Iterable[zarr.Array | zarr.Group]:
    if isinstance(path, str):
        path = [path]

    for p in path:
        n = path_to_node(p)
        if n is not None:
            yield n


def _get_n5_reader(path: PathOrPaths) -> None | ReaderFunction:
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    has_array = False
    has_group = False
    for node in paths_to_nodes(path):
        if isinstance(node, zarr.Group):
            has_group = True
            s0 = node.get("s0")
            if not isinstance(s0, zarr.Array):
                logger.debug("Path is an N5 group but not recognised as a multiscale")
                return None
        elif isinstance(node, zarr.Array):
            has_array = True
        else:
            logger.warning("Unknown zarr node type %s", type(node).__qualname__)
            return None

    match (has_array, has_group):
        case (True, True):
            logger.debug("Given paths were a mixture of N5 arrays and groups; skipping")
        case (True, False):
            return n5_array_reader
        case (False, True):
            return n5viewer_multiscale_reader
        case (False, False):
            logger.debug("No N5 arrays or groups found")

    return None


napari_get_reader: ReaderGetter = _get_n5_reader


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    # paths = [path] if isinstance(path, str) else path
    # # load all files into array
    # arrays = [np.load(_path) for _path in paths]
    # # stack arrays into single array
    # data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}
    data = []

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
