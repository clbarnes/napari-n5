#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imageio>=2.37.3",
#     "numpy>=2.4.6",
#     "scikit-image>=0.26.0",
#     "tensorstore>=0.1.84",
# ]
# ///
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
from imageio.v3 import imread
from skimage.transform import rescale

DATA_PATH = Path(__file__).resolve().parent


def write_array(path: Path, data: np.ndarray, attributes: dict[str, Any] | None = None):
    ensure_empty_directory(path, True)

    metadata: dict[str, Any] = {
        "dimensions": data.shape,
        "dataType": str(data.dtype),
        "blockSize": [32] * data.ndim,
        "compression": {"type": "gzip", "level": 9},
    }
    if attributes:
        metadata.update(attributes)

    dataset = ts.open(
        {
            "driver": "n5",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": metadata,
            "create": True,
        }
    ).result()
    dataset.write(data).result()


def write_multiscales(
    path: Path,
    data: np.ndarray,
    n_scales: int = 3,
    attributes: dict[str, Any] | None = None,
):
    ensure_empty_directory(path)
    if attributes is None:
        attributes = dict()
    attributes.setdefault("n5", "4.0.0")
    (path / "attributes.json").write_text(json.dumps(attributes))
    for s_idx in range(n_scales):
        p = path / f"s{s_idx}"
        factors = [2**s_idx] * data.ndim
        if s_idx == 0:
            scaled_data = data
        else:
            scaled_data = rescale(
                data, tuple(1 / f for f in factors), anti_aliasing=True
            ).astype(data.dtype)
        write_array(p, scaled_data, {"downsamplingFactors": factors})


def ensure_empty_directory(p: Path, parents_only: bool = False):
    if p.is_dir():
        shutil.rmtree(p)

    if parents_only:
        p = p.parent
    p.mkdir(exist_ok=True, parents=True)


def main() -> None:
    stent = imread("imageio:stent.npz").astype("float32")
    stent -= stent.min()
    stent /= stent.max()
    stent *= 255
    stent = stent.astype("uint8")

    root = DATA_PATH / "n5viewer.n5"
    ensure_empty_directory(root)
    (root / "attributes.json").write_text(
        json.dumps(
            {
                "n5": "4.0.0",
                "pixelResolution": {"unit": "mm", "dimensions": [1.5, 1.5, 1.5]},
            }
        )
    )
    write_multiscales(
        root,
        stent,
        3,
        {"pixelResolution": {"unit": "mm", "dimensions": [1.5, 1.5, 1.5]}},
    )


if __name__ == "__main__":
    main()
