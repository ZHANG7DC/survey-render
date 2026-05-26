from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .config import TargetBand, group_targets_by_instrument


class ZarrSurveyWriter:
    def __init__(
        self,
        output_path: Path,
        targets: list[TargetBand],
        *,
        image_size: int,
        dtype: str,
        chunk_size: int,
        overwrite: bool,
        write_clean: bool,
        attrs: dict,
    ) -> None:
        import numcodecs
        import zarr

        self.output_path = Path(output_path)
        if self.output_path.exists() and overwrite:
            shutil.rmtree(self.output_path)
        mode = "w" if overwrite else "w-"
        self.root = zarr.open_group(str(self.output_path), mode=mode)
        self.targets = targets
        self.grouped = group_targets_by_instrument(targets)
        self.image_size = int(image_size)
        self.dtype = np.dtype(dtype)
        self.chunk_size = max(1, int(chunk_size))
        self.write_clean = bool(write_clean)
        self.count = 0
        self.root.attrs.update(attrs)
        self.root.attrs["targets"] = json.dumps([target.__dict__ for target in targets], sort_keys=True)
        self.ids = self.root.create_dataset(
            "id", shape=(0,), chunks=(self.chunk_size,), dtype=object, object_codec=numcodecs.VLenUTF8()
        )
        self.paths = self.root.create_dataset(
            "source_path", shape=(0,), chunks=(self.chunk_size,), dtype=object, object_codec=numcodecs.VLenUTF8()
        )
        self.images = self.root.create_group("images")
        self.clean_images = self.root.create_group("clean_images") if self.write_clean else None
        self.arrays = {}
        self.clean_arrays = {}
        for instrument, items in self.grouped.items():
            shape = (0, self.image_size, self.image_size, len(items))
            chunks = (self.chunk_size, self.image_size, self.image_size, len(items))
            arr = self.images.create_dataset(instrument, shape=shape, chunks=chunks, dtype=self.dtype)
            arr.attrs["bands"] = [target.band for target in items]
            arr.attrs["target_keys"] = [target.key for target in items]
            self.arrays[instrument] = arr
            if self.clean_images is not None:
                clean = self.clean_images.create_dataset(instrument, shape=shape, chunks=chunks, dtype=self.dtype)
                clean.attrs["bands"] = [target.band for target in items]
                clean.attrs["target_keys"] = [target.key for target in items]
                self.clean_arrays[instrument] = clean

    def append(
        self,
        *,
        sample_id: str,
        source_path: str,
        observed_by_key: dict[str, np.ndarray],
        clean_by_key: dict[str, np.ndarray] | None = None,
    ) -> None:
        idx = self.count
        end = idx + 1
        self.ids.resize((end,))
        self.paths.resize((end,))
        self.ids[idx] = sample_id
        self.paths[idx] = source_path
        for instrument, items in self.grouped.items():
            stack = np.stack([observed_by_key[target.key] for target in items], axis=-1)
            arr = self.arrays[instrument]
            arr.resize((end, self.image_size, self.image_size, len(items)))
            arr[idx] = stack.astype(self.dtype, copy=False)
            if self.write_clean and clean_by_key is not None:
                clean_stack = np.stack([clean_by_key[target.key] for target in items], axis=-1)
                clean_arr = self.clean_arrays[instrument]
                clean_arr.resize((end, self.image_size, self.image_size, len(items)))
                clean_arr[idx] = clean_stack.astype(self.dtype, copy=False)
        self.count = end
        self.root.attrs["count"] = self.count
