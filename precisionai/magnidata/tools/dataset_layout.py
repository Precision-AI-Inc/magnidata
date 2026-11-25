# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Locate an image's dataset root and its sibling label and mask files.

Shared by the feature extractor and the API's annotation overlay so both find the same
``labels/`` and ``masks/`` files for an image, including images in subfolders of
``images/``.
"""

from __future__ import annotations

import os

IMAGES_DIRNAME = "images"


def dataset_layout(image_path: str) -> tuple[str, str]:
    """Return (dataset_dir, subdir) for an image under ``<DATASET>/images/[<subdir>/]``.

    The nearest ancestor folder named ``images`` marks the dataset root, so images in
    subfolders keep their dataset. Without one, the image's grandparent is the dataset,
    as for any ``<DATASET>/<folder>/<STEM>.png``.

    Parameters
    ----------
    image_path : str
        Path to the image, local or a URL path.

    Returns
    -------
    tuple[str, str]
        The dataset directory and the image's subfolder below ``images/`` (empty when
        the image sits directly in it or there is no ``images/`` folder).
    """
    parts = os.path.dirname(image_path).split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == IMAGES_DIRNAME:
            return os.sep.join(parts[:i]), os.sep.join(parts[i + 1 :])
    return os.path.dirname(os.path.dirname(image_path)), ""


def legacy_dataset_dir(image_path: str) -> str | None:
    """Return the image's grandparent when that is another place its dataset may be, else None.

    Before subfolders of ``images/`` were supported the grandparent was always the dataset.
    An ``images`` folder far up the path (``/srv/images/<DATASET>/raw/<STEM>.png``) is not
    the dataset's own, so the grandparent stays a place to look. It is skipped when it is
    the ``images`` folder itself: files inside the images tree are not labels.

    Parameters
    ----------
    image_path : str
        Path to the image, local or a URL path.

    Returns
    -------
    str or None
        The grandparent directory, or None when `dataset_layout` already returns it or it
        is the ``images`` folder.
    """
    grandparent = os.path.dirname(os.path.dirname(image_path))
    if grandparent == dataset_layout(image_path)[0] or os.path.basename(grandparent) == IMAGES_DIRNAME:
        return None
    return grandparent


def sibling_candidates(image_path: str, folder: str, filename: str) -> list[str]:
    """Return where ``<folder>/<filename>`` may live for an image, in lookup order.

    Tries ``<DATASET>/<folder>/<subdir>/<filename>`` (mirroring the image's subfolder),
    then ``<DATASET>/<folder>/<filename>``, then the same file under the image's
    grandparent (see `legacy_dataset_dir`). Duplicates are dropped.

    Parameters
    ----------
    image_path : str
        Path to the image, local or a URL path.
    folder : str
        Sibling folder name, e.g. ``labels`` or ``masks``.
    filename : str
        File name to look for inside that folder.

    Returns
    -------
    list[str]
        Candidate paths, most specific first.
    """
    dataset_dir, subdir = dataset_layout(image_path)
    candidates = [
        os.path.join(dataset_dir, folder, subdir, filename),
        os.path.join(dataset_dir, folder, filename),
    ]
    legacy = legacy_dataset_dir(image_path)
    if legacy is not None:
        candidates.append(os.path.join(legacy, folder, filename))
    return list(dict.fromkeys(candidates))
