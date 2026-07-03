# weights.py
"""
Fetches released model checkpoints on demand.

The published KO-BCE / gKO-BCE weights live on the Hugging Face Hub and are pulled
from their direct resolve URLs (no `huggingface_hub` dependency required). Files are
cached under ``config.WEIGHTS_CACHE_DIR`` so each file is only downloaded once. The
scheme is host-agnostic: change ``config.WEIGHTS_BASE_URL`` to serve the same file
layout from Zenodo, a GitHub Release, or any static host.
"""
import os
import sys
import shutil
from pathlib import Path

import requests

import config


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest`` atomically, showing a simple progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    sys.stdout.write(f"\r    {done / 1e6:7.1f} / {total / 1e6:7.1f} MB ({pct:5.1f}%)")
                    sys.stdout.flush()
        if total:
            sys.stdout.write("\n")
    os.replace(tmp, dest)


def ensure_pca_models() -> None:
    """Ensure the fitted ESM PCA reducers are present (they ship with the repo,
    but fall back to the remote host if a user deleted them)."""
    needed = []
    if config.REDUCE_ESM2_DIM:
        needed.append(f"esm2_pca_{config.ESM2_DIM_TARGET}.pkl")
    if config.REDUCE_ESM1V_DIM:
        needed.append(f"esm1v_pca_{config.ESM1V_DIM_TARGET}.pkl")
    for fname in needed:
        local = Path(config.PCA_MODEL_CACHE_DIR) / fname
        if not local.exists():
            _download_file(f"{config.WEIGHTS_BASE_URL}/pca/{fname}", local)


def ensure_model(model_key: str) -> Path:
    """Download (if needed) and return the local directory holding a checkpoint's files.

    Args:
        model_key: one of ``config.MODELS`` (e.g. "ko-bce" or "gko-bce").

    Returns:
        Path to the local directory containing best_model.pt, normalizer_quantile.pkl,
        results.json and info.json.
    """
    if model_key not in config.MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(config.MODELS)}.")

    spec = config.MODELS[model_key]
    remote_subdir = spec["remote_subdir"]
    local_dir = Path(config.WEIGHTS_CACHE_DIR) / model_key
    local_dir.mkdir(parents=True, exist_ok=True)

    for fname in config.WEIGHT_FILES:
        dest = local_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{config.WEIGHTS_BASE_URL}/{remote_subdir}/{fname}"
        _download_file(url, dest)

    ensure_pca_models()
    return local_dir


def _clear_cache(model_key: str = None) -> None:
    """Utility to wipe cached downloads (e.g. after a bad/interrupted fetch)."""
    if model_key:
        shutil.rmtree(Path(config.WEIGHTS_CACHE_DIR) / model_key, ignore_errors=True)
    else:
        shutil.rmtree(Path(config.WEIGHTS_CACHE_DIR), ignore_errors=True)
