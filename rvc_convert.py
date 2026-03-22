#!/usr/bin/env python3
"""RVC voice conversion for individual vocal segments.

Uses the RVC Python API directly for inference, avoiding CLI subprocess
issues with torch.load weights_only defaults in PyTorch >= 2.6.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_TORCH_LOAD_PATCHED = False


def _patch_torch_load():
    """Patch torch.load to default to weights_only=False for fairseq compat."""
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
    _TORCH_LOAD_PATCHED = True


def _default_rvc_assets_root() -> Path:
    env = os.environ.get("RVC_ASSETS_ROOT")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "rvc_assets"


RVC_ASSETS_ROOT = _default_rvc_assets_root()

_vc_instance = None


def _get_vc(pth_path: Path, index_path: Optional[Path] = None):
    """Get or reuse a VC instance loaded with the given checkpoint."""
    global _vc_instance
    _patch_torch_load()

    os.environ.setdefault("index_root", str(RVC_ASSETS_ROOT / "logs"))
    os.environ.setdefault("weight_root", str(RVC_ASSETS_ROOT / "weights"))
    os.environ.setdefault("hubert_path", str(RVC_ASSETS_ROOT / "assets" / "hubert" / "hubert_base.pt"))
    os.environ.setdefault("rmvpe_root", str(RVC_ASSETS_ROOT / "assets" / "rmvpe"))

    from rvc.modules.vc.modules import VC

    if _vc_instance is None or getattr(_vc_instance, "_loaded_pth", None) != str(pth_path):
        _vc_instance = VC()
        _vc_instance.get_vc(str(pth_path))
        _vc_instance._loaded_pth = str(pth_path)

    return _vc_instance


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_string(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def run_rvc_infer(
    pth_path: Path,
    input_wav: Path,
    output_wav: Path,
    index_path: Optional[Path] = None,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
) -> bool:
    """Run RVC inference using the Python API directly."""
    try:
        vc = _get_vc(pth_path, index_path)
        tgt_sr, audio_opt, times, _ = vc.vc_inference(
            sid=0,
            input_audio_path=str(input_wav),
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_file=str(index_path) if index_path and index_path.exists() else None,
            hubert_path=os.environ.get("hubert_path"),
        )
        if audio_opt is None:
            logger.warning("RVC returned None audio for %s", input_wav.name)
            return False

        from scipy.io import wavfile
        wavfile.write(str(output_wav), tgt_sr, audio_opt)
        if not output_wav.exists() or output_wav.stat().st_size < 1000:
            logger.warning("RVC output missing or too small: %s", output_wav)
            return False
        return True
    except Exception as e:
        logger.warning("RVC inference failed for %s: %s", input_wav.name, e)
        return False


def convert_segments(
    segment_paths: List[Path],
    output_dir: Path,
    pth_path: Path,
    index_path: Optional[Path] = None,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
) -> List[Path]:
    """Convert a list of vocal segment wavs through RVC.

    Each output file keeps the same filename as its input so that
    reassembly via timestamps is straightforward.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: List[Path] = []

    for seg_path in segment_paths:
        out_path = output_dir / seg_path.name
        ok = run_rvc_infer(
            pth_path, seg_path, out_path,
            index_path=index_path,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
        )
        if ok:
            converted.append(out_path)
            logger.info("Converted segment: %s", seg_path.name)
        else:
            logger.warning("Failed to convert segment: %s", seg_path.name)

    logger.info("Converted %d / %d segments", len(converted), len(segment_paths))
    return converted


def find_checkpoint_files(
    checkpoint_dir: Path,
) -> tuple[List[Path], Optional[Path]]:
    """Find .pth and .index files in a local checkpoint directory.

    Returns (pth_candidates, best_index_path).
    """
    pth_files = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.name)
    excluded = ("._D_", "._G_", "_D_", "_G_", "D_", "G_")
    inference_pth = [
        p for p in pth_files
        if not any(p.name.startswith(prefix) for prefix in excluded)
    ]
    if not inference_pth:
        inference_pth = [p for p in pth_files if "D_" not in p.name]

    index_files = sorted(checkpoint_dir.glob("*.index"), key=lambda p: p.name)
    preferred = [p for p in index_files if p.name.startswith("added")]
    best_index = preferred[0] if preferred else (index_files[0] if index_files else None)

    return inference_pth, best_index


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="RVC convert vocal segments")
    p.add_argument("--segments-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--f0-up-key", type=int, default=0)
    p.add_argument("--f0-method", default="rmvpe")
    args = p.parse_args()

    pth_candidates, best_index = find_checkpoint_files(args.checkpoint_dir)
    if not pth_candidates:
        raise FileNotFoundError(f"No .pth files found in {args.checkpoint_dir}")

    seg_wavs = sorted(args.segments_dir.glob("*.wav"))
    convert_segments(
        seg_wavs, args.output_dir,
        pth_path=pth_candidates[0],
        index_path=best_index,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
    )
