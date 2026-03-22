#!/usr/bin/env python3
"""Vocal/instrumental separation with diverse model backends.

Randomly selects among separation models (Demucs htdemucs_ft, Roformer
MelBandRoformer) so the downstream SVDD detector sees varied separation
artefacts and learns to be invariant to them.
"""
from __future__ import annotations

import logging
import random
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


class SeparationBackend(Enum):
    DEMUCS = "demucs"
    ROFORMER = "roformer"


def _load_audio_universal(path: Path, sr: int = 44100) -> np.ndarray:
    """Load audio as (channels, samples) at target sr, converting if needed."""
    try:
        mix, _ = librosa.load(str(path), sr=sr, mono=False)
    except Exception:
        tmp = path.parent / "_converted_for_sep.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", str(sr), "-ac", "2", str(tmp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        mix, _ = librosa.load(str(tmp), sr=sr, mono=False)
        tmp.unlink(missing_ok=True)

    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=0)
    elif mix.shape[0] == 1:
        mix = np.concatenate([mix, mix], axis=0)
    elif mix.shape[0] > 2:
        mix = mix[:2, :]
    return mix


def separate_demucs(
    input_path: Path,
    vocals_out: Path,
    instrumental_out: Path,
    device: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Separate using Demucs htdemucs_ft."""
    from demucs.apply import apply_model
    from demucs.audio import save_audio
    from demucs.pretrained import get_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("htdemucs_ft").to(device)

    wav, sr = torchaudio.load(str(input_path))
    if sr != model.samplerate:
        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    wav = wav.unsqueeze(0).to(device)

    with torch.inference_mode():
        stems = apply_model(
            model, wav, shifts=0, split=True, overlap=0.20, device=device,
        ).squeeze(0)

    vocals = stems[model.sources.index("vocals")].cpu()
    instrumental = sum(
        stems[i].cpu() for i, s in enumerate(model.sources) if s != "vocals"
    )

    save_audio(vocals, vocals_out, model.samplerate)
    save_audio(instrumental, instrumental_out, model.samplerate)
    logger.info("Demucs separation complete: %s", input_path.name)
    return vocals_out, instrumental_out


def separate_roformer(
    input_path: Path,
    vocals_out: Path,
    instrumental_out: Path,
    device: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Separate using MelBandRoformer via mel_band_roformer package."""
    from mel_band_roformer import MODEL_REGISTRY, demix_track
    from mel_band_roformer.download import download_model_assets, DATA_ROOT
    from mel_band_roformer.inference import get_model_from_config
    import yaml

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_entry = MODEL_REGISTRY.get("melband-roformer-kim-vocals")
    download_model_assets([model_entry], DATA_ROOT)

    config_path = DATA_ROOT / model_entry.config
    with open(config_path) as f:
        from ml_collections import ConfigDict
        config = ConfigDict(yaml.safe_load(f))

    model = get_model_from_config("mel_band_roformer", config)
    ckpt_path = DATA_ROOT / model_entry.checkpoint
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "state" in state_dict:
        state_dict = state_dict["state"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)

    mix = _load_audio_universal(input_path, sr=44100)
    mixture = torch.tensor(mix, dtype=torch.float32)
    res, _ = demix_track(config, model, mixture, device)

    instruments = list(res.keys())
    target_key = config.training.target_instrument if hasattr(config.training, "target_instrument") and config.training.target_instrument else instruments[0]

    vocals = res[target_key]
    sf.write(str(vocals_out), vocals.T, 44100, subtype="PCM_16")

    instrumental = mix - vocals
    sf.write(str(instrumental_out), instrumental.T, 44100, subtype="PCM_16")

    logger.info("Roformer separation complete: %s", input_path.name)
    return vocals_out, instrumental_out


def separate(
    input_path: Path,
    vocals_out: Path,
    instrumental_out: Path,
    backend: Optional[SeparationBackend] = None,
    device: Optional[str] = None,
) -> Tuple[Path, Path, SeparationBackend]:
    """Run separation with specified or randomly chosen backend.

    Returns (vocals_path, instrumental_path, backend_used).
    """
    if backend is None:
        backend = random.choice(list(SeparationBackend))
    logger.info("Separation backend: %s for %s", backend.value, input_path.name)

    if backend == SeparationBackend.DEMUCS:
        v, i = separate_demucs(input_path, vocals_out, instrumental_out, device)
    elif backend == SeparationBackend.ROFORMER:
        v, i = separate_roformer(input_path, vocals_out, instrumental_out, device)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return v, i, backend


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Separate vocals and instrumentals")
    p.add_argument("input", type=Path)
    p.add_argument("--vocals-out", type=Path, default=None)
    p.add_argument("--instrumental-out", type=Path, default=None)
    p.add_argument("--backend", choices=["demucs", "roformer"], default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    vout = args.vocals_out or args.input.parent / f"{args.input.stem}_vocals.wav"
    iout = args.instrumental_out or args.input.parent / f"{args.input.stem}_instrumental.wav"
    backend = SeparationBackend(args.backend) if args.backend else None
    separate(args.input, vout, iout, backend=backend, device=args.device)
