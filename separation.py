#!/usr/bin/env python3
"""Vocal/instrumental separation with diverse model backends.

Randomly selects among separation models (Demucs htdemucs_ft, Roformer
MelBandRoformer, MDX23C+MDXNET chain, UVR5 VR-Net) so the downstream
SVDD detector sees varied separation artefacts and learns to be invariant
to them.
"""
from __future__ import annotations

import logging
import random
import shutil
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

MODELS_DIR = Path(__file__).parent / "models" / "audio-separator"


class SeparationBackend(Enum):
    DEMUCS = "demucs"
    ROFORMER = "roformer"
    MDXCHAIN = "mdxchain"
    UVR5 = "uvr5"


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

    model_dir = DATA_ROOT / model_entry.slug
    config_path = model_dir / model_entry.config
    with open(config_path) as f:
        from ml_collections import ConfigDict
        config = ConfigDict(yaml.safe_load(f))

    model = get_model_from_config("mel_band_roformer", config)
    ckpt_path = model_dir / model_entry.checkpoint
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


def _find_stem(files: list, keyword: str, directory: str = "") -> str:
    """Find an output file whose name contains keyword (case-insensitive).

    audio-separator may return bare filenames; *directory* is prepended when
    the matched path does not already exist on disk.
    """
    for f in files:
        if keyword.lower() in Path(f).name.lower():
            p = Path(f)
            if not p.exists() and directory:
                p = Path(directory) / p.name
            return str(p)
    raise FileNotFoundError(f"No stem matching '{keyword}' in {files}")


def _make_separator(tmpdir: str):
    """Create an audio-separator Separator with standard settings."""
    from audio_separator.separator import Separator
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return Separator(
        output_dir=tmpdir,
        output_format="WAV",
        model_file_dir=str(MODELS_DIR),
    )


def separate_mdxchain(
    input_path: Path,
    vocals_out: Path,
    instrumental_out: Path,
    device: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Two-stage: MDX23C for vocal/instrumental, then MDXNET KARA for lead/backing.

    Stage 1 splits the mix into vocals and instrumentals.  Stage 2 refines
    the vocal stem by separating lead from backing vocals.  Backing vocals
    are folded back into the instrumental output.
    """
    with tempfile.TemporaryDirectory(prefix="mdxchain_") as tmpdir:
        sep = _make_separator(tmpdir)

        # Stage 1: MDX23C vocal/instrumental split
        sep.load_model(model_filename="MDX23C-8KFFT-InstVoc_HQ.ckpt")
        s1 = sep.separate(str(input_path))
        logger.info("MDXCHAIN stage 1 outputs: %s", s1)
        s1_vocals = _find_stem(s1, "(Vocals)", tmpdir)
        s1_inst = _find_stem(s1, "(Instrumental)", tmpdir)

        # Stage 2: MDXNET KARA splits vocals → lead + backing
        sep.load_model(model_filename="UVR_MDXNET_KARA_2.onnx")
        s2 = sep.separate(s1_vocals)
        logger.info("MDXCHAIN stage 2 outputs: %s", s2)
        s2_lead = _find_stem(s2, "(Vocals)", tmpdir)
        s2_back = _find_stem(s2, "(Instrumental)", tmpdir)

        shutil.copy2(s2_lead, vocals_out)

        # Merge backing vocals into instrumentals
        inst_wav, sr = sf.read(s1_inst)
        back_wav, _ = sf.read(s2_back)
        min_len = min(len(inst_wav), len(back_wav))
        sf.write(
            str(instrumental_out),
            inst_wav[:min_len] + back_wav[:min_len],
            sr,
            subtype="PCM_16",
        )

    logger.info("MDXCHAIN separation complete: %s", input_path.name)
    return vocals_out, instrumental_out


def separate_uvr5(
    input_path: Path,
    vocals_out: Path,
    instrumental_out: Path,
    device: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Separate using UVR5 VR-Net architecture (HP2 model)."""
    with tempfile.TemporaryDirectory(prefix="uvr5_") as tmpdir:
        sep = _make_separator(tmpdir)
        sep.load_model(model_filename="2_HP-UVR.pth")
        output_files = sep.separate(str(input_path))
        logger.info("UVR5 outputs: %s", output_files)

        shutil.copy2(_find_stem(output_files, "(Vocals)", tmpdir), vocals_out)
        shutil.copy2(_find_stem(output_files, "(Instrumental)", tmpdir), instrumental_out)

    logger.info("UVR5 separation complete: %s", input_path.name)
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

    dispatch = {
        SeparationBackend.DEMUCS: separate_demucs,
        SeparationBackend.ROFORMER: separate_roformer,
        SeparationBackend.MDXCHAIN: separate_mdxchain,
        SeparationBackend.UVR5: separate_uvr5,
    }
    fn = dispatch.get(backend)
    if fn is None:
        raise ValueError(f"Unknown backend: {backend}")
    v, i = fn(input_path, vocals_out, instrumental_out, device)
    return v, i, backend


_ALL_BACKENDS = [b.value for b in SeparationBackend]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Separate vocals and instrumentals")
    p.add_argument("input", type=Path)
    p.add_argument("--vocals-out", type=Path, default=None)
    p.add_argument("--instrumental-out", type=Path, default=None)
    p.add_argument("--backend", choices=_ALL_BACKENDS, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    vout = args.vocals_out or args.input.parent / f"{args.input.stem}_vocals.wav"
    iout = args.instrumental_out or args.input.parent / f"{args.input.stem}_instrumental.wav"
    backend = SeparationBackend(args.backend) if args.backend else None
    separate(args.input, vout, iout, backend=backend, device=args.device)
