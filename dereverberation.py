#!/usr/bin/env python3
"""Dereverberation with selectable backends, applied stochastically.

Two backends are supported:

  * **vrnet** – UVR5 VR-DeEchoDeReverb (VR-Net architecture via audio-separator).
  * **mbr** – MelBandRoformer de-reverb via audio-separator.

Both backends auto-download model weights on first use.

In the SVDD pipeline, dereverberation is applied 53 % of the time so
the detection model sees a mixture of reverberant and dry vocals.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, NoiseGate

logger = logging.getLogger(__name__)

DEREVERB_PROBABILITY = 0.53
MODELS_DIR = Path(__file__).parent / "models" / "audio-separator"


def _find_dereverb_stem(files: list, output_dir: str, *keywords: str) -> str:
    """Find the de-reverbed output file by keyword match."""
    for f in files:
        name = Path(f).name.lower()
        if any(k.lower() in name for k in keywords):
            p = Path(f)
            if not p.exists() and output_dir:
                p = Path(output_dir) / p.name
            return str(p)
    raise FileNotFoundError(
        f"No dereverb stem matching {keywords} in {files}"
    )


def _make_separator(output_dir: str):
    from audio_separator.separator import Separator
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return Separator(
        output_dir=output_dir,
        output_format="WAV",
        model_file_dir=str(MODELS_DIR),
    )


class Dereverberation:
    """UVR5 VR-Net dereverberation via audio-separator.

    Uses UVR-DeEcho-DeReverb.pth (CascadedNet with LSTM).
    Model is downloaded automatically on first use.
    """

    MODEL_FILENAME = "UVR-DeEcho-DeReverb.pth"

    def __init__(self, device: str = "cuda", agg: int = 5):
        self.device = device
        self.agg = agg
        self.board = Pedalboard([NoiseGate(threshold_db=-40)])
        self._separator = None
        self._output_dir = None

    def load(self):
        self._output_dir = tempfile.mkdtemp(prefix="dereverb_vr_")
        self._separator = _make_separator(self._output_dir)
        self._separator.load_model(model_filename=self.MODEL_FILENAME)
        logger.info("Loaded VR-Net dereverb model: %s", self.MODEL_FILENAME)

    def process(self, wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        if self._separator is None:
            self.load()

        tmp_in = os.path.join(self._output_dir, "_dereverb_input.wav")
        sf.write(tmp_in, wav, sr, subtype="PCM_16")

        output_files = self._separator.separate(tmp_in)
        logger.info("VR-Net dereverb outputs: %s", output_files)

        stem = _find_dereverb_stem(
            output_files, self._output_dir,
            "(No Reverb)", "no_reverb", "noreverb",
        )
        wav_out, sr_out = sf.read(stem)
        os.unlink(tmp_in)

        input_len = wav.shape[0]
        if wav_out.shape[0] > input_len:
            wav_out = wav_out[:input_len]
        elif wav_out.shape[0] < input_len:
            pad = [(0, input_len - wav_out.shape[0])]
            if wav_out.ndim == 2:
                pad.append((0, 0))
            wav_out = np.pad(wav_out, pad)

        wav_out = self.board(wav_out, sample_rate=sr_out)
        return wav_out, sr_out


class DereverbMelBandRoformer:
    """MelBandRoformer dereverberation via audio-separator.

    Uses the anvuew de-reverb checkpoint (SDR 19.17).
    Model is downloaded automatically on first use.
    """

    MODEL_FILENAME = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.board = Pedalboard([NoiseGate(threshold_db=-40)])
        self._separator = None
        self._output_dir = None

    def load(self):
        self._output_dir = tempfile.mkdtemp(prefix="dereverb_mbr_")
        self._separator = _make_separator(self._output_dir)
        self._separator.load_model(model_filename=self.MODEL_FILENAME)
        logger.info("Loaded MBR dereverb model: %s", self.MODEL_FILENAME)

    def process(self, wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        if self._separator is None:
            self.load()

        tmp_in = os.path.join(self._output_dir, "_dereverb_input.wav")
        sf.write(tmp_in, wav, sr, subtype="PCM_16")

        output_files = self._separator.separate(tmp_in)
        logger.info("MBR dereverb outputs: %s", output_files)

        stem = _find_dereverb_stem(
            output_files, self._output_dir,
            "(noreverb)", "no_reverb", "(dry)", "(No Reverb)",
        )
        wav_out, sr_out = sf.read(stem)
        os.unlink(tmp_in)

        input_len = wav.shape[0]
        if wav_out.shape[0] > input_len:
            wav_out = wav_out[:input_len]
        elif wav_out.shape[0] < input_len:
            pad = [(0, input_len - wav_out.shape[0])]
            if wav_out.ndim == 2:
                pad.append((0, 0))
            wav_out = np.pad(wav_out, pad)

        wav_out = self.board(wav_out, sample_rate=sr_out)
        return wav_out, sr_out


def maybe_dereverb(
    vocals_path: Path,
    output_path: Path,
    dereverb_model=None,
    probability: float = DEREVERB_PROBABILITY,
    force: Optional[bool] = None,
) -> Tuple[Path, bool]:
    """Conditionally apply dereverberation.

    Args:
        vocals_path: input vocal wav.
        output_path: where to write (possibly dereverberated) output.
        dereverb_model: a loaded Dereverberation or DereverbMelBandRoformer.
        probability: chance of applying dereverb (default 0.53).
        force: if True always apply, if False never apply, None = stochastic.

    Returns:
        (output_path, was_applied).
    """
    import random, shutil

    apply = force if force is not None else (random.random() < probability)

    if not apply:
        if vocals_path != output_path:
            shutil.copy2(vocals_path, output_path)
        logger.info("Dereverberation skipped (p=%.2f) for %s", probability, vocals_path.name)
        return output_path, False

    if dereverb_model is None:
        raise RuntimeError(
            "Dereverberation was selected but no model instance was provided. "
            "Pass --dereverb-backend to enable dereverberation."
        )

    import librosa
    wav, sr = librosa.load(str(vocals_path), sr=None, mono=False)
    if wav.ndim == 2:
        wav = wav.T

    wav_out, sr_out = dereverb_model.process(wav, sr)

    if wav_out.ndim == 2 and wav_out.shape[1] == 1:
        wav_out = wav_out.squeeze(1)

    sf.write(str(output_path), wav_out, sr_out, subtype="PCM_16")
    logger.info("Dereverberation applied to %s -> %s", vocals_path.name, output_path.name)
    return output_path, True


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Apply dereverberation to vocals")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backend", choices=["vrnet", "mbr"], default="mbr",
                   help="Dereverb backend (default: mbr)")
    p.add_argument("--force", action="store_true", help="Always apply (ignore probability)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = args.output or args.input.parent / f"{args.input.stem}_dereverb.wav"
    model = DereverbMelBandRoformer(device=args.device) if args.backend == "mbr" \
        else Dereverberation(device=args.device)
    model.load()
    maybe_dereverb(args.input, out, dereverb_model=model, force=args.force or None)
