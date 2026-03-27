#!/usr/bin/env python3
"""Orchestrator pipeline that automatically determines the optimal pitch
transposition for an RVC model before running the voice conversion pipeline.

Steps:
  1. Convert a sweep test audio file (vocal scales) through RVC at f0_up_key=0.
  2. Analyse the RVC output with HNR range detection to find the model's
     usable pitch range.
  3. Separate vocals from the target song so pitch analysis is not
     confused by instrumental content.
  4. Analyse the separated vocals to determine the singer's average pitch
     and compute the nearest-octave transposition needed.
  5. Run vc_pipeline.py with the computed transposition value.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def _short_hash(path: Path) -> str:
    """Return a short MD5 hash of a file path for cache keying."""
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:10]


def _run_rvc_sweep(
    sweep_audio: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    f0_method: str = "rmvpe",
    force: bool = False,
) -> Path | None:
    """Run the sweep test file through RVC and return the output path."""
    from rvc_convert import find_checkpoint_files, run_rvc_infer

    pth_candidates, best_index = find_checkpoint_files(checkpoint_dir)
    if not pth_candidates:
        logger.error("No .pth checkpoint found in %s", checkpoint_dir)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_hash = _short_hash(checkpoint_dir)
    output_wav = output_dir / f"sweep_rvc_{ckpt_hash}.wav"

    if output_wav.exists() and not force:
        logger.info("Sweep RVC output already exists, reusing: %s", output_wav)
        return output_wav
    if output_wav.exists() and force:
        logger.info("Force mode: removing cached sweep output")
        output_wav.unlink()

    logger.info("Running sweep test through RVC...")
    ok = run_rvc_infer(
        pth_path=pth_candidates[0],
        input_wav=sweep_audio,
        output_wav=output_wav,
        index_path=best_index,
        f0_up_key=0,
        f0_method=f0_method,
    )
    if not ok:
        logger.error("RVC conversion of sweep test failed")
        return None

    logger.info("Sweep RVC output: %s", output_wav)
    return output_wav


def _analyse_hnr(
    rvc_sweep_wav: Path,
    method: str = "autocorrelation",
    threshold: float = 2.5,
) -> dict | None:
    """Run HNR range analysis and return the usable range."""
    from hnr_range import analyse_hnr_by_note

    logger.info("Analysing HNR range (method=%s, threshold=%.1f dB)...", method, threshold)
    results = analyse_hnr_by_note(
        rvc_sweep_wav,
        method=method,
        hnr_threshold_db=threshold,
    )

    if "error" in results:
        logger.error("HNR analysis failed: %s", results["error"])
        return None

    usable = results["usable_range"]
    if not usable.get("low"):
        logger.error("No usable range detected above %.1f dB threshold", threshold)
        return None

    logger.info(
        "Usable range: %s (%s Hz) — %s (%s Hz)",
        usable["low"], usable["low_hz"],
        usable["high"], usable["high_hz"],
    )
    return usable


def _separate_vocals(
    input_audio: Path,
    output_dir: Path,
    sep_backend: str,
    device: str | None = None,
    force: bool = False,
) -> Path | None:
    """Run separation to extract vocals for pitch analysis and downstream RVC.

    The separated vocals are reused as input to vc_pipeline.py (with
    --no-separation) so the same track is not separated twice.
    """
    from separation import SeparationBackend, separate

    input_hash = _short_hash(input_audio)
    vocals_out = output_dir / f"separated_vocals_{input_hash}.wav"
    instrumental_out = output_dir / f"separated_instrumental_{input_hash}.wav"

    if vocals_out.exists() and not force:
        logger.info("Separated vocals already cached, reusing: %s", vocals_out)
        return vocals_out

    output_dir.mkdir(parents=True, exist_ok=True)
    backend = SeparationBackend(sep_backend)

    logger.info("Separating vocals from %s ...", input_audio.name)
    try:
        vocals_path, _, _ = separate(
            input_audio, vocals_out, instrumental_out,
            backend=backend, device=device,
        )
    except Exception as e:
        logger.error("Separation failed: %s", e)
        return None

    return vocals_path


def _compute_transposition(
    vocals_audio: Path,
    range_low_hz: float,
    range_high_hz: float,
) -> dict | None:
    """Analyse separated vocals and compute nearest-octave transposition."""
    from pitch_match import analyse_singer_pitch, compute_pitch_shift

    logger.info("Analysing singer pitch in %s...", vocals_audio.name)
    singer = analyse_singer_pitch(vocals_audio)

    if "error" in singer:
        logger.error("Pitch analysis failed: %s", singer["error"])
        return None

    shift = compute_pitch_shift(singer, range_low_hz, range_high_hz)

    logger.info(
        "Singer median: %s (%s Hz) | Target center: %s (%s Hz)",
        singer["median_note"], singer["median_hz"],
        shift["target_range"]["center_note"], shift["target_range"]["center_hz"],
    )
    logger.info(
        "Recommended transpose: %+d semitones (%+d octaves)",
        shift["shift_semitones"], shift["shift_octaves"],
    )
    return {"singer": singer, "shift": shift}


def _run_vc_pipeline(
    input_file: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    f0_up_key: int,
    sep_backend: str,
    f0_method: str = "rmvpe",
    force: bool = False,
    pre_separated_vocals: Path | None = None,
    dereverb_backend: str | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run vc_pipeline.py as a subprocess with the computed transposition.

    When *pre_separated_vocals* is provided, it is passed as the input
    file with ``--no-separation`` so the track is not separated a second
    time.
    """
    if pre_separated_vocals is not None:
        vc_input = pre_separated_vocals
        skip_sep = True
    else:
        vc_input = input_file
        skip_sep = False

    cmd = [
        sys.executable, str(SCRIPT_DIR / "vc_pipeline.py"),
        "--input-file", str(vc_input),
        "--checkpoint-dir", str(checkpoint_dir),
        "--output-dir", str(output_dir),
        "--f0-up-key", str(f0_up_key),
        "--f0-method", f0_method,
        "--no-desilence",
        "--no-reassembly",
    ]
    if skip_sep:
        cmd.append("--no-separation")
    else:
        cmd.extend(["--sep-backend", sep_backend])
    if dereverb_backend:
        cmd.extend(["--dereverb-backend", dereverb_backend])
    if force:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running vc_pipeline: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Auto-transpose pipeline: sweep -> HNR analysis -> pitch match -> vc_pipeline",
    )
    p.add_argument("--sweep-audio", type=Path, required=True,
                   help="Vocal scale sweep test file (WAV)")
    p.add_argument("--input-file", type=Path, required=True,
                   help="Song to process through the VC pipeline")
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="RVC model checkpoint directory")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for final results")
    p.add_argument("--sep-backend", default="mdxchain",
                   choices=["demucs", "roformer", "mdxchain", "uvr5"],
                   help="Separation backend (default: mdxchain)")
    p.add_argument("--f0-method", default="rmvpe")
    p.add_argument("--hnr-method", default="autocorrelation",
                   choices=["autocorrelation", "cepstral", "spectral"])
    p.add_argument("--hnr-threshold", type=float, default=5.0,
                   help="HNR threshold in dB (default: 5.0)")
    p.add_argument("--device", default=None,
                   help="Torch device for separation (default: auto)")
    p.add_argument("--dereverb-backend", choices=["vrnet", "mbr"], default=None,
                   help="Dereverb backend passed to vc_pipeline.py (omit to skip)")
    p.add_argument("--override-transpose", type=int, default=None,
                   help="Skip auto-detection and use this semitone value directly")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore cached intermediates (sweep, separation) and regenerate them")
    p.add_argument("--force", action="store_true",
                   help="Regenerate final vc_pipeline output even if it already exists")
    args = p.parse_args()

    if args.no_cache:
        args.force = True

    sweep_dir = args.output_dir / "_sweep_analysis"

    # ── Step 1: RVC sweep ──────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  Step 1: Running sweep test through RVC")
    print(f"{'=' * 64}\n")

    vocals_path = None

    if args.override_transpose is not None:
        f0_up_key = args.override_transpose
        print(f"  Transpose override: {f0_up_key:+d} semitones (skipping steps 1-4)\n")
    else:
        sweep_wav = _run_rvc_sweep(
            args.sweep_audio, args.checkpoint_dir, sweep_dir,
            f0_method=args.f0_method,
            force=args.no_cache,
        )
        if sweep_wav is None:
            sys.exit(1)

        # ── Step 2: HNR analysis ───────────────────────────────────────
        print(f"\n{'=' * 64}")
        print("  Step 2: Analysing RVC model usable range (HNR)")
        print(f"{'=' * 64}\n")

        usable = _analyse_hnr(sweep_wav, args.hnr_method, args.hnr_threshold)
        if usable is None:
            sys.exit(1)

        print(f"  Usable range: {usable['low']} ({usable['low_hz']} Hz) — "
              f"{usable['high']} ({usable['high_hz']} Hz)\n")

        # ── Step 3: Separate vocals for pitch analysis ──────────────────
        print(f"{'=' * 64}")
        print("  Step 3: Separating vocals for pitch analysis")
        print(f"{'=' * 64}\n")

        vocals_path = _separate_vocals(
            args.input_file, sweep_dir,
            sep_backend=args.sep_backend,
            device=args.device,
            force=args.no_cache,
        )
        if vocals_path is None:
            sys.exit(1)

        # ── Step 4: Pitch match ────────────────────────────────────────
        print(f"{'=' * 64}")
        print("  Step 4: Computing transposition from separated vocals")
        print(f"{'=' * 64}\n")

        match = _compute_transposition(
            vocals_path, usable["low_hz"], usable["high_hz"],
        )
        if match is None:
            sys.exit(1)

        shift = match["shift"]
        singer = match["singer"]
        f0_up_key = shift["shift_semitones"]

        print(f"  Singer median:    {singer['median_note']} ({singer['median_hz']} Hz)")
        print(f"  Target center:    {shift['target_range']['center_note']} "
              f"({shift['target_range']['center_hz']} Hz)")
        print(f"  Transpose:        {f0_up_key:+d} semitones "
              f"({shift['shift_octaves']:+d} octaves)")
        print(f"  Exact offset:     {shift['shift_exact_semitones']:+.2f} semitones\n")

    # ── Step 5: Run vc_pipeline ────────────────────────────────────────
    print(f"{'=' * 64}")
    print(f"  Step 5: Running VC pipeline (f0_up_key={f0_up_key:+d})")
    print(f"{'=' * 64}\n")

    rc = _run_vc_pipeline(
        input_file=args.input_file,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        f0_up_key=f0_up_key,
        sep_backend=args.sep_backend,
        f0_method=args.f0_method,
        force=args.force,
        pre_separated_vocals=vocals_path,
        dereverb_backend=args.dereverb_backend,
    )

    if rc != 0:
        logger.error("vc_pipeline exited with code %d", rc)
        sys.exit(rc)

    print(f"\n{'=' * 64}")
    print("  Pipeline complete")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
