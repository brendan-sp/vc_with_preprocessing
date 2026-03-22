#!/usr/bin/env python3
"""End-to-end SVDD dataset generation pipeline.

For each input audio track this pipeline:

  1. **Separates** vocals and instrumentals using a randomly chosen backend
     (Demucs or Roformer) to introduce diverse separation artefacts.
  2. **Dereverberation** is applied to the separated vocals with 53%
     probability (configurable) so the SVDD model sees both reverberant and
     dry examples.
  3. **Desilencing** splits the vocals on silence, recording segment
     timestamps in a CSV file co-located with the audio.
  4. **RVC conversion** runs each vocal segment through the target voice
     checkpoint.
  5. **Reassembly** places converted segments at their original timestamps
     and **mixes** with the instrumental track.
  6. (Optional) **Noise evaluation** flags outputs whose HNR or CREPE
     confidence fall below quality thresholds.

All outputs for a given track are placed in a single directory:
    {output_root}/source_{audio_md5}_rvcmodel_{rvc_md5}/
containing the final wav, timestamps CSV, and per-track metadata JSON.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment

from separation import SeparationBackend, separate
from dereverberation import DEREVERB_PROBABILITY, Dereverberation, maybe_dereverb
from desilence import desilence_and_track, reassemble_from_segments, VocalSegment
from rvc_convert import (
    convert_segments,
    find_checkpoint_files,
    md5_of_file,
    md5_of_string,
)
from noise_eval import evaluate_noise

try:
    import boto3
    _s3 = boto3.client("s3")
except ImportError:
    _s3 = None

logger = logging.getLogger(__name__)


def _md5_of_audio(audio_path: Path) -> str:
    return md5_of_file(audio_path)


def _output_stem(source_md5: str, rvc_md5: str) -> str:
    return f"source_{source_md5}_rvcmodel_{rvc_md5}"


def process_track(
    input_audio: Path,
    checkpoint_dir: Path,
    output_root: Path,
    *,
    rvc_md5: Optional[str] = None,
    sep_backend: Optional[SeparationBackend] = None,
    dereverb_model: Optional[Dereverberation] = None,
    dereverb_probability: float = DEREVERB_PROBABILITY,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    noise_eval: bool = False,
    hnr_threshold_db: float = 10.0,
    crepe_threshold: float = 0.6,
    device: Optional[str] = None,
    min_silence_len: int = 2000,
    silence_thresh: int = -40,
    keep_silence: int = 100,
    min_segment_len: int = 3000,
) -> Optional[dict]:
    """Run the full pipeline on a single audio track.

    Returns a metadata dict on success, None on failure.
    """
    source_md5 = _md5_of_audio(input_audio)

    pth_candidates, best_index = find_checkpoint_files(checkpoint_dir)
    if not pth_candidates:
        logger.error("No .pth checkpoint found in %s", checkpoint_dir)
        return None

    if rvc_md5 is None:
        rvc_md5 = md5_of_file(pth_candidates[0])

    stem = _output_stem(source_md5, rvc_md5)
    output_dir = output_root / stem
    final_wav = output_dir / f"{stem}.wav"

    if final_wav.exists():
        logger.info("Already processed, skipping: %s", final_wav.name)
        return {"output": str(final_wav), "source_md5": source_md5, "rvc_md5": rvc_md5, "skipped": True}

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="svdd_") as tmpdir:
        tmp = Path(tmpdir)

        # --- 1. Separation ---
        vocals_raw = tmp / "vocals_raw.wav"
        instrumental = tmp / "instrumental.wav"
        try:
            _, _, backend_used = separate(
                input_audio, vocals_raw, instrumental,
                backend=sep_backend, device=device,
            )
        except Exception as e:
            logger.error("Separation failed for %s: %s", input_audio.name, e)
            return None

        # --- 2. Dereverberation (53% probability) ---
        vocals_processed = tmp / "vocals_processed.wav"
        try:
            _, dereverb_applied = maybe_dereverb(
                vocals_raw, vocals_processed,
                dereverb_model=dereverb_model,
                probability=dereverb_probability,
            )
        except Exception as e:
            logger.warning("Dereverberation failed, using raw vocals: %s", e)
            shutil.copy2(vocals_raw, vocals_processed)
            dereverb_applied = False

        # --- 3. Desilence + timestamp tracking ---
        segments_dir = tmp / "segments"
        seg_paths, segments, csv_path = desilence_and_track(
            vocals_processed, segments_dir,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
            min_segment_len=min_segment_len,
        )
        if not seg_paths:
            logger.warning("No vocal segments found for %s", input_audio.name)
            return None

        # Copy timestamps CSV to output directory
        output_csv = output_dir / "timestamps.csv"
        shutil.copy2(csv_path, output_csv)

        # --- 4. RVC conversion ---
        converted_dir = tmp / "converted"
        converted = convert_segments(
            seg_paths, converted_dir,
            pth_path=pth_candidates[0],
            index_path=best_index,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
        )
        if not converted:
            logger.error("RVC conversion produced no output for %s", input_audio.name)
            return None

        # --- 5. Reassemble + mix with instrumental ---
        instrumental_audio = AudioSegment.from_file(str(instrumental))
        total_duration_ms = len(instrumental_audio)

        reassembled_vocals = reassemble_from_segments(
            converted_dir, segments, total_duration_ms,
        )

        mixed = instrumental_audio.overlay(reassembled_vocals)
        mixed.export(str(final_wav), format="wav")
        logger.info("Final mix exported: %s", final_wav.name)

    # --- 6. Noise evaluation (optional) ---
    noise_result = None
    if noise_eval:
        try:
            noise_result = evaluate_noise(
                final_wav,
                hnr_threshold_db=hnr_threshold_db,
                crepe_threshold=crepe_threshold,
            )
        except Exception as e:
            logger.warning("Noise evaluation failed: %s", e)

    # --- Write per-track metadata ---
    metadata = {
        "source_audio": str(input_audio),
        "source_md5": source_md5,
        "rvc_checkpoint_dir": str(checkpoint_dir),
        "rvc_md5": rvc_md5,
        "separation_backend": backend_used.value,
        "dereverberation_applied": dereverb_applied,
        "dereverb_probability": dereverb_probability,
        "f0_up_key": f0_up_key,
        "f0_method": f0_method,
        "num_segments": len(segments),
        "num_converted": len(converted),
        "output_wav": str(final_wav),
        "timestamps_csv": str(output_csv),
    }
    if noise_result:
        metadata["noise_eval"] = noise_result.details
        metadata["noise_is_clean"] = noise_result.is_clean

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written to %s", meta_path)

    metadata["_local_output_dir"] = str(output_dir)
    return metadata


def upload_to_s3(local_output_root: Path, stem: str, s3_dest: str) -> List[str]:
    """Upload all files in the ``{stem}/`` directory to S3.

    Args:
        local_output_root: parent directory containing the ``{stem}/`` folder.
        stem: the ``source_{md5}_rvcmodel_{md5}`` name.
        s3_dest: S3 URI prefix, e.g. ``s3://bucket/prefix/``.

    Returns:
        list of uploaded S3 URIs.
    """
    if _s3 is None:
        raise RuntimeError("boto3 is not installed; cannot upload to S3")

    if not s3_dest.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_dest}")

    parts = s3_dest[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    uploaded: List[str] = []

    subdir = local_output_root / stem
    if subdir.is_dir():
        for f in subdir.iterdir():
            if not f.is_file():
                continue
            key = f"{prefix}/{stem}/{f.name}" if prefix else f"{stem}/{f.name}"
            _s3.upload_file(str(f), bucket, key)
            uri = f"s3://{bucket}/{key}"
            uploaded.append(uri)
            logger.info("Uploaded %s -> %s", f.name, uri)

    return uploaded


def process_batch(
    input_dir: Path,
    checkpoint_dir: Path,
    output_root: Path,
    *,
    glob_pattern: str = "*.wav",
    dereverb_model_path: Optional[str] = None,
    dereverb_params_path: Optional[str] = None,
    dereverb_device: str = "cuda",
    **kwargs,
) -> List[dict]:
    """Process all audio files in a directory."""
    audio_files = sorted(input_dir.glob(glob_pattern))
    if not audio_files:
        logger.warning("No audio files found matching %s in %s", glob_pattern, input_dir)
        return []

    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %d tracks from %s", len(audio_files), input_dir)

    dereverb = None
    if dereverb_model_path and dereverb_params_path:
        dereverb = Dereverberation(
            model_path=dereverb_model_path,
            params_path=dereverb_params_path,
            device=dereverb_device,
        )
        dereverb.load()

    results = []
    for audio_path in audio_files:
        try:
            result = process_track(
                audio_path, checkpoint_dir, output_root,
                dereverb_model=dereverb,
                **kwargs,
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error("Failed to process %s: %s", audio_path.name, e, exc_info=True)

    logger.info("Batch complete: %d / %d succeeded", len(results), len(audio_files))
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="SVDD dataset generation pipeline: separate -> dereverb -> desilence -> RVC -> remix",
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path, help="Directory of source audio tracks")
    input_group.add_argument("--input-file", type=Path, help="Single source audio track")
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="RVC checkpoint directory (.pth + .index)")
    p.add_argument("--output-dir", type=Path, required=True, help="Output root directory")
    p.add_argument("--glob", default="*.wav", help="Glob pattern for input files (default *.wav)")
    p.add_argument("--s3-dest", default=None, help="S3 URI prefix to upload outputs (e.g. s3://bucket/prefix/)")

    sep_group = p.add_argument_group("separation")
    sep_group.add_argument("--sep-backend", choices=["demucs", "roformer"], default=None,
                           help="Force separation backend (default: random)")
    sep_group.add_argument("--device", default=None, help="Torch device (default: auto)")

    dereverb_group = p.add_argument_group("dereverberation")
    dereverb_group.add_argument("--dereverb-model-path", default=None,
                                help="Path to VR-DeEchoDeReverb.pth (omit to skip dereverberation)")
    dereverb_group.add_argument("--dereverb-params-path", default=None,
                                help="Path to 4band_v3.json params file")
    dereverb_group.add_argument("--dereverb-device", default="cuda")
    dereverb_group.add_argument("--dereverb-probability", type=float, default=DEREVERB_PROBABILITY)

    silence_group = p.add_argument_group("desilencing")
    silence_group.add_argument("--min-silence-len", type=int, default=2000)
    silence_group.add_argument("--silence-thresh", type=int, default=-40)
    silence_group.add_argument("--keep-silence", type=int, default=100)
    silence_group.add_argument("--min-segment-len", type=int, default=3000)

    rvc_group = p.add_argument_group("rvc")
    rvc_group.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones")
    rvc_group.add_argument("--f0-method", default="rmvpe")

    eval_group = p.add_argument_group("noise evaluation")
    eval_group.add_argument("--noise-eval", action="store_true", help="Run HNR/CREPE noise evaluation on outputs")
    eval_group.add_argument("--hnr-threshold", type=float, default=10.0)
    eval_group.add_argument("--crepe-threshold", type=float, default=0.6)

    args = p.parse_args()

    sep_backend = SeparationBackend(args.sep_backend) if args.sep_backend else None
    common_kwargs = dict(
        sep_backend=sep_backend,
        dereverb_probability=args.dereverb_probability,
        device=args.device,
        f0_up_key=args.f0_up_key,
        f0_method=args.f0_method,
        noise_eval=args.noise_eval,
        hnr_threshold_db=args.hnr_threshold,
        crepe_threshold=args.crepe_threshold,
        min_silence_len=args.min_silence_len,
        silence_thresh=args.silence_thresh,
        keep_silence=args.keep_silence,
        min_segment_len=args.min_segment_len,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        dereverb = None
        if args.dereverb_model_path and args.dereverb_params_path:
            dereverb = Dereverberation(
                model_path=args.dereverb_model_path,
                params_path=args.dereverb_params_path,
                device=args.dereverb_device,
            )
            dereverb.load()

        result = process_track(
            args.input_file, args.checkpoint_dir, args.output_dir,
            dereverb_model=dereverb,
            **common_kwargs,
        )
        results = [result] if result else []
    else:
        results = process_batch(
            input_dir=args.input_dir,
            checkpoint_dir=args.checkpoint_dir,
            output_root=args.output_dir,
            glob_pattern=args.glob,
            dereverb_model_path=args.dereverb_model_path,
            dereverb_params_path=args.dereverb_params_path,
            dereverb_device=args.dereverb_device,
            **common_kwargs,
        )

    if args.s3_dest and results:
        for r in results:
            if r.get("skipped"):
                continue
            stem = _output_stem(r["source_md5"], r["rvc_md5"])
            try:
                s3_uris = upload_to_s3(args.output_dir, stem, args.s3_dest)
                r["s3_uris"] = s3_uris
            except Exception as e:
                logger.error("S3 upload failed for %s: %s", stem, e)

    summary_path = args.output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDone. {len(results)} tracks processed. Summary: {summary_path}")
