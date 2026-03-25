# vc_with_preprocessing

End-to-end pipeline for generating voice-converted audio with diverse
preprocessing artefacts, designed for SVDD (Singing Voice Deepfake Detection)
dataset generation.

## Pipeline stages

```
Input audio
  │
  ├─ 1. Separation ──────── vocals + instrumentals
  │       (demucs | roformer | mdxchain | uvr5)
  │
  ├─ 2. Dereverberation ─── dry vocals  (stochastic, 53 % by default)
  │       (vrnet | mbr | skip)
  │
  ├─ 3. Desilencing ─────── vocal segments + timestamp CSV
  │
  ├─ 4. RVC conversion ──── voice-converted segments
  │
  ├─ 5. Reassembly ──────── converted vocals overlaid on instrumentals
  │
  └─ 6. Noise evaluation ── optional HNR / CREPE quality gate
```

## Environment setup

### Local (macOS / Linux)

```bash
# Create a dedicated conda environment (Python 3.11 recommended)
conda create -n vc_preproc python=3.11 -y
conda activate vc_preproc

# Install av + ffmpeg via conda (avoids C compilation issues)
conda install -c conda-forge av ffmpeg -y

# Install PyTorch
pip install torch torchaudio torchcodec

# Install remaining dependencies
pip install -r requirements.txt

# fairseq 0.12.2 has a known dataclass bug with Python 3.11+.
# Install a patched fork instead:
pip install git+https://github.com/One-sixth/fairseq.git --no-deps

# macOS: if you see OpenMP duplicate-library errors, set:
export KMP_DUPLICATE_LIB_OK=TRUE
```

### EC2 (GPU)

Recommended instance: **g4dn.xlarge** (T4 GPU, 16 GB VRAM) or larger.
CPU-only instances work but separation and dereverberation will be 10–20x
slower.

```bash
# System dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg

# Python environment
conda create -n vc_preproc python=3.11 -y
conda activate vc_preproc
conda install -c conda-forge av ffmpeg -y

# Install PyTorch with CUDA:
pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu121

# Install the rest:
pip install -r requirements.txt

# Patched fairseq for Python 3.11+:
pip install git+https://github.com/One-sixth/fairseq.git --no-deps

# GPU ONNX acceleration (used by audio-separator):
pip install onnxruntime-gpu
```

All separation and dereverberation model weights are downloaded
automatically on first use to `models/audio-separator/`.

## Separation backends

| Backend | Architecture | Models | Notes |
|---------|-------------|--------|-------|
| `demucs` | Hybrid transformer/CNN (htdemucs_ft) | 1 | 4-stem model, sums non-vocal stems |
| `roformer` | MelBandRoformer (transformer) | 1 | High quality, ~3 min/track on CPU |
| `mdxchain` | MDX23C → MDXNET KARA (two-stage) | 2 | Isolates lead vocals from backing |
| `uvr5` | UVR5 VR-Net (CNN) | 1 | Fastest (~15 s/track on CPU) |

## Dereverberation backends

| Backend | Architecture | Model | Notes |
|---------|-------------|-------|-------|
| `vrnet` | CascadedNet (CNN + LSTM) | UVR-DeEcho-DeReverb.pth | Fast, good for echo + reverb |
| `mbr` | MelBandRoformer (transformer) | anvuew SDR 19.17 | Higher quality, slower |

Both apply a NoiseGate (−40 dB) after dereverberation.

## Usage examples

### Standalone separation

```bash
# Separate with a specific backend
python separation.py input.wav --backend roformer

# Separate with custom output paths
python separation.py input.mp3 --backend mdxchain \
    --vocals-out vocals.wav --instrumental-out instrumental.wav
```

### Standalone dereverberation

```bash
# MelBandRoformer dereverberation (default)
python dereverberation.py vocals.wav --backend mbr --force

# VR-Net dereverberation
python dereverberation.py vocals.wav --backend vrnet --force
```

### Full pipeline (single track)

```bash
python pipeline.py \
    --input-file "song.wav" \
    --checkpoint-dir /path/to/rvc_model/ \
    --output-dir /output/ \
    --sep-backend roformer \
    --dereverb-backend mbr \
    --dereverb-probability 0.53 \
    --f0-up-key 0 \
    --f0-method rmvpe
```

### Full pipeline (batch)

```bash
python pipeline.py \
    --input-dir /data/tracks/ \
    --checkpoint-dir /path/to/rvc_model/ \
    --output-dir /output/ \
    --glob "*.wav" \
    --dereverb-backend vrnet \
    --noise-eval
```

### S3 integration (for EC2)

```bash
python pipeline.py \
    --input-dir s3://bucket/input-tracks/ \
    --checkpoint-dir s3://bucket/rvc-checkpoints/singer-a/ \
    --output-dir s3://bucket/output/ \
    --sep-backend mdxchain \
    --dereverb-backend mbr
```

## Randomization strategies

The pipeline is designed to produce diverse artefacts so downstream
detectors learn to be invariant to processing choices. There are several
ways to introduce controlled randomness:

### 1. Built-in random selection

When `--sep-backend` is omitted, the pipeline already picks uniformly at
random from all four separation backends. Dereverberation is already
stochastic — applied with probability 0.53 by default (adjustable via
`--dereverb-probability`).

### 2. Weighted backend selection (recommended)

Wrap the pipeline in a launcher script that samples parameters from
weighted distributions:

```python
import random, subprocess

SEP_WEIGHTS = {
    "roformer": 0.35,
    "mdxchain": 0.30,
    "demucs":   0.20,
    "uvr5":     0.15,
}

DEREVERB_WEIGHTS = {
    "mbr":   0.30,
    "vrnet": 0.20,
    None:    0.50,   # 50 % chance of skipping dereverberation entirely
}

sep = random.choices(list(SEP_WEIGHTS), weights=SEP_WEIGHTS.values())[0]
dereverb = random.choices(list(DEREVERB_WEIGHTS), weights=DEREVERB_WEIGHTS.values())[0]

cmd = [
    "python", "pipeline.py",
    "--input-file", "song.wav",
    "--checkpoint-dir", "/path/to/rvc/",
    "--output-dir", "/output/",
    "--sep-backend", sep,
]
if dereverb:
    cmd += ["--dereverb-backend", dereverb]

subprocess.run(cmd)
```

### 3. Config-file driven randomization

Define a YAML config with per-stage probability distributions:

```yaml
separation:
  weights: { roformer: 0.35, mdxchain: 0.30, demucs: 0.20, uvr5: 0.15 }

dereverberation:
  apply_probability: 0.53
  weights: { mbr: 0.6, vrnet: 0.4 }

desilencing:
  min_silence_len: [1500, 2000, 2500]   # sample uniformly
  silence_thresh: [-45, -40, -35]

rvc:
  f0_up_key: [-2, -1, 0, 1, 2]          # random pitch shift
  f0_method: [rmvpe, crepe]
```

Then write a small driver that loads the config, samples each parameter,
and invokes the pipeline. This keeps randomization logic separate from
the core pipeline code.

### 4. Per-parameter jitter

For continuous parameters, add small perturbations rather than discrete
choices:

```python
import random

dereverb_prob = 0.53 + random.gauss(0, 0.1)  # jitter around 0.53
dereverb_prob = max(0.0, min(1.0, dereverb_prob))

silence_thresh = -40 + random.randint(-5, 5)  # dB jitter
```

### 5. Multi-checkpoint randomization

If you have multiple RVC checkpoints (different speakers / models), the
launcher can also randomize which voice is used:

```python
import random
from pathlib import Path

checkpoints = list(Path("/rvc-models/").iterdir())
ckpt = random.choice(checkpoints)
```

This produces maximum diversity in the training data since each track
gets a random combination of separation method, dereverberation
treatment, silence parameters, and target voice.
