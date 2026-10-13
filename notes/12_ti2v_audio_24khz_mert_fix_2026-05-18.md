# TI2V Audio 24kHz MERT Fix

Date: 2026-05-18

## Goal

Make `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh` use a true 24kHz waveform path for MERT during training instead of downsampling through the generic 16kHz audio loader.

## Problem

Before this change:

- `build_wan_dataset(...)` always used `CachedLoadAudio(sr=16000)`
- cached WAVs were written under `audio_cache/wavs/`
- TI2V-Audio training therefore fed 16kHz waveforms into the MERT path
- `WanVideoUnit_Audio` also had no way to distinguish a loader-provided sample rate from the default value path

This meant the TI2V-Audio training chain did not actually preserve MERT's native 24kHz input assumption.

## Changes

### 1. MERT-aware dataset audio loader selection

File: `examples/wanvideo/model_training/train.py`

- detect `use_mert_audio = args.mert_encoder_path is not None`
- use `CachedLoadAudio(sr=24000, return_sample_rate=True)` for that path
- keep the previous `16kHz` behavior for non-MERT tasks
- write MERT waveform cache into `audio_cache/wavs_24k/` to avoid accidentally reusing old 16kHz cached WAVs

### 2. Cached loader now optionally returns sample rate

File: `diffsynth/core/data/operators.py`

- `CachedLoadAudio` now supports `return_sample_rate`
- default return format is unchanged: `(audio, cache_key)`
- MERT path can now return `(audio, sample_rate, cache_key)`

### 3. WanVideoUnit_Audio now consumes explicit sample rate

File: `diffsynth/pipelines/wan_video.py`

- tuple unpacking now accepts:
  - `(audio, cache_key)`
  - `(audio, sample_rate, cache_key)`
- when sample rate is present, it overrides the fallback/default path before calling MERT

## Validation

Validation performed locally after patching:

- confirmed `CachedLoadAudio(sr=24000, return_sample_rate=True)` is used for MERT-enabled training builds
- confirmed MERT-specific WAV cache directory is `wavs_24k`
- confirmed `WanVideoUnit_Audio` now accepts loader tuples carrying the explicit sample rate

No end-to-end training job was launched in this change record.

## Expected Behavior After Fix

For `Wan2.2-TI2V-5B-Audio` training:

- raw/cached waveform entering training data flow is 24kHz
- cached waveform is stored separately under `audio_cache/wavs_24k/`
- `WanVideoUnit_Audio` forwards `audio_sample_rate=24000` into `MERTAudioEncoder`
- MERT no longer receives a silently downsampled 16kHz waveform from the generic loader path

## Notes

- Existing `audio_cache/wavs/` contents remain untouched and continue to serve non-MERT tasks.
- Existing `mert_features/` cache may still contain features computed from the old 16kHz path. If training must be fully clean after this change, regenerate or clear the old MERT feature cache before the next run.
