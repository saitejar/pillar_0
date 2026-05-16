# Pillar-0 HeadCT Mobile Runner

This repo packages the Pillar-0 HeadCT teacher-on-mobile work into two runnable
entrypoints:

```bash
make mac-infer
make iphone-infer
```

The model path preserves the Pillar-0 HeadCT input contract:

```text
input:  1 x 11 x 128 x 256 x 256 fp16 windowed HeadCT tensor
output: 1 x 1152 Pillar-0 image embedding
```

The iOS app and Mac runner currently validate **encoder equivalence**: they
compare the int8 Core ML embedding against the PyTorch/TorchScript teacher
embedding with cosine similarity and absolute-difference metrics. A clinical
yes/no classifier is a separate RATE linear probe on top of this embedding.

## Quick Start

Clone with submodules:

```bash
git clone --recurse-submodules <github-url>
cd <repo>
```

From the repo root:

```bash
make mac-infer
```

That runs the int8 Core ML package on macOS and compares it to the TorchScript
teacher using `CPU_ONLY` and `CPU_AND_NE`.

For the iPhone Simulator app:

```bash
make iphone-infer
```

That prepares the bundled resources, builds the Xcode app, installs it into the
booted simulator, launches it, waits for full-volume inference, and saves a
screenshot to:

```text
artifacts/pillar0_headct_coreml/iphone_simulator_latest.png
```

The simulator build uses DerivedData under `/tmp` by default so macOS
File Provider extended attributes from `Documents/` do not break codesigning.
Override with `PILLAR0_DERIVED_DATA=/path/to/DerivedData` if needed.

Double-click launchers are also included for local macOS use:

```text
run_mac_inference.command
run_iphone_inference.command
```

## Prerequisites

- macOS with Xcode command line tools.
- `uv`.
- Python dependencies already synced in `rate-evals`, or run with
  `PILLAR0_UV_ARGS=` to let `uv run` resolve/sync dependencies instead of using
  the default `--no-sync`.
- Hugging Face login with accepted access to `YalaLab/Pillar0-HeadCT` if you
  need to regenerate artifacts.
- Prepared HeadCT benchmark data if you need to export a fresh real sample.

The generated model/input artifacts are intentionally **not committed**:

```text
artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage
artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlmodelc
artifacts/pillar0_headct_coreml/pillar0_headct_vision_coreml.pt
artifacts/pillar0_headct_coreml/sample_windowed_headct.npy
artifacts/pillar0_headct_coreml/simulator_sample_windowed_headct_f16.raw
artifacts/pillar0_headct_coreml/simulator_expected_torchscript_f32.raw
```

They are large, and the source model is gated. The scripts use existing local
artifacts when present.

The scripts automatically apply [patches/rate-evals-mobile.patch](patches/rate-evals-mobile.patch)
to the `rate-evals` submodule. That patch removes the Mac-hostile `flash-attn`
dependency, switches default device selection from CUDA-only to `auto`, and
keeps CPU/MPS runs from using CUDA-only DataLoader settings.

## First-Time Artifact Build

If the artifacts are missing and you have HF access plus the prepared benchmark
data:

```bash
make artifacts
```

This checks Hugging Face access, exports one real Pillar-windowed HeadCT sample,
traces the Core ML friendly Pillar-0 vision encoder, converts to Core ML,
applies int8 weight quantization, compiles the `.mlmodelc`, and copies the iOS
resources into the Xcode app bundle directory.

If you already have prebuilt artifacts, place them under:

```text
artifacts/pillar0_headct_coreml/
```

Then run either one-command entrypoint.

## iOS App

The packaged Xcode app lives at:

```text
ios/test_pillar_0/test_pillar_0.xcodeproj
```

The app loads:

```text
Pillar0Resources/Pillar0HeadCTVision_int8.mlmodelc
Pillar0Resources/simulator_sample_windowed_headct_f16.raw
Pillar0Resources/simulator_expected_torchscript_f32.raw
```

Simulator builds use `.cpuOnly`. Real-device builds use `.cpuAndNeuralEngine`.
Do not default to `.all` yet; GPU delegation on this Mac produced numerically
wrong embeddings during validation.

## Current Known Good Result

Latest simulator run:

```text
cosine:        0.999575455
max abs diff: 0.004834097
mean abs diff: 0.000457874
finite:        true
output count: 1152
prediction:   66.75s on iPhone 15 simulator CPU-only
```

Packaged `make iphone-infer` run after moving DerivedData to `/tmp`:

```text
status:       passed
prediction:   56.74s on iPhone 15 simulator CPU-only
```

The simulator timing is not a real iPhone 13 performance measurement. It proves
the iOS app path and full input tensor path work.

## Useful Commands

```bash
make help
make package-check
make agreement-status
./scripts/build_pillar0_mobile_artifacts.sh
./scripts/run_pillar0_mac_inference.sh
./scripts/run_pillar0_iphone_simulator.sh
```

To target a specific simulator:

```bash
PILLAR0_SIM_UDID=<simulator-udid> make iphone-infer
```

To let `uv` sync dependencies instead of using the default `--no-sync`:

```bash
PILLAR0_UV_ARGS= make mac-infer
```

## Publishing

No GitHub remote is configured in this local checkout yet. After you create a
repo, add it and push:

```bash
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

Do not push the generated artifacts directly. Use GitHub Releases, Hugging Face,
or another artifact store if you want to distribute the prebuilt `.mlpackage`,
`.mlmodelc`, and sample input.
