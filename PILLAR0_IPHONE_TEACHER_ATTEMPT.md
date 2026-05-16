# Pillar-0 HeadCT Teacher on iPhone Attempt

Goal: run the released `YalaLab/Pillar0-HeadCT` teacher model with the same
Pillar HeadCT model input tensor:

```text
B x C x D x H x W = 1 x 11 x 128 x 256 x 256
```

This preserves the 3D volume, all CT windows, and Atlas/Pillar vision path. The
current Core ML graph starts after Pillar/RATE preprocessing, so the app must
provide the already-windowed tensor named `windowed_headct`.

## Artifacts

```text
artifacts/pillar0_headct_coreml/sample_windowed_headct.npy
artifacts/pillar0_headct_coreml/pillar0_headct_vision.pt
artifacts/pillar0_headct_coreml/pillar0_headct_vision_coreml.pt
artifacts/pillar0_headct_coreml/Pillar0HeadCTVision.mlpackage
artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_fp32.mlpackage
artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage
artifacts/pillar0_headct_coreml/int8_split_agreement_cpu_smoke.json
artifacts/pillar0_headct_coreml/int8_split_agreement_cpu_ne_smoke.json
artifacts/pillar0_headct_coreml/simulator_sample_windowed_headct_f16.raw
artifacts/pillar0_headct_coreml/simulator_expected_torchscript_f32.raw
artifacts/pillar0_headct_coreml/Pillar0SimulatorValidate
```

Sizes observed locally:

```text
Pillar0HeadCTVision.mlpackage        86M   fp16 input / fp16 compute
Pillar0HeadCTVision_fp32.mlpackage  172M   fp32 input / fp32 compute
Pillar0HeadCTVision_int8.mlpackage   44M   fp16 input / fp16 compute / int8 weights
sample_windowed_headct.npy          176M   1 x 11 x 128 x 256 x 256 fp16
```

## Reproduce

Run from the project root unless noted.

```bash
cd rate-evals
uv run --no-sync python ../scripts/export_pillar0_headct_sample_input.py
```

Load gated HF model files:

```bash
uv run --no-sync python -u ../scripts/export_pillar0_headct_coreml.py \
  --load-model-only \
  --local-files-only \
  --torch-dtype float16 \
  --trace-device cpu
```

Trace a Core ML friendly TorchScript graph. This keeps the model math equivalent
but replaces one `repeat_interleave` pattern with `unsqueeze + expand + reshape`
because Core ML's PyTorch frontend lowered the original repeat incorrectly.

```bash
uv run --no-sync python -u ../scripts/export_pillar0_headct_coreml.py \
  --skip-dry-run \
  --sample-input artifacts/pillar0_headct_coreml/sample_windowed_headct.npy \
  --trace-device cpu \
  --torch-dtype float32 \
  --trace-name pillar0_headct_vision_coreml.pt
```

Convert the iPhone-candidate fp16 package:

```bash
uv run --no-sync python -u ../scripts/export_pillar0_headct_coreml.py \
  --skip-trace \
  --convert-coreml \
  --trace-name pillar0_headct_vision_coreml.pt \
  --coreml-name Pillar0HeadCTVision.mlpackage \
  --coreml-input-dtype float16 \
  --coreml-compute-precision float16 \
  --minimum-deployment-target iOS16
```

Convert the higher-precision reference package:

```bash
uv run --no-sync python -u ../scripts/export_pillar0_headct_coreml.py \
  --skip-trace \
  --convert-coreml \
  --trace-name pillar0_headct_vision_coreml.pt \
  --coreml-name Pillar0HeadCTVision_fp32.mlpackage \
  --coreml-input-dtype float32 \
  --coreml-compute-precision float32 \
  --minimum-deployment-target iOS16
```

Convert and weight-quantize an int8-weight teacher package:

```bash
uv run --no-sync python -u ../scripts/export_pillar0_headct_coreml.py \
  --skip-trace \
  --convert-coreml \
  --trace-name pillar0_headct_vision_coreml.pt \
  --coreml-name Pillar0HeadCTVision.mlpackage \
  --coreml-input-dtype float16 \
  --coreml-compute-precision float16 \
  --minimum-deployment-target iOS16 \
  --quantize-coreml-weights
```

Validate a Core ML package against the TorchScript trace:

```bash
uv run --no-sync python ../scripts/validate_pillar0_coreml.py \
  --mlpackage artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage \
  --input-dtype float16 \
  --compute-units CPU_ONLY CPU_AND_NE
```

Validate across train/valid/test splits with explicit sample counts:

```bash
uv run --no-sync python ../scripts/validate_pillar0_coreml_splits.py \
  --mlpackage artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage \
  --samples-per-split 2 \
  --compute-units CPU_ONLY \
  --input-dtype float16 \
  --output artifacts/pillar0_headct_coreml/int8_split_agreement_cpu_smoke.json

uv run --no-sync python ../scripts/validate_pillar0_coreml_splits.py \
  --mlpackage artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage \
  --samples-per-split 1 \
  --compute-units CPU_AND_NE \
  --input-dtype float16 \
  --output artifacts/pillar0_headct_coreml/int8_split_agreement_cpu_ne_smoke.json
```

Run the long 100/100/100 gate in detached `screen` sessions:

```bash
scripts/run_int8_agreement_100_background.sh
screen -list
```

Run the standalone iPhone Simulator validation harness:

```bash
ios/run_pillar0_simulator_validation.sh
```

Current local Simulator status: the standalone runner compiles, the Core ML
package compiles to `Pillar0HeadCTVision_int8.mlmodelc`, and the full input
validation passes in the manually booted iPhone 15 simulator with CPU-only
Core ML:

```text
output_count: 1152
cosine_similarity: 0.99957545462993602
max_abs_diff: 0.0048340968787670135
mean_abs_diff: 0.00045787378996288279
prediction_sec: 78.96
```

Run the Xcode app validation:

```bash
cd /Users/saitejaranuva/code/shin.ai/ios/test_pillar_0
xcodebuild -project test_pillar_0.xcodeproj \
  -scheme test_pillar_0 \
  -destination 'platform=iOS Simulator,id=7F78C3E2-0578-4D11-98A8-AB12A6F0772F' \
  -configuration Debug build

xcrun simctl install booted \
  /Users/saitejaranuva/Library/Developer/Xcode/DerivedData/test_pillar_0-aoxltqpnnkhyavargmskzsclxdkt/Build/Products/Debug-iphonesimulator/test_pillar_0.app
xcrun simctl launch booted test-pillar-0.test-pillar-0
```

The app bundles the compiled int8 package and one full already-windowed Pillar
HeadCT tensor:

```text
test_pillar_0/Pillar0Resources/Pillar0HeadCTVision_int8.mlmodelc
test_pillar_0/Pillar0Resources/simulator_sample_windowed_headct_f16.raw
test_pillar_0/Pillar0Resources/simulator_expected_torchscript_f32.raw
```

Latest Xcode app simulator result:

```text
device: iPhone 15 simulator, iOS 17.5
computeUnits: cpuOnly
status: passed
cosine_similarity: 0.999575455
max_abs_diff: 0.004834097
mean_abs_diff: 0.000457874
finite: true
output_count: 1152
load_sec: 0.90
prediction_sec: 66.75
expected_norm: 1.000000
actual_norm: 1.000303
```

This confirms the same full Pillar HeadCT input can run through the quantized
teacher package in an iOS app. Simulator timing is a CPU-only sanity check, not
a real iPhone 13 latency measurement.

## Local Validation

Real sample export:

```text
shape: 1 x 11 x 128 x 256 x 256
dtype: float16
finite: true
range: [0, 1]
```

PyTorch teacher:

```text
HF model load: passed
MPS fp16: failed, MPSGraph dtype/broadcast issue
MPS fp32 + CPU fallback: passed, output 1 x 1152
CPU TorchScript trace: passed
```

Core ML runtime comparison against TorchScript on the same sample:

```text
Pillar0HeadCTVision.mlpackage, CPU_ONLY:
  cosine_similarity: 0.999957
  max_abs_diff: 0.001599

Pillar0HeadCTVision.mlpackage, CPU_AND_NE:
  cosine_similarity: 0.999990
  max_abs_diff: 0.000997

Pillar0HeadCTVision_fp32.mlpackage, CPU_ONLY:
  cosine_similarity: ~1.000000
  max_abs_diff: 2.31e-7

Pillar0HeadCTVision_fp32.mlpackage, CPU_AND_NE:
  cosine_similarity: ~1.000000
  max_abs_diff: 3.43e-7

Pillar0HeadCTVision_int8.mlpackage, CPU_ONLY:
  cosine_similarity: 0.999546
  max_abs_diff: 0.004580

Pillar0HeadCTVision_int8.mlpackage, CPU_AND_NE:
  cosine_similarity: 0.999563
  max_abs_diff: 0.004285

Pillar0HeadCTVision_int8.mlpackage, train/valid/test split smoke, CPU_ONLY:
  samples: 2 random per split, 6 total
  cosine range: 0.999290 to 0.999701
  max_abs_diff range: 0.003809 to 0.006863
  status: passed finite/cosine>=0.999/max_abs_diff<=0.01

Pillar0HeadCTVision_int8.mlpackage, train/valid/test split smoke, CPU_AND_NE:
  samples: 1 random per split, 3 total
  cosine range: 0.999474 to 0.999743
  max_abs_diff range: 0.003220 to 0.006558
  status: passed finite/cosine>=0.999/max_abs_diff<=0.01

CPU_AND_GPU / ALL:
  cosine_similarity: ~0.525
  status: do not use until debugged on target hardware
```

Use `.cpuAndNeuralEngine` or `.cpuOnly` in the iOS app. Do not use `.all` as
the default for this model yet, because Core ML delegated part of the graph to
GPU on Mac and produced a numerically wrong embedding.

The int8 package is still a candidate, not a full benchmark-certified package.
The full gate is train/valid/test or at least a much larger stratified sample,
then downstream RATE metric agreement on the generated embeddings.

As of 2026-05-16, the 100/100/100 CPU-only split agreement gate is running in
detached `screen` sessions. Current progress:

```text
train: 65 / 100, failures: 0, min cosine: 0.999129713, max abs diff: 0.006921856
valid: 45 / 100, failures: 0, min cosine: 0.999428749, max abs diff: 0.006146483
test:  45 / 100, failures: 0, min cosine: 0.999354780, max abs diff: 0.006298747
```

## Remaining Work

1. Drop `Pillar0HeadCTVision_int8.mlpackage` into a real iOS app and run on
   iPhone 13 with `.cpuAndNeuralEngine`.
2. Port the raw CT/RVE preprocessing to Swift/Metal if the app must start from
   raw volume data. The converted model currently accepts the already-windowed
   Pillar tensor.
3. Run several samples through PyTorch and Core ML and compare embeddings, not
   only one sample.
4. If device latency/memory is still too high, move to distillation/QAT while
   keeping this teacher package as the reference.
