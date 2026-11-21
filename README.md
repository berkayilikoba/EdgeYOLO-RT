# EdgeYOLO-RT

**CPU-Constrained, Embedded-Aware Object Detection: Model Optimization, C++ Streaming Inference, and Deadline-Aware Benchmarking**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-supported-lightgrey)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-lightgrey)
![Code License](https://img.shields.io/badge/code%20license-MIT-green)

> **Status: work in progress.** Results are added only after they are measured. See [`ROADMAP.md`](ROADMAP.md).
>
> **Scope note:** All measurements are taken on general-purpose CPU hardware with software-imposed limits, not on a physical embedded device. The project applies embedded engineering practices (memory discipline, bounded queues, deadline metrics, ARM cross-compilation) and is designed so the benchmark can later be repeated on a real target. See [Hardware and scope](#hardware-and-scope).

---

## Research question

> Which YOLO variant and deployment configuration gives the best accuracy under a real-time budget on constrained CPU compute, and does the system actually meet its frame deadlines under sustained streaming?

Most edge-AI projects report average FPS and mAP. A deployed system also has to answer: what is the **worst-case** latency, how much **memory** does it hold, how fast does it **cold start**, and how often does it **miss a frame deadline**? EdgeYOLO-RT measures all of these.

## What it does

1. **Trains** YOLOv8 variants on a small detection dataset (deep learning).
2. **Optimizes** them for constrained inference: input scaling, ONNX export, INT8 quantization, optional pruning or distillation.
3. **Deploys** them in a C++17 streaming inference application (no Python at runtime) with preallocated buffers, a bounded frame queue and a drop policy.
4. **Benchmarks** every variant under CPU and memory limits, reporting accuracy, latency percentiles, memory footprint, cold-start time, jitter and deadline-miss rate.

## Embedded-oriented design

| Concern | Approach |
|---|---|
| Memory | Preallocated input/output buffers; no per-frame dynamic allocation in the hot loop (verified by measurement) |
| Real-time behavior | Fixed frame budget (e.g. 33 ms at 30 FPS); deadline-miss rate and jitter reported |
| Streaming | Capture thread and inference thread connected by a bounded queue; configurable drop policy (drop oldest / drop newest) |
| Footprint | Peak RSS, steady-state RSS, model file size, cold-start time, long-run leak check |
| Portability | Docker builds for x86-64 and aarch64 (cross-compiled in CI, build verification only) |
| Constraints | Container CPU and memory limits, core pinning, fixed CPU frequency where possible |

## Pipeline

```
Video / camera frames
        │
  [Capture thread] ──► bounded queue (drop policy) ──► [Inference thread]
                                                          │
                                  preprocess → ONNX Runtime → decode + NMS
                                                          │
                                           detections + per-stage timestamps
                                                          │
                                   metrics logger (latency, jitter, misses)
```

## Compared variants

| ID | Model | Format / precision | Input size |
|---|---|---|---|
| v0 | YOLOv8s | PyTorch FP32 | 640 |
| v1 | YOLOv8n | PyTorch FP32 | 640 |
| v2 | YOLOv8n | PyTorch FP32 | 480 / 320 |
| v3 | YOLOv8n | ONNX FP32 | best of v2 |
| v4 | YOLOv8n | ONNX INT8 (static quantization) | best of v2 |
| v5 | YOLOv8n | Pruned or distilled (optional) | best of v2 |

## Metrics

| Group | Metrics |
|---|---|
| Accuracy | mAP@0.5, mAP@0.5:0.95, small-object AP |
| Latency | p50 / p95 / p99 end-to-end (preprocess + inference + NMS), per-stage breakdown |
| Real-time | Deadline-miss rate at 15 / 30 FPS budgets, jitter, dropped frames |
| Resources | Peak RSS, steady-state RSS, model size, parameters, FLOPs, cold-start time |
| Constraint levels | Unconstrained, 4 cores, 2 cores / 2 GB |

The main figure is a Pareto plot: mAP vs. p95 latency, marker size = model size, with a second panel showing deadline-miss rate.

## Hardware and scope

| Item | Value |
|---|---|
| Measurement machine | _CPU model, core count, RAM, OS (fill in)_ |
| Environment | _Local machine / dedicated VM / shared VM (fill in)_ |
| Frequency policy | _Boost on/off, governor (fill in)_ |
| Constraint method | Docker `--cpus` / `--memory`, `taskset` |

**What the measurements can support**

- Relative comparisons between variants on the same machine (e.g. "INT8 reduces deadline misses by X compared with FP32 under the same constraint").
- Memory footprint, cold-start behavior, allocation behavior and leak checks.
- Accuracy impact of quantization and input scaling.

**What they cannot support**

- Absolute latency or FPS on any physical embedded board.
- Power, energy or thermal behavior.
- Hardware accelerator (NPU, GPU, DSP) behavior or camera/sensor interfacing.

If measurements are taken on a shared cloud VM, results can fluctuate because of neighboring workloads; in that case tail-latency numbers (p99, jitter) are reported as indicative only, with the number of repetitions stated.

## Results

Results are published here after they are measured. Each table states hardware, constraints, dataset, number of runs and warm-up policy.

| Experiment | Status |
|---|---|
| Baseline training (v0, v1) | not started |
| Input-size sweep | not started |
| ONNX / INT8 accuracy impact | not started |
| Constrained-CPU benchmark | not started |
| C++ streaming runtime | not started |
| Deadline / jitter analysis | not started |
| Memory footprint and leak check | not started |
| aarch64 build in CI | not started |

## Repository structure

```
edgeyolo-rt/
├── README.md
├── ROADMAP.md
├── configs/              # dataset, training, benchmark settings
├── train/                # training and evaluation scripts
├── export/               # ONNX export, quantization, parity checks
├── benchmark/            # Python benchmark harness, plotting
├── cpp/
│   ├── include/ src/     # streaming runtime (capture, queue, inference)
│   ├── tests/            # GoogleTest unit tests
│   └── CMakeLists.txt
├── docker/               # x86-64 and aarch64 images
├── docs/
│   ├── methodology.md
│   ├── measurement_protocol.md
│   ├── hardware.md       # measurement machine, frequency policy, constraints
│   ├── limitations.md
│   └── experiments.md    # includes what did not work
└── results/              # raw logs, configs, seeds
```

## Quick start

> Commands will be finalized as scripts are written.

```bash
git clone https://github.com/<your-username>/edgeyolo-rt.git
cd edgeyolo-rt
pip install -e .

# Train and evaluate
python train/train.py --config configs/yolov8n_640.yaml

# Export and quantize
python export/to_onnx.py --weights runs/yolov8n/best.pt
python export/quantize_int8.py --onnx models/yolov8n.onnx --calib data/calib

# Build and run the C++ runtime
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/edge_infer --model models/yolov8n_int8.onnx --input demo.mp4 --fps 30
```

## Measurement protocol (summary)

- Warm-up frames discarded before timing; at least several hundred measured frames per run.
- Batch size 1, fixed input resolution, identical preprocessing and NMS settings across variants.
- CPU limits via container flags and core pinning; CPU model and frequency behavior recorded in `docs/hardware.md`.
- Background applications closed; laptops plugged in with a fixed power plan.
- Each configuration repeated; median and spread reported.
- Full protocol in [`docs/measurement_protocol.md`](docs/measurement_protocol.md).

## Limitations

- Measurements are taken on general-purpose hardware with software-imposed limits. This approximates, but does not replace, a physical embedded target. No power or thermal measurements are claimed.
- The aarch64 build is verified for compilation only (emulated); emulated timings are not reported as performance.
- Standard Linux scheduling is not a hard real-time system; deadline metrics describe soft real-time behavior.
- Single small dataset; results may not transfer to other domains.

See [`docs/limitations.md`](docs/limitations.md).

## Future work: real-device validation

The code and configs are structured so that the benchmark stage can be repeated on a physical target (for example a Raspberry Pi or a Jetson-class board) without changing the runtime. Real-device results, if added, will be reported in a separate table with the board, power mode and cooling stated.

## Licenses

- EdgeYOLO-RT's own code: MIT.
- **Ultralytics YOLOv8 is licensed under AGPL-3.0.** Review its terms before redistributing trained weights or building on this repository commercially.
- Datasets keep their own licenses and are not included in this repository.

## References

- Jocher et al., *Ultralytics YOLOv8*
- Lin et al., *Microsoft COCO: Common Objects in Context*
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*
- Han et al., *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding*
- Hinton et al., *Distilling the Knowledge in a Neural Network*
- ONNX Runtime documentation

## Author

**<Your Name>**, Computer Engineering student. [LinkedIn](#) · [Email](#)
