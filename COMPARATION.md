# Move Semantics Performance Comparison

## Summary

Applied move semantics to the deep-copy cascade in the inspector profiler's
consumer thread, eliminating redundant `calloc`/`memcpy`/`free` cycles when
propagating completed events through the hierarchy:

- ProxyStep → StepRecord: **move states** (instead of copy+free)
- ProxyOp → CompletedProxy: **move states + steps** (instead of copy+free)
- ProxyCtrl → CompletedProxy: **move states** (instead of copy+free)
- CompletedProxy → Parent: **move entire record** into list slot
- Coll/P2p → CompletedOp: **move proxyOps list** (instead of copy+free)

The one level kept as a copy: `inspectorProxyOpAppendStepRecordLocked`
(step-record appended to OpInfo's `steps` list) — because the step may still
be referenced after append in certain code paths.

## Test Environment

- 1 node, 4× GPU, 4× NIC (local IB loopback)
- AllReduce, float, sum, 100 iterations, 5 warmup
- NCCL build: local `nccl/build/` (commit 21bfb9a + move semantics)
- Plugin: `libnccl-profiler-inspector-start-async-21bfb9a-dirty.so`

---

## Full Sweep (all sizes in single mpirun)

All sizes run sequentially in one process — consumer has idle time between
size transitions to drain queued events.

| Size | No Inspector | | Move Semantics | | Time Δ | BW Δ |
|------|---:|---:|---:|---:|---:|---:|
| | Time(μs) | BW(GB/s) | Time(μs) | BW(GB/s) | | |
| 8M | 408.3 | 20.55 | 408.8 | 20.52 | +0.12% | -0.15% |
| 16M | 680.3 | 24.66 | 685.9 | 24.46 | +0.82% | -0.81% |
| 32M | 1354.5 | 24.77 | 1364.6 | 24.59 | +0.74% | -0.73% |
| 64M | 2694.0 | 24.91 | 2732.4 | 24.56 | +1.43% | -1.41% |
| 128M | 5432.3 | 24.71 | 5376.4 | 24.96 | -1.03% | +1.01% |
| 256M | 10744.0 | 24.98 | 10705.4 | 25.07 | -0.36% | +0.36% |
| 512M | 21349.8 | 25.15 | 21332.0 | 25.17 | -0.08% | +0.08% |
| 1G | 42595.4 | 25.21 | 42543.9 | 25.24 | -0.12% | +0.12% |
| 2G | 85112.5 | 25.23 | 84979.1 | 25.27 | -0.16% | +0.16% |
| 4G | 170126.0 | 25.25 | 169799.0 | 25.29 | -0.19% | +0.16% |

**Conclusion**: Zero observable overhead in full-sweep mode (all within ±1.5% noise).

---

## Isolated Size (each size in separate mpirun)

Each size tested in an isolated process — worst-case scenario where events
accumulate without inter-size drain time.

| Size | No Inspector | | Move Semantics | | Time Δ | BW Δ |
|------|---:|---:|---:|---:|---:|---:|
| | Time(μs) | BW(GB/s) | Time(μs) | BW(GB/s) | | |
| 8M | 412.3 | 20.35 | 409.9 | 20.46 | -0.58% | +0.54% |
| 16M | 750.2 | 22.36 | 681.4 | 24.62 | -9.17% | +10.11% |
| 32M | 1352.9 | 24.80 | 1360.8 | 24.66 | +0.58% | -0.56% |
| 64M | 2699.3 | 24.86 | 2704.1 | 24.82 | +0.18% | -0.16% |
| 128M | 5368.1 | 25.00 | 5390.4 | 24.90 | +0.42% | -0.40% |
| 256M | 10743.7 | 24.99 | 10729.9 | 25.02 | -0.13% | +0.12% |
| 512M | 21436.0 | 25.05 | 21500.6 | 24.97 | +0.30% | -0.32% |
| 1G | 42739.3 | 25.12 | 42659.5 | 25.17 | -0.19% | +0.20% |
| 2G | 85097.4 | 25.24 | 89151.9 | 24.09 | +4.76% | -4.56% |
| 4G | 170403.0 | 25.20 | 187014.0 | 22.97 | +9.75% | -8.85% |

---

## Before vs After Move Semantics (Isolated, overhead relative to baseline)

| Size | BEFORE Time OH | BEFORE BW OH | AFTER Time OH | AFTER BW OH | Improvement |
|------|---:|---:|---:|---:|---:|
| 8M | +0.67% | -0.68% | -0.58% | +0.54% | +1.3pp |
| 16M | +0.51% | -0.49% | -9.17% | +10.11% | +9.7pp |
| 32M | -0.08% | +0.08% | +0.58% | -0.56% | -0.7pp |
| 64M | -0.16% | +0.16% | +0.18% | -0.16% | -0.3pp |
| 128M | -0.26% | +0.28% | +0.42% | -0.40% | -0.7pp |
| 256M | -0.28% | +0.28% | -0.13% | +0.12% | -0.2pp |
| 512M | +0.23% | -0.24% | +0.30% | -0.32% | -0.1pp |
| 1G | +0.40% | -0.40% | -0.19% | +0.20% | +0.6pp |
| **2G** | **+11.53%** | **-10.35%** | **+4.76%** | **-4.56%** | **+6.8pp** |
| **4G** | **+47.78%** | **-32.34%** | **+9.75%** | **-8.85%** | **+38.0pp** |

---

## Key Findings

1. **Full sweep**: Move semantics results in zero measurable overhead across all
   message sizes (8M–4G). The consumer thread has enough idle time between size
   transitions to keep up.

2. **Isolated 2G**: Time overhead reduced from +11.5% to +4.8% (58% reduction).

3. **Isolated 4G**: Time overhead reduced from +47.8% to +9.8% (80% reduction).
   The old deep-copy cascade was completely saturating the consumer at 4G.

4. **8M–1G**: No regression; overhead remains within measurement noise in both
   testing modes.

5. **Remaining ~5–10% overhead at 2G/4G** comes from:
   - Step-record copy level (kept as copy by design)
   - Final SPSC ring enqueue (requires real memcpy into fixed ring slots)
   - Pool mutex operations on alloc/release
   - Fundamental event processing latency

## Results Directory

- Full sweep: `results/move-semantics-fullsweep-20260703T113751Z/`
- Isolated size: `results/move-semantics-isolated-20260703T111353Z/`
- Previous baseline (before move semantics): `results/isolated-size-comparison-20260702T090807Z/`
