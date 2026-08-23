---
toc: true
layout: post
description: A CUDA backend for kornia-rs. Runtime kernel compilation, device-aware memory, and the three optimizations I was sure about that turned out to be wrong.
categories: [gsoc, announcement]
image: images/gsoc2026-gpu/benchmarks.png
author: [Inchara J]
title: "GSoC 2026: GPU acceleration for kornia-rs"
---

I'm Inchara J, and this summer I worked on kornia-rs as a Google Summer of Code
contributor. I built a CUDA backend for
[kornia-rs](https://github.com/kornia/kornia-rs): GPU kernels for resize, warp,
remap and colour conversion, plus a device-aware tensor memory model to hang
them on.

Rust has no native GPU vision library. A Rust project needing accelerated
`resize` or `warp_affine` today either calls out to C++ or pays Python interop
overhead. The interesting problem wasn't algorithmic, CUDA bilinear
interpolation is well understood but structural: how do you make CUDA kernels
first-class in a Rust library, with `unsafe` confined to a thin layer, device
memory obeying ownership rules, and no seconds-long compile stall on the first
image?

## Picking a backend, then changing my mind

I initially worked with [CubeCL](https://github.com/tracel-ai/cubecl), a Rust GPU
kernel DSL. Writing kernels in Rust instead of CUDA C is genuinely appealing, and
I shipped three PRs on it, allocator scaffold, backend impl, nearest and
bilinear resize.

The kernels worked and were slow. Downscale ran at **8-12 GB/s** on a card with
192 GB/s of bandwidth.

The fix needed `__ldg`, the read-only cache load intrinsic 
and control over
the L1 cache configuration. CubeCL exposes neither, because it targets CUDA,
WGPU and others through a common abstraction, and those are CUDA-specific. I
rewrote the kernels as native CUDA C compiled at runtime through
[NVRTC](https://docs.nvidia.com/cuda/nvrtc/), with
[cudarc](https://github.com/coreylowman/cudarc) for safe driver bindings.

Downscale went to **~70 GB/s**. Roughly six times faster, from getting access to
two intrinsics.

There was a second argument for the switch that I hadn't weighed. A reviewer
pointed out that enabling the CubeCL feature pulled its whole compiler stack which is an MLIR pipeline and an LLVM downloader into every downstream build. It had
also dragged a `js-sys` dependency into the workspace, which is a good sign
you've taken a wrong turn in a computer vision crate. kornia-rs standardised on
cudarc shortly after.

## How the backend works

![]({{ site.baseurl }}/images/gsoc2026-gpu/architecture.png "Backend architecture")

Each kernel is a Rust `&'static str` of CUDA C, compiled to PTX by NVRTC on first
call and cached in a per-process `OnceLock<CudaKernel>`. Later calls get the
cached kernel with no per-call or per-size recompilation. NVRTC costs 300-800 ms
for the largest kernels, invisible amortised across a process lifetime, painful
if paid per image.

Two details worth calling out. Compute capability is detected at runtime and
memoised, so the same build runs on any NVIDIA GPU without recompiling the crate.
And after compilation each kernel sets `CU_FUNC_CACHE_PREFER_L1`, which grows
Turing's L1 from 32 KB to 64 KB and lifts hit rates for the scattered reads that
bicubic (4x4 taps) and Lanczos (6x6) do, and it costs nothing in the source.

![]({{ site.baseurl }}/images/gsoc2026-gpu/memory-model.png "Tensor memory model")

On the memory side, a `Tensor<T, N>` owns a `Box<dyn MemoryResource>` carrying a
`MemoryDomain`: `Host`, `Device { id }`, or `Unified { id }`. Host slice access
asserts the domain is host-accessible, so handing a device tensor to a CPU API
fails loudly instead of dereferencing device memory from the host.

That domain also drives dispatch. The public `resize`, `warp_affine`,
`warp_perspective` and colour functions are unchanged from a caller's view: host
operands take the CPU path, device operands launch a kernel, and a mixed pair is
a typed error. **Nothing transfers implicitly**, if data crosses PCIe, you wrote
the line that moved it.

## Texture objects, tried twice, removed twice

Warp-affine with `BORDER_CONSTANT` normally needs a bounds check in every thread:
if the inverse-mapped coordinate falls outside the source, write zero. On a 45°
rotation about half the output pixels are black corners, so that branch diverges
badly.

Binding the source as a CUDA texture with `CU_TR_ADDRESS_MODE_BORDER` deletes the
branch, out-of-bounds fetches return the border colour *in hardware*. It worked,
it was faster, and it shipped.

So I tried the same thing for resize. It made it **~10% slower**.

Resize has none of the properties that make textures pay. `tex2D` costs roughly
100 cycles of instruction latency on Turing against about 30 for `__ldg`, and it
earns that back through 2D spatial caching and hardware boundary handling. But
sequential downscale reads consecutive source columns, which the unified L1
already coalesces just as well, and downscale has no out-of-bounds divergence to
remove. You pay the latency and get nothing back.

Then the warp-affine texture path had to go too, not for performance, for
correctness. A 1-channel pitch-2D texture makes `pitchInBytes = src_w × 3 × 4`,
and CUDA requires that to be a multiple of `CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT`
(32 bytes). The rows are densely packed, so **the kernels failed outright for
every source width not a multiple of 8**. Satisfying the alignment would have
meant a padded device-to-device copy on every call, which costs more than the
divergence it saves.

So textures are gone from the whole backend and everything reads through `__ldg`.
An optimization that is correct only on convenient input sizes is not an
optimization.

## One design question worth the detour

Warp-perspective and remap overlap heavily: remap samples a source image at
caller-supplied `(map_x, map_y)` coordinates, and a perspective warp is just a
particular coordinate map. So should perspective be a thin wrapper that
precomputes a map and calls remap, or its own fused kernel?

The generic version is better for maintenance, so I benchmarked before
committing: remap-bilinear landed within 6% of the fused kernel, but
remap-nearest was **30-51% slower**. Reading a precomputed coordinate map costs
two extra float loads per pixel, nothing beside bilinear's four samples, a lot
beside nearest-neighbour's single fetch.

So both exist: remap as the general primitive that lens undistortion builds on,
and fused warps where the arithmetic is cheap enough that map traffic dominates.

## Results

![]({{ site.baseurl }}/images/gsoc2026-gpu/benchmarks.png "Kernel times against OpenCV CUDA and PyTorch, and round-trip economics on two cards")

GTX 1650, against OpenCV 4.12 CUDA and PyTorch 2.9+cu128, all three re-measured
on the same card with the same loop: 50 warmup, 200 timed iterations, one sync
after the batch.

Resize 1920×1080 → 960×540, and warp-affine at 45°, kernel time:

| Operation | kornia-rs | cv2 CUDA | PyTorch | vs cv2 | vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| resize nearest | 0.107 ms | 0.217 ms | 0.109 ms | **2.0×** | 1.0× |
| resize bilinear | 0.178 ms | 0.291 ms | 0.182 ms | **1.6×** | 1.0× |
| resize bicubic | 0.245 ms | 0.569 ms | 1.493 ms | **2.3×** | **6.1×** |
| warp-affine | 0.592 ms | 0.763 ms | 3.177 ms | 1.3× | **5.4×** |

kornia-rs beats OpenCV CUDA on every operation and is level with PyTorch on the
cheap interpolants, pulling well ahead on the expensive ones. The PyTorch gap is
widest on warp because `F.affine_grid` + `F.grid_sample` allocates an
intermediate coordinate-grid tensor on every call; kornia-rs allocates nothing
intermediate.

Colour conversion reaches **170 GB/s at 1080p, 89% of the card's theoretical
peak** which is the number I'm happiest with, because it's a roofline claim rather than
a relative one. There is not much left on the table when you are that close to
what the memory system can deliver.

### What byte-exactness cost

There is one number above I would have reported differently a few months ago.
When I analyzed the benchmarks mid-project, bilinear resize measured 0.101 ms. Today the
same kernel on the same card measures 0.178 ms which is about 70% slower.

Nothing regressed by accident. `CudaKernel::compile` now passes **`--fmad=false`**
to NVRTC, which disables implicit fused multiply-add contraction so that a plain
`a*b + c` in kernel source rounds twice, exactly as the same expression rounds on
the CPU. That is what makes the GPU output *bit-identical* to the CPU path rather
than merely close, and bit-identical is the contract the parity tests check.

FMA contraction is not a small optimization to give up, and the cost landed
unevenly. Bicubic didn't regress at all (0.245 ms then and now) because its Horner
weights were rewritten to call `fmaf()` explicitly, so kernels that want fusion keep
it by saying so. Bilinear's tap accumulation is plain `a*b + c`, so it pays.

Was it worth 70% of one kernel? I think yes. A GPU path that is *almost* the same
as the CPU path is a source of bug reports nobody can reproduce, and "almost"
compounds through a pipeline. But it is a real trade, it is not free, and a
faster-looking number in an older document does not mean the code got worse, it
means it got stricter.

### The result that reframed the project

Those are all *kernel* times, which is the fair comparison against cv2 CUDA. But
it isn't what a caller experiences if the data starts on the host. So I built a
benchmark separating host-to-device transfer, kernel, and device-to-host.

On my GTX 1650, the answer was blunt:

| Operation | Resolution | CPU | H2D | Kernel | D2H | Kernel | Round-trip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resize (f32) bilinear | 1080p to 540p | 5.28 ms | 9.16 ms | 0.18 ms | 2.21 ms | **28.7x** | **0.46x** |
| gray_from_rgb (u8) | 4K | 6.18 ms | 9.61 ms | 0.30 ms | 8.78 ms | **20.8x** | **0.3x** |

The kernel is 20-30x faster and the round trip is *slower than staying on the
CPU*. The transfers cost eleven times what the kernel costs. That is the whole
argument for refusing implicit transfers: a backend that quietly uploaded and
downloaded around every call would be slower than the CPU path while looking
like an optimization, and nobody would be able to see it happening.

I was ready to state that as a general property of bandwidth-bound GPU work.
Then a friend ran the same benchmark, same commit, on an RTX 3090:

| resize f32 bilinear, 1080p | CPU | H2D | Kernel | D2H | Round-trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| GTX 1650 | 5.28 ms | 9.16 ms | 0.18 ms | 2.21 ms | **0.46x** |
| RTX 3090 | 6.95 ms | 2.71 ms | 0.04 ms | 1.11 ms | **1.8x** |

The kernel got 4.5x faster, which I expected. The transfers also got 3.4x faster
on the upload and 2x on the download, which I had not thought about at all. On
the 3090 the round trip **wins on 37 of the 58 benchmarked operations**, by up to
38x. Bicubic resize goes from 2.1x to 8.4x.

So "PCIe dominates" was never a fact about GPUs. It was a fact about my GPU, on
a narrower link, and I had been about to generalise from a sample of one. What
actually holds is the weaker and more useful claim: whether a transfer pays for
itself depends on the ratio between your link and your CPU, it varies by more
than an order of magnitude across cards, and the only way to know is to measure
the machine you are shipping on. Which is why the benchmark reports the split
instead of a single number.

One honest caveat on that 3090 run. Its host is a virtualised Haswell vCPU, so
the CPU baseline is weak and every CPU-relative ratio on that machine is
optimistic by some amount I can't quantify without a second run on real
hardware. The kernel and transfer columns are unaffected. The direction of the
result survives the caveat comfortably, but the exact multipliers should be read
as a range, not a measurement.

Either way, the device path is at its best when a tensor becomes device-resident
and *stays* there across a chain of operations, because then the transfer is
paid once instead of per call. That is what the domain dispatch exists to allow.

## Three things I got wrong

Halfway through I wrote a paper on this work whose "future work" section made
predictions I then got to test. I don't fully recommend the experience.

**Shared-memory tiling. Predicted ~20% faster; measured 1.5× slower at 2×
downscale and 7× slower at 4×.** The textbook optimization: neighbouring threads
read overlapping source pixels, so stage a tile in shared memory and compute the
block from there. But bilinear samples exactly **four source pixels per output
pixel regardless of scale factor**, at 4× downscale, adjacent outputs read
regions four pixels apart, so there's almost no overlap to exploit, while the
tile load fetches everything in the region including the three quarters nobody
wanted. I'd replaced a sparse, well-cached access pattern with a dense one. The
clue was in my own benchmark output weeks earlier: downscale was already running
at 84-91% of achievable DRAM bandwidth, so there was no headroom for a cache
trick to recover.

**`INTER_AREA`. Predicted 15-30% faster; no improvement.** The follow-up that was
meant to justify tiling, since area-averaging for integer ratios should be cheaper
than bilinear. It wasn't, and the branch is still sitting unopened.

**Unified memory. A real win, on half the hardware.** `cudaMallocManaged` gives
one pointer valid on host and device, which on a Jetson, where CPU and GPU share
physical DRAM, should eliminate transfers entirely. It does, and it loses badly
on a discrete card.

| Size | Jetson explicit | Jetson unified | | GTX 1650 explicit | GTX 1650 unified | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGA 640×480 | 2.509 ms | 1.813 ms | **1.38×** | 2.329 ms | 4.053 ms | 0.57× |
| FHD 1920×1080 | 12.639 ms | 7.410 ms | **1.71×** | 17.199 ms | 26.041 ms | 0.66× |
| 4K 3840×2160 | 47.318 ms | 27.517 ms | **1.72×** | 67.520 ms | 102.180 ms | 0.66× |

Output is bit-identical on both. On the integrated part it removes a real copy;
on the discrete one the driver demand-pages over PCIe and pays more than the
copies it removed. There's a second trap in allocation: `cuMemAllocManaged` is
about **3000× more expensive** than a pooled device allocation (42 ms vs 0.013 ms
for a 4K buffer), so unified buffers must be allocated once and reused, an
allocate-per-frame loop erases the whole benefit.

It shipped with that framing, and the benchmark reports `cudaDevAttrIntegrated`
so any run says which case it measured.

## The bug that wasn't

The same pursuit of exactness produced my favourite non-bug. Bilinear
warp-affine was diverging from OpenCV by up to 0.82 at non-identity rotations,
appearing right after a byte-exactness rewrite landed. It looked serious.

![]({{ site.baseurl }}/images/gsoc2026-gpu/parity.png "CPU/GPU parity and the OpenCV border seam")

It was neither a regression nor new. The middle panel above is what
`--fmad=false` bought: the same call on host and device operands, and not one of
262,144 pixels differs. The right panel is where the 0.82 lives. The CUDA kernel
clamps the `+1` tap at source edges, `BORDER_REPLICATE`, because that is what the
CPU `warp_affine` loop does, and matching the CPU was the contract. OpenCV uses
`BORDER_CONSTANT`.
The two disagree at exactly one pixel of border, and only where the
inverse-mapped coordinate lands on an edge. At 45° that edge is the diamond
outline you can see, and it is why identity transforms matched to 8e-9 while
rotations didn't. The rewrite hadn't introduced the
divergence; it had tightened the CPU/GPU match enough to expose a pre-existing
CPU-vs-OpenCV difference. The fix was to the *test*, not the kernel.

The parity suite existed at all because my mentor asked for it early: *"make sure
all the algorithms match output with the reference libraries. I found in some
cases that the algorithms get faster but the results are not exactly the same."*
Best process advice I got all summer.

## What shipped

- **Resize**: nearest, bilinear, bicubic, Lanczos-3, plus byte-exact u8 paths
- **Warp-affine and warp-perspective**: same four interpolants
- **Remap**: the generic primitive behind lens undistortion, f32 and u8
- **Colour conversion**: RGB↔gray, HSV, HLS, YCbCr, BGR, Bayer demosaic
- **Device-aware tensor storage**: `MemoryDomain`, pinned and unified allocators, DLPack-compatible foreign import
- **Correctness tooling**: pixel-level parity against OpenCV CUDA and NVIDIA VPI
- **A reproducible benchmark suite** with the H2D/kernel/D2H split, 1080p and 4K, desktop and Jetson

Every pull request, in order:

| PR | What it does |
| --- | --- |
| [#925](https://github.com/kornia/kornia-rs/pull/925) | GPU backend scaffold: feature flags, the `Backend` trait, a first allocator |
| [#927](https://github.com/kornia/kornia-rs/pull/927) | CubeCL backend implementation plus a GPU allocator smoke test |
| [#938](https://github.com/kornia/kornia-rs/pull/938) | Domain-aware tensor storage, and a home for experimental GPU imgproc |
| [#943](https://github.com/kornia/kornia-rs/pull/943) | `gray_from_rgb_f32` with an AVX2+FMA path, on the CPU side |
| [#946](https://github.com/kornia/kornia-rs/pull/946) | First GPU resize kernels, nearest and bilinear, on CubeCL |
| [#958](https://github.com/kornia/kornia-rs/pull/958) | Warp-affine via NVRTC. The switch to native CUDA starts here |
| [#967](https://github.com/kornia/kornia-rs/pull/967) | Benchmark results for the NVRTC resize and warp-affine kernels |
| [#971](https://github.com/kornia/kornia-rs/pull/971) | CPU warp-affine: incremental coordinates, valid-range skip, 16-row Rayon chunks |
| [#974](https://github.com/kornia/kornia-rs/pull/974) | Bicubic kernels for resize and warp-affine, Keys cubic with `a = -0.5` |
| [#979](https://github.com/kornia/kornia-rs/pull/979) | Texture-object warp-affine kernels, later removed (see above) |
| [#992](https://github.com/kornia/kornia-rs/pull/992) | `block_dim` override on the bicubic launchers, which had hardcoded it |
| [#993](https://github.com/kornia/kornia-rs/pull/993) | Auto-clamp block dims for small images so thumbnails don't collapse occupancy |
| [#994](https://github.com/kornia/kornia-rs/pull/994) | Lanczos-3, separable two-pass for resize and full 6x6 for warp |
| [#998](https://github.com/kornia/kornia-rs/pull/998) | The remap primitive, with the benchmark behind the fused-vs-generic decision |
| [#999](https://github.com/kornia/kornia-rs/pull/999) | Warp-perspective, bilinear, nearest and bicubic |
| [#1032](https://github.com/kornia/kornia-rs/pull/1032) | Pixel-level correctness checks against OpenCV for resize and warp-affine |
| [#1034](https://github.com/kornia/kornia-rs/pull/1034) | `cudaMallocManaged` support: the unified allocator and its Jetson numbers |
| [#1035](https://github.com/kornia/kornia-rs/pull/1035) | The benchmark suite with the H2D / kernel / D2H breakdown |
| [#1066](https://github.com/kornia/kornia-rs/pull/1066) | Post-merge review fixes on remap |
| [#1068](https://github.com/kornia/kornia-rs/pull/1068) | u8 remap with fixed-point quantisation, byte-exact on CPU and CUDA |

Four are still open at the time of writing:

| PR | What it does |
| --- | --- |
| [#1115](https://github.com/kornia/kornia-rs/pull/1115) | Benchmarks for remap, colour conversion and the unified-memory path |
| [#1116](https://github.com/kornia/kornia-rs/pull/1116) | Laplacian and integral filters with GPU support |
| [#1121](https://github.com/kornia/kornia-rs/pull/1121) | Two optimizations to the existing CUDA SIFT kernels: register-tiling the vertical blur, and warp-aggregating the descriptor histogram atomics so lanes hitting the same bin combine before the atomic |
| [#1122](https://github.com/kornia/kornia-rs/pull/1122) | Element-wise and reduction tensor ops dispatched on `MemoryDomain`, with in-place variants |

Six were closed rather than merged. Most were superseded by a later PR, like
#945 by #946 and #954 by #958, or were small CI fixes that got folded elsewhere.
Two of them show up in this post.
[#1017](https://github.com/kornia/kornia-rs/pull/1017) was the shared-memory
tiling attempt, closed once it benchmarked slower.
[#1036](https://github.com/kornia/kornia-rs/pull/1036) was the first version of
the tensor ops, which a stale bot closed before anyone reviewed it. It came back
as #1122 at a third of the size, once the parts that had been superseded upstream
were stripped out.


## Appendix: the full sweep

Everything above is a handful of rows chosen to make a point. This is the whole
`bench_cuda_imgproc` sweep on the RTX 3090, all 58 operations, so the selection
above can be checked against the rest.

Reproduce it with:

```sh
git clone -b bench/all-gpu-work https://github.com/Incharajayaram/kornia-rs
cd kornia-rs && cargo bench --bench bench_cuda_imgproc --features cuda
```

RTX 3090 (sm_86, 24 GB, driver 580.173.02, CUDA 12.8), commit `b887ffd`,
30 warmup and 100 timed iterations, CUDA events, rotating source buffers.
All times in milliseconds. The host is a virtualised Haswell vCPU, so treat the
CPU column and the two speedup columns as optimistic; the H2D, kernel and D2H
columns are hardware measurements and stand on their own.

| Operation | Interp | Resolution | CPU | H2D | Kernel | D2H | Total GPU | Kernel | Round trip |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize (f32) | bilinear | 1920×1080→960×540 | 6.95 | 2.71 | 0.04 | 1.11 | 3.86 | 157.5x | 1.8x |
| resize (f32) | bilinear | 3840×2160→1920×1080 | 20.66 | 11.36 | 0.15 | 2.20 | 13.71 | 137.0x | 1.5x |
| resize (f32) | nearest | 1920×1080→960×540 | 2.85 | 2.67 | 0.03 | 0.71 | 3.41 | 97.9x | 0.8x |
| resize (f32) | nearest | 3840×2160→1920×1080 | 8.54 | 11.29 | 0.10 | 2.01 | 13.39 | 89.7x | 0.6x |
| resize (f32) | bicubic | 1920×1080→960×540 | 29.24 | 2.83 | 0.05 | 0.62 | 3.50 | 633.2x | 8.4x |
| resize (f32) | bicubic | 3840×2160→1920×1080 | 87.69 | 10.61 | 0.16 | 2.01 | 12.77 | 559.4x | 6.9x |
| resize (f32) | lanczos | 1920×1080→960×540 | 5.25 | 3.09 | 0.30 | 0.55 | 3.94 | 17.7x | 1.3x |
| resize (f32) | lanczos | 3840×2160→1920×1080 | 18.67 | 11.50 | 0.87 | 2.16 | 14.53 | 21.5x | 1.3x |
| resize (u8) | bilinear | 1920×1080→960×540 | 5.27 | 0.57 | 0.02 | 0.20 | 0.79 | 295.7x | 6.7x |
| resize (u8) | bilinear | 3840×2160→1920×1080 | 18.43 | 2.52 | 0.05 | 0.67 | 3.24 | 398.9x | 5.7x |
| resize (u8) | nearest | 1920×1080→960×540 | 2.07 | 0.56 | 0.01 | 0.19 | 0.75 | 169.4x | 2.7x |
| resize (u8) | nearest | 3840×2160→1920×1080 | 6.03 | 2.67 | 0.03 | 0.67 | 3.37 | 200.2x | 1.8x |
| warp_affine (30° rot, f32) | bilinear | 1920×1080 | 4.45 | 2.52 | 0.06 | 2.11 | 4.69 | 69.2x | 0.9x |
| warp_affine (30° rot, f32) | bilinear | 3840×2160 | 19.61 | 10.94 | 0.25 | 12.75 | 23.94 | 78.2x | 0.8x |
| warp_affine (30° rot, u8) | bilinear | 1920×1080 | 2.28 | 0.57 | 0.05 | 0.71 | 1.33 | 41.6x | 1.7x |
| warp_affine (30° rot, u8) | bilinear | 3840×2160 | 9.56 | 2.78 | 0.19 | 2.21 | 5.17 | 51.4x | 1.8x |
| warp_perspective (30° rot, f32) | bilinear | 1920×1080 | 13.78 | 2.59 | 0.07 | 2.12 | 4.78 | 211.8x | 2.9x |
| warp_perspective (30° rot, f32) | bilinear | 3840×2160 | 50.39 | 10.86 | 0.26 | 12.56 | 23.68 | 193.5x | 2.1x |
| warp_perspective (30° rot, u8) | bilinear | 1920×1080 | 1.82 | 0.56 | 0.07 | 0.58 | 1.21 | 26.4x | 1.5x |
| warp_perspective (30° rot, u8) | bilinear | 3840×2160 | 8.03 | 2.44 | 0.25 | 2.09 | 4.77 | 32.7x | 1.7x |
| remap (f32) | bilinear | 1920×1080 | 12.88 | 2.44 | 0.09 | 2.10 | 4.63 | 148.8x | 2.8x |
| remap (f32) | bilinear | 3840×2160 | 50.15 | 10.69 | 0.32 | 12.70 | 23.71 | 157.2x | 2.1x |
| gaussian_blur (5x5, f32) |  | 1920×1080 | 42.09 | 2.47 | 0.13 | 1.98 | 4.58 | 329.4x | 9.2x |
| gaussian_blur (3x3, u8) |  | 1920×1080 | 0.44 | 0.57 | 0.04 | 0.56 | 1.17 | 10.2x | 0.4x |
| box_blur (3x3, u8) |  | 1920×1080 | 6.28 | 0.57 | 0.04 | 0.56 | 1.17 | 140.9x | 5.4x |
| sobel (3x3, f32) |  | 1920×1080 | 106.46 | 2.46 | 0.34 | 1.98 | 4.78 | 313.5x | 22.3x |
| laplacian (3x3, u8) |  | 1920×1080 | 1.59 | 0.25 | 0.02 | 0.42 | 0.68 | 95.9x | 2.3x |
| integral (u8) |  | 1920×1080 | 3.08 | 0.34 | 1.19 | 1.30 | 2.83 | 2.6x | 1.1x |
| gaussian_blur (5x5, f32) |  | 3840×2160 | 230.99 | 10.52 | 0.48 | 12.53 | 23.53 | 483.6x | 9.8x |
| gaussian_blur (3x3, u8) |  | 3840×2160 | 1.64 | 2.69 | 0.14 | 2.02 | 4.85 | 11.7x | 0.3x |
| box_blur (3x3, u8) |  | 3840×2160 | 23.31 | 2.63 | 0.14 | 1.99 | 4.76 | 161.5x | 4.9x |
| sobel (3x3, f32) |  | 3840×2160 | 413.50 | 10.63 | 1.30 | 12.03 | 23.96 | 317.1x | 17.3x |
| laplacian (3x3, u8) |  | 3840×2160 | 6.59 | 0.78 | 0.06 | 1.39 | 2.22 | 116.0x | 3.0x |
| integral (u8) |  | 3840×2160 | 9.88 | 0.96 | 2.41 | 4.10 | 7.46 | 4.1x | 1.3x |
| erode (3x3, u8) |  | 1920×1080 | 45.38 | 0.55 | 0.05 | 0.58 | 1.18 | 968.8x | 38.6x |
| dilate (3x3, u8) |  | 1920×1080 | 32.89 | 0.56 | 0.04 | 0.58 | 1.18 | 740.1x | 27.8x |
| erode (3x3, u8) |  | 3840×2160 | 138.34 | 2.58 | 0.13 | 2.01 | 4.72 | 1095.8x | 29.3x |
| dilate (3x3, u8) |  | 3840×2160 | 127.20 | 2.53 | 0.13 | 2.00 | 4.66 | 988.4x | 27.3x |
| gray_from_rgb (f32) |  | 1920×1080 | 0.81 | 2.42 | 0.05 | 0.71 | 3.18 | 17.8x | 0.3x |
| gray_from_rgb (f32) |  | 3840×2160 | 3.95 | 10.65 | 0.16 | 2.76 | 13.57 | 24.9x | 0.3x |
| remap (u8) | bilinear | 1920×1080 | 2.93 | 0.56 | 0.04 | 0.63 | 1.23 | 69.5x | 2.4x |
| remap (u8) | nearest | 1920×1080 | 2.93 | 0.57 | 0.04 | 0.64 | 1.25 | 70.6x | 2.3x |
| remap (u8) | bilinear | 3840×2160 | 10.97 | 2.45 | 0.14 | 1.98 | 4.58 | 75.7x | 2.4x |
| remap (u8) | nearest | 3840×2160 | 10.97 | 2.45 | 0.14 | 1.98 | 4.57 | 76.7x | 2.4x |
| gray_from_rgb (u8) |  | 1920×1080 | 0.50 | 0.55 | 0.02 | 0.25 | 0.83 | 27.9x | 0.6x |
| gray_from_rgb (u8) |  | 3840×2160 | 0.88 | 2.45 | 0.05 | 0.71 | 3.21 | 19.2x | 0.3x |
| rgb_from_gray (u8) |  | 1920×1080 | 0.56 | 0.23 | 0.02 | 0.63 | 0.88 | 35.8x | 0.6x |
| rgb_from_gray (u8) |  | 3840×2160 | 1.19 | 0.75 | 0.05 | 2.18 | 2.98 | 22.0x | 0.4x |
| hsv_from_rgb (f32) |  | 1920×1080 | 2.29 | 2.62 | 0.07 | 2.02 | 4.70 | 34.4x | 0.5x |
| hsv_from_rgb (f32) |  | 3840×2160 | 8.50 | 11.13 | 0.24 | 15.85 | 27.22 | 35.0x | 0.3x |
| hls_from_rgb (f32) |  | 1920×1080 | 2.87 | 2.63 | 0.07 | 2.20 | 4.90 | 43.3x | 0.6x |
| hls_from_rgb (f32) |  | 3840×2160 | 9.37 | 11.27 | 0.24 | 14.77 | 26.29 | 38.4x | 0.4x |
| ycc_from_rgb (u8) |  | 1920×1080 | 1.30 | 0.57 | 0.02 | 0.62 | 1.21 | 58.7x | 1.1x |
| ycc_from_rgb (u8) |  | 3840×2160 | 4.28 | 2.63 | 0.07 | 2.20 | 4.89 | 64.4x | 0.9x |
| ycc_from_rgb (f32) |  | 1920×1080 | 1.83 | 2.52 | 0.07 | 2.13 | 4.72 | 27.2x | 0.4x |
| ycc_from_rgb (f32) |  | 3840×2160 | 8.01 | 10.73 | 0.24 | 12.66 | 23.63 | 32.8x | 0.3x |
| bgr_from_rgb (u8) |  | 1920×1080 | 0.75 | 0.57 | 0.02 | 0.65 | 1.24 | 33.0x | 0.6x |
| bgr_from_rgb (u8) |  | 3840×2160 | 2.08 | 2.64 | 0.07 | 2.21 | 4.91 | 31.0x | 0.4x |

### The other three benchmarks on the same machine

**Unified memory, end-to-end `resize()` per frame.** Explicit copies against
managed memory against write-combined pinned memory:

| Size | Explicit | Unified | Pinned WC |
| --- | ---: | ---: | ---: |
| VGA 640x480 | 2.496 ms | 5.352 ms | 1.897 ms |
| HD 1280x720 | 5.443 ms | 8.879 ms | 3.263 ms |
| FHD 1920x1080 | 9.828 ms | 15.940 ms | 5.612 ms |
| 4K 3840x2160 | 26.525 ms | 37.449 ms | 18.312 ms |

Unified memory loses on this discrete card exactly as it does on the GTX 1650,
which is the result the Jetson comparison predicts. The column worth noticing is
the third one: **write-combined pinned memory beats both**, by 1.3x to 1.8x,
because the CPU writes stream out without polluting cache and the transfer is
then a straight DMA. It is the option I would reach for first on a discrete card,
and it is not the one I spent the most time on.

**SIFT `detect_and_compute`,** unified against explicit: 0.35x at VGA, 0.42x at
HD, 0.45x at FHD. Same direction, larger penalty, because SIFT allocates more
managed buffers.

**Tensor ops.** The elementwise round trips lose as expected (0.1x to 0.3x), but
`reduce` wins: 3.7x at 1M elements, 3.6x at 10M, 3.4x at 100M. A reduction
uploads a buffer and returns a scalar, so it pays H2D once and no D2H at all,
which is the one shape where the transfer arithmetic works out on any card.

The summation-accuracy probe returned bit-identical numbers on the 3090 and the
GTX 1650: CPU 48614172786688, GPU 49982858067968, against a true value of
50000000004999.8. Two different architectures, same two answers, which is what
you would hope for from a deterministic reduction and a deterministic bug.

## Thanks

To Edgar Riba and Christie Purackal for review consistently more careful than the code deserved, for
the Jetson access that made half these numbers possible, and to the friend who
ran the whole suite on an RTX 3090 and accidentally overturned one of my
conclusions. And to the kornia community. Everything
is in [kornia/kornia-rs](https://github.com/kornia/kornia-rs) behind the `cuda`
feature flag.

---

*Inchara J. GSoC 2026 contributor, kornia.
[GitHub](https://github.com/Incharajayaram) ·
[LinkedIn](https://linkedin.com/in/inchara-j-752050251)*
