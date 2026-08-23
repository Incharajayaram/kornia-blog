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
