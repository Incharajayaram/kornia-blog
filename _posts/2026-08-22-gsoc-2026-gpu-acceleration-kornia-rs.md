---
toc: true
layout: post
description: A CUDA backend for kornia-rs. Runtime kernel compilation, device-aware memory, and the three optimizations I was sure about that turned out to be wrong.
categories: [gsoc, announcement]
image: images/gsoc2026-gpu/benchmarks.png
title: "GSoC 2026: GPU acceleration for kornia-rs"
---

This summer I built a CUDA backend for [kornia-rs](https://github.com/kornia/kornia-rs),
the Rust computer vision library. GPU kernels for resize, warp, remap and colour
conversion, plus a device-aware tensor memory model to hang them on.

Rust doesn't really have a native GPU vision library. If you need an accelerated
`resize` or `warp_affine` from Rust today, you either call out to C++ or eat the
Python interop overhead. The hard part of fixing that isn't the algorithms. CUDA
bilinear interpolation is well understood. It's structural: how do you make CUDA
kernels a first-class thing in a Rust library, keep `unsafe` in a thin layer,
make device memory obey ownership rules, and not stall the first image call for
several seconds while something compiles?

## Picking a backend, then changing my mind

My proposal named [CubeCL](https://github.com/tracel-ai/cubecl), a Rust GPU
kernel DSL. Writing kernels in Rust instead of CUDA C is a nice idea and I still
think so. I shipped three PRs on it: the allocator scaffold, a backend impl, and
nearest plus bilinear resize.

The kernels worked. They were also slow. Downscale ran at 8-12 GB/s on a card
with 192 GB/s of bandwidth, which is about as far from the hardware as you can
get while still technically using it.

What I needed was `__ldg`, the read-only cache load intrinsic, and control over
the L1 cache configuration. CubeCL doesn't expose either one. It can't, really.
It targets CUDA and WGPU and others through a common abstraction, and both of
those knobs are CUDA-specific. So I rewrote the kernels as plain CUDA C compiled
at runtime through [NVRTC](https://docs.nvidia.com/cuda/nvrtc/), with
[cudarc](https://github.com/coreylowman/cudarc) for the driver bindings.

Downscale went to ~70 GB/s. Six times faster, and all of it came from being
allowed to say two things the abstraction wouldn't let me say.

There was a second argument I hadn't even weighed. A reviewer pointed out that
turning on the CubeCL feature dragged its whole compiler stack, an MLIR pipeline
and an LLVM downloader, into every downstream build. It had also pulled a
`js-sys` dependency into the workspace, which is usually a sign you've taken a
wrong turn in a computer vision crate. kornia-rs consolidated on cudarc not long
after.

Throwing away three PRs of work was annoying. But I don't think I could have
argued the point without building on CubeCL first, because "a portable
abstraction can't expose the vendor intrinsics your performance depends on" is
the kind of claim that sounds like an excuse until you have the benchmark.

## How the backend works

![]({{ site.baseurl }}/images/gsoc2026-gpu/architecture.png "Backend architecture")

Every kernel is a Rust `&'static str` holding CUDA C. On the first call NVRTC
compiles it to PTX and the result goes into a per-process
`OnceLock<CudaKernel>`. Every call after that gets the cached kernel, with no
recompilation per call or per image size. NVRTC takes 300-800 ms on the bigger
kernels, which nobody notices spread across a process lifetime and everybody
would notice per frame.

Two things fall out of this that I like. Compute capability gets detected at
runtime and memoised, so one build of the crate runs on any NVIDIA GPU. And each
kernel sets `CU_FUNC_CACHE_PREFER_L1` after compiling, taking Turing's L1 from
32 KB to 64 KB. That helps the kernels doing scattered reads, bicubic with its
4x4 taps and Lanczos with 6x6, and costs nothing in the source.

![]({{ site.baseurl }}/images/gsoc2026-gpu/memory-model.png "Tensor memory model")

For memory, a `Tensor<T, N>` owns a `Box<dyn MemoryResource>` that carries a
`MemoryDomain`: `Host`, `Device { id }`, or `Unified { id }`. Host slice access
asserts the domain is host-accessible, so passing a device tensor to a CPU API
fails loudly instead of quietly dereferencing device memory from the host.

The domain drives dispatch too. From a caller's side `resize`, `warp_affine`,
`warp_perspective` and the colour functions look exactly like they did before.
Host operands take the CPU path, device operands launch a kernel, a mixed pair is
a typed error. Nothing transfers implicitly. If your data crosses PCIe it's
because you wrote the line that moved it. There's a section further down on why
that turned out to matter more than I expected.

## Texture objects, tried twice, removed twice

Warp-affine with `BORDER_CONSTANT` normally wants a bounds check in every thread:
if the inverse-mapped coordinate falls outside the source, write zero. Rotate by
45° and roughly half your output pixels are black corners, so that branch
diverges badly.

Bind the source as a CUDA texture with `CU_TR_ADDRESS_MODE_BORDER` and the branch
goes away, because out-of-bounds fetches return the border colour in hardware. It
worked, it was faster, it shipped.

So naturally I tried the same trick on resize, where it made things about 10%
slower.

Resize has none of the properties that make textures worth it. `tex2D` costs
around 100 cycles of latency on Turing where `__ldg` costs about 30, and you earn
that back through 2D spatial caching and hardware boundary handling. But
sequential downscale reads consecutive source columns, which the unified L1
already coalesces fine, and there's no out-of-bounds divergence to remove. You
pay the latency and get nothing.

Then the warp-affine texture path had to go as well, and not for speed. A
1-channel pitch-2D texture gives `pitchInBytes = src_w * 3 * 4`, and CUDA wants
that to be a multiple of `CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT`, which is
32 bytes. Our rows are densely packed. So the kernels failed outright on every
source width that isn't a multiple of 8. Getting the alignment would have meant a
padded device-to-device copy on every single call, which costs more than the
divergence it was saving.

Textures are gone from the whole backend now and everything reads through
`__ldg`. I'm still slightly annoyed about this one, because the warp version was
genuinely a good idea and it only failed on image sizes I hadn't thought to test.

## One design question worth the detour

Warp-perspective and remap overlap a lot. Remap samples a source image at
caller-supplied `(map_x, map_y)` coordinates, and a perspective warp is really
just one particular coordinate map. So should perspective be a thin wrapper that
precomputes a map and calls remap, or its own fused kernel?

The generic version is better to maintain, so I wanted it to win. I benchmarked
before committing to it. Remap-bilinear came within 6% of the fused kernel, but
remap-nearest was 30-51% slower. Reading a precomputed coordinate map costs two
extra float loads per pixel. Next to bilinear's four samples that's noise. Next
to nearest-neighbour's single fetch it's most of the work.

So both exist. Remap is the general primitive that lens undistortion builds on,
and the fused warps stay fused where the arithmetic is cheap enough that map
traffic dominates.

## Results

![]({{ site.baseurl }}/images/gsoc2026-gpu/benchmarks.png "Benchmarks vs OpenCV CUDA and PyTorch")

GTX 1650, against OpenCV 4.12 CUDA and PyTorch 2.9+cu128. All three re-measured
on the same card with the same loop: 50 warmup, 200 timed iterations, one sync
after the batch.

Resize 1920x1080 to 960x540, and warp-affine at 45°, kernel time:

| Operation | kornia-rs | cv2 CUDA | PyTorch | vs cv2 | vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| resize nearest | 0.107 ms | 0.217 ms | 0.109 ms | **2.0x** | 1.0x |
| resize bilinear | 0.178 ms | 0.291 ms | 0.182 ms | **1.6x** | 1.0x |
| resize bicubic | 0.245 ms | 0.569 ms | 1.493 ms | **2.3x** | **6.1x** |
| warp-affine | 0.592 ms | 0.763 ms | 3.177 ms | 1.3x | **5.4x** |

We beat OpenCV CUDA on everything and roughly tie PyTorch on the cheap
interpolants, then pull well ahead on the expensive ones. The PyTorch gap is
widest on warp because `F.affine_grid` plus `F.grid_sample` allocates an
intermediate coordinate-grid tensor on every call, and kornia-rs allocates
nothing intermediate.

Colour conversion hits 170 GB/s at 1080p, which is 89% of the card's theoretical
peak. That's the number I like most, because it isn't a comparison against
anyone. It just says there isn't much left to get.

### What byte-exactness cost

There's one number in that table I'd have written differently a few months ago.
When I wrote a paper on this work mid-project, bilinear resize measured 0.101 ms.
Today the same kernel on the same card measures 0.178 ms, about 70% slower.

Nothing broke. `CudaKernel::compile` now passes `--fmad=false` to NVRTC, which
turns off implicit fused multiply-add contraction so a plain `a*b + c` in the
kernel rounds twice, exactly the way the same expression rounds on the CPU.
That's what makes the GPU output bit-identical to the CPU path instead of merely
close, and bit-identical is what the parity tests actually check.

Giving up FMA contraction is not a small thing, and the cost landed unevenly.
Bicubic didn't regress at all, 0.245 ms then and now, because its Horner weights
were rewritten to call `fmaf()` explicitly. Kernels that want fusion can still
have it, they just have to ask. Bilinear's tap accumulation is plain `a*b + c`,
so bilinear pays.

Was that worth 70% of one kernel? I think so, though I went back and forth. A GPU
path that's *almost* the same as the CPU path generates bug reports nobody can
reproduce, and "almost" compounds as you chain operations. It's still a real
trade, and I'd rather say so than keep quoting the faster number from the older
document.

### The result that reframed the project

Everything above is kernel time, which is the fair comparison against cv2 CUDA.
It is not what a caller experiences if the data starts on the host. So I built a
benchmark that separates host-to-device transfer, kernel, and device-to-host:

| Operation | Resolution | CPU | H2D | Kernel | D2H | Kernel | Round-trip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resize (f32) bilinear | 1080p to 540p | 5.28 ms | 9.16 ms | 0.18 ms | 2.21 ms | **28.7x** | **0.5x** |
| gray_from_rgb (u8) | 4K | 6.18 ms | 9.61 ms | 0.30 ms | 8.78 ms | **20.8x** | **0.3x** |

The kernel is 20-30x faster. The round trip is slower than just doing the work on
the CPU. For anything bandwidth-bound, PCIe dominates so completely that the
kernel time barely shows up.

This was deflating for about a day, then it became the most useful thing I
learned. It's the whole argument for refusing implicit transfers. A backend that
quietly uploaded and downloaded around every call would be slower than the CPU
path while looking like an optimization, and nobody would be able to see it
happening. The device path earns its keep when a tensor goes to the device and
stays there across a chain of operations, which is what the domain dispatch is
built to allow.

## Three things I got wrong

That mid-project paper had a "future work" section, which means I got to test my
own predictions four months later. I don't fully recommend the experience.

**Shared-memory tiling. I predicted ~20% faster. It measured 1.5x slower at 2x
downscale and 7x slower at 4x.** This is the textbook optimization: neighbouring
threads read overlapping source pixels, so stage a tile in shared memory and let
the block compute from there. The problem is that bilinear samples exactly four
source pixels per output pixel no matter what the scale factor is. At 4x
downscale, adjacent outputs are reading regions four pixels apart, so there's
barely any overlap to exploit, while the tile load faithfully fetches everything
in the region including the three quarters nobody asked for. I took a sparse,
well-cached access pattern and made it dense. The clue had been sitting in my own
benchmark output for weeks: downscale was already running at 84-91% of achievable
DRAM bandwidth, so there was no headroom for a cache trick to find.

**`INTER_AREA`. I predicted 15-30% faster. It did nothing.** This was the
follow-up that tiling was supposed to enable, since area-averaging for integer
ratios ought to be cheaper than bilinear. It wasn't, and that branch is still
sitting there unopened.

**Unified memory. A real win, on half the hardware.** `cudaMallocManaged` gives
you one pointer valid on both host and device. On a Jetson, where CPU and GPU
share physical DRAM, that should remove the transfers completely. It does. On a
discrete card it loses badly:

| Size | Jetson explicit | Jetson unified | | GTX 1650 explicit | GTX 1650 unified | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGA 640x480 | 2.509 ms | 1.813 ms | **1.38x** | 2.329 ms | 4.053 ms | 0.57x |
| FHD 1920x1080 | 12.639 ms | 7.410 ms | **1.71x** | 17.199 ms | 26.041 ms | 0.66x |
| 4K 3840x2160 | 47.318 ms | 27.517 ms | **1.72x** | 67.520 ms | 102.180 ms | 0.66x |

Output is bit-identical on both machines. On the integrated part it removes a
real copy. On the discrete one the driver demand-pages over PCIe and ends up
paying more than the copies it removed.

There's a second trap hiding in the allocation itself. `cuMemAllocManaged` is
roughly 3000x more expensive than a pooled device allocation, 42 ms against
0.013 ms for a 4K buffer. It amortises after about one frame, but an
allocate-per-frame loop wipes out the entire benefit. Unified buffers have to be
allocated once and reused.

So it shipped with that caveat attached, and the benchmark reports
`cudaDevAttrIntegrated` so any run tells you which case it's measuring.

## The bug that wasn't

Chasing exactness produced my favourite non-bug of the summer. Bilinear
warp-affine was diverging from OpenCV by up to 0.82 at non-identity rotations,
and it showed up right after a byte-exactness rewrite landed. It looked bad.

It turned out to be neither a regression nor new. The CUDA kernel clamps the `+1`
tap at source edges, `BORDER_REPLICATE`, because that's what the CPU
`warp_affine` loop does and matching the CPU was the contract. OpenCV uses
`BORDER_CONSTANT`. The two disagree at exactly one pixel of border, and only
where the inverse-mapped coordinate lands on an edge, which is why identity
transforms matched to 8e-9 while rotations didn't. The rewrite hadn't introduced
anything. It had tightened the CPU/GPU match enough to make a pre-existing
CPU-versus-OpenCV difference visible. The fix went into the test, not the kernel.

The parity suite only existed because my mentor asked for it early on: *"make
sure all the algorithms match output with the reference libraries. I found in
some cases that the algorithms get faster but the results are not exactly the
same."* Best process advice I got all summer.

## What shipped

- **Resize**: nearest, bilinear, bicubic, Lanczos-3, plus byte-exact u8 paths
- **Warp-affine and warp-perspective**: same four interpolants
- **Remap**: the generic primitive behind lens undistortion, f32 and u8
- **Colour conversion**: RGB to gray, HSV, HLS, YCbCr, BGR, Bayer demosaic
- **Device-aware tensor storage**: `MemoryDomain`, pinned and unified allocators, DLPack-compatible foreign import
- **Correctness tooling**: pixel-level parity against OpenCV CUDA and NVIDIA VPI
- **A reproducible benchmark suite** with the H2D/kernel/D2H split, at 1080p and 4K, on desktop and Jetson

Element-wise and reduction tensor ops using the same domain dispatch, along with
Laplacian and integral filters, are in review.

Next on my list: `INTER_AREA` done properly, tiling revisited for the kernels
that actually reuse a tile, and Python bindings for the Jetson unified-memory
path. Edgar also floated the idea of letting users register their own CUDA
kernels at runtime. The `OnceLock` cache extends to that fairly naturally, and it
would make the backend extensible without anyone having to fork kornia-rs.

## Thanks

To Edgar Riba, for reviews that caught things I'd have shipped, for the Jetson
access that made half of these numbers possible, and for asking about byte-exact
tests months before I'd have thought to. And to the kornia community generally.
All of it lives in [kornia/kornia-rs](https://github.com/kornia/kornia-rs) behind
the `cuda` feature flag.

---

*AI-use disclosure: portions of this post were drafted with assistance from
Claude. All technical content, benchmarks, and design decisions are my own work.*
