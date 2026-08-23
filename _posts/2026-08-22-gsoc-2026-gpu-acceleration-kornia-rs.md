---
toc: true
layout: post
description: A CUDA backend for kornia-rs, and the three optimizations I was sure about that turned out to be wrong.
categories: [gsoc, announcement]
image: images/gsoc2026-gpu/benchmarks.png
author: [Inchara J]
title: "GSoC 2026: GPU acceleration for kornia-rs"
---

I'm Inchara J, and this summer I worked on kornia-rs for Google Summer of Code.
I built a CUDA backend for it: GPU kernels for resize, warp, remap and colour
conversion, plus the memory model underneath them.

Rust doesn't really have a GPU vision library. If you want a fast `resize` or
`warp_affine` from Rust today, you either call into C++ or go through Python and
pay for it. The tricky part isn't the maths, it's fitting CUDA into Rust
properly: keeping `unsafe` in one small place, making device memory follow Rust's
ownership rules, and not freezing the first call while something compiles.

## Picking a backend, then changing my mind

I started with [CubeCL](https://github.com/tracel-ai/cubecl), a Rust GPU kernel
DSL. Writing kernels in Rust instead of CUDA C sounded great, and I shipped three
PRs on it: the allocator scaffold, the backend, and nearest plus bilinear resize.

The kernels worked. They were also slow. Downscale ran at 8-12 GB/s on a card
that can do 192 GB/s.

To fix it I needed `__ldg`, the read-only cache load, and control over the L1
cache setup. CubeCL doesn't give you either, and it can't really, because it
targets CUDA and WGPU and others through one common layer and both of those are
CUDA-only things. So I rewrote the kernels in plain CUDA C and compiled them at
runtime with [NVRTC](https://docs.nvidia.com/cuda/nvrtc/), using
[cudarc](https://github.com/coreylowman/cudarc) for the driver bindings.

Downscale jumped to about 70 GB/s. Six times faster, just from being allowed to
write two things the abstraction wouldn't let me write.

There was a second reason I hadn't even thought about. A reviewer pointed out
that turning on the CubeCL feature pulled its whole compiler stack, an MLIR
pipeline and an LLVM downloader, into everyone's build. It had also dragged a
`js-sys` dependency into the workspace, which is usually a sign you've gone wrong
somewhere in a computer vision crate. kornia-rs moved to cudarc soon after.

Throwing away three PRs stung. But I don't think I could have made the argument
without building on CubeCL first.

## How it works

![]({{ site.baseurl }}/images/gsoc2026-gpu/architecture.png "Backend architecture")

Every kernel is a Rust `&'static str` holding CUDA C. The first call compiles it
to PTX with NVRTC and stores it in a `OnceLock<CudaKernel>`. Every call after
that gets the cached one, with no recompiling per call or per image size. NVRTC
takes 300-800 ms on the bigger kernels, which nobody notices once per process and
everybody would notice once per frame.

Compute capability is checked at runtime, so one build runs on any NVIDIA GPU.
Each kernel also sets `CU_FUNC_CACHE_PREFER_L1`, which takes Turing's L1 from
32 KB to 64 KB and helps the kernels doing scattered reads.

![]({{ site.baseurl }}/images/gsoc2026-gpu/memory-model.png "Tensor memory model")

For memory, a `Tensor<T, N>` owns a `Box<dyn MemoryResource>` that carries a
`MemoryDomain`: `Host`, `Device`, or `Unified`. Host slice access checks that the
domain is host-readable, so passing a device tensor into a CPU function fails
loudly instead of quietly reading device memory from the CPU.

That same domain picks the path. `resize`, `warp_affine`, `warp_perspective` and
the colour functions look exactly like they did before. Host inputs go to the CPU
code, device inputs launch a kernel, and mixing the two is an error. Nothing
moves between host and device on its own. If your data crosses PCIe, you wrote
the line that moved it.

## Texture objects, tried twice, dropped twice

Warp-affine with `BORDER_CONSTANT` normally needs a bounds check in every thread:
if the mapped coordinate lands outside the image, write zero. Rotate by 45 degrees
and about half your output pixels are black corners, so that branch splits the
warp badly.

Bind the source as a CUDA texture with `CU_TR_ADDRESS_MODE_BORDER` and the branch
disappears, because out-of-bounds reads return the border colour in hardware. It
worked, it was faster, it shipped.

So I tried it on resize too, where it made things about 10% slower. `tex2D` costs
around 100 cycles on Turing where `__ldg` costs about 30, and you're meant to win
that back through 2D caching. But downscale reads straight along rows, which L1
handles fine already, and there's no branch to remove. You pay and get nothing.

Then the warp version had to go as well, and not because it was slow. A
1-channel pitch-2D texture makes `pitchInBytes = src_w * 3 * 4`, and CUDA needs
that to be a multiple of 32 bytes. Our rows are packed tight, so the kernels just
failed on any image width that isn't a multiple of 8. Fixing the alignment would
mean copying the image into a padded buffer on every call, which costs more than
the branch it was saving.

So textures are gone everywhere and everything reads through `__ldg`. I'm still a
bit annoyed about that one, because the warp version was a good idea and it only
broke on image sizes I hadn't tested.

## Results

![]({{ site.baseurl }}/images/gsoc2026-gpu/benchmarks.png "Kernel times against OpenCV CUDA and PyTorch, and round-trip on two cards")

GTX 1650, against OpenCV 4.12 CUDA and PyTorch 2.9. All three measured on the
same card with the same loop: 50 warmup, 200 timed runs, one sync at the end.

Resize 1920x1080 to 960x540, and warp-affine at 45 degrees, kernel time:

| Operation | kornia-rs | cv2 CUDA | PyTorch | vs cv2 | vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| resize nearest | 0.107 ms | 0.217 ms | 0.109 ms | **2.0x** | 1.0x |
| resize bilinear | 0.178 ms | 0.291 ms | 0.182 ms | **1.6x** | 1.0x |
| resize bicubic | 0.245 ms | 0.569 ms | 1.493 ms | **2.3x** | **6.1x** |
| warp-affine | 0.592 ms | 0.763 ms | 3.177 ms | 1.3x | **5.4x** |

We beat OpenCV CUDA on all of them, tie PyTorch on the cheap ones and pull ahead
on the expensive ones. The PyTorch gap is biggest on warp because
`F.affine_grid` plus `F.grid_sample` builds a whole coordinate grid tensor every
call, and we build nothing.

Colour conversion hits 170 GB/s at 1080p, which is 89% of what the card can do.
That's the number I like most, because it isn't a comparison with anyone. It just
says there isn't much left to get.

### What byte-exactness cost

Bilinear resize used to measure 0.101 ms. It's 0.178 ms now, about 70% slower,
and nothing broke.

`CudaKernel::compile` passes `--fmad=false` to NVRTC now, which stops the
compiler fusing multiply and add. A plain `a*b + c` then rounds twice, exactly
like it does on the CPU. That's what makes the GPU output bit-identical to the
CPU output instead of just close, and bit-identical is what the tests check.

It didn't hit everything equally. Bicubic didn't slow down at all, because its
weights were changed to call `fmaf()` directly, so kernels that want fusion can
still ask for it. Bilinear's plain `a*b + c` pays.

Worth 70% of one kernel? I think so, though I went back and forth. A GPU path
that's *almost* the same as the CPU one gives you bug reports nobody can
reproduce.

### The result that changed how I think about this

Those are kernel times, which is the fair comparison against cv2 CUDA. It isn't
what you actually get if your data starts on the CPU. So I built a benchmark that
splits out upload, kernel, and download.

On my GTX 1650:

| Operation | Resolution | CPU | H2D | Kernel | D2H | Kernel | Round trip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resize (f32) bilinear | 1080p to 540p | 5.37 ms | 9.22 ms | 0.18 ms | 2.29 ms | **29.2x** | **0.5x** |
| gray_from_rgb (u8) | 4K | 2.57 ms | 9.22 ms | 0.19 ms | 2.94 ms | **13.2x** | **0.2x** |

The kernel is 13 to 29 times faster and the round trip is still slower than just
using the CPU. The transfers cost about sixty times what the kernel does. That's
the whole reason the API refuses to move data on its own: a backend that quietly
uploaded and downloaded around every call would be slower than the CPU while
looking like a speedup, and you'd have no way to see it.

I nearly wrote that up as a general fact about GPUs. Then a friend ran the same
benchmark, same commit, on an RTX 3090:

| resize f32 bilinear, 1080p | CPU | H2D | Kernel | D2H | Round trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| GTX 1650 | 5.37 ms | 9.22 ms | 0.18 ms | 2.29 ms | **0.5x** |
| RTX 3090 | 6.95 ms | 2.71 ms | 0.04 ms | 1.11 ms | **1.8x** |

The kernel got 4.5x faster, which I expected. The transfers got 3.4x faster on
the way up and 2x on the way down, which I hadn't thought about at all. On the
3090 the round trip wins on 37 of the 58 operations, by up to 38x.

So "PCIe dominates" was never a fact about GPUs. It was a fact about my GPU, and
I was one sample away from writing it down as a law. What actually holds is
smaller: whether a transfer pays for itself depends on your link and your CPU, it
changes by more than 10x between cards, and you have to measure the machine
you're shipping on. Which is why the benchmark prints the split instead of one
number.

Either way, the GPU path is at its best when a tensor goes to the device and
*stays* there for several operations, so the transfer is paid once.

## Three things I got wrong

Halfway through I wrote a paper on this, and its future work section made
predictions I then got to test. I don't fully recommend the experience.

**Shared memory tiling. I said ~20% faster. It was 1.5x slower at 2x downscale
and 7x slower at 4x.** The classic move: neighbouring threads read overlapping
pixels, so load a tile into shared memory once and work from there. But bilinear
reads exactly four source pixels per output pixel whatever the scale. At 4x those
four are spread far apart, so there's barely any overlap to reuse, while the tile
load fetches everything in between anyway. I turned a sparse, well-cached read
pattern into a dense one. The clue had been in my own benchmark output for weeks:
downscale was already at 84-91% of the DRAM bandwidth I could get, so there was
nothing left for a cache trick to find.

**`INTER_AREA`. I said 15-30% faster. It did nothing.** This was the follow-up
that tiling was supposed to enable. It wasn't faster, and the branch is still
sitting there unopened.

**Unified memory. A real win, on half the hardware.** `cudaMallocManaged` gives
you one pointer that works on both sides. On a Jetson, where CPU and GPU share
the same RAM, that should remove the copies completely. It does. On a discrete
card it loses badly:

| Size | Jetson explicit | Jetson unified | | GTX 1650 explicit | GTX 1650 unified | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGA 640x480 | 2.509 ms | 1.813 ms | **1.38x** | 2.329 ms | 4.053 ms | 0.57x |
| FHD 1920x1080 | 12.639 ms | 7.410 ms | **1.71x** | 17.199 ms | 26.041 ms | 0.66x |
| 4K 3840x2160 | 47.318 ms | 27.517 ms | **1.72x** | 67.520 ms | 102.180 ms | 0.66x |

Output is bit-identical on both. On the Jetson it removes a real copy. On the
discrete card the driver pages memory over PCIe on demand and ends up paying more
than the copies it removed.

There's a second trap in the allocation. `cuMemAllocManaged` is about 3000x more
expensive than a normal device alloc, 42 ms against 0.013 ms for a 4K buffer. So
you allocate once and reuse, or you lose the whole benefit. The benchmark reports
whether the GPU is integrated, so a run tells you which case you're in.

## The bug that wasn't

Chasing exactness turned up my favourite non-bug. Bilinear warp-affine was off
from OpenCV by up to 0.82 at rotated angles, right after a byte-exactness rewrite
landed. It looked bad.

![]({{ site.baseurl }}/images/gsoc2026-gpu/parity.png "CPU/GPU parity and the OpenCV border seam")

It wasn't a regression and it wasn't new. The middle panel is what `--fmad=false`
bought: same call on CPU and GPU, and not one of 262,144 pixels differs. The
right panel is where the 0.82 lives. Our kernel clamps the `+1` tap at the image
edge because that's what the CPU code does, and OpenCV zero-fills instead. They
disagree on one pixel of border, only where the mapped coordinate lands on an
edge. At 45 degrees that edge is the diamond you can see, which is why identity
transforms matched to 8e-9 and rotations didn't.

The fix went into the test, not the kernel.

That test suite only existed because my mentor asked for it early on: *"make sure
all the algorithms match output with the reference libraries. I found in some
cases that the algorithms get faster but the results are not exactly the same."*
Best advice I got all summer.

## What shipped

Resize with four interpolations plus byte-exact u8 paths, warp-affine and
warp-perspective with the same four, remap in f32 and u8, six colour conversions,
the device-aware tensor storage under all of it, correctness checks against
OpenCV CUDA and NVIDIA VPI, and a benchmark suite that splits upload, kernel and
download at 1080p and 4K on desktop and Jetson.

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
`bench_cuda_imgproc` sweep, all 58 operations, on both cards, so the selection
can be checked against the rest.

```sh
git clone -b bench/all-gpu-work https://github.com/Incharajayaram/kornia-rs
cd kornia-rs && cargo bench --bench bench_cuda_imgproc --features cuda
```

Commit `b887ffd` on both machines, 30 warmup and 100 timed iterations, CUDA
events, rotating source buffers. All times in milliseconds.

A measurement note that cost me several hours. Kernel times are hardware
determined and reproduce exactly: across five runs on the GTX 1650 at different
system loads, governors and memory states, the kernel column never moved by more
than a rounding digit. Host-side numbers are a different story. A browser left
open in the background doubled the CPU baseline **and** the H2D time, because
pageable transfers are CPU-bound memcpy and the machine has only 7 GB of RAM. The
1650 figures below come from a run with the desktop otherwise idle, which
reproduces an earlier clean run to within 2%. The 3090 host is a virtualised
Haswell vCPU, so its CPU baseline is weak and its CPU-relative ratios are
optimistic. Compare kernel columns freely; treat host columns as belonging to the
machine and the moment they were taken on.

**The gain from a bigger GPU is not a single number.** Median 4.9x, ranging from
11.6x on `warp_affine u8` down to 1.0x on `integral` at 1080p. Point-sampling
kernels that are memory-bound scale with the card. A prefix sum does not, because
the dependency chain is serial no matter how many SMs you have. Lanczos sits near
the bottom too (1.4x) for the same reason: it is compute-bound on a fixed tap
count rather than starved for bandwidth.

| Operation | Interp | Resolution | 1650 CPU | 1650 kernel | 1650 trip | 3090 CPU | 3090 kernel | 3090 trip | 3090/1650 kernel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize (f32) | bilinear | 1920×1080→960×540 | 5.37 | 0.18 | 0.5x | 6.95 | 0.04 | 1.8x | 4.5x |
| resize (f32) | bilinear | 3840×2160→1920×1080 | 22.03 | 0.71 | 0.5x | 20.66 | 0.15 | 1.5x | 4.7x |
| resize (f32) | nearest | 1920×1080→960×540 | 2.51 | 0.11 | 0.2x | 2.85 | 0.03 | 0.8x | 3.7x |
| resize (f32) | nearest | 3840×2160→1920×1080 | 13.44 | 0.43 | 0.3x | 8.54 | 0.10 | 0.6x | 4.3x |
| resize (f32) | bicubic | 1920×1080→960×540 | 25.44 | 0.24 | 2.2x | 29.24 | 0.05 | 8.4x | 4.8x |
| resize (f32) | bicubic | 3840×2160→1920×1080 | 95.17 | 0.93 | 2.0x | 87.69 | 0.16 | 6.9x | 5.8x |
| resize (f32) | lanczos | 1920×1080→960×540 | 5.32 | 0.38 | 0.4x | 5.25 | 0.30 | 1.3x | 1.3x |
| resize (f32) | lanczos | 3840×2160→1920×1080 | 22.29 | 1.48 | 0.5x | 18.67 | 0.87 | 1.3x | 1.7x |
| resize (u8) | bilinear | 1920×1080→960×540 | 5.18 | 0.07 | 1.8x | 5.27 | 0.02 | 6.7x | 3.5x |
| resize (u8) | bilinear | 3840×2160→1920×1080 | 22.21 | 0.24 | 1.9x | 18.43 | 0.05 | 5.7x | 4.8x |
| resize (u8) | nearest | 1920×1080→960×540 | 2.55 | 0.04 | 0.9x | 2.07 | 0.01 | 2.7x | 4.0x |
| resize (u8) | nearest | 3840×2160→1920×1080 | 13.37 | 0.12 | 1.1x | 6.03 | 0.03 | 1.8x | 4.0x |
| warp_affine (30° rot, f32) | bilinear | 1920×1080 | 9.93 | 0.51 | 0.5x | 4.45 | 0.06 | 0.9x | 8.5x |
| warp_affine (30° rot, f32) | bilinear | 3840×2160 | 43.73 | 2.11 | 0.6x | 19.61 | 0.25 | 0.8x | 8.4x |
| warp_affine (30° rot, u8) | bilinear | 1920×1080 | 2.88 | 0.55 | 0.6x | 2.28 | 0.05 | 1.7x | 11.0x |
| warp_affine (30° rot, u8) | bilinear | 3840×2160 | 14.38 | 2.21 | 0.7x | 9.56 | 0.19 | 1.8x | 11.6x |
| warp_perspective (30° rot, f32) | bilinear | 1920×1080 | 19.37 | 0.50 | 1.0x | 13.78 | 0.07 | 2.9x | 7.1x |
| warp_perspective (30° rot, f32) | bilinear | 3840×2160 | 83.79 | 2.04 | 1.1x | 50.39 | 0.26 | 2.1x | 7.8x |
| warp_perspective (30° rot, u8) | bilinear | 1920×1080 | 3.08 | 0.60 | 0.6x | 1.82 | 0.07 | 1.5x | 8.6x |
| warp_perspective (30° rot, u8) | bilinear | 3840×2160 | 15.46 | 2.43 | 0.8x | 8.03 | 0.25 | 1.7x | 9.7x |
| remap (f32) | bilinear | 1920×1080 | 18.86 | 0.39 | 1.0x | 12.88 | 0.09 | 2.8x | 4.3x |
| remap (f32) | bilinear | 3840×2160 | 75.46 | 1.57 | 1.0x | 50.15 | 0.32 | 2.1x | 4.9x |
| gaussian_blur (5x5, f32) |  | 1920×1080 | 26.93 | 0.59 | 1.4x | 42.09 | 0.13 | 9.2x | 4.5x |
| gaussian_blur (3x3, u8) |  | 1920×1080 | 0.63 | 0.26 | 0.1x | 0.44 | 0.04 | 0.4x | 6.5x |
| box_blur (3x3, u8) |  | 1920×1080 | 7.45 | 0.27 | 1.6x | 6.28 | 0.04 | 5.4x | 6.8x |
| sobel (3x3, f32) |  | 1920×1080 | 55.96 | 1.60 | 2.9x | 106.46 | 0.34 | 22.3x | 4.7x |
| laplacian (3x3, u8) |  | 1920×1080 | 1.69 | 0.10 | 0.7x | 1.59 | 0.02 | 2.3x | 5.0x |
| integral (u8) |  | 1920×1080 | 1.47 | 1.24 | 0.3x | 3.08 | 1.19 | 1.1x | 1.0x |
| gaussian_blur (5x5, f32) |  | 3840×2160 | 131.38 | 2.46 | 1.8x | 230.99 | 0.48 | 9.8x | 5.1x |
| gaussian_blur (3x3, u8) |  | 3840×2160 | 5.49 | 1.01 | 0.3x | 1.64 | 0.14 | 0.3x | 7.2x |
| box_blur (3x3, u8) |  | 3840×2160 | 30.04 | 1.06 | 1.6x | 23.31 | 0.14 | 4.9x | 7.6x |
| sobel (3x3, f32) |  | 3840×2160 | 277.13 | 6.42 | 3.5x | 413.50 | 1.30 | 17.3x | 4.9x |
| laplacian (3x3, u8) |  | 3840×2160 | 6.76 | 0.39 | 0.7x | 6.59 | 0.06 | 3.0x | 6.5x |
| integral (u8) |  | 3840×2160 | 10.79 | 3.57 | 0.6x | 9.88 | 2.41 | 1.3x | 1.5x |
| erode (3x3, u8) |  | 1920×1080 | 48.56 | 0.29 | 10.2x | 45.38 | 0.05 | 38.6x | 5.8x |
| dilate (3x3, u8) |  | 1920×1080 | 47.55 | 0.29 | 9.9x | 32.89 | 0.04 | 27.8x | 7.2x |
| erode (3x3, u8) |  | 3840×2160 | 199.04 | 0.93 | 10.4x | 138.34 | 0.13 | 29.3x | 7.2x |
| dilate (3x3, u8) |  | 3840×2160 | 205.33 | 0.93 | 10.7x | 127.20 | 0.13 | 27.3x | 7.2x |
| gray_from_rgb (f32) |  | 1920×1080 | 2.60 | 0.19 | 0.2x | 0.81 | 0.05 | 0.3x | 3.8x |
| gray_from_rgb (f32) |  | 3840×2160 | 11.68 | 0.75 | 0.2x | 3.95 | 0.16 | 0.3x | 4.7x |
| remap (u8) | bilinear | 1920×1080 | 4.76 | 0.23 | 1.0x | 2.93 | 0.04 | 2.4x | 5.8x |
| remap (u8) | nearest | 1920×1080 | 4.76 | 0.21 | 1.0x | 2.93 | 0.04 | 2.3x | 5.2x |
| remap (u8) | bilinear | 3840×2160 | 18.46 | 0.89 | 1.0x | 10.97 | 0.14 | 2.4x | 6.4x |
| remap (u8) | nearest | 3840×2160 | 18.46 | 0.82 | 1.0x | 10.97 | 0.14 | 2.4x | 5.9x |
| gray_from_rgb (u8) |  | 1920×1080 | 0.16 | 0.05 | 0.1x | 0.50 | 0.02 | 0.6x | 2.5x |
| gray_from_rgb (u8) |  | 3840×2160 | 2.57 | 0.19 | 0.2x | 0.88 | 0.05 | 0.3x | 3.8x |
| rgb_from_gray (u8) |  | 1920×1080 | 0.24 | 0.10 | 0.1x | 0.56 | 0.02 | 0.6x | 5.0x |
| rgb_from_gray (u8) |  | 3840×2160 | 4.64 | 0.36 | 0.4x | 1.19 | 0.05 | 0.4x | 7.2x |
| hsv_from_rgb (f32) |  | 1920×1080 | 5.84 | 0.30 | 0.3x | 2.29 | 0.07 | 0.5x | 4.3x |
| hsv_from_rgb (f32) |  | 3840×2160 | 24.96 | 1.16 | 0.3x | 8.50 | 0.24 | 0.3x | 4.8x |
| hls_from_rgb (f32) |  | 1920×1080 | 5.99 | 0.30 | 0.3x | 2.87 | 0.07 | 0.6x | 4.3x |
| hls_from_rgb (f32) |  | 3840×2160 | 24.95 | 1.16 | 0.3x | 9.37 | 0.24 | 0.4x | 4.8x |
| ycc_from_rgb (u8) |  | 1920×1080 | 1.20 | 0.08 | 0.3x | 1.30 | 0.02 | 1.1x | 4.0x |
| ycc_from_rgb (u8) |  | 3840×2160 | 5.80 | 0.30 | 0.3x | 4.28 | 0.07 | 0.9x | 4.3x |
| ycc_from_rgb (f32) |  | 1920×1080 | 6.02 | 0.30 | 0.3x | 1.83 | 0.07 | 0.4x | 4.3x |
| ycc_from_rgb (f32) |  | 3840×2160 | 25.27 | 1.16 | 0.3x | 8.01 | 0.24 | 0.3x | 4.8x |
| bgr_from_rgb (u8) |  | 1920×1080 | 0.69 | 0.08 | 0.2x | 0.75 | 0.02 | 0.6x | 4.0x |
| bgr_from_rgb (u8) |  | 3840×2160 | 6.03 | 0.30 | 0.3x | 2.08 | 0.07 | 0.4x | 4.3x |

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
you would hope for from a reduction that behaves the same way twice, and a bug that does too.

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
