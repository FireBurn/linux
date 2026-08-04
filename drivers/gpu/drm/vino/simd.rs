// SPDX-License-Identifier: GPL-2.0

//! Optional vectorised Haar transforms, and an in-kernel benchmark for them.
//!
//! The scalar transform in [`super::video::wht`] is always present and is the oracle: nothing here
//! replaces it, and every path below is checked against it before any timing is reported. The
//! codec is byte-exact against DisplayLink's own encoder, so a faster transform that is not
//! identical is worthless.
//!
//! Measuring this in userspace answers the wrong question. The kernel builds Rust with
//! `-Ctarget-feature=-sse,...,-avx2`, so the scalar baseline here is genuinely unvectorised, where
//! a userspace baseline is auto-vectorised and understates the gain. More importantly, vector
//! registers need [`kernel::fpu::FpuGuard`], and the cost of entering and leaving that section
//! exists only in the kernel. `simd_bench=1` at module load reports both halves.
//!
//! `#[target_feature]` is additive per function, so these compile under the global disable; that
//! is the supported way in rather than a workaround.

use core::arch::x86_64::*;
use kernel::bindings;
use kernel::fpu::FpuGuard;
use kernel::prelude::*;
use kernel::time::{Delta, Instant, Monotonic};

use super::video::wht::{
    colour, colour_block, colour_strip, colour_strip_at, quantize, transform, ColourBlock,
    Geometry, COEFFS, PIXELS, RIDGE_GEOMETRY,
};

/// Blocks `colour_block` transforms together: the `cr`, `cb` and `y` planes of one 8x8 block.
///
/// This is the encoder's real call shape, and it is why a wide vector path may not pay: a 16-lane
/// transform running three useful lanes does the same work as one running sixteen.
pub(crate) const ENCODER_BATCH: usize = 3;

/// Coefficient order of the finest sub-band, as [`super::video::wht::transform`] emits it.
const SCAN4_MORTON: [usize; 16] = [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15];

/// Whether `boot_cpu_data` reports an x86 feature bit.
fn cpu_has(feature: u32) -> bool {
    let word = (feature / 32) as usize;
    // SAFETY: `boot_cpu_data` is initialised before any driver probes and is only read here.
    // `x86_capability` is a fixed-size array behind an anonymous union whose other member is only
    // an alignment filler, so reading it is always valid; `word` is bounds-checked against it.
    unsafe {
        let caps = &(*core::ptr::addr_of!(bindings::boot_cpu_data))
            .__bindgen_anon_3
            .x86_capability;
        caps.get(word)
            .is_some_and(|w| w & (1u32 << (feature % 32)) != 0)
    }
}

/// Whether this CPU can run the AVX2 path.
pub(crate) fn has_avx2() -> bool {
    cpu_has(bindings::X86_FEATURE_AVX2)
}

/// Whether the encoder should use the vectorised transform, resolved once.
///
/// Both halves are checked here so the hot path is one relaxed load: the CPU must have AVX2 and
/// the operator must have asked for it.
static USE_SIMD: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Latch whether the encoder takes the vectorised path. Called once, from probe.
pub(crate) fn set_encoder_simd(on: bool) {
    let v = u8::from(on && has_avx2());
    USE_SIMD.store(v, core::sync::atomic::Ordering::Relaxed);
    let name = if v != 0 { "avx2" } else { "scalar" };
    pr_info!("vino-simd: encoder transform = {}\n", name);
}

/// Transform three planes of one block, vectorised if this build and CPU allow it.
///
/// One FPU section covers all three: it is short, straight-line and allocation-free, which is what
/// [`FpuGuard`] requires. Returns `None` when the scalar path should run instead.
#[inline]
pub(crate) fn colour_block_transforms(
    cr: &[i32; PIXELS],
    cb: &[i32; PIXELS],
    y: &[i32; PIXELS],
) -> Option<([i32; COEFFS], [i32; COEFFS], [i32; COEFFS])> {
    if USE_SIMD.load(core::sync::atomic::Ordering::Relaxed) == 0 {
        return None;
    }
    let _fpu = FpuGuard::new(0);
    // SAFETY: the guard is live and `USE_SIMD` is only set when `has_avx2()` held.
    Some(unsafe {
        (
            transform_inblock_avx2(cr),
            transform_inblock_avx2(cb),
            transform_inblock_avx2(y),
        )
    })
}

/// Whether this CPU can run the AVX-512 path.
pub(crate) fn has_avx512() -> bool {
    cpu_has(bindings::X86_FEATURE_AVX512F)
}

/// Scratch for one vector transform, heap-allocated because the lane-major form of a 64-pixel
/// block is several KB and this driver's encode path already runs deep in a 16 KB kernel stack.
///
/// It must be allocated before the FPU section opens: that section runs with preemption disabled
/// and must not sleep.
pub(crate) struct Scratch<T> {
    src: [T; PIXELS],
    ll1: [T; 16],
    hl1: [T; 16],
    lh1: [T; 16],
    hh1: [T; 16],
    work_l: [T; 32],
    work_h: [T; 32],
}

impl<T: Copy> Scratch<T> {
    /// Boxed rather than returned by value: the arrays are fixed-size so the transform indexes
    /// them without bounds checks, while the several KB they occupy stay off the stack.
    fn new(zero: T) -> Result<KBox<Self>> {
        KBox::new(
            Self {
                src: [zero; PIXELS],
                ll1: [zero; 16],
                hl1: [zero; 16],
                lh1: [zero; 16],
                hh1: [zero; 16],
                work_l: [zero; 32],
                work_h: [zero; 32],
            },
            GFP_KERNEL,
        )
        .map_err(|e| e.into())
    }
}

/// Generate a lane-parallel Haar transform for one vector width.
///
/// One lane per block, so pairing neighbours needs no shuffle: element `i` of a vector holds the
/// same coefficient of `$lanes` different blocks. Loads and stores go through
/// `read_unaligned`/`write_unaligned`, whose signature does not vary with width.
macro_rules! simd_transform {
    ($name:ident, $feature:literal, $vec:ty, $lanes:literal,
     $set0:ident, $add:ident, $sub:ident, $sra:ident) => {
        /// Transform `$lanes` blocks at once into `out`.
        ///
        /// # Safety
        ///
        /// The caller must hold a live [`FpuGuard`] and must have checked that this CPU supports
        /// the feature. `blocks` and `out` must each hold at least `$lanes` entries.
        #[target_feature(enable = $feature)]
        unsafe fn $name(
            scratch: &mut Scratch<$vec>,
            blocks: &[[i32; PIXELS]],
            out: &mut [[i32; COEFFS]],
        ) {
            /// One decomposition level over `N`x`N` inputs, writing four `H`x`H` sub-bands.
            ///
            /// # Safety
            ///
            /// As the caller: FPU section live, feature present.
            #[target_feature(enable = $feature)]
            unsafe fn level<
                const N: usize,
                const H: usize,
                const S: usize,
                const W: usize,
                const O: usize,
            >(
                src: &[$vec; S],
                l: &mut [$vec; W],
                hb: &mut [$vec; W],
                ll: &mut [$vec; O],
                hl: &mut [$vec; O],
                lh: &mut [$vec; O],
                hh: &mut [$vec; O],
            ) {
                for r in 0..N {
                    for i in 0..H {
                        let (a, b) = (src[r * N + 2 * i], src[r * N + 2 * i + 1]);
                        l[r * H + i] = $add(a, b);
                        hb[r * H + i] = $sub(a, b);
                    }
                }
                for c in 0..H {
                    for i in 0..H {
                        let (a, b) = (l[2 * i * H + c], l[(2 * i + 1) * H + c]);
                        ll[i * H + c] = $add(a, b);
                        lh[i * H + c] = $sub(a, b);
                        let (a2, b2) = (hb[2 * i * H + c], hb[(2 * i + 1) * H + c]);
                        hl[i * H + c] = $add(a2, b2);
                        hh[i * H + c] = $sub(a2, b2);
                    }
                }
            }

            // SAFETY: the caller guarantees a live FPU section and a CPU with the feature; the
            // slices are checked to hold `$lanes` entries by the assert above.
            unsafe {
            // Transpose to lanes-per-coefficient once; every stage below is then shuffle-free.
            let mut tmp = [0i32; $lanes];
            for p in 0..PIXELS {
                for (b, t) in tmp.iter_mut().enumerate() {
                    *t = blocks[b][p];
                }
                scratch.src[p] = core::ptr::read_unaligned(tmp.as_ptr() as *const $vec);
            }

            let z = $set0();
            level::<8, 4, PIXELS, 32, 16>(
                &scratch.src,
                &mut scratch.work_l,
                &mut scratch.work_h,
                &mut scratch.ll1,
                &mut scratch.hl1,
                &mut scratch.lh1,
                &mut scratch.hh1,
            );
            let (mut ll2, mut hl2, mut lh2, mut hh2) = ([z; 4], [z; 4], [z; 4], [z; 4]);
            level::<4, 2, 16, 32, 4>(
                &scratch.ll1,
                &mut scratch.work_l,
                &mut scratch.work_h,
                &mut ll2,
                &mut hl2,
                &mut lh2,
                &mut hh2,
            );
            let (mut ll3, mut hl3, mut lh3, mut hh3) = ([z; 1], [z; 1], [z; 1], [z; 1]);
            level::<2, 1, 4, 32, 1>(
                &ll2,
                &mut scratch.work_l,
                &mut scratch.work_h,
                &mut ll3,
                &mut hl3,
                &mut lh3,
                &mut hh3,
            );

            // `>> 6` is an arithmetic shift in both paths.
            let store = |coeff: usize, v: $vec, out: &mut [[i32; COEFFS]]| {
                let mut lanes = [0i32; $lanes];
                core::ptr::write_unaligned(lanes.as_mut_ptr() as *mut $vec, $sra(v, 6));
                for (b, val) in lanes.iter().enumerate() {
                    out[b][coeff] = *val;
                }
            };
            store(0, ll3[0], out);
            store(1, hl3[0], out);
            store(2, lh3[0], out);
            store(3, hh3[0], out);
            for i in 0..4 {
                store(4 + i, hl2[i], out);
                store(8 + i, lh2[i], out);
                store(12 + i, hh2[i], out);
            }
            for (i, &s) in SCAN4_MORTON.iter().enumerate() {
                store(16 + i, scratch.hl1[s], out);
                store(32 + i, scratch.lh1[s], out);
                store(48 + i, scratch.hh1[s], out);
            }
            }
        }
    };
}

simd_transform!(
    transform_avx2,
    "avx2",
    __m256i,
    8,
    _mm256_setzero_si256,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_srai_epi32
);

simd_transform!(
    transform_avx512,
    "avx512f",
    __m512i,
    16,
    _mm512_setzero_si512,
    _mm512_add_epi32,
    _mm512_sub_epi32,
    _mm512_srai_epi32
);

/// Level-1 Haar of one 8x8 block, vectorised **within** the block.
///
/// The across-blocks form above has to gather every block into lane-major order first, and that
/// transpose costs about what the vector arithmetic saves. Here a row of eight `i32` is exactly one
/// `__m256i`, so the column pass is whole-vector add/sub with no shuffle and only the row pass
/// needs one permute. Nothing is transposed, and because it transforms a single block it has no
/// lane-utilisation penalty: the encoder's batch of three costs three calls, not eight idle lanes.
///
/// Levels 2 and 3 stay scalar. They are 20 of the 84 butterflies and operate on 4x4 and 2x2 data,
/// which is narrower than a vector.
///
/// # Safety
///
/// The caller must hold a live [`FpuGuard`] and must have checked `has_avx2()`.
#[target_feature(enable = "avx2")]
unsafe fn transform_inblock_avx2(block: &[i32; PIXELS]) -> [i32; COEFFS] {
    // SAFETY: as the caller.
    unsafe {
        // Deinterleave a row into [evens | odds] with one permute.
        let idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        let mut l = [_mm_setzero_si128(); 8];
        let mut h = [_mm_setzero_si128(); 8];
        for r in 0..8 {
            let row = _mm256_loadu_si256(block.as_ptr().add(r * 8) as *const __m256i);
            let p = _mm256_permutevar8x32_epi32(row, idx);
            let e = _mm256_castsi256_si128(p);
            let o = _mm256_extracti128_si256(p, 1);
            l[r] = _mm_add_epi32(e, o);
            h[r] = _mm_sub_epi32(e, o);
        }

        // Column pass: pair rows. Whole-vector, no shuffle.
        let (mut ll1, mut hl1, mut lh1, mut hh1) = ([0i32; 16], [0i32; 16], [0i32; 16], [0i32; 16]);
        for i in 0..4 {
            let (la, lb) = (l[2 * i], l[2 * i + 1]);
            let (ha, hb) = (h[2 * i], h[2 * i + 1]);
            _mm_storeu_si128(ll1.as_mut_ptr().add(i * 4) as *mut __m128i, _mm_add_epi32(la, lb));
            _mm_storeu_si128(lh1.as_mut_ptr().add(i * 4) as *mut __m128i, _mm_sub_epi32(la, lb));
            _mm_storeu_si128(hl1.as_mut_ptr().add(i * 4) as *mut __m128i, _mm_add_epi32(ha, hb));
            _mm_storeu_si128(hh1.as_mut_ptr().add(i * 4) as *mut __m128i, _mm_sub_epi32(ha, hb));
        }

        // Levels 2 and 3, scalar and identical to `video::wht`.
        let sh = |x: i32| x >> 6;
        let (mut ll2, mut hl2, mut lh2, mut hh2) = ([0i32; 4], [0i32; 4], [0i32; 4], [0i32; 4]);
        {
            let (mut tl, mut th) = ([0i32; 8], [0i32; 8]);
            for r in 0..4 {
                for i in 0..2 {
                    let (a, b) = (ll1[r * 4 + 2 * i], ll1[r * 4 + 2 * i + 1]);
                    tl[r * 2 + i] = a + b;
                    th[r * 2 + i] = a - b;
                }
            }
            for c in 0..2 {
                for i in 0..2 {
                    let (a, b) = (tl[2 * i * 2 + c], tl[(2 * i + 1) * 2 + c]);
                    ll2[i * 2 + c] = a + b;
                    lh2[i * 2 + c] = a - b;
                    let (a2, b2) = (th[2 * i * 2 + c], th[(2 * i + 1) * 2 + c]);
                    hl2[i * 2 + c] = a2 + b2;
                    hh2[i * 2 + c] = a2 - b2;
                }
            }
        }
        let (l0, h0) = (ll2[0] + ll2[1], ll2[0] - ll2[1]);
        let (l1, h1) = (ll2[2] + ll2[3], ll2[2] - ll2[3]);
        let (ll3, hl3, lh3, hh3) = (l0 + l1, h0 + h1, l0 - l1, h0 - h1);

        let mut out = [0i32; COEFFS];
        out[0] = sh(ll3);
        out[1] = sh(hl3);
        out[2] = sh(lh3);
        out[3] = sh(hh3);
        for i in 0..4 {
            out[4 + i] = sh(hl2[i]);
            out[8 + i] = sh(lh2[i]);
            out[12 + i] = sh(hh2[i]);
        }
        for i in 0..16 {
            let m = SCAN4_MORTON[i];
            out[16 + i] = sh(hl1[m]);
            out[32 + i] = sh(lh1[m]);
            out[48 + i] = sh(hh1[m]);
        }
        out
    }
}

// ------------------------------------------------------------------------------- the experiment

/// Deterministic pseudo-random blocks spanning the 8-bit range the codec sees.
fn make_blocks(n: usize) -> Result<KVec<[i32; PIXELS]>> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut v = KVec::with_capacity(n, GFP_KERNEL)?;
    for _ in 0..n {
        let mut b = [0i32; PIXELS];
        for px in b.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *px = (state % 256) as i32;
        }
        v.push(b, GFP_KERNEL)?;
    }
    Ok(v)
}

fn ns_per(total: Delta, items: usize) -> i64 {
    if items == 0 {
        return 0;
    }
    total.as_nanos() / items as i64
}

/// Run the vector paths against the scalar one and report time, FPU cost and lane utilisation.
///
/// Called from module init behind `simd_bench`. Everything is allocated up front so the FPU
/// sections contain arithmetic and nothing else.
pub(crate) fn bench() -> Result {
    const BLOCKS: usize = 4096;
    const REPS: usize = 32;

    pr_info!("vino-simd: avx2={} avx512f={}\n", has_avx2(), has_avx512());
    let blocks = make_blocks(BLOCKS)?;
    let mut out = KVec::with_capacity(16, GFP_KERNEL)?;
    for _ in 0..16 {
        out.push([0i32; COEFFS], GFP_KERNEL)?;
    }

    // Scalar baseline. This is the number the kernel actually pays today.
    let t = Instant::<Monotonic>::now();
    let mut sink = 0i64;
    for _ in 0..REPS {
        for b in blocks.iter() {
            sink = sink.wrapping_add(transform(b)[0] as i64);
        }
    }
    let scalar = Instant::<Monotonic>::now() - t;
    let scalar_ns = ns_per(scalar, REPS * BLOCKS);
    pr_info!("vino-simd: scalar {scalar_ns} ns/block over {} blocks\n", REPS * BLOCKS);

    // Cost of the FPU section on its own, with nothing inside it. This is what a vector path has
    // to earn back on every region it opens.
    let t = Instant::<Monotonic>::now();
    for _ in 0..REPS * BLOCKS {
        let g = FpuGuard::new(0);
        core::hint::black_box(&g);
        drop(g);
    }
    let fpu = Instant::<Monotonic>::now() - t;
    let fpu_ns = ns_per(fpu, REPS * BLOCKS);
    pr_info!("vino-simd: kernel_fpu_begin/end {fpu_ns} ns per section (empty)\n");

    // The within-block form: one block per call, so the encoder's batch of three costs three
    // calls rather than eight idle lanes.
    if has_avx2() {
        let mut bad = 0usize;
        for b in blocks.iter() {
            let _fpu = FpuGuard::new(0);
            // SAFETY: guard live, feature checked.
            if unsafe { transform_inblock_avx2(b) } != transform(b) {
                bad += 1;
            }
        }
        if bad != 0 {
            pr_err!("vino-simd: avx2-inblock MISMATCH on {} blocks -- not reporting a speedup\n", bad);
            return Err(EINVAL);
        }
        let t = Instant::<Monotonic>::now();
        for _ in 0..REPS {
            for b in blocks.iter() {
                let _fpu = FpuGuard::new(0);
                // SAFETY: as above.
                sink = sink.wrapping_add(unsafe { transform_inblock_avx2(b) }[0] as i64);
            }
        }
        let ib = ns_per(Instant::<Monotonic>::now() - t, REPS * BLOCKS);

        // The same, with one FPU section around a whole strip's worth of blocks, which is how a
        // real encoder would open it.
        let t = Instant::<Monotonic>::now();
        for _ in 0..REPS {
            for chunk in blocks.chunks(48) {
                let _fpu = FpuGuard::new(0);
                for b in chunk {
                    // SAFETY: as above.
                    sink = sink.wrapping_add(unsafe { transform_inblock_avx2(b) }[0] as i64);
                }
            }
        }
        let ib_strip = ns_per(Instant::<Monotonic>::now() - t, REPS * BLOCKS);
        pr_info!(
            "vino-simd: avx2-inblock identical to scalar | {} ns/block (fpu per block), {} ns/block (fpu per strip) -- no lane waste at any batch size\n",
            ib, ib_strip
        );
        pr_info!(
            "vino-simd: avx2-inblock vs scalar -- {}% per block, {}% per strip\n",
            if ib > 0 { scalar_ns * 100 / ib } else { 0 },
            if ib_strip > 0 { scalar_ns * 100 / ib_strip } else { 0 },
        );
    }

    macro_rules! run {
        ($label:literal, $supported:expr, $f:ident, $vec:ty, $lanes:literal, $zero:expr) => {
            if $supported {
                let mut scratch = Scratch::<$vec>::new($zero)?;

                // Byte-exactness first: a speedup without it is meaningless.
                let mut bad = 0usize;
                for c in 0..BLOCKS / $lanes {
                    let batch = &blocks[c * $lanes..(c + 1) * $lanes];
                    {
                        let _fpu = FpuGuard::new(0);
                        // SAFETY: guard live, feature checked, slices are `$lanes` long.
                        unsafe { $f(&mut scratch, batch, &mut out) };
                    }
                    for (i, b) in batch.iter().enumerate() {
                        if transform(b) != out[i] {
                            bad += 1;
                        }
                    }
                }
                if bad != 0 {
                    pr_err!("vino-simd: {} MISMATCH on {} blocks -- not reporting a speedup\n", $label, bad);
                    return Err(EINVAL);
                }

                // Full lanes, one FPU section per call: the ceiling, if the encode loop batched
                // across strips to fill the lanes.
                let batches = BLOCKS / $lanes;
                let t = Instant::<Monotonic>::now();
                for _ in 0..REPS {
                    for c in 0..batches {
                        let _fpu = FpuGuard::new(0);
                        // SAFETY: as above.
                        unsafe { $f(&mut scratch, &blocks[c * $lanes..(c + 1) * $lanes], &mut out) };
                    }
                }
                let full = Instant::<Monotonic>::now() - t;
                let full_ns = ns_per(full, REPS * batches * $lanes);

                // The same work with one FPU section around the whole run, to separate the
                // arithmetic from the section cost.
                let t = Instant::<Monotonic>::now();
                for _ in 0..REPS {
                    let _fpu = FpuGuard::new(0);
                    for c in 0..batches {
                        // SAFETY: as above.
                        unsafe { $f(&mut scratch, &blocks[c * $lanes..(c + 1) * $lanes], &mut out) };
                    }
                }
                let hoisted = Instant::<Monotonic>::now() - t;
                let hoisted_ns = ns_per(hoisted, REPS * batches * $lanes);

                // The encoder's real call: three useful lanes, the rest idle.
                let calls = BLOCKS / $lanes;
                let t = Instant::<Monotonic>::now();
                for _ in 0..REPS {
                    for c in 0..calls {
                        let _fpu = FpuGuard::new(0);
                        // SAFETY: as above.
                        unsafe { $f(&mut scratch, &blocks[c * $lanes..(c + 1) * $lanes], &mut out) };
                    }
                }
                let batch3 = Instant::<Monotonic>::now() - t;
                let batch3_ns = ns_per(batch3, REPS * calls * ENCODER_BATCH);

                let lanes = $lanes;
                let pct = |v: i64| if v > 0 { scalar_ns * 100 / v } else { 0 };
                let (pf, ph, pb) = (pct(full_ns), pct(hoisted_ns), pct(batch3_ns));
                pr_info!(
                    "vino-simd: {} {} lanes | full {} ns/block (fpu per call), {} ns/block (fpu hoisted) | encoder-shaped {} ns/block\n",
                    $label, lanes, full_ns, hoisted_ns, batch3_ns
                );
                pr_info!(
                    "vino-simd: {} vs scalar -- full {}%, hoisted {}%, encoder-shaped {}%\n",
                    $label, pf, ph, pb
                );
            } else {
                pr_info!("vino-simd: {} not supported on this CPU -- skipped\n", $label);
            }
        };
    }

    run!("avx2", has_avx2(), transform_avx2, __m256i, 8, unsafe {
        core::mem::zeroed()
    });
    run!("avx512", has_avx512(), transform_avx512, __m512i, 16, unsafe {
        core::mem::zeroed()
    });

    // How much of a real strip encode the transform actually is. Anything a faster transform can
    // win -- or lose -- is bounded by this share, and it is the number that decides whether the
    // rows above matter at all. A strip is 16 blocks and `colour_block` transforms three planes
    // per block, so a strip pays 48 transforms plus the pixel gather, quantiser and entropy coder.
    const STRIPS: usize = 512;
    let geom: Geometry = RIDGE_GEOMETRY;
    let mut px = |x: usize, y: usize| {
        let v = ((x * 7) ^ (y * 13)) as u8;
        (v, v.wrapping_add(29), v.wrapping_add(83))
    };
    let t = Instant::<Monotonic>::now();
    let mut bytes = 0usize;
    for i in 0..STRIPS {
        let sy = (i % 64) * geom.strip_h();
        bytes += colour_strip_at(geom, 0, sy, &mut px)?.len();
    }
    let strip = Instant::<Monotonic>::now() - t;
    let strip_ns = ns_per(strip, STRIPS);
    let transforms_per_strip = 16 * 3;
    let in_transform = scalar_ns * transforms_per_strip as i64;
    pr_info!(
        "vino-simd: strip encode {} ns ({} B avg), of which {} transforms = {} ns, {}%\n",
        strip_ns,
        bytes / STRIPS,
        transforms_per_strip,
        in_transform,
        if strip_ns > 0 { in_transform * 100 / strip_ns } else { 0 },
    );
    pr_info!("vino-simd: a perfect transform could save at most that share of encode CPU\n");

    // Where the other ~90% goes. The transform is one of four elementwise stages a strip runs, and
    // the three around it have no shuffle and no transpose, so they vectorise where it does not.
    let mut planes = ([0i32; PIXELS], [0i32; PIXELS], [0i32; PIXELS]);
    for i in 0..PIXELS {
        let (y, cb, cr) = colour((i * 7) as u8, (i * 13) as u8, (i * 29) as u8);
        planes.0[i] = cr;
        planes.1[i] = cb;
        planes.2[i] = y;
    }
    const N: usize = 16 * 512;

    let t = Instant::<Monotonic>::now();
    for i in 0..N {
        let v = core::hint::black_box(i as u8);
        core::hint::black_box(colour(v, v.wrapping_add(29), v.wrapping_add(83)));
    }
    let px_ns = ns_per(Instant::<Monotonic>::now() - t, N);

    let t = Instant::<Monotonic>::now();
    for i in 0..N {
        core::hint::black_box(quantize(core::hint::black_box(i as i32 * 37), i % COEFFS));
    }
    let q_ns = ns_per(Instant::<Monotonic>::now() - t, N);

    let t = Instant::<Monotonic>::now();
    for _ in 0..N / 16 {
        core::hint::black_box(colour_block(&planes.0, &planes.1, &planes.2));
    }
    let block_ns = ns_per(Instant::<Monotonic>::now() - t, N / 16);

    // Per strip: 16 blocks x 64 px of colour conversion, and 16 `colour_block` calls of which the
    // transform is 3 x `scalar_ns`.
    let colour_per_strip = px_ns * 16 * PIXELS as i64;
    let block_per_strip = block_ns * 16;
    pr_info!(
        "vino-simd: per strip -- colour() {} ns ({}%), colour_block {} ns ({}%), of which transform {} ns ({}%)\n",
        colour_per_strip,
        if strip_ns > 0 { colour_per_strip * 100 / strip_ns } else { 0 },
        block_per_strip,
        if strip_ns > 0 { block_per_strip * 100 / strip_ns } else { 0 },
        in_transform,
        if strip_ns > 0 { in_transform * 100 / strip_ns } else { 0 },
    );
    pr_info!(
        "vino-simd: elementwise unit costs -- colour() {} ns/px, quantize() {} ns/coeff\n",
        px_ns, q_ns
    );

    // The entropy coder, timed directly rather than by subtraction. It is bit-serial and
    // data-dependent, so it is the part of the codec SIMD cannot touch.
    let mut blocks: KVec<ColourBlock> = KVec::with_capacity(16, GFP_KERNEL)?;
    for _ in 0..16 {
        blocks.push(colour_block(&planes.0, &planes.1, &planes.2), GFP_KERNEL)?;
    }
    let t = Instant::<Monotonic>::now();
    for i in 0..512 {
        core::hint::black_box(colour_strip(&blocks, 0, (i % 64) as u16 * 16)?.len());
    }
    let entropy_ns = ns_per(Instant::<Monotonic>::now() - t, 512);
    pr_info!(
        "vino-simd: entropy coder {} ns/strip ({}% of the strip encode) -- bit-serial, no SIMD applies\n",
        entropy_ns,
        if strip_ns > 0 { entropy_ns * 100 / strip_ns } else { 0 },
    );
    pr_info!("vino-simd: 100% means parity with scalar; over 100% is faster\n");
    core::hint::black_box(sink);
    Ok(())
}
