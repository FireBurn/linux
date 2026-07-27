// SPDX-License-Identifier: GPL-2.0

//! DisplayLink video encoders and EP08 framing.
//!
//! The live scanout path uses [`wht`], a clean-room implementation of DLM's full-colour
//! 8x8 Haar codec and 64x16 strip/record grammar. As of 2026-07-22, deterministic
//! full-spectrum RGB noise is byte-identical to DLM for all 3600/3600 strips of a
//! 2560x1440 frame, both offline and from the bytes emitted by the loaded kernel module.
//!
//! The older Raw/RLX mode-2 encoder below remains as a tested protocol implementation,
//! but is not selected by live scanout because no real mode-2 capture exists to establish
//! its outer wire contract.
#![allow(dead_code)] // Retained protocol helpers and offline/KUnit entry points.

use super::*;

pub(crate) const MAGIC_RAW16: u16 = 0x68af;
pub(crate) const MAGIC_RLE16: u16 = 0x69af;
/// Frame-init `0x40af` (`FUN_003330fc`: u32 `0xaf0440af` + u16 `0x0840`).
pub(crate) const FRAME_INIT: [u8; 6] = [0xaf, 0x40, 0x04, 0xaf, 0x40, 0x08];
/// Bare `0xa0af` sync (`FUN_00332a38`).
pub(crate) const SYNC: [u8; 2] = [0xaf, 0xa0];
/// Frame-end section->code table `DAT_005b7860`, indexed by `mode - 1`.
pub(crate) const SECTION_CODE: [u8; 7] = [0x01, 0x00, 0x03, 0x00, 0x05, 0x07, 0x07];
pub(crate) const MAX_BLOCK_PIXELS: usize = 256;

/// Per-run strategy: mode 0 raw-only, 1 RLE-only, 2 adaptive (sec 8.4).
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Raw = 0,
    Rle = 1,
    Adaptive = 2,
}

/// Pack 8-bit RGB into RGB565 (the XRGB framebuffer reduced for the
/// `0x68af`/`0x69af` path).
pub(crate) fn rgb565(r: u8, g: u8, b: u8) -> u16 {
    ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3)
}

/// 6-byte block header: magic LE, 24-bit coord BE, count u8 (256 -> 0).
fn block_header(out: &mut KVec<u8>, magic: u16, coord: u32, count: usize) -> Result {
    out.extend_from_slice(&magic.to_le_bytes(), GFP_KERNEL)?;
    out.push(((coord >> 16) & 0xff) as u8, GFP_KERNEL)?;
    out.push(((coord >> 8) & 0xff) as u8, GFP_KERNEL)?;
    out.push((coord & 0xff) as u8, GFP_KERNEL)?;
    out.push((count & 0xff) as u8, GFP_KERNEL)?;
    Ok(())
}

fn encode_raw_into(out: &mut KVec<u8>, coord: u32, pix: &[u16]) -> Result {
    block_header(out, MAGIC_RAW16, coord, pix.len())?;
    for &p in pix {
        out.extend_from_slice(&p.to_be_bytes(), GFP_KERNEL)?;
    }
    Ok(())
}

fn encode_rle_into(out: &mut KVec<u8>, coord: u32, pix: &[u16]) -> Result {
    block_header(out, MAGIC_RLE16, coord, pix.len())?;
    let mut i = 0;
    while i < pix.len() {
        let v = pix[i];
        let mut run = 1;
        while i + run < pix.len() && pix[i + run] == v && run < 255 {
            run += 1;
        }
        out.push(run as u8, GFP_KERNEL)?;
        out.extend_from_slice(&v.to_be_bytes(), GFP_KERNEL)?;
        i += run;
    }
    Ok(())
}

fn run_count(pix: &[u16]) -> usize {
    let mut c = 0;
    let mut i = 0;
    while i < pix.len() {
        let v = pix[i];
        let mut j = i + 1;
        while j < pix.len() && pix[j] == v {
            j += 1;
        }
        c += 1;
        i = j;
    }
    c
}

fn encode_run_into(out: &mut KVec<u8>, mode: Mode, coord: u32, pix: &[u16]) -> Result {
    match mode {
        Mode::Raw => encode_raw_into(out, coord, pix),
        Mode::Rle => encode_rle_into(out, coord, pix),
        Mode::Adaptive => {
            let l = pix.len();
            let c = run_count(pix);
            if 2 * l < 3 * c + 1 {
                encode_raw_into(out, coord, pix)
            } else {
                encode_rle_into(out, coord, pix)
            }
        }
    }
}

/// Mode-2 frame encoder holding the shadow (previous-frame) buffer.
pub(crate) struct Encoder {
    width: usize,
    height: usize,
    mode: Mode,
    // vmalloc-backed: a `width*height` u16 buffer is ~4 MiB at 1080p, far above the
    // contiguous-kmalloc order limit (the page allocator WARNs and fails on it).
    shadow: VVec<u16>,
}

impl Encoder {
    pub(crate) fn new(width: usize, height: usize, mode: Mode) -> Result<Self> {
        let shadow = VVec::from_elem(0u16, width * height, GFP_KERNEL)?;
        Ok(Self {
            width,
            height,
            mode,
            shadow,
        })
    }

    /// Encode `cur` (RGB565) into a mode-2 marker stream; updates the shadow.
    /// Change-detection is per row; changed runs chunk into <=256-px blocks.
    pub(crate) fn encode(&mut self, cur: &[u16]) -> Result<KVec<u8>> {
        let mut s = KVec::new();
        self.encode_into(cur, &mut s)?;
        Ok(s)
    }

    /// Like [`encode`](Self::encode) but appends the marker stream to a caller-owned
    /// `out` instead of allocating a fresh `KVec`. The hot scanout path
    /// ([`encode_and_send`](super::drm_sink::encode_and_send)) uses this to encode
    /// straight into a buffer that already reserves the EP08 transport header, so a
    /// frame costs one allocation with no separate framing copy.
    pub(crate) fn encode_into(&mut self, cur: &[u16], s: &mut KVec<u8>) -> Result {
        s.extend_from_slice(&FRAME_INIT, GFP_KERNEL)?;
        for y in 0..self.height {
            let row = y * self.width;
            let mut x = 0;
            while x < self.width {
                while x < self.width && cur[row + x] == self.shadow[row + x] {
                    x += 1;
                }
                if x >= self.width {
                    break;
                }
                let run_start = x;
                while x < self.width && cur[row + x] != self.shadow[row + x] {
                    x += 1;
                }
                let run_end = x;
                let mut p = run_start;
                while p < run_end {
                    let n = (run_end - p).min(MAX_BLOCK_PIXELS);
                    let coord = (((row + p) * 2) & 0xff_ffff) as u32;
                    encode_run_into(s, self.mode, coord, &cur[row + p..row + p + n])?;
                    p += n;
                }
            }
        }
        // `SECTION_CODE` (the decompile's `DAT_005b7860`) is indexed by the DL3 1-based
        // mode number minus 1; our 0-based `Mode` discriminant already equals that index
        // (Raw=0->0x01, Rle=1->0x00, Adaptive=2->0x03). The previous `saturating_sub(1)`
        // double-subtracted, collapsing Raw and Rle onto the same code.
        let code = SECTION_CODE[(self.mode as usize).min(SECTION_CODE.len() - 1)];
        s.extend_from_slice(&SYNC, GFP_KERNEL)?;
        s.extend_from_slice(&[0xaf, 0x20, 0x1f, code], GFP_KERNEL)?;
        s.extend_from_slice(&[0xaf, 0x20, 0xff, 0x00], GFP_KERNEL)?;
        s.extend_from_slice(&SYNC, GFP_KERNEL)?;
        // Commit the shadow ONLY after the whole frame has been emitted successfully.
        // Updating it incrementally (per changed run) left it half-updated if a later
        // `extend_from_slice` hit OOM, permanently desyncing every subsequent diff frame.
        // The encoder reads the pre-frame shadow throughout, so a single end-of-frame copy
        // is equivalent to the per-run writes on the success path. The loop already indexes
        // `cur` up to `width*height == shadow.len()`, so this slice is always in bounds.
        let n = self.shadow.len();
        self.shadow.copy_from_slice(&cur[..n]);
        Ok(())
    }
}

/// Vino (`0x2801`) Walsh-Hadamard codec -- DLM's live full-colour scanout format.
/// See `docs/WHT-CODEC.md` + `docs/VIDEO.md` +
/// `captures/codec-vlc-table-breakthrough-20260623.md`.
///
/// **Current verification (2026-07-22).** The implementation covers the complete 64-coefficient
/// transform, position-dependent luma/chroma quantizers, three-plane significance tree, DC DPCM,
/// AC run/magnitude stream, strip layouts, and DLM's outer image records. It reproduces DLM's
/// controlled chromatic ramps (11/11 in the retained cramp corpus) and a deterministic arbitrary
/// RGB-noise frame (3600/3600 strips and lengths) byte-for-byte. A usbmon capture of both live Vino
/// heads reproduces the same 3600/3600 result, proving the kernel input/encode/wire path rather than
/// only a userspace model. Historical partial-codec notes remain in `docs/WHT-CODEC.md` as provenance.
#[allow(dead_code)] // Live codec plus offline/KUnit helpers retained in the same module.
pub(crate) mod wht {
    use super::*;

    /// Transform block geometry (recovered + byte-exact-verified 2026-06-23, see
    /// `captures/codec-vlc-table-breakthrough-20260623.md`): an **8x8 pixel** block is the input
    /// (`DIM` x `DIM` = `PIXELS` luma samples); the wavelet emits all **64** coefficients
    /// (`COEFFS`). `BLOCK` aliases `PIXELS` for the `transform()` input length.
    pub(crate) const DIM: usize = 8;
    pub(crate) const PIXELS: usize = DIM * DIM;
    pub(crate) const COEFFS: usize = 64;
    pub(crate) const BLOCK: usize = PIXELS;

    /// Vino colour transform, in the codec's 64x fixed point: `Cb = 64(R-G)`,
    /// `Cr = 64(B-G)` (achromatic R=G=B -> Cb=Cr=0), and the reversible luma
    ///
    /// ```text
    ///     Y = 64*G + 64*((Cb_raw + Cr_raw) >> 2)   where Cb_raw=R-G, Cr_raw=B-G
    /// ```
    ///
    /// The `>> 2` is an arithmetic shift (floor toward -inf). This **replaces** the
    /// earlier `Y = 16R + 32G + 16B` form, which `validate-transform-encoderio.py`
    /// showed runs 16..48 HIGH for chromatic blocks (`16R+32G+16B = 64G + 16(Cb+Cr)`,
    /// i.e. the un-floored sum). The floored form reproduces DLM's transform DC for
    /// **every** measured colour -- the 6 saturated primaries/secondaries (incl. the
    /// signed green/cyan cases the floor must round toward -inf), grey, white and
    /// black (`scripts/validate-transform-encoderio.py`); achromatic input is
    /// unchanged (`64*G` since `Cb=Cr=0`).
    pub(crate) fn colour(r: u8, g: u8, b: u8) -> (i32, i32, i32) {
        let (r, g, b) = (r as i32, g as i32, b as i32);
        let (cb, cr) = (r - g, b - g);
        (64 * g + 64 * ((cb + cr) >> 2), 64 * cb, 64 * cr)
    }

    /// Per-coefficient `(step, bias)` quantization table, **derived 2026-06-23 from DLM's
    /// ground-truth `quant_leave` pre/post buffers** (captures/sig-library) -- the old
    /// controlled ramps first recovered positions 0..31; deterministic full-spectrum noise then
    /// exposed the retained LH/HH bands and distinguished every rounding rule through position 63.
    fn step_bias(i: usize) -> (i32, i32) {
        match i {
            0 => (16, 8),
            1 | 2 => (16, 8),
            3 => (32, 16),
            4..=11 => (4, 2),
            12..=15 => (8, 4),
            16..=47 => (2, 0),
            _ => (4, 2), // 48..=63
        }
    }

    /// Quantize coefficient `coeff` at position `i`: `sign(coeff) * floor((|coeff| + bias) / step)`
    /// (byte-exact vs DLM, 25/25 library strips). Clamped to the 12-bit signed long-token range.
    pub(crate) fn quantize(coeff: i32, i: usize) -> i32 {
        let (step, bias) = step_bias(i);
        // DLM rounds the coarse 0..15 bands half-up on the signed value (ties toward
        // +infinity), not half-away from zero.  The finest HL band (16..31) truncates
        // toward zero.  Deterministic full-spectrum noise exposed both distinctions;
        // the old ramp vectors happened to land on exact divisors.
        let q = if bias == 0 {
            let q = coeff.abs() / step;
            if coeff < 0 {
                -q
            } else {
                q
            }
        } else {
            (coeff + bias).div_euclid(step)
        };
        q.clamp(-2048, 2047)
    }

    /// One separable 2-D Haar step over the top-left `n`x`n` of `src` (row-major, `n` columns,
    /// `n` in {8,4,2}). The 1-D Haar butterfly is `lo = a + b`, `hi = a - b`; applied to rows then
    /// columns it splits the `n`x`n` block into four `(n/2)`x`(n/2)` subbands written to
    /// `ll`/`hl`/`lh`/`hh` (row-major, stride `n/2`). Unnormalized -- `transform()` floor-divides
    /// the final coefficients by 64. Mirrors `scripts/wht-transform.py` (verified byte-exact).
    ///
    /// **`#[inline(never)]` (2026-07-17, root cause of the `video.rs:294`/`encode_and_send_wht`
    /// kernel stack overflow -- see `project_stack_overflow_root_cause_found_20260717` memory):**
    /// LLVM was inlining this whole codec pipeline (`colour_frame_ep08` -> `colour_strip_blocks`
    /// -> `colour_block` -> `transform` -> `haar2d`, called from a loop over every strip in a
    /// frame) into a single function, summing every inlined callee's locals into ONE static stack
    /// frame instead of reusing space between sequential calls. `CONFIG_VMAP_STACK` +
    /// `CONFIG_SCHED_STACK_END_CHECK` caught it cleanly: `encode_and_send_wht`'s prologue alone
    /// allocated 0x3ed8 (16,088) bytes -- on a 16KB kernel stack, several frames deep inside a
    /// workqueue callback, that's a guaranteed overflow regardless of pixel content (explaining
    /// why a userspace repro with an 8MB stack never reproduced it). Forcing this and the other
    /// codec functions below to stay real, separate calls (each with its own small frame that is
    /// entered/freed per call, not accumulated) fixes this by construction -- no logic changed.
    #[inline(never)]
    fn haar2d(
        src: &[i32],
        n: usize,
        ll: &mut [i32],
        hl: &mut [i32],
        lh: &mut [i32],
        hh: &mut [i32],
    ) {
        let h = n / 2;
        // Diagnostic (2026-07-16, see `project_bringup_video_panic_20260716` memory): a live HW
        // run hit `index out of bounds: the len is 0 but the index is 0` at this function's
        // column-pass write below, but every call site in `transform()` was hand-verified to pass
        // correctly-sized fixed arrays (never empty) -- so the exact call/size that actually
        // failed was never pinned down. This makes any future recurrence self-diagnosing instead
        // of a bare index panic: report `n`, the derived `h`, and every buffer's ACTUAL length
        // against what this function requires, so the message alone says which call site (and
        // which buffer) was wrong. Unconditional (not `debug_assert!`) so it fires regardless of
        // build profile until the real cause is confirmed and this can be removed/downgraded.
        assert!(
            src.len() == n * n
                && ll.len() == h * h
                && hl.len() == h * h
                && lh.len() == h * h
                && hh.len() == h * h,
            "vino: haar2d(n={n}, h={h}) buffer size mismatch -- src.len()={} (want {}), \
             ll.len()={} hl.len()={} lh.len()={} hh.len()={} (want {} each)",
            src.len(),
            n * n,
            ll.len(),
            hl.len(),
            lh.len(),
            hh.len(),
            h * h
        );
        // Row pass: L = row-lo, H = row-hi (each n rows x h cols).
        let mut l = [0i32; PIXELS];
        let mut hb = [0i32; PIXELS];
        for r in 0..n {
            for i in 0..h {
                let (a, b) = (src[r * n + 2 * i], src[r * n + 2 * i + 1]);
                l[r * h + i] = a + b;
                hb[r * h + i] = a - b;
            }
        }
        // Column pass: LL/LH = col-lo/hi of L, HL/HH = col-lo/hi of H (each h x h).
        for c in 0..h {
            for i in 0..h {
                let (a, b) = (l[2 * i * h + c], l[(2 * i + 1) * h + c]);
                ll[i * h + c] = a + b;
                lh[i * h + c] = a - b;
                let (a2, b2) = (hb[2 * i * h + c], hb[(2 * i + 1) * h + c]);
                hl[i * h + c] = a2 + b2;
                hh[i * h + c] = a2 - b2;
            }
        }
    }

    /// DLM's video transform (`FUN_007a7b60`), reverse-engineered + **verified byte-exact**
    /// (2026-06-23 ramps; completed 2026-07-22 with deterministic full-spectrum noise): an **8x8
    /// 2-D Haar (Mallat) wavelet, floor-divided by 64**. The `block` is 8x8 luma (`Y` in the
    /// codec's x64 fixed point); the output is 64 coefficients in DLM's Mallat layout:
    /// `c[0]` = LL; `c[1..4]` = level-3 HL/LH/HH; `c[4..8]/[8..12]/[12..16]` = level-2 HL/LH/HH
    /// (2x2 row-major each); `c[16..32]`, `c[32..48]`, and `c[48..64]` are the level-1 HL, LH,
    /// and HH 4x4 bands. Each level-1 band uses the same 2x2 Morton scan. A uniform block yields
    /// `DC = mean`, all AC = 0.
    ///
    /// `#[inline(never)]`: see `haar2d`'s doc comment -- part of the kernel-stack-overflow fix.
    #[inline(never)]
    pub(crate) fn transform(block: &[i32; PIXELS]) -> [i32; COEFFS] {
        let sh = |x: i32| x >> 6; // arithmetic shift = floor division by 64 (matches DLM/`//64`)
        let mut c = [0i32; COEFFS];
        // Level 1: 8x8 -> three 4x4 detail bands. Smooth horizontal ramps only excite HL and left
        // its scan ambiguous; deterministic noise makes every position and all three bands unique.
        let (mut ll1, mut hl1, mut lh1, mut hh1) = ([0i32; 16], [0i32; 16], [0i32; 16], [0i32; 16]);
        haar2d(block, DIM, &mut ll1, &mut hl1, &mut lh1, &mut hh1);
        // DLM scans every 4x4 level-one band in 2x2 Morton order.  A horizontal
        // ramp only distinguished this from row-major and led to the earlier
        // (incorrect) transpose; full-spectrum deterministic noise makes all 16
        // positions unique and proves this permutation exactly.
        const SCAN4_MORTON: [usize; 16] = [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15];
        for p in 0..16 {
            c[16 + p] = sh(hl1[SCAN4_MORTON[p]]);
            c[32 + p] = sh(lh1[SCAN4_MORTON[p]]);
            c[48 + p] = sh(hh1[SCAN4_MORTON[p]]);
        }
        // Level 2: LL1 (4x4) -> 2x2 subbands; c[4..8]/[8..12]/[12..16].
        let (mut ll2, mut hl2, mut lh2, mut hh2) = ([0i32; 4], [0i32; 4], [0i32; 4], [0i32; 4]);
        haar2d(&ll1, 4, &mut ll2, &mut hl2, &mut lh2, &mut hh2);
        for i in 0..4 {
            c[4 + i] = sh(hl2[i]);
            c[8 + i] = sh(lh2[i]);
            c[12 + i] = sh(hh2[i]);
        }
        // Level 3: LL2 (2x2) -> 1x1 subbands; the DC c[0] and coarse c[1..4].
        let (mut ll3, mut hl3, mut lh3, mut hh3) = ([0i32; 1], [0i32; 1], [0i32; 1], [0i32; 1]);
        haar2d(&ll2, 2, &mut ll3, &mut hl3, &mut lh3, &mut hh3);
        c[0] = sh(ll3[0]);
        c[1] = sh(hl3[0]);
        c[2] = sh(lh3[0]);
        c[3] = sh(hh3[0]);
        c
    }

    // ====================================================================================
    // ★ 2026-06-23 (live HW): the REAL entropy code, recovered + byte-exact-verified.
    //
    // The earlier MSB-first 5-bit "short/long token" model (now removed) was REFUTED: a value-axis
    // amplitude sweep (`scripts/codec-sweep-plan.py`) showed its decoded tokens were invariant to
    // the coefficient VALUE (identical `L589` across a 128x AC range) -- it never matched the coder.
    // The dock's entropy coder (DLM leaf `0x5e68b0`) is a **memory-resident unary-prefix VLC,
    // written LSB-first**, dumped from DLM and reproduced byte-for-byte by [`Vlc`] + [`CODEBOOK`]
    // (`scripts/wht-block-codec.py` reproduces DLM's per-block output 5/5; see
    // `captures/codec-vlc-table-breakthrough-20260623.md`). A coefficient's magnitude category is
    // `bit_length(|coeff|)` (verified across the sweep), code = unary(c)+0-terminator+remainder.
    //
    // VERIFIED here (KUnit): the codebook, the LSB-first packing, the 1-bit final padding, and the
    // magnitude-category rule. NOT yet generalized (the open work): the coeff->token GRAMMAR for
    // arbitrary content -- DC DPCM, the 2-D scan (incl. the real horizontal/vertical asymmetry),
    // and block modes -- so this is the byte-exact OUTPUT stage, not yet a wired general encoder.
    // ====================================================================================

    /// The dumped Vino entropy VLC, indexed by symbol: `(code, nbits)`, emitted **LSB-first**.
    /// Symbol 0 = the 1-bit code `0` (zero / most common); symbol 31 = the all-ones escape prefix.
    pub(crate) const CODEBOOK: [(u32, u8); 32] = [
        (0, 1),
        (1, 3),
        (5, 3),
        (3, 5),
        (19, 5),
        (11, 5),
        (27, 5),
        (7, 7),
        (71, 7),
        (39, 7),
        (103, 7),
        (23, 7),
        (87, 7),
        (55, 7),
        (119, 7),
        (15, 8),
        (143, 8),
        (79, 8),
        (207, 8),
        (47, 8),
        (175, 8),
        (111, 8),
        (239, 8),
        (31, 8),
        (159, 8),
        (95, 8),
        (223, 8),
        (63, 8),
        (191, 8),
        (127, 8),
        (255, 9),
        (511, 9),
    ];

    /// The constant 13-byte per-block sync literal (emitted twice), recovered from the wire.
    pub(crate) const SYNC13: [u8; 13] = [
        0x7c, 0x93, 0x6f, 0xf2, 0x4d, 0xbe, 0xc9, 0x37, 0xf9, 0x26, 0xdf, 0xe4, 0x9b,
    ];

    /// LSB-first VLC bit packer matching the dock (final byte padded with **1-bits** -- a
    /// truncated all-ones code, as DLM emits).
    pub(crate) struct Vlc {
        out: KVec<u8>,
        acc: u32,
        nbits: u32,
    }

    impl Vlc {
        pub(crate) fn new() -> Self {
            Self {
                out: KVec::new(),
                acc: 0,
                nbits: 0,
            }
        }

        /// Append one bit (LSB-first within each byte).
        fn bit(&mut self, b: u32) -> Result {
            self.acc |= (b & 1) << self.nbits;
            self.nbits += 1;
            if self.nbits == 8 {
                self.out.push((self.acc & 0xff) as u8, GFP_KERNEL)?;
                self.acc = 0;
                self.nbits = 0;
            }
            Ok(())
        }

        /// Emit codebook `sym`'s code, least-significant bit first.
        pub(crate) fn symbol(&mut self, sym: usize) -> Result {
            let (code, n) = CODEBOOK[sym];
            for k in 0..n as u32 {
                self.bit(code >> k)?;
            }
            Ok(())
        }

        /// Emit one quantized coefficient as DLM's JPEG-SSSS-style magnitude code (LSB-first),
        /// verified byte-exact against DLM's per-coefficient wire bits (q-4/q-8/q-16 -- see
        /// `scripts/wht-strip-encoder.py`). A zero coefficient is the 1-bit symbol 0. A nonzero
        /// `q` emits the unary category `c = bit_length(|q|)` (c ones + a 0 terminator), then the
        /// `(c-1)`-bit magnitude offset `|q| - 2^(c-1)` (MSB-first within the field), then a sign
        /// bit (`0` = negative, the captured polarity). This is the retained early luma codebook
        /// helper; the live full-colour path uses [`Bits::esc`], including its recovered escapes.
        /// This helper rejects categories >= 9 instead of silently mixing the two grammars.
        pub(crate) fn coeff(&mut self, q: i32) -> Result {
            if q == 0 {
                return self.symbol(0);
            }
            let c = mag_category(q); // bit_length(|q|)
            if c >= 9 {
                return Err(kernel::error::code::EOVERFLOW); // escape long form -- open RE
            }
            for _ in 0..c {
                self.bit(1)?; // unary category
            }
            self.bit(0)?; // terminator
            let offset = q.unsigned_abs() - (1 << (c - 1));
            for i in (0..c - 1).rev() {
                self.bit(offset >> i)?; // (c-1)-bit magnitude offset, MSB-first
            }
            self.bit(if q < 0 { 0 } else { 1 }) // sign bit (0 = negative)
        }

        /// Flush, padding the final byte with 1-bits (matches the dock's truncated all-ones code).
        pub(crate) fn finish(mut self) -> Result<KVec<u8>> {
            if self.nbits > 0 {
                while self.nbits < 8 {
                    self.acc |= 1 << self.nbits;
                    self.nbits += 1;
                }
                self.out.push((self.acc & 0xff) as u8, GFP_KERNEL)?;
            }
            Ok(self.out)
        }
    }

    /// Magnitude category of a quantized coefficient: `bit_length(|coeff|)` (verified 2026-06-23 --
    /// e.g. |4|->3, |8|->4, |255|->8). 0 for a zero coefficient.
    pub(crate) fn mag_category(coeff: i32) -> u32 {
        coeff.unsigned_abs().checked_ilog2().map_or(0, |l| l + 1)
    }

    // ====================================================================================
    // ★ 2026-06-23: the SOLID-colour strip encoder -- byte-exact-verified end to end vs DLM
    // (3508/3508 strips of grey128.bin + white). A solid 64x16 strip (16 uniform 8x8 blocks) is the
    // most common desktop case (backgrounds, flat UI). DLM codes it with a NO-sync framing distinct
    // from the AC-stripe path: a 16-byte header, a CONSTANT 30-byte "main frame" (identical for any
    // solid colour), the strip's absolute DC ESCAPE-coded (long form), then a fixed trailer. The DC
    // rule was cracked offline (`scripts/wht-strip-encoder.py`): code = unary(c) ++ offset(c-1,
    // MSB-first) ++ sign, c=bit_length(qDC), qDC=quantize(DC,0), DC=c3[0]=64*gray (achromatic).
    // ====================================================================================

    /// Bit offset where the per-plane DC escape begins (after the 16-byte header + the constant
    /// 30-byte main frame = byte 46). Verified 2026-06-23 across the grey sweep and solid primaries.
    const SOLID_DC_BIT: usize = 368;
    /// Maximum DC escape category; the maximum category omits the unary 0-terminator (a complete
    /// prefix code on categories `1..=SOLID_DC_CMAX`). `|qY| <= 1020 => c <= 10`, `|qCb| <= 255`.
    const SOLID_DC_CMAX: u32 = 10;
    /// Minimum trailing bits after the escape before the 2-byte length tail (empirical: fits all 14
    /// solid/colour DCs of `codec-grammar-20260623`; ~= the 16 blocks' empty-AC/EOB run).
    const SOLID_DC_MINPAD: usize = 61;
    /// Header (16 B, X/Y/w18/w1c patched per strip) + the constant 30-byte main frame (bytes 16..46)
    /// of a solid strip (the 16-block all-zero DC-residual / empty-AC structure, identical for any
    /// solid colour). Byte 46 (bit 368) onward carries the (Cr, Cb, Y) DC escape + trailer.
    const SOLID_MAIN: [u8; 46] = [
        0x01, 0x28, 0, 0, 0, 0, 0, 0, 0, 0, 0x3a, 0, 0x3a, 0, 0,
        0, // header (magic,X,Y,resv,w18,w1c,z)
        0xfc, 0x00, 0x7e, 0x00, 0x3f, 0x80, 0x1f, 0xc0, 0x0f, 0xe0, 0x07, 0xf0, 0x03, 0xf8, 0x01,
        0xfc, 0x00, 0x7e, 0x00, 0x3f, 0x80, 0x1f, 0xc0, 0x0f, 0xe0, 0x07, 0xf0, 0x03, 0xf8,
        0x01, // main frame
    ];

    /// Max AC magnitude category (the maximum category omits the unary 0-terminator).
    const AC_CMAX: u32 = 9;

    /// LSB-first bit accumulator for the AC-strip coder (no final padding, unlike [`Vlc`]).
    struct Bits {
        out: KVec<u8>,
        n: usize,
    }

    impl Bits {
        fn new() -> Self {
            Self {
                out: KVec::new(),
                n: 0,
            }
        }

        fn bit(&mut self, b: u32) -> Result {
            if self.n % 8 == 0 {
                self.out.push(0, GFP_KERNEL)?;
            }
            self.out[self.n / 8] |= ((b & 1) as u8) << (self.n % 8);
            self.n += 1;
            Ok(())
        }

        /// The shared escape value code: a 0 is one `0` bit; else `unary(c) ++ [0-term IFF c<cmax]
        /// ++ offset(c-1, MSB-first) ++ sign(1=positive)`. `c = bit_length(|v|)`.
        fn esc(&mut self, v: i32, cmax: u32) -> Result {
            if v == 0 {
                return self.bit(0);
            }
            // Saturate a magnitude whose category exceeds the codebook maximum (`cmax`) to the
            // largest value that category `cmax` encodes. This keeps the unary prefix at most
            // `cmax` ones: a decoder that stops after `cmax` ones (the max-category escape, whose
            // 0-terminator is omitted) would otherwise read the (cmax+1)th one as an offset bit
            // and desync the rest of the strip. In-range coefficients (`c <= cmax`) are
            // unaffected; this only bounds the out-of-range case, which the recovered grammar does
            // not otherwise exercise.
            let c = mag_category(v).min(cmax);
            let off = v.unsigned_abs().min((1 << c) - 1) - (1 << (c - 1));
            for _ in 0..c {
                self.bit(1)?;
            }
            if c < cmax {
                self.bit(0)?;
            }
            for i in (0..c.saturating_sub(1)).rev() {
                self.bit(off >> i)?;
            }
            self.bit(u32::from(v > 0))
        }

        /// Per-block luma significance code recovered across every last position in 1..63.
        /// For `k=floor(log2(64-last))`, emit `00`, `k` one bits, `0`, then the `k`-bit
        /// MSB-first value `(64-2^k)-last`. A flat block retains its separate 15-bit code.
        fn sync_unit_after(&mut self, last: usize, mut skip: usize) -> Result {
            fn emit(bits: &mut Bits, value: u32, skip: &mut usize) -> Result {
                if *skip != 0 {
                    *skip -= 1;
                    Ok(())
                } else {
                    bits.bit(value)
                }
            }
            if last == 0 {
                for &bit in &[0u32, 0, 1, 1, 1, 1, 1, 1] {
                    emit(self, bit, &mut skip)?;
                }
                for _ in 0..7 {
                    emit(self, 0, &mut skip)?;
                }
            } else {
                debug_assert!(last < COEFFS);
                emit(self, 0, &mut skip)?;
                emit(self, 0, &mut skip)?;
                let remaining = COEFFS - last;
                let k = usize::BITS - 1 - remaining.leading_zeros();
                for _ in 0..k {
                    emit(self, 1, &mut skip)?;
                }
                emit(self, 0, &mut skip)?;
                let end = COEFFS - (1usize << k);
                let v = (end - last) as u32;
                for i in (0..k).rev() {
                    emit(self, (v >> i) & 1, &mut skip)?;
                }
            }
            debug_assert_eq!(skip, 0);
            Ok(())
        }

        fn sync_unit(&mut self, last: usize) -> Result {
            self.sync_unit_after(last, 0)
        }
    }

    /// Number of blocks across a 64-px-wide strip and the rows of blocks within its 16-px height.
    const STRIP_BLOCKS_X: usize = 8;
    const STRIP_ROW_BLOCKS: usize = 8; // blocks in one 8-row half (8 across x 1 down)
    const STRIP_BLOCKS: usize = 16; // 8 across x 2 down

    /// Round a byte count up to an even number (every coder sub-region is even-aligned).
    fn round_even(n: usize) -> usize {
        n + (n & 1)
    }

    /// Emit the 16-block DPCM DC plane into `b`: per block, the `(Cr, Cb, Y)` quantized DC as a
    /// RESIDUAL escape (residual = this block's DC minus the previous block's DC, previous = 0 for
    /// block 0). Byte-exact vs DLM on the varying-DC solid strips (vstripe8/hstripe8/checker8) and
    /// the uniform case (all residuals 0). `dc[k] = (qCr, qCb, qY)`.
    fn dc_plane(b: &mut Bits, dc: &[(i32, i32, i32); STRIP_BLOCKS]) -> Result {
        let (mut pcr, mut pcb, mut py) = (0i32, 0i32, 0i32);
        for &(cr, cb, y) in dc {
            b.esc(cr - pcr, SOLID_DC_CMAX)?;
            b.esc(cb - pcb, SOLID_DC_CMAX)?;
            b.esc(y - py, SOLID_DC_CMAX)?;
            (pcr, pcb, py) = (cr, cb, y);
        }
        Ok(())
    }

    /// Emit one block's luma AC coefficients (positions `1..=last`) into `b`: a run bit `0` for an
    /// insignificant coefficient, else the magnitude escape. No EOB -- the block's `last` (from the
    /// significance sync) bounds the loop. This helper is intentionally achromatic; the live
    /// three-plane counterpart is [`Bits::colour_block_ac`].
    fn block_ac(b: &mut Bits, q: &[i32; COEFFS], last: usize) -> Result {
        for i in 1..=last {
            if q[i] == 0 {
                b.bit(0)?;
            } else {
                b.esc(q[i], AC_CMAX)?;
            }
        }
        Ok(())
    }

    // ====================================================================================
    // COLOUR strip codec (Cb/Cr planes). Recovered byte-exact 2026-06-27/28 from DLM
    // sink-hook captures (cramp/rramp/bramp period sweeps, captures/codec-sink-sweep-*):
    // 2700/2872 strips byte-identical across every significance combination. Tooling +
    // proof: scripts/codec-re/{coeffs,model,colourstrip,verify-colour-ac}.py.
    //
    // Per block the 3 planes are (Cr=64*(B-G), Cb=64*(R-G), Y=64*G + 64*((Cb+Cr)>>2)).
    //  * SYNC unit = [Cr field][Cb field][Y field]; chroma fields present only when last>0
    //    (the per-block plane mask), Y field always present (luma `sync_unit`).
    //  * DC plane = 16-block DPCM (Cr,Cb,Y), 3 tokens/block, chroma step 64 / luma step 16,
    //    round-half-up on the signed value.
    //  * AC rows (row0 blocks 0..8, row1 8..16): per block (Cr,Cb,Y) present planes, chroma
    //    quant flat step 16 (truncate toward zero), positions 1..last, run-bit `0` for zeros.
    //  * Strip length = w1c + round_even(row1) (the 2-byte tail overlaps row1's tail).
    // ====================================================================================

    /// Chroma AC quantizer recovered from deterministic full-spectrum DLM output.
    /// Coarse bands 1/2 and 4..11 use step 16; coefficient 3 and positions 12..47 use
    /// step 32; the final HH band (48..63) uses step 64. All use signed half-up rounding. The former flat
    /// step-16/truncate model was indistinguishable on the smooth horizontal ramps,
    /// but disagreed on essentially every textured colour block.
    fn quantize_chroma_ac(coeff: i32, i: usize) -> i32 {
        let step = if matches!(i, 1 | 2 | 4..=11) {
            16
        } else if i >= 48 {
            64
        } else {
            32
        };
        (coeff + step / 2).div_euclid(step)
    }

    /// Per-plane DC quantizer, round-half-up on the SIGNED value (toward +inf): luma (plane 0)
    /// step 16, chroma step 64. `+224/64 = 3.5 -> 4`; `-8416/64 = -131.5 -> -131`.
    fn quantize_dc_round(plane: usize, v: i32) -> i32 {
        let step = if plane == 0 { 16 } else { 64 };
        (v + step / 2).div_euclid(step)
    }

    impl Bits {
        /// Exact chroma last-position tree node. For
        /// `c=floor(log2(last+1))`, emit `1`x`c`, `0`, then the `c`-bit MSB-first
        /// offset `last-(2^c-1)`. At category endpoints this composes to the older
        /// `1`x`c` ++ `0`x`c` field; arbitrary noise reveals the offset bits.
        fn chroma_base(&mut self, last: usize) -> Result {
            debug_assert!(last > 0 && last < COEFFS);
            let c = usize::BITS - 1 - (last + 1).leading_zeros();
            for _ in 0..c {
                self.bit(1)?;
            }
            self.bit(0)?;
            let offset = (last - ((1usize << c) - 1)) as u32;
            for bit in (0..c).rev() {
                self.bit((offset >> bit) & 1)?;
            }
            Ok(())
        }

        /// One block's three-plane significance tree. The luma code begins with two
        /// zero root branches. A present Cr replaces the first with its chroma node;
        /// a present Cb replaces the second. This explains why the old category-end
        /// fixtures looked like independently concatenated plane fields while exact
        /// non-boundary last positions did not.
        fn colour_sync_unit(&mut self, lcr: usize, lcb: usize, ly: usize) -> Result {
            match (lcr != 0, lcb != 0) {
                (false, false) => self.sync_unit(ly),
                (true, false) => {
                    self.chroma_base(lcr)?;
                    self.sync_unit_after(ly, 1)
                }
                (false, true) => {
                    self.bit(0)?;
                    self.chroma_base(lcb)?;
                    self.sync_unit_after(ly, 2)
                }
                (true, true) => {
                    self.chroma_base(lcr)?;
                    self.chroma_base(lcb)?;
                    self.sync_unit_after(ly, 2)
                }
            }
        }

        /// One block's colour AC: present planes in (Cr, Cb, Y) order, positions `1..=last`,
        /// run-bit `0` for an insignificant coefficient else the magnitude escape (cmax `AC_CMAX`).
        fn colour_block_ac(
            &mut self,
            qcr: &[i32; COEFFS],
            qcb: &[i32; COEFFS],
            qy: &[i32; COEFFS],
            lcr: usize,
            lcb: usize,
            ly: usize,
        ) -> Result {
            for &(q, last) in &[(qcr, lcr), (qcb, lcb), (qy, ly)] {
                for i in 1..=last {
                    if q[i] == 0 {
                        self.bit(0)?;
                    } else {
                        self.esc(q[i], AC_CMAX)?;
                    }
                }
            }
            Ok(())
        }
    }

    /// One quantized colour block: the three planes' 64 coefficients and exact last-significant
    /// AC positions. Built by [`colour_block`] from a block's per-plane samples.
    pub(crate) struct ColourBlock {
        qcr: [i32; COEFFS],
        qcb: [i32; COEFFS],
        qy: [i32; COEFFS],
        lcr: usize,
        lcb: usize,
        ly: usize,
    }

    /// Return the exact chroma AC extent. Kept as a named helper for KUnit and the userspace
    /// chimera: deterministic noise proved that the significance tree carries the offset within
    /// each power-of-two category, rather than only the category endpoint.
    pub(crate) fn chroma_last(q: &[i32; COEFFS]) -> usize {
        (1..COEFFS).rev().find(|&i| q[i] != 0).unwrap_or(0)
    }

    /// Transform + quantize one block's three planes (each 64 samples in the codec's x64 fixed
    /// point: `cr[i] = 64*(B-G)`, `cb[i] = 64*(R-G)`, `y[i] = 64*G + 64*((Cb+Cr)>>2)`). Luma uses
    /// the per-position `quantize`; chroma AC uses its recovered per-band
    /// `quantize_chroma_ac`; all DCs use `quantize_dc_round`.
    ///
    /// `#[inline(never)]`: see `haar2d`'s doc comment -- part of the kernel-stack-overflow fix.
    #[inline(never)]
    pub(crate) fn colour_block(
        cr: &[i32; PIXELS],
        cb: &[i32; PIXELS],
        y: &[i32; PIXELS],
    ) -> ColourBlock {
        let tcr = transform(cr);
        let tcb = transform(cb);
        let ty = transform(y);
        let mut qcr = [0i32; COEFFS];
        let mut qcb = [0i32; COEFFS];
        let mut qy = [0i32; COEFFS];
        qcr[0] = quantize_dc_round(2, tcr[0]);
        qcb[0] = quantize_dc_round(1, tcb[0]);
        qy[0] = quantize_dc_round(0, ty[0]);
        for i in 1..COEFFS {
            qcr[i] = quantize_chroma_ac(tcr[i], i);
            qcb[i] = quantize_chroma_ac(tcb[i], i);
            qy[i] = quantize(ty[i], i);
        }
        let last = |q: &[i32; COEFFS]| (1..COEFFS).rev().find(|&i| q[i] != 0).unwrap_or(0);
        let (lcr, lcb, ly) = (chroma_last(&qcr), chroma_last(&qcb), last(&qy));
        ColourBlock {
            qcr,
            qcb,
            qy,
            lcr,
            lcb,
            ly,
        }
    }

    /// Encode one 64x16 COLOUR strip at pixel `(x, y)` from its 16 quantized colour blocks
    /// (raster: 0..8 top 8-px half, 8..16 bottom). Byte-exact vs DLM on all measured chromatic
    /// content (see the section header). The 2-byte tail is this strip's own `L-2`;
    /// [`encode_frame`]/the scanout path overwrites it with the next strip's forward length hint.
    ///
    /// `#[inline(never)]`: see `haar2d`'s doc comment -- part of the kernel-stack-overflow fix.
    #[inline(never)]
    pub(crate) fn colour_strip(blocks: &[ColourBlock], x: u16, y: u16) -> Result<KVec<u8>> {
        let mut main = Bits::new();
        for b in blocks {
            main.colour_sync_unit(b.lcr, b.lcb, b.ly)?;
        }
        let (mut pcr, mut pcb, mut py) = (0i32, 0i32, 0i32);
        for b in blocks {
            let (cr, cb, yv) = (b.qcr[0], b.qcb[0], b.qy[0]);
            main.esc(cr - pcr, SOLID_DC_CMAX)?;
            main.esc(cb - pcb, SOLID_DC_CMAX)?;
            main.esc(yv - py, SOLID_DC_CMAX)?;
            (pcr, pcb, py) = (cr, cb, yv);
        }
        let mut row0 = Bits::new();
        for b in &blocks[..STRIP_ROW_BLOCKS] {
            row0.colour_block_ac(&b.qcr, &b.qcb, &b.qy, b.lcr, b.lcb, b.ly)?;
        }
        let mut row1 = Bits::new();
        for b in &blocks[STRIP_ROW_BLOCKS..] {
            row1.colour_block_ac(&b.qcr, &b.qcb, &b.qy, b.lcr, b.lcb, b.ly)?;
        }

        let r0 = round_even(row0.out.len());
        let r1 = round_even(row1.out.len());
        let main_b = round_even(main.out.len()) + 2;
        let w18 = 16 + main_b;
        let w1c = w18 + r0;
        // The 2-byte tail overlaps the end of the row1 region (len = w1c + round_even(row1)).
        let len = w1c + r1;

        let mut out = KVec::new();
        out.resize(len, 0, GFP_KERNEL)?;
        out[0] = 0x01;
        out[1] = 0x28;
        out[2..4].copy_from_slice(&x.to_le_bytes());
        // Strip-y = the band's TOP edge. (An earlier "bottom edge = y+STRIP_H" reading from ONE
        // cold@120 capture was refuted 2026-07-20 by a cramp256 capture whose first band is y=0 with
        // top row 0 -- DLM's y just follows the actual painted band, which differed between the two
        // captures. `validate-frame`'s record 0 matched with y=top once the record framing was fixed.)
        out[4..6].copy_from_slice(&y.to_le_bytes());
        out[10..12].copy_from_slice(&(w18 as u16).to_le_bytes());
        out[12..14].copy_from_slice(&(w1c as u16).to_le_bytes());
        out[16..16 + main.out.len()].copy_from_slice(&main.out);
        out[w18..w18 + row0.out.len()].copy_from_slice(&row0.out);
        out[w1c..w1c + row1.out.len()].copy_from_slice(&row1.out);
        // No forward-hint tail: on the EP08 wire the strip's last 2 bytes are the natural row1
        // bit-packing. The record framing carries the length as `strip_id == len`, so the in-strip
        // echo the sink hook showed is not transmitted on the wire. See `frame_records`.
        Ok(out)
    }

    /// Strip pixel geometry: a strip is 64 px wide and 16 px tall
    /// (`STRIP_BLOCKS_X * DIM` x `2 * DIM`).
    pub(crate) const STRIP_W: usize = STRIP_BLOCKS_X * DIM; // 64
    pub(crate) const STRIP_H: usize = 2 * DIM; // 16

    /// DLM's damage MACRO-TILE: 256 px wide x 64 px tall = 4x4 = 16 strips. Reverse-engineered
    /// 2026-07-25 (`docs/DLM-DAMAGE-TILING.md`, `scripts/dlm-damage-{probe,atomic}.c`): DLM quantises
    /// ALL damage to this `(256,64)`-aligned grid and always re-sends every strip of a touched
    /// macro-tile -- never a sub-macro-tile partial. A single-strip clip on the atomic
    /// `FB_DAMAGE_CLIPS` path still produced the full 16-strip macro-tile. Sending fewer (per-strip)
    /// leaves stale strips on the dock -> the on-screen torn "bad updates". So damage selection snaps
    /// to this grid.
    pub(crate) const MACRO_W: usize = 4 * STRIP_W; // 256
    pub(crate) const MACRO_H: usize = 4 * STRIP_H; // 64

    /// Gather one 64x16 strip's 16 colour blocks from a pixel source. `px(x, y)` returns the
    /// 8-bit `(R, G, B)` at absolute frame coordinate `(x, y)`; `(ox, oy)` is the strip's
    /// top-left pixel. Each block's three planes are built in the codec's x64 fixed point via
    /// [`colour`] (per-pixel `(Y, Cb, Cr)`, stored `(Cr, Cb, Y)` for [`colour_block`]). Blocks are
    /// raster order within the strip (0..8 top 8-px half, 8..16 bottom), matching [`colour_strip`].
    ///
    /// `#[inline(never)]`: see `haar2d`'s doc comment -- part of the kernel-stack-overflow fix.
    /// Returns a **heap-allocated** `KVec` rather than `[ColourBlock; STRIP_BLOCKS]` by value
    /// (~6.5KB): a live crash (`BUG: TASK stack guard page was hit`, `CONFIG_VMAP_STACK`/
    /// `CONFIG_SCHED_STACK_END_CHECK`) showed that a large by-value array return does not
    /// reliably get constructed directly into the caller's slot (no guaranteed RVO in Rust) --
    /// `colour_frame_ep08`'s own frame held a second ~6.5KB copy of the same data on top of this
    /// function's, and being nested calls (not siblings), those frames coexist on the stack
    /// simultaneously rather than reusing space. Heap-allocating this one, genuinely large,
    /// per-strip buffer removes it from the stack entirely.
    #[inline(never)]
    fn colour_strip_blocks(
        ox: usize,
        oy: usize,
        px: &mut impl FnMut(usize, usize) -> (u8, u8, u8),
    ) -> Result<KVec<ColourBlock>> {
        let mut blocks = KVec::with_capacity(STRIP_BLOCKS, GFP_KERNEL)?;
        for k in 0..STRIP_BLOCKS {
            let (bx, by) = (k % STRIP_BLOCKS_X, k / STRIP_BLOCKS_X);
            let (mut cr, mut cb, mut y) = ([0i32; PIXELS], [0i32; PIXELS], [0i32; PIXELS]);
            for r in 0..DIM {
                for c in 0..DIM {
                    let (rr, gg, bb) = px(ox + bx * DIM + c, oy + by * DIM + r);
                    let (yv, cbv, crv) = colour(rr, gg, bb);
                    let i = r * DIM + c;
                    (cr[i], cb[i], y[i]) = (crv, cbv, yv);
                }
            }
            blocks.push(colour_block(&cr, &cb, &y), GFP_KERNEL)?;
        }
        Ok(blocks)
    }

    /// Encode a full `width`x`height` 8-bit-RGB frame into the Vino WHT **colour** EP08 transfer(s)
    /// -- the colour counterpart of the luma [`encode_frame`] and the live scanout assembler.
    /// `px(x, y)` yields the source pixel's `(R, G, B)`; the
    /// caller applies any rotation / gamma / format conversion (so this stays a pure codec). The
    /// surface is tiled into 64x16 strips in raster order, each built from [`colour_block`] +
    /// [`colour_strip`], and the strip stream is framed for the wire by [`frame_records`] using
    /// DLM's real EP08 TLV record layout (one record per single-Y band, each strip prefixed with
    /// its `strip_id == len`), then chunked into `<= 65536`-byte USB transfers. There is **no**
    /// per-transfer [`write_ep08_header`] (that older framing made the dock fault -- see
    /// [`frame_records`]). `seq0` is the logical frame number used by [`frame_trailer`] and the
    /// returned value is advanced for the next frame.
    ///
    /// `width`/`height` must be multiples of 64 and 16 (`EINVAL` otherwise). Live scanout pads a
    /// non-aligned mode with black to this strip grid while preserving the real mode dimensions.
    /// The complete colour grammar is byte-exact against deterministic full-spectrum DLM output.
    ///
    /// `#[inline(never)]`: see `haar2d`'s doc comment -- part of the kernel-stack-overflow fix.
    #[inline(never)]
    pub(crate) fn colour_frame_ep08(
        width: usize,
        height: usize,
        seq0: u32,
        head: u8,
        mut px: impl FnMut(usize, usize) -> (u8, u8, u8),
    ) -> Result<(KVec<KVec<u8>>, u32)> {
        if width % STRIP_W != 0 || height % STRIP_H != 0 {
            return Err(kernel::error::code::EINVAL);
        }
        // Build every strip body (raster order; each strip's natural row1 tail, no echo).
        let mut strips: KVec<KVec<u8>> = KVec::new();
        let mut sy = 0usize;
        while sy < height {
            let mut sx = 0usize;
            while sx < width {
                let blocks = colour_strip_blocks(sx, sy, &mut px)?;
                strips.push(colour_strip(&blocks, sx as u16, sy as u16)?, GFP_KERNEL)?;
                sx += STRIP_W;
            }
            sy += STRIP_H;
        }
        Ok((frame_records(&strips, head)?, seq0.wrapping_add(1)))
    }

    /// Build a valid all-black WHT frame without sampling or transforming a framebuffer.
    ///
    /// This is the prompt post-mode-set training carrier. A real 1440p compositor keyframe can
    /// take hundreds of milliseconds to hash and encode, but the D6000's lighting capture starts
    /// video only ~110 ms after the mode-set, while its stream-enable bracket is still active.
    /// Every black strip uses the live full-colour [`colour_strip`] grammar, but all sixteen zero
    /// coefficient blocks are constructed once and reused. Constructing this frame is therefore
    /// proportional to the 64x16 strip grid rather than to every source pixel and reproduces the
    /// captured 203,040-byte 1440p black image. The caller still sends the real framebuffer as a
    /// full keyframe afterwards; this frame exists only to deliver ARM plus continuous,
    /// correctly-framed video inside the dock's activation window.
    pub(crate) fn black_frame_ep08(
        width: usize,
        height: usize,
        head: u8,
    ) -> Result<KVec<KVec<u8>>> {
        if width % STRIP_W != 0 || height % STRIP_H != 0 {
            return Err(kernel::error::code::EINVAL);
        }
        let mut blocks: KVec<ColourBlock> = KVec::with_capacity(STRIP_BLOCKS, GFP_KERNEL)?;
        for _ in 0..STRIP_BLOCKS {
            blocks.push(
                ColourBlock {
                    qcr: [0; COEFFS],
                    qcb: [0; COEFFS],
                    qy: [0; COEFFS],
                    lcr: 0,
                    lcb: 0,
                    ly: 0,
                },
                GFP_KERNEL,
            )?;
        }
        let mut strips: KVec<KVec<u8>> = KVec::new();
        let mut sy = 0usize;
        while sy < height {
            let mut sx = 0usize;
            while sx < width {
                strips.push(colour_strip(&blocks, sx as u16, sy as u16)?, GFP_KERNEL)?;
                sx += STRIP_W;
            }
            sy += STRIP_H;
        }
        frame_records(&strips, head)
    }

    /// Encode ONE 64x16 strip whose top-left output pixel is `(sx, sy)`.
    ///
    /// This is the unit of work both frame encoders are built from, and it is the reason a
    /// parallel encode is possible at all: a strip reads only its own 64x16 region through `px`
    /// and produces its own independent byte vector, sharing no state with any other strip. The
    /// scanout encoder in `drm_sink.rs` fans batches of these across CPUs; see `EncodeChunk`.
    pub(crate) fn colour_strip_at(
        sx: usize,
        sy: usize,
        px: &mut impl FnMut(usize, usize) -> (u8, u8, u8),
    ) -> Result<KVec<u8>> {
        let blocks = colour_strip_blocks(sx, sy, px)?;
        colour_strip(&blocks, sx as u16, sy as u16)
    }

    /// The raster-ordered top-left coordinate of every strip a damage set selects.
    ///
    /// Split out of [`colour_frame_ep08_damage`] so the serial and parallel encoders select
    /// exactly the same strips in exactly the same order. **The order is load-bearing:**
    /// [`frame_records`] groups strips into one record per single-Y band and requires them
    /// x-ordered within each band, so reordering here changes the wire format.
    ///
    /// A strip is selected iff a clip overlaps the 256x64 MACRO-TILE containing it (DLM's damage
    /// granularity -- see `MACRO_W`/`MACRO_H` and `docs/DLM-DAMAGE-TILING.md`). DLM re-sends every
    /// strip of a touched macro-tile; matching that avoids the stale-strip torn updates a
    /// per-strip selection produced.
    pub(crate) fn damage_strip_coords(
        width: usize,
        height: usize,
        clips: &[(usize, usize, usize, usize)],
    ) -> Result<KVec<(usize, usize)>> {
        let mut coords: KVec<(usize, usize)> = KVec::new();
        let mut sy = 0usize;
        while sy < height {
            let mut sx = 0usize;
            while sx < width {
                let mx = (sx / MACRO_W) * MACRO_W;
                let my = (sy / MACRO_H) * MACRO_H;
                let hit = clips.iter().any(|&(x0, y0, x1, y1)| {
                    mx < x1 && x0 < mx + MACRO_W && my < y1 && y0 < my + MACRO_H
                });
                if hit {
                    coords.push((sx, sy), GFP_KERNEL)?;
                }
                sx += STRIP_W;
            }
            sy += STRIP_H;
        }
        Ok(coords)
    }

    /// Every strip of a full frame, in the same raster order as [`damage_strip_coords`].
    pub(crate) fn all_strip_coords(width: usize, height: usize) -> Result<KVec<(usize, usize)>> {
        let mut coords: KVec<(usize, usize)> = KVec::new();
        let mut sy = 0usize;
        while sy < height {
            let mut sx = 0usize;
            while sx < width {
                coords.push((sx, sy), GFP_KERNEL)?;
                sx += STRIP_W;
            }
            sy += STRIP_H;
        }
        Ok(coords)
    }

    /// Call counter for the per-strip encode breakdown logged by the damage encoder.
    static ENCODE_LOG_N: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

    /// Damage-aware variant of [`colour_frame_ep08`]: encodes **only** the 64x16 strips that
    /// intersect a client damage rectangle, producing a *partial* frame (the dock updates the tiles
    /// it receives at their self-encoded positions and keeps the rest -- how DLM does partial
    /// updates; see `docs/VIDEO-PARTIAL-UPDATE-DESIGN.md`). A typical desktop change (caret, cursor,
    /// small window) touches a handful of strips, so this is KB not MB.
    ///
    /// `clips` are `(x0, y0, x1, y1)` half-open rectangles in output/source pixels (identity
    /// rotation only -- the caller sends a full [`colour_frame_ep08`] otherwise). A strip at
    /// `(sx, sy)` is included iff some clip overlaps `[sx, sx+STRIP_W) x [sy, sy+STRIP_H)`. Raster
    /// iteration keeps strips x-ordered within each y-band, so [`frame_records`] groups them exactly
    /// as the full-frame path does. Returns an **empty** frame list when no strip is touched -- the
    /// caller must skip the USB write in that case (no-op flip). **The first frame after a mode-set
    /// must still be a full keyframe** (the dock's framebuffer is undefined until then).
    ///
    /// `#[inline(never)]`: same kernel-stack-overflow guard as [`colour_frame_ep08`].
    #[inline(never)]
    pub(crate) fn colour_frame_ep08_damage(
        width: usize,
        height: usize,
        seq0: u32,
        head: u8,
        clips: &[(usize, usize, usize, usize)],
        mut px: impl FnMut(usize, usize) -> (u8, u8, u8),
    ) -> Result<(KVec<KVec<u8>>, u32)> {
        if width % STRIP_W != 0 || height % STRIP_H != 0 {
            return Err(kernel::error::code::EINVAL);
        }
        let coords = damage_strip_coords(width, height, clips)?;
        let mut strips: KVec<KVec<u8>> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
        let (mut blocks_us, mut packs_us, nstrips) = (0i64, 0i64, coords.len());
        for &(sx, sy) in coords.iter() {
            let t0 = Instant::<Monotonic>::now();
            let blocks = colour_strip_blocks(sx, sy, &mut px)?;
            let t1 = Instant::<Monotonic>::now();
            strips.push(colour_strip(&blocks, sx as u16, sy as u16)?, GFP_KERNEL)?;
            let t2 = Instant::<Monotonic>::now();
            blocks_us += (t1 - t0).as_micros_ceil();
            packs_us += (t2 - t1).as_micros_ceil();
        }
        let t3 = Instant::<Monotonic>::now();
        let records = frame_records(&strips, head)?;
        // ★ Is the encode cost in the per-strip work (which parallelises across CPUs, since strips
        // are independent) or in the serial framing? This split is what decides whether threading
        // is the right lever -- see the phase log in `drm_sink.rs`.
        let n = ENCODE_LOG_N.fetch_add(1, core::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 2 || n % 32 == 0 {
            pr_info!(
                "vino: encode strips={nstrips} blocks(transform+quant)={blocks_us}us \
                 pack(vlc)={packs_us}us records={}us\n",
                (Instant::<Monotonic>::now() - t3).as_micros_ceil()
            );
        }
        Ok((records, seq0.wrapping_add(1)))
    }

    /// A strip's `y` (the EP08 record bands group strips by row). Reads the `y` field the strip
    /// builders write at byte offset 4 ([`colour_strip`] / [`solid_strip`]).
    fn strip_y(s: &[u8]) -> u16 {
        u16::from_le_bytes([s[4], s[5]])
    }

    /// Frame a raster-ordered list of strip bodies into EP08 USB transfers using DLM's real wire
    /// record framing (RE'd 2026-06-28 from a passive capture of DLM's EP08 output):
    ///
    /// ```text
    /// record (one per single-Y band of strips):
    ///   u16 pad   = 0
    ///   u16 size  = total record length (TLV..trailer, excludes the inter-record gap)
    ///   u32 type  = 4
    ///   u16 sub   = head | (((y / 16) & 1) << 4)
    ///   u16 aux   = zero-padding byte count
    ///   u32 fseq  = 0
    ///   per strip: u16 strip_id (== strip length) ++ strip bytes
    ///   u8[aux]   zero padding so the complete `size + 4` stride is 16-byte aligned
    /// ```
    ///
    /// The record stream is chunked internally into small record-aligned buffers. The sink streams
    /// those fragments directly into the persistent queue's 65536-byte URBs; these internal
    /// boundaries are therefore not USB framing. There is **no** per-transfer header (the old
    /// `write_ep08_header sub=0x30` over concatenated strips was wrong and made the dock fault).
    /// Every full-payload DLM desktop capture examined on 2026-07-22 has a maximum complete record
    /// stride of 4080 bytes (not 4096); retaining the extra 16 bytes made Vino emit a shape DLM
    /// never does. The previous 0x4000 cap produced 16-KiB records for real desktop content;
    /// black/grey happened to remain below the real limit and were accepted, while the first
    /// oversized live record reset the dock.
    pub(crate) fn frame_records(strips: &[KVec<u8>], head: u8) -> Result<KVec<KVec<u8>>> {
        const PREFIX: usize = 8;
        const STRIDE_CAP: usize = 0x0ff0;
        // Allocation boundary only, not wire framing. This removes the multi-megabyte contiguous
        // grow/realloc that warned in `__alloc_frozen_pages_noprof` and returned ENOMEM from the
        // DRM commit worker on 2026-07-22.
        const CHUNK: usize = 0x4000;
        let mut frames: KVec<KVec<u8>> = KVec::new();
        let mut chunk: KVec<u8> = KVec::new();
        let mut i = 0usize;
        while i < strips.len() {
            let y0 = strip_y(&strips[i]);
            let mut record: KVec<u8> = KVec::new();
            record.extend_from_slice(&[0u8; 8 + PREFIX], GFP_KERNEL)?; // TLV(8) + prefix(8)
            record[4..8].copy_from_slice(&4u32.to_le_bytes()); // type = 4
            // Bit 4 on odd 16-row bands. DLM's steady-state frames do this, but its frame 0 and
            // its field-ordered frames (all even bands, then all odd) leave it clear on every
            // record -- see docs/WHT-CODEC.md. Kept as the `record_sub_bit4` module parameter so
            // that divergence can be A/B'd against the hardware in one reload; the default is the
            // long-standing behaviour, so this changes nothing unless it is asked to.
            let parity = if *crate::module_parameters::record_sub_bit4.value() != 0 {
                (y0 / STRIP_H as u16) & 1
            } else {
                0
            };
            let sub = head as u16 | (parity << 4);
            record[8..10].copy_from_slice(&sub.to_le_bytes());
            let mut n = 0usize;
            while i < strips.len() && strip_y(&strips[i]) == y0 {
                let s = &strips[i];
                let projected = record.len() + 2 + s.len();
                let projected_aligned = (projected + 15) & !15;
                if n > 0 && projected_aligned > STRIDE_CAP {
                    break;
                }
                record.extend_from_slice(&(s.len() as u16).to_le_bytes(), GFP_KERNEL)?;
                record.extend_from_slice(s, GFP_KERNEL)?;
                n += 1;
                i += 1;
            }
            // DLM pads each complete record stride to 16 bytes and carries that exact padding count
            // in `aux`. There is no additional trailer or inter-record gap: `size` counts from
            // after the four-byte pad+size prefix, so `stride = size + 4` lands on the next record.
            let pad = (16 - (record.len() & 15)) & 15;
            record[10..12].copy_from_slice(&(pad as u16).to_le_bytes());
            record.extend_from_slice(&[0u8; 15][..pad], GFP_KERNEL)?;
            let size = (record.len() - 4) as u16;
            record[2..4].copy_from_slice(&size.to_le_bytes());

            if !chunk.is_empty() && chunk.len() + record.len() > CHUNK {
                frames.push(chunk, GFP_KERNEL)?;
                chunk = KVec::new();
            }
            chunk.extend_from_slice(&record, GFP_KERNEL)?;
        }
        if !chunk.is_empty() {
            frames.push(chunk, GFP_KERNEL)?;
        }
        Ok(frames)
    }

    /// Build DLM's three 32-byte end-of-frame records.
    ///
    /// Fresh untruncated cold-plug captures on both D6000 video endpoints show these after every
    /// logical frame, including the ARM-prefixed first frame. They are the actual frame delimiter;
    /// ending after the last image band leaves the dock waiting for them and it resets about
    /// 140 ms later. The first record carries an 8-bit frame counter (one-based, wrapping), while
    /// all three carry a three-slot phase (`0,2,4`) and the selected head in header byte 9.
    pub(crate) fn frame_trailer(head: u8, seq0: u32) -> [u8; 96] {
        let phase = ((seq0 % 3) as u8) * 2;
        let next_phase = (phase + 2) % 6;
        let phase_off = phase * 4;
        let next_off = next_phase * 4;
        let frame_no = (seq0 as u8).wrapping_add(1);
        let mut out = [0u8; 96];

        for (i, head_byte) in [head, head, head | 0x10].into_iter().enumerate() {
            let o = i * 32;
            out[o + 2] = 0x1c; // size=28 -> 32-byte record
            out[o + 4] = 0x04; // type=4
                               // `sub` is the little-endian u16 at bytes 8..10.  Head 0 hid this bug because both
                               // encodings are zero; for head 1, writing byte 9 produced 0x0100/0x1100 instead of
                               // DLM's 0x0001/0x0011 and the second display never presented the completed frame.
            out[o + 8] = head_byte;
            out[o + 10] = if i == 0 { 0x06 } else { 0x04 };
        }

        // Record A: frame-present marker + current ring phase + one-based u8 frame number.
        out[16] = 0x08;
        out[18] = 0x05;
        out[19] = phase;
        out[23] = phase_off;
        out[25] = frame_no;

        // Records B/C are identical apart from C's head|0x10 header selector.
        for o in [32usize, 64] {
            out[o + 16] = 0x0a;
            out[o + 18] = 0x04;
            out[o + 19] = next_phase;
            out[o + 23] = next_off;
            out[o + 27] = phase_off;
        }
        out
    }
}

/// Length of the EP08 transport header ([`write_ep08_header`]).
pub(crate) const EP08_HDR_LEN: usize = 16;

/// Write the 16-byte EP08 transport header into `hdr` for a `payload_len`-byte codec
/// stream: `type=4 sub=0x30 sub_len_dw=0` sec 3 framing (matches the live capture).
/// `size = payload_len + 12`. Used by the in-place scanout path. `hdr` must be at
/// least 16 bytes.
///
/// The wire `size` field is 16-bit, so a frame is limited to `u16::MAX - 12` payload
/// bytes; a larger codec stream cannot be expressed in a single frame and returns
/// `EOVERFLOW` rather than silently truncating `size` (which would desync the dock's
/// parser). This helper belongs to the retained mode-2/RLE encoder; WHT scanout uses
/// [`wht::frame_records`] and is not limited by this 16-bit header.
pub(crate) fn write_ep08_header(hdr: &mut [u8], payload_len: usize, seq: u32) -> Result {
    let size = payload_len
        .checked_add(12)
        .filter(|&s| s <= u16::MAX as usize);
    let size = size.ok_or(kernel::error::code::EOVERFLOW)?;
    hdr[0] = 0;
    hdr[1] = 0;
    hdr[2..4].copy_from_slice(&(size as u16).to_le_bytes());
    hdr[4..8].copy_from_slice(&4u32.to_le_bytes());
    hdr[8..10].copy_from_slice(&0x30u16.to_le_bytes());
    hdr[10..12].copy_from_slice(&0u16.to_le_bytes());
    hdr[12..16].copy_from_slice(&seq.to_le_bytes());
    Ok(())
}
