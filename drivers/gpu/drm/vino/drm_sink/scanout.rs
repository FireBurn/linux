// SPDX-License-Identifier: GPL-2.0

//! Turning a committed framebuffer into bytes on a video endpoint.
//!
//! The compositor's atomic callback does no work beyond snapshotting: everything below runs on the
//! deferred scanout worker. In order, a frame goes through damage selection against the previous
//! frame's per-strip hashes, encoding (fanned across CPUs as [`EncodeChunk`]s), record framing, and
//! submission through the connector's persistent URB queue.

use super::mode_objects::rot_src;
use super::*;

/// Consecutive video stalls cleared before a connector is parked for a fresh mode-set.
const VIDEO_STALL_LIMIT: u64 = 4;

/// Compress and submit one coalesced primary-plane flip on the deferred worker. Keeping all slow
/// work here makes the DRM atomic callback bounded to state inspection plus an `ARef` increment.
pub(super) fn run_pending_scanout(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    frame: PendingScanout,
) {
    use core::sync::atomic::Ordering::Relaxed;

    let connector_index = frame.connector as usize;
    if data.modeset_requested[connector_index].load(Ordering::Acquire) == 0 {
        scanout_gate(
            frame.connector,
            "worker: connector has no mode-set requested",
        );
        return;
    }
    let requested_geometry_matches = data.last_timing.lock()[connector_index]
        .is_some_and(|t| t.hactive as usize == frame.w && t.vactive as usize == frame.h);
    if !requested_geometry_matches {
        // stale framebuffer from a different-size mode generation
        scanout_gate(
            frame.connector,
            "worker: framebuffer size differs from the cached mode",
        );
        return;
    }
    // Was this the mode-set's owed keyframe? Read before sending, since a successful send clears
    // the bit.
    let was_keyframe =
        data.keyframe_pending.load(Ordering::Acquire) & (1u32 << frame.connector) != 0;
    let settle_copy = was_keyframe.then(|| frame.clone());
    let slot = frame.shadow_idx;
    let generation = frame.shadow_generation;
    let (source_w, source_h) = src_dims(frame.rotation, frame.w, frame.h);
    let shadow = {
        let mut pool = data.shadow[connector_index].lock();
        // Split the validation so the counters say *which* invariant failed.
        let in_range = slot < SHADOW_SLOTS;
        let not_inflight = pool.inflight.is_none();
        let gen_ok = in_range && pool.slots[slot].generation == generation;
        let dims_ok = in_range
            && pool.slots[slot]
                .surface
                .as_ref()
                .is_some_and(|surface| surface.w == source_w && surface.h == source_h);
        if !not_inflight {
            scanout_gate(frame.connector, "slot busy: another encode inflight");
        } else if !gen_ok {
            scanout_gate(frame.connector, "slot generation moved under us");
        } else if !dims_ok {
            scanout_gate(frame.connector, "slot surface missing or wrong size");
        }
        let valid = in_range && not_inflight && gen_ok && dims_ok;
        if !valid {
            None
        } else {
            pool.inflight = Some(slot);
            pool.slots[slot].surface.take()
        }
    };
    let Some(shadow) = shadow else {
        scanout_gate(
            frame.connector,
            "worker: committed surface is no longer available",
        );
        return;
    };

    // `pixels` and `hashes` are lent to the encoder and moved back below; the band is scratch that
    // the encoder has no use for, so it just waits here to be reunited with them.
    let ShadowSurface {
        w: source_w,
        h: source_h,
        pixels,
        hashes,
        band,
    } = shadow;
    let color = data.color_snapshot(connector_index);
    let direct = direct_pixel_map(frame.rotation, &color, source_w, source_h, frame.w, frame.h);
    let src = match Arc::new(
        PixelSource {
            pixels,
            pitch: source_w * 4,
            w: source_w,
            h: source_h,
            output_w: frame.w,
            output_h: frame.h,
            rotation: frame.rotation,
            color,
            direct,
            hashes,
            depth: data.geometry_for_connector(frame.connector).depth(),
            source_depth: data.connector_buffer_depth(frame.connector),
        },
        GFP_KERNEL,
    ) {
        Ok(src) => src,
        Err(_) => {
            let mut pool = data.shadow[connector_index].lock();
            if pool.inflight == Some(slot) {
                pool.inflight = None;
            }
            scanout_gate(frame.connector, "worker: pixel source allocation failed");
            return;
        }
    };
    let result = encode_and_send(
        dev,
        data,
        frame.connector,
        &src,
        frame.rotation,
        &frame.clips[..frame.nclips],
        frame.w,
        frame.h,
    );
    data.last_frame.lock()[connector_index] = Some(Instant::<Monotonic>::now());
    let returned = Arc::into_unique_or_drop(src).map(|src| {
        let mut src = core::pin::Pin::into_inner(src);
        ShadowSurface {
            w: source_w,
            h: source_h,
            pixels: core::mem::replace(&mut src.pixels, KVVec::new()),
            hashes: core::mem::replace(&mut src.hashes, KVVec::new()),
            band,
        }
    });
    {
        let mut pool = data.shadow[connector_index].lock();
        if pool.inflight == Some(slot) {
            if pool.slots[slot].generation == generation && pool.slots[slot].surface.is_none() {
                pool.slots[slot].surface = returned;
            }
            pool.inflight = None;
        }
    }
    match result {
        Ok(()) => {
            let n = data.scanout_fails[connector_index].swap(0, Relaxed);
            data.scanout_skip[connector_index].store(0, Relaxed);
            if n > 0 {
                pr_info!("vino: connector {connector_index} scanout recovered after {n} failed frame(s)\n");
            }
            // Arm the one-shot settle repaint. A compositor that goes idle right after enabling an
            // output can otherwise remain on the initial keyframe indefinitely.
            if let Some(mut copy) = settle_copy {
                copy.clips[0] = (0, 0, copy.w, copy.h);
                copy.nclips = 1;
                // During the post-mode-set training window, repaint at frame cadence so the dock
                // receives the sustained stream needed to program the downstream pixel clock.
                // Outside that window, use the bounded settle repaint.
                let sustaining = data.sustain_until.lock()[connector_index]
                    .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
                // Training repaints at the fast cadence and is exempt from the budget; everything
                // else charges one settle repaint against this connector's keyframe obligation and
                // stops when it runs out. See `SETTLE_REPAINTS` for the unbounded keyframe loop
                // that made a static desktop stream ~2.7 MB/s per connector. A dock that tears the
                // link down over a silent video endpoint has to be re-fed whether or not anything
                // changed, so its repaint is periodic and unbudgeted.
                let keepalive = data.video_keepalive();
                let unbudgeted = sustaining || keepalive;
                let charged = unbudgeted
                    || data.settle_budget[connector_index]
                        .fetch_update(Relaxed, Relaxed, |b| b.checked_sub(1))
                        .is_ok();
                if charged {
                    let delay = if sustaining {
                        data.frame_period_ms()
                    } else if keepalive {
                        NAVARRO_KEEPALIVE_MS
                    } else {
                        SETTLE_REPAINT_MS
                    };
                    // Only a repaint that exists to put bytes on the wire has to be a keyframe:
                    // training programs the downstream clock with them, and a keepalive dock tears
                    // its link down over a silent endpoint, so both need a frame whether or not
                    // anything changed. The ordinary settle repaint is there to replace whatever
                    // the compositor happened to have mapped when the mode set went out, and a
                    // keyframe has since reached every dock buffer -- so what it owes is the
                    // difference, which on a desktop that did not change is nothing at all.
                    // Sending a second full surface instead costs `dock_buffers` presentations of
                    // it, and on a dock that shares its control pipe that is the transfer the dock
                    // stops accepting.
                    let as_keyframe = sustaining || keepalive;
                    data.settle_repaint.lock()[connector_index] = Some((
                        Instant::<Monotonic>::now() + Delta::from_millis(delay),
                        copy,
                        as_keyframe,
                    ));
                }
            } else {
                // Repaint the same framebuffer while strips still have retransmit debt. This is a
                // delta, not a keyframe, and terminates after at most the profile's
                // `damage_frames` accepted submissions because each pass decrements the ledger.
                let owes = data.dirty_ttl.lock()[connector_index]
                    .as_ref()
                    .is_some_and(|debt| debt.iter().any(|&d| d > 0));
                if owes {
                    data.settle_repaint.lock()[connector_index] = Some((
                        Instant::<Monotonic>::now() + Delta::from_millis(data.frame_period_ms()),
                        frame.clone(),
                        false,
                    ));
                }
            }
        }
        Err(e) => {
            // Log at exponentially sparser points and back off future worker attempts. An error is
            // transport state, not a reason to stall the compositor's pageflip path.
            let n = data.scanout_fails[connector_index].fetch_add(1, Relaxed) + 1;
            if n == 1 || n.is_power_of_two() {
                pr_err!("vino: connector {connector_index} scanout frame failed ({e:?}) [x{n}] -- throttling\n");
            }
            data.scanout_skip[connector_index].store(core::cmp::min(n, 120), Relaxed);
            // The failed queue was synchronously retired at the exact `q.send()` error site while
            // that physical pipe was still owned. Give up after a few consecutive stalls and wait
            // for the next mode set rather than repeatedly driving a pipe the dock is refusing.
            if e == EPIPE || e == EPROTO {
                if n == VIDEO_STALL_LIMIT + 1 {
                    pr_err!(
                        "vino: connector {connector_index} stalled {n} times; parking it until the next mode set\n"
                    );
                    data.modeset_active[connector_index].store(0, Ordering::Release);
                    data.programmed_timing.lock()[connector_index] = None;
                }
            }
        }
    }
}

/// Copy a whole cursor framebuffer for the dock.
///
/// The dock takes DRM `ARGB8888` unchanged; [`crate::cp::cursor_image`] owns the wire placement.
/// The complete bitmap is sent every time rather than the helper's clipped rectangle: the dock is
/// configured for a fixed cursor size (`mode_config.cursor_width/height`) and clips at the panel
/// edge itself.
pub(super) fn read_cursor_bgra(
    fb: &kms::framebuffer::Framebuffer<VinoDrmDriver>,
    w: usize,
    h: usize,
) -> Result<KVec<u8>> {
    let vmap = fb.vmap::<VinoObject>()?;
    let view = vmap.view();
    let pitch = vmap.pitch();
    let row = w.checked_mul(4).ok_or(EINVAL)?;
    let len = row.checked_mul(h).ok_or(EINVAL)?;
    let mut out = KVec::new();
    out.resize(len, 0, GFP_KERNEL)?;
    for dy in 0..h {
        let src = dy.checked_mul(pitch).ok_or(EINVAL)?;
        view.try_copy_to_slice(src, &mut out[dy * row..(dy + 1) * row])?;
    }
    Ok(out)
}

/// Source (framebuffer) dimensions for an output of `ow`x`oh` pixels under plane `rotation`.
/// The 90/270 rotations swap width and height between the framebuffer and the displayed output;
/// the others preserve them.
pub(super) fn src_dims(rotation: plane::Rotation, ow: usize, oh: usize) -> (usize, usize) {
    if matches!(
        rotation.angle(),
        plane::Rotation::ROTATE_90 | plane::Rotation::ROTATE_270
    ) {
        (oh, ow)
    } else {
        (ow, oh)
    }
}

/// Copy a committed framebuffer into this connector's [`ShadowSurface`], reusing the existing
/// allocation whenever the geometry is unchanged.
///
/// Runs in the atomic commit path, so everything else (damage selection, rotation, gamma, the
/// codec) stays in the worker and reads this private surface instead of the compositor's live
/// buffer.
///
/// The traversal is band-major: for each row of strips, the source's rows are pulled into
/// [`ShadowSurface::band`] a full row per read, and only then are that band's strips hashed and --
/// where the hash moved -- copied on into `pixels`. The obvious strip-major order costs far more
/// for the same result, because a strip is 64 px wide but the source row is `pitch` bytes apart: it
/// reads the source in `STRIP_W * 4`-byte fragments (57,600 of them per 1440p frame, versus 1,440
/// full-row reads here), it walks those fragments against the row stride rather than sequentially,
/// and it has to read a changed strip out of the source a second time to copy it, because the first
/// read went to a fragment-sized scratch that could not be kept.
///
/// Strips whose hash is unchanged are still not written, so an idle desktop moves no more memory
/// than the strip-major order would, and a busy one reads the source once.
#[inline(never)]
pub(super) fn snapshot_to_shadow(
    geometry: crate::video::haar::Geometry,
    slot: &mut Option<ShadowSurface>,
    source: &kms::framebuffer::FramebufferVMapOwned<VinoObject>,
    w: usize,
    h: usize,
) -> Result {
    if w == 0 || h == 0 {
        return Err(EINVAL);
    }
    let row = w.checked_mul(4).ok_or(EINVAL)?;
    let need = row.checked_mul(h).ok_or(EINVAL)?;
    // GEM dumb buffers pad the pitch, so the source stride is not necessarily `w * 4`.
    let pitch = source.pitch();
    let view = source.view();

    let (sw, sh) = (geometry.strip_w(), geometry.strip_h());
    let padded_width = (w + sw - 1) & !(sw - 1);
    let padded_height = (h + sh - 1) & !(sh - 1);
    let tiles_x = padded_width / sw;
    let tiles_y = padded_height / sh;

    // A freshly allocated surface holds zeros, not the previous frame, so nothing in it may be
    // treated as already up to date however its stored hashes compare.
    let band_len = sh.checked_mul(row).ok_or(EINVAL)?;
    let mut fresh = false;
    if !matches!(slot, Some(s) if s.w == w && s.h == h) {
        let mut pixels: KVVec<u8> = KVVec::new();
        pixels.resize(need, 0, GFP_KERNEL)?;
        let mut hashes: KVVec<u64> = KVVec::new();
        hashes.resize(tiles_x * tiles_y, 0, GFP_KERNEL)?;
        let mut band: KVVec<u8> = KVVec::new();
        band.resize(band_len, 0, GFP_KERNEL)?;
        *slot = Some(ShadowSurface {
            w,
            h,
            pixels,
            hashes,
            band,
        });
        fresh = true;
    }
    let shadow = slot.as_mut().ok_or(kernel::error::code::ENOMEM)?;
    if shadow.hashes.len() != tiles_x * tiles_y {
        shadow.hashes.resize(tiles_x * tiles_y, 0, GFP_KERNEL)?;
    }
    if shadow.band.len() != band_len {
        shadow.band.resize(band_len, 0, GFP_KERNEL)?;
    }

    // Borrow the three buffers as disjoint fields: the band is read while `pixels` is written.
    let ShadowSurface {
        pixels,
        hashes,
        band,
        ..
    } = shadow;
    for ty in 0..tiles_y {
        let sy = ty * sh;
        let y_end = (sy + sh).min(h);
        // The final band is short whenever the height is not a whole number of strips.
        let rows = y_end - sy;
        // Pull the band out of the source through the checked I/O view, one full row per read.
        for dy in 0..rows {
            let dst = &mut band[dy * row..dy * row + row];
            view.try_copy_to_slice((sy + dy) * pitch, dst)?;
        }
        for tx in 0..tiles_x {
            let sx = tx * sw;
            let x_end = (sx + sw).min(w);
            let seed = 0x9e37_79b1_85eb_ca87u64
                ^ (sx as u64).rotate_left(17)
                ^ (sy as u64).rotate_left(43);
            let mut hasher = xxhash::Xxh64::new(seed);
            let bytes = (x_end - sx) * 4;
            // Hash exactly the bytes, in the order, that the strip-major traversal did, so a
            // surface's stored hashes stay comparable across this change.
            if sx < x_end {
                for dy in 0..rows {
                    let off = dy * row + sx * 4;
                    hasher.update(&band[off..off + bytes])?;
                }
            }
            let hash = hasher.digest();
            let idx = ty * tiles_x + tx;
            if sx < x_end && (fresh || hashes[idx] != hash) {
                for dy in 0..rows {
                    let src = dy * row + sx * 4;
                    let dst = (sy + dy) * row + sx * 4;
                    pixels[dst..dst + bytes].copy_from_slice(&band[src..src + bytes]);
                }
            }
            hashes[idx] = hash;
        }
    }
    Ok(())
}

/// Convert changed strip hashes into a compact set of damage rectangles.
///
/// The rectangles are built on the macro-tile grid, not the strip grid, because a touched
/// macro-tile is resent whole either way -- see `Geometry::macro_w`. Describing damage at a
/// granularity finer than it is transmitted at costs nothing and fragments the list by up to the
/// sixteen strips a macro-tile holds, which is enough for a handful of scattered updates to
/// exhaust the rectangle ceiling and take the whole surface with them.
///
/// Horizontal runs are joined, then equal runs on adjacent macro-rows are extended vertically. A
/// frame still fragmented past the ceiling falls back to one full-output rectangle rather than
/// growing an unbounded allocation or spending more time testing rectangles than encoding strips.
#[inline(never)]
pub(crate) fn changed_strip_rects(
    geometry: crate::video::haar::Geometry,
    old: &[u64],
    new: &[u64],
    padded_width: usize,
    padded_height: usize,
) -> Result<KVec<DamageRect>> {
    const MAX_RECTS: usize = 128;
    let tiles_x = padded_width >> geometry.strip_w_shift();
    let tiles_y = padded_height >> geometry.strip_h_shift();
    if old.len() != tiles_x * tiles_y || new.len() != old.len() {
        return Err(EINVAL);
    }
    let (mw, mh) = (geometry.macro_w(), geometry.macro_h());
    let per_x = mw / geometry.strip_w();
    let per_y = mh / geometry.strip_h();
    let macros_x = tiles_x.div_ceil(per_x);
    let macros_y = tiles_y.div_ceil(per_y);
    // The macro-tile grid as a changed/unchanged bitmap. One byte a tile: a 4K surface is 544 of
    // them, so the map is cheaper than the rectangle list it replaces.
    let mut touched: KVVec<u8> = KVVec::new();
    touched.resize(macros_x * macros_y, 0, GFP_KERNEL)?;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            if old[ty * tiles_x + tx] != new[ty * tiles_x + tx] {
                touched[(ty / per_y) * macros_x + tx / per_x] = 1;
            }
        }
    }
    let mut rects: KVec<DamageRect> = KVec::new();
    for my in 0..macros_y {
        let mut mx = 0usize;
        while mx < macros_x {
            if touched[my * macros_x + mx] == 0 {
                mx += 1;
                continue;
            }
            let run_start = mx;
            while mx < macros_x && touched[my * macros_x + mx] != 0 {
                mx += 1;
            }
            let x0 = run_start * mw;
            let x1 = (mx * mw).min(padded_width);
            let y0 = my * mh;
            let y1 = (y0 + mh).min(padded_height);
            let mut merged = false;
            for prior in rects.iter_mut().rev() {
                if prior.0 == x0 && prior.2 == x1 && prior.3 == y0 {
                    prior.3 = y1;
                    merged = true;
                    break;
                }
            }
            if !merged {
                if rects.len() == MAX_RECTS {
                    let mut full: KVec<DamageRect> = KVec::new();
                    full.push((0, 0, padded_width, padded_height), GFP_KERNEL)?;
                    return Ok(full);
                }
                rects.push((x0, y0, x1, y1), GFP_KERNEL)?;
            }
        }
    }
    Ok(rects)
}

/// Maximum packet size of a SuperSpeed bulk endpoint.
///
/// Only its role as a divisor matters here: a transfer that is a whole number of these ends on a
/// full packet and so terminates nothing. A device running below SuperSpeed uses a smaller value,
/// which divides this one, so a transfer short by this measure is short by that one too.
const BULK_MAX_PACKET: usize = 1024;

/// How much of a frame's last transfer is split off behind it so the frame ends short.
///
/// Any value that is neither zero nor a multiple of [`BULK_MAX_PACKET`] works; a record stride is
/// a multiple of sixteen, so this keeps the split on a stride boundary where a record allows one.
const FRAME_TAIL_BYTES: usize = 16;

/// Hard ceiling on work items per frame, purely to bound per-frame allocation.
///
/// Each chunk owns synchronization and a coordinate list, so the count must be
/// bounded. At `ENCODE_MIN_STRIPS_PER_CHUNK`, a full 1440p frame needs about
/// 112 chunks.
const ENCODE_MAX_WORK_ITEMS: usize = 256;

/// Fewest strips per chunk worth dispatching. Below this the allocation, enqueue and completion
/// cost more than the strips themselves; a small delta stays on the serial path.
const ENCODE_MIN_STRIPS_PER_CHUNK: usize = 32;

/// Immutable driver-owned pixel source shared by parallel encode workers.
pub(super) struct PixelSource {
    pixels: KVVec<u8>,
    pitch: usize,
    /// Dimensions of the untransformed framebuffer snapshot.
    w: usize,
    h: usize,
    /// Dimensions and transform of the image presented to the dock.
    output_w: usize,
    output_h: usize,
    rotation: plane::Rotation,
    color: Option<crate::color::ColorPipeline>,
    /// True when an output pixel is the source pixel at the same coordinates: identity rotation, no
    /// gamma table, and output dimensions equal to the snapshot's.
    ///
    /// Fullscreen video makes `px` the third-hottest symbol in the kernel (13.8% of the machine
    /// on a 4K clip), because every one of the ~3.7 M pixels per frame pays a `rot_src` match and
    /// a gamma branch that are constant for the whole frame. Deciding once per frame lets the
    /// common case read straight out of the snapshot.
    direct: bool,
    /// Strip hashes computed during the snapshot -- see [`ShadowSurface::hashes`]. Carried through
    /// so the encoder does not re-read the whole surface just to re-derive them.
    hashes: KVVec<u64>,
    /// Bits per channel of the snapshot's pixels, and therefore of the frame the codec produces.
    ///
    /// Both layouts this handles are four bytes per pixel, so the snapshot copy above is
    /// depth-agnostic and only the unpack here differs: `XRGB8888` is three bytes in the low 24
    /// bits, `XRGB2101010` three 10-bit fields in the low 30.
    depth: crate::video::haar::Depth,
    /// Bits per channel of the framebuffer itself, which decides how a pixel is decoded.
    ///
    /// Equal to `depth` except when userspace asked for a deeper link than the surface it handed
    /// over, which is the ordinary way a compositor drives a ten-bit link from an eight-bit
    /// desktop. Samples are widened after decoding so the codec always sees `depth`.
    source_depth: crate::video::haar::Depth,
}

/// Whether the encoder can read output pixels straight out of the snapshot. See
/// [`PixelSource::direct`].
fn direct_pixel_map(
    rotation: plane::Rotation,
    color: &Option<crate::color::ColorPipeline>,
    w: usize,
    h: usize,
    output_w: usize,
    output_h: usize,
) -> bool {
    color.is_none() && rotation == plane::Rotation::ROTATE_0 && output_w == w && output_h == h
}

/// Widen an eight-bit sample to ten bits.
///
/// Replicating the top two bits into the low ones keeps both endpoints exact: black stays zero and
/// full white stays full white, where a plain shift would land three codes short of it and tint
/// every highlight.
#[inline]
fn widen_8_to_10(v: u16) -> u16 {
    (v << 2) | (v >> 6)
}

impl PixelSource {
    /// Split one packed 32-bit pixel into channels at this source's depth.
    #[inline]
    fn unpack(&self, p: u32) -> (u16, u16, u16) {
        let (r, g, b) = self.unpack_source(p);
        self.widen(r, g, b)
    }

    /// Widen a decoded sample from the framebuffer's depth to the link's.
    #[inline]
    fn widen(&self, r: u16, g: u16, b: u16) -> (u16, u16, u16) {
        match (self.source_depth, self.depth) {
            (crate::video::haar::Depth::Eight, crate::video::haar::Depth::Ten) => {
                (widen_8_to_10(r), widen_8_to_10(g), widen_8_to_10(b))
            }
            _ => (r, g, b),
        }
    }

    /// Split one packed 32-bit pixel into channels at the framebuffer's own depth.
    #[inline]
    fn unpack_source(&self, p: u32) -> (u16, u16, u16) {
        match self.source_depth {
            // Little-endian XRGB8888.
            crate::video::haar::Depth::Eight => (
                ((p >> 16) & 0xff) as u16,
                ((p >> 8) & 0xff) as u16,
                (p & 0xff) as u16,
            ),
            // XRGB2101010: two ignored bits, then R, G, B ten bits each.
            crate::video::haar::Depth::Ten => (
                ((p >> 20) & 0x3ff) as u16,
                ((p >> 10) & 0x3ff) as u16,
                (p & 0x3ff) as u16,
            ),
        }
    }

    /// Read one gamma-corrected pixel in untransformed framebuffer coordinates.
    #[inline]
    fn source_px(&self, sx: usize, sy: usize) -> (u16, u16, u16) {
        if sx >= self.w || sy >= self.h {
            return (0, 0, 0);
        }
        let off = sy * self.pitch + sx * 4;
        // Bounds-checked once per pixel instead of the serial path's raw `read_unaligned`. The
        // check is noise next to the 64-coefficient transform each pixel feeds.
        let Some(chunk) = self.pixels.get(off..off + 4) else {
            return (0, 0, 0);
        };
        let Ok(bytes) = <[u8; 4]>::try_from(chunk) else {
            return (0, 0, 0);
        };
        let (r, g, b) = self.unpack(u32::from_le_bytes(bytes));
        match &self.color {
            // The colour pipeline's tables are 8-bit, so a 10-bit surface is corrected at 8-bit
            // precision and scaled back. That loses up to two low bits of a corrected pixel -- but ignoring the
            // correction instead would leave a compositor's night-colour shift silently unapplied
            // on exactly the outputs most likely to be colour-managed. Revisit with a 10-bit LUT if
            // banding is ever measured on a corrected HDR connector.
            Some(pipeline) => match self.depth {
                crate::video::haar::Depth::Eight => {
                    let (r, g, b) = pipeline.apply(r as u8, g as u8, b as u8);
                    (r as u16, g as u16, b as u16)
                }
                crate::video::haar::Depth::Ten => {
                    let (r, g, b) = pipeline.apply((r >> 2) as u8, (g >> 2) as u8, (b >> 2) as u8);
                    (
                        (r as u16) << 2 | (r as u16 >> 6),
                        (g as u16) << 2 | (g as u16 >> 6),
                        (b as u16) << 2 | (b as u16 >> 6),
                    )
                }
            },
            None => (r, g, b),
        }
    }

    /// Read one output pixel after applying the plane transform.
    ///
    /// Keeping the transform in the immutable shared source gives serial and parallel encoding
    /// exactly the same sampler. Codec padding is black and never reads beyond the snapshot.
    #[inline]
    fn px(&self, dx: usize, dy: usize) -> (u16, u16, u16) {
        if self.direct {
            // The codec pads the surface up to whole strips and expects black outside the image, so
            // the bounds check stays: without it a read past the row wraps into the next one.
            if dx >= self.w || dy >= self.h {
                return (0, 0, 0);
            }
            let off = dy * self.pitch + dx * 4;
            let Some(chunk) = self.pixels.get(off..off + 4) else {
                return (0, 0, 0);
            };
            if let crate::video::haar::Depth::Eight = self.source_depth {
                // Little-endian XRGB8888: byte 0 is blue, 1 green, 2 red.
                return self.widen(chunk[2] as u16, chunk[1] as u16, chunk[0] as u16);
            }
            let Ok(bytes) = <[u8; 4]>::try_from(chunk) else {
                return (0, 0, 0);
            };
            return self.unpack(u32::from_le_bytes(bytes));
        }
        if dx >= self.output_w || dy >= self.output_h {
            return (0, 0, 0);
        }
        let (sx, sy) = rot_src(self.rotation, dx, dy, self.w, self.h);
        self.source_px(sx, sy)
    }
}

/// vino's own workqueue for the parallel strip encode.
///
/// On the shared `system_unbound` pool the encode's CPU time is anonymous: the kernel
/// composes worker thread names from the workqueue's, so shared-pool work appears only as
/// `kworker/uN:M-events_unbound`, indistinguishable from every other user of that pool. On a queue
/// of our own the same threads appear as `kworker/uN:M-vino_encode`, so `ps`/`top`/`perf`
/// attribute the codec's cost to vino, and the fan-out does not compete with unrelated work for
/// the shared pool's concurrency budget.
///
/// `WQ_UNBOUND` because strip encoding is pure compute with no CPU affinity worth preserving --
/// that property is what let the fan-out reach ~7.4x. Allocated once on first use and never
/// destroyed: it is driver-wide, costs one `workqueue_struct`, and outliving every `EncodeChunk` is
/// exactly what makes the join safe.
///
/// `max_active` is the CPU count. An unbound queue built without it takes `WQ_DFL_ACTIVE`, which is
/// half of `WQ_MAX_ACTIVE` and so far above any useful degree of parallelism that it does not bound
/// the fan-out at all: a full frame is a few hundred chunks per connector, and every one of them
/// becomes a runnable CPU-bound worker at once. On a machine that is already busy that is not
/// parallelism, it is a thundering herd -- it evicts the caches the codec depends on and stalls
/// unrelated work, including the compositor thread this driver is waiting on. Fine-grained chunks
/// are still the right unit (see `encode_across_cpus`); how many run at once is a separate
/// decision, and this is where it is made.
///
/// Falls back to `system_unbound` if the allocation ever fails, so a failure here costs the thread
/// *name*, not the driver.
fn encode_queue() -> Option<&'static workqueue::Queue> {
    static ENCODE_WQ: kernel::sync::SetOnce<workqueue::OwnedQueue> = kernel::sync::SetOnce::new();
    if let Some(q) = ENCODE_WQ.as_ref() {
        return Some(q);
    }
    // A concurrent racer may win the `SetOnce`; its queue is dropped and the winner's is used.
    if let Ok(q) = workqueue::Queue::new_unbound()
        .max_active(kernel::cpu::nr_cpu_ids().max(1))
        .build(kernel::c_str!("vino_encode"))
    {
        let _ = ENCODE_WQ.populate(q);
    }
    ENCODE_WQ.as_ref().map(|q| &**q)
}

/// Encode a batch of strips from `src`, in the order given.
fn encode_coords(
    geometry: crate::video::haar::Geometry,
    src: &PixelSource,
    coords: &[(usize, usize)],
) -> Result<KVec<KVec<u8>>> {
    let mut out = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
    for &(sx, sy) in coords.iter() {
        let mut px = |dx, dy| src.px(dx, dy);
        out.push(
            crate::video::haar::colour_strip_at(geometry, sx, sy, &mut px)?,
            GFP_KERNEL,
        )?;
    }
    Ok(out)
}

/// One contiguous batch of strips, encoded on whichever CPU the unbound workqueue picks.
///
/// Chunks share nothing but the read-only [`PixelSource`]: each writes only its own `out`, so no
/// locking is needed on the hot path and the results reassemble by chunk order.
#[pin_data]
struct EncodeChunk {
    #[pin]
    work: Work<EncodeChunk>,
    #[pin]
    done: Completion,
    src: Arc<PixelSource>,
    coords: KVec<(usize, usize)>,
    /// The dock's strip layout, carried per chunk so two docks of different generations can
    /// encode concurrently on the same workqueue.
    geometry: crate::video::haar::Geometry,
    /// Encoded strip bodies. Written once by the worker, read once by the joiner after `done`;
    /// the lock is uncontended and taken twice per chunk per frame.
    #[pin]
    out: Mutex<KVec<KVec<u8>>>,
}

impl_has_work! {
    impl HasWork<Self> for EncodeChunk { self.work }
}

impl EncodeChunk {
    fn new(
        geometry: crate::video::haar::Geometry,
        src: Arc<PixelSource>,
        coords: KVec<(usize, usize)>,
    ) -> Result<Arc<Self>> {
        Arc::pin_init(
            pin_init!(EncodeChunk {
                work <- new_work!("vino::EncodeChunk::work"),
                done <- Completion::new(),
                src,
                coords,
                geometry,
                out <- new_mutex!(KVec::new(), "vino::EncodeChunk::out"),
            }),
            GFP_KERNEL,
        )
    }
}

impl WorkItem for EncodeChunk {
    type Pointer = Arc<EncodeChunk>;

    fn run(this: Arc<EncodeChunk>) {
        if let Ok(strips) = encode_coords(this.geometry, &this.src, &this.coords) {
            *this.out.lock() = strips;
        }
        // Complete unconditionally. On failure `out` stays short and the joiner detects that by
        // length -- but it must never be left blocked on a completion that cannot fire.
        this.done.complete_all();
    }
}

/// Encode `coords` across CPUs and return the strip bodies in the same order.
///
/// Order is not a nicety: [`crate::video::haar::frame_records`] groups strips into one wire record
/// per single-Y band and needs them x-ordered within a band, so the chunks are contiguous slices
/// of the raster-ordered coordinate list and are reassembled strictly in chunk order.
///
/// Returns `Ok(None)` when the frame is too small to be worth splitting, so the caller falls
/// through to the serial encoder rather than paying dispatch cost for a handful of strips.
fn parallel_strip_encode(
    geometry: crate::video::haar::Geometry,
    src: &Arc<PixelSource>,
    coords: &[(usize, usize)],
) -> Result<Option<KVec<KVec<u8>>>> {
    // Size chunks from the amount of work, not from the CPU count. Handing the queue more, smaller
    // items than there are CPUs costs a little dispatch overhead but improves load balancing: with
    // one chunk per CPU a single slow chunk holds up the whole join, whereas fine-grained items let
    // idle workers pick up the remainder.
    //
    // This is only safe because `encode_queue` caps `max_active` at the CPU count. The chunk count
    // is the unit of work; that cap is what keeps the count from also becoming the number of
    // threads competing for the machine.
    let nchunks = (coords.len() / ENCODE_MIN_STRIPS_PER_CHUNK).min(ENCODE_MAX_WORK_ITEMS);
    if nchunks < 2 {
        return Ok(None);
    }
    let per = coords.len().div_ceil(nchunks);

    let mut chunks: KVec<Arc<EncodeChunk>> = KVec::with_capacity(nchunks, GFP_KERNEL)?;
    let mut queued: KVec<bool> = KVec::with_capacity(nchunks, GFP_KERNEL)?;
    let mut start = 0usize;
    while start < coords.len() {
        let end = (start + per).min(coords.len());
        let mut mine: KVec<(usize, usize)> = KVec::with_capacity(end - start, GFP_KERNEL)?;
        for &c in &coords[start..end] {
            mine.push(c, GFP_KERNEL)?;
        }
        let chunk = EncodeChunk::new(geometry, src.clone(), mine)?;
        // `enqueue` gives the item back if it is already pending -- impossible for one allocated
        // a line ago, but if it ever happened, waiting on its completion would hang the scanout
        // worker forever. Record it and encode that chunk inline instead.
        let ok = encode_queue()
            .map_or_else(
                || workqueue::system_unbound().enqueue(chunk.clone()),
                |q| q.enqueue(chunk.clone()),
            )
            .is_ok();
        queued.push(ok, GFP_KERNEL)?;
        chunks.push(chunk, GFP_KERNEL)?;
        start = end;
    }

    // The scanout worker runs on the per-device scanout queue, so blocking here cannot deadlock
    // against the separate unbound pool the chunks run on.
    let mut strips: KVec<KVec<u8>> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
    for (i, chunk) in chunks.iter().enumerate() {
        let mine = if queued[i] {
            chunk.done.wait_for_completion();
            core::mem::take(&mut *chunk.out.lock())
        } else {
            encode_coords(chunk.geometry, &chunk.src, &chunk.coords)?
        };
        if mine.len() != chunk.coords.len() {
            // A chunk failed to allocate. Sending a frame with strips missing would paint a
            // partial image the dock would keep, so fail the whole encode and let the caller's
            // retry/backoff handle it.
            return Err(ENOMEM);
        }
        for s in mine {
            strips.push(s, GFP_KERNEL)?;
        }
    }
    Ok(Some(strips))
}

/// Verify that workqueue fan-out produces the same strip bytes as the serial transformed sampler.
///
/// This is kept behind the Vino KUnit option so the production driver carries no test allocation
/// or dispatch path. The deliberately unaligned output also verifies that both paths produce
/// identical black codec padding.
#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
pub(crate) fn parallel_rotation_matches_serial(rotation: plane::Rotation) -> Result {
    let (output_w, output_h) = (500usize, 123usize);
    let (w, h) = src_dims(rotation, output_w, output_h);
    let len = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .ok_or(EINVAL)?;
    let mut pixels: KVVec<u8> = KVVec::new();
    pixels.resize(len, 0, GFP_KERNEL)?;
    for sy in 0..h {
        for sx in 0..w {
            let off = (sy * w + sx) * 4;
            pixels[off] = ((sx * 3 + sy * 5) & 0xff) as u8;
            pixels[off + 1] = ((sx * 7 + sy * 11) & 0xff) as u8;
            pixels[off + 2] = ((sx * 13 + sy * 17) & 0xff) as u8;
            pixels[off + 3] = 0xff;
        }
    }
    let src = Arc::new(
        PixelSource {
            pixels,
            pitch: w * 4,
            w,
            h,
            output_w,
            output_h,
            rotation,
            color: None,
            direct: direct_pixel_map(rotation, &None, w, h, output_w, output_h),
            hashes: KVVec::new(),
            depth: crate::video::haar::Depth::Eight,
            source_depth: crate::video::haar::Depth::Eight,
        },
        GFP_KERNEL,
    )?;
    let geometry = crate::video::haar::RIDGE_GEOMETRY;
    let padded_width = output_w.next_multiple_of(geometry.strip_w());
    let padded_height = output_h.next_multiple_of(geometry.strip_h());
    let coords = crate::video::haar::all_strip_coords(geometry, padded_width, padded_height)?;

    let mut serial: KVec<KVec<u8>> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
    for &(strip_x, strip_y) in coords.iter() {
        let mut px = |dx, dy| {
            if dx >= output_w || dy >= output_h {
                return (0, 0, 0);
            }
            let (sx, sy) = rot_src(rotation, dx, dy, w, h);
            src.source_px(sx, sy)
        };
        serial.push(
            crate::video::haar::colour_strip_at(geometry, strip_x, strip_y, &mut px)?,
            GFP_KERNEL,
        )?;
    }

    let parallel = parallel_strip_encode(geometry, &src, &coords)?.ok_or(EINVAL)?;
    if serial.len() != parallel.len()
        || serial
            .iter()
            .zip(parallel.iter())
            .any(|(expected, actual)| expected[..] != actual[..])
    {
        return Err(EINVAL);
    }
    Ok(())
}

/// A frame that trips one of these is dropped between the compositor's commit and the wire.
fn scanout_gate(connector: u8, reason: &str) {
    vino_debug!("vino: scanout connector={connector} deferred: {reason}\n");
}

#[inline(never)]
fn encode_and_send_haar(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    connector: u8,
    src: &Arc<PixelSource>,
    rotation: plane::Rotation,
    _clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    let geometry = data.geometry_for_connector(connector);
    let connector_index = connector as usize;
    // The vendor sends the frames that open a stream without the steady-state record bit and
    // every frame after them with it, whether they carry a whole surface or one damaged strip.
    // The training window is that opening.
    let opening = data.sustain_until.lock()[connector_index]
        .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
    let geometry = if opening {
        geometry.opening()
    } else {
        geometry
    };
    // Gate video on the matching mode-set reaching the dock. Plane updates run before the CRTC
    // enable queues that mode-set, and the dock rejects video on an unconfigured stream. Deferring
    // does not advance the codec sequence; the next scanout pass retries the frame.
    let want = data.modeset_requested[connector_index].load(core::sync::atomic::Ordering::Acquire);
    if want == 0 {
        scanout_gate(connector, "no mode-set requested (modeset_requested == 0)");
        return Ok(());
    }
    // A dock that cannot report downstream presence offers every connector it has, because its
    // activation is one dock-wide transaction and a connector nobody described is not the same dock
    // state as a connector that does not exist. The empty socket still recovers no EDID, and
    // painting it spends a full surface per repaint on a sink that is not there -- on this family
    // over the endpoint its control plane also needs. So configure the connector and leave it
    // showing the black its carrier put there, rather than streaming to nothing.
    if !data.reports_presence() && !data.connector_has_edid(connector_index) {
        scanout_gate(connector, "no monitor has described this socket");
        return Ok(());
    }
    let cached = data.last_timing.lock()[connector_index];
    if !cached.is_some_and(|t| {
        timing_key(&t) == want && t.hactive as usize == w && t.vactive as usize == h
    }) {
        scanout_gate(
            connector,
            "cached timing does not match the requested mode generation",
        );
        return Ok(());
    }
    if data.modeset_active[connector_index].load(core::sync::atomic::Ordering::Acquire) != want {
        // A failed command-worker activation leaves the desired generation intact. This worker is
        // sleepable, so retry the same transaction before submitting its pending framebuffer.
        let timing = cached.ok_or(EINVAL)?;
        data.activate_head(dev, connector, &timing, want)?;
        // A successful inline retry has made this very commit safe to send: continue into the
        // encoder instead of waiting for another page flip.  A completely static connector may not
        // receive another atomic update after its enabling commit.
        if data.modeset_active[connector_index].load(core::sync::atomic::Ordering::Acquire) != want
        {
            scanout_gate(
                connector,
                "mode-set not active and the inline re-send did not land",
            );
            return Ok(());
        }
    }
    let seq0 = data.scanout_seq.lock()[connector_index];
    // Source dimensions (swapped from the output for 90/270 rotation).
    let (sw, sh) = src_dims(rotation, w, h);
    if src.w != sw
        || src.h != sh
        || src.output_w != w
        || src.output_h != h
        || src.rotation != rotation
    {
        return Err(EINVAL);
    }
    // Full keyframe vs damage delta. A mode-set requires a keyframe; rotation/reflection remains
    // conservative because the content shadow is deliberately stored in unrotated framebuffer
    // space. For identity rotation, compare the actual framebuffer instead of trusting optional
    // FB_DAMAGE_CLIPS: KWin commonly changes framebuffer objects without publishing that blob.
    let kf_bit = 1u32 << connector_index;
    let identity = rotation.angle() == plane::Rotation::ROTATE_0
        && !rotation.contains(plane::Rotation::REFLECT_X | plane::Rotation::REFLECT_Y);
    let owes_keyframe = data
        .keyframe_pending
        .load(core::sync::atomic::Ordering::Acquire)
        & kf_bit
        != 0;
    let mut full = owes_keyframe || !identity;
    // The codec operates on complete 64x16 strips. Pad non-aligned modes to the next strip
    // boundary; the mode-set retains the visible dimensions and the sampler supplies black for
    // pixels outside them.
    let padded_width = (w + geometry.strip_w() - 1) & !(geometry.strip_w() - 1);
    let padded_height = (h + geometry.strip_h() - 1) & !(geometry.strip_h() - 1);
    let mut content_hashes: Option<KVVec<u64>> = None;
    let mut content_damage: KVec<DamageRect> = KVec::new();
    // Strips whose pixels moved since the last accepted frame, before the retransmit debt and the
    // macro-tile rounding widen that into what is actually sent. Reported next to the selected
    // count so an oversized delta says which of the two widened it.
    let mut moved_strips = 0usize;
    if identity {
        let expected = (padded_width >> geometry.strip_w_shift())
            * (padded_height >> geometry.strip_h_shift());
        if src.hashes.len() != expected {
            return Err(EINVAL);
        }
        let mut hashes: KVVec<u64> = KVVec::new();
        hashes.resize(expected, 0, GFP_KERNEL)?;
        hashes.copy_from_slice(&src.hashes);
        if !full {
            let previous = data.strip_hashes.lock();
            if let Some(state) = &previous[connector_index] {
                if state.padded_width == padded_width && state.padded_height == padded_height {
                    // Charge every strip whose content moved with the profile's logical-frame
                    // debt, then select every strip that still owes a transmission -- including
                    // ones that changed on an earlier frame and have not reached every dock
                    // buffer. See `dirty_ttl` and `FrameDelivery`.
                    let mut ttl = data.dirty_ttl.lock();
                    if !ttl[connector_index]
                        .as_ref()
                        .is_some_and(|t| t.len() == hashes.len())
                    {
                        let mut fresh: KVVec<u8> = KVVec::new();
                        fresh.resize(hashes.len(), 0, GFP_KERNEL)?;
                        ttl[connector_index] = Some(fresh);
                    }
                    let debt = ttl[connector_index]
                        .as_mut()
                        .ok_or(kernel::error::code::ENOMEM)?;
                    let damage_frames = data.frame_delivery().damage_frames.max(1);
                    for i in 0..hashes.len() {
                        if state.hashes[i] != hashes[i] {
                            debt[i] = damage_frames;
                            moved_strips += 1;
                        }
                    }
                    // Reuse the hash differ: mark an owed strip by handing it a baseline value that
                    // cannot match, and an unowed one its own value.
                    let mut baseline: KVVec<u64> = KVVec::new();
                    baseline.resize(hashes.len(), 0, GFP_KERNEL)?;
                    for i in 0..hashes.len() {
                        baseline[i] = if debt[i] > 0 { !hashes[i] } else { hashes[i] };
                    }
                    content_damage = changed_strip_rects(
                        geometry,
                        &baseline,
                        &hashes,
                        padded_width,
                        padded_height,
                    )?;
                } else {
                    full = true;
                }
            } else {
                full = true;
            }
        }
        content_hashes = Some(hashes);
    }
    if !full && content_damage.is_empty() {
        scanout_gate(connector, "no keyframe owed and no strip content changed");
        return Ok(());
    }
    // Serial fallback and parallel workers share the same transformed sampler.
    let px = |dx: usize, dy: usize| src.px(dx, dy);
    // Damage selection and encoded-strip reuse remain identity-only. Rotated and reflected frames
    // are conservative full updates, but their independent strips can use the same workqueue
    // fan-out as an identity keyframe.
    // What the encoded bytes depend on besides the strip pixels themselves; see
    // `StripHashState::tag`. Identity rotation is a precondition of caching at all, so it needs no
    // representation here.
    let encode_tag = {
        let gamma = match &src.color {
            Some(pipeline) => pipeline.tag(),
            None => 0,
        };
        // The sample depth belongs here for the same reason the colour transform does, and is the
        // more dangerous of the two: changing it re-maps every sample on the way into the codec
        // and moves the escape ceiling, while leaving the framebuffer byte for byte identical. A
        // cache keyed on the pixels alone therefore serves bodies encoded at the old depth inside
        // a stream declared at the new one, and the dock decodes part of the frame as noise.
        let deep = matches!(src.depth, crate::video::haar::Depth::Ten);
        gamma ^ (u64::from(deep) * 0x9e37_79b9_7f4a_7c15)
    };
    // Strips carried over verbatim from the previous frame's encode, and the strips actually
    // handed to the codec; kept for the post-send cache publish below.
    let mut encoded: Option<(KVec<(usize, usize)>, KVec<KVec<u8>>)> = None;
    let parallel = if !identity {
        let coords = crate::video::haar::all_strip_coords(geometry, padded_width, padded_height)?;
        match parallel_strip_encode(geometry, src, &coords)? {
            Some(strips) => {
                let records = if geometry.connector_selector_shift != 0 {
                    crate::video::haar::frame_records_navarro_ordinary(
                        geometry, &strips, connector,
                    )?
                } else {
                    crate::video::haar::frame_records(geometry, &strips, connector)?
                };
                Some(records)
            }
            None => None,
        }
    } else {
        let coords = if full {
            crate::video::haar::all_strip_coords(geometry, padded_width, padded_height)?
        } else {
            crate::video::haar::damage_strip_coords(
                geometry,
                padded_width,
                padded_height,
                &content_damage,
            )?
        };
        // What this frame costs the dock, and why. A delta that selects far more strips than moved
        // was widened by the retransmit debt or by rounding out to whole macro-tiles; one that
        // selects the surface from a handful of rectangles hit the rectangle ceiling instead.
        vino_debug!(
            "vino: scanout connector={} {} {}/{} strips from {} rect(s), {} moved\n",
            connector,
            if full { "keyframe" } else { "delta" },
            coords.len(),
            (padded_width >> geometry.strip_w_shift())
                * (padded_height >> geometry.strip_h_shift()),
            content_damage.len(),
            moved_strips
        );
        // Reuse an encoded strip body when its pixels and gamma tag are unchanged. Encode only
        // misses, then restore the required x-order within each Y band.
        let tiles_x = padded_width >> geometry.strip_w_shift();
        let mut reuse: KVec<Option<KVec<u8>>> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
        let mut misses: KVec<(usize, usize)> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
        {
            let cache = data.strip_hashes.lock();
            let usable = cache[connector_index].as_ref().filter(|c| {
                c.padded_width == padded_width
                    && c.padded_height == padded_height
                    && c.tag == encode_tag
            });
            for &(sx, sy) in coords.iter() {
                let idx =
                    (sy >> geometry.strip_h_shift()) * tiles_x + (sx >> geometry.strip_w_shift());
                let hit = usable.and_then(|c| {
                    // Same pixels as when this body was produced, and a body was kept.
                    let same = c.hashes.get(idx).zip(content_hashes.as_ref()?.get(idx));
                    let body = c.bodies.get(idx)?;
                    (same.is_some_and(|(a, b)| a == b) && !body.is_empty()).then_some(body)
                });
                match hit {
                    Some(body) => {
                        let mut copy: KVec<u8> = KVec::with_capacity(body.len(), GFP_KERNEL)?;
                        copy.extend_from_slice(body, GFP_KERNEL)?;
                        reuse.push(Some(copy), GFP_KERNEL)?;
                    }
                    None => {
                        reuse.push(None, GFP_KERNEL)?;
                        misses.push((sx, sy), GFP_KERNEL)?;
                    }
                }
            }
        }
        let fresh = match parallel_strip_encode(geometry, src, &misses)? {
            Some(s) => Some(s),
            // Too few misses to be worth splitting: encode them here rather than dropping to
            // the whole-frame serial path, which would re-encode the cache hits as well.
            None if !misses.is_empty() => Some(encode_coords(geometry, src, &misses)?),
            None => Some(KVec::new()),
        };
        match fresh {
            Some(fresh) if fresh.len() == misses.len() => {
                let mut strips: KVec<KVec<u8>> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
                let mut next = fresh.into_iter();
                for slot in reuse {
                    match slot {
                        Some(body) => strips.push(body, GFP_KERNEL)?,
                        None => strips.push(next.next().ok_or(EINVAL)?, GFP_KERNEL)?,
                    }
                }
                let records = if full && geometry.connector_selector_shift != 0 {
                    crate::video::haar::frame_records_navarro_ordinary(
                        geometry, &strips, connector,
                    )?
                } else {
                    crate::video::haar::frame_records(geometry, &strips, connector)?
                };
                encoded = Some((coords, strips));
                Some(records)
            }
            _ => None,
        }
    };
    let frames = match parallel {
        Some(r) => r,
        None if full && geometry.connector_selector_shift != 0 => {
            crate::video::haar::colour_frame_ep08_navarro_ordinary(
                geometry,
                padded_width,
                padded_height,
                connector,
                px,
            )?
        }
        None if full => crate::video::haar::colour_frame_ep08(
            geometry,
            padded_width,
            padded_height,
            connector,
            px,
        )?,
        None => crate::video::haar::colour_frame_ep08_damage(
            geometry,
            padded_width,
            padded_height,
            connector,
            &content_damage,
            px,
        )?,
    };
    // A damage delta that touched no aligned strip = nothing to send this flip: skip the write
    // (no seq advance, no arm, keyframe obligation untouched). Full frames always have strips.
    if frames.is_empty() {
        scanout_gate(connector, "encoder produced zero records");
        return Ok(());
    }
    if data.shutting_down.load(Ordering::Acquire)
        || data.modeset_requested[connector_index].load(Ordering::Acquire) != want
        || data.modeset_active[connector_index].load(Ordering::Acquire) != want
    {
        scanout_gate(
            connector,
            "mode generation changed between encode and submit",
        );
        return Ok(());
    }
    // A frame is one continuous bulk stream: intermediate transfers end on a full 1024-byte packet
    // and only the final transfer is short. A short packet at a record boundary terminates the
    // frame early and desynchronises the dock. The first frame after a mode set also prepends the
    // connector's ten-record arm burst; clear that obligation only after a successful submission.
    let connector_bit = 1u32 << connector;
    // The 2560-byte arm burst appears only on frame zero after a mode set. Later frames begin
    // directly with video records.
    let arm = if data
        .arm_prefix_pending
        .load(core::sync::atomic::Ordering::Acquire)
        & connector_bit
        != 0
    {
        Some(data.build_stream_prefix_buf(connector_index)?)
    } else {
        None
    };
    let arm_len = arm.as_ref().map_or(0, |a| a.len());
    // Revalidate at the actual wire boundary too. The encoded bytes and ARM prefix are specific to
    // this mode generation; submitting them after a concurrent disable/re-enable poisons the next
    // stream even though every USB URB can still complete successfully.
    if data.shutting_down.load(Ordering::Acquire)
        || data.modeset_requested[connector_index].load(Ordering::Acquire) != want
        || data.modeset_active[connector_index].load(Ordering::Acquire) != want
    {
        vino_debug!(
            "vino: scanout connector={} superseded before video submit; frame discarded\n",
            connector
        );
        return Ok(());
    }
    // Preserve the last readiness-to-video adjacency from the VINO session that lit both panels.
    // These are real CP status transactions (with EP84 replies drained by `send_cp`), paced at the
    // required cadence, and only run for frame zero while the ARM prefix is present.
    if arm.is_some() {
        for _ in 0..VinoDrmData::PREWRITE_POLLS {
            data.poll_status(dev)?;
            fsleep(Delta::from_millis(VinoDrmData::PREWRITE_POLL_MS as i64));
        }
        vino_debug!(
            "vino: inline pre-write paced poll ({}x @{}ms) before first video connector={}\n",
            VinoDrmData::PREWRITE_POLLS,
            VinoDrmData::PREWRITE_POLL_MS,
            connector
        );
    }
    // Frame zero starts with an arm record; later frames start with video records. Record fragments
    // are allocation boundaries only and are joined into exact 64-KiB transfers below without a
    // whole-frame coalescing allocation.
    let frame_count = frames.len();
    let image_len: usize = frames.iter().take(frame_count).map(|f| f.len()).sum();
    // The DL7400's per-strip parameter map. DLM and Windows both include it in frame zero after
    // at least some image records; the deterministic image-then-map ordering is used below. Ridge
    // has no equivalent record and gets an empty slice.
    let params: KVec<u8> = if geometry.connector_selector_shift == 0 {
        KVec::new()
    } else {
        crate::video::haar::navarro_strip_params(
            geometry,
            connector,
            padded_width,
            padded_height,
            &frames,
            &mut data.strip_classes.lock()[connector_index],
        )?
    };
    vino_debug!(
        "vino: connector={} shift={} params={} B ({}x{} pad)\n",
        connector,
        geometry.connector_selector_shift,
        params.len(),
        padded_width,
        padded_height
    );
    if arm.is_some() {
        data.send_stream_open(dev, connector_index)?;
    }
    let startup = arm.is_some();
    // A cold link requires a bounded back-to-back full-frame burst until the downstream clock is
    // programmed. Reuse the encoded image and advance only its frame trailer and per-frame control
    // sync. The arm prefix remains exclusive to presentation zero.
    let training = full
        && data.sustain_until.lock()[connector_index]
            .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
    // A dock that shares its control pipe cannot be given the training window's presentation
    // count: a whole-surface keyframe is 1.88 MB there, and eight back-to-back copies of it are
    // 15 MB in one scanout against a dock DLM feeds at 0.86 MB/s. Measured, the dock accepts
    // three such frames and then stops answering the control plane entirely. Every buffer still
    // gets the keyframe; it just does not get it eight times.
    // A keyframe has to initialise every buffer before its content shadow can become authoritative.
    // Ordinary updates are different: DLM sends one Ella presentation per logical frame and lets
    // successive frames walk the ring.  The profile keeps those two requirements separate and the
    // damage ledger schedules later logical frames if the compositor becomes idle.
    let repeat_count = frame_presentation_count(
        data.frame_delivery(),
        full,
        training,
        data.video_on_ctrl_pipe(),
    );
    let first_opener_len = data
        .build_frame_opener(connector, seq0, startup)
        .as_ref()
        .map_or(0, |o| o.len());
    // Every ordinary Navarro presentation pairs its frame-sub records with one authenticated
    // report on the stream sub. Build presentation zero once here so the diagnostic length and
    // the bytes submitted below refer to the same live counter reservation. The prologue itself
    // has no report; presentation one will allocate its own inside the loop.
    let mut first_report = if startup {
        None
    } else {
        data.build_stream_report_buf(connector_index, seq0)?
    };
    let first_report_len = first_report.as_ref().map_or(0, |r| r.len());
    let first_wire_len = arm_len
        + first_opener_len
        + first_report_len
        + params.len()
        + image_len
        + data.build_frame_trailer(connector, seq0).len();
    let (rec_count, max_stride, max_strip) =
        crate::video::haar::record_stats(&frames[..frame_count]);
    vino_debug!(
        "vino: connector={} chunks={} arm={} first={} presentations={} records={} max_stride={} max_strip={}\n",
        connector,
        frame_count,
        arm_len,
        first_wire_len,
        repeat_count,
        rec_count,
        max_stride,
        max_strip
    );
    // Split at 65536-byte boundaries, a multiple of the endpoint's 1024-byte maximum packet size,
    // so only the final transfer terminates short. Submit through a persistent eight-deep queue to
    // keep the frame continuous across transfer boundaries. Do not flush between frames: slot reuse
    // reaps completions without introducing a pipeline gap.
    const XFER: usize = VIDEO_XFER;
    let pipe_i = dev.video_pipe_index(connector_index)?;
    let mut last_wire_len = 0usize;
    // Presentations that named a ring slot, which is what the frame counter counts. See
    // `names_ring_slot`.
    let mut named = 0u32;
    for repeat in 0..repeat_count {
        // Pace the copies apart on a dock whose control plane shares this endpoint. Back-to-back
        // presentations hold it for as long as it takes to push several megabytes, which is
        // exactly when the dock has to be able to answer, and it stops answering at all.
        if repeat > 0 && data.video_on_ctrl_pipe() {
            fsleep(Delta::from_millis(data.frame_period_ms()));
        }
        // A compositor mode change can arrive while the presentation is in flight. Never let the
        // old frame cross the new mode generation.
        if data.shutting_down.load(Ordering::Acquire)
            || data.modeset_requested[connector_index].load(Ordering::Acquire) != want
            || data.modeset_active[connector_index].load(Ordering::Acquire) != want
        {
            vino_debug!(
                "vino: scanout connector={} superseded during presentation; stopped at {}/{}\n",
                connector,
                repeat,
                repeat_count
            );
            return Ok(());
        }

        let repeat_seq = seq0.wrapping_add(named);
        let frame_trailer = data.build_frame_trailer(connector, repeat_seq);
        // Prefix ARM only to presentation zero. Every later presentation starts directly at the
        // image records and carries a freshly advanced three-record frame trailer.
        let arm_slice: &[u8] = if repeat == 0 {
            arm.as_ref().map_or(&[], |a| &a[..])
        } else {
            &[]
        };
        let prologue_frame = startup && repeat == 0;
        let frame_opener = data.build_frame_opener(connector, repeat_seq, prologue_frame);
        let opener_slice: &[u8] = frame_opener.as_ref().map_or(&[], |o| &o[..]);
        if super::names_ring_slot(opener_slice, &frame_trailer) {
            named = named.wrapping_add(1);
        }
        let report = if prologue_frame {
            None
        } else if repeat == 0 {
            first_report.take()
        } else {
            data.build_stream_report_buf(connector_index, repeat_seq)?
        };
        let report_slice: &[u8] = report.as_ref().map_or(&[], |r| &r[..]);
        let wire_len = arm_slice.len()
            + opener_slice.len()
            + report_slice.len()
            + params.len()
            + image_len
            + frame_trailer.len();
        last_wire_len = wire_len;
        {
            // Take this connector's staging buffer while submitting, then restore it. The queue
            // mutex stays locked for the complete frame: two connectors that address the same
            // physical endpoint must never submit their record streams concurrently.
            let mut staging = match data.video_staging.lock()[connector_index].take() {
                Some(s) => s,
                None => {
                    let mut s = KVec::new();
                    s.resize(XFER, 0, GFP_KERNEL)?;
                    s
                }
            };
            let submitted = {
                // One writer owns a shared pipe for the whole frame; see `own_pipe`.
                let _pipe = data.own_pipe();
                // A shared-pipe queue failure is terminal for its complete control/video session.
                // It may have happened while this worker waited behind its sibling, so recheck
                // under pipe ownership before recreating or touching the canonical queue.
                if data.video_on_ctrl_pipe() && !data.cp_link_alive() {
                    data.video_staging.lock()[connector_index] = Some(staging);
                    return Err(ENODEV);
                }
                let mut queue_slot = data.video_q[pipe_i].lock();
                if queue_slot.is_none() {
                    match dev.video_queue(connector_index, 8, XFER) {
                        Ok(q) => {
                            *queue_slot = Some(q);
                            vino_debug!(
                                "vino: connector={} endpoint={:#04x} persistent video queue opened (depth=8, {} B URBs)\n",
                                connector,
                                dev.endpoints.video[connector_index].address(),
                                XFER
                            );
                        }
                        // Nothing was taken from the shared pipe slot, but return this connector's
                        // staging allocation before propagating the open error.
                        Err(e) => {
                            data.video_staging.lock()[connector_index] = Some(staging);
                            return Err(e);
                        }
                    }
                }
                // Keep both borrows inside the mutex scope. Dropping the queue or unlocking it
                // mid-frame would permit a second connector to interleave URBs on this endpoint.
                let submit = |staging: &mut KVec<u8>, q: &mut crate::usb::BulkOutQueue| -> Result {
                    let staging = &mut staging[..];
                    let q = &mut *q;
                    // Scatter/gather cursor over
                    // [optional ARM][optional Navarro opener][authenticated stream report]
                    // [record chunks, with the parameter map among them][trailer]. Join only one
                    // transfer at a time in the reusable bounded staging allocation, avoiding a
                    // contiguous allocation spanning the complete frame.
                    let arm_parts = usize::from(!arm_slice.is_empty());
                    let opener_parts = usize::from(!opener_slice.is_empty());
                    let report_parts = usize::from(!report_slice.is_empty());
                    let param_parts = usize::from(!params.is_empty());
                    let trailer_parts = 1usize;
                    let opener_end = arm_parts + opener_parts;
                    let lead = opener_end + report_parts;
                    // The map describes the records around it and goes where the vendor puts it,
                    // part-way through them; see `param_map_chunk_split`. A dock with no map has
                    // no split to make.
                    let param_after = if param_parts == 0 {
                        frame_count
                    } else {
                        super::param_map_chunk_split(&frames[..frame_count])
                    };
                    let param_end = lead + param_after + param_parts;
                    let part_count = lead + frame_count + param_parts + trailer_parts;
                    let mut part_i = 0usize;
                    let split_tail = data.split_full_packet_frame();
                    let mut part_off = 0usize;
                    let mut wire_off = 0usize;
                    while wire_off < wire_len {
                        let mut data_len = (wire_len - wire_off).min(XFER);
                        // The dock delimits a frame by the short packet its last transfer ends on.
                        // A frame whose length is a whole number of maximum-size packets ends on a
                        // full one, delimits nothing, and is read as running into the next frame.
                        // Split one short tail off it so there is always a boundary; a record may
                        // span transfers, so where the split falls does not matter.
                        if split_tail
                            && data_len == wire_len - wire_off
                            && data_len % BULK_MAX_PACKET == 0
                        {
                            data_len -= FRAME_TAIL_BYTES;
                        }
                        let dst = &mut staging[..data_len];
                        let mut dst_off = 0usize;
                        while dst_off < dst.len() && part_i < part_count {
                            let part: &[u8] = if part_i < arm_parts {
                                arm_slice
                            } else if part_i < opener_end {
                                opener_slice
                            } else if part_i < lead {
                                report_slice
                            } else if part_i < lead + param_after {
                                &frames[part_i - lead][..]
                            } else if part_i < param_end {
                                &params[..]
                            } else if part_i < part_count - trailer_parts {
                                &frames[part_i - lead - param_parts][..]
                            } else {
                                &frame_trailer[..]
                            };
                            let n = (part.len() - part_off).min(dst.len() - dst_off);
                            dst[dst_off..dst_off + n]
                                .copy_from_slice(&part[part_off..part_off + n]);
                            dst_off += n;
                            part_off += n;
                            if part_off == part.len() {
                                part_i += 1;
                                part_off = 0;
                            }
                        }
                        if let Err(e) = q.send(dev.io(), dst, crate::timeout()) {
                            // The queue is eight URBs deep and `send` reaps a completion when its
                            // slot is reused, so this offset is where the error surfaced, not
                            // where the dock objected -- that was about eight transfers earlier.
                            // Report the frame's shape instead: a dock refuses a malformed record
                            // by halting the endpoint, which arrives here as a transport error and
                            // gets blamed on the transport.
                            let (records, max_stride, max_strip) =
                                crate::video::haar::record_stats(&frames[..frame_count]);
                            pr_warn!(
                                "vino: scanout connector={} pipeline submit at off={}/{} failed; frame has {} records, largest stride {}, largest strip {}\n",
                                connector,
                                wire_off,
                                wire_len,
                                records,
                                max_stride,
                                max_strip
                            );
                            return Err(e);
                        }
                        wire_off += data_len;
                    }
                    Ok(())
                };
                let submitted = {
                    let mut queue = queue_slot.as_mut().get_mut().as_mut().ok_or(ENODEV)?;
                    submit(&mut staging, &mut queue)
                };
                match submitted {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        // Stay inside the existing physical-pipe and canonical-queue guards.  The
                        // queue drop kills and drains every later URB before halt-clear, so neither
                        // this connector's unframed tail nor a sibling's new frame can race
                        // recovery.
                        let clear_halt = (e == EPIPE || e == EPROTO)
                            && data.scanout_fails[connector_index].load(Ordering::Relaxed)
                                < VIDEO_STALL_LIMIT;
                        if let Err(recovery) = data.retire_failed_video_queue(
                            dev,
                            connector_index,
                            &mut *queue_slot,
                            e,
                            clear_halt,
                        ) {
                            pr_err!(
                                "vino: connector {connector_index} video queue recovery failed ({recovery:?})\n"
                            );
                        }
                        Err(e)
                    }
                }
            };
            // Restore the per-connector staging allocation after the endpoint transaction.
            data.video_staging.lock()[connector_index] = Some(staging);
            submitted?;
        }

        // The ARM burst was delivered with presentation zero. Clear it immediately rather than
        // after the whole replay: if a later copy fails, retrying ARM would corrupt a pipe that is
        // already armed.
        if repeat == 0 && startup {
            data.arm_prefix_pending
                .fetch_and(!connector_bit, core::sync::atomic::Ordering::Release);
            // The cold-link requirement is measured from the start of continuous VIDEO, not from
            // the earlier mode-set. Refresh the complete training window here so modeset bracket
            // latency, cross-connector serialization, and encoder time cannot make it
            // intermittently too short. Subsequent cadence-selected compositor flips and idle
            // settle repaints are both promoted to full keyframes while this deadline is live.
            // Re-armed from the first frame the dock actually received, but only where
            // `sustain_window` granted one: a dock that shares its control pipe is granted none.
            {
                let mut sustain = data.sustain_until.lock();
                if sustain[connector_index].is_some() {
                    sustain[connector_index] =
                        Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
                }
            }
            vino_debug!(
                "vino: scanout connector={} initial ARM+keyframe accepted ({} B on the wire)\n",
                connector,
                wire_len
            );
            // The dock expects two stream-commit messages on EP02 immediately after accepting the
            // video arm burst.
            for _ in 0..2 {
                match data.send_cp(dev, 0x16, 0, |ctr| crate::cp::stream_commit(ctr, connector)) {
                    Ok(()) => vino_debug!("vino: stream-commit connector={} ok\n", connector),
                    Err(e) => pr_warn!(
                        "vino: stream-commit connector={} failed ({e:?})\n",
                        connector
                    ),
                }
            }
        }

        // A dedicated video endpoint can cheaply sample status after streaming.  On a shared
        // pipe, the long-lived keepalive already owns this same poll and DLM permits long runs of
        // pixels without an inline CP transaction.  Duplicating it here increases EP02 pressure
        // and blocks the scanout worker waiting for a reply from the endpoint it just filled.
        if !data.video_on_ctrl_pipe() {
            let due = {
                let mut last = data.last_status_poll.lock();
                let due = last.is_none_or(|t| t.elapsed().as_millis() >= STATUS_POLL_MIN_MS);
                if due {
                    *last = Some(Instant::<Monotonic>::now());
                }
                due
            };
            if due {
                if let Err(e) =
                    data.send_cp(dev, 0x14, 0, |ctr| crate::cp::device_query_req(ctr, 0x000c))
                {
                    vino_debug!(
                        "vino: scanout connector={} CP status poll failed ({e:?})\n",
                        connector
                    );
                }
            }
        }
        // Do not drain here. The eight-URB ring spans frame boundaries; `send()` reaps a
        // completion when its slot is reused, so transport errors surface after the ring wraps
        // without introducing a per-frame pipeline bubble.
    }
    // Publish the new codec sequence only after every URB for this frame was submitted. A stale
    // generation or transport failure above leaves the old sequence intact for the next keyframe.
    //
    data.scanout_seq.lock()[connector_index] = seq0.wrapping_add(named);
    // The USB path accepted the complete image. Publish its content shadow only now; every early
    // return and transport error above deliberately leaves the previous dock-visible state intact.
    // The frame reached the dock, so every strip it carried has paid one transmission. A full
    // keyframe is presented twice and rewrites the whole surface, so it clears the ledger outright.
    {
        let mut ttl = data.dirty_ttl.lock();
        if let Some(debt) = ttl[connector_index].as_mut() {
            pay_damage_debt(debt, full);
        }
    }
    // Publish the content shadow, and with it the encoded body of every strip this frame carried,
    // so the retransmissions `damage_frames` owes can re-use the bytes instead of re-running the
    // codec (see `StripHashState::bodies`). Bodies for strips this frame did not touch are carried
    // forward from the previous state -- they are still what the dock holds, and a later debt pass
    // may select them. Best-effort throughout: a failed allocation costs a cache miss, never a
    // frame, so the hashes are published either way.
    {
        let mut state = data.strip_hashes.lock();
        let carried = state[connector_index]
            .take()
            .filter(|c| {
                c.padded_width == padded_width
                    && c.padded_height == padded_height
                    && c.tag == encode_tag
            })
            .map(|c| (c.bodies, c.hashes));
        state[connector_index] = content_hashes.map(|hashes| {
            let (mut bodies, old) = match carried {
                Some((b, h)) => (b, Some(h)),
                None => (KVec::new(), None),
            };
            // A carried body is only still valid if that strip's content has not moved since it
            // was encoded. Every strip whose hash changes IS selected for this frame and so is
            // overwritten below -- but do not rely on that invariant holding as the selection
            // logic evolves: a body left paired with a newer hash would be served as a cache hit
            // and paint stale pixels the dock would then keep, with nothing scheduled to repair
            // it. Cheap to make airtight, and the failure it prevents is permanent corruption.
            if let Some(old) = &old {
                if old.len() == bodies.len() && old.len() == hashes.len() {
                    for i in 0..bodies.len() {
                        if old[i] != hashes[i] {
                            bodies[i] = KVec::new();
                        }
                    }
                }
            }
            if bodies.len() != hashes.len() {
                bodies = KVec::new();
                let _ = bodies.reserve(hashes.len(), GFP_KERNEL);
                while bodies.len() < hashes.len() && bodies.push(KVec::new(), GFP_KERNEL).is_ok() {}
            }
            if bodies.len() == hashes.len() {
                if let Some((coords, strips)) = encoded {
                    let tiles_x = padded_width >> geometry.strip_w_shift();
                    for (&(sx, sy), body) in coords.iter().zip(strips) {
                        let idx = (sy >> geometry.strip_h_shift()) * tiles_x
                            + (sx >> geometry.strip_w_shift());
                        if let Some(slot) = bodies.get_mut(idx) {
                            *slot = body;
                        }
                    }
                }
            }
            StripHashState {
                padded_width,
                padded_height,
                hashes,
                bodies,
                tag: encode_tag,
            }
        });
    }
    // A full keyframe was accepted -- this connector may now send damage deltas until the next
    // mode-set.
    if full {
        data.keyframe_pending
            .fetch_and(!kf_bit, core::sync::atomic::Ordering::Release);
    }

    // Charge every presentation, not the frame once: each one is a separate copy on the wire and
    // the dock decodes all of them.
    data.charge_stream_budget(last_wire_len.saturating_mul(repeat_count as usize));

    vino_debug!(
        "vino: scanout connector={} frame ok ({} presentation(s), {} B final write)\n",
        connector,
        repeat_count,
        last_wire_len
    );
    Ok(())
}

/// Convert the mapped XRGB8888 frame to RGB565, Vino-encode it against the previous frame,
/// and bulk-write the resulting EP08 frame to the dock.
pub(super) fn encode_and_send(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    connector: u8,
    src: &Arc<PixelSource>,
    rotation: plane::Rotation,
    // The client's changed rectangles (identity rotation only; empty means no pixel update).
    // `encode_and_send_haar` uses these to send a damage delta (only changed strips) after the
    // first full keyframe because the dock surface is undefined after a mode set.
    clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    // Non-64x16-aligned modes are padded to complete codec strips. The dock clips the padded image
    // to the active timing, matching the validated 68-band wire layout for 1080-line modes.
    encode_and_send_haar(dev, data, connector, src, rotation, clips, w, h)
}

// ---- Encoder ----------------------------------------------------------------

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
#[kunit_tests(vino_scanout)]
mod tests {
    use super::*;
    use crate::*;

    /// An eight-bit desktop driven over a ten-bit link must keep both endpoints exact.
    ///
    /// A plain shift leaves full white at 1020 of 1023, which tints every highlight and is the
    /// kind of error that looks like a panel problem rather than an encoder one.
    #[test]
    fn widening_an_eight_bit_sample_keeps_the_endpoints() {
        assert_eq!(widen_8_to_10(0), 0);
        assert_eq!(widen_8_to_10(255), 1023);
        // Mid grey stays mid grey rather than drifting low.
        assert_eq!(widen_8_to_10(128), 514);
        // Monotonic, and never outside the ten-bit range.
        let mut previous = 0;
        for v in 0..=255u16 {
            let w = widen_8_to_10(v);
            assert!(w <= 1023);
            assert!(v == 0 || w > previous);
            previous = w;
        }
    }

    /// Scattered damage must cost the strips it touches, not the surface.
    ///
    /// The rectangle list is an intermediate representation between the per-strip hash comparison
    /// and the strip coordinates the encoder is given, and it has a ceiling. Built on the strip
    /// grid, an update spread across the screen fragments into more runs than that ceiling holds
    /// and the whole surface is sent instead -- roughly thirty times the bytes, on a dock that
    /// halts its endpoint when it is given too many. Built on the macro-tile grid, which is the
    /// granularity a touched strip is resent at anyway, the same update stays inside it.
    #[test]
    fn scattered_damage_does_not_cost_the_whole_surface() -> Result {
        let geometry = profile::PROFILE_ELLA.geometry();
        // 1920x1088, this dock's whole top mode padded to strips: 30 x 68 = 2040 strips.
        let (padded_width, padded_height) = (1920usize, 1088usize);
        let tiles_x = padded_width >> geometry.strip_w_shift();
        let tiles_y = padded_height >> geometry.strip_h_shift();
        assert_eq!(tiles_x * tiles_y, 2040);

        let mut old: KVVec<u64> = KVVec::new();
        old.resize(tiles_x * tiles_y, 0, GFP_KERNEL)?;
        let mut new: KVVec<u64> = KVVec::new();
        new.resize(tiles_x * tiles_y, 0, GFP_KERNEL)?;

        // Nothing moved: no rectangles, and the caller skips the write entirely.
        assert!(changed_strip_rects(geometry, &old, &new, padded_width, padded_height)?.is_empty());

        // Every other strip on every other band: 510 changed strips, and 510 separate runs on the
        // strip grid -- four times the ceiling. On the macro-tile grid they collapse into whole
        // rows of tiles.
        let mut changed = 0usize;
        for ty in (0..tiles_y).step_by(2) {
            for tx in (0..tiles_x).step_by(2) {
                new[ty * tiles_x + tx] = 1;
                changed += 1;
            }
        }
        assert_eq!(changed, 510);
        let rects = changed_strip_rects(geometry, &old, &new, padded_width, padded_height)?;
        assert!(!rects.is_empty());
        let selected =
            video::haar::damage_strip_coords(geometry, padded_width, padded_height, &rects)?.len();
        // This pattern really does reach into every macro-tile, so the surface is the right
        // answer; what matters is that it was reached by rounding and not by giving up.
        assert_eq!(selected, 2040);

        // A realistic update instead: a window redraw, a cursor and a clock. Three regions, and
        // the cost is the macro-tiles they cover.
        for h in new.iter_mut() {
            *h = 0;
        }
        for (x0, y0, x1, y1) in [
            (4usize, 8usize, 14usize, 30usize),
            (20, 40, 22, 42),
            (26, 2, 29, 4),
        ] {
            for ty in y0..y1 {
                for tx in x0..x1 {
                    new[ty * tiles_x + tx] = 1;
                }
            }
        }
        let rects = changed_strip_rects(geometry, &old, &new, padded_width, padded_height)?;
        // Every rectangle is macro-tile aligned, so the selector's own rounding adds nothing.
        // Aligned on every edge the grid has one, and clamped to the surface where it does not:
        // 1920 is seven whole macro-tiles and half of an eighth.
        for &(x0, y0, x1, y1) in rects.iter() {
            assert_eq!(x0 % geometry.macro_w(), 0);
            assert_eq!(y0 % geometry.macro_h(), 0);
            assert!(x1 == padded_width || x1 % geometry.macro_w() == 0);
            assert!(y1 == padded_height || y1 % geometry.macro_h() == 0);
        }
        let selected =
            video::haar::damage_strip_coords(geometry, padded_width, padded_height, &rects)?.len();
        assert!(selected < 2040 / 2);
        Ok(())
    }

    #[test]
    fn parallel_encoder_matches_serial_for_every_plane_transform() -> Result {
        use drm::kms::plane::Rotation;

        let transforms = [
            Rotation::ROTATE_0,
            Rotation::ROTATE_90,
            Rotation::ROTATE_180,
            Rotation::ROTATE_270,
            Rotation::ROTATE_0 | Rotation::REFLECT_X,
            Rotation::ROTATE_90 | Rotation::REFLECT_X,
            Rotation::ROTATE_180 | Rotation::REFLECT_X,
            Rotation::ROTATE_270 | Rotation::REFLECT_X,
            Rotation::ROTATE_0 | Rotation::REFLECT_Y,
            Rotation::ROTATE_90 | Rotation::REFLECT_Y,
            Rotation::ROTATE_180 | Rotation::REFLECT_Y,
            Rotation::ROTATE_270 | Rotation::REFLECT_Y,
            Rotation::ROTATE_0 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
            Rotation::ROTATE_90 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
            Rotation::ROTATE_180 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
            Rotation::ROTATE_270 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
        ];
        for transform in transforms {
            parallel_rotation_matches_serial(transform)?;
        }
        Ok(())
    }
}
