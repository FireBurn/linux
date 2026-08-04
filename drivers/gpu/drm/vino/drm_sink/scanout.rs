// SPDX-License-Identifier: GPL-2.0

//! Turning a committed framebuffer into bytes on a video endpoint.
//!
//! The compositor's atomic callback does no work beyond snapshotting: everything below runs on the
//! deferred scanout worker. In order, a frame goes through damage selection against the previous
//! frame's per-strip hashes, encoding (fanned across CPUs as [`EncodeChunk`]s), record framing, and
//! submission through the head's persistent URB queue.

use super::*;
use super::mode_objects::rot_src;

/// Compress and submit one coalesced primary-plane flip on the deferred worker. Keeping all slow
/// work here makes the DRM atomic callback bounded to state inspection plus an `ARef` increment.
pub(super) fn run_pending_scanout(dev: &BoundInterface<'_>, data: &VinoDrmData, frame: PendingScanout) {
    use core::sync::atomic::Ordering::Relaxed;

    let head_i = frame.head as usize;
    if data.modeset_requested[head_i].load(Ordering::Acquire) == 0 {
        scanout_gate(frame.head, "worker: head has no mode-set requested");
        return;
    }
    let requested_geometry_matches = data.last_timing.lock()[head_i]
        .is_some_and(|t| t.hactive as usize == frame.w && t.vactive as usize == frame.h);
    if !requested_geometry_matches {
        // stale framebuffer from a different-size mode generation
        scanout_gate(
            frame.head,
            "worker: framebuffer size differs from the cached mode",
        );
        return;
    }
    // Was this the mode-set's owed keyframe? Read before sending, since a successful send clears
    // the bit.
    let was_keyframe = data.keyframe_pending.load(Ordering::Acquire) & (1u32 << frame.head) != 0;
    let settle_copy = was_keyframe.then(|| frame.clone());
    let slot = frame.shadow_idx;
    let generation = frame.shadow_generation;
    let (source_w, source_h) = src_dims(frame.rotation, frame.w, frame.h);
    let shadow = {
        let mut pool = data.shadow[head_i].lock();
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
            scanout_gate(frame.head, "slot busy: another encode inflight");
        } else if !gen_ok {
            scanout_gate(frame.head, "slot generation moved under us");
        } else if !dims_ok {
            scanout_gate(frame.head, "slot surface missing or wrong size");
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
            frame.head,
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
    let color = data.color_snapshot(head_i);
    let direct = direct_pixel_map(
        frame.rotation,
        &color,
        source_w,
        source_h,
        frame.w,
        frame.h,
    );
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
        },
        GFP_KERNEL,
    ) {
        Ok(src) => src,
        Err(_) => {
            let mut pool = data.shadow[head_i].lock();
            if pool.inflight == Some(slot) {
                pool.inflight = None;
            }
            scanout_gate(frame.head, "worker: pixel source allocation failed");
            return;
        }
    };
    let result = encode_and_send(
        dev,
        data,
        frame.head,
        &src,
        frame.rotation,
        &frame.clips[..frame.nclips],
        frame.w,
        frame.h,
    );
    data.last_frame.lock()[head_i] = Some(Instant::<Monotonic>::now());
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
        let mut pool = data.shadow[head_i].lock();
        if pool.inflight == Some(slot) {
            if pool.slots[slot].generation == generation
                && pool.slots[slot].surface.is_none()
            {
                pool.slots[slot].surface = returned;
            }
            pool.inflight = None;
        }
    }
    match result {
        Ok(()) => {
            let n = data.scanout_fails[head_i].swap(0, Relaxed);
            data.scanout_skip[head_i].store(0, Relaxed);
            if n > 0 {
                pr_info!("vino: head {head_i} scanout recovered after {n} failed frame(s)\n");
            }
            // Arm the one-shot settle repaint. A compositor that goes idle right after enabling an
            // output can otherwise remain on the initial keyframe indefinitely.
            if let Some(mut copy) = settle_copy {
                copy.clips[0] = (0, 0, copy.w, copy.h);
                copy.nclips = 1;
                // During the post-mode-set training window, repaint at frame cadence so the dock
                // receives the sustained stream needed to program the downstream pixel clock.
                // Outside that window, use the bounded settle repaint.
                let sustaining = data.sustain_until.lock()[head_i]
                    .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
                // Training repaints at the fast cadence and is exempt from the budget; everything
                // else charges one settle repaint against this
                // head's keyframe obligation and stops when it runs out. See `SETTLE_REPAINTS` for
                // the unbounded keyframe loop that made a static desktop stream ~2.7 MB/s per head.
                // A dock that tears the link down over a silent video endpoint has to be re-fed
                // whether or not anything changed, so its repaint is periodic and unbudgeted.
                let keepalive = data.is_navarro();
                let unbudgeted = sustaining || keepalive;
                let charged = unbudgeted
                    || data.settle_budget[head_i]
                        .fetch_update(Relaxed, Relaxed, |b| b.checked_sub(1))
                        .is_ok();
                if charged {
                    let delay = if sustaining {
                        FRAME_PERIOD_MS
                    } else if keepalive {
                        NAVARRO_KEEPALIVE_MS
                    } else {
                        SETTLE_REPAINT_MS
                    };
                    data.settle_repaint.lock()[head_i] = Some((
                        Instant::<Monotonic>::now() + Delta::from_millis(delay),
                        copy,
                        true,
                    ));
                }
            } else {
                // Repaint the same framebuffer while strips still have retransmit debt. This is a
                // delta, not a keyframe, and terminates after at most `damage_repeats()` accepted
                // presentations because each pass decrements the debt ledger.
                let owes = data.dirty_ttl.lock()[head_i]
                    .as_ref()
                    .is_some_and(|debt| debt.iter().any(|&d| d > 0));
                if owes {
                    data.settle_repaint.lock()[head_i] = Some((
                        Instant::<Monotonic>::now() + Delta::from_millis(FRAME_PERIOD_MS),
                        frame.clone(),
                        false,
                    ));
                }
            }
        }
        Err(e) => {
            // Log at exponentially sparser points and back off future worker attempts. An error is
            // transport state, not a reason to stall the compositor's pageflip path.
            let n = data.scanout_fails[head_i].fetch_add(1, Relaxed) + 1;
            if n == 1 || n.is_power_of_two() {
                pr_err!("vino: head {head_i} scanout frame failed ({e:?}) [x{n}] -- throttling\n");
            }
            data.scanout_skip[head_i].store(core::cmp::min(n, 120), Relaxed);
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

/// Copy a committed framebuffer into this head's [`ShadowSurface`], reusing the existing allocation
/// whenever the geometry is unchanged.
///
/// Runs in the atomic commit path, so everything else (damage selection, rotation, gamma, the
/// codec) stays in the worker and reads this private surface instead of the compositor's live
/// buffer.
///
/// The traversal is **band-major**: for each row of strips, the source's rows are pulled into
/// [`ShadowSurface::band`] a full row per read, and only then are that band's strips hashed and --
/// where the hash moved -- copied on into `pixels`. The obvious strip-major order costs far more
/// for the same result, because a strip is 64 px wide but the source row is `pitch` bytes apart:
/// it reads the source in `STRIP_W * 4`-byte fragments (57,600 of them per 1440p frame, versus
/// 1,440 full-row reads here), it walks those fragments against the row stride rather than
/// sequentially, and it has to read a changed strip out of the source a second time to copy it,
/// because the first read went to a fragment-sized scratch that could not be kept.
///
/// Strips whose hash is unchanged are still not written, so an idle desktop moves no more memory
/// than before; a busy one no longer reads the source twice.
#[inline(never)]
pub(super) fn snapshot_to_shadow(
    geom: crate::video::wht::Geometry,
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

    let (sw, sh) = (geom.strip_w(), geom.strip_h());
    let w_pad = (w + sw - 1) & !(sw - 1);
    let h_pad = (h + sh - 1) & !(sh - 1);
    let tiles_x = w_pad / sw;
    let tiles_y = h_pad / sh;

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

/// Convert changed strip hashes into a compact set of already tile-aligned damage rectangles.
/// Horizontal runs are joined, then equal runs on adjacent bands are extended vertically. A very
/// fragmented frame falls back to one full-output rectangle rather than growing an unbounded
/// allocation or spending more time testing rectangles than encoding strips.
#[inline(never)]
fn changed_strip_rects(
    geom: crate::video::wht::Geometry,
    old: &[u64],
    new: &[u64],
    w_pad: usize,
    h_pad: usize,
) -> Result<KVec<DamageRect>> {
    const MAX_RECTS: usize = 128;
    let tiles_x = w_pad >> geom.strip_w_shift();
    let tiles_y = h_pad >> geom.strip_h_shift();
    if old.len() != tiles_x * tiles_y || new.len() != old.len() {
        return Err(EINVAL);
    }
    let mut rects: KVec<DamageRect> = KVec::new();
    for ty in 0..tiles_y {
        let mut tx = 0usize;
        while tx < tiles_x {
            if old[ty * tiles_x + tx] == new[ty * tiles_x + tx] {
                tx += 1;
                continue;
            }
            let run_start = tx;
            while tx < tiles_x && old[ty * tiles_x + tx] != new[ty * tiles_x + tx] {
                tx += 1;
            }
            let x0 = run_start * geom.strip_w();
            let x1 = tx * geom.strip_w();
            let y0 = ty * geom.strip_h();
            let y1 = y0 + geom.strip_h();
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
                    full.push((0, 0, w_pad, h_pad), GFP_KERNEL)?;
                    return Ok(full);
                }
                rects.push((x0, y0, x1, y1), GFP_KERNEL)?;
            }
        }
    }
    Ok(rects)
}

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

impl PixelSource {
    /// Read one gamma-corrected pixel in untransformed framebuffer coordinates.
    #[inline]
    fn source_px(&self, sx: usize, sy: usize) -> (u8, u8, u8) {
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
        let p = u32::from_le_bytes(bytes);
        let (r, g, b) = (
            ((p >> 16) & 0xff) as u8,
            ((p >> 8) & 0xff) as u8,
            (p & 0xff) as u8,
        );
        match &self.color {
            Some(pipeline) => pipeline.apply(r, g, b),
            None => (r, g, b),
        }
    }

    /// Read one output pixel after applying the plane transform.
    ///
    /// Keeping the transform in the immutable shared source gives serial and parallel encoding
    /// exactly the same sampler. Codec padding is black and never reads beyond the snapshot.
    #[inline]
    fn px(&self, dx: usize, dy: usize) -> (u8, u8, u8) {
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
            // Little-endian XRGB8888: byte 0 is blue, 1 green, 2 red.
            return (chunk[2], chunk[1], chunk[0]);
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
/// The encode ran on `system_unbound`, where its CPU time is anonymous: the kernel composes worker
/// thread names from the workqueue's, so shared-pool work appears only as
/// `kworker/uN:M-events_unbound`, indistinguishable from every other user of that pool. On a queue
/// of our own the same threads appear as **`kworker/uN:M-vino_encode`**, so `ps`/`top`/`perf`
/// attribute the codec's cost to vino, and the fan-out no longer competes with unrelated work for
/// the shared pool's concurrency budget.
///
/// `WQ_UNBOUND` because strip encoding is pure compute with no CPU affinity worth preserving --
/// that property is what let the fan-out reach ~7.4x. Allocated once on first use and never
/// destroyed: it is driver-wide, costs one `workqueue_struct`, and outliving every `EncodeChunk` is
/// exactly what makes the join safe.
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
        .cpu_intensive()
        .build(kernel::c_str!("vino_encode"))
    {
        let _ = ENCODE_WQ.populate(q);
    }
    ENCODE_WQ.as_ref().map(|q| &**q)
}

/// Encode a batch of strips from `src`, in the order given.
fn encode_coords(
    geom: crate::video::wht::Geometry,
    src: &PixelSource,
    coords: &[(usize, usize)],
) -> Result<KVec<KVec<u8>>> {
    let mut out = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
    for &(sx, sy) in coords.iter() {
        let mut px = |dx, dy| src.px(dx, dy);
        out.push(
            crate::video::wht::colour_strip_at(geom, sx, sy, &mut px)?,
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
    geom: crate::video::wht::Geometry,
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
        geom: crate::video::wht::Geometry,
        src: Arc<PixelSource>,
        coords: KVec<(usize, usize)>,
    ) -> Result<Arc<Self>> {
        Arc::pin_init(
            pin_init!(EncodeChunk {
                work <- new_work!("vino::EncodeChunk::work"),
                done <- Completion::new(),
                src,
                coords,
                geom,
                out <- new_mutex!(KVec::new(), "vino::EncodeChunk::out"),
            }),
            GFP_KERNEL,
        )
    }
}

impl WorkItem for EncodeChunk {
    type Pointer = Arc<EncodeChunk>;

    fn run(this: Arc<EncodeChunk>) {
        if let Ok(strips) = encode_coords(this.geom, &this.src, &this.coords) {
            *this.out.lock() = strips;
        }
        // Complete unconditionally. On failure `out` stays short and the joiner detects that by
        // length -- but it must never be left blocked on a completion that cannot fire.
        this.done.complete_all();
    }
}

/// Encode `coords` across CPUs and return the strip bodies in the **same** order.
///
/// Order is not a nicety: [`crate::video::wht::frame_records`] groups strips into one wire record
/// per single-Y band and needs them x-ordered within a band, so the chunks are contiguous slices
/// of the raster-ordered coordinate list and are reassembled strictly in chunk order.
///
/// Returns `Ok(None)` when the frame is too small to be worth splitting, so the caller falls
/// through to the serial encoder rather than paying dispatch cost for a handful of strips.
fn parallel_strip_encode(
    geom: crate::video::wht::Geometry,
    src: &Arc<PixelSource>,
    coords: &[(usize, usize)],
) -> Result<Option<KVec<KVec<u8>>>> {
    // Size chunks from the amount of work. The unbound workqueue controls how many execute in
    // parallel, so the encoder does not need to model the host CPU topology.
    //
    // This is not thread oversubscription: these are work items, and `system_unbound()` decides how
    // many run at once. Handing it more, smaller items than there are CPUs costs a little dispatch
    // overhead but improves load balancing -- with one chunk per CPU a single slow chunk holds up
    // the whole join, whereas fine-grained items let idle workers pick up the remainder.
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
        let chunk = EncodeChunk::new(geom, src.clone(), mine)?;
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
            encode_coords(chunk.geom, &chunk.src, &chunk.coords)?
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
pub(super) fn parallel_rotation_matches_serial(rotation: plane::Rotation) -> Result {
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
        },
        GFP_KERNEL,
    )?;
    let geom = crate::video::wht::RIDGE_GEOMETRY;
    let w_pad = output_w.next_multiple_of(geom.strip_w());
    let h_pad = output_h.next_multiple_of(geom.strip_h());
    let coords = crate::video::wht::all_strip_coords(geom, w_pad, h_pad)?;

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
            crate::video::wht::colour_strip_at(geom, strip_x, strip_y, &mut px)?,
            GFP_KERNEL,
        )?;
    }

    let parallel = parallel_strip_encode(geom, &src, &coords)?.ok_or(EINVAL)?;
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
fn scanout_gate(head: u8, reason: &str) {
    vino_debug!("vino: scanout head={head} deferred: {reason}\n");
}

#[inline(never)]
fn encode_and_send_wht(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    head: u8,
    src: &Arc<PixelSource>,
    rotation: plane::Rotation,
    _clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    let geom = data.geometry();
    // Gate video on the matching mode-set reaching the dock. Plane updates run before the CRTC
    // enable queues that mode-set, and the dock rejects video on an unconfigured stream. Deferring
    // does not advance the codec sequence; the next scanout pass retries the frame.
    let head_i = head as usize;
    let want = data.modeset_requested[head_i].load(core::sync::atomic::Ordering::Acquire);
    if want == 0 {
        scanout_gate(head, "no mode-set requested (modeset_requested == 0)");
        return Ok(());
    }
    let cached = data.last_timing.lock()[head_i];
    if !cached.is_some_and(|t| {
        timing_key(&t) == want && t.hactive as usize == w && t.vactive as usize == h
    }) {
        scanout_gate(
            head,
            "cached timing does not match the requested mode generation",
        );
        return Ok(());
    }
    if data.modeset_active[head_i].load(core::sync::atomic::Ordering::Acquire) != want {
        // A failed command-worker activation leaves the desired generation intact. This worker is
        // sleepable, so retry the same transaction before submitting its pending framebuffer.
        let timing = cached.ok_or(EINVAL)?;
        data.activate_head(dev, head, &timing, want)?;
        // A successful inline retry has made this very commit safe to send: continue into the
        // encoder instead of waiting for another page flip.  A completely static head may not
        // receive another atomic update after its enabling commit.
        if data.modeset_active[head_i].load(core::sync::atomic::Ordering::Acquire) != want {
            scanout_gate(
                head,
                "mode-set not active and the inline re-send did not land",
            );
            return Ok(());
        }
    }
    let seq0 = data.scanout_seq.lock()[head_i];
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
    let kf_bit = 1u32 << head_i;
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
    let w_pad = (w + geom.strip_w() - 1) & !(geom.strip_w() - 1);
    let h_pad = (h + geom.strip_h() - 1) & !(geom.strip_h() - 1);
    let mut content_hashes: Option<KVVec<u64>> = None;
    let mut content_damage: KVec<DamageRect> = KVec::new();
    if identity {
        let expected = (w_pad >> geom.strip_w_shift()) * (h_pad >> geom.strip_h_shift());
        if src.hashes.len() != expected {
            return Err(EINVAL);
        }
        let mut hashes: KVVec<u64> = KVVec::new();
        hashes.resize(expected, 0, GFP_KERNEL)?;
        hashes.copy_from_slice(&src.hashes);
        if !full {
            let previous = data.strip_hashes.lock();
            if let Some(state) = &previous[head_i] {
                if state.w_pad == w_pad && state.h_pad == h_pad {
                    // Charge every strip whose content moved with a fresh debt, then select every
                    // strip that still owes a transmission -- including ones that changed on an
                    // earlier frame and have not yet reached both dock buffers. See `dirty_ttl`.
                    let mut ttl = data.dirty_ttl.lock();
                    if !ttl[head_i]
                        .as_ref()
                        .is_some_and(|t| t.len() == hashes.len())
                    {
                        let mut fresh: KVVec<u8> = KVVec::new();
                        fresh.resize(hashes.len(), 0, GFP_KERNEL)?;
                        ttl[head_i] = Some(fresh);
                    }
                    let debt = ttl[head_i].as_mut().ok_or(kernel::error::code::ENOMEM)?;
                    for i in 0..hashes.len() {
                        if state.hashes[i] != hashes[i] {
                            debt[i] = damage_repeats(geom);
                        }
                    }
                    // Reuse the hash differ: mark an owed strip by handing it a baseline value that
                    // cannot match, and an unowed one its own value.
                    let mut baseline: KVVec<u64> = KVVec::new();
                    baseline.resize(hashes.len(), 0, GFP_KERNEL)?;
                    for i in 0..hashes.len() {
                        baseline[i] = if debt[i] > 0 { !hashes[i] } else { hashes[i] };
                    }
                    content_damage = changed_strip_rects(geom, &baseline, &hashes, w_pad, h_pad)?;
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
        scanout_gate(head, "no keyframe owed and no strip content changed");
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
    let gamma_tag = match &src.color {
        Some(pipeline) => pipeline.tag(),
        None => 0,
    };
    // Strips carried over verbatim from the previous frame's encode, and the strips actually
    // handed to the codec; kept for the post-send cache publish below.
    let mut encoded: Option<(KVec<(usize, usize)>, KVec<KVec<u8>>)> = None;
    let parallel = if !identity {
        let coords = crate::video::wht::all_strip_coords(geom, w_pad, h_pad)?;
        match parallel_strip_encode(geom, src, &coords)? {
            Some(strips) => {
                let records = if geom.head_sub_shift != 0 {
                    crate::video::wht::frame_records_navarro_ordinary(geom, &strips, head)?
                } else {
                    crate::video::wht::frame_records(geom, &strips, head)?
                };
                Some((records, seq0.wrapping_add(1)))
            }
            None => None,
        }
    } else {
        let coords = if full {
            crate::video::wht::all_strip_coords(geom, w_pad, h_pad)?
        } else {
            crate::video::wht::damage_strip_coords(geom, w_pad, h_pad, &content_damage)?
        };
        // Reuse an encoded strip body when its pixels and gamma tag are unchanged. Encode only
        // misses, then restore the required x-order within each Y band.
        let tiles_x = w_pad >> geom.strip_w_shift();
        let mut reuse: KVec<Option<KVec<u8>>> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
        let mut misses: KVec<(usize, usize)> = KVec::with_capacity(coords.len(), GFP_KERNEL)?;
        {
            let cache = data.strip_hashes.lock();
            let usable = cache[head_i]
                .as_ref()
                .filter(|c| c.w_pad == w_pad && c.h_pad == h_pad && c.tag == gamma_tag);
            for &(sx, sy) in coords.iter() {
                let idx = (sy >> geom.strip_h_shift()) * tiles_x + (sx >> geom.strip_w_shift());
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
        let fresh = match parallel_strip_encode(geom, src, &misses)? {
            Some(s) => Some(s),
            // Too few misses to be worth splitting: encode them here rather than dropping to
            // the whole-frame serial path, which would re-encode the cache hits as well.
            None if !misses.is_empty() => Some(encode_coords(geom, src, &misses)?),
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
                let records = if full && geom.head_sub_shift != 0 {
                    crate::video::wht::frame_records_navarro_ordinary(geom, &strips, head)?
                } else {
                    crate::video::wht::frame_records(geom, &strips, head)?
                };
                encoded = Some((coords, strips));
                Some((records, seq0.wrapping_add(1)))
            }
            _ => None,
        }
    };
    let (frames, next_seq) = match parallel {
        Some(r) => r,
        None if full && geom.head_sub_shift != 0 => {
            crate::video::wht::colour_frame_ep08_navarro_ordinary(
                geom, w_pad, h_pad, seq0, head, px,
            )?
        }
        None if full => {
            crate::video::wht::colour_frame_ep08(geom, w_pad, h_pad, seq0, head, px)?
        }
        None => crate::video::wht::colour_frame_ep08_damage(
            geom,
            w_pad,
            h_pad,
            seq0,
            head,
            &content_damage,
            px,
        )?,
    };
    // A damage delta that touched no aligned strip = nothing to send this flip: skip the write
    // (no seq advance, no arm, keyframe obligation untouched). Full frames always have strips.
    if frames.is_empty() {
        scanout_gate(head, "encoder produced zero records");
        return Ok(());
    }
    if data.shutting_down.load(Ordering::Acquire)
        || data.modeset_requested[head_i].load(Ordering::Acquire) != want
        || data.modeset_active[head_i].load(Ordering::Acquire) != want
    {
        scanout_gate(head, "mode generation changed between encode and submit");
        return Ok(());
    }
    // A frame is one continuous bulk stream: intermediate transfers end on a full 1024-byte packet
    // and only the final transfer is short. A short packet at a record boundary terminates the
    // frame early and desynchronises the dock. The first frame after a mode set also prepends the
    // head's ten-record arm burst; clear that obligation only after a successful submission.
    let head_bit = 1u32 << head;
    // The 2560-byte arm burst appears only on frame zero after a mode set. Later frames begin
    // directly with video records.
    let arm = if data
        .arm_prefix_pending
        .load(core::sync::atomic::Ordering::Acquire)
        & head_bit
        != 0
    {
        Some(data.build_stream_prefix_buf(head_i)?)
    } else {
        None
    };
    let arm_len = arm.as_ref().map_or(0, |a| a.len());
    // Revalidate at the actual wire boundary too. The encoded bytes and ARM prefix are specific to
    // this mode generation; submitting them after a concurrent disable/re-enable poisons the next
    // stream even though every USB URB can still complete successfully.
    if data.shutting_down.load(Ordering::Acquire)
        || data.modeset_requested[head_i].load(Ordering::Acquire) != want
        || data.modeset_active[head_i].load(Ordering::Acquire) != want
    {
        vino_debug!(
            "vino: scanout head={} superseded before video submit; frame discarded\n",
            head
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
            "vino: inline pre-write paced poll ({}x @{}ms) before first video head={}\n",
            VinoDrmData::PREWRITE_POLLS,
            VinoDrmData::PREWRITE_POLL_MS,
            head
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
    let params: KVec<u8> = if geom.head_sub_shift == 0 {
        KVec::new()
    } else {
        crate::video::wht::navarro_strip_params(
            geom,
            head,
            w_pad,
            h_pad,
            &frames,
            &mut data.strip_classes.lock()[head_i],
        )?
    };
    vino_debug!(
        "vino: head={} shift={} params={} B ({}x{} pad)\n",
        head,
        geom.head_sub_shift,
        params.len(),
        w_pad,
        h_pad
    );
    if arm.is_some() {
        data.send_stream_open(dev, head_i)?;
    }
    let startup = arm.is_some();
    // A cold link requires a bounded back-to-back full-frame burst until the downstream clock is
    // programmed. Reuse the encoded image and advance only its frame trailer and per-frame control
    // sync. The arm prefix remains exclusive to presentation zero.
    let training = full
        && data.sustain_until.lock()[head_i]
            .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
    let repeat_count = if training {
        COLD_TRAINING_PRESENTATIONS
    } else if full {
        // A keyframe must reach EVERY dock buffer. One presentation updates only one of them,
        // while later deltas repair only their selected regions -- and the ledger is cleared
        // outright below on the strength of this, so a keyframe that comes up short leaves stale
        // pixels nothing will ever repair.
        u32::from(geom.dock_buffers)
    } else {
        // Consecutive copies land in the same dock buffer. Spread delta retransmissions across
        // successive frames through `damage_repeats` and the debt repaint instead.
        1
    };
    let first_opener_len = data
        .build_frame_opener(head, seq0, startup)
        .as_ref()
        .map_or(0, |o| o.len());
    // Every ordinary Navarro presentation pairs its frame-sub records with one authenticated
    // report on the stream sub. Build presentation zero once here so the diagnostic length and
    // the bytes submitted below refer to the same live counter reservation. The prologue itself
    // has no report; presentation one will allocate its own inside the loop.
    let mut first_report = if startup {
        None
    } else {
        data.build_stream_report_buf(head_i)?
    };
    let first_report_len = first_report.as_ref().map_or(0, |r| r.len());
    let first_wire_len = arm_len
        + first_opener_len
        + first_report_len
        + params.len()
        + image_len
        + data.build_frame_trailer(head, seq0).len();
    vino_debug!(
        "vino: head={} chunks={} arm={} first={} presentations={}\n",
        head,
        frame_count,
        arm_len,
        first_wire_len,
        repeat_count
    );
    // Split at 65536-byte boundaries, a multiple of the endpoint's 1024-byte maximum packet size,
    // so only the final transfer terminates short. Submit through a persistent eight-deep queue to
    // keep the frame continuous across transfer boundaries. Do not flush between frames: slot reuse
    // reaps completions without introducing a pipeline gap.
    const XFER: usize = VIDEO_XFER;
    let pipe_i = dev.video_pipe_index(head_i)?;
    let mut last_wire_len = 0usize;
    for repeat in 0..repeat_count {
        // A compositor mode change can arrive while the presentation is in flight. Never let the
        // old frame cross the new mode generation.
        if data.shutting_down.load(Ordering::Acquire)
            || data.modeset_requested[head_i].load(Ordering::Acquire) != want
            || data.modeset_active[head_i].load(Ordering::Acquire) != want
        {
            vino_debug!(
                "vino: scanout head={} superseded during presentation; stopped at {}/{}\n",
                head,
                repeat,
                repeat_count
            );
            return Ok(());
        }

        let repeat_seq = seq0.wrapping_add(repeat);
        let frame_trailer = data.build_frame_trailer(head, repeat_seq);
        // Prefix ARM only to presentation zero. Every later presentation starts directly at the
        // image records and carries a freshly advanced three-record frame trailer.
        let arm_slice: &[u8] = if repeat == 0 {
            arm.as_ref().map_or(&[], |a| &a[..])
        } else {
            &[]
        };
        let frame_opener = data.build_frame_opener(head, repeat_seq, !arm_slice.is_empty());
        let opener_slice: &[u8] = frame_opener.as_ref().map_or(&[], |o| &o[..]);
        let report = if arm_slice.is_empty() {
            if repeat == 0 {
                first_report.take()
            } else {
                data.build_stream_report_buf(head_i)?
            }
        } else {
            None
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
            // Take this head's staging buffer while submitting, then restore it. The queue mutex
            // stays locked for the complete frame: two connectors that address the same physical
            // endpoint must never submit their record streams concurrently.
            let mut staging = match data.video_staging.lock()[head_i].take() {
                Some(s) => s,
                None => {
                    let mut s = KVec::new();
                    s.resize(XFER, 0, GFP_KERNEL)?;
                    s
                }
            };
            let submitted = {
                let mut queue_slot = data.video_q[pipe_i].lock();
                if queue_slot.is_none() {
                    if *crate::module_parameters::video_clear_halt.value() != 0 {
                        let _ = dev.clear_video_halt(head_i);
                    }
                    match dev.video_queue(head_i, 8, XFER) {
                        Ok(q) => {
                            *queue_slot = Some(q);
                            vino_debug!(
                                "vino: head={} endpoint={:#04x} persistent video queue opened (depth=8, {} B URBs)\n",
                                head,
                                dev.eps.video[head_i].address(),
                                XFER
                            );
                        }
                        // Nothing was taken from the shared pipe slot, but return this head's
                        // staging allocation before propagating the open error.
                        Err(e) => {
                            data.video_staging.lock()[head_i] = Some(staging);
                            return Err(e);
                        }
                    }
                }
                let mut queue = queue_slot.as_mut().get_mut().as_mut().ok_or(ENODEV)?;
                // Keep both borrows inside the mutex scope. Dropping the queue or unlocking it
                // mid-frame would permit a second connector to interleave URBs on this endpoint.
                let submit = |staging: &mut KVec<u8>, q: &mut crate::usb::BulkOutQueue| -> Result {
                    let staging = &mut staging[..];
                    let q = &mut *q;
                    // Scatter/gather cursor over
                    // [optional ARM][optional Navarro opener][authenticated stream report]
                    // [record chunks][parameter map][trailer]. Join only one
                    // transfer at a time in the reusable bounded staging allocation, avoiding a
                    // contiguous allocation spanning the complete frame.
                    let arm_parts = usize::from(!arm_slice.is_empty());
                    let opener_parts = usize::from(!opener_slice.is_empty());
                    let report_parts = usize::from(!report_slice.is_empty());
                    let param_parts = usize::from(!params.is_empty());
                    let trailer_parts = 1usize;
                    let opener_end = arm_parts + opener_parts;
                    let lead = opener_end + report_parts;
                    let part_count = lead + frame_count + param_parts + trailer_parts;
                    let mut part_i = 0usize;
                    let mut part_off = 0usize;
                    let mut wire_off = 0usize;
                    while wire_off < wire_len {
                        let data_len = (wire_len - wire_off).min(XFER);
                        let dst = &mut staging[..data_len];
                        let mut dst_off = 0usize;
                        while dst_off < dst.len() && part_i < part_count {
                            let part: &[u8] = if part_i < arm_parts {
                                arm_slice
                            } else if part_i < opener_end {
                                opener_slice
                            } else if part_i < lead {
                                report_slice
                            } else if part_i < lead + frame_count {
                                &frames[part_i - lead][..]
                            } else if part_i < lead + frame_count + param_parts {
                                &params[..]
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
                            pr_warn!(
                                "vino: scanout head={} pipeline submit at off={}/{} failed\n",
                                head,
                                wire_off,
                                wire_len
                            );
                            if *crate::module_parameters::video_clear_halt.value() != 0 {
                                let _ = dev.clear_video_halt(head_i);
                            }
                            return Err(e);
                        }
                        wire_off += data_len;
                    }
                    Ok(())
                };
                submit(&mut staging, &mut queue)
            };
            // Restore the per-connector staging allocation after the endpoint transaction.
            data.video_staging.lock()[head_i] = Some(staging);
            submitted?;
        }

        // The ARM burst was delivered with presentation zero. Clear it immediately rather than
        // after the whole replay: if a later copy fails, retrying ARM would corrupt a pipe that is
        // already armed.
        if repeat == 0 && startup {
            data.arm_prefix_pending
                .fetch_and(!head_bit, core::sync::atomic::Ordering::Release);
            // The cold-link requirement is measured from the start of continuous VIDEO, not from
            // the earlier mode-set. Refresh the complete training window here so modeset bracket
            // latency, cross-head serialization, and encoder time cannot make it intermittently
            // too short. Subsequent cadence-selected compositor flips and idle settle repaints are
            // both promoted to full keyframes while this deadline is live.
            data.sustain_until.lock()[head_i] =
                Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
            vino_debug!(
                "vino: scanout head={} initial ARM+keyframe accepted ({} B on the wire)\n",
                head,
                wire_len
            );
            // The dock expects two stream-commit messages on EP02 immediately after accepting the
            // video arm burst.
            for _ in 0..2 {
                match data.send_cp(dev, 0x16, 0, |ctr| crate::cp::stream_commit(ctr, head)) {
                    Ok(()) => vino_debug!("vino: stream-commit head={} ok\n", head),
                    Err(e) => pr_warn!("vino: stream-commit head={} failed ({e:?})\n", head),
                }
            }
        }

        // Rate-limited: see `STATUS_POLL_MIN_MS`. Sending this per presentation is what starved
        // the other head of the control link.
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
                vino_debug!("vino: scanout head={} CP status poll failed ({e:?})\n", head);
            }
        }
        // Do not drain here. The eight-URB ring spans frame boundaries; `send()` reaps a
        // completion when its slot is reused, so transport errors surface after the ring wraps
        // without introducing a per-frame pipeline bubble.
    }
    // Publish the new codec sequence only after every URB for this frame was submitted. A stale
    // generation or transport failure above leaves the old sequence intact for the next keyframe.
    data.scanout_seq.lock()[head_i] = next_seq.wrapping_add(repeat_count - 1);
    // The USB path accepted the complete image. Publish its content shadow only now; every early
    // return and transport error above deliberately leaves the previous dock-visible state intact.
    // The frame reached the dock, so every strip it carried has paid one transmission. A full
    // keyframe is presented twice and rewrites the whole surface, so it clears the ledger outright.
    {
        let mut ttl = data.dirty_ttl.lock();
        if let Some(debt) = ttl[head_i].as_mut() {
            if full {
                debt.fill(0);
            } else {
                for d in debt.iter_mut() {
                    *d = d.saturating_sub(1);
                }
            }
        }
    }
    // Publish the content shadow, and with it the encoded body of every strip this frame carried,
    // so the retransmissions `damage_repeats` owes can re-use the bytes instead of re-running the
    // codec (see `StripHashState::bodies`). Bodies for strips this frame did NOT touch are carried
    // forward from the previous state -- they are still what the dock holds, and a later debt pass
    // may select them. Best-effort throughout: a failed allocation costs a cache miss, never a
    // frame, so the hashes are published either way.
    {
        let mut state = data.strip_hashes.lock();
        let carried = state[head_i]
            .take()
            .filter(|c| c.w_pad == w_pad && c.h_pad == h_pad && c.tag == gamma_tag)
            .map(|c| (c.bodies, c.hashes));
        state[head_i] = content_hashes.map(|hashes| {
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
                    let tiles_x = w_pad >> geom.strip_w_shift();
                    for (&(sx, sy), body) in coords.iter().zip(strips) {
                        let idx = (sy >> geom.strip_h_shift()) * tiles_x
                            + (sx >> geom.strip_w_shift());
                        if let Some(slot) = bodies.get_mut(idx) {
                            *slot = body;
                        }
                    }
                }
            }
            StripHashState {
                w_pad,
                h_pad,
                hashes,
                bodies,
                tag: gamma_tag,
            }
        });
    }
    // A full keyframe was accepted -- this head may now send damage deltas until the next mode-set.
    if full {
        data.keyframe_pending
            .fetch_and(!kf_bit, core::sync::atomic::Ordering::Release);
    }

    vino_debug!(
        "vino: scanout head={} frame ok ({} presentation(s), {} B final write)\n",
        head,
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
    head: u8,
    src: &Arc<PixelSource>,
    rotation: plane::Rotation,
    // The client's changed rectangles (identity rotation only; empty means no pixel update).
    // `encode_and_send_wht` uses these to send a damage delta (only changed strips) after the first
    // full keyframe because the dock surface is undefined after a mode set.
    clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    // Non-64x16-aligned modes are padded to complete codec strips. The dock clips the padded image
    // to the active timing, matching the validated 68-band wire layout for 1080-line modes.
    encode_and_send_wht(dev, data, head, src, rotation, clips, w, h)
}

// ---- Encoder ----------------------------------------------------------------
