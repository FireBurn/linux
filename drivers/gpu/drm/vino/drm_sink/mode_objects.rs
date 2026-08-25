// SPDX-License-Identifier: GPL-2.0

//! The KMS objects the compositor drives: CRTC, primary and cursor planes, encoder and connector,
//! plus the software vblank timer that paces them.
//!
//! These callbacks run under the DRM atomic lock and must not block, so anything that talks to the
//! dock is queued for [`super::VinoDrmData`]'s workers rather than done here.

use super::*;
// `hdr_output_eotf` reads the connector's HDR metadata from the CRTC enable path; the trait is
// implemented for every connector state but has to be in scope to be called on an opaque one.
use kernel::drm::kms::connector::RawConnectorState;

/// A software vblank source: an hrtimer that fires once per frame and drives
/// `drm_crtc_handle_vblank()`. It stops when vblank is disabled and is also cancelled
/// unconditionally by [`VinoDrmData::shutdown`].
#[pin_data]
pub(crate) struct VblankTimer {
    #[pin]
    timer: HrTimer<Self>,
    /// Owned CRTC reference used by the hard-timer callback.
    ///
    /// This reference forms a cycle through the DRM device, so shutdown clears it after cancelling
    /// the timer. The IRQ-aware lock permits access from both process and hard-timer context.
    #[pin]
    pub(super) crtc: SpinLockIrq<Option<crtc::CrtcRef<VinoCrtc>>>,
    /// One scanout frame in nanoseconds (from the mode's `framedur_ns`).
    interval_ns: AtomicI64,
    /// Whether vblanks should currently be delivered (toggled by enable/disable_vblank).
    pub(super) enabled: AtomicBool,
}

impl VblankTimer {
    fn new() -> impl PinInit<Self> {
        pin_init!(VblankTimer {
            timer <- HrTimer::new(),
            crtc <- new_spinlock_irq!(None, "vino::vblank_crtc"),
            interval_ns: AtomicI64::new(16_666_666), // ~60 Hz until a mode sets it
            enabled: AtomicBool::new(false),
        })
    }
}

impl HrTimerCallback for VblankTimer {
    type Pointer<'a> = Arc<Self>;

    fn run(this: ArcBorrow<'_, Self>, mut ctx: HrTimerCallbackContext<'_, Self>) -> HrTimerRestart {
        // Vblank is off: let the timer die instead of ticking uselessly; `enable_vblank` re-arms
        // it. A concurrent re-arm racing this return is safe -- hrtimer keeps a timer that was
        // re-queued during its callback enqueued even on NORESTART.
        if !this.enabled.load(Ordering::Relaxed) {
            return HrTimerRestart::NoRestart;
        }
        // Take an owned copy of the published handle and release the lock *before* delivering the
        // vblank. `drm_crtc_handle_vblank()` takes `dev->vblank_time_lock`, and `enable_vblank`
        // runs the other way round -- it is called with the DRM vblank locks already held and
        // acquires this one -- so holding this lock across the delivery would be a lock inversion.
        // Cloning is just a `drm_dev_get()`, and the clone cannot drop the last reference: the
        // handle we cloned from stays published for the whole callback, because the only code that
        // clears it (`VinoDrmData::shutdown`) does so after `hrtimer_cancel` has waited for this
        // callback to return.
        let crtc = this.crtc.lock_with(ctx.local_interrupt_disabled()).clone();
        if let Some(crtc) = crtc {
            crtc.crtc().handle_vblank();
        }
        let interval = this.interval_ns.load(Ordering::Relaxed).max(1_000_000);
        ctx.forward_now(Delta::from_nanos(interval));
        HrTimerRestart::Restart
    }
}

impl_has_hr_timer! {
    impl HasHrTimer<Self> for VblankTimer {
        mode: RelativeHardMode<Monotonic>, field: self.timer
    }
}

#[pin_data]
pub(crate) struct VinoCrtc {
    /// Which display connector (0-based) this CRTC drives. Names the connector in diagnostics, and
    /// maps a CRTC in an atomic commit onto the connector whose rate it spends from the dock-wide
    /// budget.
    pub(super) connector: u8,
    /// The software vblank source for this CRTC.
    vblank: Arc<VblankTimer>,
    /// One driver-owned DRM vblank reference held for the whole active interval. A USB display has
    /// no hardware interrupt to bootstrap the compositor's first post-modeset presentation; if no
    /// initial page-flip event is attached, the DRM core never calls `enable_vblank`, the software
    /// timer never starts, and KWin leaves the first framebuffer frozen forever. Pinning one ref
    /// while active starts the clock deterministically; `atomic_disable` balances it before off.
    /// The vblank reference held for the whole time this CRTC is active. Taken in
    /// `atomic_enable` and released in `atomic_disable`, which is longer than a borrowed
    /// `VblankRef` can live, so an owned one is stored here.
    #[pin]
    pub(super) vblank_pinned: Mutex<Option<OwnedVblankRef<VinoCrtc>>>,
}

#[derive(Clone, Default)]
pub(crate) struct VinoCrtcState;

impl crtc::DriverCrtcState for VinoCrtcState {
    type Crtc = VinoCrtc;
}

#[vtable]
impl crtc::DriverCrtc for VinoCrtc {
    type Args = u8;
    type Driver = VinoDrmDriver;
    type State = VinoCrtcState;
    type VblankImpl = Self;

    fn new(_device: &drm::Device<Self::Driver>, connector: &u8) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoCrtc {
            connector: *connector,
            vblank: Arc::pin_init(VblankTimer::new(), GFP_KERNEL)?,
            vblank_pinned <- new_mutex!(None),
        })
    }

    /// The display is turning on (scanout begins). Enables vblank pacing, pushes a live mode-set CP
    /// message for the negotiated mode. The command is queued and is a no-op until CP engages.
    const HAS_ATOMIC_CHECK: bool = true;

    /// Reject a commit that exceeds the dock's combined active-connector budget.
    ///
    /// `mode_valid` checks each connector against the complete budget because the
    /// advertised modes must not depend on another connector's current state.
    /// A commit that does not increase this connector's rate is always allowed (it can only hold or
    /// reduce the combined total); no limiting when the budget is 0 (unknown).
    fn atomic_check(check: CrtcAtomicCheck<'_, Self>) -> Result {
        let crtc = check.crtc();
        let connector = crtc.connector as usize;
        let data: &VinoDrmData = crtc.drm_dev();
        let budget = data.dock_budget();
        let (state, old, mut new) = check.take_all();
        // The colour description reaches the dock in the mode-set message, which is only sent from
        // `atomic_enable`. Toggling HDR on a live output changes nothing the core considers a mode
        // change, so ask for one: otherwise the compositor starts encoding PQ into a sink that was
        // never told, and the picture washes out.
        let colour_changed = {
            let eotf = |c: Option<&_>| -> (u32, Option<connector::Eotf>) {
                c.map_or(
                    (0, None),
                    |c: &kernel::drm::kms::connector::OpaqueConnectorState<VinoDrmDriver>| {
                        (c.colorspace(), c.hdr_output_eotf())
                    },
                )
            };
            eotf(state.old_connector_state_for_crtc(crtc))
                != eotf(state.new_connector_state_for_crtc(crtc))
        };
        if colour_changed {
            new.set_mode_changed(true);
        }
        let old_rate = if old.active() {
            let m = old.mode();
            active_pixel_rate(m.hdisplay(), m.vdisplay(), m.vrefresh())
        } else {
            0
        };
        let new_rate = if new.active() {
            let m = new.mode();
            if (!old.active() || new.mode_changed()) && !crate::cp::mode_supported(m) {
                pr_warn!(
                    "vino: socket {socket} mode {}x{}@{} has no dock profile\n",
                    m.hdisplay(),
                    m.vdisplay(),
                    m.vrefresh(),
                    socket = connector + 1
                );
                return Err(EINVAL);
            }
            active_pixel_rate(m.hdisplay(), m.vdisplay(), m.vrefresh())
        } else {
            0
        };
        // Refresh ceiling, enforced here as well as in `mode_valid`, because pruning the mode list
        // is not a limit: a client can commit a user-defined mode that was never advertised
        // (`xrandr --newmode`, a modeline in a compositor config, `drm_mode_setcrtc` with its own
        // timing). This is the check that actually stops the dock being driven at a rate it goes
        // dark on, and it costs one comparison on the commit path.
        //
        // Only a commit that raises the refresh is rejected, exactly as the budget check below only
        // examines a commit that raises the rate. Every page flip carries the CRTC state through
        // here, so revalidating an unchanged rate would only add work to the hot path.
        let old_refresh = if old.active() {
            old.mode().vrefresh()
        } else {
            0
        };
        let new_refresh = if new.active() {
            new.mode().vrefresh()
        } else {
            0
        };
        if new_refresh > old_refresh && !data.refresh_within_limit(new_refresh) {
            let limit = data.max_refresh_hz();
            pr_warn!(
                "vino: socket {socket} refresh {new_refresh} exceeds {limit} Hz\n",
                socket = connector + 1
            );
            return Err(EINVAL);
        }
        // Price the budget for the depth this commit will actually be driven at. The recorded
        // depth is a commit behind -- it is written when a connector is enabled -- so asking for it
        // here admits a pair on the eight-bit budget and then drives it at ten, which the dock
        // answers by accepting every byte of the second stream and never lighting it.
        // Decide the link depth here rather than refusing the mode. Ten bits costs a third more
        // per pixel, so a pair that fits at eight may not fit at ten; a compositor answers a
        // rejected mode by disabling the output, so the depth gives way and both connectors light.
        if budget != 0 {
            let wants_deep = state
                .new_connector_state_for_crtc(crtc)
                .is_some_and(|conn| {
                    conn.max_requested_bpc() >= 10
                        && conn.hdr_output_eotf() == Some(connector::Eotf::SmpteSt2084)
                });
            let combined_now =
                new_rate.saturating_add(data.other_connectors_rate(&state, connector));
            let fits_deep = combined_now <= data.budget_at_depth(budget, true);
            data.set_connector_ten_bit_denied(connector as u8, wants_deep && !fits_deep);
        }
        if budget == 0 || new_rate <= old_rate {
            return Ok(());
        }
        let others = data.other_connectors_rate(&state, connector);
        let combined = new_rate.saturating_add(others);
        if combined > budget {
            pr_warn!(
                "vino: socket {socket} combined rate {combined} exceeds {budget}\n",
                socket = connector + 1
            );
            return Err(EINVAL);
        }
        Ok(())
    }

    fn atomic_enable(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        crtc.vblank_on();
        // Keep the software presentation clock running for the complete active interval. The
        // reference is stored as an owned one because it must outlive this callback and is only
        // released in `atomic_disable`. Page-flip events take their own additional refs.
        let mut pinned = crtc.vblank_pinned.lock();
        if pinned.is_none() {
            match crtc.vblank_get() {
                Ok(vblank_ref) => *pinned = Some(vblank_ref.into_owned()),
                Err(e) => pr_warn!(
                    "vino: failed to start connector {} software vblank clock ({e:?})\n",
                    crtc.connector
                ),
            }
        }
        drop(pinned);
        let connector = crtc.connector;
        let dev: &VinoDrmDevice = crtc.drm_dev();
        let data: &VinoDrmData = dev;
        let (state, new) = commit.take_state_new_state();
        // The transfer function is a connector property, but the mode set that carries it to the
        // dock is built here, so read it across from the connector routed to this CRTC. Only PQ
        // is distinguished: the dock's flags word has exactly one HDR bit, and every other EOTF
        // (including HLG, which it cannot express) is carried as SDR rather than mislabelled.
        let st2084 = state
            .new_connector_state_for_crtc(crtc)
            .and_then(|conn| conn.hdr_output_eotf())
            == Some(connector::Eotf::SmpteSt2084);
        data.set_connector_st2084(connector, st2084);
        // The link depth userspace asked for, which is a separate question from the framebuffer's
        // format: an eight-bit surface over a ten-bit link is the ordinary case, and reading the
        // format alone silently ignores the request.
        let requested_bpc = state
            .new_connector_state_for_crtc(crtc)
            .map_or(0, |conn| conn.max_requested_bpc());
        data.set_connector_max_bpc(connector, requested_bpc);
        // Cache this connector's colour transform for the scanout to apply.
        data.update_color(connector as usize, new.gamma_lut(), new.ctm());
        // Whatever this connector's sink state was, the enable path re-runs the full bracket and
        // mode-set, so any silence from here is the dock's news, not vino's.
        data.set_self_blanked(connector as usize, false);
        // The depth the dock is told must match the depth the plane will actually send.
        // `connector_ten_bit` is set from the committed framebuffer's fourcc, so a connector that
        // never gets a 10-bit buffer is never announced as 30 bpp. It goes in rather than being
        // patched on afterwards because the framebuffer allocation the set-mode states is derived
        // from it.
        let ten_bit = data.connector_is_ten_bit(connector as usize);
        let timing = match crate::cp::timing_from_drm_mode(new.mode(), data.allocation(), ten_bit) {
            Ok(mut timing) => {
                // Read back rather than reusing the local, so both halves of the colour
                // description come from the same per-connector state every other path consults.
                timing.st2084 = data.connector_is_st2084(connector as usize);
                // Declare the shared video endpoint. Four connectors are multiplexed onto two bulk
                // endpoints, and DLM names bit 2 of the flags word `Dual NIVO`; a connector whose
                // partner connector is also live has to say so, or the dock drives only one of the
                // two streams it is being sent.
                timing.dual_nivo = data.endpoint_is_shared(connector as usize);
                timing
            }
            Err(e) => {
                pr_err!(
                    "vino: connector {} reached atomic enable with an unsupported mode ({e:?})\n",
                    connector
                );
                return;
            }
        };
        vino_debug!(
            "vino: KMS CRTC enable -- connector {} display ON, mode {}x{}@{} {} bpc{} (scanout begins)\n",
            connector,
            timing.hactive,
            timing.vactive,
            timing.refresh_hz,
            if timing.ten_bit { 10 } else { 8 },
            if timing.st2084 { " PQ" } else { "" }
        );
        // Publish the desired timing; atomic callbacks must not block on USB.
        let mode_key = timing_key(&timing);
        data.last_timing.lock()[connector as usize] = Some(timing);
        data.modeset_requested[connector as usize].store(mode_key, Ordering::Release);
        data.queue_cmd(dev, KmsCmd::ModeSet { connector, timing });
    }

    /// The display is turning off (DPMS-off/blank/suspend all land here in atomic KMS).
    /// Resets the scanout state so a later re-enable sends a full keyframe rather than diffing
    /// against a shadow the dock may have dropped. Do not send the monitor's DDC/CI VCP 0xd6
    /// here: hard standby is separate from stopping the DisplayLink stream and can leave a panel
    /// asleep across a dock power cycle.
    fn atomic_disable(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        // Dropping the stored reference releases the vblank reference `atomic_enable` took.
        drop(crtc.vblank_pinned.lock().take());
        crtc.vblank_off();
        let connector = crtc.connector;
        let dev: &VinoDrmDevice = crtc.drm_dev();
        let data: &VinoDrmData = dev;
        data.update_color(connector as usize, None, None);
        // The stream is torn down; a later re-enable must re-send the mode-set before any video
        // write (the dock EPIPEs a write onto an unconfigured stream). Forget the active mode so
        // the scanout gate defers until the re-enable's mode-set lands.
        data.modeset_requested[connector as usize].store(0, Ordering::Release);
        data.modeset_active[connector as usize].store(0, core::sync::atomic::Ordering::Release);
        data.programmed_timing.lock()[connector as usize] = None;
        // Drop a framebuffer queued while this CRTC was active. Otherwise the deferred worker can
        // retry its old mode and paint after DPMS-off.
        data.pending_scanout.lock()[connector as usize] = None;
        data.settle_repaint.lock()[connector as usize] = None;
        // Spend nothing on a connector that is off; the re-enable's mode-set refills it.
        data.settle_budget[connector as usize].store(0, Ordering::Relaxed);
        // Release the ~14.7 MB private copy; a re-enable owes a keyframe and re-snapshots.
        data.shadow[connector as usize].lock().discard();
        data.sustain_until.lock()[connector as usize] = None;
        data.strip_hashes.lock()[connector as usize] = None;
        data.dirty_ttl.lock()[connector as usize] = None;
        vino_debug!(
            "vino: KMS CRTC disable -- socket {socket} display OFF (scanout stopped)\n",
            socket = connector + 1
        );
        // Stopping locally is not enough: the dock goes on scanning out whatever it last received,
        // so a DPMS-off left the panel lit on a frozen desktop. Queue the dock-side take-down for
        // the command worker -- this callback must not block on USB (see `KmsCmd`). It is queued
        // last, after the mode generation has been zeroed, because `blank_connector` keys its write
        // on exactly that zero.
        data.queue_cmd(dev, KmsCmd::Blank { connector });
    }

    /// Arm the page-flip completion event to be sent by the next vblank tick, so userspace is paced
    /// to the refresh rate rather than signalled immediately.
    fn atomic_flush(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        let data: &VinoDrmData = crtc.drm_dev();
        let mut new = commit.take_new_state();
        // Re-cache the colour transform on every commit that touches this CRTC, so a dynamic
        // GAMMA_LUT or CTM change on an already-enabled connector (which does not re-run
        // atomic_enable) is picked up rather than deferred to the next full modeset. A night-light
        // corrector ramps its CTM continuously, so this is the path that carries it, not
        // atomic_enable.
        data.update_color(crtc.connector as usize, new.gamma_lut(), new.ctm());
        if let Some(pending) = new.get_pending_vblank_event() {
            match crtc.vblank_get() {
                Ok(vbl_ref) => pending.arm(vbl_ref),
                // Vblank couldn't be enabled (e.g. mid-teardown): fall back to sending now.
                Err(_) => pending.send(),
            }
        }
    }
}

impl VblankSupport for VinoCrtc {
    type Crtc = VinoCrtc;

    fn enable_vblank(
        crtc: &crtc::Crtc<Self::Crtc>,
        vblank_guard: &VblankGuard<'_, Self::Crtc>,
        irq: &LocalInterruptDisabled,
    ) -> Result {
        let data: &VinoCrtc = crtc;
        // Track the mode's real frame duration so the tick matches the negotiated refresh rate.
        let fd = vblank_guard.frame_duration();
        if fd > 0 {
            data.vblank.interval_ns.store(fd as i64, Ordering::Relaxed);
        }
        // Publish the CRTC for the timer callback. Only the first enable stores it; the CRTC a
        // given timer serves never changes, and re-taking the reference on every enable would just
        // leak one `drm_dev_get()` per DPMS cycle. `lock_with` because the DRM core already called
        // us with local interrupts disabled -- proven by the `irq` token.
        {
            let mut published = data.vblank.crtc.lock_with(irq);
            if published.is_none() {
                *published = Some(crtc.to_owned_ref());
            }
        }
        data.vblank.enabled.store(true, Ordering::Relaxed);
        let interval = data.vblank.interval_ns.load(Ordering::Relaxed);
        // The started timer is registered on the DEVICE, so teardown can cancel it without
        // depending on the DRM core calling `disable_vblank` -- see `VinoDrmData::vblank`.
        let drm_data: &VinoDrmData = crtc.drm_dev();
        let connector = usize::from(data.connector);
        if connector >= MAX_CONNECTORS {
            return Ok(());
        }
        let mut slots = drm_data.vblank.lock();
        match &slots[connector] {
            None => {
                // First enable: start the timer and keep the handle as its sole owner.
                slots[connector] = Some((
                    data.vblank.clone(),
                    data.vblank.clone().start(Delta::from_nanos(interval)),
                ));
            }
            Some((_, h)) => {
                // Re-enable after `disable_vblank` let the timer die (NoRestart): re-queue it in
                // place. `restart` removes and re-inserts a still-pending timer, so this is
                // correct whether the final disabled tick has already fired or not, and it never
                // blocks on the callback -- which matters because we are called under the vblank
                // locks with interrupts disabled.
                h.restart(Delta::from_nanos(interval));
            }
        }
        Ok(())
    }

    fn disable_vblank(
        crtc: &crtc::Crtc<Self::Crtc>,
        _vblank_guard: &VblankGuard<'_, Self::Crtc>,
        _irq: &LocalInterruptDisabled,
    ) {
        let data: &VinoCrtc = crtc;
        data.vblank.enabled.store(false, Ordering::Relaxed);
    }

    fn get_vblank_timestamp(
        _crtc: &crtc::Crtc<Self::Crtc>,
        _in_vblank_irq: bool,
    ) -> Option<VblankTimestamp> {
        // Let DRM estimate the timestamp from the mode timings.
        None
    }
}

// ---- Planes: primary (scanout) + cursor -------------------------------------
//
// The safe KMS layer allows one `DriverPlane` type per driver, so `VinoPlane` serves both the
// primary and cursor planes, told apart by `is_cursor` (from the plane's `Args`).

/// Constructor arguments for a [`VinoPlane`]: which connector it belongs to and whether it is that
/// connector's cursor plane (vs. its primary scanout plane).
#[derive(Clone, Copy)]
pub(crate) struct PlaneArgs {
    pub(super) connector: u8,
    pub(super) is_cursor: bool,
}

#[pin_data]
pub(crate) struct VinoPlane {
    /// Which display connector (0-based) this plane belongs to. Selects the scanout video endpoint
    /// (see `DockProfile::video_endpoints`) and the cursor CP `connector` field.
    connector: u8,
    /// Whether this is the cursor plane (vs. the primary scanout plane).
    is_cursor: bool,
    /// The framebuffer region last uploaded as the cursor bitmap.
    #[pin]
    cursor_last: Mutex<Option<CursorUpload>>,
}

struct CursorUpload {
    framebuffer: ARef<kms::framebuffer::Framebuffer<VinoDrmDriver>>,
    /// Value of this connector's `cursor_epoch` when the bitmap was sent. A newer epoch means the
    /// dock has since been reconfigured and is no longer holding it.
    epoch: u32,
}

#[derive(Clone, Default)]
pub(crate) struct VinoPlaneState;

impl plane::DriverPlaneState for VinoPlaneState {
    type Plane = VinoPlane;
}

#[vtable]
impl plane::DriverPlane for VinoPlane {
    type Args = PlaneArgs;
    type Driver = VinoDrmDriver;
    type State = VinoPlaneState;

    fn new(_device: &drm::Device<Self::Driver>, args: PlaneArgs) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoPlane {
            connector: args.connector,
            is_cursor: args.is_cursor,
            cursor_last <- new_mutex!(None),
        })
    }

    /// Validate plane geometry and populate `drm_plane_state.visible`, which the damage iterator
    /// requires before it can report changed rectangles.
    fn atomic_check(check: PlaneAtomicCheck<'_, Self>) -> Result {
        let plane = check.plane();
        let (state, _old, mut new) = check.take_all();
        let Some(crtc) = new.crtc::<VinoDrmDriver>() else {
            // A disabled plane is not visible and needs no geometry validation.
            return Ok(());
        };
        let crtc_state = match state.get_new_crtc_state(crtc) {
            Some(s) => s,
            None => state.add_crtc_state(crtc)?,
        };
        // Vino supports 1:1 scanout only. Primary planes must cover the CRTC; cursor planes may be
        // positioned and clipped by the helper. Updates on a disabled CRTC remain disallowed.
        new.atomic_helper_check::<_, VinoDrmDriver>(&crtc_state, plane.is_cursor, false)?;

        // The transfer paths currently consume a full framebuffer, not an arbitrary source crop.
        // Require exactly that at the UAPI boundary. Cursor clipping performed by the helper above
        // is represented separately in its derived source rectangle.
        if let Some(fb) = new.framebuffer::<VinoDrmDriver>() {
            let full_width = fb.width().checked_shl(16).ok_or(EINVAL)?;
            let full_height = fb.height().checked_shl(16).ok_or(EINVAL)?;
            if new.source_x_16_16() != 0
                || new.source_y_16_16() != 0
                || new.source_width_16_16() != full_width
                || new.source_height_16_16() != full_height
            {
                return Err(EINVAL);
            }
        }

        Ok(())
    }

    /// A new framebuffer was flipped in. Maps it, converts XRGB8888 -> RGB565 (or feeds the
    /// Haar colour codec directly for an aligned mode), and bulk-writes the resulting EP08
    /// frame(s).
    ///
    /// EP08 writes happen only after CP engagement and a matching mode-set has landed.
    fn atomic_update(commit: PlaneAtomicCommit<'_, Self>) {
        let plane = commit.plane();
        let connector = plane.connector;
        let dev: &VinoDrmDevice = plane.drm_dev();
        let data: &VinoDrmData = dev;
        if !data.cp_engaged.load(core::sync::atomic::Ordering::SeqCst) {
            return;
        }

        // Cursor plane: publish bitmap and position commands for the asynchronous control worker.
        // The protocol uses id=0x1b for create, id=0x1c with the inner bitmap flag set for image,
        // and id=0x1a for movement.
        if plane.is_cursor {
            let new = commit.take_new_state();
            match new.framebuffer::<VinoDrmDriver>() {
                Some(fb) => {
                    let Some(source) = new.visible_source().ok().flatten() else {
                        *plane.cursor_last.lock() = None;
                        data.queue_cmd(
                            dev,
                            KmsCmd::CursorMove {
                                connector,
                                x: 0,
                                y: 0,
                                visible: false,
                            },
                        );
                        return;
                    };
                    let Some(destination) = new.visible_destination() else {
                        return;
                    };
                    // The complete framebuffer, not the helper's clipped rectangle: the dock
                    // expects a fixed-size cursor and clips at the panel edge itself.
                    let Ok(w) = u16::try_from(fb.width()) else {
                        return;
                    };
                    let Ok(h) = u16::try_from(fb.height()) else {
                        return;
                    };
                    let epoch = data.cursor_epoch[usize::from(connector)].load(Ordering::Acquire);
                    let mut last = plane.cursor_last.lock();
                    // Re-send when the bitmap changed, or when a reconfigure means the dock is no
                    // longer holding it: `CursorMove` succeeds against a cursor that no longer
                    // exists, so a stale match is silent on the wire.
                    let unchanged = last.as_ref().is_some_and(|last| {
                        core::ptr::eq(&*last.framebuffer, fb) && last.epoch == epoch
                    });
                    if !unchanged {
                        if let Ok(bgra) = read_cursor_bgra(fb, usize::from(w), usize::from(h)) {
                            // One shared bitmap per device: announce geometry only when it
                            // changes, not on every shape change.
                            let hi = usize::from(connector);
                            if data.cursor_geometry.lock()[hi].replace((w, h)) != Some((w, h)) {
                                data.queue_cmd(dev, KmsCmd::CursorCreate { connector, w, h });
                            }
                            data.queue_cmd(
                                dev,
                                KmsCmd::CursorImage {
                                    connector,
                                    w,
                                    h,
                                    bgra,
                                },
                            );
                            *last = Some(CursorUpload {
                                framebuffer: ARef::from(fb),
                                epoch,
                            });
                        }
                    }
                    // The dock positions the whole bitmap by its top-left, so this is the
                    // unclipped origin: scanout is 1:1, so how far into the source the helper
                    // started is how far off-screen the origin is. Clamped because the wire
                    // coordinates are unsigned -- the pointer stops at the edge.
                    let Ok(x) = u16::try_from((destination.x1 - source.x1).max(0)) else {
                        return;
                    };
                    let Ok(y) = u16::try_from((destination.y1 - source.y1).max(0)) else {
                        return;
                    };
                    data.queue_cmd(
                        dev,
                        KmsCmd::CursorMove {
                            connector,
                            x,
                            y,
                            visible: true,
                        },
                    );
                }
                // Cursor disabled: clear the dock's visible flag and forget the bitmap so a later
                // enable uploads it again.
                None => {
                    *plane.cursor_last.lock() = None;
                    data.queue_cmd(
                        dev,
                        KmsCmd::CursorMove {
                            connector,
                            x: 0,
                            y: 0,
                            visible: false,
                        },
                    );
                }
            }
            return;
        }

        // Primary plane: take both old and new state so the frame-damage clips can be merged.
        let (old, new) = commit.take_old_new_state();
        let Some(fb) = new.framebuffer::<VinoDrmDriver>() else {
            return;
        };
        // Plane rotation/reflection (identity unless the compositor set the rotation property).
        let rotation = new.rotation();
        // atomic_check has already rejected scaling, positioning, and partial source rectangles,
        // so these destination dimensions describe the complete output and cannot overrun the
        // framebuffer under any advertised rotation.
        let (w, h) = (new.crtc_w() as usize, new.crtc_h() as usize);
        // Collect the client's individual frame-damage clips (the rectangles that
        // `damage_merged()` would collapse into one bounding box), each clamped to the output, so
        // only the genuinely changed rectangles are re-converted from the source rather than their
        // whole enclosing box. Only for identity rotation (the clips are in un-rotated source
        // space; mapping them through 90/270 is not worth it for the throttled fallback path), and
        // never on the Haar keyframe path -- see `encode_and_send`. A fixed stack array keeps the
        // atomic-commit path allocation-free; on overflow the clips collapse into one bounding box.
        // An empty list means the client reported no changed pixels. Rotation/reflection still
        // promotes it to a full frame in `encode_and_send_haar`, because source-space clips cannot
        // yet be transformed safely for those cases.
        let mut clips = [(0usize, 0usize, 0usize, 0usize); MAX_DAMAGE_CLIPS];
        let mut nclips = 0usize;
        if rotation.angle() == plane::Rotation::ROTATE_0
            && !rotation.contains(plane::Rotation::REFLECT_X | plane::Rotation::REFLECT_Y)
        {
            new.for_each_damage_clip(old, |r| {
                let c = (
                    (r.x1.max(0) as usize).min(w),
                    (r.y1.max(0) as usize).min(h),
                    (r.x2.max(0) as usize).min(w),
                    (r.y2.max(0) as usize).min(h),
                );
                if nclips < MAX_DAMAGE_CLIPS {
                    clips[nclips] = c;
                    nclips += 1;
                } else {
                    // Overflow: collapse ALL accumulated clips plus `c` into a single bounding box
                    // in clips[0]. Folding only clips[0] with `c` here would silently drop the
                    // damage in clips[1..], leaving those regions stale on screen; union every
                    // pending clip so the whole changed area is still repainted.
                    let mut bb = c;
                    for &r in &clips[..nclips] {
                        bb = (bb.0.min(r.0), bb.1.min(r.1), bb.2.max(r.2), bb.3.max(r.3));
                    }
                    clips[0] = bb;
                    nclips = 1;
                }
            });
        }

        use core::sync::atomic::Ordering::Relaxed;
        // Throttle: while scanout is failing (dock NAKing because CP isn't engaged), skip the
        // upcoming pageflips set by the backoff below instead of converting+encoding+sending a
        // frame the dock will just drop.
        let skip = data.scanout_skip[connector as usize].load(Relaxed);
        if skip > 0 {
            data.scanout_skip[connector as usize].store(skip - 1, Relaxed);
            return;
        }
        data.queue_scanout(
            dev,
            fb,
            PendingScanout {
                connector,
                rotation,
                clips,
                nclips,
                w,
                h,
                shadow_idx: 0,
                shadow_generation: 0,
            },
        );
    }
}

/// The connector's fixed encoder. The dock has no encoder configuration of its own.
#[pin_data]
pub(crate) struct VinoEncoder;

#[vtable]
impl encoder::DriverEncoder for VinoEncoder {
    type Driver = VinoDrmDriver;
    type Args = ();

    fn new(_device: &drm::Device<Self::Driver>, _args: ()) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoEncoder {})
    }
}

// ---- Connector --------------------------------------------------------------

#[pin_data]
pub(crate) struct VinoConnector {
    /// Index into the owning device's per-connector EDID/presence arrays.
    connector: u8,
}

#[derive(Clone, Default)]
pub(crate) struct VinoConnectorState;

impl connector::DriverConnectorState for VinoConnectorState {
    type Connector = VinoConnector;
}

#[vtable]
impl connector::DriverConnector for VinoConnector {
    type Args = u8;
    type Driver = VinoDrmDriver;
    type State = VinoConnectorState;

    fn new(_device: &drm::Device<Self::Driver>, connector: u8) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoConnector { connector })
    }

    /// Install the dock's real EDID (read during probe) when available; otherwise fall back
    /// to a single 1920x1080@60 CVT mode. Reading the real EDID gives the true monitor
    /// name/size and its native mode list; the fallback keeps the connector usable when
    /// nothing is plugged into the dock or the CP channel has not yet delivered the EDID.
    fn get_modes<'a>(
        connector: ConnectorGuard<'a, Self>,
        guard: &ModeConfigGuard<'a, Self::Driver>,
    ) -> i32 {
        let data: &VinoDrmData = connector.drm_dev();
        let edids = data.cached_edids.lock();
        if let Some(blob) = edids
            .get(connector.connector as usize)
            .and_then(Option::as_ref)
        {
            // A failed EDID update adds no modes; fall through to the built-in list rather than
            // reporting a count the core did not actually get.
            match connector.add_edid_modes(blob) {
                Ok(n) if n > 0 => return n,
                other => {
                    // A cached EDID that yields no modes is indistinguishable downstream from a
                    // socket with nothing in it: both land on the synthesised list. Say which.
                    vino_debug!(
                        "vino: connector {} EDID ({} B) produced no modes ({:?}); using the built-in list\n",
                        connector.connector,
                        blob.len(),
                        other.map(|n| n)
                    );
                }
            }
        }
        drop(edids);
        let _ = guard;
        // A connector handed to DRM's EDID override must report NO modes here: the core applies the
        // override (`drm_kms_helper.edid_firmware=`, or a debugfs `edid_override` write) purely as
        // a fallback for a connector that is connected and produced none. Synthesising the
        // built-in list below would satisfy the probe helper and the override would never be
        // consulted.
        if data.edid_from_userspace(connector.connector as usize) {
            return 0;
        }
        // No downstream EDID yet: advertise the standard mode list up to the fallback resolution
        // and prefer it, keeping the connector usable until the dock delivers a real EDID.
        let n = connector.add_modes_noedid((FALLBACK_W as u32, FALLBACK_H as u32));
        connector.set_preferred_mode((FALLBACK_W as u32, FALLBACK_H as u32));
        n
    }

    /// Report the connector connected once the dock has delivered this connector's downstream EDID
    /// (a real monitor is attached and described) OR the bring-up work item confirmed CP engagement
    /// + this connector's DISPLAY-CAP push (`connectors_present`: on 3.4.26 the raw-EDID path can
    /// fail, so cached EDID alone would leave every connector permanently disconnected despite a
    /// fully-engaged dock). A connector with neither stays disconnected rather than advertising a
    /// phantom output.
    fn detect(connector: &Connector<Self>, _force: bool) -> Status {
        let data: &VinoDrmData = connector.drm_dev();
        let connector = connector.connector as usize;
        if connector >= data.connector_count() {
            return Status::Disconnected;
        }
        // On a dock that reports no downstream presence, both signals are absent for a connector
        // that has a monitor on it, and the whole measured choreography -- the framebuffer
        // allocation the set-mode states, the mode-set bracket, the sink states -- is the vendor
        // driving every connector it has. Offering a subset of that leaves the dock configured for
        // a display arrangement nobody described; `get_modes` supplies the fallback list until an
        // EDID arrives.
        if !data.reports_presence() {
            // The dock recovers an EDID for the socket a monitor is in and nothing for the one
            // that is empty, so the EDID is the whole presence signal here. The socket with
            // nothing in it is still configured -- it joins the dock-wide transaction at its
            // sibling's mode -- so it no longer has to be advertised to hold its place.
            //
            // Bring-up owns the endpoint until encrypted setup publishes the runtime CP link, so
            // a connector must not be offered before then whatever its EDID says.
            return if data.connector_has_edid(connector) && data.kms_activation_ready() {
                Status::Connected
            } else {
                Status::Disconnected
            };
        }
        let has_edid = data
            .cached_edids
            .lock()
            .get(connector)
            .is_some_and(Option::is_some);
        let present = data
            .connectors_present
            .load(core::sync::atomic::Ordering::Acquire)
            & (1 << connector)
            != 0;
        if has_edid || present {
            Status::Connected
        } else {
            Status::Disconnected
        }
    }

    /// Prune modes whose pixel clock exceeds a single connector's bandwidth ceiling
    /// ([`MAX_HEAD_CLOCK_KHZ`], ~4K@60), whose refresh rate exceeds what the dock has ever been
    /// shown to display ([`DOCK_MAX_REFRESH_HZ`]), or whose pixel rate exceeds the dock's budget.
    fn mode_valid(connector: ConnectorModeValidation<'_, Self>, mode: &DisplayMode) -> ModeStatus {
        let data: &VinoDrmData = connector.drm_dev();
        if mode.clock() < 0 || mode.clock() as u32 > data.max_connector_clock_khz() {
            return ModeStatus::ClockHigh;
        }
        if !data.refresh_within_limit(mode.vrefresh()) {
            return ModeStatus::ClockHigh;
        }
        if !crate::cp::mode_supported(mode) {
            return ModeStatus::Bad;
        }
        // Reject a mode only when that connector exceeds the dock's whole pixel budget. The atomic
        // CRTC check enforces the combined rate of simultaneously active connectors.
        let budget = data.dock_budget();
        let connector_rate = active_pixel_rate(mode.hdisplay(), mode.vdisplay(), mode.vrefresh());
        if budget != 0 && connector_rate > budget {
            return ModeStatus::Bad;
        }
        ModeStatus::Ok
    }
}

/// Apply the cached gamma ramp (three 256-entry 8-bit LUTs) to an `(r, g, b)` pixel, or return it
/// unchanged when no gamma is programmed.
#[inline]

/// Map an output pixel `(dx, dy)` back to its source-framebuffer pixel `(sx, sy)` under a DRM
/// plane `rotation` bitmask (`DRM_MODE_ROTATE_*` | `DRM_MODE_REFLECT_*`, the values the
/// standard `drm_plane_create_rotation_property` exposes). `sw`/`sh` are the SOURCE
/// (framebuffer) dimensions. Rotation is clockwise; reflection is applied in source space
/// after rotation. Pure and total (saturating), so it is unit-tested directly. Applied per source
/// pixel in [`encode_and_send`]/[`encode_and_send_haar`] for the plane's rotation property.
pub(crate) fn rot_src(
    rotation: plane::Rotation,
    dx: usize,
    dy: usize,
    sw: usize,
    sh: usize,
) -> (usize, usize) {
    let xmax = sw.saturating_sub(1);
    let ymax = sh.saturating_sub(1);
    let rot = rotation.angle();
    let (mut sx, mut sy) = if rot == plane::Rotation::ROTATE_90 {
        (dy, ymax.saturating_sub(dx))
    } else if rot == plane::Rotation::ROTATE_180 {
        (xmax.saturating_sub(dx), ymax.saturating_sub(dy))
    } else if rot == plane::Rotation::ROTATE_270 {
        (xmax.saturating_sub(dy), dx)
    } else {
        (dx, dy) // ROTATE_0 / unset
    };
    if rotation.contains(plane::Rotation::REFLECT_X) {
        sx = xmax.saturating_sub(sx);
    }
    if rotation.contains(plane::Rotation::REFLECT_Y) {
        sy = ymax.saturating_sub(sy);
    }
    (sx, sy)
}

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
#[kunit_tests(vino_mode_objects)]
mod tests {
    use super::*;

    #[test]
    fn rotation_pixel_mapping() {
        use drm::kms::plane::Rotation;

        // Source 2x3 (sw=2, sh=3). 0deg is identity; 180deg mirrors both axes.
        assert_eq!(rot_src(Rotation::ROTATE_0, 0, 0, 2, 3), (0, 0));
        assert_eq!(rot_src(Rotation::ROTATE_0, 1, 2, 2, 3), (1, 2));
        assert_eq!(rot_src(Rotation::ROTATE_180, 0, 0, 2, 3), (1, 2));
        assert_eq!(rot_src(Rotation::ROTATE_180, 1, 2, 2, 3), (0, 0));
        // 90deg: output dims are (sh,sw)=(3,2); (dx,dy) -> (dy, sh-1-dx).
        assert_eq!(rot_src(Rotation::ROTATE_90, 0, 0, 2, 3), (0, 2));
        assert_eq!(rot_src(Rotation::ROTATE_90, 2, 1, 2, 3), (1, 0));
        // 270deg: (dx,dy) -> (sw-1-dy, dx).
        assert_eq!(rot_src(Rotation::ROTATE_270, 0, 0, 2, 3), (1, 0));
        assert_eq!(rot_src(Rotation::ROTATE_270, 2, 1, 2, 3), (0, 2));
        // Reflect-X composes on top of the rotation (here identity): sx -> sw-1-sx.
        assert_eq!(
            rot_src(Rotation::ROTATE_0 | Rotation::REFLECT_X, 0, 0, 2, 3),
            (1, 0)
        );
    }
}
