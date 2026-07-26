// SPDX-License-Identifier: GPL-2.0

//! DRM/KMS sink: register a real `struct drm_device` with an atomic mode-setting
//! pipeline so the dock appears to userspace as a `card`/`renderD` node that can be
//! `drmModeSetCrtc`'d. Two independent display heads (see [`HEADS`]), each a CRTC driven by a
//! primary plane ([`VinoPlane::atomic_update`] -> per-head video endpoint scanout), a cursor plane,
//! a virtual encoder, and a virtual connector whose mode list comes from the dock's real EDID
//! (falling back to 1080p), with GEM-shmem dumb buffers and `drm_gem_fb_create` framebuffers.
//!
//! Built on the safe KMS mode-object layer (`kernel::drm::kms`), not the raw
//! `bindings::drm_*` C API: `VinoDrmDriver` implements `drm::kms::KmsDriver`, and each
//! mode object (`VinoCrtc`/`VinoPlane`/`VinoConnector`/`VinoEncoder`) implements the
//! matching `Driver*` trait rather than hand-assembling a C vtable.
//!
//! Wired onto the safe KMS layer:
//! - Per-head primary plane scanout to that head's video endpoint ([`VIDEO_EPS`]) and a
//!   `Type::Cursor` plane (bitmap + position forwarded via `cp::cursor_{create,image,move}` with
//!   the head as the CP `head` field).
//! - A 256-entry CRTC `GAMMA_LUT` (applied in the scanout) and a full plane rotation property
//!   (all four 90-degree rotations plus X/Y reflection), applied per source pixel via `rot_src`
//!   (90/270 swap the source/output dimensions -- see [`src_dims`]).
//! - Frame-damage clips: the WHT scanout emits only intersecting 64x16 strips
//!   (`RawPlaneState::for_each_damage_clip`) for identity rotation.
//! - Connector `detect()` (connected once the head's EDID arrives) and `mode_valid()` (prune modes
//!   above [`MAX_HEAD_CLOCK_KHZ`]).
//! - A DDC/CI virtual I2C adapter ([`VinoI2c`], tunnelling monitor-control writes to the dock over
//!   CP -- explicit brightness/contrast/etc. writes from userspace via `ddcutil`).
//!
//! Not yet done (requires further hardware characterization):
//! - Per-head mode-set / DDC differentiation on the wire -- the CP mode-set (`id=0x48`) has no
//!   decoded head/stream field, so the head is conveyed only by the video endpoint carrying frames.
//! - Damage under rotation is conservatively promoted to a full WHT frame; source-space damage
//!   rectangles are not yet transformed into rotated strip coordinates.
//! - DDC/CI *reads* (Get-VCP) -- need the dock's CP reply path; and brightness/contrast as
//!   *connector properties* (the I2C adapter is the interface for now).
//!
//! CP, per-head EDID/modesetting and dual-head EP08 scanout are live on hardware.

use core::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use kernel::{
    bindings, drm,
    drm::kms::{
        self,
        connector::{self, Connector, ConnectorGuard, ModeStatus, Status},
        crtc::{self, CrtcAtomicCheck, CrtcAtomicCommit, RawCrtc as _, RawCrtcState as _},
        encoder,
        modes::DisplayMode,
        plane::{self, PlaneAtomicCheck, PlaneAtomicCommit, RawPlaneState as _},
        vblank::{
            OwnedVblankRef, RawVblankCrtcState as _, VblankGuard, VblankSupport, VblankTimestamp,
        },
        KmsDriver, ModeConfigGuard, ModeConfigInfo, ModeObject as _, UnregisteredKmsDevice,
    },
    error::code::{EINVAL, ENOTSUPP, ENXIO},
    i2c, impl_has_hr_timer,
    interrupt::LocalInterruptDisabled,
    prelude::*,
    sync::{
        aref::ARef, new_mutex, new_spinlock, new_spinlock_irq, Arc, ArcBorrow, Mutex, SpinLock,
        SpinLockIrq,
    },
    time::{
        delay::fsleep,
        hrtimer::{
            ArcHrTimerHandle, HrTimer, HrTimerCallback, HrTimerCallbackContext, HrTimerPointer,
            HrTimerRestart, RelativeMode,
        },
        Delta, Instant, Monotonic,
    },
    workqueue::{self, impl_has_work, new_work, Work, WorkItem},
    xxhash,
};

/// Fallback connector mode advertised by `get_modes` when the dock has not delivered a real
/// downstream EDID yet. The live scanout geometry follows the actual framebuffer/negotiated
/// mode (see [`scanout_one`]), so this is only the no-EDID default, not a hard scanout limit.
///
/// 2560x1440: this dock's actual downstream monitor's real native/preferred timing
/// (ground-truthed 2026-07-16 via a direct-HDMI `ddcutil` read of the same MSI MAG 27CQ6F --
/// see `captures/rr-out-sequence-20260716/full-session-trace1/msi-mag27cq6f-direct-hdmi-edid.md`),
/// AND `64x16`-aligned (`video::wht::STRIP_W`/`STRIP_H`). Non-aligned modes are now safely padded
/// to the strip grid, but the native mode avoids padding and unnecessary work.
const FALLBACK_W: i32 = 2560;
const FALLBACK_H: i32 = 1440;

/// `DRM_FORMAT_XRGB8888` (`fourcc_code('X','R','2','4')`); the dock scans out 32bpp.
const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258;
/// Primary-plane format list (opaque 32bpp scanout).
static PRIMARY_FORMATS: [u32; 1] = [DRM_FORMAT_XRGB8888];

/// `DRM_FORMAT_ARGB8888` (`fourcc_code('A','R','2','4')`); the dock's cursor bitmap carries alpha.
const DRM_FORMAT_ARGB8888: u32 = 0x3432_5241;
/// Cursor-plane format list.
static CURSOR_FORMATS: [u32; 1] = [DRM_FORMAT_ARGB8888];

/// Per-mode pixel-clock ceiling (kHz) -- about 4K@60 (CEA 594 MHz). Modes above this are pruned
/// by the connector `mode_valid` hook ([`VinoConnector::mode_valid`]).
const MAX_HEAD_CLOCK_KHZ: i32 = 600_000;

/// **Hardware experiment 2026-07-23 — RESULT: NEGATIVE, both left `false`.**
///
/// Ran on hardware as 1280x720@60 on both heads, 877,440-byte full frames, ~5 fps each
/// (EP08 45.9 MB + EP0b 45.5 MB per 10 s), every URB status 0, dock firmware trace silent.
/// **The panels still did not light.** So neither the damage-delta path nor the pixel rate
/// is what keeps the dock from programming its downstream clock. Kept as switches because
/// they are a cheap way to re-pin the head for future experiments.
///
/// Hypothesis (user, from the one run where the panels did light): the working configuration was
/// sending *whole* frames continuously, not damage-delta slices, and was visibly sluggish -- which
/// is what a low frame rate of full-frame updates looks like. Every "accepted but dark" run since
/// has been sending small deltas after the first keyframe.
///
/// [`TEST_ONLY_720P60`] prunes the connector's mode list down to 1280x720@60, cutting the pixel
/// rate ~4x versus 2560x1440@120 so a full frame every cycle is comfortably affordable.
/// [`TEST_FORCE_FULL_FRAMES`] makes every scanout a full keyframe, so the dock is never asked to
/// composite a partial update onto a frame it may never have displayed.
///
/// Together with `FRAME_PERIOD_MS` = 140 this reproduces the "full frames, ~7 fps, sluggish"
/// shape. If the panels light, the delta path is implicated and the next step is finding what the
/// dock requires before it will accept a partial update.
const TEST_ONLY_720P60: bool = false;
const TEST_FORCE_FULL_FRAMES: bool = false;

/// **The single head-count knob** for the whole driver. Every per-head array/loop is sized by this
/// (`connectors`, `gamma`, `video_keys`, the CP setup in `vino.rs` via `CP_SETUP_HEADS = HEADS`, the
/// scanout keyframe/arm bitmasks). Which heads actually light up is decided at runtime by per-head
/// monitor detection (the `id=0x78` DISPLAY-CAP → `heads_present`), so this is only the MAXIMUM the
/// device can drive — extra heads with no monitor simply never connect.
///
/// The D6000 (DL-6000) is a 2-video-head device over USB (DLM drove only EP08+EP0b for two
/// monitors; EP0a/EP0c saw zero video traffic). Newer DisplayLink devices expose up to ~5 heads. To
/// raise this: **(1)** bump `HEADS`; **(2)** extend [`VIDEO_EPS`] to `HEADS` real bulk-OUT video
/// endpoints for the target device (the array literal must match `HEADS` or it won't compile — a
/// deliberate guard); **(3)** verify on hardware that the dock tolerates the per-head CP setup
/// (`send_cp_setup` runs it for every `HEADS`, even heads with no monitor) — an over-count once
/// hardware-reset the dock, so confirm with a capture before shipping >2.
pub(crate) const HEADS: usize = 2;

/// Per-head video bulk-OUT endpoint, indexed by head: head 0 -> EP08, head 1 -> EP0b (captured from
/// DLM driving a two-monitor D6000). The endpoint is the head selector for scanout; the cursor uses
/// the CP `head` field (see [`VinoPlane::atomic_update`]). **Must have exactly [`HEADS`] entries.**
/// For a device with more heads, add its real video endpoints here (they are device-specific — the
/// D6000's other bulk-OUT endpoints 0x0a/0x0c are NOT video heads; capture the target device driving
/// N monitors to learn its head->endpoint map).
pub(crate) const VIDEO_EPS: [u8; HEADS] = [0x08, 0x0b];

/// Maximum number of individual frame-damage rectangles re-converted per flip before they are
/// collapsed into a single bounding box. Bounds the stack array used on the atomic-commit path
/// (no per-flip allocation); a compositor that reports more clips than this just gets a coarser
/// (still correct) repaint.
const MAX_DAMAGE_CLIPS: usize = 16;
/// Minimum interval between normal frames for one head.
///
/// **Reverted 2026-07-23 from 16 ms (~60 Hz) to 140 ms**, matching the older steady-state Vino
/// cadence. Long DLM captures average roughly 7 full fps, but the paired physical COLD capture now
/// proves that activation is a separate burst phase: frames start 24--35 ms apart until the
/// downstream clock is programmed. Do not lower this ordinary-desktop throttle to reproduce that
/// burst -- doing so makes every compositor flip pay for a multi-megabyte encode. The bounded
/// [`COLD_TRAINING_PRESENTATIONS`] replay below reuses one encoding to keep the endpoint busy during
/// training, then returns to this normal cadence.
const FRAME_PERIOD_MS: i64 = 140;
/// Absolute timing of the first pre-encoded activation carrier and bracket close relative to the
/// mode-set submission.
///
/// The full physical lifecycle capture (`dlm-hotplug-sequence-20260725-143903`, step 2) is precise:
/// DLM finishes the open half at +26 ms, sends polls at +89/+95/+110 ms, starts video with the last
/// poll, then closes at +123/+125 ms while that frame is still in flight. The first paced Vino
/// attempt still let the independent keepalive interleave inside this sequence; its open half
/// stretched to +39 ms and video slipped to +131 ms. Drive the whole interval from one absolute
/// clock while the background CP loop is quiesced.
const PROMPT_VIDEO_MS: i64 = 110;
const PROMPT_CLOSE_2F_MS: i64 = 123;
const PROMPT_CLOSE_2E_MS: i64 = 125;
/// Worst-case one background keepalive iteration is a poll + heartbeat + two presence probes.
/// After publishing the exclusion flag, allow that already-started iteration to finish before
/// anchoring the mode-set. This delay is before the reference clock and does not alter DLM's
/// mode-relative timeline.
const PROMPT_KEEPALIVE_QUIESCE_MS: i64 = 40;
const PROMPT_TRAINING_OPEN_MS: i64 = 0;
const PROMPT_TRAINING_TAIL_MS: i64 = 400;
/// Absolute millisecond offsets from the **head-0** mode-set for the dual-head cold wake, decoded
/// from `captures/dlm-coldplug-withmon-160457` — the only capture in the corpus that provably
/// lights *both* panels of this dock from a physical cold plug.
///
/// Every earlier paced attempt used `dlm-hotplug-sequence-20260725-143903`, a **one-monitor**
/// lifecycle capture whose whole activation fits in 125 ms. The two references disagree by an
/// order of magnitude, and only this one covers the case we are trying to fix. Decoded with
/// `scripts/analyze-mon-cp.py --around dlm-started --full`; the offsets below are that listing
/// verbatim, relative to `+1.822 OUT 0x48/0x22 MODESET`.
///
/// Two properties of this schedule had never been reproduced:
///
///  * **EP02 goes completely silent from +29 ms to +1016 ms** — not even a status poll. vino has
///    always polled at ~66/s without interruption, so the dock has never seen this window.
///  * **DLM re-probes and re-fetches head 1's EDID in the middle of the bracket** (+1033/+1059).
///    The dock's firmware trace answers those with `2fab9`/`3e54c`/`30297`, three events that
///    appear in the lit trace and in no vino trace, immediately before it programs the clock.
mod cold {
    pub(super) const H1_MODE: i64 = 29;
    /// End of the silent window. Nothing may be sent on EP02 between `H1_MODE` and here.
    pub(super) const QUIET_END: i64 = 1016;
    pub(super) const H0_VIDEO: i64 = 1159;
    pub(super) const H1_VIDEO: i64 = 1233;
    /// `(offset_ms, head, sub, state)` stream markers. `sub` 0x2f/0x2e as on the wire.
    pub(super) const MARKERS: &[(i64, u8, u16, u8)] = &[
        (17, 0, 0x2f, 1),
        (21, 0, 0x2e, 3),
        (1016, 0, 0x2f, 1),
        (1021, 1, 0x2f, 1),
        (1023, 0, 0x2e, 3),
        (1029, 1, 0x2e, 3),
        (1056, 0, 0x2f, 1),
        (1057, 1, 0x2f, 1),
        (1064, 0, 0x2e, 0),
        (1124, 1, 0x2e, 3),
        (1132, 1, 0x2f, 1),
        (1135, 1, 0x2e, 0),
        (1195, 0, 0x2f, 0),
        (1208, 0, 0x2e, 0),
        (1220, 1, 0x2f, 1),
        (1225, 1, 0x2e, 0),
        (1295, 1, 0x2f, 0),
        (1298, 1, 0x2e, 0),
    ];
    /// `id=0x14 sub=0x0c` status polls.
    pub(super) const POLLS: &[i64] = &[5, 26, 1019, 1130, 1192, 1204, 1222, 1235, 1253, 1270, 1287, 1304];
    /// `(offset_ms, head, is_fetch)` — `false` is the `0x15/0x20` probe, `true` the `0x15/0x21`
    /// fetch.
    pub(super) const EDID: &[(i64, u8, bool)] = &[(1033, 1, false), (1059, 1, true)];
    /// Keep both carriers alternating for this long after the last close marker. DLM's trace
    /// programs the head-0 clock 377 ms after its first video bytes and head 1's 589 ms after
    /// its own, so the stream must outlive the bracket by a comfortable margin.
    pub(super) const CARRIER_TAIL_MS: i64 = 800;
}
/// Number of back-to-back presentations of one already-encoded full frame while a newly-mode-set
/// downstream is training.
///
/// The physical DLM cold capture starts full frames only 24--35 ms apart, with 0.4--5 ms between
/// the end of one frame and the start of the next. Vino's WHT encode is intentionally off the KMS
/// path but takes long enough that re-encoding at [`FRAME_PERIOD_MS`] produced 420--1430 ms gaps in
/// the failing cold capture. Replaying the already-encoded frame eight times keeps this endpoint's
/// persistent USB ring busy for roughly 0.45 s -- the measured interval before DLM's cold trace
/// reaches `2807a`/`2990d` -- without raising the normal steady-state cadence.
const COLD_TRAINING_PRESENTATIONS: u32 = 8;
type DamageRect = (usize, usize, usize, usize);
type BoundInterface<'a> = super::UsbLink<'a>;

/// Exact-enough generation key for one timing. Width/height alone is not a generation: switching
/// 2560x1440@60 -> @120 used to leave the same key and allowed the old frame to cross the new
/// mode-set. The four u16 wire fields fit losslessly in one atomic u64.
fn timing_key(t: &super::cp::Timing) -> u64 {
    ((t.hactive as u64) << 48)
        | ((t.vactive as u64) << 32)
        | ((t.refresh_hz as u64) << 16)
        | t.pixel_clock_10khz as u64
}

/// Content of the last frame successfully submitted for one head, represented in the dock's native
/// 64x16 strip grid. KWin frequently omits `FB_DAMAGE_CLIPS` when switching framebuffer objects; a
/// raw-content shadow is therefore the authoritative way to distinguish an unchanged flip from a
/// real repaint. `KVVec` permits vmalloc fallback for the roughly 29-KiB 1440p hash table.
struct StripHashState {
    w_pad: usize,
    h_pad: usize,
    hashes: KVVec<u64>,
}

/// The DRM driver marker type.
pub(super) struct VinoDrmDriver;

/// Convenience alias for our concrete `drm::Device`.
pub(super) type VinoDrmDevice = drm::Device<VinoDrmDriver>;

/// The live CP session the bring-up work item publishes once the dock engages the cipher
/// (`acks > 0`), so the KMS callbacks can seal+send runtime CP messages (a mode-set when the
/// compositor switches mode) that continue the SAME keystream the bring-up setup left off at.
/// `wire_seq` is the AES-CTR block counter (advanced by the content blocks of each send; the
/// appended Dl3Cmac tag is not part of the keystream) and `counter` the dock-echoed inner CP
/// counter. Both advance per send under the mutex.
pub(super) struct CpLink {
    ks: [u8; 16],
    riv: [u8; 8],
    wire_seq: u32,
    counter: u16,
    ep84_q: Option<super::usb::BulkInQueue>,
}

/// A deferred runtime CP send, queued by a DRM atomic-commit callback and executed by
/// [`VinoDrmData`]'s async command worker instead of blocking the calling commit.
///
/// **Why this exists (2026-07-16, see `project_ep08_silent_freeze_after_crtc_enable_20260716`
/// memory):** `atomic_enable`/`atomic_disable`/the cursor path used to call
/// [`VinoDrmData::send_cp`]/[`VinoDrmData::set_vcp`] directly -- real blocking USB I/O (a mutex
/// plus a hardware round-trip) executed inline in the DRM atomic-commit calling context. Compared
/// against `revdi`'s equivalent callbacks (`EvdiCrtc::atomic_enable`/`atomic_disable`), which only
/// ever queue a `drm_event` for userspace DLM to consume asynchronously and never touch hardware
/// at all -- vino cannot offload to a userspace daemon the way revdi does (vino's whole point is
/// to replace it), but the actual hardware write does not need to happen inline in the callback
/// either. A live freeze landed right after a rapid CRTC mode-switch with no oops/panic at all,
/// consistent with a compositor's synchronous commit ioctl blocking on a slow/stuck USB
/// transfer -- this makes the callbacks fast and non-blocking by construction, matching DRM
/// driver-writing convention (slow hardware programming belongs in a deferred worker, not the
/// atomic-commit callback), mirroring the pattern `BringUp`'s workqueue already uses for bring-up.
enum KmsCmd {
    ModeSet {
        head: u8,
        timing: super::cp::Timing,
    },
    CursorCreate {
        head: u8,
        w: u16,
        h: u16,
    },
    CursorImage {
        head: u8,
        w: u16,
        h: u16,
        bgra: KVec<u8>,
    },
    CursorMove {
        head: u8,
        x: u16,
        y: u16,
    },
}

/// Latest primary-plane flip awaiting compression on the deferred worker. The framebuffer is
/// refcounted, so it remains valid after the atomic commit callback returns. There is one slot per
/// head: a newer flip replaces an older unsent flip instead of building an unbounded queue behind
/// a slow encoder. When replacement could lose accumulated damage, the newer flip is promoted to a
/// full-output damage rectangle.
struct PendingScanout {
    head: u8,
    fb: ARef<kms::framebuffer::Framebuffer<VinoDrmDriver>>,
    rotation: u32,
    clips: [DamageRect; MAX_DAMAGE_CLIPS],
    nclips: usize,
    w: usize,
    h: usize,
}

impl Clone for PendingScanout {
    fn clone(&self) -> Self {
        Self {
            head: self.head,
            fb: self.fb.clone(),
            rotation: self.rotation,
            clips: self.clips,
            nclips: self.nclips,
            w: self.w,
            h: self.h,
        }
    }
}

/// How long after a keyframe to repaint the same head once more.
///
/// **Why this exists (HW-observed 2026-07-22):** on a freshly enabled output the only frame either
/// head ever received was 205,696/205,968 bytes -- the known ARM+all-black sizes -- and then *zero*
/// video bytes for minutes. A Wayland compositor stops committing when nothing changes, so if the
/// one keyframe a mode-set owes happens to catch a blank buffer, that blank image is what the panel
/// keeps forever: there is no later flip to correct it. Damage deltas were proven to work in the
/// same session (a window moving on the output produced immediate EP08 traffic), so this is
/// specifically a *first* frame problem.
///
/// One extra full repaint of the head's newest known framebuffer, once, closes that hole for the
/// cost of a single redundant keyframe per mode-set.
const SETTLE_REPAINT_MS: i64 = 1200;

/// Print the live CP session key/nonces to the kernel log at session publish so a usbmon capture
/// of **vino's own** CP dialogue can be decrypted offline (`scripts/decrypt-dlm-cp.py`) and diffed
/// against DLM's. Development aid only -- it puts a session key in `dmesg`.
const CP_KEY_DEBUG: bool = true;

/// DRM device-private data: the bound USB interface, engaged CP session, connector state, deferred
/// scanout slots and per-head transport state.
#[pin_data]
pub(super) struct VinoDrmData {
    /// The USB I/O-permitted window for this device's interface, shared with the persistent
    /// queues. `disconnect()` closes it, after which every transfer path here fails cleanly
    /// instead of touching an unbound interface.
    pub(super) io: Arc<super::usb::IoWindow>,
    /// The dock's endpoints, resolved and direction/type-checked once during probe.
    pub(super) eps: super::Endpoints,
    /// Stops every producer before unplug drains the embedded work item. This is checked while
    /// holding the producer's queue lock so a late atomic callback cannot enqueue a self-owning
    /// `ARef<VinoDrmDevice>` after `cancel_sync()` has already returned.
    shutting_down: AtomicBool,
    #[pin]
    cp_link: Mutex<Option<CpLink>>,
    /// Pending [`KmsCmd`]s an atomic-commit callback queued, drained in order by
    /// [`VinoDrmData`]'s [`WorkItem::run`] on the system workqueue. See [`KmsCmd`]'s doc comment.
    #[pin]
    cmd_queue: Mutex<KVec<KmsCmd>>,
    /// Coalescing per-head scanout slots consumed by `cmd_work`. Compression and USB submission
    /// must not run in `atomic_update`: doing so blocked KWin's commit worker for hundreds of
    /// milliseconds per frame and eventually produced multi-second pageflip timeouts.
    #[pin]
    pending_scanout: Mutex<[Option<PendingScanout>; HEADS]>,
    /// A one-shot repaint of the head's newest known framebuffer, armed when a keyframe is sent and
    /// due `SETTLE_REPAINT_MS` later. Cleared as soon as it is taken, or whenever a real flip
    /// arrives (that flip already carries newer content, so the redundant repaint is pointless).
    /// See [`SETTLE_REPAINT_MS`] for the hardware observation that motivates it.
    #[pin]
    settle_repaint: Mutex<[Option<(Instant<Monotonic>, PendingScanout)>; HEADS]>,
    /// Count of primary-plane flips seen per head, purely diagnostic -- see `queue_scanout`.
    flips: [core::sync::atomic::AtomicU64; HEADS],
    /// Every started software vblank timer, with the handle whose drop is the full
    /// `hrtimer_cancel`. Owned **here**, not on `VinoCrtc`, so `shutdown()` can stop the timers on
    /// the one teardown path that is guaranteed to run before the module can be unloaded.
    ///
    /// **HW-confirmed panic, 2026-07-22 (pstore `Oops#2`/`Panic#3`):** the previous design let the
    /// timer self-stop when `disable_vblank` cleared `enabled`, on the stated assumption that
    /// `unplug()` -> `drm_atomic_helper_shutdown()` always reaches `disable_vblank` first. The
    /// crash log falsifies that: the unload sequence logged the keepalive stopping, the deferred
    /// work draining, both interfaces disconnecting and `usbcore: deregistering interface driver
    /// vino` with **no** `KMS CRTC disable` line at all. `atomic_disable` never ran, so
    /// `vblank_pinned` was never released, the DRM vblank refcount never reached zero,
    /// `disable_vblank` was never called, and the timer kept re-arming itself straight through
    /// `modprobe -r`. 17 ms after the deregister it fired into freed module text
    /// (`RIP: 0xffffffffa002b2e1`, `Code: cc cc cc ...`, `<IRQ> __hrtimer_run_queues`,
    /// `Modules linked in: usbmon [last unloaded: vino]`) -> fatal exception in interrupt -> reboot.
    ///
    /// A `SpinLock`, not a `Mutex`: `enable_vblank` runs under the DRM vblank locks with local
    /// interrupts disabled and must not sleep.
    #[pin]
    vblank: SpinLock<[Option<(Arc<VblankTimer>, ArcHrTimerHandle<VblankTimer>)>; HEADS]>,
    /// The work item that drains `cmd_queue`. Embedded on the `drm::Device` (not `VinoDrmData`
    /// directly) per the safe KMS binding's `WorkItem`/`HasWork` blanket impls -- see
    /// `queue_cmd`'s doc comment for how it's enqueued.
    #[pin]
    cmd_work: Work<VinoDrmDevice>,
    /// Downstream EDID per head. Connector callbacks use their head index to read this owned state;
    /// publishing EDID therefore requires no raw pointer back into a DRM mode object.
    #[pin]
    cached_edids: Mutex<[Option<KVec<u8>>; HEADS]>,
    /// Bit N is set once CP confirms that a real downstream monitor is present on head N.
    heads_present: core::sync::atomic::AtomicU32,
    /// Each head's gamma ramp cached from its CRTC atomic hook as three 256-entry 8-bit LUTs
    /// (`[r; 256] ++ [g; 256] ++ [b; 256]`), or `None` for identity. Cached here (not read from
    /// the CRTC state) because scanout runs in the plane path; each entry is `Copy`, so the scanout
    /// snapshots its head's entry under the lock and applies it without holding the lock in the
    /// pixel loop. Per head so a second display's gamma cannot clobber the first's.
    #[pin]
    gamma: Mutex<[Option<[u8; 768]>; HEADS]>,
    /// Per-head strip hashes for the last frame accepted by the USB submission path. Updated only
    /// after the complete frame has been queued, so a failed transfer can never advance the shadow
    /// beyond what the dock may actually display.
    #[pin]
    strip_hashes: Mutex<[Option<StripHashState>; HEADS]>,
    /// Set once the dock engages the CP cipher (`wsub=0x45` acks > 0); EP08 scanout is gated on it.
    /// Per device, so a second connected dock does not share one dock's engagement state.
    cp_engaged: core::sync::atomic::AtomicBool,
    /// Excludes the independent keepalive loop while the mode worker emits DLM's measured,
    /// mode-relative activation timeline. Without this, a keepalive poll can win `cp_link` between
    /// two explicitly paced markers and stretch/reorder the sequence.
    cp_timeline_exclusive: core::sync::atomic::AtomicBool,
    /// The mode-set (`id=0x48`) that has actually been SENT to each dock head, encoded as
    /// `hactive:vactive:refresh:pclk`; `0` = none sent. Set by the async worker AFTER a
    /// `KmsCmd::ModeSet` send completes, reset on `atomic_disable`. The scanout path gates the
    /// first EP08 write on this MATCHING the frame's `w`x`h`: on an enabling modeset commit the
    /// plane update (`commit_planes`) runs BEFORE the CRTC enable (`commit_modeset_enables`) queues
    /// the mode-set, so the video write would otherwise race ahead of the mode-set and the dock
    /// EPIPEs the write onto an unconfigured stream (HW-observed 2026-07-17). Gating defers the
    /// write a frame until the mode-set has landed -- DLM's own order is mode-set -> video.
    /// (This replaced the old `ep08_test_fired` one-shot latch, which existed because the first
    /// EP08 write used to *disconnect* the dock; with the per-head CP fix that is now a recoverable
    /// EPIPE, so continuous scanout gated by this per-device state plus the backoff is safe.)
    modeset_active: [core::sync::atomic::AtomicU64; HEADS],
    /// Latest mode userspace currently requests per head, encoded like `modeset_active`; zero
    /// means the CRTC is disabled. The deferred worker uses this generation key to discard stale
    /// mode-set commands and framebuffers left by a rapid disable/re-enable sequence.
    modeset_requested: [core::sync::atomic::AtomicU64; HEADS],
    /// Per-head monotonic timestamp of the last accepted EP08 frame, for rate limiting when a
    /// compositor submits faster than DLM's roughly 7-fps video cadence. This is a minimum interval,
    /// not a refresh timer: a static KWin desktop produced one frame in a 22-second capture without
    /// resetting once the separate continuous CP keepalive was present.
    #[pin]
    last_frame: SpinLock<[Option<Instant<Monotonic>>; HEADS]>,
    /// ★ Cold-downstream training deadline per head (2026-07-25). On a mode-set the dock needs a
    /// SUSTAINED video stream (~0.4 s of continuous frames) before it programs its downstream pixel
    /// clock and lights the panel -- ONE keyframe is not enough on a COLD link. Proven by capturing
    /// DLM activating a cold dock (`captures/dlm-coldplug-withmon-*`): its `2807a`/`2990d` fire ~0.4 s
    /// AFTER a continuous EP08/EP0b stream begins. vino sent one keyframe then idled, so a cold link
    /// never activated. A cadence-selected full frame is replayed
    /// [`COLD_TRAINING_PRESENTATIONS`] times without re-encoding, giving the endpoint the genuinely
    /// gapless burst seen under DLM; the settle/live paths keep selecting fresh keyframes until this
    /// deadline expires. Afterwards scanout backs off to its normal cadence. Set on every mode-set;
    /// harmless on a warm link (which activates on the first frame anyway).
    #[pin]
    sustain_until: SpinLock<[Option<Instant<Monotonic>>; HEADS]>,
    /// Logical WHT frame sequence per head. This used to live on `VinoPlane`; moving scanout to the
    /// device worker means the sequence belongs with the rest of the deferred transport state.
    #[pin]
    scanout_seq: Mutex<[u32; HEADS]>,
    /// One PERSISTENT async bulk-OUT queue per head, created on first scanout and kept for the life
    /// of the device. DLM streams video as a continuously pipelined ring of <=65536-B URBs with
    /// **8 outstanding at all times** (xHCI trace: 5835 URBs of exactly 65536, max-in-flight 8, mean
    /// latency 2.06 ms) -- it never drains between frames. Creating a queue per frame instead makes
    /// `Drop` `usb_kill_urb` every slot and tears the ring down each frame, which the dock rejects.
    /// See `docs/DLM-WIRE-ANNOTATION.md`.
    #[pin]
    video_q: Mutex<[Option<super::usb::BulkOutQueue>; HEADS]>,
    /// One reusable 64-KiB coalescing window per head. `frame_records` deliberately stores a frame
    /// as small allocations so encoding never asks kmalloc for multi-megabyte physically contiguous
    /// memory; scanout joins those fragments into this bounded window before `BulkOutQueue::send`
    /// copies it into the persistent DMA ring. Internal record boundaries remain invisible on USB.
    #[pin]
    video_staging: Mutex<[Option<KVec<u8>>; HEADS]>,
    /// The last timing `atomic_enable` computed for this device, cached so the scanout path can
    /// RE-SEND the mode-set if it finds `modeset_active` unset. Without this, a mode-set whose send
    /// failed (e.g. the CRTC was enabled while CP was still re-engaging after a dock reset) is never
    /// retried, and every subsequent flip defers forever -- HW-observed 2026-07-20 as ~5000
    /// consecutive "mode-set not yet sent" deferrals with the compositor happily flipping.
    #[pin]
    last_timing: SpinLock<[Option<super::cp::Timing>; HEADS]>,
    /// Per-head bitmask (`1 << head`) of heads that still owe the video-pipe **arm burst** prefixed
    /// to their next EP08 write. Set (all heads) after a `KmsCmd::ModeSet` send; a head's bit is
    /// cleared once its arm-prefixed frame is accepted. DLM ships the 10-record arm burst
    /// concatenated with frame 0's video in ONE EP08 URB (RE 2026-07-18 of the cold-plug pcap's
    /// first 65536-B URB -- see `project_ep08_two_faults_video_pipe_arm_20260718`); prior arm-burst
    /// attempts sent it as a SEPARATE transfer and disconnected the dock. This makes the first write
    /// after each mode-set carry `[arm burst][video]` as one contiguous stream, matching DLM.
    arm_prefix_pending: core::sync::atomic::AtomicU32,
    /// Per-head "owes a full keyframe" bitmask (bit `h` = head `h`). Set (all heads) after a
    /// `KmsCmd::ModeSet` send: a new mode leaves the dock's framebuffer undefined, so the first
    /// scanout after it must be a FULL frame ([`super::video::wht::colour_frame_ep08`]), not a
    /// damage delta -- otherwise the un-redrawn strips stay garbage. Cleared for a head once its
    /// keyframe is sent; subsequent flips send only changed strips
    /// ([`super::video::wht::colour_frame_ep08_damage`]). See `docs/VIDEO-PARTIAL-UPDATE-DESIGN.md`.
    keyframe_pending: core::sync::atomic::AtomicU32,
    /// Dock-wide pixel-rate budget (pixels/sec), already multiplied by the compression headroom.
    /// The DisplayLink chip has ONE total-throughput budget shared across all heads; without this
    /// two heads can each pass the per-head [`MAX_HEAD_CLOCK_KHZ`] cap while together overrunning the
    /// dock. Split evenly across the currently-connected heads ([`own_pixel_budget`]) and enforced
    /// in [`VinoConnector::mode_valid`] + [`VinoCrtc::atomic_check`]. `0` = unknown → no limiting
    /// (so a wrong/absent value never causes a false mode rejection). Ported from revdi's proven
    /// dual-head manager; see `docs/CROSS-HEAD-BANDWIDTH-DESIGN.md`.
    dock_pixel_budget: core::sync::atomic::AtomicU32,
    /// Consecutive failed live-scanout frames on this device, for log rate-limiting.
    scanout_fails: core::sync::atomic::AtomicU64,
    /// Upcoming pageflips to skip before this device's next scanout attempt (backoff while the
    /// dock NAKs). A single successful frame clears it.
    scanout_skip: core::sync::atomic::AtomicU64,
    /// Per-head video key delivered in the bring-up CP setup's `id=0x32` message (decoded dump:
    /// `captures/rr-out-sequence-20260716/cp-dialogue-decoded.txt`). ITS CRYPTOGRAPHIC ROLE IN
    /// EP08 IS NOT YET REVERSE-ENGINEERED -- only the wire slot it belongs in is proven, not
    /// whether/how it keys the video stream. Stashed here so the scanout path
    /// (`encode_and_send`/`encode_and_send_wht`) can start consuming it once that is known; they
    /// do not read it today.
    #[pin]
    video_keys: Mutex<[[u8; 32]; HEADS]>,
}

impl VinoDrmData {
    pub(super) fn new(
        io: Arc<super::usb::IoWindow>,
        eps: super::Endpoints,
    ) -> impl PinInit<Self, Error> {
        try_pin_init!(Self {
            io,
            eps,
            shutting_down: AtomicBool::new(false),
            cp_link <- new_mutex!(Option::<CpLink>::None),
            cmd_queue <- new_mutex!(KVec::new()),
            pending_scanout <- new_mutex!([const { None }; HEADS]),
            settle_repaint <- new_mutex!([const { None }; HEADS]),
            flips: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            vblank <- new_spinlock!([const { None }; HEADS]),
            cmd_work <- new_work!("vino::kms_cmd"),
            cached_edids <- new_mutex!([const { None }; HEADS]),
            heads_present: core::sync::atomic::AtomicU32::new(0),
            gamma <- new_mutex!([None; HEADS]),
            strip_hashes <- new_mutex!([const { None }; HEADS]),
            cp_engaged: core::sync::atomic::AtomicBool::new(false),
            cp_timeline_exclusive: core::sync::atomic::AtomicBool::new(false),
            modeset_active: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            modeset_requested: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            last_frame <- new_spinlock!([const { None }; HEADS]),
            sustain_until <- new_spinlock!([const { None }; HEADS]),
            scanout_seq <- new_mutex!([0; HEADS]),
            video_q <- new_mutex!([const { None }; HEADS]),
            video_staging <- new_mutex!([const { None }; HEADS]),
            last_timing <- new_spinlock!([None; HEADS]),
            arm_prefix_pending: core::sync::atomic::AtomicU32::new(0),
            keyframe_pending: core::sync::atomic::AtomicU32::new(0),
            // D6000 default: 442,368,000 px/s (one 1440p@120) x2 compression headroom = dual
            // 1440p@120. Overwrite from the dock once its budget field is RE'd (CONTROL-PLANE.md).
            dock_pixel_budget: core::sync::atomic::AtomicU32::new(884_736_000),
            scanout_fails: core::sync::atomic::AtomicU64::new(0),
            scanout_skip: core::sync::atomic::AtomicU64::new(0),
            video_keys <- new_mutex!([[0u8; 32]; HEADS]),
        })
    }

    /// Stop deferred DRM work while the parent USB interface is still bound. `cmd_work` is
    /// embedded in this DRM device and each successful enqueue temporarily owns an
    /// `ARef<VinoDrmDevice>`; pending scanouts also retain compositor framebuffers. Quiesce both
    /// producers, reclaim any queued work pointer, and drop those framebuffers before the final
    /// device references disappear during devres teardown.
    pub(super) fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        self.cp_timeline_exclusive.store(false, Ordering::Release);
        for mode in &self.modeset_requested {
            mode.store(0, Ordering::Release);
        }

        // Stop the software vblank clocks FIRST, and do it here rather than relying on the DRM
        // core reaching `disable_vblank` -- see the `vblank` field for the crash that proves it
        // does not always get there. Take the registry out from under the spinlock before
        // dropping the handles: `ArcHrTimerHandle`'s drop is a full `hrtimer_cancel`, which waits
        // for a running callback and therefore must not run in atomic context.
        let timers = {
            let mut slots = self.vblank.lock();
            core::mem::replace(&mut *slots, [const { None }; HEADS])
        };
        // Clear `enabled` before cancelling so a callback racing the cancel returns `NoRestart`
        // instead of re-arming behind it.
        for (timer, _) in timers.iter().flatten() {
            timer.enabled.store(false, Ordering::Relaxed);
        }
        // Split the registry: drop every `ArcHrTimerHandle` (each drop == `hrtimer_cancel`, which
        // waits for a running callback), but keep the `Arc<VblankTimer>`s alive so the published
        // CRTC handles can be released below. From here on no vblank callback can run or be
        // re-armed, because `VinoCrtc::vblank` is only reachable through a CRTC of this device and
        // every producer is already refusing work.
        let timers = timers.map(|slot| {
            slot.map(|(timer, handle)| {
                drop(handle);
                timer
            })
        });

        // Break the two device-to-itself reference cycles that keep the `drm_device` -- and with it
        // this card's DRM minor -- alive forever after unplug. Both run through a
        // `crtc::CrtcRef`, which owns an `ARef<VinoDrmDevice>`:
        //
        //   1. `VblankTimer::crtc`, published by the first `enable_vblank` and never released. The
        //      timer is owned by `VinoCrtc`, which lives inside the DRM device allocation.
        //   2. `VinoCrtc::vblank_pinned`, the driver-held vblank reference. `atomic_disable`
        //      normally releases it, but the 2026-07-22 unload crash proved `atomic_disable` does
        //      not always run, so teardown must not depend on it.
        //
        // Safe to do here even though these were the last self-references: `shutdown()`'s only
        // caller is `VinoDriver::disconnect`, which reaches it through the `drm::Registration`
        // still held in the bound data -- and that owns an `ARef<VinoDrmDevice>` of its own -- so
        // `&self` outlives this function regardless of what is dropped below. The taken values
        // are dropped outside both locks: `drm_dev_put()` can end in `drm_dev_release()` and
        // `drm_crtc_vblank_put()` takes the DRM vblank locks, neither of which may run under our
        // spinlock.
        for timer in timers.iter().flatten() {
            let published = timer.crtc.lock().take();
            if let Some(crtc_ref) = published {
                let crtc: &VinoCrtc = crtc_ref.crtc();
                drop(crtc.vblank_pinned.lock().take());
                drop(crtc_ref);
            }
        }
        drop(timers);

        *self.cmd_queue.lock() = KVec::new();
        *self.pending_scanout.lock() = [const { None }; HEADS];
        *self.settle_repaint.lock() = [const { None }; HEADS];
        *self.strip_hashes.lock() = [const { None }; HEADS];
        // Cancel the queued drain and reclaim the `ARef<VinoDrmDevice>` the enqueue handed to
        // the workqueue, if it was still pending. Dropping it here releases the self-reference
        // that would otherwise keep this device alive until the work ran.
        drop(self.cmd_work.cancel_sync());

        // A running callback may have taken a batch just before shutdown was published. It has
        // finished now; clear anything it left behind and tear the USB queues down while their
        // parent interface is still in Bound context.
        *self.cmd_queue.lock() = KVec::new();
        *self.pending_scanout.lock() = [const { None }; HEADS];
        *self.settle_repaint.lock() = [const { None }; HEADS];
        *self.strip_hashes.lock() = [const { None }; HEADS];
        *self.video_q.lock() = [const { None }; HEADS];
        *self.video_staging.lock() = [const { None }; HEADS];
        *self.cp_link.lock() = None;
        pr_info!("vino: deferred KMS/video work drained for unplug\n");
    }

    /// Cache `head`'s CRTC gamma LUT (from `RawCrtcState::gamma_lut`) for the scanout to apply, or
    /// clear it (identity) with `None`. Each `drm_color_lut` channel is reduced to 8 bits.
    pub(super) fn update_gamma(&self, head: usize, lut: Option<&[bindings::drm_color_lut]>) {
        let cached = lut.map(|entries| {
            let mut t = [0u8; 768];
            for i in 0..256 {
                // Identity past the end of a short LUT.
                let e = entries.get(i);
                t[i] = e.map_or(i as u8, |c| (c.red >> 8) as u8);
                t[256 + i] = e.map_or(i as u8, |c| (c.green >> 8) as u8);
                t[512 + i] = e.map_or(i as u8, |c| (c.blue >> 8) as u8);
            }
            t
        });
        let changed = if let Some(slot) = self.gamma.lock().get_mut(head) {
            if *slot == cached {
                false
            } else {
                *slot = cached;
                true
            }
        } else {
            false
        };
        if changed {
            self.strip_hashes.lock()[head] = None;
            self.keyframe_pending
                .fetch_or(1u32 << head, Ordering::Release);
        }
    }

    /// Snapshot `head`'s cached gamma LUT for a scanout pass (`Copy`, so no lock is held
    /// afterwards).
    pub(super) fn gamma_snapshot(&self, head: usize) -> Option<[u8; 768]> {
        self.gamma.lock().get(head).copied().flatten()
    }

    /// Record whether this dock has engaged its CP cipher (`wsub=0x45` acks > 0). The plane
    /// scanout path is gated on it, so pushing frames at a dock whose CP channel is dead cannot
    /// fault it. Set by the bring-up work item once the CP setup completes.
    pub(super) fn set_cp_engaged(&self, engaged: bool) {
        self.cp_engaged
            .store(engaged, core::sync::atomic::Ordering::SeqCst);
    }

    /// Pause the background CP loop and let any iteration which passed its check finish. The mode
    /// worker calls this before taking its timestamp anchor; the fixed delay therefore cannot move
    /// any event relative to the mode-set itself.
    fn begin_cp_timeline(&self) {
        self.cp_timeline_exclusive.store(true, Ordering::Release);
        fsleep(Delta::from_millis(PROMPT_KEEPALIVE_QUIESCE_MS));
    }

    fn end_cp_timeline(&self) {
        self.cp_timeline_exclusive.store(false, Ordering::Release);
    }

    /// Used by `BringUp`'s long-lived keepalive worker. It deliberately remains cheap because that
    /// worker checks it every millisecond while an activation sequence is in progress.
    pub(super) fn cp_timeline_exclusive(&self) -> bool {
        self.cp_timeline_exclusive.load(Ordering::Acquire)
    }

    /// Publish the engaged CP session so the KMS callbacks can send runtime CP messages.
    /// Called once by the bring-up work item after the dock acks (`acks > 0`). `wire_seq`/
    /// `counter` are the next free values past the bring-up CP setup.
    pub(super) fn publish_session(
        &self,
        dev: &BoundInterface<'_>,
        ks: &[u8; 16],
        riv: &[u8; 8],
        wire_seq: u32,
        counter: u16,
    ) {
        // DLM keeps a 4096-byte EP84 read POSTED essentially all the time, not just around each
        // runtime EP02 write: a corpus-wide cadence survey puts its read-outstanding duty cycle at
        // 36-100% of wall time against vino's 3-22%. Its max-in-flight is 1, so this is about the
        // read being continuously pending rather than about depth -- but a depth-1 queue drained
        // synchronously leaves the endpoint un-posted between calls, which is exactly the gap.
        // Match `super::EP84_QUEUE_DEPTH` so one URB stays posted while others are reaped.
        let ep84_q = match dev.ctrl_in_queue(super::EP84_QUEUE_DEPTH, 4096) {
            Ok(q) => Some(q),
            Err(e) => {
                pr_warn!("vino: persistent EP84 queue open failed ({e:?}); using sync fallback\n");
                None
            }
        };
        if CP_KEY_DEBUG {
            // Dev-only: print the live CP session key and both direction nonces so a usbmon
            // capture of vino's own session can be decrypted offline with
            // `scripts/decrypt-dlm-cp.py` -- the same tool used on DLM. Without this the OUT/IN
            // dialogue is opaque (the key never appears on the wire: it is the whitened SKE key
            // vino generated itself), so a DLM-vs-vino dialogue diff, and the dock's own
            // `sub=0x0c` firmware trace, are unreadable. Turn off before any non-development
            // build. See `docs/BLOCKER.md` (dark-panel triage).
            let mut riv_in = *riv;
            riv_in[7] ^= 0x01;
            pr_info!(
                "vino: CPKEYS key={:02x?} riv_out={:02x?} riv_in={:02x?} wire_seq={wire_seq} ctr={counter}\n",
                ks,
                riv,
                &riv_in
            );
        }
        *self.cp_link.lock() = Some(CpLink {
            ks: *ks,
            riv: *riv,
            wire_seq,
            counter,
            ep84_q,
        });
    }

    /// Stash the per-head video keys the bring-up CP setup generated at the `id=0x32` slot (see
    /// the doc comment on the `video_keys` field). Called once, alongside
    /// [`publish_session`](Self::publish_session), when the dock has engaged CP.
    pub(super) fn set_video_keys(&self, keys: [[u8; 32]; HEADS]) {
        *self.video_keys.lock() = keys;
    }

    /// Status polls issued immediately before the first video presentation of a mode generation.
    ///
    /// **Cut from 75x16ms (1.2 s) to 2 on 2026-07-23.** The 1.2 s window was read off DLM as a
    /// "readiness wait", but a merged host/dock timeline of
    /// `captures/dlm-wake-ab-20260722-150209/` (DLM lighting both panels) shows DLM does the
    /// opposite: its stream-enable bracket runs 6.264-6.505 s and its **first EP08 video goes out
    /// at 6.406 s -- during the bracket**, 140 ms after the first marker. The dock then programs
    /// the downstream pixel clock at 7.278 s (`2990d`/`3f32a` in its firmware trace) and the panels
    /// light. vino was taking 3.9 s from bracket to first frame, 28x DLM's gap; the dock gave up on
    /// activation and fell back to its `38c41` idle tick (210x per capture, never once in DLM's).
    /// The polls are kept -- DLM does interleave status traffic here -- but not the stall.
    /// See `docs/BLOCKER.md`.
    ///
    /// **2026-07-23:** the readiness window now lives back at the end of
    /// [`modeset_bracket_post`], where the known-lit `a6f4c8b` build had it, so this inline
    /// pre-write copy is down to a token two polls. Do not restore 75x16ms *here* -- having it in
    /// both places is what produced a 3.9 s bracket-to-first-frame gap against DLM's 140 ms.
    const PREWRITE_POLLS: u32 = 2;
    const PREWRITE_POLL_MS: u64 = 1;
    /// One `id=0x14 sub=0x000c` device-status poll — the same message the EDID readiness loop uses.
    fn poll_status(&self, dev: &BoundInterface<'_>) {
        let _ = self.send_cp(dev, 0x14, 0, |ctr| super::cp::device_query_req(ctr, 0x000c));
    }

    /// One `id=0x16 sub=0x2e|0x2f` stream/display marker. **State lives in byte 23**, not byte 22
    /// (byte 22 is constantly `1` — reading it makes every marker look like state=1).
    fn stream_marker(&self, dev: &BoundInterface<'_>, head: u8, sub: u16, st: u8) {
        let _ = self.send_cp(dev, 0x16, 0, |ctr| {
            super::cp::stream_marker(ctr, head, sub, st)
        });
    }

    /// Emit a run of DLM's `(2f, 2e)` marker pairs for one head, polling between pairs.
    ///
    /// In both decrypted DLM captures the two markers of a pair go out back-to-back (~1 ms apart)
    /// and consecutive pairs are separated by tens of milliseconds of status/EDID traffic, so the
    /// poll belongs *between* pairs rather than inside one.
    /// Currently unused: the brackets were reverted on 2026-07-23 to the known-lit `a6f4c8b`
    /// shape, which is not expressible as clean `(2f, 2e)` pairs (it has an unpaired `2f(1)`).
    /// Kept because it is the exact form the decrypted DLM captures decode to, and is what the
    /// brackets should return to once a lit panel gives a baseline to A/B against.
    #[allow(dead_code)]
    fn stream_marker_pairs(&self, dev: &BoundInterface<'_>, head: u8, pairs: &[(u8, u8)]) {
        for (i, &(state_2f, state_2e)) in pairs.iter().enumerate() {
            if i > 0 {
                self.poll_status(dev);
            }
            self.stream_marker(dev, head, 0x2f, state_2f);
            self.stream_marker(dev, head, 0x2e, state_2e);
        }
    }

    /// First half of the per-head stream-enable bracket, as `(2f state, 2e state)` pairs.
    ///
    /// **Corrected 2026-07-22** from `captures/max-cold-20260721-235609/cp-decrypted.json` (a real
    /// DLM mode change on head 1), which sends four pairs before the `id=0x48` mode-set:
    ///
    /// ```text
    /// (2f:1, 2e:3)  (2f:1, 2e:0)  (2f:0, 2e:0)  (2f:1, 2e:3)
    /// ```
    ///
    /// vino previously sent only the first of those four. A **wake** (the first mode-set since the
    /// CRTC was enabled) has no pre-half at all -- see [`modeset_bracket_post`].
    /// **Reverted 2026-07-23 to the shape of `drm` commit `a6f4c8b` ("Pixels :D", 2026-07-22
    /// 01:36) -- the last build observed to actually light both panels.** That build sent this
    /// half unconditionally, for a wake and a mode change alike, as a single `2f(1) 2e(3)` pair
    /// followed by one status poll. The four-pair form below it, and the wake/mode-change split in
    /// [`modeset_bracket_post`], were derived from capture analysis *after* the panels had gone
    /// dark; no build carrying them has ever lit a panel. Restoring the known-lit shape first, and
    /// re-deriving from captures only once pixels are back, is the cheaper order.
    fn modeset_bracket_pre(&self, dev: &BoundInterface<'_>, head: u8) {
        self.stream_marker(dev, head, 0x2f, 1);
        self.stream_marker(dev, head, 0x2e, 3);
        self.poll_status(dev);
    }

    /// Sleep until an absolute millisecond offset from the mode-set anchor.
    ///
    /// Chaining relative sleeps let scheduler delay accumulate: the first paced hardware capture
    /// reached video at +131 ms despite requesting the right individual delays. Absolute deadlines
    /// make later events catch up rather than carrying every earlier overrun forward.
    fn wait_mode_offset(anchor: Instant<Monotonic>, target_ms: i64) {
        let elapsed_ms = (Instant::<Monotonic>::now() - anchor).as_millis();
        if elapsed_ms < target_ms {
            fsleep(Delta::from_millis(target_ms - elapsed_ms));
        }
    }

    /// Emit the post-mode-set half of the bracket plus DLM's three pre-video status polls, stopping
    /// at the exact point where DLM begins its first video presentation.
    ///
    /// A **wake** (`captures/dlm-wake-ab-20260722-150209/cp-decrypted.json`, the only decrypted
    /// capture of DLM actually lighting the panels) sends the `id=0x48` mode-set for both heads
    /// *first*, with no pre-half, and then five pairs per head -- note `2e:3` twice:
    ///
    /// ```text
    /// (2f:1, 2e:3)  (2f:1, 2e:3)  (2f:1, 2e:0)  (2f:1, 2e:0)  (2f:0, 2e:0)
    /// ```
    ///
    /// A **mode change** on an already-live head sends four pairs, ramping back down to `(0, 0)`:
    ///
    /// ```text
    /// (2f:1, 2e:0)  (2f:0, 2e:0)  (2f:1, 2e:0)  (2f:0, 2e:0)
    /// ```
    ///
    /// vino previously sent one hybrid sequence for both cases, with an unpaired `2f:1` and no
    /// second `2e:3`, and never distinguished a wake from a mode change.
    /// **Reverted 2026-07-23 to `a6f4c8b`'s known-lit order** (capture record numbers n=7..36):
    /// `poll 2f(1) 2e(0) poll 2f(1) poll 2e(0) 2f(0) 2e(0)`, then [`POST_MODESET_POLLS`] paced
    /// polls. Note the deliberately **unpaired `2f(1)`** in the middle -- a later pass "corrected"
    /// that to clean pairs and split wake from mode-change, and the panels have been dark ever
    /// since. `wake` is retained for logging only; both cases send this one sequence, as the lit
    /// build did.
    ///
    ///
    /// The full physical DLM lifecycle capture (`step2-plug-monitor1`) is unambiguous: mode-set at
    /// +29.989 s, the open markers through +30.015 s, status polls at +30.078/+30.084/+30.099 s,
    /// first EP0b video also at +30.099 s, then final `(2f:0,2e:0)` at +30.112/+30.114 s while
    /// video remains in flight.
    fn modeset_bracket_post_open(
        &self,
        dev: &BoundInterface<'_>,
        head: u8,
        anchor: Instant<Monotonic>,
    ) {
        self.poll_status(dev);
        Self::wait_mode_offset(anchor, 5);
        self.stream_marker(dev, head, 0x2f, 1);
        Self::wait_mode_offset(anchor, 9);
        self.stream_marker(dev, head, 0x2e, 3);
        Self::wait_mode_offset(anchor, 12);
        self.stream_marker(dev, head, 0x2f, 1);
        Self::wait_mode_offset(anchor, 14);
        self.stream_marker(dev, head, 0x2e, 3);
        Self::wait_mode_offset(anchor, 20);
        self.stream_marker(dev, head, 0x2f, 1);
        // DLM puts a status poll at the same millisecond as the last 2f(1).
        self.poll_status(dev);
        Self::wait_mode_offset(anchor, 26);
        self.stream_marker(dev, head, 0x2e, 0);
        // There is a measured 63-ms quiet interval, then three polls at +89/+95/+110 ms. The last
        // poll and first video bytes share the same capture millisecond.
        Self::wait_mode_offset(anchor, 89);
        self.poll_status(dev);
        Self::wait_mode_offset(anchor, 95);
        self.poll_status(dev);
        Self::wait_mode_offset(anchor, PROMPT_VIDEO_MS);
        self.poll_status(dev);
    }

    /// Close the post-mode-set bracket after prompt video has started.
    ///
    /// The first close marker is +13 ms from video and the second is +15 ms. Background keepalive
    /// resumes immediately after this pair and supplies DLM's continuing ~60-Hz status dialogue.
    fn modeset_bracket_post_close(
        &self,
        dev: &BoundInterface<'_>,
        head: u8,
        anchor: Instant<Monotonic>,
    ) {
        Self::wait_mode_offset(anchor, PROMPT_CLOSE_2F_MS);
        self.stream_marker(dev, head, 0x2f, 0);
        Self::wait_mode_offset(anchor, PROMPT_CLOSE_2E_MS);
        self.stream_marker(dev, head, 0x2e, 0);
    }

    /// Continuously present one already-encoded activation carrier for at least `duration_ms`.
    ///
    /// This deliberately performs no CP transaction between presentations. `BringUp` runs the
    /// status/heartbeat dialogue concurrently on another work item; doing it here would put a
    /// 10--15 ms control round-trip between tiny black frames and recreate the endpoint starvation
    /// this path exists to remove. A persistent eight-URB queue bounds how far submission can run
    /// ahead, so elapsed wall time closely follows actual endpoint progress rather than merely
    /// copying an arbitrary number of frames into unbounded memory.
    fn submit_prompt_training(
        &self,
        dev: &BoundInterface<'_>,
        head: u8,
        want: u64,
        frames: &[KVec<u8>],
        duration_ms: i64,
        with_arm: bool,
    ) -> Result<u32> {
        if frames.is_empty() {
            return Err(kernel::error::code::EINVAL);
        }
        const XFER: usize = 65536;
        let head_i = head as usize;
        let head_bit = 1u32 << head;
        let image_len: usize = frames.iter().map(|f| f.len()).sum();
        let arm = if with_arm
            && self.arm_prefix_pending.load(Ordering::Acquire) & head_bit != 0
        {
            self.build_arm_burst_buf(head_i)
        } else {
            None
        };
        if with_arm && arm.is_none() {
            return Err(kernel::error::code::ENOMEM);
        }
        let startup = arm.is_some();
        let seq0 = self.scanout_seq.lock()[head_i];
        let started = Instant::<Monotonic>::now();
        let mut repeat = 0u32;

        loop {
            if self.shutting_down.load(Ordering::Acquire)
                || self.modeset_requested[head_i].load(Ordering::Acquire) != want
                || self.modeset_active[head_i].load(Ordering::Acquire) != want
            {
                return Err(kernel::error::code::ENODEV);
            }

            let seq = seq0.wrapping_add(repeat);
            let trailer = super::video::wht::frame_trailer(head, seq);
            let arm_slice: &[u8] = if repeat == 0 {
                arm.as_ref().map_or(&[], |a| &a[..])
            } else {
                &[]
            };
            let wire_len = arm_slice.len() + image_len + trailer.len();
            {
                let mut staging_slots = self.video_staging.lock();
                let staging_slot = &mut staging_slots[head_i];
                if staging_slot.is_none() {
                    let mut staging = KVec::new();
                    staging.resize(XFER, 0, GFP_KERNEL)?;
                    *staging_slot = Some(staging);
                }
                let staging = staging_slot.as_mut().ok_or(kernel::error::code::ENOMEM)?;

                let mut queues = self.video_q.lock();
                let queue_slot = &mut queues[head_i];
                if queue_slot.is_none() {
                    *queue_slot = Some(dev.video_queue(head_i, 8, XFER)?);
                    pr_info!(
                        "vino: head={} persistent video queue opened by prompt training\n",
                        head
                    );
                }
                let queue = queue_slot
                    .as_mut()
                    .ok_or(kernel::error::code::ENODEV)?;

                let arm_parts = usize::from(!arm_slice.is_empty());
                let part_count = arm_parts + frames.len() + 1;
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
                        } else if part_i < arm_parts + frames.len() {
                            &frames[part_i - arm_parts][..]
                        } else {
                            &trailer[..]
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
                    if let Err(e) = queue.send(dev.io(), dst, super::timeout()) {
                        let _ = dev.clear_video_halt(head_i);
                        return Err(e);
                    }
                    wire_off += data_len;
                }
            }

            if repeat == 0 && startup {
                self.arm_prefix_pending
                    .fetch_and(!head_bit, Ordering::Release);
                self.sustain_until.lock()[head_i] =
                    Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
                pr_info!(
                    "vino: prompt head={} ARM+black frame submitted {} ms after bracket-open ({} B)\n",
                    head,
                    (Instant::<Monotonic>::now() - started).as_millis(),
                    wire_len
                );
            }
            repeat = repeat.wrapping_add(1);
            self.scanout_seq.lock()[head_i] = seq0.wrapping_add(repeat);

            if repeat > 0
                && (Instant::<Monotonic>::now() - started).as_millis() >= duration_ms
            {
                break;
            }
        }

        pr_info!(
            "vino: prompt head={} training phase complete ({} presentations, {} ms)\n",
            head,
            repeat,
            (Instant::<Monotonic>::now() - started).as_millis()
        );
        Ok(repeat)
    }

    /// Activate both downstream heads as one dock-wide wake transaction.
    ///
    /// DLM does not finish one head before starting the other. Both the physical-cold capture
    /// (`dlm-coldplug-withmon-160457`) and the successful takeover of a Vino-dark dock
    /// (`dlm-takeover-dark-20260726-0820`) send the two `id=0x48` mode-sets first (29 ms and
    /// 71 ms apart respectively), then interleave both heads' stream markers and video. Vino used
    /// to run a complete mode/marker/440-ms-carrier transaction for head 0 before sending head 1's
    /// mode-set; the measured gap was 595 ms and the same dock remained dark until DLM took over.
    ///
    /// The schedule is now a literal replay of the cold capture's decoded timeline (see the
    /// [`cold`] module). Batching the two heads was necessary but not sufficient: the batched
    /// build compressed DLM's 1.3-second choreography into 27 ms and the dock's firmware trace
    /// still took the idle branch. The dock is given the same event *ordering* it always was —
    /// what changes here is that it also gets DLM's silent window and its mid-bracket EDID
    /// re-read.
    ///
    /// This is deliberately only the *dual wake* path. A single-head hotplug or an already-live
    /// mode change keeps using the per-head, absolute timeline below, which was derived from the
    /// detailed one-monitor lifecycle capture.
    fn activate_dual_wake(
        &self,
        dev: &BoundInterface<'_>,
        timings: [Option<super::cp::Timing>; HEADS],
    ) {
        let mut prompts: [Option<KVec<KVec<u8>>>; HEADS] =
            core::array::from_fn(|_| None);
        let mut keys = [0u64; HEADS];
        let mut valid = 0u32;

        // Pre-encode both tiny carriers before excluding the keepalive or starting either
        // mode-set. Encoding work must not turn DLM's back-to-back mode pair into a serial pair.
        for head in 0..HEADS {
            let Some(timing) = timings[head] else {
                continue;
            };
            let key = timing_key(&timing);
            if self.modeset_requested[head].load(Ordering::Acquire) != key
                || self.modeset_active[head].load(Ordering::Acquire) != 0
            {
                continue;
            }
            pr_info!(
                "vino: DUAL-WAKE timing head={} h={}+{} fp={} sw={} v={}+{} fp={} sw={} refresh={} pclk10k={} field42=0x{:04x}\n",
                head,
                timing.hactive,
                timing.hblank,
                timing.hsync_front,
                timing.hsync_width,
                timing.vactive,
                timing.vblank,
                timing.vsync_front,
                timing.vsync_width,
                timing.refresh_hz,
                timing.pixel_clock_10khz,
                timing.field42
            );
            let w_pad = (timing.hactive as usize + super::video::wht::STRIP_W - 1)
                & !(super::video::wht::STRIP_W - 1);
            let h_pad = (timing.vactive as usize + super::video::wht::STRIP_H - 1)
                & !(super::video::wht::STRIP_H - 1);
            prompts[head] = super::video::wht::black_frame_ep08(w_pad, h_pad, head as u8).ok();
            keys[head] = key;
            valid |= 1u32 << head;
        }
        if valid.count_ones() < 2 {
            return;
        }

        // One clock for the whole transaction. Every event below is scheduled against this
        // anchor, so a slow send makes the next event catch up instead of pushing the rest of the
        // schedule out -- the failure mode that invalidated the first paced hardware A/B.
        self.begin_cp_timeline();
        let anchor = Instant::<Monotonic>::now();
        let mut sent = 0u32;
        let mut started = 0u32;

        // Two cursors walk the sorted schedules; `run_cp_until` drains everything due at or
        // before a given offset. That keeps markers, polls and EDID reads correctly interleaved
        // without writing the merge out by hand.
        let mut mi = 0usize;
        let mut pi = 0usize;
        let mut ei = 0usize;

        macro_rules! cp_until {
            ($limit:expr) => {{
                let limit: i64 = $limit;
                loop {
                    let nm = cold::MARKERS.get(mi).map(|m| m.0);
                    let np = cold::POLLS.get(pi).copied();
                    let ne = cold::EDID.get(ei).map(|e| e.0);
                    let next = [nm, np, ne]
                        .into_iter()
                        .flatten()
                        .filter(|&o| o <= limit)
                        .min();
                    let Some(off) = next else { break };
                    Self::wait_mode_offset(anchor, off);
                    if nm == Some(off) {
                        let (_, head, sub, state) = cold::MARKERS[mi];
                        if sent & (1u32 << head) != 0 {
                            self.stream_marker(dev, head, sub, state);
                        }
                        mi += 1;
                    } else if np == Some(off) {
                        self.poll_status(dev);
                        pi += 1;
                    } else {
                        let (_, head, fetch) = cold::EDID[ei];
                        // Re-read the sink's EDID exactly where DLM does. The reply is
                        // deliberately discarded: this is the dock-side DDC transaction the lit
                        // trace answers with `2fab9`/`3e54c`/`30297`, not a source of new modes,
                        // and re-parsing it here would risk a hotplug in the middle of a
                        // mode-set.
                        let _ = self.send_cp(dev, 0x15, 0, |ctr| {
                            if fetch {
                                super::cp::get_edid_req(ctr, head)
                            } else {
                                super::cp::get_edid_req_sub(ctr, 0x0020, head)
                            }
                        });
                        ei += 1;
                    }
                }
            }};
        }

        // Both mode-sets go out first, 29 ms apart, before any video.
        for head in 0..HEADS {
            let bit = 1u32 << head;
            if valid & bit == 0 {
                continue;
            }
            let Some(timing) = timings[head] else {
                continue;
            };
            if head == 1 {
                cp_until!(cold::H1_MODE - 1);
                Self::wait_mode_offset(anchor, cold::H1_MODE);
            }
            let mode_ok = self
                .send_cp(dev, 0x48, 0, |ctr| {
                    super::cp::set_mode(ctr, head as u8, &timing)
                })
                .is_ok();
            if !mode_ok || self.modeset_requested[head].load(Ordering::Acquire) != keys[head] {
                continue;
            }
            self.modeset_active[head].store(keys[head], Ordering::Release);
            self.sustain_until.lock()[head] =
                Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
            self.arm_prefix_pending.fetch_or(bit, Ordering::Release);
            self.keyframe_pending.fetch_or(bit, Ordering::Release);
            self.strip_hashes.lock()[head] = None;
            sent |= bit;
        }

        // *** The silent window. *** Nothing goes out on EP02 between the head-1 mode-set and
        // +1016 ms. The background keepalive is already excluded, and this loop simply has no
        // scheduled events in the interval, so the dock sees the ~1 s of quiet that every lit
        // capture contains and no vino run ever has.
        Self::wait_mode_offset(anchor, cold::QUIET_END);

        // Bracket, status polls and the mid-bracket EDID re-read, up to the first video.
        cp_until!(cold::H0_VIDEO - 1);

        for (head, at) in [(0usize, cold::H0_VIDEO), (1usize, cold::H1_VIDEO)] {
            cp_until!(at - 1);
            Self::wait_mode_offset(anchor, at);
            let bit = 1u32 << head;
            if sent & bit == 0 {
                continue;
            }
            // Exactly one ARM+carrier presentation, so the closing markers that DLM sends 33 ms
            // later are not pushed out behind a blocking multi-frame submission.
            if prompts[head].as_ref().is_some_and(|frames| {
                self.submit_prompt_training(
                    dev,
                    head as u8,
                    keys[head],
                    frames,
                    PROMPT_TRAINING_OPEN_MS,
                    true,
                )
                .is_ok()
            }) {
                started |= bit;
            }
        }

        // Remaining polls and the closing markers.
        cp_until!(i64::MAX);
        self.end_cp_timeline();

        // Keep both endpoints busy well past the close. DLM programs the head-0 clock 377 ms
        // after its first video bytes and head 1's 589 ms after its own, so the carrier has to
        // outlive the bracket rather than stopping with it.
        let tail_started = Instant::<Monotonic>::now();
        while (Instant::<Monotonic>::now() - tail_started).as_millis() < cold::CARRIER_TAIL_MS {
            for head in 0..HEADS {
                if started & (1u32 << head) == 0 {
                    continue;
                }
                if let Some(frames) = prompts[head].as_ref() {
                    if let Err(e) = self.submit_prompt_training(
                        dev,
                        head as u8,
                        keys[head],
                        frames,
                        PROMPT_TRAINING_OPEN_MS,
                        false,
                    ) {
                        pr_info!(
                            "vino: dual-wake prompt head={} training failed ({e:?})\n",
                            head
                        );
                    }
                }
            }
        }
        pr_info!(
            "vino: DLM cold-replay dual wake complete after {} ms (mode/started masks 0x{:x}/0x{:x})\n",
            (Instant::<Monotonic>::now() - anchor).as_millis(),
            sent,
            started
        );
    }

    /// Build one head's **cold** video-arm burst (2560 B), matching the 4-way-confirmed cold-plug
    /// pcaps (`docs/EP08-ARM-BURST.md`, 2026-07-19) — DLM PREPENDS this to frame 0's video in one URB.
    /// Records: #0/#1/#4/#5 plaintext; #6/#7 fixed `type=4` `0a 00 04 …`; #2/#3/#8/#9 sealed under
    /// THIS head's video channel key/nonce (from the per-head SKE plus
    /// `riv_h ^ (0x08 | head) @ byte7`) sharing ONE block counter (seq 0,1,2,71). #2/#3 = 16 B
    /// (`04 00 08 04 03 00` + 10 host-random). #8/#9 = the fixed 1090-byte video-decode template followed by 14
    /// host-random bytes. The four variable fields are separate DLM CTR-DRBG requests of lengths
    /// 10, 10, 14, and 14 respectively (live Frida capture, 2026-07-21).
    fn build_arm_burst_buf(&self, head: usize) -> Option<KVec<u8>> {
        let keys = *self.video_keys.lock();
        let vkey: [u8; 16] = keys[head][0..16].try_into().ok()?;
        let vnonce: [u8; 8] = keys[head][16..24].try_into().ok()?;
        let h = head as u16;
        // ONE running block counter shared across the sealed records #2/#3/#8/#9 (the dock's video
        // channel advances a single counter): #2 seq0(+1), #3 seq1(+1), #8 seq2(+69), #9 seq71.
        let mut seal_seq: u32 = 0;
        let mut buf = KVec::with_capacity(2560, GFP_KERNEL).ok()?;
        for (i, &(_wire_type, sub_base, aux, body_len)) in
            super::cp::VIDEO_ARM_BURST.iter().enumerate()
        {
            let sub = sub_base.wrapping_add(h);
            match i {
                2 | 3 => {
                    // Sealed under the per-head video key/nonce. Content = fixed 6-byte header +
                    // 10 host-random bytes; seq is the shared block counter (16 B = 1 block each).
                    let mut content = [0u8; 16];
                    content[..6].copy_from_slice(&[0x04, 0x00, 0x08, 0x04, 0x03, 0x00]);
                    super::rng::fill(&mut content[6..]);
                    let frame =
                        super::cp::seal_video_arm(&vkey, &vnonce, sub, aux, seal_seq, &content)
                            .ok()?;
                    seal_seq += 1;
                    buf.extend_from_slice(&frame, GFP_KERNEL).ok()?;
                }
                6 | 7 => {
                    // type=4 but FIXED plaintext (not encrypted, no MAC): a 32-byte frame whose
                    // 16-byte body is the constant `0a 00 04 00 …00 10 00 00 00 00` (0x10 @ byte 11).
                    let mut f = [0u8; 32];
                    f[2..4].copy_from_slice(&0x001cu16.to_le_bytes());
                    f[4..8].copy_from_slice(&4u32.to_le_bytes());
                    f[8..10].copy_from_slice(&sub.to_le_bytes());
                    f[10..12].copy_from_slice(&aux.to_le_bytes());
                    f[16..32].copy_from_slice(&[
                        0x0a, 0x00, 0x04, 0x00, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0, 0,
                    ]);
                    buf.extend_from_slice(&f, GFP_KERNEL).ok()?;
                }
                8 | 9 => {
                    // Sealed 1104-byte records = the per-frame video-decode setup. Sealed under the
                    // per-head video key with the shared block counter (69 blocks each).
                    // PLAINTEXT RECOVERED 2026-07-19 (`video_arm_content.rs`): a live gdb capture of
                    // DLM read the per-head video seal key (setter DLM+0x85c830) + content riv (AES fn
                    // DLM+0x269dd0) while usbmon captured frame-0; the sealed #8/#9 were then AES-CTR
                    // decrypted offline (entropy 7.8 -> 1.78). The fixed prefix is per-mode
                    // (2560x1440), assumed head-invariant. See captures/ep08-1104b-decoded-20260719/.
                    debug_assert_eq!(body_len, 1104);
                    let base: &[u8] = if i == 8 {
                        &super::video_arm_content::VIDEO_ARM_8_CONTENT
                    } else {
                        &super::video_arm_content::VIDEO_ARM_9_CONTENT
                    };
                    // ★ 2026-07-21: all decoded sessions/heads/records have the same bytes 0..1090
                    // and differ only in bytes 1090..1104. A filtered live trace then caught the
                    // decisive source: immediately before sealing #8 and #9, DLM calls its mbedTLS
                    // CTR-DRBG for exactly 14 bytes; each output is byte-identical to that record's
                    // tail. Build a fresh tail for each record. Keep the large fixed portion in
                    // static storage and assemble into a KVec so we do not put 1104 bytes on the
                    // kernel stack.
                    let tail_offset = super::video_arm_content::VIDEO_ARM_RANDOM_TAIL_OFFSET;
                    debug_assert_eq!(body_len - tail_offset, 14);
                    let mut content = KVec::with_capacity(body_len, GFP_KERNEL).ok()?;
                    content
                        .extend_from_slice(&base[..tail_offset], GFP_KERNEL)
                        .ok()?;
                    let mut random_tail = [0u8; 14];
                    super::rng::fill(&mut random_tail);
                    content.extend_from_slice(&random_tail, GFP_KERNEL).ok()?;
                    debug_assert_eq!(content.len(), body_len);
                    let frame =
                        super::cp::seal_video_arm(&vkey, &vnonce, sub, aux, seal_seq, &content)
                            .ok()?;
                    seal_seq += (body_len / 16) as u32;
                    buf.extend_from_slice(&frame, GFP_KERNEL).ok()?;
                }
                _ => {
                    // wire_type==2 plaintext records (#0/#1/#4/#5).
                    let body = super::cp::video_arm_plaintext_body(i, h);
                    let frame = super::cp::video_arm_plain_frame(sub, &body);
                    buf.extend_from_slice(&frame, GFP_KERNEL).ok()?;
                }
            }
        }
        Some(buf)
    }

    /// Seal and send one interactive CP message on EP02, advancing the session keystream.
    /// `build(counter)` produces the inner CP message for the dock-echoed `counter` it is
    /// handed (e.g. [`super::cp::set_mode`]); `tag_reserved` trailing bytes are dropped before
    /// the live Dl3Cmac is appended. Returns `Ok(())` as a **no-op when CP is not engaged**.
    /// The `cp_link` mutex serialises the deferred KMS worker and continuous keepalive. Callers are
    /// sleepable; DRM atomic callbacks queue commands instead of invoking this blocking path.
    pub(super) fn send_cp(
        &self,
        dev: &BoundInterface<'_>,
        id: u16,
        tag_reserved: usize,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
    ) -> Result {
        let mut guard = self.cp_link.lock();
        let Some(link) = (&mut *guard).as_mut() else {
            return Ok(()); // CP not engaged -- nothing to send
        };
        let msg = build(link.counter)?;
        let content = &msg[..msg.len().saturating_sub(tag_reserved)];
        let frame = super::cp::seal_interactive(&link.ks, &link.riv, id, link.wire_seq, content)?;
        // Checkpoint (2026-07-16, post-silent-freeze investigation): this is the ONE blocking I/O
        // call shared by every runtime CP send (mode-set, DDC/CI VCP, cursor) -- bracketing it
        // here covers all call sites for free. A live run froze with no oops/panic right after a
        // rapid CRTC mode-switch cycle; if this is where it hangs, "entering bulk_send" will be
        // the last line ever logged (bulk_send's own `timeout()` bound means it should not be
        // able to block indefinitely on its own -- if it does, that is itself a finding).
        dev.ctrl_send(&frame, super::timeout(), GFP_KERNEL)?;
        link.wire_seq = link
            .wire_seq
            .wrapping_add(((content.len() + 15) / 16) as u32);
        link.counter = link.counter.wrapping_add(1);
        // ★ 2026-07-20: DRAIN the dock's reply on EP84 after every send. DLM runs strict CP
        // lockstep -- for every EP02-OUT it submits an EP84 IN read and reaps the sealed reply
        // (usbmon during a mode-set: 76 OUT / 76 IN). vino's runtime `send_cp` previously ONLY
        // wrote and NEVER read (usbmon: 855 OUT / 0 IN), so the dock's reply buffer was never
        // drained; the dock stops answering and hard-resets a few seconds later with the panels
        // never lighting. The bring-up's async EP84 queue is a local that's dropped when bring-up
        // returns, so runtime sends had no reader at all. Read one reply here (best-effort:
        // NAK/short/timeout is fine -- not every message elicits one, and the point is to keep the
        // channel drained, not to parse it). This is the CP half of the lockstep DLM keeps for the
        // whole session; see `docs/DLM-WIRE-ANNOTATION.md`.
        // DLM always posts one 4096-byte EP84 URB. Keep the request length identical even when the
        // expected acknowledgement is short; larger logical replies are naturally fragmented.
        let mut reply = KVec::from_elem(0u8, 4096, GFP_KERNEL)?;
        if let Some(q) = link.ep84_q.as_mut() {
            let _ = q.recv(dev.io(), &mut reply, super::cp_reply_timeout());
        } else {
            let _ = dev.ctrl_recv(&mut reply, super::cp_reply_timeout(), GFP_KERNEL);
        }
        Ok(())
    }

    /// Consume the dock's *unprompted* EP84 pushes, i.e. reads that are not the reply to any of our
    /// writes. Returns how many frames were drained.
    ///
    /// **Why (2026-07-23, corpus cadence survey).** The EP84-submits-per-EP02-submit ratio is a
    /// stable signature: DLM on a cold plug runs **1.116-1.157**, vino ran **1.000**. DLM's surplus
    /// is unpaired reads that consume what the dock sends on its own initiative -- the cert, the
    /// `id=0x4c` and `id=0x78` capability blocks, `id=0x2 sub=0x86` heartbeats. vino read exactly
    /// once per write, so anything the dock pushed *between* writes sat in its buffer until vino
    /// happened to send something, and a dock with a bounded queue has no reason to keep offering
    /// it. Raising the queue depth fixed how often a URB is *posted* (3-22% -> 99.9% of wall time)
    /// but not this: every read was still paired.
    ///
    /// Bounded, and each read is an immediate completion check, so an idle channel does not delay
    /// the keepalive and a chatty one cannot monopolise the caller.
    pub(super) fn drain_cp_pushes(&self, dev: &BoundInterface<'_>, max: usize) -> usize {
        let mut guard = self.cp_link.lock();
        let Some(link) = (&mut *guard).as_mut() else {
            return 0;
        };
        let Ok(mut reply) = KVec::from_elem(0u8, 4096, GFP_KERNEL) else {
            return 0;
        };
        let mut n = 0;
        while n < max {
            let got = match link.ep84_q.as_mut() {
                // Every queue slot is already posted. This is an opportunistic drain, so never
                // spend the ordinary 8-ms paired-reply timeout waiting for a push which has not
                // happened. That timeout plus the keepalive's 13-ms sleep stretched Vino's
                // measured activation cadence to ~23 ms; DLM polls at ~17 ms in the physical
                // lifecycle capture. A zero timeout still reaps and re-posts any completed slot.
                Some(q) => q.recv(dev.io(), &mut reply, Delta::from_millis(0)),
                None => dev
                    .ctrl_recv(&mut reply, Delta::from_millis(0), GFP_KERNEL)
                    .map(Some),
            };
            // `Ok(None)` is the queue's timeout: nothing pending, so the dock has nothing more to
            // say right now. Any error is treated the same -- this is best-effort drainage.
            match got {
                Ok(Some(len)) if len > 0 => n += 1,
                _ => break,
            }
        }
        n
    }

    /// Cache a head's downstream EDID (read during probe). Bring-up publishes all heads with one
    /// hotplug only after both presence and EDID state are complete; firing here exposed KWin to a
    /// transient no-EDID mode list (including synthetic 1920x1440) before the real EDID arrived.
    /// Out-of-range heads are ignored.
    pub(super) fn set_edid(&self, dev: &VinoDrmDevice, head: usize, blob: KVec<u8>) {
        let mut edids = self.cached_edids.lock();
        let Some(slot) = edids.get_mut(head) else {
            return;
        };
        *slot = Some(blob);
        drop(edids);
        let _ = dev;
    }

    /// Mark a head connected from CP engagement alone (no raw EDID). Bring-up fires one hotplug
    /// after every head's EDID has also been cached, so the compositor never probes partial state.
    /// Called once the head's DISPLAY-CAP push confirms monitor presence.
    pub(super) fn set_connected(&self, dev: &VinoDrmDevice, head: usize) {
        if head >= HEADS {
            return;
        }
        self.heads_present
            .fetch_or(1 << head, core::sync::atomic::Ordering::Release);
        let _ = dev;
    }

    /// Stage 2 (runtime monitor removal): clear a head's connected state. `detect()` reports
    /// Connected when `cached_edid || heads_present bit`, so a genuine disconnect must clear BOTH,
    /// or a head that lost its monitor keeps advertising a phantom output from the stale EDID.
    pub(super) fn set_disconnected(&self, head: usize) {
        if head >= HEADS {
            return;
        }
        self.heads_present
            .fetch_and(!(1u32 << head), core::sync::atomic::Ordering::Release);
        if let Some(slot) = self.cached_edids.lock().get_mut(head) {
            *slot = None;
        }
    }

    /// Re-run head `head`'s EDID probe/fetch/**engage** sequence on the live CP link.
    ///
    /// **Why a runtime monitor replug needs this (2026-07-26).** The dock enables a head's
    /// downstream sink from the EDID *engage* (`id=0x16 sub=0x23`); its firmware trace answers that
    /// message with `27525 <h> <h>` / `2a887 <h> <h>`, and only after that does it reach `2807a` and
    /// start the pixel clock `2990d`. Unplugging the monitor tears that sink state down. vino used to
    /// handle the replug by setting the presence bit, polling readiness and firing a hotplug event --
    /// KWin then re-enabled the CRTC and vino re-sent the mode-set and video, all of which the dock
    /// accepted while leaving the panel dark, because nothing had re-enabled its sink. A *dock*
    /// replug recovered only because full bring-up re-runs the engage.
    ///
    /// So repeat bring-up's per-head sequence -- probe, kick, fetch, engage, post-EDID query -- at
    /// bring-up's measured spacing. Every message is fire-and-forget: the dock's CP reply to each is
    /// a contentless generic ack, so there is nothing to wait for here (that property is exactly what
    /// hid the off23 bug for a month). Errors are ignored; a dropped message just leaves the head
    /// dark as before, and the next presence cycle tries again.
    pub(super) fn reengage_head(&self, dev: &BoundInterface<'_>, head: u8) {
        let step = |id: u16, build: &dyn Fn(u16) -> Result<KVec<u8>>, gap_ms: i64| {
            let _ = self.send_cp(dev, id, 0, |ctr| build(ctr));
            fsleep(Delta::from_millis(gap_ms));
        };
        step(0x15, &|c| super::cp::get_edid_req_sub(c, 0x0020, head), 117);
        step(0x15, &|c| super::cp::get_edid_req_sub(c, 0x0020, head), 115);
        step(0x16, &|c| super::cp::edid_readiness_kick(c, head), 107);
        step(0x15, &|c| super::cp::get_edid_req(c, head), 11);
        step(0x16, &|c| super::cp::edid_engage_req(c, head), 118);
        step(0x16, &|c| super::cp::edid_engage_req(c, head), 107);
        step(0x15, &|c| super::cp::post_edid_query(c, head), 11);
    }

    /// Stage 2 (runtime monitor hotplug): probe whether head `head` currently has a monitor.
    ///
    /// Sends the per-head EDID probe (`id=0x15 sub=0x20`, byte22 = head selector -- the same
    /// selector that unblocked the whole EDID path) and decodes the dock's sealed `0x45` reply.
    /// A present monitor routes the probe to the dock's trusted EDID/display-capability handler
    /// (`id=0x44`/`id=0x194`/`id=0x78`); an empty port gets a bare generic `id=0x14` ack. Returns
    /// `Some(true/false)` on a decodable reply, `None` if CP is down or no reply decoded (caller
    /// treats `None` as "no change", and debounces `Some` transitions). Reuses the live session
    /// `ks/riv/counter` exactly like `send_cp`, so it stays in CP lockstep.
    pub(super) fn probe_head_present(&self, dev: &BoundInterface<'_>, head: u8) -> Option<bool> {
        let mut guard = self.cp_link.lock();
        let link = (&mut *guard).as_mut()?;
        let msg = super::cp::get_edid_req_sub(link.counter, 0x0020, head).ok()?;
        let frame =
            super::cp::seal_interactive(&link.ks, &link.riv, 0x15, link.wire_seq, &msg).ok()?;
        dev.ctrl_send(&frame, super::timeout(), GFP_KERNEL).ok()?;
        link.wire_seq = link.wire_seq.wrapping_add(((msg.len() + 15) / 16) as u32);
        link.counter = link.counter.wrapping_add(1);
        let mut reply = KVec::from_elem(0u8, 4096, GFP_KERNEL).ok()?;
        let got = match link.ep84_q.as_mut() {
            Some(q) => match q.recv(dev.io(), &mut reply, super::cp_reply_timeout()) {
                Ok(Some(n)) if n > 16 => n,
                _ => return None,
            },
            None => match dev.ctrl_recv(&mut reply, super::cp_reply_timeout(), GFP_KERNEL) {
                Ok(n) if n > 16 => n,
                _ => return None,
            },
        };
        let (id, _sub, _ctr) = super::cp::decode_in_lenient(&link.ks, &link.riv, &reply[..got])?;
        Some(matches!(id, 0x44 | 0x194 | 0x78))
    }

    /// Whether head `head`'s presence bit is currently set (for the keepalive to seed its baseline
    /// before watching for runtime connect/remove transitions).
    pub(super) fn head_present(&self, head: usize) -> bool {
        head < HEADS
            && self
                .heads_present
                .load(core::sync::atomic::Ordering::Acquire)
                & (1u32 << head)
                != 0
    }

    /// How many heads on this dock currently have a monitor attached (`present`), for splitting the
    /// shared pixel-rate budget. At least 1, so a lone head (or a momentary all-disconnected read
    /// racing `detect`) gets the whole budget rather than dividing by zero.
    fn connected_heads(&self) -> u32 {
        self.heads_present
            .load(core::sync::atomic::Ordering::Acquire)
            .count_ones()
            .max(1)
    }

    /// This head's even share of the dock's total pixel-rate budget (pixels/sec). `0` = no limit set
    /// yet (budget unknown). Because every head only ever checks its own share, the sum across heads
    /// is guaranteed to stay within the total without a global cross-CRTC check.
    fn own_pixel_budget(&self) -> u32 {
        let total = self
            .dock_pixel_budget
            .load(core::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        total / self.connected_heads()
    }

    /// Queue a [`KmsCmd`] for the async worker and make sure it runs. Pushing never blocks (a
    /// plain `Mutex` push, no I/O); `workqueue::system().enqueue` is a no-op if the worker is
    /// already pending/running, which is exactly what we want -- it will drain this command too
    /// once it gets to run, whether that's this dispatch or the next one. Called from DRM
    /// atomic-commit callbacks, which is why nothing here may block.
    fn queue_cmd(&self, dev: &VinoDrmDevice, cmd: KmsCmd) {
        let mut queue = self.cmd_queue.lock();
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let mut coalesced = false;
        if let KmsCmd::CursorMove { head, x, y } = &cmd {
            // A moving pointer can queue hundreds of positions while a full frame is being
            // compressed. Only the newest unsent position matters; retaining every intermediate
            // point creates an unbounded CP backlog in front of video.
            for queued in queue.iter_mut().rev() {
                if let KmsCmd::CursorMove {
                    head: queued_head, ..
                } = queued
                {
                    if queued_head == head {
                        *queued = KmsCmd::CursorMove {
                            head: *head,
                            x: *x,
                            y: *y,
                        };
                        coalesced = true;
                        break;
                    }
                }
            }
        }
        if !coalesced && queue.push(cmd, GFP_KERNEL).is_err() {
            pr_warn!("vino: KMS command queue allocation failed -- dropping a send\n");
            return;
        }
        // Enqueue while the queue lock still serializes us with `shutdown()`. Otherwise shutdown
        // could cancel an idle work item between this unlock and enqueue, leaving a late work-owned
        // device reference behind after teardown.
        let _ = workqueue::system().enqueue(ARef::from(dev));
        drop(queue);
    }

    /// Publish the latest framebuffer for one head and wake the same deferred worker used by the
    /// blocking runtime CP commands. Replacing an unsent flip is deliberate backpressure: the dock
    /// needs the newest desktop, not every historical compositor buffer. If damaged flips are
    /// coalesced, carry the unsent damage into the newest framebuffer so no intermediate update is
    /// lost without needlessly promoting every busy compositor interval to a full-screen refresh.
    fn queue_scanout(&self, dev: &VinoDrmDevice, mut frame: PendingScanout) {
        let head = frame.head as usize;
        if head >= HEADS {
            return;
        }
        // A real flip carries newer content than any armed settle repaint of an older buffer.
        self.settle_repaint.lock()[head] = None;
        // Flip accounting. The failure this instruments is "the panel keeps the blank buffer
        // forever": the distinguishing question is whether the compositor stopped flipping or
        // whether its flips stopped reaching us, and only a count answers it. Logged at
        // exponentially sparser points so a busy desktop costs nothing.
        let n = self.flips[head].fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_power_of_two() {
            pr_info!("vino: head {head} primary-plane flip #{n}\n");
        }
        let mut pending = self.pending_scanout.lock();
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        if let Some(old) = pending[head].take() {
            if old.w != frame.w || old.h != frame.h || old.rotation != frame.rotation {
                // Damage coordinates are not comparable across a geometry transform. A mode-set
                // already owes a keyframe, but keep this conservative for a rotation-only commit.
                frame.clips[0] = (0, 0, frame.w, frame.h);
                frame.nclips = 1;
            } else if old.nclips + frame.nclips <= MAX_DAMAGE_CLIPS {
                // `frame` names the newest complete framebuffer. Repainting the union of its own
                // damage and every unsent older clip reproduces all intermediate changes directly
                // from that newest image.
                for &clip in &old.clips[..old.nclips] {
                    frame.clips[frame.nclips] = clip;
                    frame.nclips += 1;
                }
            } else {
                // Too many rectangles for the bounded atomic-state payload: collapse their union
                // to one bounding box. This may repaint extra strips, but unlike the previous
                // full-output fallback it remains small for typical pointer/window motion.
                let mut bb = (frame.w, frame.h, 0usize, 0usize);
                for &r in &frame.clips[..frame.nclips] {
                    bb = (bb.0.min(r.0), bb.1.min(r.1), bb.2.max(r.2), bb.3.max(r.3));
                }
                for &r in &old.clips[..old.nclips] {
                    bb = (bb.0.min(r.0), bb.1.min(r.1), bb.2.max(r.2), bb.3.max(r.3));
                }
                if bb.0 < bb.2 && bb.1 < bb.3 {
                    frame.clips[0] = bb;
                    frame.nclips = 1;
                } else {
                    frame.nclips = 0;
                }
            }
        }
        pending[head] = Some(frame);
        let _ = workqueue::system().enqueue(ARef::from(dev));
        drop(pending);
    }
}

impl_has_work! {
    impl HasWork<VinoDrmDevice> for VinoDrmData { self.cmd_work }
}

impl WorkItem for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;

    /// Drain every [`KmsCmd`] queued since the last run, executing each one's real (blocking) CP
    /// send here -- off the DRM atomic-commit calling thread that queued it. See [`KmsCmd`]'s doc
    /// comment for why this exists. Runs in order; a failed send is logged and does not stop the
    /// rest of the batch (matches the previous inline calls, which were each already independent
    /// best-effort sends via `let _ =` / a logged warning).
    fn run(this: ARef<VinoDrmDevice>) {
        // `drm_dev_enter()` holds the parent USB interface in Bound typestate until this worker
        // finishes. If unplug has begun, discard queued transport work without touching USB.
        let data: &VinoDrmData = &this;
        // The I/O window is closed by `disconnect()` before it returns, so once it refuses a token
        // there is no USB left to do: discard the queued transport work.
        let Ok(link) = super::UsbLink::open(&data.io, data.eps) else {
            return;
        };
        let dev = &link;
        let mut next_head = 0usize;
        loop {
            let cmds = core::mem::replace(&mut *data.cmd_queue.lock(), KVec::new());
            // A cold dual-head atomic commit queues both enables together. DLM treats that as one
            // dock-wide wake: both mode-sets precede either head's video. Detect that shape before
            // consuming the owned commands; `Timing` is Copy, so cursor payloads remain untouched
            // for the ordinary command loop below.
            let mut dual_timings: [Option<super::cp::Timing>; HEADS] = [None; HEADS];
            for cmd in cmds.iter() {
                if let KmsCmd::ModeSet { head, timing } = cmd {
                    let head_i = *head as usize;
                    if head_i < HEADS
                        && data.modeset_active[head_i].load(Ordering::Acquire) == 0
                        && data.modeset_requested[head_i].load(Ordering::Acquire)
                            == timing_key(timing)
                    {
                        dual_timings[head_i] = Some(*timing);
                    }
                }
            }
            let dual_wake = dual_timings.iter().flatten().count() >= 2;
            if dual_wake {
                data.activate_dual_wake(dev, dual_timings);
            }
            // Control-plane ordering comes first. An enabling atomic commit queues the plane flip
            // before its CRTC mode-set; finish this head's captured pre/mode/post transaction before
            // selecting a pending framebuffer.
            for cmd in cmds {
                let res = match cmd {
                    KmsCmd::ModeSet { head, timing } => {
                        if dual_wake {
                            // `activate_dual_wake` consumed the current generation for both heads.
                            // A superseding generation queued while it ran is still present in
                            // `cmd_queue` and gets priority on the next outer iteration.
                            continue;
                        }
                        let head_i = head as usize;
                        let key = timing_key(&timing);
                        if data.modeset_requested[head_i].load(Ordering::Acquire) != key {
                            Ok(()) // superseded or disabled while queued
                        } else {
                            pr_info!(
                                "vino: MODESET timing h={}+{} fp={} sw={} v={}+{} fp={} sw={} refresh={} pclk10k={} field42=0x{:04x}\n",
                                timing.hactive, timing.hblank, timing.hsync_front, timing.hsync_width,
                                timing.vactive, timing.vblank, timing.vsync_front, timing.vsync_width,
                                timing.refresh_hz, timing.pixel_clock_10khz, timing.field42
                            );
                            let wake = data.modeset_active[head_i].load(Ordering::Acquire) == 0;
                            // Pre-encode a tiny, valid activation carrier BEFORE starting the
                            // mode-set clock. DLM begins video ~110 ms after a wake mode-set; the
                            // real framebuffer can take 0.7--1.7 s to hash+encode at 1440p.
                            let w_pad = (timing.hactive as usize
                                + super::video::wht::STRIP_W
                                - 1)
                                & !(super::video::wht::STRIP_W - 1);
                            let h_pad = (timing.vactive as usize
                                + super::video::wht::STRIP_H
                                - 1)
                                & !(super::video::wht::STRIP_H - 1);
                            let prompt =
                                super::video::wht::black_frame_ep08(w_pad, h_pad, head).ok();
                            if prompt.is_none() {
                                pr_info!(
                                    "vino: prompt head={} black-frame preparation failed; continuing with real keyframe\n",
                                    head
                                );
                            }
                            // Prevent the independent CP worker from inserting a poll into DLM's
                            // measured marker/video sequence. `begin_cp_timeline` also drains one
                            // already-started keepalive iteration before the timestamp is taken.
                            data.begin_cp_timeline();
                            // DLM's first/wake mode-set has no pre-half. Its already-live
                            // mode-change path does bracket on both sides.
                            if !wake {
                                data.modeset_bracket_pre(dev, head);
                            }
                            let mode_anchor = Instant::<Monotonic>::now();
                            let r = data.send_cp(dev, 0x48, 0, |ctr| {
                                super::cp::set_mode(ctr, head, &timing)
                            });
                            if r.is_ok()
                                && data.modeset_requested[head_i].load(Ordering::Acquire) == key
                            {
                                data.modeset_active[head_i].store(key, Ordering::Release);
                                // Mark this mode generation as needing cold-link training. The
                                // deadline is refreshed when its initial ARM+keyframe is accepted,
                                // so slow encoding or a second head's mode-set cannot consume the
                                // useful part of the window before video actually starts.
                                data.sustain_until.lock()[head_i] =
                                    Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
                                let bit = 1u32 << head;
                                data.arm_prefix_pending.fetch_or(bit, Ordering::Release);
                                data.keyframe_pending.fetch_or(bit, Ordering::Release);
                                data.strip_hashes.lock()[head_i] = None;
                                data.modeset_bracket_post_open(dev, head, mode_anchor);
                                let prompt_started = prompt.as_ref().is_some_and(|frames| {
                                    match data.submit_prompt_training(
                                        dev,
                                        head,
                                        key,
                                        frames,
                                        PROMPT_TRAINING_OPEN_MS,
                                        true,
                                    ) {
                                        Ok(_) => true,
                                        Err(e) => {
                                            pr_info!(
                                                "vino: prompt head={} opening phase failed ({e:?})\n",
                                                head
                                            );
                                            false
                                        }
                                    }
                                });
                                data.modeset_bracket_post_close(dev, head, mode_anchor);
                                // DLM resumes its ordinary polling immediately after the close;
                                // the tail carrier continues concurrently with that dialogue.
                                data.end_cp_timeline();
                                if prompt_started {
                                    if let Some(frames) = prompt.as_ref() {
                                        if let Err(e) = data.submit_prompt_training(
                                            dev,
                                            head,
                                            key,
                                            frames,
                                            PROMPT_TRAINING_TAIL_MS,
                                            false,
                                        ) {
                                            pr_info!(
                                                "vino: prompt head={} tail phase failed ({e:?})\n",
                                                head
                                            );
                                        }
                                    }
                                }
                                pr_info!(
                                    "vino: sent DLM {} stream-enable bracket around head {} mode-set\n",
                                    if wake { "wake" } else { "mode-change" },
                                    head
                                );
                            } else {
                                data.end_cp_timeline();
                            }
                            r
                        }
                    }
                    KmsCmd::CursorCreate { head, w, h } => data.send_cp(dev, 0x1b, 0, |ctr| {
                        super::cp::cursor_create(ctr, head, w, h)
                    }),
                    KmsCmd::CursorImage { head, w, h, bgra } => {
                        data.send_cp(dev, 0x401c, 0, |ctr| {
                            super::cp::cursor_image(ctr, head, w, h, &bgra)
                        })
                    }
                    KmsCmd::CursorMove { head, x, y } => {
                        data.send_cp(dev, 0x1a, 0, |ctr| super::cp::cursor_move(ctr, head, x, y))
                    }
                };
                if let Err(e) = res {
                    pr_info!("vino: async KMS command failed ({e:?})\n");
                }
            }

            // A second head's mode-set may have arrived while the first head's blocking CP batch
            // was in flight. Give stream-control commands priority over starting a multi-megabyte
            // encode; cursor-only traffic may wait behind one frame and is coalesced at enqueue.
            let urgent = data
                .cmd_queue
                .lock()
                .iter()
                .any(|cmd| matches!(cmd, KmsCmd::ModeSet { .. }));
            if urgent {
                continue;
            }

            // Consume at most one framebuffer before checking the CP queue again. Round-robin
            // selection prevents an animated head 0 from starving a static head 1 keyframe.
            //
            // Do not TAKE and discard a frame which arrived before the per-head cadence deadline.
            // KWin may stop committing immediately after that flip (for example, when a window
            // stops moving), so no later callback would wake us and the newest image would remain
            // missing forever. Leave it in the coalescing slot, wait for the short remainder, and
            // then pick the newest version of it. Concurrent flips continue replacing/merging that
            // slot while this worker sleeps.
            let (frame, cadence_wait_ms) = {
                let mut pending = data.pending_scanout.lock();
                let mut selected = None;
                let mut wait_ms: Option<i64> = None;
                for offset in 0..HEADS {
                    let head = (next_head + offset) % HEADS;
                    if data.modeset_requested[head].load(Ordering::Acquire) != 0
                        && pending[head].is_some()
                    {
                        let owes_keyframe =
                            data.keyframe_pending.load(Ordering::Acquire) & (1u32 << head) != 0;
                        let elapsed_ms = data.last_frame.lock()[head]
                            .map_or(FRAME_PERIOD_MS, |t| t.elapsed().as_millis());
                        if owes_keyframe || elapsed_ms >= FRAME_PERIOD_MS {
                            selected = pending[head].take();
                            // A busy compositor continuously replaces `settle_repaint` via
                            // `queue_scanout`, so relying on the idle timer alone does NOT produce
                            // the sustained full-frame stream a cold downstream needs. Force the
                            // cadence-selected live framebuffer to be a keyframe while training.
                            // The elapsed check above still caps this at FRAME_PERIOD_MS; setting
                            // the bit any earlier would bypass the cadence limiter.
                            let sustaining = data.sustain_until.lock()[head].is_some_and(|until| {
                                (until - Instant::<Monotonic>::now()).as_millis() > 0
                            });
                            if sustaining {
                                data.keyframe_pending
                                    .fetch_or(1u32 << head, Ordering::Release);
                            }
                            next_head = (head + 1) % HEADS;
                            break;
                        }
                        let remaining = (FRAME_PERIOD_MS - elapsed_ms).max(1);
                        wait_ms = Some(wait_ms.map_or(remaining, |old| old.min(remaining)));
                    }
                }
                // Nothing flipped in. Fall back to the one-shot settle repaint if one is due, so a
                // compositor that went idle straight after enabling the output still ends up with
                // its real desktop on the panel rather than the buffer that happened to be current
                // when the mode-set's keyframe went out.
                if selected.is_none() {
                    let mut settle = data.settle_repaint.lock();
                    for offset in 0..HEADS {
                        let head = (next_head + offset) % HEADS;
                        if data.modeset_requested[head].load(Ordering::Acquire) == 0 {
                            settle[head] = None;
                            continue;
                        }
                        let Some((due, _)) = settle[head].as_ref() else {
                            continue;
                        };
                        let remaining = *due - Instant::<Monotonic>::now();
                        if remaining.as_millis() <= 0 {
                            selected = settle[head].take().map(|(_, f)| f);
                            data.keyframe_pending
                                .fetch_or(1u32 << head, Ordering::Release);
                            next_head = (head + 1) % HEADS;
                            if !TEST_FORCE_FULL_FRAMES {
                                pr_info!("vino: head {head} settle repaint (compositor idle after mode-set)\n");
                            }
                            break;
                        }
                        let remaining = remaining.as_millis().max(1);
                        wait_ms = Some(wait_ms.map_or(remaining, |old| old.min(remaining)));
                    }
                }
                (selected, wait_ms)
            };
            if let Some(frame) = frame {
                run_pending_scanout(dev, data, frame);
                continue;
            }
            if let Some(ms) = cadence_wait_ms {
                fsleep(Delta::from_millis(ms));
                continue;
            }

            if data.cmd_queue.lock().is_empty() {
                break;
            }
        }
    }
}

/// GEM object inner data. Empty: the shmem-backed `drm::gem::shmem::Object` (which
/// wires `drm_gem_shmem_dumb_create`, so userspace `DRM_IOCTL_MODE_CREATE_DUMB`
/// works) is enough until the EP08 scanout path consumes the framebuffers.
#[pin_data]
pub(super) struct VinoObject {}

impl drm::gem::DriverObject for VinoObject {
    type Driver = VinoDrmDriver;
    type Args = ();

    fn new(
        _dev: &drm::Device<VinoDrmDriver>,
        _size: usize,
        _args: (),
    ) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoObject {})
    }
}

/// Per-open DRM client state. The generic DRM fops pin the owning module for the file lifetime.
#[pin_data]
pub(super) struct VinoDrmFile {}

impl drm::file::DriverFile for VinoDrmFile {
    type Driver = VinoDrmDriver;

    fn open(_dev: &drm::Device<Self::Driver>) -> Result<Pin<KBox<Self>>> {
        KBox::try_pin_init(try_pin_init!(Self {}), GFP_KERNEL)
    }
}

const INFO: drm::DriverInfo = drm::DriverInfo {
    major: 0,
    minor: 1,
    patchlevel: 0,
    name: c"vino",
    desc: c"DisplayLink DL3 (Dell D6000) DRM driver",
};

#[vtable]
impl drm::Driver for VinoDrmDriver {
    type Data = VinoDrmData;
    type File = VinoDrmFile;
    type Object = drm::gem::shmem::Object<VinoObject>;
    type ParentDevice<Ctx: kernel::device::DeviceContext> = super::usb::Interface<Ctx>;
    type RegistrationData<'a> = ();
    type Kms = Self;

    const INFO: drm::DriverInfo = INFO;

    // The dock composites the cursor plane itself (sent over CP, not blended into the primary
    // scanout), so advertise the cursor hotspot properties and forward the client-set offset -- else
    // the pointer's click point is off from where it's drawn. See `VinoPlane::atomic_update`.
    const FEAT_CURSOR_HOTSPOT: bool = true;

    // No driver-private ioctls (GEM/dumb + KMS handled by the DRM core).
    kernel::declare_drm_ioctls! {}
}

#[vtable]
impl KmsDriver for VinoDrmDriver {
    type Connector = VinoConnector;
    type Plane = VinoPlane;
    type Crtc = VinoCrtc;
    type Encoder = VinoEncoder;

    fn mode_config_info(
        _dev: &kernel::device::Device,
        _drm_data: &Self::Data,
    ) -> Result<ModeConfigInfo> {
        Ok(ModeConfigInfo {
            min_resolution: (0, 0),
            max_resolution: (4096, 4096),
            max_cursor: (64, 64),
            preferred_depth: 32,
            preferred_fourcc: Some(DRM_FORMAT_XRGB8888),
        })
    }

    fn create_objects(dev: &UnregisteredKmsDevice<'_, Self>) -> Result {
        // Build one independent head (CRTC + primary/cursor plane + encoder + connector) per
        // wired display, each pinned to its own video endpoint via its head index.
        for head in 0..HEADS {
            // `possible_crtcs` for the plane/encoder is a bitmask of CRTC *indices*, which only
            // exist once `UnregisteredCrtc::new` runs -- but planes must exist before the CRTC
            // that references them. CRTCs are created here one per head in order, so this head's
            // CRTC index is `head` and its mask is `1 << head`.
            let crtc_mask = 1u32 << head;
            let primary = plane::UnregisteredPlane::<VinoPlane>::new(
                dev,
                crtc_mask,
                &PRIMARY_FORMATS,
                None,
                plane::Type::Primary,
                None,
                PlaneArgs {
                    head: head as u8,
                    is_cursor: false,
                },
            )?;
            // Tell compositors that this primary plane accepts the standard FB_DAMAGE_CLIPS
            // property. The scanout path already consumes those clips and emits only intersecting
            // 64x16 WHT strips, but without attaching the property KWin cannot provide them:
            // unchanged commits arrive with an empty clip list while real updates fall back to
            // ambiguous framebuffer swaps. That left the first keyframe frozen when empty damage
            // was correctly treated as a no-op, or forced multi-megabyte full frames when it was
            // treated as a repaint. EVDI exposes the same property before plane registration.
            primary.enable_fb_damage_clips();
            // Advertise every rotation vino's re-encode can produce by remapping source pixels
            // (`rot_src`): the four 90-degree rotations plus the two reflections.
            primary.create_rotation_property(
                bindings::DRM_MODE_ROTATE_0,
                bindings::DRM_MODE_ROTATE_0
                    | bindings::DRM_MODE_ROTATE_90
                    | bindings::DRM_MODE_ROTATE_180
                    | bindings::DRM_MODE_ROTATE_270
                    | bindings::DRM_MODE_REFLECT_X
                    | bindings::DRM_MODE_REFLECT_Y,
            )?;
            let cursor = plane::UnregisteredPlane::<VinoPlane>::new(
                dev,
                crtc_mask,
                &CURSOR_FORMATS,
                None,
                plane::Type::Cursor,
                None,
                PlaneArgs {
                    head: head as u8,
                    is_cursor: true,
                },
            )?;
            // The cursor plane advertises ARGB8888, and any plane exposing an alpha format must
            // also say how that alpha is blended -- `drm_mode_config_validate()` WARNs at
            // registration otherwise, tainting the kernel `W` and making every later crash report
            // harder to read. The dock composites the cursor itself from a premultiplied bitmap
            // (`cp::cursor_image`), so premultiplied is the only mode offered.
            cursor.create_blend_mode_property(1 << bindings::DRM_MODE_BLEND_PREMULTI)?;
            let crtc_obj = crtc::UnregisteredCrtc::<VinoCrtc>::new(
                dev,
                primary,
                Some(&cursor),
                None,
                head as u8,
            )?;
            // Advertise a 256-entry GAMMA_LUT; the scanout applies it (cached via the CRTC hooks).
            crtc_obj.enable_gamma(256);
            let enc = encoder::UnregisteredEncoder::<VinoEncoder>::new(
                dev,
                encoder::Type::Virtual,
                crtc_obj.mask(),
                0,
                None,
                (),
            )?;
            let conn = connector::UnregisteredConnector::<VinoConnector>::new(
                dev,
                // A DisplayLink dock output is a real DP/HDMI display, not a virtual one. Crucially,
                // `__drm_connector_init` SKIPS attaching the standard EDID property for
                // `DRM_MODE_CONNECTOR_VIRTUAL` (and WRITEBACK) -- so with `Virtual`, the dock's real
                // EDID could never be installed: `drm_edid_connector_update` had no `edid_property` to
                // write, leaving `edid_blob_ptr` NULL and `drm_edid_connector_add_modes` returning 0
                // (2026-07-20: EDID read fine but connector kept advertising only the fallback mode
                // list). DisplayPort gets the EDID property, so `get_modes` installs the monitor's real
                // native modes.
                connector::Type::DisplayPort,
                head as u8,
            )?;
            conn.attach_encoder(&*enc)?;
        }
        Ok(())
    }
}

// ---- CRTC -----------------------------------------------------------------

/// A software vblank source: an hrtimer that fires once per frame and drives
/// `drm_crtc_handle_vblank()`, so the atomic helpers pace page-flips against a real vblank
/// (via `drm_crtc_arm_vblank_event()` in [`VinoCrtc::atomic_flush`]) instead of completing them
/// immediately with a fake vblank.
///
/// The timer stops itself when vblank is disabled (mirroring the C core's
/// `drm_vblank_timer_function`, which returns `HRTIMER_NORESTART` on a zeroed interval): the
/// callback sees `enabled == false` and returns [`HrTimerRestart::NoRestart`], so an idle or
/// DPMS-off output costs no wakeups. `enable_vblank` re-arms it with a raw
/// [`ArcHrTimerHandle::restart`], which re-queues the timer whether it is dead or still pending -- no new
/// handle is minted, so the single `ArcHrTimerHandle` taken at first start remains the sole owner
/// and its drop (at CRTC teardown, before the `drm_crtc` is freed) is the only full
/// `hrtimer_cancel`. The *arming* half of this is the design `revdi`'s `EvdiCrtc`/`VblankTimer`
/// uses; the two drivers deliberately diverge on how the callback reaches its CRTC (see the
/// [`VblankTimer::crtc`] field) and on where the cancel is guaranteed (see the 2026-07-22
/// correction below).
///
/// **2026-07-16 history:** this used to free-run unconditionally (`Restart` regardless of
/// `enabled`) with an explicit `__module_get`/`module_put` pinning the module for as long as the
/// timer could fire, because `disconnect()` (`Device::unplug()`) does call
/// `drm_atomic_helper_shutdown()` -> `disable_vblank`, but that free-running timer never actually
/// stopped ticking, so nothing else guaranteed the module couldn't unload while it was still
/// armed (HW-confirmed panic: `rmmod` succeeded post-disconnect while the timer kept firing and
/// eventually jumped into freed module memory). That pin's release was tied to `VinoCrtc`'s
/// `PinnedDrop`, which turned out to never run at mere driver-unbind (the CRTC mode-object
/// outlives it -- see `project_release_bug_partial_fix_one_ref_remains_20260716` memory), so the
/// module could then never unload either -- trading a crash for a permanent "module in use" leak.
/// Comparing against `revdi`, which has no such pin and no leak, found the actual bug: `run()`
/// never implemented the self-stop-on-disable behavior its own (stale, copied) doc comment
/// claimed. Fixed to genuinely self-stop below.
///
/// **2026-07-22 correction: self-stopping is necessary but NOT sufficient.** The paragraph above
/// used to end by claiming the timer is "provably non-armed within one frame interval of
/// `disable_vblank` (called synchronously inside `unplug()`, always well before any subsequent
/// `rmmod`)". That premise is false, and it cost a machine reboot: on the observed unload
/// `atomic_disable` never ran, so the vblank refcount never dropped to zero, so `disable_vblank`
/// was never called and `enabled` stayed `true` -- the timer re-armed itself right through
/// `modprobe -r` and fired into freed module text. Self-stop only helps once someone actually
/// disables vblank. Teardown therefore no longer depends on it: the handle now lives in
/// [`VinoDrmData::vblank`] and `VinoDrmData::shutdown()` cancels it unconditionally.
#[pin_data]
pub(super) struct VblankTimer {
    #[pin]
    timer: HrTimer<Self>,
    /// The CRTC to deliver vblanks to, published the first time vblank is enabled. An owned
    /// handle rather than a raw pointer: it keeps the DRM device alive, so the timer callback can
    /// safely reach the CRTC from outside any DRM callback.
    ///
    /// **This is a reference cycle, and it must be broken explicitly.** A
    /// [`crtc::CrtcRef`] holds an `ARef<VinoDrmDevice>`, while this timer is itself owned by the
    /// device (via [`VinoCrtc::vblank`], which lives inside the DRM device allocation). Once
    /// `enable_vblank` publishes the handle the device therefore holds a reference to itself, so
    /// `drm_dev_put()` never reaches zero, `drm_dev_release()` never runs, and the card's minor is
    /// never returned to `drm_minors_xa`. That is exactly the long-standing "DRM minor advances
    /// 2, 3, 4, ... on every replug" anomaly: `drm_dev_unplug()` did run (the `card`/`renderD`
    /// nodes do disappear), but the `drm_device` behind it was immortal.
    /// [`VinoDrmData::shutdown`] clears this slot after `hrtimer_cancel`, which is what actually
    /// frees the device.
    ///
    /// A [`SpinLockIrq`], not a [`SetOnce`], purely so it can be cleared: the publisher
    /// (`enable_vblank`) runs with local interrupts already disabled, the reader is an hrtimer
    /// callback in hardirq context, and the one clearer is process context with interrupts on.
    ///
    /// `revdi` solves the same problem the other way -- `EvdiCrtc`'s timer stores a bare
    /// `AtomicPtr<bindings::drm_crtc>` and calls `drm_crtc_handle_vblank()` on it unsafely, so
    /// there is no reference to leak and never was a cycle. Vino keeps the owned handle (no unsafe
    /// deref in the callback) and pays for it with this lock plus the explicit clear in
    /// `shutdown()`. See `docs/MODULE-LIFECYCLE.md` "Root cause 4" for the comparison.
    #[pin]
    crtc: SpinLockIrq<Option<crtc::CrtcRef<VinoCrtc>>>,
    /// One scanout frame in nanoseconds (from the mode's `framedur_ns`).
    interval_ns: AtomicI64,
    /// Whether vblanks should currently be delivered (toggled by enable/disable_vblank).
    enabled: AtomicBool,
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
        let crtc = {
            // SAFETY: an hrtimer callback registered with `RelativeMode` (`HRTIMER_MODE_REL`, not
            // `..._SOFT`) is run from `hrtimer_interrupt()` in hardirq context, so local
            // interrupts are disabled for the whole of the borrow taken here.
            let irq = unsafe { LocalInterruptDisabled::assume_disabled() };
            this.crtc.lock_with(irq).clone()
        };
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
        mode: RelativeMode<Monotonic>, field: self.timer
    }
}

#[pin_data]
pub(super) struct VinoCrtc {
    /// Which display head (0-based) this CRTC drives. Used for diagnostics; the mode-set/DDC CP
    /// messages this CRTC sends are not yet head-differentiated on the wire (see the module doc).
    head: u8,
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
    vblank_pinned: Mutex<Option<OwnedVblankRef<VinoCrtc>>>,
}

#[derive(Clone, Default)]
pub(super) struct VinoCrtcState;

impl crtc::DriverCrtcState for VinoCrtcState {
    type Crtc = VinoCrtc;
}

#[vtable]
impl crtc::DriverCrtc for VinoCrtc {
    type Args = u8;
    type Driver = VinoDrmDriver;
    type State = VinoCrtcState;
    type VblankImpl = Self;

    fn new(_device: &drm::Device<Self::Driver>, head: &u8) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoCrtc {
            head: *head,
            vblank: Arc::pin_init(VblankTimer::new(), GFP_KERNEL)?,
            vblank_pinned <- new_mutex!(None),
        })
    }

    /// The display is turning on (scanout begins). Enables vblank pacing, pushes a live mode-set CP
    /// message for the negotiated mode. The command is queued and is a no-op until CP engages.
    const HAS_ATOMIC_CHECK: bool = true;

    /// Cross-head bandwidth admission: reject a commit whose pixel rate exceeds THIS head's even
    /// share of the dock's shared budget. Because each head only checks its own share, the sum
    /// across heads stays within the total without a global cross-CRTC check (revdi's proven model;
    /// `docs/CROSS-HEAD-BANDWIDTH-DESIGN.md`). A commit that does not increase this head's rate is
    /// always allowed (it can only hold or reduce the combined total); no limiting when budget is 0.
    fn atomic_check(check: CrtcAtomicCheck<'_, Self>) -> Result {
        let data: &VinoDrmData = check.crtc().drm_dev();
        let own_budget = data.own_pixel_budget();
        let (old, new) = check.take_old_new_state();
        let old_rate = if old.active() {
            let m = old.mode();
            u32::from(m.hdisplay())
                .saturating_mul(u32::from(m.vdisplay()))
                .saturating_mul(m.vrefresh().max(0) as u32)
        } else {
            0
        };
        let new_rate = if new.active() {
            let m = new.mode();
            u32::from(m.hdisplay())
                .saturating_mul(u32::from(m.vdisplay()))
                .saturating_mul(m.vrefresh().max(0) as u32)
        } else {
            0
        };
        if own_budget == 0 || new_rate <= old_rate {
            return Ok(());
        }
        if new_rate > own_budget {
            pr_warn!(
                "vino: rejecting commit -- head pixel rate {new_rate} exceeds this head's budget share ({own_budget})\n"
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
                    "vino: failed to start head {} software vblank clock ({e:?})\n",
                    crtc.head
                ),
            }
        }
        drop(pinned);
        let head = crtc.head;
        let dev: &VinoDrmDevice = crtc.drm_dev();
        let data: &VinoDrmData = dev;
        let new = commit.take_new_state();
        // Cache this head's gamma ramp for the scanout to apply.
        data.update_gamma(head as usize, new.gamma_lut());
        let timing = super::cp::timing_from_drm_mode(new.mode());
        pr_info!(
            "vino: KMS CRTC enable -- head {} display ON, mode {}x{}@{} (scanout begins)\n",
            head,
            timing.hactive,
            timing.vactive,
            timing.refresh_hz
        );
        // 2026-07-16: these used to be direct blocking `send_cp`/`set_vcp` calls here -- real USB
        // I/O inline in the DRM atomic-commit callback. See `KmsCmd`'s doc comment for why that's
        // risky (a live freeze landed right after a rapid mode-switch with no oops at all) and
        // why queuing for the async worker instead makes this callback fast and non-blocking by
        // construction. The actual sends (and their success/failure logging) now happen in
        // `VinoDrmData::run`.
        let mode_key = timing_key(&timing);
        data.last_timing.lock()[head as usize] = Some(timing);
        data.modeset_requested[head as usize].store(mode_key, Ordering::Release);
        data.queue_cmd(dev, KmsCmd::ModeSet { head, timing });
    }

    /// The display is turning off (DPMS-off/blank/suspend all land here in atomic KMS).
    /// Resets the scanout state so a later re-enable sends a full keyframe rather than diffing
    /// against a shadow the dock may have dropped. Do not send the monitor's DDC/CI VCP 0xd6
    /// here: no such write exists in the DLM capture, and hard-standby is separate from stopping
    /// the DisplayLink stream (it can also leave a panel asleep across a dock cold plug).
    fn atomic_disable(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        // Dropping the stored reference releases the vblank reference `atomic_enable` took.
        drop(crtc.vblank_pinned.lock().take());
        crtc.vblank_off();
        let head = crtc.head;
        let dev: &VinoDrmDevice = crtc.drm_dev();
        let data: &VinoDrmData = dev;
        data.update_gamma(head as usize, None);
        // The stream is torn down; a later re-enable must re-send the mode-set before any video
        // write (the dock EPIPEs a write onto an unconfigured stream). Forget the active mode so
        // the scanout gate defers until the re-enable's mode-set lands.
        data.modeset_requested[head as usize].store(0, Ordering::Release);
        data.modeset_active[head as usize].store(0, core::sync::atomic::Ordering::Release);
        // Drop a framebuffer queued while this CRTC was active. Otherwise the deferred worker can
        // retry its old mode and paint after DPMS-off.
        data.pending_scanout.lock()[head as usize] = None;
        data.settle_repaint.lock()[head as usize] = None;
        data.sustain_until.lock()[head as usize] = None;
        data.strip_hashes.lock()[head as usize] = None;
        pr_info!("vino: KMS CRTC disable -- head {head} display OFF (scanout stopped)\n");
    }

    /// Arm the page-flip completion event to be sent by the next vblank tick, so userspace is paced
    /// to the refresh rate rather than signalled immediately.
    fn atomic_flush(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        let data: &VinoDrmData = crtc.drm_dev();
        let mut new = commit.take_new_state();
        // Re-cache the gamma ramp on every commit that touches this CRTC, so a dynamic GAMMA_LUT
        // change on an already-enabled head (which does not re-run atomic_enable) is picked up
        // rather than deferred to the next full modeset.
        data.update_gamma(crtc.head as usize, new.gamma_lut());
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
        let head = usize::from(data.head);
        if head >= HEADS {
            return Ok(());
        }
        let mut slots = drm_data.vblank.lock();
        match &slots[head] {
            None => {
                // First enable: start the timer and keep the handle as its sole owner.
                slots[head] = Some((
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

/// Constructor arguments for a [`VinoPlane`]: which head it belongs to and whether it is that
/// head's cursor plane (vs. its primary scanout plane).
#[derive(Clone, Copy)]
pub(super) struct PlaneArgs {
    head: u8,
    is_cursor: bool,
}

#[pin_data]
pub(super) struct VinoPlane {
    /// Which display head (0-based) this plane belongs to. Selects the scanout video endpoint
    /// ([`VIDEO_EPS`]) and the cursor CP `head` field.
    head: u8,
    /// Whether this is the cursor plane (vs. the primary scanout plane).
    is_cursor: bool,
    /// The framebuffer last uploaded to the dock as the cursor bitmap (raw address, `0` = none),
    /// so a bare cursor move only re-sends the position, not the whole image. Cursor plane only.
    cursor_last: core::sync::atomic::AtomicUsize,
}

#[derive(Clone, Default)]
pub(super) struct VinoPlaneState;

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
            head: args.head,
            is_cursor: args.is_cursor,
            cursor_last: core::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Validate the plane geometry and, importantly, let the DRM helper compute
    /// `drm_plane_state.visible`. The damage iterator deliberately returns no rectangles for a
    /// plane whose `visible` bit was never established. Without this hook the initial keyframe
    /// worked (it ignores damage), but every later KWin flip appeared to have empty damage and was
    /// discarded, leaving a permanently static image.
    fn atomic_check(check: PlaneAtomicCheck<'_, Self>) -> Result {
        let (state, _old, mut new) = check.take_all();
        let Some(crtc) = new.crtc::<VinoDrmDriver>() else {
            // A disabled plane is not visible and needs no geometry validation.
            return Ok(());
        };
        let crtc_state = match state.get_new_crtc_state(crtc) {
            Some(s) => s,
            None => state.add_crtc_state(crtc)?,
        };
        // VINO currently supports 1:1 scanout only. Like udl/gud, keep positioning and updates on
        // a disabled CRTC disallowed; this also clips the source and publishes `visible` for the
        // frame-damage helper used by atomic_update.
        new.atomic_helper_check::<_, VinoDrmDriver>(&crtc_state, false, false)
    }

    /// A new framebuffer was flipped in. Maps it, converts XRGB8888 -> RGB565 (or feeds the
    /// WHT colour codec directly for an aligned mode), and bulk-writes the resulting EP08
    /// frame(s).
    ///
    /// EP08 writes happen only after CP engagement and a matching mode-set has landed.
    fn atomic_update(commit: PlaneAtomicCommit<'_, Self>) {
        let plane = commit.plane();
        let head = plane.head;
        let dev: &VinoDrmDevice = plane.drm_dev();
        let data: &VinoDrmData = dev;
        if !data.cp_engaged.load(core::sync::atomic::Ordering::SeqCst) {
            return;
        }

        // Cursor plane: forward the cursor bitmap/position to the dock over CP (id=0x1b create,
        // 0x401c image, 0x1a move -- see `cp::cursor_*`). A no-op until CP engages, like scanout.
        // 2026-07-16: these used to be direct blocking `send_cp` calls -- see `KmsCmd`'s doc
        // comment. Queuing means `cursor_last` is now updated optimistically at queue time rather
        // than only after a confirmed-successful send (the previous retry-on-failure behaviour):
        // a minor regression (a deferred send that fails silently won't be retried until the
        // bitmap changes again), acceptable for a cosmetic path that was never implicated in any
        // incident, in exchange for not blocking this callback on cursor-image I/O either.
        if plane.is_cursor {
            use core::sync::atomic::Ordering::Relaxed;
            let new = commit.take_new_state();
            match new.framebuffer::<VinoDrmDriver>() {
                Some(fb) => {
                    let w = fb.width() as u16;
                    let h = fb.height() as u16;
                    let key = fb as *const _ as usize;
                    if plane.cursor_last.load(Relaxed) != key {
                        if let Ok(bgra) = read_cursor_bgra(fb, w as usize, h as usize) {
                            data.queue_cmd(dev, KmsCmd::CursorCreate { head, w, h });
                            data.queue_cmd(dev, KmsCmd::CursorImage { head, w, h, bgra });
                            plane.cursor_last.store(key, Relaxed);
                        }
                    }
                    // Cursor hotspot (DRIVER_CURSOR_HOTSPOT): with the client cap set, `crtc_x/y` is
                    // where the pointer's hot pixel should land, and `hotspot_x/y` is that pixel's
                    // offset inside the bitmap. The dock composites the bitmap by its top-left, so
                    // shift the sent position by `-hotspot` to put the hot pixel at `crtc_x/y`.
                    // Without the cap `hotspot_x/y` are 0, so this is a no-op (legacy behaviour).
                    let x = (new.crtc_x() - new.hotspot_x()).max(0) as u16;
                    let y = (new.crtc_y() - new.hotspot_y()).max(0) as u16;
                    data.queue_cmd(dev, KmsCmd::CursorMove { head, x, y });
                }
                // Cursor disabled: forget the last bitmap so it re-uploads if it comes back.
                None => plane.cursor_last.store(0, Relaxed),
            }
            return;
        }

        // Primary plane: take both old and new state so the frame-damage clips can be merged.
        let (old, new) = commit.take_old_new_state();
        let Some(fb) = new.framebuffer::<VinoDrmDriver>() else {
            return;
        };
        // The plane's destination geometry mirrors the negotiated mode (the compositor sizes the
        // primary plane 1:1 with the virtual output), so this drives the dynamic scanout
        // resolution.
        // Plane rotation/reflection (identity unless the compositor set the rotation property).
        let rotation = new.rotation();
        // Clamp the output rectangle to the framebuffer so scanout never samples past the source.
        // crtc_w/crtc_h are the userspace plane destination; the safe KMS layer exposes no
        // atomic_check to reject scaling, so a destination larger than the attached framebuffer
        // would otherwise drive the pixel loop past the end of the mapped GEM buffer. Under 90/270
        // the source axes are swapped, so bound each output axis by the framebuffer axis it
        // samples. This also keeps `w * h` (the shadow-buffer size) within the framebuffer's
        // bounds, so it cannot overflow a 32-bit `usize`.
        let (fbw, fbh) = (fb.width() as usize, fb.height() as usize);
        let (w, h) = match rotation & bindings::DRM_MODE_ROTATE_MASK {
            r if r == bindings::DRM_MODE_ROTATE_90 || r == bindings::DRM_MODE_ROTATE_270 => (
                (new.crtc_w() as usize).min(fbh),
                (new.crtc_h() as usize).min(fbw),
            ),
            _ => (
                (new.crtc_w() as usize).min(fbw),
                (new.crtc_h() as usize).min(fbh),
            ),
        };
        // Collect the client's individual frame-damage clips (the rectangles that
        // `damage_merged()` would collapse into one bounding box), each clamped to the output, so
        // only the genuinely changed rectangles are re-converted from the source rather than their
        // whole enclosing box. Only for identity rotation (the clips are in un-rotated source
        // space; mapping them through 90/270 is not worth it for the throttled fallback path), and
        // never on the WHT keyframe path -- see `encode_and_send`. A fixed stack array keeps the
        // atomic-commit path allocation-free; on overflow the clips collapse into one bounding box.
        // An empty list means the client reported no changed pixels. Rotation/reflection still
        // promotes it to a full frame in `encode_and_send_wht`, because source-space clips cannot
        // yet be transformed safely for those cases.
        let mut clips = [(0usize, 0usize, 0usize, 0usize); MAX_DAMAGE_CLIPS];
        let mut nclips = 0usize;
        if rotation & bindings::DRM_MODE_ROTATE_MASK == bindings::DRM_MODE_ROTATE_0
            && rotation & (bindings::DRM_MODE_REFLECT_X | bindings::DRM_MODE_REFLECT_Y) == 0
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
        let skip = data.scanout_skip.load(Relaxed);
        if skip > 0 {
            data.scanout_skip.store(skip - 1, Relaxed);
            return;
        }
        data.queue_scanout(
            dev,
            PendingScanout {
                head,
                fb: ARef::from(fb),
                rotation,
                clips,
                nclips,
                w,
                h,
            },
        );
    }
}

/// Compress and submit one coalesced primary-plane flip on the deferred worker. Keeping all slow
/// work here makes the DRM atomic callback bounded to state inspection plus an `ARef` increment.
fn run_pending_scanout(dev: &BoundInterface<'_>, data: &VinoDrmData, frame: PendingScanout) {
    use core::sync::atomic::Ordering::Relaxed;

    let head_i = frame.head as usize;
    if data.modeset_requested[head_i].load(Ordering::Acquire) == 0 {
        scanout_gate(frame.head, 6, "worker: head has no mode-set requested");
        return;
    }
    let requested_geometry_matches = data.last_timing.lock()[head_i]
        .is_some_and(|t| t.hactive as usize == frame.w && t.vactive as usize == frame.h);
    if !requested_geometry_matches {
        // stale framebuffer from a different-size mode generation
        scanout_gate(frame.head, 7, "worker: framebuffer size differs from the cached mode");
        return;
    }
    // Was this the mode-set's owed keyframe? Read before sending, since a successful send clears
    // the bit.
    let was_keyframe = data.keyframe_pending.load(Ordering::Acquire) & (1u32 << frame.head) != 0;
    // `TEST_FORCE_FULL_FRAMES` also needs a continuous *source* of frames. Normally a repaint is
    // only re-armed after the mode-set's keyframe, so once the compositor goes idle -- which it
    // does immediately when the panels are dark and nothing animates -- vino stops transmitting
    // entirely (HW-observed: zero EP08 bytes after the first keyframe). DLM never goes silent; it
    // streams full frames continuously at ~7 fps. Re-arm unconditionally so this experiment
    // actually reproduces that.
    let settle_copy = (was_keyframe || TEST_FORCE_FULL_FRAMES).then(|| frame.clone());
    match scanout_one(
        dev,
        data,
        frame.head,
        &frame.fb,
        frame.rotation,
        &frame.clips[..frame.nclips],
        frame.w,
        frame.h,
    ) {
        Ok(()) => {
            let n = data.scanout_fails.swap(0, Relaxed);
            data.scanout_skip.store(0, Relaxed);
            if n > 0 {
                pr_info!("vino: scanout recovered after {n} failed frame(s)\n");
            }
            // Arm the one-shot settle repaint. A compositor that goes idle right after enabling an
            // output would otherwise leave whatever it had drawn at keyframe time -- in the
            // observed failure, an all-black buffer -- on the panel permanently.
            if let Some(mut copy) = settle_copy {
                copy.clips[0] = (0, 0, copy.w, copy.h);
                copy.nclips = 1;
                // ★ Cold-link training: within the post-mode-set `sustain_until` window, repaint at
                // the fast frame cadence so the dock sees the SUSTAINED continuous stream a cold
                // downstream needs to train its link and program the pixel clock (see `sustain_until`
                // and `captures/dlm-coldplug-withmon-*`). Outside the window (link already up), back
                // off to the sparse one-shot settle so a static desktop is not refreshed forever.
                let sustaining = data.sustain_until.lock()[head_i]
                    .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
                let delay = if TEST_FORCE_FULL_FRAMES || sustaining {
                    FRAME_PERIOD_MS
                } else {
                    SETTLE_REPAINT_MS
                };
                data.settle_repaint.lock()[head_i] =
                    Some((Instant::<Monotonic>::now() + Delta::from_millis(delay), copy));
            }
        }
        Err(e) => {
            // Log at exponentially sparser points and back off future worker attempts. An error is
            // transport state, not a reason to stall the compositor's pageflip path.
            let n = data.scanout_fails.fetch_add(1, Relaxed) + 1;
            if n == 1 || n.is_power_of_two() {
                pr_err!("vino: scanout frame failed ({e:?}) [x{n}] -- throttling\n");
            }
            data.scanout_skip.store(core::cmp::min(n, 120), Relaxed);
        }
    }
}

/// Map the cursor framebuffer and convert it to the `w*h*4` BGRA bitmap the dock expects
/// (`cp::cursor_image`). The source is XRGB/ARGB8888; each pixel is written out as B, G, R, A.
fn read_cursor_bgra(
    fb: &kms::framebuffer::Framebuffer<VinoDrmDriver>,
    w: usize,
    h: usize,
) -> Result<KVec<u8>> {
    let vmap = fb.vmap::<VinoObject>()?;
    let view = vmap.view();
    let src = view.as_ptr().cast::<u8>();
    let pitch = vmap.pitch();
    let mut out = KVec::with_capacity(w * h * 4, GFP_KERNEL)?;
    for dy in 0..h {
        for dx in 0..w {
            // SAFETY: `dy*pitch + dx*4 + 3` is within the mapped cursor framebuffer (`pitch*h`
            // bytes); `dx < w <= pitch/4`, `dy < h`.
            // ARGB8888 is little-endian in memory (B,G,R,A); normalise the word so the channel
            // shifts below are correct on big-endian hosts too.
            let px = u32::from_le(unsafe {
                (src.add(dy * pitch + dx * 4) as *const u32).read_unaligned()
            });
            out.push((px & 0xff) as u8, GFP_KERNEL)?; // B
            out.push(((px >> 8) & 0xff) as u8, GFP_KERNEL)?; // G
            out.push(((px >> 16) & 0xff) as u8, GFP_KERNEL)?; // R
            out.push(((px >> 24) & 0xff) as u8, GFP_KERNEL)?; // A
        }
    }
    Ok(out)
}

/// Source (framebuffer) dimensions for an output of `ow`x`oh` pixels under plane `rotation`.
/// The 90/270 rotations swap width and height between the framebuffer and the displayed output;
/// the others preserve them.
fn src_dims(rotation: u32, ow: usize, oh: usize) -> (usize, usize) {
    let rot = rotation & bindings::DRM_MODE_ROTATE_MASK;
    if rot == bindings::DRM_MODE_ROTATE_90 || rot == bindings::DRM_MODE_ROTATE_270 {
        (oh, ow)
    } else {
        (ow, oh)
    }
}

/// vmap `fb`, encode it, and push one video frame to the head's endpoint. Split out so `?` can be
/// used. `w`/`h` are the OUTPUT (displayed) dimensions; `clips` are the client's changed
/// rectangles in output space (empty = no changed pixels for identity rotation).
fn scanout_one(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    head: u8,
    fb: &kms::framebuffer::Framebuffer<VinoDrmDriver>,
    rotation: u32,
    clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    if w == 0 || h == 0 {
        return Err(EINVAL);
    }
    // Map the framebuffer's backing pages into the kernel address space; the guard unmaps on
    // drop, including on an early return below.
    let vmap = fb.vmap::<VinoObject>()?;
    // The real source stride: GEM dumb buffers pad the pitch (alignment), so it is not necessarily
    // `w * 4`. Take it from the validated mapping, which has already checked the offset, plane
    // count and object size for us.
    let pitch = vmap.pitch();
    let view = vmap.view();
    encode_and_send(
        dev,
        data,
        head,
        view.as_ptr().cast::<u8>(),
        pitch,
        rotation,
        clips,
        w,
        h,
    )
}

/// Encode the mapped frame with the byte-exact Vino WHT **colour** codec and bulk-write the
/// resulting EP08 frame(s). Reads the source XRGB8888 at full 8-bit precision (no RGB565
/// reduction -- the codec works in 8-bit RGB). The first frame after each mode-set is a full
/// keyframe; subsequent identity-rotation flips send only the changed strips (damage delta, via
/// `keyframe_pending` + `colour_frame_ep08_damage`; see `docs/VIDEO-PARTIAL-UPDATE-DESIGN.md`).
/// `w`/`h` are the mode's real size; non-64x16-aligned modes are padded up to the next strip
/// multiple (`w_pad`/`h_pad`) and encoded as full bands (DLM's shape -- the dock shows only `w`x`h`
/// and ignores the black-padded rows/cols).
///
/// `#[inline(never)]` (2026-07-17): this function's own stack frame was the direct site of a
/// real kernel stack overflow (`BUG: TASK stack guard page was hit`, caught by
/// `CONFIG_VMAP_STACK`/`CONFIG_SCHED_STACK_END_CHECK`) -- see the `#[inline(never)]`s in
/// `video.rs`'s codec functions for the root cause and
/// `project_stack_overflow_root_cause_found_20260717` memory for the full story. Keeping this a
/// real call (not flattened into `scanout_one`) means its frame is freed on return instead of
/// staying part of a caller that itself runs several frames deep inside a workqueue callback.
///
/// Hash every native 64x16 strip in an identity-rotation framebuffer. The hash deliberately uses
/// Geometry is stored alongside the result by [`StripHashState`], so two modes can never be
/// compared even if their tile counts happen to match. XRGB's unused byte is included: a compositor
/// changing it can cause a harmless extra update but can never hide a visible pixel change.
#[inline(never)]
fn framebuffer_strip_hashes(
    vaddr: *const u8,
    pitch: usize,
    w: usize,
    h: usize,
    w_pad: usize,
    h_pad: usize,
) -> Result<KVVec<u64>> {
    let tiles_x = w_pad / super::video::wht::STRIP_W;
    let tiles_y = h_pad / super::video::wht::STRIP_H;
    let mut hashes: KVVec<u64> = KVVec::new();
    hashes.resize(tiles_x * tiles_y, 0, GFP_KERNEL)?;

    for ty in 0..tiles_y {
        let sy = ty * super::video::wht::STRIP_H;
        for tx in 0..tiles_x {
            let sx = tx * super::video::wht::STRIP_W;
            // Include tile position in the seed so moving equal-colour tiles does not create a
            // large family of identical hashes. Padded rows/columns are represented by this seed;
            // their black content is immutable and need not be read.
            let mut hash = 0x9e37_79b1_85eb_ca87u64
                ^ (sx as u64).rotate_left(17)
                ^ (sy as u64).rotate_left(43);
            let y_end = (sy + super::video::wht::STRIP_H).min(h);
            let x_end = (sx + super::video::wht::STRIP_W).min(w);
            let mut y = sy;
            while y < y_end {
                // SAFETY: `sx < x_end <= w`, `y < h`, and the XRGB8888 pitch covers every visible
                // pixel, so this row slice lies inside the mapped framebuffer.
                let row = unsafe {
                    core::slice::from_raw_parts(vaddr.add(y * pitch + sx * 4), (x_end - sx) * 4)
                };
                // Chaining the previous hash in as the next seed fingerprints the whole tile
                // without copying its rows into one contiguous buffer.
                hash = xxhash::xxh64(row, hash);
                y += 1;
            }
            hashes[ty * tiles_x + tx] = hash;
        }
    }
    Ok(hashes)
}

/// Convert changed strip hashes into a compact set of already tile-aligned damage rectangles.
/// Horizontal runs are joined, then equal runs on adjacent bands are extended vertically. A very
/// fragmented frame falls back to one full-output rectangle rather than growing an unbounded
/// allocation or spending more time testing rectangles than encoding strips.
#[inline(never)]
fn changed_strip_rects(
    old: &[u64],
    new: &[u64],
    w_pad: usize,
    h_pad: usize,
) -> Result<KVec<DamageRect>> {
    const MAX_RECTS: usize = 128;
    let tiles_x = w_pad / super::video::wht::STRIP_W;
    let tiles_y = h_pad / super::video::wht::STRIP_H;
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
            let x0 = run_start * super::video::wht::STRIP_W;
            let x1 = tx * super::video::wht::STRIP_W;
            let y0 = ty * super::video::wht::STRIP_H;
            let y1 = y0 + super::video::wht::STRIP_H;
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

/// One bit per (gate, head): has this "returned Ok without writing anything" reason been reported?
static SCANOUT_GATE_SEEN: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Report, once per (gate, head), that a scanout attempt produced no wire traffic.
///
/// Every deferral gate in [`encode_and_send_wht`] returns `Ok(())` -- correctly, since a deferred
/// frame is not a transport error. The side effect is that a head which defers *every* frame
/// forever is indistinguishable from a healthy idle one: the panel stays dark, no error is logged,
/// and the dock's own trace shows its stream starving. Naming the gate turns that silent state
/// into a diagnosable one. One-shot per reason so a permanently-stuck head cannot spam the log.
fn scanout_gate(head: u8, idx: u32, reason: &str) {
    let bit = 1u32 << (idx * HEADS as u32 + head as u32);
    if SCANOUT_GATE_SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        pr_info!("vino: scanout head={head} sent NOTHING -- deferred at gate: {reason}\n");
    }
}

#[inline(never)]
fn encode_and_send_wht(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    head: u8,
    vaddr: *const u8,
    pitch: usize,
    rotation: u32,
    _clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    // Gate the EP08 write on the mode-set (id=0x48) for THIS mode having actually been sent to
    // the dock. On an enabling-modeset commit the plane update (`commit_planes`) runs BEFORE the
    // CRTC enable queues the mode-set, so without this the video write races ahead of the mode-set
    // and the dock EPIPEs it onto an unconfigured stream (HW-observed 2026-07-17). Defer this frame
    // (no write, no error/backoff, no seq advance) until the async worker has sent the mode-set;
    // the next flip then proceeds. DLM's own order is mode-set -> video.
    let head_i = head as usize;
    let want = data.modeset_requested[head_i].load(core::sync::atomic::Ordering::Acquire);
    if want == 0 {
        scanout_gate(head, 0, "no mode-set requested (modeset_requested == 0)");
        return Ok(());
    }
    let cached = data.last_timing.lock()[head_i];
    if !cached.is_some_and(|t| {
        timing_key(&t) == want && t.hactive as usize == w && t.vactive as usize == h
    }) {
        scanout_gate(head, 1, "cached timing does not match the requested mode generation");
        return Ok(());
    }
    if data.modeset_active[head_i].load(core::sync::atomic::Ordering::Acquire) != want {
        // RETRY the mode-set rather than deferring forever. The queued `KmsCmd::ModeSet` can fail
        // (CP not engaged yet while the dock re-enumerates), and nothing else ever re-sends it, so
        // the scanout would defer every frame for the life of the device. This path is sleepable
        // (it already does a blocking `bulk_send`), so send it inline and latch `modeset_active` on
        // success -- exactly what the async worker does. Rate-limited by only retrying when the
        // cached timing matches the mode the compositor is actually flipping.
        if let Some(t) = cached {
            if timing_key(&t) == want && t.hactive as usize == w && t.vactive as usize == h {
                let wake =
                    data.modeset_active[head_i].load(core::sync::atomic::Ordering::Acquire) == 0;
                let w_pad = (w + super::video::wht::STRIP_W - 1)
                    & !(super::video::wht::STRIP_W - 1);
                let h_pad = (h + super::video::wht::STRIP_H - 1)
                    & !(super::video::wht::STRIP_H - 1);
                let prompt = super::video::wht::black_frame_ep08(w_pad, h_pad, head).ok();
                data.begin_cp_timeline();
                if !wake {
                    data.modeset_bracket_pre(dev, head);
                }
                let mode_anchor = Instant::<Monotonic>::now();
                let mode_sent = data
                    .send_cp(dev, 0x48, 0, |ctr| super::cp::set_mode(ctr, head, &t))
                    .is_ok();
                if mode_sent
                    && data.modeset_requested[head_i].load(Ordering::Acquire) == want
                {
                    data.modeset_active[head_i].store(want, core::sync::atomic::Ordering::Release);
                    data.sustain_until.lock()[head_i] =
                        Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
                    data.arm_prefix_pending
                        .fetch_or(1u32 << head, core::sync::atomic::Ordering::Release);
                    data.keyframe_pending
                        .fetch_or(1u32 << head, core::sync::atomic::Ordering::Release);
                    data.strip_hashes.lock()[head_i] = None;
                    data.modeset_bracket_post_open(dev, head, mode_anchor);
                    let prompt_started = prompt.as_ref().is_some_and(|frames| {
                        match data.submit_prompt_training(
                            dev,
                            head,
                            want,
                            frames,
                            PROMPT_TRAINING_OPEN_MS,
                            true,
                        ) {
                            Ok(_) => true,
                            Err(e) => {
                                pr_info!(
                                    "vino: inline prompt head={} opening phase failed ({e:?})\n",
                                    head
                                );
                                false
                            }
                        }
                    });
                    data.modeset_bracket_post_close(dev, head, mode_anchor);
                    data.end_cp_timeline();
                    if prompt_started {
                        if let Some(frames) = prompt.as_ref() {
                            if let Err(e) = data.submit_prompt_training(
                                dev,
                                head,
                                want,
                                frames,
                                PROMPT_TRAINING_TAIL_MS,
                                false,
                            ) {
                                pr_info!(
                                    "vino: inline prompt head={} tail phase failed ({e:?})\n",
                                    head
                                );
                            }
                        }
                    }
                    pr_info!("vino: scanout re-sent bracketed mode-set {w}x{h}\n");
                } else {
                    data.end_cp_timeline();
                }
            }
        }
        // A successful inline retry has made this very commit safe to send: continue into the
        // encoder instead of waiting for another page flip.  A completely static head may not
        // receive another atomic_update after this enabling commit (HW-observed on head 1), so
        // returning unconditionally here leaves that monitor configured but permanently dark.
        // If the retry did not land, keep deferring exactly as before.
        if data.modeset_active[head_i].load(core::sync::atomic::Ordering::Acquire) != want {
            scanout_gate(head, 2, "mode-set not active and the inline re-send did not land");
            return Ok(());
        }
    }
    let seq0 = data.scanout_seq.lock()[head_i];
    let gamma = data.gamma_snapshot(head_i);
    // Source dimensions (swapped from the output for 90/270 rotation).
    let (sw, sh) = src_dims(rotation, w, h);
    // Full keyframe vs damage delta. A mode-set requires a keyframe; rotation/reflection remains
    // conservative because the content shadow is deliberately stored in unrotated framebuffer
    // space. For identity rotation, compare the actual framebuffer instead of trusting optional
    // FB_DAMAGE_CLIPS: KWin commonly changes framebuffer objects without publishing that blob.
    let kf_bit = 1u32 << head_i;
    let identity = rotation & bindings::DRM_MODE_ROTATE_MASK == bindings::DRM_MODE_ROTATE_0
        && rotation & (bindings::DRM_MODE_REFLECT_X | bindings::DRM_MODE_REFLECT_Y) == 0;
    let owes_keyframe = data
        .keyframe_pending
        .load(core::sync::atomic::Ordering::Acquire)
        & kf_bit
        != 0;
    // `TEST_FORCE_FULL_FRAMES`: never send a damage delta, so the dock always receives a complete
    // frame rather than a partial update onto content it may never have displayed.
    let mut full = owes_keyframe || !identity || TEST_FORCE_FULL_FRAMES;
    // Non-64x16-aligned modes (e.g. 1920x1080, 1080%16=8) are padded up to the next strip multiple
    // and encoded as full bands -- exactly DLM's shape (1080p pcap = 68 bands = 1088 rows for a 1080p
    // mode-set). The mode-set carries the real w/h, so the dock displays only that and ignores the
    // pad rows/cols; the sampler below returns black for the padded region (its content is never
    // shown). See `docs/VIDEO-TODO.md`.
    let w_pad = (w + super::video::wht::STRIP_W - 1) & !(super::video::wht::STRIP_W - 1);
    let h_pad = (h + super::video::wht::STRIP_H - 1) & !(super::video::wht::STRIP_H - 1);
    let mut content_hashes: Option<KVVec<u64>> = None;
    let mut content_damage: KVec<DamageRect> = KVec::new();
    if identity {
        let hashes = framebuffer_strip_hashes(vaddr, pitch, w, h, w_pad, h_pad)?;
        if !full {
            let previous = data.strip_hashes.lock();
            if let Some(state) = &previous[head_i] {
                if state.w_pad == w_pad && state.h_pad == h_pad {
                    content_damage = changed_strip_rects(&state.hashes, &hashes, w_pad, h_pad)?;
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
        scanout_gate(head, 3, "no keyframe owed and no strip content changed");
        return Ok(());
    }
    // Shared pixel sampler: output (dx,dy) -> gamma-corrected source RGB. Padding beyond the real
    // frame is black (dock-ignored) and, crucially, never dereferences the framebuffer out of bounds.
    let px = |dx: usize, dy: usize| {
        if dx >= w || dy >= h {
            return (0, 0, 0);
        }
        // Map the output pixel back to its source pixel under the plane rotation/reflection.
        let (sx, sy) = rot_src(rotation, dx, dy, sw, sh);
        // SAFETY: `dx<w, dy<h` (checked above), so `rot_src` returns `sx < sw <= pitch/4`, `sy < sh`
        // and `sy*pitch + sx*4 + 3` is within the mapped source framebuffer (`pitch*sh` bytes).
        // XRGB8888 is little-endian in memory; normalise so the channel shifts are endian-safe.
        let p = u32::from_le(unsafe {
            (vaddr.add(sy * pitch + sx * 4) as *const u32).read_unaligned()
        });
        apply_gamma(
            &gamma,
            ((p >> 16) & 0xff) as u8,
            ((p >> 8) & 0xff) as u8,
            (p & 0xff) as u8,
        )
    };
    let (frames, next_seq) = if full {
        super::video::wht::colour_frame_ep08(w_pad, h_pad, seq0, head, px)?
    } else {
        super::video::wht::colour_frame_ep08_damage(w_pad, h_pad, seq0, head, &content_damage, px)?
    };
    // A damage delta that touched no aligned strip = nothing to send this flip: skip the write
    // (no seq advance, no arm, keyframe obligation untouched). Full frames always have strips.
    if frames.is_empty() {
        scanout_gate(head, 4, "encoder produced zero records");
        return Ok(());
    }
    if data.shutting_down.load(Ordering::Acquire)
        || data.modeset_requested[head_i].load(Ordering::Acquire) != want
        || data.modeset_active[head_i].load(Ordering::Acquire) != want
    {
        scanout_gate(head, 5, "mode generation changed between encode and submit");
        return Ok(());
    }
    // Coalesce the whole frame into ONE contiguous bulk_send. DLM sends each frame as a single
    // continuous bulk-OUT stream with NO short packet until the true end of the frame (its captured
    // URBs are exact 65536-byte multiples of the 1024 wMaxPacketSize, so no intermediate transfer
    // terminates short). vino used to issue one separate `bulk_send` per record-chunk, each capped
    // at 65024 (a *non*-multiple of 1024) so every chunk ended in a short packet -- on the theory
    // the dock used short packets as record delimiters. HW-observed 2026-07-17 refutes that: with
    // per-chunk short packets the dock accepts chunk 0 of a frame, then disconnects the instant
    // chunk 1 arrives as a fresh short-packet-terminated transfer mid-frame (protocol desync --
    // `chunk 0 ok` immediately followed by USB disconnect + ESHUTDOWN). A single bulk_send of the
    // whole frame fragments into full max-packet USB packets with only ONE final short packet,
    // matching DLM's wire shape exactly. On rejection the dock halts the endpoint (EPIPE); clear the
    // stall so the toggle resets to DATA0 and the next flip is a fresh attempt, not an EPROTO wedge.
    // On the first EP08 write after a mode-set, PREPEND this head's 10-record video-pipe arm burst
    // so the wire carries `[arm burst][video]` as one contiguous URB -- exactly DLM's cold-plug
    // frame 0 (RE 2026-07-18; see `arm_prefix_pending`). Without it every write EPIPEs (pipe not
    // armed) even for byte-exact content. The bit is cleared only on a successful write, so a failed
    // attempt re-prepends next flip.
    let head_bit = 1u32 << head;
    // Fresh untruncated DLM cold-plug captures show the 2560-byte ARM burst only on frame zero after
    // a mode-set. Normal subsequent frames begin directly with video records. The earlier
    // arm-every-frame conclusion came from submit-only Vino logs and is not wire evidence.
    let arm = if data
        .arm_prefix_pending
        .load(core::sync::atomic::Ordering::Acquire)
        & head_bit
        != 0
    {
        data.build_arm_burst_buf(head_i)
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
        pr_info!(
            "vino: scanout head={} superseded before video submit; frame discarded\n",
            head
        );
        return Ok(());
    }
    // Preserve the last readiness-to-video adjacency from the VINO session that lit both panels.
    // These are real CP status transactions (with EP84 replies drained by `send_cp`), paced at the
    // captured cadence, and only run for frame zero while the ARM prefix is present.
    if arm.is_some() {
        for _ in 0..VinoDrmData::PREWRITE_POLLS {
            data.poll_status(dev);
            fsleep(Delta::from_millis(VinoDrmData::PREWRITE_POLL_MS as i64));
        }
        pr_info!(
            "vino: inline pre-write paced poll ({}x @{}ms) before first video head={}\n",
            VinoDrmData::PREWRITE_POLLS,
            VinoDrmData::PREWRITE_POLL_MS,
            head
        );
    }
    // DLM starts frame zero directly with the ARM record (`00 00 1c 00 ...`) and later frames
    // directly with video records (`00 00 cc 08 ...`). The 16 zero bytes in the old extracted
    // `*_frame0.bin` files were introduced by that extractor, not present in usbmon payloads.
    // Record fragments are allocation boundaries only and are joined below into exact 64-KiB URBs
    // without a whole-frame coalescing allocation.
    let frame_count = frames.len();
    let image_len: usize = frames.iter().take(frame_count).map(|f| f.len()).sum();
    let startup = arm.is_some();
    // ★ Cold training + double-buffer replay (2026-07-25, `docs/DLM-DAMAGE-TILING.md`).
    //
    // A cold DLM link receives full frames back-to-back (24--35 ms start cadence, at most ~5 ms
    // between frames) until its downstream clock is programmed about 0.4 s later. The first Vino
    // sustain implementation re-encoded at FRAME_PERIOD_MS; the paired failing capture measured
    // 476--1489 ms between full-frame starts (418--1431 ms with no traffic after each frame), so it
    // was sustained in duration but never continuous on the wire. Reuse this already-encoded full
    // frame for a bounded eight-presentation burst.
    // Every presentation gets a fresh trailer/frame number and one per-frame CP sync below; ARM is
    // still present only on presentation zero.
    //
    // For ordinary partial damage, the dock is double-buffered,
    // so a one-shot damage written to only one buffer FLICKERS/tears between the two on scanout.
    // DLM was measured sending each damage EXACTLY TWICE (~100 ms apart, then idle) -- once per dock
    // buffer -- via the damage harness. Send each DELTA twice too (the existing `repeat_count`
    // re-sends with an advanced frame trailer), so both dock buffers get the change.
    let training = full
        && data.sustain_until.lock()[head_i]
            .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
    let repeat_count = if training {
        COLD_TRAINING_PRESENTATIONS
    } else if full {
        1
    } else {
        2
    };
    let first_wire_len = arm_len + image_len + super::video::wht::frame_trailer(head, seq0).len();
    pr_debug!(
        "vino: scanout head={} {} chunk(s) + {} B arm-prefix = {} B first write, {} presentation(s)\n",
        head,
        frame_count,
        arm_len,
        first_wire_len,
        repeat_count
    );
    // ★ Split the frame on EXACT 65536-B boundaries, not record boundaries. 65536 is a multiple of
    // the EP08 wMaxPacketSize (1024), so every transfer but the last ends on a full packet and does
    // NOT short-packet -- the dock therefore reads the whole sequence as ONE continuous frame, with
    // the single trailing short packet marking the end. This is byte-for-byte DLM's wire shape
    // (its pcap shows 65536+65536+65536+short per frame).
    //
    // Splitting at RECORD boundaries (sizes like 63786/61226, not 1024 multiples) made every chunk
    // end in a short packet, so the dock saw each as a separate full-screen frame and rejected the
    // mid-screen ones -- HW-observed as "chunk 0 sent, chunk 1 -> ESHUTDOWN".
    // ...and submit them ASYNC/PIPELINED, not one-at-a-time. Synchronous `bulk_send` waits for each
    // URB to complete before submitting the next, leaving a gap the dock reads as end-of-frame: HW
    // shows transfer 0 (65536 B) accepted and transfer 1 rejected, even on exact 1024-multiple
    // boundaries. DLM keeps several 65536-B URBs in flight back-to-back (libusb), so the dock sees
    // one uninterrupted frame. `bulk_out_queue` gives the same submit/reap pipelining.
    // ★ Stream the frame exactly like DLM: split on EXACT 65536-B boundaries (a 1024 multiple, so
    // only the FINAL short chunk terminates the frame) and push them through this head's PERSISTENT
    // 8-deep pipeline. Do NOT flush -- `send` reaps a slot only when the cursor wraps onto it, which
    // keeps ~8 URBs outstanding continuously, exactly as the xHCI trace shows DLM behaving. Draining
    // between frames (or rebuilding the queue) is what the dock rejects.
    const XFER: usize = 65536;
    let mut last_wire_len = 0usize;
    for repeat in 0..repeat_count {
        // A compositor mode change can arrive while the presentation is in flight. Never let the
        // old frame cross the new mode generation.
        if data.shutting_down.load(Ordering::Acquire)
            || data.modeset_requested[head_i].load(Ordering::Acquire) != want
            || data.modeset_active[head_i].load(Ordering::Acquire) != want
        {
            pr_info!(
                "vino: scanout head={} superseded during presentation; stopped at {}/{}\n",
                head,
                repeat,
                repeat_count
            );
            return Ok(());
        }

        let repeat_seq = seq0.wrapping_add(repeat);
        let frame_trailer = super::video::wht::frame_trailer(head, repeat_seq);
        // DLM prefixes ARM only to presentation zero. Every later presentation starts directly at
        // the image records and carries a freshly advanced three-record frame trailer.
        let arm_slice: &[u8] = if repeat == 0 {
            arm.as_ref().map_or(&[], |a| &a[..])
        } else {
            &[]
        };
        let wire_len = arm_slice.len() + image_len + frame_trailer.len();
        last_wire_len = wire_len;
        {
            let mut staging_slots = data.video_staging.lock();
            let staging_slot = &mut staging_slots[head_i];
            if staging_slot.is_none() {
                let mut staging = KVec::new();
                staging.resize(XFER, 0, GFP_KERNEL)?;
                *staging_slot = Some(staging);
            }
            let staging = staging_slot.as_mut().ok_or(kernel::error::code::ENOMEM)?;

            let mut qs = data.video_q.lock();
            let slot = &mut qs[head_i];
            if slot.is_none() {
                match dev.video_queue(head_i, 8, XFER) {
                    Ok(q) => {
                        *slot = Some(q);
                        pr_info!(
                            "vino: head={} persistent video queue opened (depth=8, {} B URBs)\n",
                            head,
                            XFER
                        );
                    }
                    Err(e) => return Err(e),
                }
            }
            let q = slot.as_mut().ok_or(kernel::error::code::ENODEV)?;
            // Scatter/gather cursor over [optional ARM][record chunks][trailer]. Join only one
            // transfer at a time in the reusable bounded staging allocation, eliminating the
            // former `KVec` whose capacity was the entire multi-megabyte frame.
            let arm_parts = usize::from(!arm_slice.is_empty());
            let trailer_parts = 1usize;
            let part_count = arm_parts + frame_count + trailer_parts;
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
                    } else if part_i < arm_parts + frame_count {
                        &frames[part_i - arm_parts][..]
                    } else {
                        &frame_trailer[..]
                    };
                    let n = (part.len() - part_off).min(dst.len() - dst_off);
                    dst[dst_off..dst_off + n].copy_from_slice(&part[part_off..part_off + n]);
                    dst_off += n;
                    part_off += n;
                    if part_off == part.len() {
                        part_i += 1;
                        part_off = 0;
                    }
                }
                if let Err(e) = q.send(dev.io(), dst, super::timeout()) {
                    pr_info!(
                        "vino: scanout head={} pipeline submit at off={}/{} failed\n",
                        head,
                        wire_off,
                        wire_len
                    );
                    let _ = dev.clear_video_halt(head_i);
                    return Err(e);
                }
                wire_off += data_len;
            }
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
            pr_info!(
                "vino: scanout head={} initial ARM+keyframe accepted ({} B on the wire)\n",
                head,
                wire_len
            );
            // Post-burst "stream commit" (`id=0x16 sub=0x4c`; see `cp::stream_commit`'s doc comment
            // and `project_post_burst_stream_commit_found_20260717` memory): DLM sends this twice
            // per head on the MAIN EP02 channel right after that head's video-endpoint arm burst,
            // before real video starts. In drm-v3 the arm burst is fused into the frame-zero EP08
            // URB, so the closest wire-equivalent point is here, immediately after that URB is
            // accepted. Dropped in the v3 rebase (defined in cp.rs, no call sites); restored to
            // re-enable the dock's downstream display engine (`2807a`/`2990d` fork -- see BLOCKER.md).
            for _ in 0..2 {
                match data.send_cp(dev, 0x16, 0, |ctr| super::cp::stream_commit(ctr, head)) {
                    Ok(()) => pr_info!("vino: stream-commit head={} ok\n", head),
                    Err(e) => pr_info!("vino: stream-commit head={} failed ({e:?})\n", head),
                }
            }
        }

        if let Err(e) = data.send_cp(dev, 0x14, 0, |ctr| super::cp::device_query_req(ctr, 0x000c)) {
            pr_info!(
                "vino: scanout head={} per-frame CP sync failed ({e:?})\n",
                head
            );
        }
        // Do not drain here. DLM maintains an eight-URB ring across frame boundaries; `send()`
        // reaps the old completion when a slot is reused, so transport errors still surface after
        // the ring wraps without introducing a per-frame pipeline bubble.
    }
    // Publish the new codec sequence only after every URB for this frame was submitted. A stale
    // generation or transport failure above leaves the old sequence intact for the next keyframe.
    data.scanout_seq.lock()[head_i] = next_seq.wrapping_add(repeat_count - 1);
    // The USB path accepted the complete image. Publish its content shadow only now; every early
    // return and transport error above deliberately leaves the previous dock-visible state intact.
    data.strip_hashes.lock()[head_i] = content_hashes.map(|hashes| StripHashState {
        w_pad,
        h_pad,
        hashes,
    });
    // A full keyframe was accepted -- this head may now send damage deltas until the next mode-set.
    if full {
        data.keyframe_pending
            .fetch_and(!kf_bit, core::sync::atomic::Ordering::Release);
    }
    // ★ 2026-07-20 RE: DLM keeps a CP dialogue running for the WHOLE streaming session -- one
    // `id=0x14 sub=0x0c` video-pipe sync per video frame (measured in the cold pcap: 33 CP-OUT
    // messages against 37 video frames in 0.45 s, ~74/sec, ~1:1 with frames). vino previously went
    // CP-SILENT the moment scanout started, so the dock's session timed out and it hard-reset a few
    // seconds after an otherwise-accepted write -- exactly the observed DELAYED reset. The status
    // dialogue must continue per frame; a one-shot burst does not keep the session alive.
    data.last_frame.lock()[head_i] = Some(Instant::<Monotonic>::now());
    pr_debug!(
        "vino: scanout head={} frame ok ({} presentation(s), {} B final write)\n",
        head,
        repeat_count,
        last_wire_len
    );
    Ok(())
}

/// Convert the mapped XRGB8888 frame to RGB565, Vino-encode it against the previous frame,
/// and bulk-write the resulting EP08 frame to the dock.
fn encode_and_send(
    dev: &BoundInterface<'_>,
    data: &VinoDrmData,
    head: u8,
    vaddr: *const u8,
    pitch: usize,
    rotation: u32,
    // The client's changed rectangles (identity rotation only; empty means no pixel update).
    // `encode_and_send_wht` uses these to send a damage delta (only changed strips) after the first
    // full keyframe -- see `docs/VIDEO-PARTIAL-UPDATE-DESIGN.md`.
    clips: &[(usize, usize, usize, usize)],
    w: usize,
    h: usize,
) -> Result {
    // WHT colour codec path -- the only scanout codec. Non-64x16-aligned modes (e.g. 1920x1080,
    // 1080%16=8) are handled by `encode_and_send_wht` PADDING the frame up to the next 64x16 multiple
    // and emitting full strip bands: DLM does exactly this (cold-pcap `ep08-replay-1080p.bin` has 68
    // bands = 1088 rows for a 1080p mode-set, the extra 8 rows padded), and the dock displays only
    // the mode's real size, ignoring the pad. (The old RLE fallback that faulted the dock on 1080p is
    // gone; this replaces it byte-shape-correctly.)
    encode_and_send_wht(dev, data, head, vaddr, pitch, rotation, clips, w, h)
}

// ---- Encoder ----------------------------------------------------------------

#[pin_data]
pub(super) struct VinoEncoder;

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
pub(super) struct VinoConnector {
    /// Index into the owning device's per-head EDID/presence arrays.
    head: u8,
}

#[derive(Clone, Default)]
pub(super) struct VinoConnectorState;

impl connector::DriverConnectorState for VinoConnectorState {
    type Connector = VinoConnector;
}

#[vtable]
impl connector::DriverConnector for VinoConnector {
    type Args = u8;
    type Driver = VinoDrmDriver;
    type State = VinoConnectorState;

    fn new(_device: &drm::Device<Self::Driver>, head: u8) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoConnector { head })
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
        if let Some(blob) = edids.get(connector.head as usize).and_then(Option::as_ref) {
            // A failed EDID update adds no modes; fall through to the built-in list rather than
            // reporting a count the core did not actually get.
            if let Ok(n) = connector.add_edid_modes(blob) {
                if n > 0 {
                    return n;
                }
            }
        }
        drop(edids);
        let _ = guard;
        // No downstream EDID yet: advertise the standard mode list up to the fallback resolution
        // and prefer it, keeping the connector usable until the dock delivers a real EDID.
        let n = connector.add_modes_noedid((FALLBACK_W as u32, FALLBACK_H as u32));
        connector.set_preferred_mode((FALLBACK_W as u32, FALLBACK_H as u32));
        n
    }

    /// Report the head connected once the dock has delivered this head's downstream EDID (a real
    /// monitor is attached and described) OR the bring-up work item confirmed CP engagement +
    /// this head's DISPLAY-CAP push (`heads_present`: on 3.4.26 the raw-EDID path can fail, so
    /// cached EDID alone would leave every connector permanently
    /// disconnected despite a fully-engaged dock). A head with neither stays disconnected rather
    /// than advertising a phantom output.
    fn detect(connector: &Connector<Self>, _force: bool) -> Status {
        let data: &VinoDrmData = connector.drm_dev();
        let head = connector.head as usize;
        let has_edid = data
            .cached_edids
            .lock()
            .get(head)
            .is_some_and(Option::is_some);
        let present = data
            .heads_present
            .load(core::sync::atomic::Ordering::Acquire)
            & (1 << head)
            != 0;
        if has_edid || present {
            Status::Connected
        } else {
            Status::Disconnected
        }
    }

    /// Prune modes whose pixel clock exceeds a single head's bandwidth ceiling
    /// ([`MAX_HEAD_CLOCK_KHZ`], ~4K@60).
    fn mode_valid(connector: &Connector<Self>, mode: &DisplayMode) -> ModeStatus {
        // Hard single-link ceiling (~4K@60) first.
        if mode.clock() > MAX_HEAD_CLOCK_KHZ {
            return ModeStatus::ClockHigh;
        }
        // Head 1 was clamped to 60 Hz here on 2026-07-25 because "allowing KWin to select 120 Hz on
        // both heads leaves the dock in the pre-activation `3fa43` state". That was a
        // misattribution: the dock stayed pre-activation because vino's EDID engage
        // (`id=0x16 sub=0x23`) carried a random byte in off23 and was rejected, so the downstream
        // sink was never enabled -- not because of the refresh rate. With that fixed both heads
        // reach `2807a`/`2990d` at 1440p120, and the cross-head budget below is the only rate limit
        // that belongs here.
        // Experiment: pin the head to 1280x720@60 (see `TEST_ONLY_720P60`).
        if TEST_ONLY_720P60 {
            let is_720p60 = mode.hdisplay() == 1280
                && mode.vdisplay() == 720
                && (59..=61).contains(&mode.vrefresh());
            if !is_720p60 {
                return ModeStatus::Bad;
            }
        }
        // Cross-head bandwidth: prune modes whose pixel rate exceeds THIS head's even share of the
        // dock's shared budget, so two heads can't each pass the per-head cap yet together overrun
        // the DisplayLink chip. Skipped when the budget is unknown (0) -- never a false rejection.
        // See `docs/CROSS-HEAD-BANDWIDTH-DESIGN.md`.
        let data: &VinoDrmData = connector.drm_dev();
        let budget = data.own_pixel_budget();
        if budget != 0 {
            let rate = u32::from(mode.hdisplay())
                .saturating_mul(u32::from(mode.vdisplay()))
                .saturating_mul(mode.vrefresh().max(0) as u32);
            if rate > budget {
                return ModeStatus::Bad;
            }
        }
        ModeStatus::Ok
    }
}

// ---- DDC/CI I2C adapter -----------------------------------------------------

/// Per-adapter DDC/CI context. DLM/EVDI exposes one I2C adapter per display head; the head is part
/// of every DisplayLink DDC control message and therefore cannot be inferred from a shared bus.
pub(super) struct VinoI2cContext {
    pub(super) dev: ARef<VinoDrmDevice>,
    pub(super) head: u8,
}

/// vino's per-head DDC/CI virtual I2C bus: a userspace monitor-control tool (`ddcutil`, the desktop
/// brightness slider via the I2C DDC path) writes a DDC/CI transaction to the monitor's I2C slave
/// on this adapter, and vino tunnels it to the downstream monitor over the dock's CP channel
/// (`cp::ddc_forward`, `id=0x36 sub=0x26`). Writes only for now (Get-VCP reads need the CP reply
/// path); a no-op until CP engages. KMS enable/disable deliberately does not generate DDC writes.
pub(super) struct VinoI2c;

impl i2c::BusController for VinoI2c {
    type Context = VinoI2cContext;

    fn master_xfer(ctx: &VinoI2cContext, msgs: &mut [i2c::Msg]) -> Result<usize> {
        let data: &VinoDrmData = &ctx.dev;
        let link = super::UsbLink::open(&data.io, data.eps)?;
        let dev = &link;
        // The I2C contract is all-or-nothing: return the number of messages transferred on full
        // success, or a negative errno on the first message that cannot be fulfilled -- never a
        // short positive count, which the core (and tools like ddcutil) would read as success for
        // the messages that were actually skipped.
        for msg in msgs.iter_mut() {
            if msg.addr() != super::cp::DDCCI_I2C_ADDR as u16 {
                // Only the DDC/CI slave exists on this virtual bus; nothing acks other addresses.
                return Err(ENXIO);
            }
            // Direction-typed access: a write message yields a shared slice, a read message a
            // mutable one, so this cannot accidentally write through a write-only message.
            let i2c::MsgBuffer::Write(write) = msg.buffer() else {
                // DDC/CI reads (Get-VCP) require decoding the dock's CP reply -- not wired yet.
                return Err(ENOTSUPP);
            };
            data.send_cp(dev, 0x36, 0, |ctr| {
                super::cp::ddc_forward(ctr, ctx.head, write)
            })?;
        }
        Ok(msgs.len())
    }

    fn functionality(_ctx: &VinoI2cContext) -> u32 {
        i2c::FUNC_I2C
    }
}

/// Apply the cached gamma ramp (three 256-entry 8-bit LUTs) to an `(r, g, b)` pixel, or return it
/// unchanged when no gamma is programmed.
#[inline]
fn apply_gamma(gamma: &Option<[u8; 768]>, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    match gamma {
        Some(t) => (t[r as usize], t[256 + g as usize], t[512 + b as usize]),
        None => (r, g, b),
    }
}

/// Map an output pixel `(dx, dy)` back to its source-framebuffer pixel `(sx, sy)` under a DRM
/// plane `rotation` bitmask (`DRM_MODE_ROTATE_*` | `DRM_MODE_REFLECT_*`, the values the
/// standard `drm_plane_create_rotation_property` exposes). `sw`/`sh` are the SOURCE
/// (framebuffer) dimensions. Rotation is clockwise; reflection is applied in source space
/// after rotation. Pure and total (saturating), so it is unit-tested directly. Applied per source
/// pixel in [`encode_and_send`]/[`encode_and_send_wht`] for the plane's rotation property.
pub(super) fn rot_src(rotation: u32, dx: usize, dy: usize, sw: usize, sh: usize) -> (usize, usize) {
    let xmax = sw.saturating_sub(1);
    let ymax = sh.saturating_sub(1);
    let rot = rotation & bindings::DRM_MODE_ROTATE_MASK;
    let (mut sx, mut sy) = if rot == bindings::DRM_MODE_ROTATE_90 {
        (dy, ymax.saturating_sub(dx))
    } else if rot == bindings::DRM_MODE_ROTATE_180 {
        (xmax.saturating_sub(dx), ymax.saturating_sub(dy))
    } else if rot == bindings::DRM_MODE_ROTATE_270 {
        (xmax.saturating_sub(dy), dx)
    } else {
        (dx, dy) // ROTATE_0 / unset
    };
    if rotation & bindings::DRM_MODE_REFLECT_X != 0 {
        sx = xmax.saturating_sub(sx);
    }
    if rotation & bindings::DRM_MODE_REFLECT_Y != 0 {
        sy = ymax.saturating_sub(sy);
    }
    (sx, sy)
}
