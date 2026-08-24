// SPDX-License-Identifier: GPL-2.0

//! DRM/KMS integration for Vino.
//!
//! Each dock connector has a primary plane, cursor plane, CRTC, encoder and connector. Framebuffers
//! are copied into driver-owned snapshots before atomic completion, then compressed and sent by
//! per-connector workers. Connector modes come from downstream EDID tunneled over the dock's
//! control protocol.

use core::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use kernel::{
    drm,
    drm::kms::{
        self,
        connector::{self, Connector, ConnectorGuard, ConnectorModeValidation, ModeStatus, Status},
        crtc::{self, CrtcAtomicCheck, CrtcAtomicCommit, RawCrtc as _, RawCrtcState as _},
        encoder,
        modes::DisplayMode,
        plane::{self, PlaneAtomicCheck, PlaneAtomicCommit, RawPlaneState as _},
        vblank::{
            OwnedVblankRef, RawVblankCrtcState as _, VblankGuard, VblankSupport, VblankTimestamp,
        },
        KmsDriver, ModeConfigGuard, ModeConfigInfo, ModeObject as _, UnregisteredKmsDevice,
    },
    error::code::{EINVAL, ENODEV, ENOMEM, ENOTSUPP},
    impl_has_hr_timer,
    interrupt::LocalInterruptDisabled,
    io::Io,
    prelude::*,
    sync::{
        aref::ARef, new_mutex, new_spinlock, new_spinlock_irq, Arc, ArcBorrow, Completion, Mutex,
        SpinLock, SpinLockIrq,
    },
    time::{
        delay::{fsleep, udelay},
        hrtimer::{
            ArcHrTimerHandle, HrTimer, HrTimerCallback, HrTimerCallbackContext, HrTimerPointer,
            HrTimerRestart, RelativeHardMode,
        },
        Delta, Instant, Monotonic,
    },
    workqueue::{
        self, impl_has_delayed_work, impl_has_work, new_delayed_work, new_work, DelayedWork, Work,
        WorkItem,
    },
    xxhash,
};

mod activation;
mod bracket;
mod cp_session;
mod dispatch;
mod driver;
mod limits;
mod mode_objects;
mod presence;
mod scanout;
mod settings;
mod stream;
mod timeline;
mod worker;

pub(crate) use driver::VinoObject;
use limits::{active_pixel_rate, timing_key, DEFAULT_MAX_HEAD_CLOCK_KHZ, DEFAULT_MAX_REFRESH_HZ};
pub(super) use mode_objects::{
    PlaneArgs, VblankTimer, VinoConnector, VinoCrtc, VinoEncoder, VinoPlane,
};
use scanout::{read_cursor_bgra, run_pending_scanout, snapshot_to_shadow, src_dims};
// Almost every item in `timeline` is read by the activation path; naming them individually
// would list the module.
pub(crate) use timeline::*;

/// Connector mode used until a downstream EDID is available.
const FALLBACK_W: i32 = 2560;
const FALLBACK_H: i32 = 1440;

/// Primary-plane format list (opaque 32bpp scanout).
static PRIMARY_FORMATS: [u32; 1] = [drm::fourcc::XRGB8888];

/// Primary-plane format list for a dock whose pipeline carries 10 bits per channel.
///
/// `XRGB8888` stays first: it is what every ordinary desktop commits, and a compositor choosing
/// between them should not be pushed towards the deeper one by list order alone.
static PRIMARY_FORMATS_HDR: [u32; 2] = [drm::fourcc::XRGB8888, drm::fourcc::XRGB2101010];

/// The only framebuffer layout any of these planes accepts.
///
/// Publishing it gives userspace an `IN_FORMATS` property; without it the plane advertises formats
/// with no modifier information at all.
static LINEAR_MODIFIER: [u64; 1] = [drm::fourcc::FORMAT_MOD_LINEAR];

/// Cursor-plane format list.
static CURSOR_FORMATS: [u32; 1] = [drm::fourcc::ARGB8888];

/// Stream-marker state which powers down a connector's downstream sink.
///
/// The resulting probe silence must be paired with [`VinoDrmData::self_blanked`] so it is not
/// mistaken for a physical disconnect.
///
/// The vendor drives a DL-3x00 sink down with `0x2f` state 1 followed by `0x2e` state 3, and back
/// up with `0x2e` state 0 then `0x2f` state 0. State 1 is what has been verified against a
/// DL-6xxx dock, so the field is likely a bitmask rather than an enumeration; a platform that
/// needs the vendor's exact value will have to carry it per profile.
const BLANK_MARKER_STATE: u8 = 1;

/// Delay before retrying a transient asynchronous control operation.
const KMS_RETRY_MS: u32 = 50;

/// Consecutive deferrals of a KMS command batch before it is dropped.
///
/// A command that fails because the dock has stopped answering will fail again for the same
/// reason, and retrying it forever reprograms a dead link twenty times a second: it buries every
/// other message in the log and keeps writing to a dock that has already abandoned its session.
/// Anything genuinely transient clears well inside this, and a later commit or hotplug queues
/// fresh work regardless. The bound is a little over the dock's own no-answer watchdog.
const KMS_RETRY_LIMIT: u32 = 128;

/// Maximum number of physical downstream connectors Vino exposes.
///
/// Ridge docks use the first two. Navarro has four physical DP sockets; connectors 0/2 share
/// bulk endpoint 0x08 and connectors 1/3 share 0x0a. `DockProfile::connectors` selects the
/// active prefix at runtime, while this constant keeps the DRM object layout fixed at registration.
pub(crate) const MAX_CONNECTORS: usize = 4;

/// Bulk transfer size on the video endpoints: a multiple of the 1024-byte maximum packet size, so
/// only a frame's final transfer terminates short.
const VIDEO_XFER: usize = 65536;

/// Stream reports a connector sends after its stream is opened, on a dock that carries video on the
/// control pipe.
///
/// The report restates the mode on a stream the dock has just been handed. DLM sends fourteen of
/// them across 7115 frames and never two together, so one per stream is the whole of it; sending
/// one per frame spends both the dock's bandwidth budget and its sealed block counter on a record
/// it is not waiting for.
const STREAM_REPORT_BURST: u32 = 1;

/// Frame of a fresh stream that carries the report, counting the prologue as zero.
///
/// DLM restates the mode with the third frame, not the first: the two frames before it carry
/// nothing but pixels and their closing record.
const STREAM_REPORT_FRAME: u32 = 2;

/// Whether a presentation named a ring slot, and so consumed a frame counter.
///
/// The counter belongs to whichever record names the slot: the frame opener on a dock that
/// carries one, the frame trailer on a dock that carries the transition there. A presentation with
/// neither says nothing about the ring, and counting it anyway puts every later record one slot
/// ahead of the buffer the host actually wrote, so the dock scans out a buffer nothing has filled.
pub(crate) fn names_ring_slot(opener: &[u8], trailer: &[u8]) -> bool {
    !opener.is_empty() || !trailer.is_empty()
}

/// Maximum number of individual frame-damage rectangles re-converted per flip before they are
/// collapsed into a single bounding box. Bounds the stack array used on the atomic-commit path
/// (no per-flip allocation); a compositor that reports more clips than this just gets a coarser
/// (still correct) repaint.
const MAX_DAMAGE_CLIPS: usize = 16;
/// Minimum interval between normal frames for one connector.
const FRAME_PERIOD_MS: i64 = 5;
/// The coalescing window in microseconds. Whole-millisecond arithmetic truncated the elapsed time
/// and forced a 1 ms minimum sleep, so a frame could wait materially longer than the window.
const FRAME_PERIOD_US: i64 = FRAME_PERIOD_MS * 1000;
/// Interval between keepalive status queries on a dock whose profile does not state one.
const STATUS_PERIOD_MS: i64 = 250;

/// Activation timing relative to the mode-set submission.
const PROMPT_VIDEO_MS: i64 = 110;
const PROMPT_CLOSE_2F_MS: i64 = 123;
const PROMPT_CLOSE_2E_MS: i64 = 125;
/// Upper bound used to quiesce an already-running keepalive iteration.
const PROMPT_KEEPALIVE_QUIESCE_MS: i64 = 40;
/// Minimum interval between streaming status polls (`id=0x14 sub=0x0c`).
///
/// Issued per presentation this would be 200-600 CP round-trips a second across two connectors at
/// ~100 fps, every one of them serialising on the single control link, and whichever connector
/// looped fastest would re-acquire it immediately and starve the other. DLM issues ~3.8 per second,
/// so a quarter-second floor matches the reference and leaves the link free for the other
/// connector.
const STATUS_POLL_MIN_MS: i64 = 250;
const PROMPT_TRAINING_OPEN_MS: i64 = 0;
const PROMPT_TRAINING_TAIL_MS: i64 = 400;
/// How long [`VinoDrmData::blank_connector`] keeps presenting black when a CRTC is disabled.
///
/// It only has to outlast the dock's buffer rotation, which is at most three presentations on the
/// current profiles; a black frame is ~200 KB and presents in a couple of milliseconds, so this is
/// generous by an order of magnitude and still finishes well inside a DPMS transition. It is not a
/// training window -- nothing downstream needs settling -- so it does not reuse
/// [`PROMPT_TRAINING_TAIL_MS`].
const BLANK_PRESENT_MS: i64 = 120;

/// `edid_target` sentinel: nobody is waiting for an EDID.
const NO_EDID_TARGET: u32 = u32::MAX;

/// How much of a DL7400 frame's image data precedes its per-strip parameter map.
///
/// The map tells the dock how to read the records around it, and where it sits in the frame is
/// load-bearing rather than cosmetic: the vendor's stream carries it this far into the image
/// records of every frame, and a frame that carries the same valid records after all of its pixels
/// instead is accepted twice and then leaves the endpoint permanently un-drained.
pub(crate) const NAVARRO_PARAM_IMAGE_OFFSET: usize = 115168;

/// How many leading record chunks precede the parameter map in a frame.
///
/// The map has to land on a record boundary, and an encoded frame's chunks are the boundaries this
/// side knows: each holds whole records, while [`NAVARRO_PARAM_IMAGE_OFFSET`] itself falls wherever
/// a frame's record lengths put it. So round the vendor's offset back to a chunk, and keep at least
/// one chunk in front of the map -- what the dock will not take is the map arriving after every
/// record it describes, not the exact byte it arrives at.
pub(crate) fn param_map_chunk_split(frames: &[KVec<u8>]) -> usize {
    let mut consumed = 0usize;
    let mut chunks = 0usize;
    for f in frames {
        if consumed + f.len() > NAVARRO_PARAM_IMAGE_OFFSET {
            break;
        }
        consumed += f.len();
        chunks += 1;
    }
    chunks.max(1).min(frames.len())
}
type DamageRect = (usize, usize, usize, usize);
type BoundInterface<'a> = super::UsbLink<'a>;

/// A dock's unspent sustained-throughput credit.
///
/// Credit accrues at the profile's rate and is capped at one second of it, so a dock idle for a
/// minute does not bank a minute of bytes and then hand them to the endpoint at once. Spending is
/// allowed to overdraw: a frame's size is known only once it is encoded, and refusing to send a
/// frame already committed to the wire would strand it. The debt is repaid before the next frame
/// is selected, which is what turns the ledger into a rate.
pub(crate) struct StreamCredit {
    bytes: i64,
    topped_up: Option<Instant<Monotonic>>,
}

impl StreamCredit {
    fn new() -> Self {
        Self {
            bytes: 0,
            topped_up: None,
        }
    }
}

/// Credit accrued over `elapsed_us` at `bps`, saturating rather than wrapping on a long idle.
pub(crate) fn stream_credit_accrued(bps: u32, elapsed_us: i64) -> i64 {
    let bps = i64::from(bps);
    elapsed_us
        .max(0)
        .saturating_mul(bps)
        .checked_div(1_000_000)
        .unwrap_or(0)
}

/// Microseconds until an overdrawn ledger is back in credit.
pub(crate) fn stream_credit_wait_us(bps: u32, bytes: i64) -> Option<i64> {
    if bytes >= 0 {
        return None;
    }
    let bps = i64::from(bps).max(1);
    Some(
        bytes
            .saturating_neg()
            .saturating_mul(1_000_000)
            .checked_div(bps)
            .unwrap_or(0)
            .saturating_add(1),
    )
}

/// Presentations made by one logical scanout submission.
///
/// Cold training remains a transport exception for docks with a dedicated video endpoint.  Normal
/// keyframe and delta counts are profile data because ring depth alone does not describe how a
/// platform expects the host to populate that ring.
pub(crate) fn frame_presentation_count(
    policy: super::profile::FrameDelivery,
    full: bool,
    cold_training: bool,
    video_on_ctrl_pipe: bool,
) -> u32 {
    if full && cold_training && !video_on_ctrl_pipe {
        COLD_TRAINING_PRESENTATIONS
    } else if full {
        u32::from(policy.keyframe_presentations.max(1))
    } else {
        u32::from(policy.delta_presentations.max(1))
    }
}

/// Account one accepted logical submission against per-strip delivery debt.
pub(crate) fn pay_damage_debt(debt: &mut [u8], full: bool) {
    if full {
        debt.fill(0);
    } else {
        for remaining in debt {
            *remaining = remaining.saturating_sub(1);
        }
    }
}

/// Content of the last frame successfully submitted for one connector, represented in the dock's
/// native 64x16 strip grid. KWin frequently omits `FB_DAMAGE_CLIPS` when switching framebuffer
/// objects; a raw-content shadow is therefore the authoritative way to distinguish an unchanged
/// flip from a real repaint. `KVVec` permits vmalloc fallback for the roughly 29-KiB 1440p hash
/// table.
struct StripHashState {
    padded_width: usize,
    padded_height: usize,
    hashes: KVVec<u64>,
    /// Encoded strip bodies, parallel to `hashes`. Retransmit debt can reuse a body while its
    /// pixels and encoding tag remain unchanged.
    bodies: KVec<KVec<u8>>,
    /// Everything other than the strip's own pixels that its encoded bytes depend on.
    ///
    /// The hash covers the raw framebuffer, so a change that alters the ENCODED output without
    /// altering the source pixels would otherwise serve a stale body. Gamma is exactly that: a new
    /// LUT re-maps every pixel on the way into the codec while leaving the framebuffer identical,
    /// and although a gamma change owes a keyframe, a keyframe re-selects every strip rather than
    /// invalidating anything, so the reuse test would still hit. Rotation is the same hazard, and
    /// is handled by only ever caching under identity rotation.
    tag: u64,
}

/// The DRM driver marker type.
pub(super) struct VinoDrmDriver;

/// Convenience alias for our concrete `drm::Device`.
pub(super) type VinoDrmDevice = drm::Device<VinoDrmDriver>;

/// Active control-protocol session.
///
/// `wire_seq` counts AES-CTR content blocks; the authentication tag does not consume keystream.
/// `counter` is the inner protocol counter. The enclosing mutex advances both atomically with a
/// complete request/reply transaction.
pub(super) struct CpLink {
    ks: kernel::crypto::Secret<16>,
    riv: [u8; 8],
    wire_seq: u32,
    counter: u16,
    ep84_q: Option<super::usb::BulkInQueue>,
}

/// How long the dock may answer nothing at all before the session is abandoned.
///
/// Measured in silence rather than in unanswered messages, because those are not the same thing. A
/// connector whose sink is down leaves its own re-engage unanswered every few seconds for the life
/// of the session while its sibling drives a lit panel and the status dialogue replies throughout;
/// counting messages tears that session down. A dock with a video pipe of its own answers a lit,
/// idle link continuously, so real silence there is unambiguous.
const CP_SILENCE_LIMIT_MS: i64 = 5000;

/// The same limit for a dock that shares its control pipe with video.
///
/// Silence is not evidence of anything on such a dock: the vendor's own capture has it say nothing
/// for 79 s while a panel is lit, and the vendor sends nothing either. Applying the short limit
/// there abandons a working session, and the reset that follows is what turns a stalled dock into
/// one that has to be unplugged.
const CP_SILENCE_LIMIT_SHARED_MS: i64 = 90_000;

/// How often the watchdog checks the silence deadline.
const CP_WATCHDOG_PERIOD_MS: u32 = 1000;

/// A control-protocol operation deferred from the non-blocking atomic callbacks.
enum KmsCmd {
    ModeSet {
        connector: u8,
        timing: super::cp::Timing,
    },
    CursorCreate {
        connector: u8,
        w: u16,
        h: u16,
    },
    CursorImage {
        connector: u8,
        w: u16,
        h: u16,
        bgra: KVec<u8>,
    },
    CursorMove {
        connector: u8,
        x: u16,
        y: u16,
        /// The dock's own visible flag. Hiding by parking the cursor at `u16::MAX` instead left a
        /// ghost pointer at the top-left of both panels: the dock wraps an out-of-range origin
        /// rather than clipping the cursor away.
        visible: bool,
    },
    /// Drive the stream black and close its control-protocol bracket.
    Blank {
        connector: u8,
    },
}

impl KmsCmd {
    fn connector(&self) -> usize {
        match self {
            Self::ModeSet { connector, .. }
            | Self::CursorCreate { connector, .. }
            | Self::CursorImage { connector, .. }
            | Self::CursorMove { connector, .. }
            | Self::Blank { connector } => *connector as usize,
        }
    }
}

/// The cursor state one connector's dock connector is holding, as last acknowledged on the wire.
///
/// A mode set leaves the dock's cursor undefined -- `owe_keyframe` says so by bumping
/// `cursor_epoch` -- but vino could only act on that when the compositor next committed the cursor
/// plane. A pointer that is not moving produces no commit, so the cursor simply vanished until it
/// was moved, and every mode set is now dock-wide, so it vanished on *both* panels for a change
/// made to one. Keeping the last accepted bitmap and position lets the mode-set path put it back
/// itself.
struct CursorShot {
    w: u16,
    h: u16,
    bgra: KVec<u8>,
    x: u16,
    y: u16,
    visible: bool,
}

struct PendingKmsHead {
    stream: Option<KmsCmd>,
    cursor_create: Option<KmsCmd>,
    cursor_image: Option<KmsCmd>,
    cursor_move: Option<KmsCmd>,
}

impl PendingKmsHead {
    const fn new() -> Self {
        Self {
            stream: None,
            cursor_create: None,
            cursor_image: None,
            cursor_move: None,
        }
    }

    fn slot(&mut self, cmd: &KmsCmd) -> &mut Option<KmsCmd> {
        match cmd {
            KmsCmd::ModeSet { .. } | KmsCmd::Blank { .. } => &mut self.stream,
            KmsCmd::CursorCreate { .. } => &mut self.cursor_create,
            KmsCmd::CursorImage { .. } => &mut self.cursor_image,
            KmsCmd::CursorMove { .. } => &mut self.cursor_move,
        }
    }

    fn update(&mut self, cmd: KmsCmd) {
        let slot = self.slot(&cmd);
        *slot = Some(cmd);
    }

    /// Restore a failed operation unless a newer desired operation already occupies its slot.
    fn retry(&mut self, cmd: KmsCmd) {
        let slot = self.slot(&cmd);
        if slot.is_none() {
            *slot = Some(cmd);
        }
    }

    fn has_stream(&self) -> bool {
        self.stream.is_some()
    }
}

struct PendingKms {
    connectors: [PendingKmsHead; MAX_CONNECTORS],
}

impl PendingKms {
    const fn new() -> Self {
        Self {
            connectors: [const { PendingKmsHead::new() }; MAX_CONNECTORS],
        }
    }

    fn is_empty(&self) -> bool {
        self.connectors.iter().all(|connector| {
            connector.stream.is_none()
                && connector.cursor_create.is_none()
                && connector.cursor_image.is_none()
                && connector.cursor_move.is_none()
        })
    }

    /// Discard every queued command. Used when the link they address is gone; see
    /// `KMS_RETRY_LIMIT`.
    fn clear(&mut self) {
        self.connectors = [const { PendingKmsHead::new() }; MAX_CONNECTORS];
    }

    fn has_stream(&self) -> bool {
        self.connectors.iter().any(PendingKmsHead::has_stream)
    }

    fn update(&mut self, cmd: KmsCmd) {
        if let Some(pending) = self.connectors.get_mut(cmd.connector()) {
            pending.update(cmd);
        }
    }

    fn retry(&mut self, cmd: KmsCmd) {
        if let Some(pending) = self.connectors.get_mut(cmd.connector()) {
            pending.retry(cmd);
        }
    }

    /// Restore a drained batch without replacing newer state published while it was executing.
    fn retry_batch(&mut self, batch: Self) {
        for connector in batch.connectors {
            let PendingKmsHead {
                stream,
                cursor_create,
                cursor_image,
                cursor_move,
            } = connector;
            for cmd in [stream, cursor_create, cursor_image, cursor_move]
                .into_iter()
                .flatten()
            {
                self.retry(cmd);
            }
        }
    }
}

fn kms_error_retryable(error: Error) -> bool {
    error != EINVAL && error != ENOTSUPP
}

/// Latest primary-plane flip awaiting compression on the deferred worker. The framebuffer is
/// refcounted, so it remains valid after the atomic commit callback returns. There is one slot per
/// connector: a newer flip replaces an older unsent flip instead of building an unbounded queue
/// behind a slow encoder. When replacement could lose accumulated damage, the newer flip is
/// promoted to a full-output damage rectangle.
struct PendingScanout {
    connector: u8,
    rotation: plane::Rotation,
    clips: [DamageRect; MAX_DAMAGE_CLIPS],
    nclips: usize,
    w: usize,
    h: usize,
    /// Which private surface contains this commit's pixels.
    shadow_idx: usize,
    /// Generation of `shadow_idx`, used to reject a slot replaced before the worker claims it.
    shadow_generation: u64,
}

impl Clone for PendingScanout {
    fn clone(&self) -> Self {
        Self {
            connector: self.connector,
            rotation: self.rotation,
            clips: self.clips,
            nclips: self.nclips,
            w: self.w,
            h: self.h,
            shadow_idx: self.shadow_idx,
            shadow_generation: self.shadow_generation,
        }
    }
}

/// Private snapshots let atomic commit copy into one while the worker encodes another. Copying
/// before flip completion ensures that the compositor cannot reuse storage while Vino reads it.
///
/// Three, not two: a slot can be reserved by the encoder (`inflight`) *and* by a snapshot that has
/// dropped the pool lock to copy (`writing`) at the same time, which with two slots left none free
/// and silently dropped the commit. Three keeps one available in that state.
const SHADOW_SLOTS: usize = 3;

/// Maximum number of prepared compositor buffers retained per connector.
///
/// Compositors normally rotate through a small swapchain. Keeping four validated mappings moves
/// vmap preparation out of repeated flips while bounding pinned memory when a client reallocates.
const SOURCE_BINDINGS: usize = 4;

struct SourceBinding {
    framebuffer: ARef<kms::framebuffer::Framebuffer<VinoDrmDriver>>,
    mapping: kms::framebuffer::FramebufferVMapOwned<VinoObject>,
}

struct SourceBindingCache {
    entries: [Option<Arc<SourceBinding>>; SOURCE_BINDINGS],
    next: usize,
}

impl SourceBindingCache {
    const fn new() -> Self {
        Self {
            entries: [const { None }; SOURCE_BINDINGS],
            next: 0,
        }
    }

    fn get(
        &mut self,
        fb: &kms::framebuffer::Framebuffer<VinoDrmDriver>,
    ) -> Result<Arc<SourceBinding>> {
        if let Some(binding) = self
            .entries
            .iter()
            .flatten()
            .find(|binding| &*binding.framebuffer == fb)
        {
            return Ok(binding.clone());
        }

        let binding = Arc::new(
            SourceBinding {
                framebuffer: ARef::from(fb),
                mapping: fb.owned_vmap::<VinoObject>()?,
            },
            GFP_KERNEL,
        )?;
        self.entries[self.next] = Some(binding.clone());
        self.next = (self.next + 1) % SOURCE_BINDINGS;
        Ok(binding)
    }

    fn discard(&mut self) {
        self.entries = [const { None }; SOURCE_BINDINGS];
        self.next = 0;
    }
}

struct ShadowSurface {
    w: usize,
    h: usize,
    pixels: KVVec<u8>,
    /// Per-strip content hashes, computed while copying into this immutable snapshot.
    hashes: KVVec<u64>,
    /// Scratch holding one `STRIP_H`-row band of the source, in this surface's packed stride.
    ///
    /// The snapshot reads the source one full row at a time into this buffer and then hashes and
    /// copies the band's strips out of it, so the source is read once per frame, sequentially, and
    /// every strip's second pass hits a buffer small enough to stay in cache. At 2560 wide it is
    /// 160 KiB against the 14.7 MB of `pixels`.
    band: KVVec<u8>,
}

struct ShadowSlot {
    generation: u64,
    surface: Option<ShadowSurface>,
}

impl ShadowSlot {
    const fn new() -> Self {
        Self {
            generation: 0,
            surface: None,
        }
    }
}

/// One connector's shadow surfaces. Locked per connector: the snapshot copies ~14.7 MB while
/// holding this, and it runs on the compositor's non-blocking commit tail, so a device-wide lock
/// made one connector's commit stall the other's -- measured at up to 4.2 ms, half a 120 Hz frame
/// budget.
struct ShadowPool {
    slots: [ShadowSlot; SHADOW_SLOTS],
    inflight: Option<usize>,
    /// Slot currently being written by a snapshot that has released the pool lock.
    ///
    /// The copy is far too long to hold the lock across, so the commit takes the slot's surface
    /// out, drops the lock and copies into it unlocked. This marks the slot reserved for that
    /// window, exactly as `inflight` does for the encoder's side.
    writing: Option<usize>,
    source_bindings: SourceBindingCache,
}

impl ShadowPool {
    const fn new() -> Self {
        Self {
            slots: [const { ShadowSlot::new() }; SHADOW_SLOTS],
            inflight: None,
            writing: None,
            source_bindings: SourceBindingCache::new(),
        }
    }

    fn discard(&mut self) {
        for slot in &mut self.slots {
            slot.generation = slot.generation.wrapping_add(1);
            slot.surface = None;
        }
        self.source_bindings.discard();
    }
}

/// How long a cold downstream link is fed keyframes at frame cadence; see `sustain_window`.
const SUSTAIN_MS: i64 = 3000;

/// Delay before the one-shot post-keyframe repaint.
const SETTLE_REPAINT_MS: i64 = 1200;

/// Number of post-keyframe repaints. Cold-link training uses its separate bounded deadline.
const SETTLE_REPAINTS: u32 = 1;

/// Longest the DL7400 tolerates a silent video endpoint before it tears the link down.
///
/// Measured twice, with very different transfer shapes: a full 204 KB frame and a single 4 KB
/// image record both ended with every outstanding URB completing `-ESHUTDOWN` 1.06 s and 1.10 s
/// after the last video byte, the dock going deaf on the control plane at the same instant. DLM
/// never gets near it -- it pairs a sealed report with every frame, a median 9-19 ms apart and at
/// most 1.0 s apart even when the desktop is still.
const NAVARRO_VIDEO_QUIET_MS: i64 = 1000;

/// Period at which an idle DL7400 connector is re-fed, comfortably inside
/// [`NAVARRO_VIDEO_QUIET_MS`].
const NAVARRO_KEEPALIVE_MS: i64 = 250;

/// Keep a missed repaint from being enough to trip the dock's teardown.
const _: () = assert!(NAVARRO_KEEPALIVE_MS * 3 <= NAVARRO_VIDEO_QUIET_MS);

/// DRM device-private data: the bound USB interface, engaged CP session, connector state, deferred
/// scanout slots and per-connector transport state.
#[pin_data]
pub(super) struct VinoDrmData {
    /// The USB I/O-permitted window for this device's interface, shared with the persistent
    /// queues. `disconnect()` closes it, after which every transfer path here fails cleanly
    /// instead of touching an unbound interface.
    pub(super) io: Arc<super::usb::IoWindow>,
    /// The dock's endpoints, resolved and direction/type-checked once during probe.
    pub(super) endpoints: super::Endpoints,
    /// Stops every producer before unplug drains the embedded work item. This is checked while
    /// holding the producer's queue lock so a late atomic callback cannot enqueue a self-owning
    /// `ARef<VinoDrmDevice>` after `cancel_sync()` has already returned.
    shutting_down: AtomicBool,
    #[pin]
    cp_link: Mutex<Option<CpLink>>,
    /// When the dock last answered anything, and whether a session exists at all.
    ///
    /// Deliberately outside `cp_link`. A dock that has stopped answering must be stopped talking
    /// to, but the thread that discovers this is the one already stuck: `usb_bulk_msg` honours its
    /// own timeout and then kills the URB, and *that* wait is unbounded, so a controller which
    /// will not retire the transfer leaves the caller blocked uninterruptibly with `cp_link` held.
    /// Everything that has to take the mutex to learn the link is stuck therefore blocks behind
    /// the very transfer it is trying to diagnose -- including the keepalive's own liveness check.
    /// A spinlock and an atomic are always available.
    #[pin]
    cp_last_reply: SpinLock<Instant<Monotonic>>,
    cp_session_live: AtomicBool,
    /// Set once this device has been asked to reset itself out of a wedged session.
    ///
    /// One attempt only. A reset that works re-probes into a fresh device with this cleared; a
    /// reset that does not must not become a loop.
    cp_reset_queued: AtomicBool,
    /// Watchdog that enforces the silence deadline from off the control path.
    ///
    /// The keepalive cannot do this itself: it *is* the thread that wedges, so its own check at
    /// the top of the loop is never reached again. Scheduled on the system queue rather than
    /// vino's, which the stuck transaction owns.
    #[pin]
    cp_watchdog: DelayedWork<VinoDrmDevice, 5>,
    /// Latest desired control/KMS state per connector.
    #[pin]
    pending_kms: Mutex<PendingKms>,
    /// Coalescing per-connector scanout slots consumed by `cmd_work`.
    ///
    /// Compression and USB submission may sleep and therefore cannot run in
    /// `atomic_update`.
    #[pin]
    pending_scanout: Mutex<[Option<PendingScanout>; MAX_CONNECTORS]>,
    /// A one-shot repaint of the connector's newest known framebuffer. Cleared as soon as it is
    /// taken, or whenever a real flip arrives (that flip already carries newer content, so the
    /// redundant repaint is pointless). See [`SETTLE_REPAINT_MS`] for the hardware observation
    /// behind it.
    ///
    /// The `bool` is "promote to a full keyframe". It is true for the post-keyframe settle repaint,
    /// whose job is to replace a stale surface. It is false for a *debt* repaint, which carries
    /// outstanding `dirty_ttl` retransmissions to the dock's second buffer without promoting them
    /// to a keyframe.
    #[pin]
    settle_repaint: Mutex<[Option<(Instant<Monotonic>, PendingScanout, bool)>; MAX_CONNECTORS]>,
    /// Private committed surfaces and their worker ownership state.
    #[pin]
    shadow: [Mutex<ShadowPool>; MAX_CONNECTORS],
    /// Active software-vblank timers. The device owns their cancellation handles so shutdown does
    /// not depend on atomic-disable callbacks running. A spinlock is required because
    /// `enable_vblank` runs with local interrupts disabled.
    #[pin]
    vblank: SpinLock<[Option<(Arc<VblankTimer>, ArcHrTimerHandle<VblankTimer>)>; MAX_CONNECTORS]>,
    /// Work item that drains control/KMS commands.
    #[pin]
    cmd_work: DelayedWork<VinoDrmDevice>,
    /// Independent per-connector scanout workers. Their work IDs are const generics, so each
    /// connector has an explicit field; transport state is taken from per-connector slots while a
    /// frame is submitted.
    #[pin]
    scanout_work_h0: Work<VinoDrmDevice, 1>,
    #[pin]
    scanout_work_h1: Work<VinoDrmDevice, 2>,
    #[pin]
    scanout_work_h2: Work<VinoDrmDevice, 3>,
    #[pin]
    scanout_work_h3: Work<VinoDrmDevice, 4>,
    /// Dedicated queue for initial authentication and the steady-state control session.
    session_queue: workqueue::OwnedQueue,
    /// Ordered queue for runtime KMS and cursor control transactions.
    kms_queue: workqueue::OwnedQueue,
    /// Per-device unbound queue for the two scanout workers.
    scanout_queue: workqueue::OwnedQueue,
    /// Downstream EDID per connector. Connector callbacks use their connector index to read this
    /// owned state; publishing EDID therefore requires no raw pointer back into a DRM mode object.
    #[pin]
    cached_edids: Mutex<[Option<KVec<u8>>; MAX_CONNECTORS]>,
    /// Bit N is set once CP confirms that a real downstream monitor is present on connector N.
    connectors_present: core::sync::atomic::AtomicU32,
    /// Each connector's gamma ramp cached from its CRTC atomic hook as three 256-entry 8-bit LUTs
    /// (`[r; 256] ++ [g; 256] ++ [b; 256]`), or `None` for identity. Cached here (not read from the
    /// CRTC state) because scanout runs in the plane path; each entry is `Copy`, so the scanout
    /// snapshots its connector's entry under the lock and applies it without holding the lock in
    /// the pixel loop. Per connector so a second display's gamma cannot clobber the first's.
    #[pin]
    color: Mutex<[Option<super::color::ColorPipeline>; MAX_CONNECTORS]>,
    /// Per-connector strip hashes for the last frame accepted by the USB submission path. Updated
    /// only after the complete frame has been queued, so a failed transfer can never advance the
    /// shadow beyond what the dock may actually display.
    #[pin]
    strip_hashes: Mutex<[Option<StripHashState>; MAX_CONNECTORS]>,
    /// The DL7400 per-strip size-class map most recently sent for each connector.
    ///
    /// The map describes the whole surface while a delta frame carries only its damaged strips, so
    /// rebuilding it from zero each frame re-declares every untouched position as class 0. See
    /// `video::haar::navarro_strip_params`.
    #[pin]
    strip_classes: Mutex<[KVec<u8>; MAX_CONNECTORS]>,
    /// Per-strip retransmit debt. Spreading repeated updates across frames reaches both of the
    /// dock's scanout buffers; consecutive presentations can target the same buffer.
    #[pin]
    dirty_ttl: Mutex<[Option<KVVec<u8>>; MAX_CONNECTORS]>,
    /// Set once the dock engages the CP cipher (`wsub=0x45` acks > 0); EP08 scanout is gated on it.
    /// Per device, so a second connected dock does not share one dock's engagement state.
    cp_engaged: core::sync::atomic::AtomicBool,
    /// Set once encrypted setup, initial sink discovery, and the platform's pre-mode-set readiness
    /// interval have all completed. KMS producers may coalesce state before this, but no activation
    /// may touch the dock until the bring-up worker publishes this one-way gate.
    kms_activation_ready: core::sync::atomic::AtomicBool,
    /// This device's codec geometry, packed; see [`VinoDrmData::geometry`] and
    /// [`super::video::haar::Geometry`].
    codec_geometry: core::sync::atomic::AtomicU32,
    /// Keyframe, delta and damage-debt presentation counts, packed one byte each; see
    /// [`super::profile::FrameDelivery`].
    frame_delivery: core::sync::atomic::AtomicU32,
    /// Whether a presence retry may reset a bracket beside a live connector; see
    /// [`super::profile::ProbeBracket`].
    probe_bracket: core::sync::atomic::AtomicU8,
    /// Bit `h` set when connector `h`'s committed framebuffer is 10 bits per channel.
    ///
    /// Separate from `codec_geometry` because it is the one part of the codec's configuration that
    /// is neither device-wide nor fixed: the DL7400 negotiates depth per connector, measured on
    /// Windows holding one connector at 8 bits while the other ran at 10.
    connector_ten_bit: core::sync::atomic::AtomicU32,
    /// Bits per channel userspace asked the link to carry, one byte per connector, from `max bpc`.
    ///
    /// Deliberately separate from `connector_ten_bit`: that one says what the framebuffer holds and
    /// decides how a pixel is decoded, this one says what the dock is told to carry.
    connector_max_bpc: core::sync::atomic::AtomicU32,
    /// Connectors whose requested ten-bit link does not fit the dock's shared bandwidth.
    ///
    /// Ten bits costs a third more per pixel, so a pair of modes that fits at eight may not fit at
    /// ten. Refusing the mode is the wrong answer -- a compositor answers `EINVAL` by disabling the
    /// output rather than choosing a shallower link -- so the depth gives way instead and both
    /// connectors light at eight bits.
    connector_deny_ten_bit: core::sync::atomic::AtomicU32,
    /// Bit `h` set when connector `h`'s connector is being driven with the SMPTE ST 2084 (PQ)
    /// transfer function, taken from the `HDR_OUTPUT_METADATA` blob userspace attached to it.
    ///
    /// Deliberately not folded into `connector_ten_bit`: depth and transfer function are two fields
    /// of the dock's set-mode message and two independent decisions by the compositor.
    head_st2084: core::sync::atomic::AtomicU32,
    /// Which protocol generation this dock speaks; see `DockProfile::generation`. The two
    /// platforms differ in their initialisation, per-connector HDCP framing, stream open and mode
    /// description, so one flag drives all of them rather than three that can disagree.
    dock_wide_modeset: core::sync::atomic::AtomicBool,
    clear_mode_before_set: core::sync::atomic::AtomicBool,
    blank_markers_held: core::sync::atomic::AtomicBool,
    video_keepalive: core::sync::atomic::AtomicBool,
    /// Whether the first frame after a mode set carries the cold ARM burst; see
    /// `DockProfile::arm_burst`.
    arm_burst: core::sync::atomic::AtomicBool,
    /// How this dock states its framebuffer allocation; see [`profile::Allocation`].
    allocation: kernel::sync::SetOnce<&'static super::profile::Allocation>,
    /// Whether video records travel on the control bulk-OUT pipe; see
    /// `DockProfile::video_on_ctrl_pipe`.
    video_on_ctrl_pipe: core::sync::atomic::AtomicBool,
    /// The `0x16/0x2e` state that takes a sink down; see `DockProfile::sink_down_state`.
    sink_down_state: core::sync::atomic::AtomicU8,
    post_mode_sink_states: core::sync::atomic::AtomicU16,
    /// `DockProfile::pre_mode_sink_state`, with `u16::MAX` standing for `None`.
    pre_mode_sink_state: core::sync::atomic::AtomicU16,
    /// Heads whose sealed video stream has been opened, as a bitmask; see `set_video_keys`.
    stream_opened: core::sync::atomic::AtomicU32,
    /// Stream reports a connector still owes after its stream was opened; see
    /// `arm_stream_prologue`.
    stream_reports_owed: [core::sync::atomic::AtomicU32; MAX_CONNECTORS],
    /// Consecutive deferrals of the asynchronous KMS batch; see `KMS_RETRY_LIMIT`.
    kms_retries: core::sync::atomic::AtomicU32,
    /// How this dock's video stream describes itself, packed: the layout word in the low sixteen
    /// bits, the stream-marker kind above it, and the code-table form in bit 24. See
    /// `set_video_stream_desc`.
    video_stream_desc: core::sync::atomic::AtomicU32,
    /// Shortest interval between frames on one connector; see `DockProfile::frame_period_ms`.
    frame_period_us: core::sync::atomic::AtomicI64,
    /// Interval between keepalive status queries; see `DockProfile::status_period_ms`.
    status_period_ms: core::sync::atomic::AtomicI64,
    /// Flat carrier frames a connector opens its stream with; see `DockProfile::carrier_frames`.
    carrier_frames: core::sync::atomic::AtomicU32,
    /// Sustained bytes per second this dock accepts; see `DockProfile::stream_pacing`.
    stream_budget_bps: core::sync::atomic::AtomicU32,
    /// Most that may leave back to back after an idle period; the credit ceiling.
    stream_burst_bytes: core::sync::atomic::AtomicU32,
    /// Unspent bytes of that budget, and when they were last topped up.
    ///
    /// One ledger for the whole dock rather than one per connector: what the budget describes is a
    /// decoder behind a single endpoint, and two connectors sharing it spend from the same pool.
    #[pin]
    stream_credit: SpinLock<StreamCredit>,
    /// How many downstream connectors this dock answers a presence probe for; see
    /// `DockProfile::connectors`. Ridge: 2; Navarro: all four physical sockets.
    connectors: core::sync::atomic::AtomicU8,
    /// Excludes the independent keepalive loop while the mode worker emits the mode-relative
    /// activation timeline. Without this, a keepalive poll can win `cp_link` between
    /// two explicitly paced markers and stretch/reorder the sequence.
    cp_timeline_exclusive: core::sync::atomic::AtomicBool,
    /// Navarro's authenticated setup transcript continues directly into the first KMS
    /// transaction: its first runtime message is a pipe clear, not a background status poll.
    /// Hold the keepalive after publishing the session until that transaction has claimed the
    /// control timeline.
    initial_modeset_quiet: core::sync::atomic::AtomicBool,
    /// Mode generation successfully programmed on each dock connector. Scanout must match it
    /// because atomic plane updates can precede the deferred mode-set transaction.
    modeset_active: [core::sync::atomic::AtomicU64; MAX_CONNECTORS],
    /// Exact timing bytes most recently programmed on each connector.
    ///
    /// Kept apart from `modeset_active`: that atomic is the producer's request token, while
    /// `dual_nivo` can be filled only after another connector on the endpoint publishes its own
    /// request. No-op detection compares this exact dock-side state instead of pretending the
    /// request token also describes a send-time topology correction.
    #[pin]
    programmed_timing: SpinLock<[Option<super::cp::Timing>; MAX_CONNECTORS]>,
    /// Latest mode userspace currently requests per connector, encoded like `modeset_active`; zero
    /// means the CRTC is disabled. The deferred worker uses this generation key to discard stale
    /// mode-set commands and framebuffers left by a rapid disable/re-enable sequence.
    modeset_requested: [core::sync::atomic::AtomicU64; MAX_CONNECTORS],
    /// Whether a frame ending on a full packet is split; see
    /// `DockProfile::split_full_packet_frame`.
    split_full_packet_frame: AtomicBool,
    /// Per-connector timestamp of the last accepted frame, used to bound scanout cadence.
    #[pin]
    last_frame: SpinLock<[Option<Instant<Monotonic>>; MAX_CONNECTORS]>,
    /// When `queue_scanout` last ran for each connector, i.e. when KWin's commit tail last handed
    /// us a framebuffer. Distinguishes "the compositor stopped committing" from "we dropped the
    /// frame".
    #[pin]
    /// When the streaming status poll last went out, device-wide. The poll keeps the control
    /// dialogue alive; it does not need to be per presentation.
    #[pin]
    last_status_poll: SpinLock<Option<Instant<Monotonic>>>,
    /// When each connector's scanout work item last began executing.
    #[pin]
    /// Rate limiter for the stall diagnostic below.
    #[pin]
    /// Deadline for the sustained full-frame stream required to train a cold downstream link.
    #[pin]
    sustain_until: SpinLock<[Option<Instant<Monotonic>>; MAX_CONNECTORS]>,
    /// Logical Haar frame sequence per connector.
    #[pin]
    scanout_seq: Mutex<[u32; MAX_CONNECTORS]>,
    /// Persistent pipelined bulk-OUT queue per physical video endpoint. It remains live between
    /// frames.
    ///
    /// The slot is the first connector whose endpoint address matches the caller's (see
    /// [`UsbLink::video_pipe_index`](super::UsbLink::video_pipe_index)); duplicate slots remain
    /// empty. Holding an individual slot mutex over a whole frame serializes connectors that share
    /// a pipe without needlessly serializing independent endpoints.
    #[pin]
    video_q: [Mutex<Option<super::usb::BulkOutQueue>>; MAX_CONNECTORS],
    /// Held by whoever is writing to a pipe that carries both planes; see [`Self::own_pipe`].
    #[pin]
    pipe_writer: Mutex<u8>,
    /// One reusable 64-KiB coalescing window per connector. `frame_records` deliberately stores a
    /// frame as small allocations so encoding never asks kmalloc for multi-megabyte physically
    /// contiguous memory; scanout joins those fragments into this bounded window before
    /// `BulkOutQueue::send` copies it into the persistent DMA ring. Internal record boundaries
    /// remain invisible on USB.
    #[pin]
    video_staging: Mutex<[Option<KVec<u8>>; MAX_CONNECTORS]>,
    /// Last requested timing, retained so scanout can retry a failed mode-set.
    #[pin]
    last_timing: SpinLock<[Option<super::cp::Timing>; MAX_CONNECTORS]>,
    /// Heads whose next video stream must be prefixed with the pipe-arm records.
    arm_prefix_pending: core::sync::atomic::AtomicU32,
    /// Heads for which the read-only endpoint status at the first video stall was logged.
    endpoint_status_logged: core::sync::atomic::AtomicU32,
    /// Connectors still owed the short sealed open that names a stream vino does not drive.
    ///
    /// Held apart from `arm_prefix_pending` because it is the complement of it: a connector vino
    /// is about to send pixels to opens its stream with the pipe descriptor instead, and both DLM
    /// captures send this record only on the stream ids of the connectors left idle. The opens go
    /// out before any connector's first frame, as DLM's do.
    stream_open_pending: core::sync::atomic::AtomicU32,
    /// Per-connector "owes a full keyframe" bitmask (bit `h` = connector `h`). Set (all connectors)
    /// after a `KmsCmd::ModeSet` send: a new mode leaves the dock's framebuffer undefined, so the
    /// first scanout after it must be a FULL frame ([`super::video::haar::colour_frame_ep08`]), not
    /// a damage delta -- otherwise the un-redrawn strips stay garbage. Cleared for a connector once
    /// its keyframe is sent; subsequent flips send only changed strips through
    /// [`super::video::haar::colour_frame_ep08_damage`].
    keyframe_pending: core::sync::atomic::AtomicU32,
    /// Per-connector generation of the dock's cursor bitmap, bumped by [`Self::owe_keyframe`].
    ///
    /// The cursor plane re-uploads only when its bitmap differs from the last one sent, so it needs
    /// to know when the dock stopped holding that bitmap. A mode-set discards it.
    cursor_epoch: [core::sync::atomic::AtomicU32; MAX_CONNECTORS],
    /// Rotates the shadow slot each commit so successive snapshots do not land in the same one.
    shadow_rr: [core::sync::atomic::AtomicU32; MAX_CONNECTORS],
    /// Geometry last announced with `cursor_create`, per connector. Whether the dock keeps one
    /// shared cursor bitmap or one per connector is not established, so each connector announces
    /// and uploads its own -- correct either way, at the cost of one extra upload per shape change.
    #[pin]
    cursor_geometry: Mutex<[Option<(u16, u16)>; MAX_CONNECTORS]>,
    /// Heads whose next activation is a *repair* of a sink the dock dropped underneath us,
    /// rather than a cold bring-up. A repair must not run the cold training window: the link
    /// is already trained, and that window presents full keyframes at [`FRAME_PERIOD_MS`]
    /// for three seconds -- measured at 1.07 GB over 12 seconds across two connectors, which is
    /// the documented way to destabilise this dock.
    repair_connectors: core::sync::atomic::AtomicU32,
    /// The cursor each connector's connector is holding; see [`CursorShot`].
    #[pin]
    cursor_shot: Mutex<[Option<CursorShot>; MAX_CONNECTORS]>,
    /// Dock-wide pixel-rate budget in pixels per second; zero means unknown.
    dock_pixel_budget: core::sync::atomic::AtomicU32,
    /// Highest refresh rate this dock is known to drive; see `DockProfile::max_refresh_hz`.
    max_refresh_hz: core::sync::atomic::AtomicU32,
    /// Highest per-mode pixel clock in kHz; see `DockProfile::max_connector_clock_khz`.
    max_connector_clock_khz: core::sync::atomic::AtomicU32,
    /// Excludes scanout while a mode-set batch can submit on a video endpoint. Paired with
    /// `video_inflight` using sequentially consistent store-then-check handshakes.
    cmd_busy: core::sync::atomic::AtomicBool,
    /// Set around `run_pending_scanout`, allowing `cmd_work` to wait for a
    /// frame already in flight when it set [`Self::cmd_busy`].
    video_inflight: [core::sync::atomic::AtomicBool; MAX_CONNECTORS],
    /// Consecutive failed live-scanout frames per connector, for log rate-limiting.
    scanout_fails: [core::sync::atomic::AtomicU64; MAX_CONNECTORS],
    /// Upcoming page flips to skip for per-connector transport backoff.
    scanout_skip: [core::sync::atomic::AtomicU64; MAX_CONNECTORS],
    /// Settle repaints this connector may still arm. See [`SETTLE_REPAINTS`].
    settle_budget: [core::sync::atomic::AtomicU32; MAX_CONNECTORS],
    /// Last inner status returned for each connector's presence probe.
    presence_reply: [core::sync::atomic::AtomicU32; MAX_CONNECTORS],
    /// Pending downstream-topology notification for this device's keepalive worker.
    downstream_event: AtomicBool,
    /// Head currently expecting an EDID from a re-engage, or [`NO_EDID_TARGET`].
    ///
    /// The EDID arrives as an `id=0x194` push, and during a re-engage it lands in `send_cp`'s
    /// own lockstep drain rather than in `drain_cp_pushes`. This says "somebody is waiting for
    /// one", so that drain can stash it instead of discarding it.
    edid_target: core::sync::atomic::AtomicU32,
    /// The blob that drain caught, handed back to [`VinoDrmData::reengage_connector`].
    #[pin]
    edid_caught: Mutex<Option<KVec<u8>>>,
    /// Heads intentionally blanked by Vino. Their expected probe silence is not a hot-unplug.
    self_blanked: core::sync::atomic::AtomicU32,
    /// Heads whose blank bracket is still open on the dock, one bit each.
    ///
    /// Distinct from [`Self::self_blanked`], which `atomic_enable` clears on the commit thread
    /// before the command worker runs; by then the wake choreography would no longer know a blank
    /// was owed a close. A bracket left open keeps the sink dark through the next mode set.
    blank_bracket_open: core::sync::atomic::AtomicU32,
    /// Whether this dock's video pipeline can carry ten bits per channel, from its profile.
    hdr_capable: AtomicBool,
    /// Whether this dock composites a cursor bitmap of its own; see [`DockProfile::hw_cursor`].
    hw_cursor: AtomicBool,
    /// Whether the dock's presence probe describes a connector; see
    /// [`DockProfile::reports_presence`].
    reports_presence: AtomicBool,
    /// Whether the connectors share one EDID handler; see [`DockProfile::shared_edid_handler`].
    shared_edid_handler: AtomicBool,
    /// Per-connector key and nonce used to seal pipe-arm records.
    #[pin]
    video_keys: Mutex<[kernel::crypto::Secret<32>; MAX_CONNECTORS]>,
    /// When each connector last put a byte on its video endpoint.
    ///
    /// Drives the DL7400 keep-alive: see [`NAVARRO_VIDEO_QUIET_MS`] for why a connector that has
    /// nothing to draw still has to say something.
    #[pin]
    last_video_at: SpinLock<[Option<Instant<Monotonic>>; MAX_CONNECTORS]>,
    /// Per-connector AES-CTR block counter for the sealed records on that connector's video stream.
    ///
    /// Every sealed video record carries this counter in its wire `seq`, and `seal_livemac` uses
    /// it both as the CTR block index and as the Dl3Cmac counter. It is stream state, not record
    /// state: DLM advances it by `ceil(plaintext / 16)` for every sealed record it sends on a
    /// stream and never rewinds it, so a re-arm continues the count rather than restarting. It is
    /// reset only when new video keys arrive, because a fresh key is a fresh keystream.
    video_seal_seq: [core::sync::atomic::AtomicU32; MAX_CONNECTORS],
}

impl VinoDrmData {
    /// `hdr_capable`, `hw_cursor` and `connectors` come from the dock's profile and must be
    /// supplied here rather than stored afterwards: `create_objects` runs inside
    /// `drm::Registration::new_static`, which is *before* probe reaches the block that publishes
    /// the rest of the profile. Set late, they were always false while the connectors and planes
    /// were being built, so the ten-bit format and the three HDR properties were silently never
    /// attached, and no dock could ever withhold its cursor plane.
    ///
    /// `connectors` decides how many connectors exist at all. A dock that advertises more
    /// connectors than it has sockets offers userspace outputs that can never carry a monitor, and
    /// a compositor that enables one makes the driver encode and transmit full frames to nothing.
    pub(super) fn new(
        io: Arc<super::usb::IoWindow>,
        endpoints: super::Endpoints,
        hdr_capable: bool,
        hw_cursor: bool,
        connectors: u8,
    ) -> impl PinInit<Self, Error> {
        try_pin_init!(Self {
            io,
            endpoints,
            shutting_down: AtomicBool::new(false),
            cp_link <- new_mutex!(Option::<CpLink>::None),
            cp_last_reply <- new_spinlock!(Instant::<Monotonic>::now()),
            cp_session_live: AtomicBool::new(false),
            cp_reset_queued: AtomicBool::new(false),
            cp_watchdog <- new_delayed_work!("vino::cp_watchdog"),
            pending_kms <- new_mutex!(PendingKms::new()),
            pending_scanout <- new_mutex!([const { None }; MAX_CONNECTORS]),
            settle_repaint <- new_mutex!([const { None }; MAX_CONNECTORS]),
            shadow <- pin_init::pin_init_array_from_fn(|_| new_mutex!(ShadowPool::new())),
            vblank <- new_spinlock!([const { None }; MAX_CONNECTORS]),
            cmd_work <- new_delayed_work!("vino::kms_cmd"),
            scanout_work_h0 <- new_work!("vino::scanout_h0"),
            scanout_work_h1 <- new_work!("vino::scanout_h1"),
            scanout_work_h2 <- new_work!("vino::scanout_h2"),
            scanout_work_h3 <- new_work!("vino::scanout_h3"),
            session_queue: workqueue::Queue::new_ordered().build(kernel::c_str!("vino_session"))?,
            // High priority: this queue carries cursor movement, and its work items are a few
            // small control messages. Left at default priority they queue behind whatever else
            // the machine is doing, and the pointer visibly stutters under load.
            kms_queue: workqueue::Queue::new_ordered()
                .highpri()
                .build(kernel::c_str!("vino_kms"))?,
            scanout_queue: workqueue::Queue::new_unbound()
                .max_active(MAX_CONNECTORS as u32)
                .build(kernel::c_str!("vino_scanout"))?,
            cached_edids <- new_mutex!([const { None }; MAX_CONNECTORS]),
            connectors_present: core::sync::atomic::AtomicU32::new(0),
            color <- new_mutex!([None; MAX_CONNECTORS]),
            strip_hashes <- new_mutex!([const { None }; MAX_CONNECTORS]),
            strip_classes <- new_mutex!(core::array::from_fn(|_| KVec::new())),
            dirty_ttl <- new_mutex!([const { None }; MAX_CONNECTORS]),
            cp_engaged: core::sync::atomic::AtomicBool::new(false),
            kms_activation_ready: core::sync::atomic::AtomicBool::new(false),
            cp_timeline_exclusive: core::sync::atomic::AtomicBool::new(false),
            initial_modeset_quiet: core::sync::atomic::AtomicBool::new(false),
            modeset_active: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            programmed_timing <- new_spinlock!([None; MAX_CONNECTORS]),
            modeset_requested: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            split_full_packet_frame: AtomicBool::new(false),
            last_frame <- new_spinlock!([const { None }; MAX_CONNECTORS]),
            last_status_poll <- new_spinlock!(None),
            sustain_until <- new_spinlock!([const { None }; MAX_CONNECTORS]),
            scanout_seq <- new_mutex!([0; MAX_CONNECTORS]),
            video_q <- pin_init::pin_init_array_from_fn(|_| new_mutex!(None)),
            pipe_writer <- new_mutex!(0u8),
            video_staging <- new_mutex!([const { None }; MAX_CONNECTORS]),
            last_timing <- new_spinlock!([None; MAX_CONNECTORS]),
            arm_prefix_pending: core::sync::atomic::AtomicU32::new(0),
            endpoint_status_logged: core::sync::atomic::AtomicU32::new(0),
            stream_open_pending: core::sync::atomic::AtomicU32::new(0),
            keyframe_pending: core::sync::atomic::AtomicU32::new(0),
            cursor_epoch: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            shadow_rr: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            cursor_geometry <- new_mutex!([None; MAX_CONNECTORS]),
            repair_connectors: core::sync::atomic::AtomicU32::new(0),
            cursor_shot <- new_mutex!([const { None }; MAX_CONNECTORS]),
            // D6000 default: 442,368,000 px/s (one 1440p@120) x2 compression headroom = dual
            // 1440p@120. Replace it if a dock capability supplies a limit.
            dock_pixel_budget: core::sync::atomic::AtomicU32::new(884_736_000),
            max_refresh_hz: core::sync::atomic::AtomicU32::new(DEFAULT_MAX_REFRESH_HZ),
            max_connector_clock_khz: core::sync::atomic::AtomicU32::new(DEFAULT_MAX_HEAD_CLOCK_KHZ),
            cmd_busy: core::sync::atomic::AtomicBool::new(false),
            video_inflight: core::array::from_fn(|_| core::sync::atomic::AtomicBool::new(false)),
            scanout_fails: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            scanout_skip: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            settle_budget: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            presence_reply: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            downstream_event: AtomicBool::new(false),
            edid_target: core::sync::atomic::AtomicU32::new(NO_EDID_TARGET),
            edid_caught <- new_mutex!(None),
            self_blanked: core::sync::atomic::AtomicU32::new(0),
            blank_bracket_open: core::sync::atomic::AtomicU32::new(0),
            hdr_capable: AtomicBool::new(hdr_capable),
            hw_cursor: AtomicBool::new(hw_cursor),
            reports_presence: AtomicBool::new(true),
            shared_edid_handler: AtomicBool::new(false),
            codec_geometry: core::sync::atomic::AtomicU32::new(0),
            // Ridge-compatible defaults until probe publishes the matched profile.
            frame_delivery: core::sync::atomic::AtomicU32::new(2 | (1 << 8) | (3 << 16)),
            probe_bracket: core::sync::atomic::AtomicU8::new(
                super::profile::ProbeBracket::Always as u8
            ),
            connector_ten_bit: core::sync::atomic::AtomicU32::new(0),
            connector_max_bpc: core::sync::atomic::AtomicU32::new(0),
            connector_deny_ten_bit: core::sync::atomic::AtomicU32::new(0),
            head_st2084: core::sync::atomic::AtomicU32::new(0),
            // A dock that names no connector count still has to expose something, so fall back to
            // the maximum rather than building a card with no connectors at all.
            connectors: core::sync::atomic::AtomicU8::new(if connectors == 0 {
                MAX_CONNECTORS as u8
            } else {
                connectors.min(MAX_CONNECTORS as u8)
            }),
            dock_wide_modeset: core::sync::atomic::AtomicBool::new(false),
            clear_mode_before_set: core::sync::atomic::AtomicBool::new(false),
            blank_markers_held: core::sync::atomic::AtomicBool::new(false),
            video_keepalive: core::sync::atomic::AtomicBool::new(false),
            arm_burst: core::sync::atomic::AtomicBool::new(true),
            allocation: kernel::sync::SetOnce::new(),
            video_on_ctrl_pipe: core::sync::atomic::AtomicBool::new(false),
            sink_down_state: core::sync::atomic::AtomicU8::new(BLANK_MARKER_STATE),
            post_mode_sink_states: core::sync::atomic::AtomicU16::new(0x0303),
            pre_mode_sink_state: core::sync::atomic::AtomicU16::new(u16::MAX),
            stream_opened: core::sync::atomic::AtomicU32::new(0),
            stream_reports_owed: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            kms_retries: core::sync::atomic::AtomicU32::new(0),
            video_stream_desc: core::sync::atomic::AtomicU32::new(0),
            frame_period_us: core::sync::atomic::AtomicI64::new(FRAME_PERIOD_US),
            status_period_ms: core::sync::atomic::AtomicI64::new(STATUS_PERIOD_MS),
            carrier_frames: core::sync::atomic::AtomicU32::new(u32::MAX),
            stream_budget_bps: core::sync::atomic::AtomicU32::new(u32::MAX),
            stream_burst_bytes: core::sync::atomic::AtomicU32::new(u32::MAX),
            stream_credit <- new_spinlock!(StreamCredit::new()),
            video_keys <- new_mutex!(core::array::from_fn(
                |_| kernel::crypto::Secret::zeroed()
            )),
            last_video_at <- new_spinlock!([None; MAX_CONNECTORS]),
            video_seal_seq: core::array::from_fn(
                |_| core::sync::atomic::AtomicU32::new(0)
            ),
        })
    }

    /// Publish the producers' stop flag and nothing else.
    ///
    /// `disconnect()` calls this *before* `IoWindow::close()`. The scanout and command workers each
    /// hold an `Io` token for as long as they loop and re-read `shutting_down` every iteration, so
    /// setting it early is what keeps them from holding `close()`'s wait open. Everything in
    /// [`shutdown`](Self::shutdown) proper must wait until USB I/O is quiesced; this must not.
    pub(super) fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        self.kms_activation_ready.store(false, Ordering::Release);
        self.cp_timeline_exclusive.store(false, Ordering::Release);
        self.initial_modeset_quiet.store(false, Ordering::Release);
    }

    /// Stand every producer down because the device is about to be reset.
    ///
    /// A reset takes the whole session with it: the dock forgets its content-protection keys, its
    /// open streams and the sinks it was driving, and nothing this driver holds describes the
    /// device on the other side of one. So the link is marked gone before the reset rather than
    /// after, which is what stops a worker submitting a transfer across it.
    pub(super) fn stop_for_reset(&self) {
        self.cp_session_live.store(false, Ordering::Release);
        self.begin_shutdown();
    }

    /// Whether the parent interface is being removed.
    pub(super) fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Queue used by the session bring-up and keepalive work item.
    pub(super) fn session_queue(&self) -> &workqueue::Queue {
        &self.session_queue
    }

    /// Take this device's pending downstream-topology notification.
    pub(super) fn take_downstream_event(&self) -> bool {
        self.downstream_event.swap(false, Ordering::Acquire)
    }

    /// Stop deferred DRM work while the parent USB interface is still bound. `cmd_work` is
    /// embedded in this DRM device and each successful enqueue temporarily owns an
    /// `ARef<VinoDrmDevice>`; pending scanouts also retain compositor framebuffers. Quiesce both
    /// producers, reclaim any queued work pointer, and drop those framebuffers before the final
    /// device references disappear during devres teardown.
    pub(super) fn shutdown(&self) {
        // Idempotent, and `disconnect()` has normally already done this: see `begin_shutdown`.
        self.begin_shutdown();
        for mode in &self.modeset_requested {
            mode.store(0, Ordering::Release);
        }

        // Stop the software vblank clocks before releasing their CRTC references. Take the
        // registry out from under the spinlock before dropping the handles:
        // Dropping `ArcHrTimerHandle` waits for a running callback and must
        // not happen in atomic context.
        let timers = {
            let mut slots = self.vblank.lock();
            core::mem::replace(&mut *slots, [const { None }; MAX_CONNECTORS])
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

        // Break the two device-to-itself reference cycles. Both run through a `crtc::CrtcRef`,
        // which owns an `ARef<VinoDrmDevice>`:
        //
        //   1. `VblankTimer::crtc`, published by the first `enable_vblank` and never released. The
        //      timer is owned by `VinoCrtc`, which lives inside the DRM device allocation.
        //   2. `VinoCrtc::vblank_pinned`, the driver-held vblank reference. Teardown cannot rely on
        //      `atomic_disable` running before unplug.
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
                // The software vblank clock has just stopped, and a page flip armed by
                // `atomic_flush` is waiting on a tick that will never come. `drm_crtc_vblank_off()`
                // both refuses further vblank references -- so `drm_atomic_helper_wait_for_vblanks`
                // skips this CRTC instead of warning -- and sends every event still queued on the
                // device's vblank list, which is exactly where `PendingVblankEvent::arm` put ours.
                //
                // Without it an unplug left the compositor's `commit_tail` blocked until DRM's own
                // deadlines expired: this boot logged 73 `vblank wait timed out` warnings and 110
                // pairs of `flip_done timed out` / `commit wait timed out`, ten seconds each, on
                // top of every dock reset. That is most of the delay between a dock coming back and
                // pixels reappearing.
                crtc_ref.crtc().vblank_off();
                let crtc: &VinoCrtc = crtc_ref.crtc();
                drop(crtc.vblank_pinned.lock().take());
                drop(crtc_ref);
            }
        }
        drop(timers);

        *self.pending_kms.lock() = PendingKms::new();
        *self.pending_scanout.lock() = [const { None }; MAX_CONNECTORS];
        *self.settle_repaint.lock() = [const { None }; MAX_CONNECTORS];
        for h in 0..MAX_CONNECTORS {
            self.shadow[h].lock().discard();
        }
        *self.strip_hashes.lock() = [const { None }; MAX_CONNECTORS];
        *self.dirty_ttl.lock() = [const { None }; MAX_CONNECTORS];
        // Cancel the queued drain and reclaim the `ARef<VinoDrmDevice>` the enqueue handed to
        // the workqueue, if it was still pending. Dropping it here releases the self-reference
        // that would otherwise keep this device alive until the work ran.
        //
        // Cancel `cmd_work` first because it can enqueue both scanout workers. `shutting_down` is
        // already visible to all workers, so cancellation only waits for work already in flight.
        drop(self.cmd_work.cancel_sync());
        drop(self.cp_watchdog.cancel_sync());
        drop(self.scanout_work_h0.cancel_sync());
        drop(self.scanout_work_h1.cancel_sync());
        drop(self.scanout_work_h2.cancel_sync());
        drop(self.scanout_work_h3.cancel_sync());

        // A running callback may have taken a batch just before shutdown was published. It has
        // finished now; clear anything it left behind and tear the USB queues down while their
        // parent interface is still in Bound context.
        *self.pending_kms.lock() = PendingKms::new();
        *self.pending_scanout.lock() = [const { None }; MAX_CONNECTORS];
        *self.settle_repaint.lock() = [const { None }; MAX_CONNECTORS];
        for h in 0..MAX_CONNECTORS {
            self.shadow[h].lock().discard();
        }
        *self.strip_hashes.lock() = [const { None }; MAX_CONNECTORS];
        *self.dirty_ttl.lock() = [const { None }; MAX_CONNECTORS];
        for queue in &self.video_q {
            *queue.lock() = None;
        }
        *self.video_staging.lock() = [const { None }; MAX_CONNECTORS];
        self.cp_session_live.store(false, Ordering::Release);
        *self.cp_link.lock() = None;
        vino_debug!("vino: deferred KMS/video work drained for unplug\n");
    }

    /// Cache `connector`'s CRTC colour transform (from `RawCrtcState::gamma_lut` and
    /// `RawCrtcState::ctm`) for the scanout to apply, or clear it to identity with two `None`s.
    pub(super) fn update_color(
        &self,
        connector: usize,
        lut: Option<&[crtc::ColorLut]>,
        ctm: Option<&crtc::ColorCtm>,
    ) {
        let socket = connector + 1;
        let cached = super::color::ColorPipeline::build(lut, ctm);
        let changed = if let Some(slot) = self.color.lock().get_mut(connector) {
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
            if cached.is_some() {
                vino_debug!("vino: socket {socket} colour transform updated\n");
            } else {
                vino_debug!("vino: socket {socket} colour transform cleared\n");
            }
            // The encoded-strip cache keys on a strip's source pixels, so a transform change that
            // leaves those pixels untouched would otherwise re-send stale bodies for the whole
            // screen. Drop the cache and owe a keyframe.
            self.strip_hashes.lock()[connector] = None;
            self.dirty_ttl.lock()[connector] = None;
            self.owe_keyframe(connector);
        }
    }

    /// Snapshot `connector`'s cached colour transform for a scanout pass (`Copy`, so no lock is
    /// held afterwards).
    pub(super) fn color_snapshot(&self, connector: usize) -> Option<super::color::ColorPipeline> {
        self.color.lock().get(connector).copied().flatten()
    }

    /// Number of physical connectors selected by the matched dock profile.
    pub(super) fn connector_count(&self) -> usize {
        usize::from(self.connectors.load(Ordering::Acquire)).min(MAX_CONNECTORS)
    }

    /// Whether `connector` represents a distinct runtime stream on this dock.
    ///
    /// Every physical connector does, including both halves of a shared endpoint: the DL7400 maps
    /// its four connectors in pairs onto two video bulk endpoints, and treating the second of each
    /// pair as an alias would make a monitor in socket 3 or 4 invisible. Sharing an endpoint is a
    /// transport detail, handled where it belongs by [`UsbLink::video_pipe_index`], which gives
    /// both connectors of a pair the same persistent queue.
    ///
    /// Empty sockets cost nothing here: the presence probe answers negative for them, and the
    /// keepalive's re-engage retry stands down permanently once it has.
    pub(super) fn runtime_connector(&self, connector: usize) -> bool {
        connector < self.connector_count()
    }

    /// Whether bring-up has reached the point where KMS may touch the dock.
    pub(super) fn kms_activation_ready(&self) -> bool {
        self.kms_activation_ready.load(Ordering::Acquire)
    }

    /// Take every connector down once the control session has been abandoned.
    ///
    /// Userspace can move its windows off a connector that has disappeared, but not off one that
    /// is merely frozen. Recovery is a replug, which rebinds and starts a fresh session.
    pub(super) fn drop_connectors_with_session(&self, drm_dev: &VinoDrmDevice) {
        let mut dropped = false;
        for connector in 0..self.connector_count() {
            if !self.runtime_connector(connector) || !self.connector_present(connector) {
                continue;
            }
            self.set_disconnected(connector);
            dropped = true;
            pr_warn!(
                "vino: socket {socket} dropped with the control session\n",
                socket = connector + 1
            );
        }
        if dropped {
            drm_dev.hotplug_event();
        }
    }

    /// Take the shared pipe for one indivisible sequence of writes.
    ///
    /// A record is never split: the vendor's control records sit between records, never inside
    /// one. On a dock with a video pipe of its own that is free -- the two planes cannot collide.
    /// Here they share an endpoint, and a control write submitted between two of a frame's URBs
    /// lands in the middle of an image record, where it desynchronises the dock's parser for the
    /// rest of the frame. The dock accepts every byte and shows nothing, which is indistinguishable
    /// from a dead sink.
    ///
    /// Returns `None` on a dock whose planes have separate endpoints, where the exclusion would
    /// only cost the control plane latency.
    ///
    /// Lock order is `cp_link` then this then `video_q`. Nothing may send a control message while
    /// holding it.
    pub(super) fn own_pipe(
        &self,
    ) -> Option<kernel::sync::lock::Guard<'_, u8, kernel::sync::lock::mutex::MutexBackend>> {
        self.video_on_ctrl_pipe().then(|| self.pipe_writer.lock())
    }

    /// Retire every outstanding URB after a physical video queue reports an error.
    ///
    /// The caller must still own this endpoint's `own_pipe()` guard (when present) and its
    /// canonical `video_q` slot.  Taking and explicitly dropping the complete queue synchronously
    /// kills every submitted URB before a stalled endpoint is cleared.  In particular, no queued
    /// frame tail may resume after `usb_clear_halt()` without the prefix that made it parseable.
    /// Keeping the canonical slot empty also makes the next writer create a fresh queue on a
    /// dedicated video endpoint.
    ///
    /// A shared control/video pipe cannot resume locally.  Submission advances the host's frame
    /// and ring counters before asynchronous URBs complete, so cancellation can leave the dock
    /// expecting an earlier frame than Vino.  There is no independent endpoint to re-arm and no
    /// safe counter-only rewind; abandon the complete session and let USB reset establish a new
    /// one instead.  Dedicated video endpoints retain their local drain/clear recovery.
    pub(super) fn retire_failed_video_queue(
        &self,
        dev: &BoundInterface<'_>,
        connector: usize,
        queue_slot: &mut Option<super::usb::BulkOutQueue>,
        cause: Error,
        clear_halt: bool,
    ) -> Result {
        let doomed = queue_slot.take();
        drop(doomed);

        vino_debug!(
            "vino: connector={} retired failed physical video queue ({:?})\n",
            connector,
            cause
        );
        if self.video_on_ctrl_pipe()
            && (cause == kernel::error::code::EPIPE
                || cause == kernel::error::code::EPROTO
                || cause == kernel::error::code::ETIMEDOUT)
        {
            // Publish the terminal state before invalidating connectors or requesting reset.  A CP
            // writer already queued behind `own_pipe()` rechecks this flag after it acquires the
            // pipe, and a scanout writer may not recreate the canonical queue once it is false.
            if self
                .cp_session_live
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                pr_warn!(
                    "vino: shared video/control pipe failed ({:?}); abandoning the session\n",
                    cause
                );
            }
            // This device instance is terminal even though USB reset is asynchronous.  Close the
            // producer gate as well as the transport gate so KMS callbacks may coalesce their
            // latest state, but cannot enqueue another activation in the reset window.  The fresh
            // probe owns a new VinoDrmData and publishes its own readiness after setup completes.
            self.kms_activation_ready.store(false, Ordering::Release);
            let mut programmed = self.programmed_timing.lock();
            for h in 0..self.connector_count() {
                self.modeset_active[h].store(0, Ordering::Release);
                programmed[h] = None;
            }
            drop(programmed);
            self.reset_after_wedge();
            return Ok(());
        }
        if clear_halt
            && (cause == kernel::error::code::EPIPE || cause == kernel::error::code::EPROTO)
        {
            dev.clear_video_halt(connector)?;
            pr_info!(
                "vino: connector {} video queue drained and endpoint halt cleared\n",
                connector
            );
        }
        Ok(())
    }

    /// Keep Navarro's setup-to-first-mode-set control stream free of background traffic.
    pub(super) fn hold_cp_for_initial_modeset(&self) {
        self.initial_modeset_quiet.store(true, Ordering::Release);
    }

    /// Whether the initial Navarro mode set still owns the next control message.
    pub(super) fn initial_modeset_quiet(&self) -> bool {
        self.initial_modeset_quiet.load(Ordering::Acquire)
    }

    /// Release the initial hold if userspace never submits a mode set.
    pub(super) fn release_initial_modeset_quiet(&self) {
        self.initial_modeset_quiet.store(false, Ordering::Release);
    }

    /// Store the per-connector video keys produced by the `id=0x32` exchange.
    ///
    /// Called with [`publish_session`](Self::publish_session) when CP engages. `opened` is the
    /// connector bitmask whose streams the setup burst opened; each of those consumed block zero of
    /// its stream, so its chain continues at block one. A connector not in the mask had no sink at
    /// setup time and still owes its open, so its chain must start at zero -- sealing at one leaves
    /// a gap the dock accounts for as a keystream it never received, and it discards the record
    /// with nothing on the wire to say so.
    pub(super) fn set_video_keys(
        &self,
        keys: [kernel::crypto::Secret<32>; MAX_CONNECTORS],
        opened: u32,
    ) {
        *self.video_keys.lock() = keys;
        self.stream_opened.store(opened, Ordering::Release);
        // A new key is a new keystream, so the block counters start over with it.
        for (connector, seq) in self.video_seal_seq.iter().enumerate() {
            let used = u32::from(opened & (1u32 << connector) != 0);
            seq.store(used, Ordering::Release);
        }
    }

    /// Make a connector owe the records that open its stream, ahead of its next frame.
    ///
    /// On a dock that shares its control pipe the prologue restarts the stream, and the dock's
    /// frame counter with it: DLM's next opener names ring slot 0 and frame 1 whatever the
    /// connector had reached before. Carrying the old count over hands the dock a slot it is still
    /// scanning out. A dock with a video pipe of its own has no such restart, and keeps counting.
    fn arm_stream_prologue(&self, connector: usize) {
        self.arm_prefix_pending
            .fetch_or(1u32 << connector, Ordering::Release);
        if self.video_on_ctrl_pipe() && connector < MAX_CONNECTORS {
            self.scanout_seq.lock()[connector] = 0;
            self.stream_reports_owed[connector].store(STREAM_REPORT_BURST, Ordering::Release);
        }
    }

    /// Reserve `blocks` AES-CTR blocks on a connector's video stream and return the counter to seal
    /// at.
    ///
    /// Sealed video records must tile the stream's keystream without gaps or overlaps: the dock
    /// tracks the same counter, and a record that repeats a block a previous record already used is
    /// a replay of that keystream. Reserving before sealing keeps that true no matter how the
    /// records are grouped into transfers.
    fn take_seal_seq(&self, connector: usize, blocks: u32) -> u32 {
        self.video_seal_seq[connector].fetch_add(blocks, Ordering::AcqRel)
    }

    /// Status polls issued immediately before the first video presentation of a mode generation.
    ///
    /// Captured sequences interleave two status messages here and begin video while the stream
    /// bracket is still active. The longer downstream training interval follows the bracket.
    const PREWRITE_POLLS: u32 = 2;
    const PREWRITE_POLL_MS: u64 = 1;
    /// Send one `id=0x14 sub=0x000c` device-status poll.
    fn poll_status(&self, dev: &BoundInterface<'_>) -> Result {
        self.send_cp(dev, 0x14, 0, |ctr| super::cp::device_query_req(ctr, 0x000c))
    }

    /// One `id=0x16 sub=0x2e|0x2f` stream/display marker. State lives in byte 23, not byte 22
    /// (byte 22 is constantly `1` -- reading it makes every marker look like state=1).
    fn stream_marker(&self, dev: &BoundInterface<'_>, connector: u8, sub: u16, st: u8) -> Result {
        self.send_cp(dev, 0x16, 0, |ctr| {
            super::cp::stream_marker(ctr, connector, sub, st)
        })
    }

    /// Send one captured Navarro sink-reset operation.
    fn navarro_cold_op(&self, dev: &BoundInterface<'_>, op: NavarroColdOp) -> Result {
        match op {
            NavarroColdOp::Poll => self.poll_status(dev),
            NavarroColdOp::EdidState(connector, state) => self.send_cp(dev, 0x16, 0, |ctr| {
                super::cp::edid_readiness_state(ctr, connector, state)
            }),
            NavarroColdOp::Probe(connector) => self.send_cp(dev, 0x15, 0, |ctr| {
                super::cp::get_edid_req_sub(ctr, 0x20, connector)
            }),
            NavarroColdOp::Fetch(connector) => {
                self.send_cp(dev, 0x15, 0, |ctr| super::cp::get_edid_req(ctr, connector))
            }
            NavarroColdOp::SinkTeardown(connector) => self.send_cp(dev, 0x16, 0, |ctr| {
                super::cp::edid_sink_state(ctr, connector, 0xff)
            }),
            NavarroColdOp::Engage(connector) => self.send_cp(dev, 0x16, 0, |ctr| {
                super::cp::edid_engage_req(ctr, connector)
            }),
            NavarroColdOp::PostEdid(connector) => self.send_cp(dev, 0x15, 0, |ctr| {
                super::cp::post_edid_query(ctr, connector)
            }),
            NavarroColdOp::Clear(connector) => {
                self.send_cp(dev, 0x48, 0, |ctr| super::cp::clear_mode(ctr, connector))
            }
        }
    }

    /// Whether another connector on `connector`'s video endpoint is also being driven.
    ///
    /// `0x08` owns connectors {0, 2} and `0x0a` owns {1, 3}, so a connector's partner is the one
    /// two away. Such a pair must declare `Dual NIVO` in both mode sets or the dock drives only one
    /// of the two streams it is sent, however correctly they are tagged.
    pub(super) fn endpoint_is_shared(&self, connector: usize) -> bool {
        self.endpoint_is_shared_in_mask(connector, self.requested_connector_mask())
    }

    /// Requested connectors as one topology snapshot.
    fn requested_connector_mask(&self) -> u32 {
        self.modeset_requested
            .iter()
            .enumerate()
            .fold(0u32, |mask, (connector, requested)| {
                mask | (u32::from(requested.load(Ordering::Acquire) != 0) << connector)
            })
    }

    /// Whether `connector` shares its endpoint with another requested connector in `mask`.
    fn endpoint_is_shared_in_mask(&self, connector: usize, mask: u32) -> bool {
        if connector >= MAX_CONNECTORS || self.connector_count() <= 2 {
            return false;
        }
        let partner = connector ^ 2;
        partner < MAX_CONNECTORS && mask & (1u32 << partner) != 0
    }
}

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
#[kunit_tests(vino_sink)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn only_a_presentation_that_names_a_ring_slot_advances_the_frame_counter() -> Result {
        // The frame counter belongs to the ring, and every generation names the ring in the
        // record that closes a frame. A presentation carrying neither an opener nor a trailer says
        // nothing about the ring and must not consume a slot, or every later record names a buffer
        // one ahead of the one the host filled.
        assert!(!names_ring_slot(&[], &video::haar::FrameTrailer::none()));
        let ella = profile::PROFILE_ELLA.geometry();
        assert!(names_ring_slot(
            &[],
            &video::haar::FrameTrailer::one(&video::haar::ella_frame_close(ella, 0, 0))
        ));

        // Both other generations close every frame, so every presentation advances the counter and
        // this rule leaves them exactly as they were.
        let ridge = profile::PROFILE_RIDGE.geometry();
        assert!(names_ring_slot(
            &[],
            &video::haar::frame_trailer(ridge, 0, 0)
        ));
        let navarro = profile::PROFILE_NAVARRO.geometry();
        assert!(names_ring_slot(
            &[],
            &video::haar::navarro_frame_trailer(navarro, 0, 0)
        ));
        Ok(())
    }

    /// The DL7400 parameter map goes among a frame's records, not after all of them.
    ///
    /// The dock reads the records around the map with what the map says, and takes a frame that
    /// carries it after every record it describes twice before it stops draining the endpoint
    /// altogether. The split lands on a chunk because that is a record boundary; the vendor's byte
    /// offset on its own is wherever a frame's record lengths put it.
    #[test]
    fn param_map_lands_among_a_frame_s_records() -> Result {
        let chunk = |len: usize| -> Result<KVec<u8>> {
            let mut c = KVec::new();
            c.resize(len, 0, GFP_KERNEL)?;
            Ok(c)
        };

        // A frame of even chunks: the split is the last chunk that fits under the vendor's offset,
        // and leaves the rest of the frame behind the map.
        let mut even: KVec<KVec<u8>> = KVec::new();
        for _ in 0..27 {
            even.push(chunk(16_000)?, GFP_KERNEL)?;
        }
        let split = param_map_chunk_split(&even);
        assert_eq!(split, 7);
        assert!(split * 16_000 <= NAVARRO_PARAM_IMAGE_OFFSET);
        assert!((split + 1) * 16_000 > NAVARRO_PARAM_IMAGE_OFFSET);

        // A frame smaller than the offset still puts records in front of the map, and never names
        // a chunk it does not have.
        let mut small: KVec<KVec<u8>> = KVec::new();
        small.push(chunk(4_000)?, GFP_KERNEL)?;
        small.push(chunk(4_000)?, GFP_KERNEL)?;
        assert_eq!(param_map_chunk_split(&small), 2);

        // A single chunk larger than the offset cannot be split, and the map goes behind it rather
        // than in front of every record in the frame.
        let mut one: KVec<KVec<u8>> = KVec::new();
        one.push(chunk(NAVARRO_PARAM_IMAGE_OFFSET * 2)?, GFP_KERNEL)?;
        assert_eq!(param_map_chunk_split(&one), 1);
        Ok(())
    }

    #[test]
    fn frame_delivery_is_profile_data_not_ring_geometry() {
        let ridge = profile::PROFILE_RIDGE.protocol.frame_delivery;
        let navarro = profile::PROFILE_NAVARRO.protocol.frame_delivery;
        let ella = profile::PROFILE_ELLA.protocol.frame_delivery;

        // Preserve both established dedicated-pipe families exactly.
        assert_eq!(ridge.keyframe_presentations, 2);
        assert_eq!(ridge.delta_presentations, 1);
        assert_eq!(ridge.damage_frames, 3);
        assert_eq!(navarro.keyframe_presentations, 3);
        assert_eq!(navarro.delta_presentations, 1);
        assert_eq!(navarro.damage_frames, 4);
        for (policy, keys, deltas) in [(ridge, 2, 1), (navarro, 3, 1)] {
            assert_eq!(frame_presentation_count(policy, true, false, false), keys);
            assert_eq!(
                frame_presentation_count(policy, false, false, false),
                deltas
            );
        }

        // Ella still initialises all three buffers, but DLM carries one ordinary presentation per
        // logical frame. Later debt frames walk the ring without multiplying each update in place.
        assert_eq!(ella.keyframe_presentations, 3);
        assert_eq!(ella.delta_presentations, 1);
        assert_eq!(ella.damage_frames, 3);
        assert_eq!(frame_presentation_count(ella, true, false, true), 3);
        assert_eq!(frame_presentation_count(ella, false, false, true), 1);

        // Dedicated endpoints retain their bounded cold-training burst. A shared control pipe
        // uses its profile keyframe count instead, so it never receives eight multi-megabyte
        // copies back to back.
        assert_eq!(
            frame_presentation_count(ridge, true, true, false),
            drm_sink::COLD_TRAINING_PRESENTATIONS
        );
        assert_eq!(frame_presentation_count(ridge, false, true, false), 1);
        assert_eq!(frame_presentation_count(ella, true, true, true), 3);

        // `damage_frames` includes the first accepted submission. Ella therefore leaves exactly
        // two scheduled debt submissions after the changed frame, covering slots 0, 1 and 2 once.
        let mut debt = [ella.damage_frames, 1, 0];
        pay_damage_debt(&mut debt, false);
        assert_eq!(debt, [2, 0, 0]);
        pay_damage_debt(&mut debt, false);
        assert_eq!(debt, [1, 0, 0]);
        pay_damage_debt(&mut debt, false);
        assert_eq!(debt, [0, 0, 0]);
        let mut full = [3, 2, 1];
        pay_damage_debt(&mut full, true);
        assert_eq!(full, [0, 0, 0]);
    }

    /// The pacing envelope is the vendor's, and only a shared-pipe dock declares one.
    #[test]
    fn stream_pacing_is_the_vendors_envelope_on_the_shared_pipe_dock_only() {
        assert!(!profile::PROFILE_RIDGE.protocol.stream_pacing.is_metered());
        assert!(!profile::PROFILE_NAVARRO.protocol.stream_pacing.is_metered());
        let pacing = profile::PROFILE_ELLA.protocol.stream_pacing;
        assert!(pacing.is_metered());
        assert_eq!(pacing.bytes_per_sec, 8_000_000);
        assert_eq!(pacing.burst_bytes, 24_000_000);
        // Room for the dock-wide activation keyframe, which is 9.56 MB inside half a second and
        // is accepted every time.
        assert!(pacing.burst_bytes > 9_560_000);
        // And within reach of the vendor's own worst second, rather than a fraction of it.
        assert!(i64::from(pacing.burst_bytes) + i64::from(pacing.bytes_per_sec) >= 30_000_000);

        let bps = pacing.bytes_per_sec;
        // A second of idle accrues exactly a second of budget, before the burst cap applies.
        assert_eq!(stream_credit_accrued(bps, 1_000_000), 8_000_000);
        assert_eq!(stream_credit_accrued(bps, 1_000), 8_000);
        // A long idle must not wrap into a negative windfall.
        assert!(stream_credit_accrued(bps, i64::MAX) > 0);
        assert_eq!(stream_credit_accrued(bps, -5), 0);

        // In credit, a frame goes now. Overdrawn, it waits for exactly the debt.
        assert_eq!(stream_credit_wait_us(bps, 1), None);
        assert_eq!(stream_credit_wait_us(bps, 0), None);
        // Overdrawn by a second's refill, a frame waits exactly a second.
        assert_eq!(stream_credit_wait_us(bps, -8_000_000), Some(1_000_001));
        assert!(stream_credit_wait_us(bps, i64::MIN).is_some());
    }
}
