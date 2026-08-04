// SPDX-License-Identifier: GPL-2.0

//! DRM/KMS integration for Vino.
//!
//! Each dock head has a primary plane, cursor plane, CRTC, encoder and connector. Framebuffers
//! are copied into driver-owned snapshots before atomic completion, then compressed and sent by
//! per-head workers. Connector modes come from downstream EDID tunneled over the dock's control
//! protocol.

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

/// The KMS objects the compositor drives, and the software vblank timer that paces them.
mod mode_objects;
/// Framebuffer to wire: damage selection, encode, record framing and URB submission.
mod scanout;

pub(super) use mode_objects::{
    PlaneArgs, VblankTimer, VinoConnector, VinoCrtc, VinoEncoder, VinoPlane,
};
use scanout::{read_cursor_bgra, run_pending_scanout, snapshot_to_shadow, src_dims};

/// Connector mode used until a downstream EDID is available.
const FALLBACK_W: i32 = 2560;
const FALLBACK_H: i32 = 1440;

/// Primary-plane format list (opaque 32bpp scanout).
static PRIMARY_FORMATS: [u32; 1] = [drm::fourcc::XRGB8888];

/// Cursor-plane format list.
static CURSOR_FORMATS: [u32; 1] = [drm::fourcc::ARGB8888];

/// Per-mode pixel-clock ceiling in kHz for a dock whose profile has not been applied yet.
///
/// Ridge's DLM never programs above 497.75 MHz, so no Ridge capture fills the high half of the
/// offset-70 `u32`; this keeps the value that half can express on its own.
pub(super) const DEFAULT_MAX_HEAD_CLOCK_KHZ: u32 = 655_350;

/// Refresh ceiling for a dock whose profile has not been applied yet.
///
/// This is Ridge's limit, which is also DLM's: asked for 2560x1440@180 it puts 119.998 Hz on the
/// wire, and asked for @85 it programs the 59.95 Hz CVT-RB timing.
pub(super) const DEFAULT_MAX_REFRESH_HZ: u32 = 120;

/// Return the active pixel rate, saturating on invalidly large modes.
pub(super) fn active_pixel_rate(hdisplay: u16, vdisplay: u16, vrefresh: i32) -> u32 {
    u32::from(hdisplay)
        .saturating_mul(u32::from(vdisplay))
        .saturating_mul(vrefresh.max(0) as u32)
}

/// Stream-marker state which powers down a head's downstream sink.
///
/// The resulting probe silence must be paired with [`VinoDrmData::self_blanked`] so it is not
/// mistaken for a physical disconnect.
const BLANK_MARKER_STATE: u8 = 1;

/// Delay before retrying a transient asynchronous control operation.
const KMS_RETRY_MS: u32 = 50;

/// Maximum number of physical downstream connectors Vino exposes.
///
/// Ridge docks use the first two. Navarro has four physical DP sockets; connectors 0/2 share
/// bulk endpoint 0x08 and connectors 1/3 share 0x0a. `DockProfile::connectors` selects the
/// active prefix at runtime, while this constant keeps the DRM object layout fixed at registration.
pub(crate) const HEADS: usize = 4;

/// Bulk transfer size on the video endpoints: a multiple of the 1024-byte maximum packet size, so
/// only a frame's final transfer terminates short.
const VIDEO_XFER: usize = 65536;

/// The repeated layout word in the DL7400's decoder configuration at 2560x1440.
///
/// Ridge uses `0x4000` for every mode. Only this value has been observed on the DL7400, and its
/// derivation from the mode is not established.
const NAVARRO_LAYOUT_WORD: u16 = 0x2100;

/// Maximum number of individual frame-damage rectangles re-converted per flip before they are
/// collapsed into a single bounding box. Bounds the stack array used on the atomic-commit path
/// (no per-flip allocation); a compositor that reports more clips than this just gets a coarser
/// (still correct) repaint.
const MAX_DAMAGE_CLIPS: usize = 16;
/// Minimum interval between normal frames for one head.
const FRAME_PERIOD_MS: i64 = 5;
/// The coalescing window in microseconds. Whole-millisecond arithmetic truncated the elapsed time
/// and forced a 1 ms minimum sleep, so a frame could wait materially longer than the window.
const FRAME_PERIOD_US: i64 = FRAME_PERIOD_MS * 1000;

/// How many consecutive frames must carry a strip after its content changes, so that every one of
/// the dock's buffers receives it.
///
/// The ring depth (`Geometry::dock_buffers`) is the theoretical minimum, plus one frame of margin
/// for a presentation the dock drops or applies to the buffer it just used. Coming up short leaves
/// a slot holding stale pixels, which the panel shows as ghosting and no wire analysis can see --
/// the bytes are correct.
#[inline]
fn damage_repeats(geom: super::video::wht::Geometry) -> u8 {
    geom.dock_buffers.saturating_add(1)
}
/// Activation timing relative to the mode-set submission.
const PROMPT_VIDEO_MS: i64 = 110;
const PROMPT_CLOSE_2F_MS: i64 = 123;
const PROMPT_CLOSE_2E_MS: i64 = 125;
/// Upper bound used to quiesce an already-running keepalive iteration.
const PROMPT_KEEPALIVE_QUIESCE_MS: i64 = 40;
/// Minimum interval between streaming status polls (`id=0x14 sub=0x0c`).
///
/// This was issued once per *presentation*, so two heads at ~100 fps with `damage_repeats()`
/// presentations each drove 200-600 CP round-trips a second, every one of them serialising on the
/// single control link. Whichever head looped fastest re-acquired it immediately and starved the
/// other for seconds at a time. DLM issues ~3.8 of these per second
/// (266 in a 70 s capture), so a quarter-second floor matches the reference and leaves the link
/// free for the other head.
const STATUS_POLL_MIN_MS: i64 = 250;
const PROMPT_TRAINING_OPEN_MS: i64 = 0;
const PROMPT_TRAINING_TAIL_MS: i64 = 400;
/// How long [`VinoDrmData::blank_head`] keeps presenting black when a CRTC is disabled.
///
/// It only has to outlast the dock's buffer rotation, which `damage_repeats` puts at three
/// presentations; a black frame is ~200 KB and presents in a couple of milliseconds, so this is
/// generous by an order of magnitude and still finishes well inside a DPMS transition. It is not a
/// training window -- nothing downstream needs settling -- so it does not reuse
/// [`PROMPT_TRAINING_TAIL_MS`].
const BLANK_PRESENT_MS: i64 = 120;

/// `edid_target` sentinel: nobody is waiting for an EDID.
const NO_EDID_TARGET: u32 = u32::MAX;
/// Dual-head cold-wake timeline relative to the head-0 mode set.
///
/// EP02 must remain quiet between `H1_MODE` and `QUIET_END`; the dock also requires a head-1 EDID
/// probe and fetch before video starts.
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
    pub(super) const POLLS: &[i64] = &[
        5, 26, 1019, 1130, 1192, 1204, 1222, 1235, 1253, 1270, 1287, 1304,
    ];
    /// `(offset_ms, head, is_fetch)` — `false` is the `0x15/0x20` probe, `true` the `0x15/0x21`
    /// fetch.
    pub(super) const EDID: &[(i64, u8, bool)] = &[(1033, 1, false), (1059, 1, true)];
    /// Keep both carriers active until downstream clock programming completes.
    pub(super) const CARRIER_TAIL_MS: i64 = 800;
}

/// A dock's cold bring-up choreography, anchored on the first head's mode set.
///
/// Ridge and Navarro differ in more than timing: Navarro opens each head's bracket with a
/// state-0 pair before the state-1/3 pair, spaces its two mode sets 757 ms apart rather than
/// 29 ms, sets head 0's mode a *second* time shortly before head 0's video, and streams head 1
/// first. Replaying one dock's timeline at the other leaves the endpoint unarmed.
pub(super) struct ColdTimeline {
    /// Offset of the second head's mode set.
    pub(super) h1_mode: i64,
    /// End of any silent window on EP02 after the second mode set.
    pub(super) quiet_end: i64,
    /// Heads in the order they start streaming, with the offset each starts at.
    pub(super) video: &'static [(usize, i64)],
    /// Mode sets repeated after the initial pair, as `(offset, head)`.
    pub(super) remode: &'static [(i64, usize)],
    /// `(offset_ms, head, sub, state)` stream markers. `sub` 0x2f/0x2e as on the wire.
    pub(super) markers: &'static [(i64, u8, u16, u8)],
    /// `id=0x14 sub=0x0c` status polls.
    pub(super) polls: &'static [i64],
    /// `(offset_ms, head, is_fetch)` EDID re-reads inside the bracket.
    pub(super) edid: &'static [(i64, u8, bool)],
}

/// Ridge's timeline, as replayed from a D6000 cold bring-up.
static COLD_RIDGE: ColdTimeline = ColdTimeline {
    h1_mode: cold::H1_MODE,
    quiet_end: cold::QUIET_END,
    video: &[(0, cold::H0_VIDEO), (1, cold::H1_VIDEO)],
    remode: &[],
    markers: cold::MARKERS,
    polls: cold::POLLS,
    edid: cold::EDID,
};

/// Navarro's timeline, measured from a DLM cold bring-up and anchored on connector 0's real
/// (`off23 = 2`) mode set, exactly as Ridge's is.
static COLD_NAVARRO: ColdTimeline = ColdTimeline {
    h1_mode: 10,
    // DLM polls continuously across this span; there is no silent window to preserve.
    quiet_end: 11,
    // Video is not a pair of one-shot events: DLM keeps head 0's carrier alive throughout the
    // still-open control bracket, starts head 1, and continues both through the closing markers.
    // A gap here makes the dock accept one frame and NAK the next forever. The activation path
    // uses its pre-encoded carrier; normal scanout replaces it as soon as activation returns.
    video: &[
        (0, 122),
        (0, 124),
        (0, 134),
        (0, 171),
        (0, 192),
        (0, 199),
        (0, 235),
        (0, 252),
        (1, 272),
        (0, 277),
        (1, 293),
        (1, 303),
    ],
    remode: &[],
    markers: &[
        (7, 0, 0x2f, 1),
        (13, 0, 0x2e, 3),
        (20, 1, 0x2f, 1),
        (21, 0, 0x2f, 1),
        (35, 0, 0x2e, 0),
        (76, 1, 0x2e, 3),
        (104, 1, 0x2f, 1),
        (128, 1, 0x2e, 3),
        (131, 0, 0x2f, 1),
        (136, 0, 0x2e, 0),
        (168, 1, 0x2f, 1),
        (181, 1, 0x2e, 0),
        (228, 0, 0x2f, 0),
        (230, 0, 0x2e, 0),
        (303, 1, 0x2f, 0),
        (304, 1, 0x2e, 0),
    ],
    polls: &[17, 78, 120, 162, 179, 223, 267, 295, 297, 297],
    // Navarro reads every connector's EDID before the anchor, not inside the bracket.
    edid: &[],
};

/// Reservation-token slots for [`COLD_NAVARRO::markers`] and [`COLD_NAVARRO::polls`].
///
/// DLM assigns counters in its per-head workers before their EP02 writes interleave. Wire AES
/// sequence remains monotonic, but the echoed inner counters consequently do not: for example the
/// wire order begins `n, n+1, n+3, n+2, n+5, n+4`. Navarro starts NAKing at the first flattened
/// counter, so retain that allocation order. These numbers index live reservation tokens; they
/// are not protocol counters and are never added to a captured/base counter.
static NAVARRO_MARKER_COUNTER_SLOTS: &[usize] =
    &[1, 2, 4, 6, 8, 7, 10, 12, 11, 14, 16, 17, 20, 21, 26, 27];
static NAVARRO_POLL_COUNTER_SLOTS: &[usize] = &[5, 9, 13, 15, 18, 19, 22, 23, 24, 25];
const NAVARRO_COLD_COUNTERS: usize = 28;

/// One operation in Navarro's cold sink-reset prelude. This is separate from [`ColdTimeline`]: it
/// runs before the first real mode set and changes downstream EDID/sink state, whereas
/// `ColdTimeline` brackets already-programmed streams.
#[derive(Clone, Copy)]
enum NavarroColdOp {
    Poll,
    EdidState(u8, u8),
    Probe(u8),
    Fetch(u8),
    SinkState(u8, u8),
    PostEdid(u8),
    Clear(u8),
}

/// Authenticated DLM transaction between Navarro's first clear pair and its first real mode.
///
/// Offsets are milliseconds from the first head-0 clear in
/// `navarro-dlm-today-124144/wire.pcapng`. Equal offsets deliberately retain wire order. Most
/// importantly, DLM stops both EDID readers and sends sink state `0xff` immediately after the
/// first clears, then re-reads and re-engages each sink before its second clear. Omitting this
/// whole state transition left the video endpoints accepting one bulk transfer and NAKing every
/// subsequent transfer.
static NAVARRO_COLD_PRELUDE: &[(i64, NavarroColdOp)] = &[
    (2, NavarroColdOp::EdidState(0, 0)),
    (3, NavarroColdOp::Probe(0)),
    (5, NavarroColdOp::EdidState(1, 0)),
    (5, NavarroColdOp::Probe(1)),
    (7, NavarroColdOp::SinkState(0, 0xff)),
    (7, NavarroColdOp::SinkState(1, 0xff)),
    (8, NavarroColdOp::Poll),
    (30, NavarroColdOp::Poll),
    (50, NavarroColdOp::Poll),
    (69, NavarroColdOp::Poll),
    (87, NavarroColdOp::Poll),
    (105, NavarroColdOp::Poll),
    (123, NavarroColdOp::Poll),
    (143, NavarroColdOp::Poll),
    (162, NavarroColdOp::Poll),
    (181, NavarroColdOp::Poll),
    (201, NavarroColdOp::Poll),
    (219, NavarroColdOp::Poll),
    (237, NavarroColdOp::Poll),
    (255, NavarroColdOp::Poll),
    (273, NavarroColdOp::Poll),
    (293, NavarroColdOp::Poll),
    (312, NavarroColdOp::Poll),
    (328, NavarroColdOp::Poll),
    (329, NavarroColdOp::Poll),
    (329, NavarroColdOp::Poll),
    (329, NavarroColdOp::Poll),
    (330, NavarroColdOp::Poll),
    (330, NavarroColdOp::Poll),
    (1216, NavarroColdOp::Poll),
    (1233, NavarroColdOp::Poll),
    (1315, NavarroColdOp::Poll),
    (1650, NavarroColdOp::Poll),
    (1667, NavarroColdOp::Poll),
    (1750, NavarroColdOp::Poll),
    (1755, NavarroColdOp::Probe(0)),
    (1757, NavarroColdOp::EdidState(0, 1)),
    (1758, NavarroColdOp::Probe(0)),
    (1805, NavarroColdOp::Fetch(0)),
    (1810, NavarroColdOp::SinkState(0, 0)),
    (1822, NavarroColdOp::Clear(0)),
    (1827, NavarroColdOp::Poll),
    (1850, NavarroColdOp::Probe(1)),
    (1853, NavarroColdOp::EdidState(1, 1)),
    (1856, NavarroColdOp::Probe(1)),
    (1902, NavarroColdOp::PostEdid(0)),
    (1903, NavarroColdOp::Fetch(1)),
    (1907, NavarroColdOp::SinkState(1, 1)),
    (1930, NavarroColdOp::Clear(1)),
    (1934, NavarroColdOp::Poll),
    (1956, NavarroColdOp::Poll),
    (1975, NavarroColdOp::Poll),
    (1994, NavarroColdOp::Poll),
    (2003, NavarroColdOp::Poll),
    (2005, NavarroColdOp::Poll),
    (2007, NavarroColdOp::PostEdid(1)),
    (2016, NavarroColdOp::Poll),
];

/// Navarro tears both pipe descriptors down first, then executes
/// [`NAVARRO_COLD_PRELUDE`] before programming the first real mode.
const NAVARRO_PRIME_CLEAR_H1_MS: i64 = 2;
const NAVARRO_REAL_MODE_H0_MS: i64 = 2978;
/// How long the KMS worker waits for the rest of a multi-head atomic commit's mode sets.
///
/// Bounded so a genuine single-head commit costs at most this before proceeding.
const MODESET_BATCH_SETTLE_MS: i64 = 20;

/// Number of back-to-back presentations of one already-encoded full frame while a newly-mode-set
/// downstream is training.
const COLD_TRAINING_PRESENTATIONS: u32 = 8;
type DamageRect = (usize, usize, usize, usize);
type BoundInterface<'a> = super::UsbLink<'a>;

/// Generation key for a complete timing, including refresh and pixel clock.
fn timing_key(t: &super::cp::Timing) -> u64 {
    ((t.hactive as u64) << 48)
        | ((t.vactive as u64) << 32)
        | (((t.refresh_hz & 0xff) as u64) << 24)
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

/// A control-protocol operation deferred from the non-blocking atomic callbacks.
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
        /// The dock's own visible flag. Hiding by parking the cursor at `u16::MAX` instead left a
        /// ghost pointer at the top-left of both panels: the dock wraps an out-of-range origin
        /// rather than clipping the cursor away.
        visible: bool,
    },
    /// Drive the stream black and close its control-protocol bracket.
    Blank {
        head: u8,
    },
}

impl KmsCmd {
    fn head(&self) -> usize {
        match self {
            Self::ModeSet { head, .. }
            | Self::CursorCreate { head, .. }
            | Self::CursorImage { head, .. }
            | Self::CursorMove { head, .. }
            | Self::Blank { head } => *head as usize,
        }
    }
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
    heads: [PendingKmsHead; HEADS],
}

impl PendingKms {
    const fn new() -> Self {
        Self {
            heads: [const { PendingKmsHead::new() }; HEADS],
        }
    }

    fn is_empty(&self) -> bool {
        self.heads.iter().all(|head| {
            head.stream.is_none()
                && head.cursor_create.is_none()
                && head.cursor_image.is_none()
                && head.cursor_move.is_none()
        })
    }

    fn has_stream(&self) -> bool {
        self.heads.iter().any(PendingKmsHead::has_stream)
    }

    fn update(&mut self, cmd: KmsCmd) {
        if let Some(pending) = self.heads.get_mut(cmd.head()) {
            pending.update(cmd);
        }
    }

    fn retry(&mut self, cmd: KmsCmd) {
        if let Some(pending) = self.heads.get_mut(cmd.head()) {
            pending.retry(cmd);
        }
    }
}

fn kms_error_retryable(error: Error) -> bool {
    error != EINVAL && error != ENOTSUPP
}

/// Latest primary-plane flip awaiting compression on the deferred worker. The framebuffer is
/// refcounted, so it remains valid after the atomic commit callback returns. There is one slot per
/// head: a newer flip replaces an older unsent flip instead of building an unbounded queue behind
/// a slow encoder. When replacement could lose accumulated damage, the newer flip is promoted to a
/// full-output damage rectangle.
struct PendingScanout {
    head: u8,
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
            head: self.head,
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

/// Maximum number of prepared compositor buffers retained per head.
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

/// One head's shadow surfaces. Locked **per head**: the snapshot copies ~14.7 MB while holding
/// this, and it runs on the compositor's non-blocking commit tail, so a device-wide lock made one
/// head's commit stall the other's -- measured at up to 4.2 ms, half a 120 Hz frame budget.
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

/// Period at which an idle DL7400 head is re-fed, comfortably inside
/// [`NAVARRO_VIDEO_QUIET_MS`].
const NAVARRO_KEEPALIVE_MS: i64 = 250;

/// Keep a missed repaint from being enough to trip the dock's teardown.
const _: () = assert!(NAVARRO_KEEPALIVE_MS * 3 <= NAVARRO_VIDEO_QUIET_MS);

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
    /// Latest desired control/KMS state per head.
    #[pin]
    pending_kms: Mutex<PendingKms>,
    /// Coalescing per-head scanout slots consumed by `cmd_work`.
    ///
    /// Compression and USB submission may sleep and therefore cannot run in
    /// `atomic_update`.
    #[pin]
    pending_scanout: Mutex<[Option<PendingScanout>; HEADS]>,
    /// A one-shot repaint of the head's newest known framebuffer. Cleared as soon as it is taken,
    /// or whenever a real flip arrives (that flip already carries newer content, so the redundant
    /// repaint is pointless). See [`SETTLE_REPAINT_MS`] for the hardware observation behind it.
    ///
    /// The `bool` is "promote to a full keyframe". It is **true** for the post-keyframe settle
    /// repaint, whose job is to replace a stale surface. It is **false** for a
    /// *debt* repaint, which carries outstanding `dirty_ttl` retransmissions
    /// to the dock's second buffer without promoting them to a keyframe.
    #[pin]
    settle_repaint: Mutex<[Option<(Instant<Monotonic>, PendingScanout, bool)>; HEADS]>,
    /// Private committed surfaces and their worker ownership state.
    #[pin]
    shadow: [Mutex<ShadowPool>; HEADS],
    /// Active software-vblank timers. The device owns their cancellation handles so shutdown does
    /// not depend on atomic-disable callbacks running. A spinlock is required because
    /// `enable_vblank` runs with local interrupts disabled.
    #[pin]
    vblank: SpinLock<[Option<(Arc<VblankTimer>, ArcHrTimerHandle<VblankTimer>)>; HEADS]>,
    /// Work item that drains control/KMS commands.
    #[pin]
    cmd_work: DelayedWork<VinoDrmDevice>,
    /// Independent per-head scanout workers. Their work IDs are const generics, so each head has
    /// an explicit field; transport state is taken from per-head slots while a frame is submitted.
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
    color: Mutex<[Option<super::color::ColorPipeline>; HEADS]>,
    /// Per-head strip hashes for the last frame accepted by the USB submission path. Updated only
    /// after the complete frame has been queued, so a failed transfer can never advance the shadow
    /// beyond what the dock may actually display.
    #[pin]
    strip_hashes: Mutex<[Option<StripHashState>; HEADS]>,
    /// The DL7400 per-strip size-class map most recently sent for each connector.
    ///
    /// The map describes the whole surface while a delta frame carries only its damaged strips, so
    /// rebuilding it from zero each frame re-declares every untouched position as class 0. See
    /// `video::wht::navarro_strip_params`.
    #[pin]
    strip_classes: Mutex<[KVec<u8>; HEADS]>,
    /// Per-strip retransmit debt. Spreading repeated updates across frames reaches both of the
    /// dock's scanout buffers; consecutive presentations can target the same buffer.
    #[pin]
    dirty_ttl: Mutex<[Option<KVVec<u8>>; HEADS]>,
    /// Set once the dock engages the CP cipher (`wsub=0x45` acks > 0); EP08 scanout is gated on it.
    /// Per device, so a second connected dock does not share one dock's engagement state.
    cp_engaged: core::sync::atomic::AtomicBool,
    /// This device's codec geometry, packed; see [`VinoDrmData::geometry`] and
    /// [`super::video::wht::Geometry`].
    codec_geometry: core::sync::atomic::AtomicU32,
    /// Which protocol generation this dock speaks; see `DockProfile::generation`. The two
    /// platforms differ in their initialisation, per-head HDCP framing, stream open and mode
    /// description, so one flag drives all of them rather than three that can disagree.
    is_navarro: core::sync::atomic::AtomicBool,
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
    /// Mode generation successfully programmed on each dock head. Scanout must match it because
    /// atomic plane updates can precede the deferred mode-set transaction.
    modeset_active: [core::sync::atomic::AtomicU64; HEADS],
    /// Latest mode userspace currently requests per head, encoded like `modeset_active`; zero
    /// means the CRTC is disabled. The deferred worker uses this generation key to discard stale
    /// mode-set commands and framebuffers left by a rapid disable/re-enable sequence.
    modeset_requested: [core::sync::atomic::AtomicU64; HEADS],
    /// Per-head timestamp of the last accepted frame, used to bound scanout cadence.
    #[pin]
    last_frame: SpinLock<[Option<Instant<Monotonic>>; HEADS]>,
    /// When `queue_scanout` last ran for each head, i.e. when KWin's commit tail last handed us a
    /// framebuffer. Distinguishes "the compositor stopped committing" from "we dropped the frame".
    #[pin]
    /// When the streaming status poll last went out, device-wide. The poll keeps the control
    /// dialogue alive; it does not need to be per presentation.
    #[pin]
    last_status_poll: SpinLock<Option<Instant<Monotonic>>>,
    /// When each head's scanout work item last began executing.
    #[pin]
    /// Rate limiter for the stall diagnostic below.
    #[pin]
    /// Deadline for the sustained full-frame stream required to train a cold downstream link.
    #[pin]
    sustain_until: SpinLock<[Option<Instant<Monotonic>>; HEADS]>,
    /// Logical WHT frame sequence per head.
    #[pin]
    scanout_seq: Mutex<[u32; HEADS]>,
    /// Persistent pipelined bulk-OUT queue per physical video endpoint. It remains live between
    /// frames.
    ///
    /// The slot is the first connector whose endpoint address matches the caller's (see
    /// [`UsbLink::video_pipe_index`](super::UsbLink::video_pipe_index)); duplicate slots remain
    /// empty. Holding an individual slot mutex over a whole frame serializes connectors that share
    /// a pipe without needlessly serializing independent endpoints.
    #[pin]
    video_q: [Mutex<Option<super::usb::BulkOutQueue>>; HEADS],
    /// One reusable 64-KiB coalescing window per head. `frame_records` deliberately stores a frame
    /// as small allocations so encoding never asks kmalloc for multi-megabyte physically contiguous
    /// memory; scanout joins those fragments into this bounded window before `BulkOutQueue::send`
    /// copies it into the persistent DMA ring. Internal record boundaries remain invisible on USB.
    #[pin]
    video_staging: Mutex<[Option<KVec<u8>>; HEADS]>,
    /// Last requested timing, retained so scanout can retry a failed mode-set.
    #[pin]
    last_timing: SpinLock<[Option<super::cp::Timing>; HEADS]>,
    /// Heads whose next video stream must be prefixed with the pipe-arm records.
    arm_prefix_pending: core::sync::atomic::AtomicU32,
    /// Heads for which the read-only endpoint status at the first video stall was logged.
    endpoint_status_logged: core::sync::atomic::AtomicU32,
    /// Connectors still owed the short sealed open that names a stream vino does not drive.
    ///
    /// Held apart from `arm_prefix_pending` because it is the complement of it: a connector vino
    /// is about to send pixels to opens its stream with the pipe descriptor instead, and both DLM
    /// captures send this record only on the stream ids of the connectors left idle. The opens go
    /// out before any head's first frame, as DLM's do.
    stream_open_pending: core::sync::atomic::AtomicU32,
    /// Per-head "owes a full keyframe" bitmask (bit `h` = head `h`). Set (all heads) after a
    /// `KmsCmd::ModeSet` send: a new mode leaves the dock's framebuffer undefined, so the first
    /// scanout after it must be a FULL frame ([`super::video::wht::colour_frame_ep08`]), not a
    /// damage delta -- otherwise the un-redrawn strips stay garbage. Cleared for a head once its
    /// keyframe is sent; subsequent flips send only changed strips through
    /// [`super::video::wht::colour_frame_ep08_damage`].
    keyframe_pending: core::sync::atomic::AtomicU32,
    /// Per-head generation of the dock's cursor bitmap, bumped by [`Self::owe_keyframe`].
    ///
    /// The cursor plane re-uploads only when its bitmap differs from the last one sent, so it needs
    /// to know when the dock stopped holding that bitmap. A mode-set discards it.
    cursor_epoch: [core::sync::atomic::AtomicU32; HEADS],
    /// Rotates the shadow slot each commit so successive snapshots do not land in the same one.
    shadow_rr: [core::sync::atomic::AtomicU32; HEADS],
    /// Geometry last announced with `cursor_create`, per head. Whether the dock keeps one shared
    /// cursor bitmap or one per head is not established, so each head announces and uploads its
    /// own -- correct either way, at the cost of one extra upload per shape change.
    #[pin]
    cursor_geometry: Mutex<[Option<(u16, u16)>; HEADS]>,
    /// Dock-wide pixel-rate budget in pixels per second; zero means unknown.
    dock_pixel_budget: core::sync::atomic::AtomicU32,
    /// Highest refresh rate this dock is known to drive; see `DockProfile::max_refresh_hz`.
    max_refresh_hz: core::sync::atomic::AtomicU32,
    /// Highest per-mode pixel clock in kHz; see `DockProfile::max_head_clock_khz`.
    max_head_clock_khz: core::sync::atomic::AtomicU32,
    /// Excludes scanout while a mode-set batch can submit on a video endpoint. Paired with
    /// `video_inflight` using sequentially consistent store-then-check handshakes.
    cmd_busy: core::sync::atomic::AtomicBool,
    /// Set around `run_pending_scanout`, allowing `cmd_work` to wait for a
    /// frame already in flight when it set [`Self::cmd_busy`].
    video_inflight: [core::sync::atomic::AtomicBool; HEADS],
    /// Consecutive failed live-scanout frames **per head**, for log rate-limiting.
    scanout_fails: [core::sync::atomic::AtomicU64; HEADS],
    /// Upcoming page flips to skip for per-head transport backoff.
    scanout_skip: [core::sync::atomic::AtomicU64; HEADS],
    /// Settle repaints this head may still arm. See [`SETTLE_REPAINTS`].
    settle_budget: [core::sync::atomic::AtomicU32; HEADS],
    /// Last inner status returned for each head's presence probe.
    presence_reply: [core::sync::atomic::AtomicU32; HEADS],
    /// Pending downstream-topology notification for this device's keepalive worker.
    downstream_event: AtomicBool,
    /// Head currently expecting an EDID from a re-engage, or [`NO_EDID_TARGET`].
    ///
    /// The EDID arrives as an `id=0x194` push, and during a re-engage it lands in `send_cp`'s
    /// own lockstep drain rather than in `drain_cp_pushes`. This says "somebody is waiting for
    /// one", so that drain can stash it instead of discarding it.
    edid_target: core::sync::atomic::AtomicU32,
    /// The blob that drain caught, handed back to [`VinoDrmData::reengage_head`].
    #[pin]
    edid_caught: Mutex<Option<KVec<u8>>>,
    /// Heads intentionally blanked by Vino. Their expected probe silence is not a hot-unplug.
    self_blanked: core::sync::atomic::AtomicU32,
    /// Per-head key and nonce used to seal pipe-arm records.
    #[pin]
    video_keys: Mutex<[kernel::crypto::Secret<32>; HEADS]>,
    /// When each head last put a byte on its video endpoint.
    ///
    /// Drives the DL7400 keep-alive: see [`NAVARRO_VIDEO_QUIET_MS`] for why a head that has
    /// nothing to draw still has to say something.
    #[pin]
    last_video_at: SpinLock<[Option<Instant<Monotonic>>; HEADS]>,
    /// Per-head AES-CTR block counter for the sealed records on that head's video stream.
    ///
    /// Every sealed video record carries this counter in its wire `seq`, and `seal_livemac` uses
    /// it both as the CTR block index and as the Dl3Cmac counter. It is stream state, not record
    /// state: DLM advances it by `ceil(plaintext / 16)` for every sealed record it sends on a
    /// stream and never rewinds it, so a re-arm continues the count rather than restarting. It is
    /// reset only when new video keys arrive, because a fresh key is a fresh keystream.
    video_seal_seq: [core::sync::atomic::AtomicU32; HEADS],
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
            pending_kms <- new_mutex!(PendingKms::new()),
            pending_scanout <- new_mutex!([const { None }; HEADS]),
            settle_repaint <- new_mutex!([const { None }; HEADS]),
            shadow <- pin_init::pin_init_array_from_fn(|_| new_mutex!(ShadowPool::new())),
            vblank <- new_spinlock!([const { None }; HEADS]),
            cmd_work <- new_delayed_work!("vino::kms_cmd"),
            scanout_work_h0 <- new_work!("vino::scanout_h0"),
            scanout_work_h1 <- new_work!("vino::scanout_h1"),
            scanout_work_h2 <- new_work!("vino::scanout_h2"),
            scanout_work_h3 <- new_work!("vino::scanout_h3"),
            session_queue: workqueue::Queue::new_ordered().build(kernel::c_str!("vino_session"))?,
            kms_queue: workqueue::Queue::new_ordered().build(kernel::c_str!("vino_kms"))?,
            scanout_queue: workqueue::Queue::new_unbound()
                .max_active(HEADS as u32)
                .build(kernel::c_str!("vino_scanout"))?,
            cached_edids <- new_mutex!([const { None }; HEADS]),
            heads_present: core::sync::atomic::AtomicU32::new(0),
            color <- new_mutex!([None; HEADS]),
            strip_hashes <- new_mutex!([const { None }; HEADS]),
            strip_classes <- new_mutex!(core::array::from_fn(|_| KVec::new())),
            dirty_ttl <- new_mutex!([const { None }; HEADS]),
            cp_engaged: core::sync::atomic::AtomicBool::new(false),
            cp_timeline_exclusive: core::sync::atomic::AtomicBool::new(false),
            initial_modeset_quiet: core::sync::atomic::AtomicBool::new(false),
            modeset_active: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            modeset_requested: core::array::from_fn(|_| core::sync::atomic::AtomicU64::new(0)),
            last_frame <- new_spinlock!([const { None }; HEADS]),
            last_status_poll <- new_spinlock!(None),
            sustain_until <- new_spinlock!([const { None }; HEADS]),
            scanout_seq <- new_mutex!([0; HEADS]),
            video_q <- pin_init::pin_init_array_from_fn(|_| new_mutex!(None)),
            video_staging <- new_mutex!([const { None }; HEADS]),
            last_timing <- new_spinlock!([None; HEADS]),
            arm_prefix_pending: core::sync::atomic::AtomicU32::new(0),
            endpoint_status_logged: core::sync::atomic::AtomicU32::new(0),
            stream_open_pending: core::sync::atomic::AtomicU32::new(0),
            keyframe_pending: core::sync::atomic::AtomicU32::new(0),
            cursor_epoch: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            shadow_rr: core::array::from_fn(|_| core::sync::atomic::AtomicU32::new(0)),
            cursor_geometry <- new_mutex!([None; HEADS]),
            // D6000 default: 442,368,000 px/s (one 1440p@120) x2 compression headroom = dual
            // 1440p@120. Replace it if a dock capability supplies a limit.
            dock_pixel_budget: core::sync::atomic::AtomicU32::new(884_736_000),
            max_refresh_hz: core::sync::atomic::AtomicU32::new(DEFAULT_MAX_REFRESH_HZ),
            max_head_clock_khz: core::sync::atomic::AtomicU32::new(DEFAULT_MAX_HEAD_CLOCK_KHZ),
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
            codec_geometry: core::sync::atomic::AtomicU32::new(0),
            connectors: core::sync::atomic::AtomicU8::new(HEADS as u8),
            is_navarro: core::sync::atomic::AtomicBool::new(false),
            video_keys <- new_mutex!(core::array::from_fn(
                |_| kernel::crypto::Secret::zeroed()
            )),
            last_video_at <- new_spinlock!([None; HEADS]),
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
        self.cp_timeline_exclusive.store(false, Ordering::Release);
        self.initial_modeset_quiet.store(false, Ordering::Release);
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
                let crtc: &VinoCrtc = crtc_ref.crtc();
                drop(crtc.vblank_pinned.lock().take());
                drop(crtc_ref);
            }
        }
        drop(timers);

        *self.pending_kms.lock() = PendingKms::new();
        *self.pending_scanout.lock() = [const { None }; HEADS];
        *self.settle_repaint.lock() = [const { None }; HEADS];
        for h in 0..HEADS {
            self.shadow[h].lock().discard();
        }
        *self.strip_hashes.lock() = [const { None }; HEADS];
        *self.dirty_ttl.lock() = [const { None }; HEADS];
        // Cancel the queued drain and reclaim the `ARef<VinoDrmDevice>` the enqueue handed to
        // the workqueue, if it was still pending. Dropping it here releases the self-reference
        // that would otherwise keep this device alive until the work ran.
        //
        // Cancel `cmd_work` first because it can enqueue both scanout workers. `shutting_down` is
        // already visible to all workers, so cancellation only waits for work already in flight.
        drop(self.cmd_work.cancel_sync());
        drop(self.scanout_work_h0.cancel_sync());
        drop(self.scanout_work_h1.cancel_sync());
        drop(self.scanout_work_h2.cancel_sync());
        drop(self.scanout_work_h3.cancel_sync());

        // A running callback may have taken a batch just before shutdown was published. It has
        // finished now; clear anything it left behind and tear the USB queues down while their
        // parent interface is still in Bound context.
        *self.pending_kms.lock() = PendingKms::new();
        *self.pending_scanout.lock() = [const { None }; HEADS];
        *self.settle_repaint.lock() = [const { None }; HEADS];
        for h in 0..HEADS {
            self.shadow[h].lock().discard();
        }
        *self.strip_hashes.lock() = [const { None }; HEADS];
        *self.dirty_ttl.lock() = [const { None }; HEADS];
        for queue in &self.video_q {
            *queue.lock() = None;
        }
        *self.video_staging.lock() = [const { None }; HEADS];
        *self.cp_link.lock() = None;
        vino_debug!("vino: deferred KMS/video work drained for unplug\n");
    }

    /// Cache `head`'s CRTC colour transform (from `RawCrtcState::gamma_lut` and
    /// `RawCrtcState::ctm`) for the scanout to apply, or clear it to identity with two `None`s.
    pub(super) fn update_color(
        &self,
        head: usize,
        lut: Option<&[crtc::ColorLut]>,
        ctm: Option<&crtc::ColorCtm>,
    ) {
        let cached = super::color::ColorPipeline::build(lut, ctm);
        let changed = if let Some(slot) = self.color.lock().get_mut(head) {
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
                vino_debug!("vino: head {head} colour transform updated\n");
            } else {
                vino_debug!("vino: head {head} colour transform cleared\n");
            }
            // The encoded-strip cache keys on a strip's source pixels, so a transform change that
            // leaves those pixels untouched would otherwise re-send stale bodies for the whole
            // screen. Drop the cache and owe a keyframe.
            self.strip_hashes.lock()[head] = None;
            self.dirty_ttl.lock()[head] = None;
            self.owe_keyframe(head);
        }
    }

    /// Snapshot `head`'s cached colour transform for a scanout pass (`Copy`, so no lock is held
    /// afterwards).
    pub(super) fn color_snapshot(&self, head: usize) -> Option<super::color::ColorPipeline> {
        self.color.lock().get(head).copied().flatten()
    }



    /// Record this device's codec geometry; see [`Self::geometry`].
    pub(super) fn set_codec_geometry(
        &self,
        strip_blocks_x: usize,
        interlaced: bool,
        band_parity: bool,
        head_sub_shift: u8,
        stream_id_mask: u8,
        dock_buffers: u8,
    ) {
        let packed = (strip_blocks_x as u32 & 0xff)
            | ((interlaced as u32) << 8)
            | ((band_parity as u32) << 9)
            | ((head_sub_shift as u32) << 16)
            | ((stream_id_mask as u32) << 20)
            | ((dock_buffers as u32) << 28);
        self.codec_geometry
            .store(packed | 0x8000, core::sync::atomic::Ordering::Release);
    }

    /// This device's codec geometry, to be passed into every codec call made on its behalf.
    ///
    /// Stored packed because the DRM device allocation is pin-initialised before `probe` knows
    /// which dock it matched. A device with no profile applied reads the Ridge layout.
    pub(super) fn geometry(&self) -> super::video::wht::Geometry {
        let p = self.codec_geometry.load(core::sync::atomic::Ordering::Acquire);
        if p & 0x8000 == 0 {
            return super::video::wht::RIDGE_GEOMETRY;
        }
        super::video::wht::Geometry::new(
            ((p & 0xff) as usize).max(1),
            p & (1 << 8) != 0,
            p & (1 << 9) != 0,
            ((p >> 16) & 0xf) as u8,
            ((p >> 20) & 0xff) as u8,
            ((p >> 28) & 0xf) as u8,
        )
    }


    /// Record which protocol generation this dock speaks.
    pub(super) fn set_navarro(&self, on: bool) {
        self.is_navarro.store(on, Ordering::Release);
    }

    /// Whether this dock speaks the Navarro protocol.
    pub(super) fn is_navarro(&self) -> bool {
        self.is_navarro.load(Ordering::Acquire)
    }

    /// Whether mode sets carry the DL7400's offset-46/48 words.
    pub(super) fn navarro_mode_words(&self) -> bool {
        self.is_navarro()
    }

    /// Whether the first frame after a mode set carries the cold ARM burst.
    pub(super) fn uses_arm_burst(&self) -> bool {
        !self.is_navarro()
    }

    /// Record how many connectors this dock exposes; see [`DockProfile::connectors`].
    pub(super) fn set_connectors(&self, n: u8) {
        self.connectors.store(
            if n == 0 {
                HEADS as u8
            } else {
                n.min(HEADS as u8)
            },
            Ordering::Release,
        );
    }

    /// Number of physical connectors selected by the matched dock profile.
    pub(super) fn connector_count(&self) -> usize {
        usize::from(self.connectors.load(Ordering::Acquire)).min(HEADS)
    }

    /// Whether `head` represents a distinct runtime stream on this dock.
    ///
    /// The DL7400 exposes four connector selectors but maps them in pairs onto two video bulk
    /// endpoints. DLM discovers all four selectors during setup, then drives and polls only the
    /// first selector for each distinct endpoint. Treating selectors 2/3 as independent heads
    /// interleaves spurious re-engage traffic into the authenticated mode-set transcript.
    pub(super) fn runtime_connector(&self, head: usize) -> bool {
        if head >= self.connector_count() {
            return false;
        }
        if !self.navarro_mode_words() {
            return true;
        }
        let address = self.eps.video[head].address();
        !self.eps.video[..head]
            .iter()
            .any(|endpoint| endpoint.address() == address)
    }

    pub(super) fn set_cp_engaged(&self, engaged: bool) {
        self.cp_engaged
            .store(engaged, core::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the current device session has engaged content protection.
    pub(super) fn cp_engaged(&self) -> bool {
        self.cp_engaged.load(core::sync::atomic::Ordering::Acquire)
    }

    /// Pause the background CP loop and let any iteration which passed its check finish. The mode
    /// worker calls this before taking its timestamp anchor; the fixed delay therefore cannot move
    /// any event relative to the mode-set itself.
    fn begin_cp_timeline(&self) {
        self.cp_timeline_exclusive.store(true, Ordering::Release);
        self.initial_modeset_quiet.store(false, Ordering::Release);
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
        ep84_depth: usize,
    ) {
        // EP84 must remain posted between runtime EP02 writes. A queue drained synchronously leaves
        // the endpoint unposted between calls and can stall the control protocol.
        // `ep84_depth` is the matched profile's `ep84_queue_depth`, so the runtime queue keeps the
        // same number of reads posted that bring-up did.
        let ep84_q = match dev.ctrl_in_queue(ep84_depth, 4096) {
            Ok(q) => Some(q),
            Err(e) => {
                pr_warn!("vino: persistent EP84 queue open failed ({e:?}); using sync fallback\n");
                None
            }
        };
        *self.cp_link.lock() = Some(CpLink {
            ks: kernel::crypto::Secret::new(*ks),
            riv: *riv,
            wire_seq,
            counter,
            ep84_q,
        });
    }

    /// Store the per-head video keys produced by the `id=0x32` exchange.
    ///
    /// Called with [`publish_session`](Self::publish_session) when CP engages.
    pub(super) fn set_video_keys(&self, keys: [kernel::crypto::Secret<32>; HEADS]) {
        *self.video_keys.lock() = keys;
        // A new key is a new keystream, so the block counters start over with it.
        for seq in &self.video_seal_seq {
            seq.store(0, Ordering::Release);
        }
    }

    /// Reserve `blocks` AES-CTR blocks on a head's video stream and return the counter to seal at.
    ///
    /// Sealed video records must tile the stream's keystream without gaps or overlaps: the dock
    /// tracks the same counter, and a record that repeats a block a previous record already used
    /// is a replay of that keystream. Reserving before sealing keeps that true no matter how the
    /// records are grouped into transfers.
    fn take_seal_seq(&self, head: usize, blocks: u32) -> u32 {
        self.video_seal_seq[head].fetch_add(blocks, Ordering::AcqRel)
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

    /// One `id=0x16 sub=0x2e|0x2f` stream/display marker. **State lives in byte 23**, not byte 22
    /// (byte 22 is constantly `1` — reading it makes every marker look like state=1).
    fn stream_marker(&self, dev: &BoundInterface<'_>, head: u8, sub: u16, st: u8) -> Result {
        self.send_cp(dev, 0x16, 0, |ctr| {
            super::cp::stream_marker(ctr, head, sub, st)
        })
    }

    /// Send one captured Navarro sink-reset operation.
    fn navarro_cold_op(&self, dev: &BoundInterface<'_>, op: NavarroColdOp) -> Result {
        match op {
            NavarroColdOp::Poll => self.poll_status(dev),
            NavarroColdOp::EdidState(head, state) => self.send_cp(dev, 0x16, 0, |ctr| {
                super::cp::edid_readiness_state(ctr, head, state)
            }),
            NavarroColdOp::Probe(head) => self.send_cp(dev, 0x15, 0, |ctr| {
                super::cp::get_edid_req_sub(ctr, 0x20, head)
            }),
            NavarroColdOp::Fetch(head) => {
                self.send_cp(dev, 0x15, 0, |ctr| super::cp::get_edid_req(ctr, head))
            }
            NavarroColdOp::SinkState(head, state) => self.send_cp(dev, 0x16, 0, |ctr| {
                super::cp::edid_sink_state(ctr, head, state)
            }),
            NavarroColdOp::PostEdid(head) => {
                self.send_cp(dev, 0x15, 0, |ctr| super::cp::post_edid_query(ctr, head))
            }
            NavarroColdOp::Clear(head) => {
                self.send_cp(dev, 0x48, 0, |ctr| super::cp::clear_mode(ctr, head))
            }
        }
    }

    /// Drive `head` to black on the dock, then close its stream bracket.
    ///
    /// Runs on the command worker for [`KmsCmd::Blank`], i.e. after `atomic_disable` has already
    /// zeroed this head's mode generation. That zero is what makes the write legal: every video
    /// path gates on `modeset_requested == modeset_active == want`, and passing `want = 0` matches
    /// exactly the disabled state and nothing else -- so a re-enable racing this blank flips both
    /// atomics to a real key and the submit loop drops out with `ENODEV` instead of painting black
    /// over the freshly enabled mode.
    ///
    /// The dock's stream itself is still configured (vino never told it otherwise), so the black
    /// frames are an ordinary accepted write, not a write onto a torn-down pipe.
    fn blank_head(&self, dev: &BoundInterface<'_>, head: u8) -> Result {
        let head_i = head as usize;
        let Some(timing) = self.last_timing.lock()[head_i] else {
            // Never modeset, so there is nothing lit to blank.
            return Ok(());
        };
        let geom = self.geometry();
        let w_pad = (timing.hactive as usize + geom.strip_w() - 1) & !(geom.strip_w() - 1);
        let h_pad = (timing.vactive as usize + geom.strip_h() - 1) & !(geom.strip_h() - 1);
        let frames = super::video::wht::black_frame_ep08(geom, w_pad, h_pad, head)?;
        let ordinary_frames =
            super::video::wht::black_frame_ep08_ordinary(geom, w_pad, h_pad, head)?;
        // Present for long enough to reach every dock buffer. The dock is multi-buffered and a
        // single presentation lands in one buffer only -- the same reason `damage_repeats` exists
        // -- so a one-shot blank leaves the other buffer holding the frozen desktop and the panel
        // alternates between black and stale content.
        let sent = self.submit_prompt_training(
            dev,
            head,
            0,
            &frames,
            &ordinary_frames,
            BLANK_PRESENT_MS,
            false,
        )?;
        // Close the stream with the validated bracket. This stops the DisplayLink stream without
        // forcing the monitor into hard standby.
        self.stream_marker(dev, head, 0x2f, 0)?;
        self.stream_marker(dev, head, 0x2e, 0)?;
        // Do not take the sink down for a head whose monitor has already gone away.
        //
        // `atomic_disable` fires for both a DPMS-off and a monitor removal, and they need opposite
        // treatment. Sending the power-down marker at a sink that is already gone is pointless, and
        // setting `self_blanked` would make the presence watcher deliberately ignore that head's
        // silence, preventing a later replug from being detected.
        let candidate = if self.head_present(head_i) {
            BLANK_MARKER_STATE
        } else {
            vino_debug!(
                "vino: head {head} blank skips the sink marker -- its monitor is already gone\n"
            );
            0
        };
        if candidate != 0 {
            // From here the dock will stop answering this head's presence probe, exactly as it does
            // for a real unplug. Claim the silence before causing it.
            self.set_self_blanked(head_i, true);
            let power_down = (|| -> Result {
                self.stream_marker(dev, head, 0x2f, 1)?;
                self.stream_marker(dev, head, 0x2e, candidate as u8)?;
                self.poll_status(dev)?;
                self.stream_marker(dev, head, 0x2f, 0)
            })();
            if let Err(e) = power_down {
                self.set_self_blanked(head_i, false);
                return Err(e);
            }
            vino_debug!("vino: head {head} downstream sink powered down\n");
        }
        vino_debug!("vino: head {head} blanked on the dock ({sent} black presentation(s))\n");
        Ok(())
    }

    /// Open the per-head stream bracket before changing an active mode.
    fn modeset_bracket_pre(&self, dev: &BoundInterface<'_>, head: u8) -> Result {
        self.stream_marker(dev, head, 0x2f, 1)?;
        self.stream_marker(dev, head, 0x2e, 3)?;
        self.poll_status(dev)
    }

    /// Sleep until an absolute millisecond offset from the mode-set anchor.
    ///
    /// Absolute deadlines keep scheduler delay from accumulating across the activation sequence.
    fn wait_mode_offset(anchor: Instant<Monotonic>, target_ms: i64) {
        let elapsed_ms = (Instant::<Monotonic>::now() - anchor).as_millis();
        if elapsed_ms < target_ms {
            fsleep(Delta::from_millis(target_ms - elapsed_ms));
        }
    }

    /// Sleep/spin until an exact microsecond offset in a short video-transport schedule.
    ///
    /// `fsleep` handles the bulk of the delay without burning a CPU; the small busy-wait tail
    /// avoids scheduling a producer boundary hundreds of microseconds late. This is used only for
    /// the four submissions of Navarro's one-shot prologue.
    fn wait_video_offset(anchor: Instant<Monotonic>, target_us: i64) {
        const SPIN_MARGIN_US: i64 = 80;
        let elapsed = anchor.elapsed().as_micros_ceil();
        if elapsed >= target_us {
            return;
        }
        if target_us - elapsed > SPIN_MARGIN_US {
            fsleep(Delta::from_micros(target_us - elapsed - SPIN_MARGIN_US));
        }
        let elapsed = anchor.elapsed().as_micros_ceil();
        if elapsed < target_us {
            udelay(Delta::from_micros(target_us - elapsed));
        }
    }

    /// Complete the stream-open markers and status polls up to the first video deadline.
    fn modeset_bracket_post_open(
        &self,
        dev: &BoundInterface<'_>,
        head: u8,
        anchor: Instant<Monotonic>,
    ) -> Result {
        self.poll_status(dev)?;
        Self::wait_mode_offset(anchor, 5);
        self.stream_marker(dev, head, 0x2f, 1)?;
        Self::wait_mode_offset(anchor, 9);
        self.stream_marker(dev, head, 0x2e, 3)?;
        Self::wait_mode_offset(anchor, 12);
        self.stream_marker(dev, head, 0x2f, 1)?;
        Self::wait_mode_offset(anchor, 14);
        self.stream_marker(dev, head, 0x2e, 3)?;
        Self::wait_mode_offset(anchor, 20);
        self.stream_marker(dev, head, 0x2f, 1)?;
        // The status poll shares the final `2f(1)` deadline.
        self.poll_status(dev)?;
        Self::wait_mode_offset(anchor, 26);
        self.stream_marker(dev, head, 0x2e, 0)?;
        // There is a measured 63-ms quiet interval, then three polls at +89/+95/+110 ms. The last
        // poll and first video bytes share one deadline.
        Self::wait_mode_offset(anchor, 89);
        self.poll_status(dev)?;
        Self::wait_mode_offset(anchor, 95);
        self.poll_status(dev)?;
        Self::wait_mode_offset(anchor, PROMPT_VIDEO_MS);
        self.poll_status(dev)
    }

    /// Close the post-mode-set bracket after prompt video has started.
    ///
    /// The first close marker is +13 ms from video and the second is +15 ms. Background keepalive
    /// resumes immediately after this pair and supplies the continuing status dialogue.
    fn modeset_bracket_post_close(
        &self,
        dev: &BoundInterface<'_>,
        head: u8,
        anchor: Instant<Monotonic>,
    ) -> Result {
        Self::wait_mode_offset(anchor, PROMPT_CLOSE_2F_MS);
        self.stream_marker(dev, head, 0x2f, 0)?;
        Self::wait_mode_offset(anchor, PROMPT_CLOSE_2E_MS);
        self.stream_marker(dev, head, 0x2e, 0)
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
        ordinary_frames: &[KVec<u8>],
        duration_ms: i64,
        with_arm: bool,
    ) -> Result<u32> {
        if frames.is_empty() {
            return Err(kernel::error::code::EINVAL);
        }
        let geom = self.geometry();
        // Diagnostic: shrink the transfer so a dock that stops after one *transfer* can be told
        // apart from one that stops after a fixed *byte count*. Both look identical at the default
        // 65536, because exactly one transfer and exactly 65536 bytes are the same event.
        let xfer: usize = match *crate::module_parameters::video_xfer.value() {
            0 => VIDEO_XFER,
            n => (n as usize).clamp(1024, VIDEO_XFER) & !1023,
        };
        let head_i = head as usize;
        let pipe_i = dev.video_pipe_index(head_i)?;
        let head_bit = 1u32 << head;
        // Diagnostic cap: send only the first N whole image records of a frame, so a bring-up
        // failure can be bisected between the stream's opening records and its pixels. Records
        // are a uniform 4048-byte stride until a frame's short last one, so this stays aligned.
        let image_cap = match *crate::module_parameters::video_records.value() {
            0 => usize::MAX,
            n => (n as usize).saturating_mul(4048),
        };
        // The DL7400's per-strip parameter map, ahead of the pixels it describes. This path --
        // the startup/prompt-training submit -- is the one the dock's first frames go out on, so
        // wiring the map only into the steady-state scanout left it absent from every frame that
        // matters. Ridge has no equivalent record and gets an empty slice.
        let params: KVec<u8> = if geom.head_sub_shift == 0 {
            KVec::new()
        } else {
            let t = self.last_timing.lock().get(head_i).copied().flatten();
            match t {
                Some(t) => {
                    let (sw, sh) = (geom.strip_w(), geom.strip_h());
                    let w_pad = (t.hactive as usize).div_ceil(sw) * sw;
                    let h_pad = (t.vactive as usize).div_ceil(sh) * sh;
                    super::video::wht::navarro_strip_params(
                        geom,
                        head,
                        w_pad,
                        h_pad,
                        &frames,
                        &mut self.strip_classes.lock()[head_i],
                    )?
                }
                None => KVec::new(),
            }
        };
        vino_debug!("vino: head={head} prompt-training parameter map {} B\n", params.len());
        let arm = if with_arm {
            if self.arm_prefix_pending.load(Ordering::Acquire) & head_bit == 0 {
                return Err(ENODEV);
            }
            Some(self.build_stream_prefix_buf(head_i)?)
        } else {
            None
        };
        if arm.is_some() {
            self.send_stream_open(dev, head_i)?;
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
            let trailer = self.build_frame_trailer(head, seq);
            let arm_slice: &[u8] = if repeat == 0 {
                arm.as_ref().map_or(&[], |a| &a[..])
            } else {
                &[]
            };
            let opener = self.build_frame_opener(head, seq, !arm_slice.is_empty());
            let opener_slice: &[u8] = opener.as_ref().map_or(&[], |o| &o[..]);
            // The dock is owed one sealed report on the stream sub for every frame on the frame
            // sub; without it the link is torn down a few seconds after the first frame.
            //
            // The prologue frame is the one exception for the report: DLM goes directly from its
            // decoder configuration into image records. It is *not* an exception for the
            // parameter map. The same-day DLM cold capture carries the map part-way through frame
            // zero, and Windows carries it after the image records. Both put it before frame
            // close. Omitting it made vino's first frame exactly 5,984 bytes short and left the
            // dock briefly scanning a partially described framebuffer.
            let report = if arm_slice.is_empty() {
                self.build_stream_report_buf(head_i)?
            } else {
                None
            };
            let report_slice: &[u8] = report.as_ref().map_or(&[], |r| &r[..]);
            let params_slice: &[u8] = &params[..];
            // The prologue and ordinary DLM carriers contain identical strips but different
            // producer record boundaries. Select by the presence of the one-shot arm rather than
            // by this helper's local `repeat`: the cold timeline invokes the helper once per
            // measured presentation, so every invocation starts with repeat zero.
            let frame_parts = if arm_slice.is_empty() {
                ordinary_frames
            } else {
                frames
            };
            let mut image_len: usize = 0;
            let mut image_parts = 0usize;
            for f in frame_parts.iter() {
                if image_len >= image_cap {
                    break;
                }
                image_len += f.len().min(image_cap - image_len);
                image_parts += 1;
            }
            let wire_len = arm_slice.len()
                + opener_slice.len()
                + report_slice.len()
                + params_slice.len()
                + image_len
                + trailer.len();
            {
                let mut staging_slots = self.video_staging.lock();
                let staging_slot = &mut staging_slots[head_i];
                if staging_slot.is_none() {
                    let mut staging = KVec::new();
                    staging.resize(xfer, 0, GFP_KERNEL)?;
                    *staging_slot = Some(staging);
                }
                let staging = staging_slot.as_mut().ok_or(kernel::error::code::ENOMEM)?;

                let mut queue_slot = self.video_q[pipe_i].lock();
                if queue_slot.is_none() {
                    // Navarro's required EP08/EP0a clears are sent together at their captured
                    // pre-commit point in `send_cp_setup`. Repeating a clear here, after stream
                    // setup has begun, is intentionally diagnostic-only: it changes SuperSpeed
                    // endpoint sequence state at a point DLM does not.
                    if *crate::module_parameters::video_clear_halt.value() != 0 {
                        let _ = dev.clear_video_halt(head_i);
                    }
                    *queue_slot = Some(dev.video_queue(head_i, 8, xfer)?);
                    vino_debug!(
                        "vino: head={} endpoint={:#04x} persistent video queue opened by prompt training\n",
                        head,
                        dev.eps.video[head_i].address()
                    );
                }
                let queue = queue_slot
                    .as_mut()
                    .get_mut()
                    .as_mut()
                    .ok_or(kernel::error::code::ENODEV)?;

                // A carrier is one protocol frame split over several URBs. Never block after
                // submitting only its prefix: DLM's video producer and control workers are
                // independent, while this cold-timeline worker also owes the marker burst that
                // surrounds the first video bytes. On a non-draining endpoint the old path filled
                // the eight slots with two complete frames plus one URB of frame three, then
                // waited a second for frame-three URB two. The due +14-ms marker consequently
                // never reached EP02 even though the dock had authenticated the preceding marker.
                // Defer the whole presentation when it cannot fit; the next scheduled carrier can
                // retry after the control messages have advanced the dock state.
                let frame_urbs = wire_len.div_ceil(xfer);
                if !queue.can_send_n(dev.io(), frame_urbs)? {
                    if self
                        .endpoint_status_logged
                        .fetch_or(head_bit, Ordering::AcqRel)
                        & head_bit
                        == 0
                    {
                        match dev.video_endpoint_status(head_i) {
                            Ok(status) => pr_info!(
                                "vino: head={} endpoint={:#04x} stopped accepting video: GET_STATUS={:#06x} halt={}\n",
                                head,
                                dev.eps.video[head_i].address(),
                                status,
                                status & 1
                            ),
                            Err(e) => pr_warn!(
                                "vino: head={} endpoint={:#04x} stopped accepting video: GET_STATUS failed ({e:?})\n",
                                head,
                                dev.eps.video[head_i].address()
                            ),
                        }
                    }
                    if duration_ms <= 0 {
                        return Ok(repeat);
                    }
                    if (Instant::<Monotonic>::now() - started).as_millis() >= duration_ms {
                        break;
                    }
                    fsleep(Delta::from_millis(1));
                    continue;
                }

                // Navarro's captured DLM stream flushes its two parameter records after exactly
                // 115168 bytes of image records in both the prologue and ordinary carriers. The
                // placement is load-bearing: putting the same valid records after all pixels lets
                // two presentations complete, then leaves the first transfer of frame three
                // permanently pending. Build a borrowed scatter list in the captured order. The
                // insertion may fall inside one allocation chunk, but is always on a wire-record
                // boundary.
                const NAVARRO_PARAM_IMAGE_OFFSET: usize = 115168;
                let mut wire_parts: KVec<&[u8]> = KVec::with_capacity(
                    image_parts + 6,
                    GFP_KERNEL,
                )?;
                if !arm_slice.is_empty() {
                    wire_parts.push(arm_slice, GFP_KERNEL)?;
                }
                if !opener_slice.is_empty() {
                    wire_parts.push(opener_slice, GFP_KERNEL)?;
                }
                if !report_slice.is_empty() {
                    wire_parts.push(report_slice, GFP_KERNEL)?;
                }
                let param_at = NAVARRO_PARAM_IMAGE_OFFSET.min(image_len);
                let mut image_off = 0usize;
                let mut param_inserted = params_slice.is_empty();
                for f in frame_parts.iter() {
                    if image_off >= image_len {
                        break;
                    }
                    let n = f.len().min(image_len - image_off);
                    let split = param_at.saturating_sub(image_off).min(n);
                    if !param_inserted && image_off + n >= param_at {
                        if split != 0 {
                            wire_parts.push(&f[..split], GFP_KERNEL)?;
                        }
                        wire_parts.push(params_slice, GFP_KERNEL)?;
                        param_inserted = true;
                        if split != n {
                            wire_parts.push(&f[split..n], GFP_KERNEL)?;
                        }
                    } else if n != 0 {
                        wire_parts.push(&f[..n], GFP_KERNEL)?;
                    }
                    image_off += n;
                }
                if !param_inserted {
                    wire_parts.push(params_slice, GFP_KERNEL)?;
                }
                wire_parts.push(&trailer[..], GFP_KERNEL)?;

                let part_count = wire_parts.len();
                let mut part_i = 0usize;
                let mut part_off = 0usize;
                let mut wire_off = 0usize;
                // DLM's authenticated first head-0 prologue does not put all four URBs on the
                // xHCI ring at once. It submits at +0/+806/+851/+873 us and receives completion
                // of the first at +104 us, so the dock gets a ~700-us ready interval before the
                // second transfer and then a three-URB pipeline. Submitting chunk two immediately
                // leaves Navarro NRDY forever after exactly one completed URB. Preserve this
                // producer boundary only for the one-shot, full-size prologue; ordinary frames
                // use the normal eight-deep queue, just as DLM does.
                const NAVARRO_PROLOGUE_SUBMIT_US: [i64; 4] = [0, 806, 851, 873];
                let pace_prologue = !arm_slice.is_empty()
                    && xfer == VIDEO_XFER
                    && geom.head_sub_shift != 0
                    && *crate::module_parameters::video_sync.value() == 0;
                let mut prologue_anchor: Option<Instant<Monotonic>> = None;
                while wire_off < wire_len {
                    let data_len = (wire_len - wire_off).min(xfer);
                    let dst = &mut staging[..data_len];
                    let mut dst_off = 0usize;
                    while dst_off < dst.len() && part_i < part_count {
                        let part = wire_parts[part_i];
                        let n = (part.len() - part_off).min(dst.len() - dst_off);
                        dst[dst_off..dst_off + n].copy_from_slice(&part[part_off..part_off + n]);
                        dst_off += n;
                        part_off += n;
                        if part_off == part.len() {
                            part_i += 1;
                            part_off = 0;
                        }
                    }
                    // Diagnostic: the dock accepts exactly one transfer and then takes no more
                    // -- 16384 bytes when the transfer is 16384, 65536 when it is 65536, so it is
                    // one *transfer*, not a byte count. Clearing the halt before each one tests
                    // whether the dock is halting the endpoint after every transfer.
                    if *crate::module_parameters::video_clear_each.value() != 0 {
                        if *crate::module_parameters::video_clear_halt.value() != 0 {
                            let _ = dev.clear_video_halt(head_i);
                        }
                    }
                    if pace_prologue {
                        let chunk = wire_off / xfer;
                        if let Some(anchor) = prologue_anchor {
                            if let Some(&target_us) = NAVARRO_PROLOGUE_SUBMIT_US.get(chunk) {
                                Self::wait_video_offset(anchor, target_us);
                            }
                        }
                    }
                    // `video_sync` is an all-synchronous diagnostic. The normal queue path uses
                    // DLM's mixed transport: prologue chunk zero is reaped below, then the rest
                    // and all ordinary frames are pipelined.
                    let sent = if *crate::module_parameters::video_sync.value() != 0 {
                        dev.video_send(head_i, dst, super::timeout(), GFP_KERNEL).map(|_| ())
                    } else {
                        queue.send(dev.io(), dst, super::timeout())
                    };
                    if let Err(e) = sent {
                        if *crate::module_parameters::video_clear_halt.value() != 0 {
                            let _ = dev.clear_video_halt(head_i);
                        }
                        return Err(e);
                    }
                    if pace_prologue && wire_off == 0 {
                        let anchor = Instant::<Monotonic>::now();
                        prologue_anchor = Some(anchor);
                        // Do not expose chunk two to xHCI before chunk zero completes. The capture
                        // has the first completion at +104 us and the next submit at +806 us.
                        queue.flush(dev.io(), super::timeout())?;
                    }
                    self.last_video_at.lock()[head_i] = Some(Instant::<Monotonic>::now());
                    wire_off += data_len;
                }
            }

            if repeat == 0 && startup {
                self.arm_prefix_pending
                    .fetch_and(!head_bit, Ordering::Release);
                self.sustain_until.lock()[head_i] =
                    Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
                vino_debug!(
                    "vino: head {} startup frame submitted after {} ms ({} bytes)\n",
                    head,
                    (Instant::<Monotonic>::now() - started).as_millis(),
                    wire_len
                );
            }
            repeat = repeat.wrapping_add(1);
            self.scanout_seq.lock()[head_i] = seq0.wrapping_add(repeat);

            if repeat > 0 && (Instant::<Monotonic>::now() - started).as_millis() >= duration_ms {
                break;
            }
        }

        vino_debug!(
            "vino: head={} training complete ({} presentations, {} ms)\n",
            head,
            repeat,
            (Instant::<Monotonic>::now() - started).as_millis()
        );
        Ok(repeat)
    }

    /// Apply one desired mode generation and its activation carrier.
    ///
    /// The control timeline is always released before returning. Any failed bracket, mode-set, arm,
    /// or carrier transfer clears `modeset_active`, allowing the desired generation to be retried.
    fn activate_head(
        &self,
        dev: &BoundInterface<'_>,
        head: u8,
        timing: &super::cp::Timing,
        want: u64,
    ) -> Result<bool> {
        let head_i = head as usize;
        if head_i >= HEADS || self.modeset_requested[head_i].load(Ordering::Acquire) != want {
            return Ok(false);
        }

        let geom = self.geometry();
        let w_pad = (timing.hactive as usize + geom.strip_w() - 1) & !(geom.strip_w() - 1);
        let h_pad = (timing.vactive as usize + geom.strip_h() - 1) & !(geom.strip_h() - 1);
        let prompt = super::video::wht::black_frame_ep08(geom, w_pad, h_pad, head)?;
        let prompt_ordinary =
            super::video::wht::black_frame_ep08_ordinary(geom, w_pad, h_pad, head)?;
        let wake = self.modeset_active[head_i].load(Ordering::Acquire) == 0;

        self.begin_cp_timeline();
        let transaction = (|| -> Result<bool> {
            if !wake {
                self.modeset_bracket_pre(dev, head)?;
            }
            let mode_anchor = Instant::<Monotonic>::now();
            // Tear the connector's pipe down before configuring it, as DLM does. See
            // `cp::clear_mode`: offset 23 is an operation code, and vino only ever sent the
            // "set this mode" form.
            if self.navarro_mode_words() {
                self.send_cp(dev, 0x48, 0, |ctr| super::cp::clear_mode(ctr, head))?;
            }
            self.send_cp(dev, 0x48, 0, |ctr| super::cp::set_mode(ctr, head, timing))?;
            if self.modeset_requested[head_i].load(Ordering::Acquire) != want {
                return Ok(false);
            }

            self.modeset_active[head_i].store(want, Ordering::Release);
            self.sustain_until.lock()[head_i] =
                Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
            let bit = 1u32 << head;
            self.arm_prefix_pending.fetch_or(bit, Ordering::Release);
            // A driven connector's stream opens with its pipe descriptor, not the idle open.
            self.stream_open_pending.fetch_and(!bit, Ordering::Release);
            self.owe_keyframe(head_i);
            self.strip_hashes.lock()[head_i] = None;
            self.dirty_ttl.lock()[head_i] = None;

            self.modeset_bracket_post_open(dev, head, mode_anchor)?;
            let opening = self.submit_prompt_training(
                dev,
                head,
                want,
                &prompt,
                &prompt_ordinary,
                PROMPT_TRAINING_OPEN_MS,
                true,
            );
            let closing = self.modeset_bracket_post_close(dev, head, mode_anchor);
            opening?;
            closing?;
            Ok(true)
        })();
        self.end_cp_timeline();

        let activated = match transaction {
            Ok(activated) => activated,
            Err(e) => {
                self.modeset_active[head_i].store(0, Ordering::Release);
                return Err(e);
            }
        };
        if !activated {
            return Ok(false);
        }
        if let Err(e) =
            self.submit_prompt_training(
                dev,
                head,
                want,
                &prompt,
                &prompt_ordinary,
                PROMPT_TRAINING_TAIL_MS,
                false,
            )
        {
            self.modeset_active[head_i].store(0, Ordering::Release);
            return Err(e);
        }

        vino_debug!(
            "vino: applied {} stream-enable sequence for head {}\n",
            if wake { "wake" } else { "mode-change" },
            head
        );
        Ok(true)
    }

    /// Activate both downstream heads using the dock-wide cold-link schedule.
    ///
    /// Both mode sets precede either head's video. Single-head activation and live mode changes
    /// use the per-head schedule.
    fn activate_dual_wake(
        &self,
        dev: &BoundInterface<'_>,
        timings: [Option<super::cp::Timing>; HEADS],
    ) -> Result<bool> {
        let geom = self.geometry();
        let mut prompts: [Option<KVec<KVec<u8>>>; HEADS] = core::array::from_fn(|_| None);
        let mut ordinary_prompts: [Option<KVec<KVec<u8>>>; HEADS] =
            core::array::from_fn(|_| None);
        let mut keys = [0u64; HEADS];
        let mut valid = 0u32;

        // Pre-encode both tiny carriers before excluding the keepalive or starting either
        // mode-set. Encoding work must not serialize the dock's back-to-back mode pair.
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
            vino_debug!(
                "vino: dual activation head={} mode={}x{}@{}\n",
                head,
                timing.hactive,
                timing.vactive,
                timing.refresh_hz
            );
            let w_pad = (timing.hactive as usize + geom.strip_w() - 1) & !(geom.strip_w() - 1);
            let h_pad = (timing.vactive as usize + geom.strip_h() - 1) & !(geom.strip_h() - 1);
            prompts[head] = Some(super::video::wht::black_frame_ep08(
                geom,
                w_pad,
                h_pad,
                head as u8,
            )?);
            ordinary_prompts[head] = Some(super::video::wht::black_frame_ep08_ordinary(
                geom,
                w_pad,
                h_pad,
                head as u8,
            )?);
            keys[head] = key;
            valid |= 1u32 << head;
        }
        if valid.count_ones() < 2 {
            return Ok(false);
        }

        // Keep the clear/settle phase and the real mode-set choreography in one exclusive control
        // transaction. Their deadlines have separate anchors because the Navarro cold timeline was
        // measured from the real head-0 mode set, 1,156 ms after its pipe clear.
        self.begin_cp_timeline();
        let activation_started = Instant::<Monotonic>::now();
        let mut anchor = activation_started;
        let mut sent = 0u32;
        let mut started = 0u32;
        let timeline = (|| -> Result<(u32, u32)> {
            if self.navarro_mode_words() {
                // DLM's first clear pair begins a dock-wide sink reset. The authenticated
                // transcript then stops/restarts each EDID reader, disengages/re-engages the
                // downstream sinks, and clears each pipe a second time before any real mode.
                for head in 0..HEADS {
                    if valid & (1u32 << head) == 0 {
                        continue;
                    }
                    if head == 1 {
                        Self::wait_mode_offset(activation_started, NAVARRO_PRIME_CLEAR_H1_MS);
                    }
                    self.send_cp(dev, 0x48, 0, |ctr| {
                        super::cp::clear_mode(ctr, head as u8)
                    })?;
                }
                for &(at, op) in NAVARRO_COLD_PRELUDE {
                    Self::wait_mode_offset(activation_started, at);
                    self.navarro_cold_op(dev, op)?;
                }
                Self::wait_mode_offset(activation_started, NAVARRO_REAL_MODE_H0_MS);
                anchor = Instant::<Monotonic>::now();
            }

            let navarro_counters = if self.navarro_mode_words() {
                Some(self.reserve_cp_counters::<NAVARRO_COLD_COUNTERS>()?)
            } else {
                None
            };

            // Three cursors walk the sorted schedules; `cp_until` drains everything due at or
            // before a given offset, preserving the ordering between markers, polls, and EDID
            // reads.
            let mut mi = 0usize;
            let mut pi = 0usize;
            let mut ei = 0usize;
            // Replaying Ridge's choreography at Navarro leaves its video endpoint unarmed, so the
            // timeline follows the dock, not the driver.
            let timeline: &ColdTimeline = if self.uses_arm_burst() {
                &COLD_RIDGE
            } else {
                &COLD_NAVARRO
            };
            let mut remoded = 0u32;

            macro_rules! cp_until {
                ($limit:expr) => {{
                    let limit: i64 = $limit;
                    loop {
                        let nm = timeline.markers.get(mi).map(|m| m.0);
                        let np = timeline.polls.get(pi).copied();
                        let ne = timeline.edid.get(ei).map(|e| e.0);
                        let next = [nm, np, ne]
                            .into_iter()
                            .flatten()
                            .filter(|&o| o <= limit)
                            .min();
                        let Some(off) = next else { break };
                        Self::wait_mode_offset(anchor, off);
                        if nm == Some(off) {
                            let (_, head, sub, state) = timeline.markers[mi];
                            if sent & (1u32 << head) != 0 {
                                if let Some(counters) = navarro_counters.as_ref() {
                                    let slot = *NAVARRO_MARKER_COUNTER_SLOTS
                                        .get(mi)
                                        .ok_or(EINVAL)?;
                                    let ctr = *counters.get(slot).ok_or(EINVAL)?;
                                    self.send_cp_reserved(dev, 0x16, ctr, |ctr| {
                                        super::cp::stream_marker(ctr, head, sub, state)
                                    })?;
                                } else {
                                    self.stream_marker(dev, head, sub, state)?;
                                }
                            }
                            mi += 1;
                        } else if np == Some(off) {
                            if let Some(counters) = navarro_counters.as_ref() {
                                let slot = *NAVARRO_POLL_COUNTER_SLOTS
                                    .get(pi)
                                    .ok_or(EINVAL)?;
                                let ctr = *counters.get(slot).ok_or(EINVAL)?;
                                self.send_cp_reserved(dev, 0x14, ctr, |ctr| {
                                    super::cp::device_query_req(ctr, 0x000c)
                                })?;
                            } else {
                                self.poll_status(dev)?;
                            }
                            pi += 1;
                        } else {
                            let (_, head, fetch) = timeline.edid[ei];
                            // Re-read the sink's EDID at its required place in the transaction.
                            // This dock-side DDC operation is not a source of new modes, so discard
                            // its reply rather than publishing a hotplug during a mode set.
                            self.send_cp(dev, 0x15, 0, |ctr| {
                                if fetch {
                                    super::cp::get_edid_req(ctr, head)
                                } else {
                                    super::cp::get_edid_req_sub(ctr, 0x0020, head)
                                }
                            })?;
                            ei += 1;
                        }
                    }
                }};
            }

            // Both real mode sets go out before any video, spaced according to this dock's
            // measured cold timeline. Navarro's pipe clears were sent during the settling phase
            // above; do not collapse them back into this loop.
            for head in 0..HEADS {
                let bit = 1u32 << head;
                if valid & bit == 0 {
                    continue;
                }
                let Some(timing) = timings[head] else {
                    continue;
                };
                if head == 1 {
                    cp_until!(timeline.h1_mode - 1);
                    Self::wait_mode_offset(anchor, timeline.h1_mode);
                }
                if let Some(counters) = navarro_counters.as_ref() {
                    let slot = if head == 0 { 0 } else { 3 };
                    let ctr = *counters.get(slot).ok_or(EINVAL)?;
                    self.send_cp_reserved(dev, 0x48, ctr, |ctr| {
                        super::cp::set_mode(ctr, head as u8, &timing)
                    })?;
                } else {
                    self.send_cp(dev, 0x48, 0, |ctr| {
                        super::cp::set_mode(ctr, head as u8, &timing)
                    })?;
                }
                if self.modeset_requested[head].load(Ordering::Acquire) != keys[head] {
                    continue;
                }
                self.modeset_active[head].store(keys[head], Ordering::Release);
                self.sustain_until.lock()[head] =
                    Some(Instant::<Monotonic>::now() + Delta::from_millis(3000));
                self.arm_prefix_pending.fetch_or(bit, Ordering::Release);
                // A driven connector's stream opens with its pipe descriptor, not the idle open.
                self.stream_open_pending.fetch_and(!bit, Ordering::Release);
                self.owe_keyframe(head);
                self.strip_hashes.lock()[head] = None;
                self.dirty_ttl.lock()[head] = None;
                sent |= bit;
            }

            // Preserve the required silent window on EP02 between the head-1 mode set and
            // `cold::QUIET_END`. The exclusive control timeline already excludes keepalives.
            Self::wait_mode_offset(anchor, timeline.quiet_end);

            // Bracket, status polls and the mid-bracket EDID re-read, up to the first video.
            cp_until!(timeline.video[0].1 - 1);

            // Name the streams vino will not drive, before any head sends pixels. DLM does this
            // for every connector without a monitor and only then starts a frame; the connectors
            // that are about to stream open with their pipe descriptor instead.
            if !self.uses_arm_burst()
                && *crate::module_parameters::idle_opens.value() != 0
            {
                let idle = !sent & ((1u32 << HEADS) - 1);
                self.stream_open_pending.fetch_or(idle, Ordering::Release);
                for head in 0..HEADS {
                    if idle & (1u32 << head) != 0 {
                        self.send_stream_open(dev, head)?;
                    }
                }
            }

            for &(head, at) in timeline.video {
                cp_until!(at - 1);
                // Some docks set a head's mode a second time shortly before its video.
                for &(off, rehead) in timeline.remode {
                    if off >= at || remoded & (1u32 << rehead) != 0 || sent & (1u32 << rehead) == 0
                    {
                        continue;
                    }
                    let Some(timing) = timings[rehead] else {
                        continue;
                    };
                    cp_until!(off - 1);
                    Self::wait_mode_offset(anchor, off);
                    self.send_cp(dev, 0x48, 0, |ctr| {
                        super::cp::set_mode(ctr, rehead as u8, &timing)
                    })?;
                    remoded |= 1u32 << rehead;
                }
                cp_until!(at - 1);
                Self::wait_mode_offset(anchor, at);
                let bit = 1u32 << head;
                if sent & bit == 0 {
                    continue;
                }
                // Exactly one ARM+carrier presentation keeps the closing markers from being
                // delayed behind a blocking multi-frame submission.
                let frames = prompts[head].as_ref().ok_or(EINVAL)?;
                let ordinary_frames = ordinary_prompts[head].as_ref().ok_or(EINVAL)?;
                let t_sub = Instant::<Monotonic>::now();
                let first_for_head = started & bit == 0;
                self.submit_prompt_training(
                    dev,
                    head as u8,
                    keys[head],
                    frames,
                    ordinary_frames,
                    PROMPT_TRAINING_OPEN_MS,
                    first_for_head,
                )?;
                // The timeline collapses right after this call: head 1's video slipped from its
                // scheduled +150 ms to +1321 ms, so the markers DLM sends inside the stream never
                // go out. Report the cost unconditionally until that is understood.
                pr_info!(
                    "vino: head {} video submit took {} ms (timeline offset {} ms, {} ms since anchor)\n",
                    head,
                    (Instant::<Monotonic>::now() - t_sub).as_millis(),
                    at,
                    (Instant::<Monotonic>::now() - anchor).as_millis()
                );
                started |= bit;
            }

            // Remaining polls and the closing markers.
            cp_until!(i64::MAX);
            Ok((sent, started))
        })();
        self.end_cp_timeline();
        let (sent, started) = match timeline {
            Ok(state) => state,
            Err(e) => {
                for head in 0..HEADS {
                    if sent & (1u32 << head) != 0
                        && self.modeset_active[head].load(Ordering::Acquire) == keys[head]
                    {
                        self.modeset_active[head].store(0, Ordering::Release);
                    }
                }
                return Err(e);
            }
        };
        if sent.count_ones() < 2 {
            for head in 0..HEADS {
                if sent & (1u32 << head) != 0
                    && self.modeset_active[head].load(Ordering::Acquire) == keys[head]
                {
                    self.modeset_active[head].store(0, Ordering::Release);
                }
            }
            return Ok(false);
        }

        // Keep both endpoints busy through downstream clock training, so the carrier outlives the
        // bracket rather than stopping with it.
        let tail_started = Instant::<Monotonic>::now();
        while (Instant::<Monotonic>::now() - tail_started).as_millis() < cold::CARRIER_TAIL_MS {
            for head in 0..HEADS {
                if started & (1u32 << head) == 0 {
                    continue;
                }
                let frames = prompts[head].as_ref().ok_or(EINVAL)?;
                let ordinary_frames = ordinary_prompts[head].as_ref().ok_or(EINVAL)?;
                if let Err(e) = self.submit_prompt_training(
                    dev,
                    head as u8,
                    keys[head],
                    frames,
                    ordinary_frames,
                    PROMPT_TRAINING_OPEN_MS,
                    false,
                ) {
                    for reset in 0..HEADS {
                        if sent & (1u32 << reset) != 0
                            && self.modeset_active[reset].load(Ordering::Acquire) == keys[reset]
                        {
                            self.modeset_active[reset].store(0, Ordering::Release);
                        }
                    }
                    return Err(e);
                }
            }
        }
        vino_debug!(
            "vino: dual-head activation complete after {} ms (mode/started masks 0x{:x}/0x{:x})\n",
            (Instant::<Monotonic>::now() - anchor).as_millis(),
            sent,
            started
        );
        Ok(true)
    }

    /// Build one head's cold video-arm burst, prepended to the first video frame after a mode set.
    /// Records #0/#1/#4/#5 are plaintext and #6/#7 contain a fixed `type=4`
    /// body. Records #2/#3/#8/#9 use this head's video key and nonce, derived
    /// from the per-head SKE with `riv_h ^ (0x08 | head)` in byte 7, and share
    /// one block counter. Records #8/#9 carry the decoder configuration and
    /// independent nonces.
    /// Build the per-head video stream-open this platform wants in place of the cold ARM burst.
    ///
    /// Sealed with the head's video key like every other video-endpoint message, and prefixed to
    /// the first frame after a mode set so it reaches the dock before any pixels.
    /// Build the prefix that opens a head's video stream after a mode set.
    ///
    /// Ridge prefixes a cold ARM burst to the first frame; Navarro prefixes a short sealed
    /// stream-open. Both occupy the same slot ahead of any pixels, and every submission path must
    /// pick between them the same way: sending one platform's opening to the other's dock leaves
    /// the stream unopened, and the dock then watchdog-resets a few seconds later.
    /// Build the records that close a frame, in this dock's format.
    fn build_frame_trailer(&self, head: u8, seq0: u32) -> super::video::wht::FrameTrailer {
        let geom = self.geometry();
        if self.uses_arm_burst() {
            super::video::wht::frame_trailer(geom, head, seq0)
        } else {
            super::video::wht::navarro_frame_trailer(geom, head, seq0)
        }
    }

    /// Build the record that starts a non-prologue DL7400 frame.
    ///
    /// Ridge carries its slot transition in the three-record trailer. Navarro terminates the old
    /// frame after its close record and starts the next USB transfer with this opener instead.
    fn build_frame_opener(&self, head: u8, seq0: u32, prologue: bool) -> Option<[u8; 32]> {
        if self.uses_arm_burst() || prologue {
            None
        } else {
            Some(super::video::wht::navarro_frame_opener(self.geometry(), head, seq0))
        }
    }

    /// Open a head's video stream, once per mode generation, ahead of any pixels.
    ///
    /// Does nothing on a dock whose opening is the ARM burst carried with the first frame.
    fn send_stream_open(&self, dev: &BoundInterface<'_>, head: usize) -> Result {
        let bit = 1u32 << head;
        if self.stream_open_pending.load(Ordering::Acquire) & bit == 0 {
            return Ok(());
        }
        let Some(open) = self.build_stream_open_buf(head)? else {
            self.stream_open_pending.fetch_and(!bit, Ordering::Release);
            return Ok(());
        };
        let pipe_i = dev.video_pipe_index(head)?;
        let mut queue_slot = self.video_q[pipe_i].lock();
        if queue_slot.is_none() {
            if *crate::module_parameters::video_clear_halt.value() != 0 {
                let _ = dev.clear_video_halt(head);
            }
            *queue_slot = Some(dev.video_queue(head, 8, VIDEO_XFER)?);
        }
        let queue = queue_slot
            .as_mut()
            .get_mut()
            .as_mut()
            .ok_or(kernel::error::code::ENODEV)?;
        queue.send(dev.io(), &open, super::timeout())?;
        self.stream_open_pending.fetch_and(!bit, Ordering::Release);
        vino_debug!("vino: head {} video stream opened\n", head);
        Ok(())
    }

    /// Keep every engaged head's video endpoint from going quiet long enough for the dock to tear
    /// the link down.
    ///
    /// The DL7400 stops answering -- video *and* control -- about a second after the last byte on
    /// a video endpoint, whatever it was doing before. A compositor with nothing to redraw leaves
    /// vino silent well past that, so each head that has not sent anything for
    /// [`NAVARRO_KEEPALIVE_MS`] sends the same sealed report DLM pairs with every frame. Called
    /// from the control keepalive, which already runs for the life of the session.
    ///
    /// Only heads whose video queue is already open are fed: a head that has never streamed has
    /// nothing to keep alive, and opening a queue here would start a stream nothing follows.
    pub(super) fn send_video_keepalive(&self, dev: &BoundInterface<'_>) {
        if self.uses_arm_burst() || self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let now = Instant::<Monotonic>::now();
        for head in 0..HEADS {
            if self.modeset_active[head].load(Ordering::Acquire) == 0 {
                continue;
            }
            let due = match self.last_video_at.lock()[head] {
                Some(last) => (now - last).as_millis() >= NAVARRO_KEEPALIVE_MS,
                None => false,
            };
            if !due {
                continue;
            }
            let Ok(report) = self.build_stream_report_buf(head) else {
                continue;
            };
            let Some(report) = report else { continue };
            let Ok(pipe_i) = dev.video_pipe_index(head) else {
                continue;
            };
            let mut queue_slot = self.video_q[pipe_i].lock();
            let Some(queue) = queue_slot.as_mut().get_mut().as_mut() else {
                continue;
            };
            if queue.send(dev.io(), &report, super::timeout()).is_ok() {
                self.last_video_at.lock()[head] = Some(Instant::<Monotonic>::now());
            }
        }
    }

    fn build_stream_prefix_buf(&self, head: usize) -> Result<KVec<u8>> {
        if self.uses_arm_burst() {
            self.build_arm_burst_buf(head)
        } else {
            self.build_navarro_prologue_buf(head)
        }
    }

    /// Build the message a head's video stream opens with, sent alone ahead of everything else.
    ///
    /// Ridge has none: its ARM burst opens the stream from within the first frame's transfer.
    ///
    /// Navarro has none either, for a head it is about to drive. The short sealed open does exist
    /// on this dock, but both DLM captures send it only on the stream ids of the connectors with
    /// no monitor -- `0x17` and `0x1f` while pixels went to connectors 0 and 1 -- each as the
    /// first and only record on its own stream, sealed with that connector's own key at block 0.
    /// A driven connector's sealed chain instead opens with the pipe descriptor at block 0.
    ///
    /// It must not go out on `stream_id | 0x10`, which for head 0 is connector 2's stream id,
    /// sealed with head 0's key. That both signed another connector's stream with the wrong key
    /// and, because the prologue then also started at block 0, used the head's first keystream
    /// block twice.
    fn build_stream_open_buf(&self, head: usize) -> Result<Option<KVec<u8>>> {
        if self.uses_arm_burst() {
            return Ok(None);
        }
        let keys = self.video_keys.lock();
        let key = keys.get(head).ok_or(EINVAL)?;
        let mut vkey = kernel::crypto::Secret::zeroed();
        vkey.copy_from_slice(&key[..16]);
        let mut vnonce = [0u8; 8];
        vnonce.copy_from_slice(&key[16..24]);
        drop(keys);
        let content = super::cp::navarro_stream_open();
        let stream = self.geometry().stream_id(head as u8);
        let seq = self.take_seal_seq(head, content.len().div_ceil(16) as u32);
        Ok(Some(super::cp::seal_video_arm(
            &vkey, &vnonce, stream, 0x0002, seq, &content,
        )?))
    }

    /// Build the sealed report a head owes its stream for one frame.
    ///
    /// DLM pairs every frame on the frame sub with one of these on the stream sub, so a stream
    /// that sends pixels and then falls silent on its stream sub is a stream the dock stops
    /// believing in. Returns `None` on a dock whose frames carry no such record.
    ///
    /// The ordinary `aux=0x000c` form is what DLM sends for all but a handful of frames; the
    /// `aux=0x0002` form restates the mode and goes out with the frame that carries the prologue,
    /// which is the frame right after a mode set.
    fn build_stream_report_buf(&self, head: usize) -> Result<Option<KVec<u8>>> {
        if self.uses_arm_burst() {
            return Ok(None);
        }
        let keys = self.video_keys.lock();
        let key = keys.get(head).ok_or(EINVAL)?;
        let mut vkey = kernel::crypto::Secret::zeroed();
        vkey.copy_from_slice(&key[..16]);
        let mut vnonce = [0u8; 8];
        vnonce.copy_from_slice(&key[16..24]);
        drop(keys);
        let stream = self.geometry().stream_id(head as u8);
        let with_mode = self.arm_prefix_pending.load(Ordering::Acquire) & (1u32 << head) != 0;
        let (aux, content): (u16, KVec<u8>) = if with_mode {
            let timing = self
                .last_timing
                .lock()
                .get(head)
                .copied()
                .flatten()
                .ok_or(ENODEV)?;
            let header = super::video_arm::mode_header(
                timing.hactive,
                timing.vactive,
                NAVARRO_LAYOUT_WORD,
            );
            let mut v = KVec::new();
            v.extend_from_slice(&super::cp::navarro_stream_report_mode(&header), GFP_KERNEL)?;
            (0x0002, v)
        } else {
            let mut v = KVec::new();
            v.extend_from_slice(&super::cp::navarro_stream_report(), GFP_KERNEL)?;
            (0x000c, v)
        };
        let seq = self.take_seal_seq(head, content.len().div_ceil(16) as u32);
        Ok(Some(super::cp::seal_video_arm(
            &vkey, &vnonce, stream, aux, seq, &content,
        )?))
    }

    /// Build the DL7400 records that precede a head's first frame.
    ///
    /// In wire order: two plaintext stream markers, the sealed pipe descriptor, a plaintext frame
    /// marker, an unsealed record naming the connector's first and fifth ring addresses, and the
    /// sealed decoder configuration. Both sealed records draw from the stream's running block
    /// counter, so on a first arm the descriptor seals at block 0 and the configuration at block
    /// 19 -- the descriptor's 304 bytes in blocks -- exactly as DLM's `0 -> 19 -> 88` chain does.
    fn build_navarro_prologue_buf(&self, head: usize) -> Result<KVec<u8>> {
        let keys = self.video_keys.lock();
        let key = keys.get(head).ok_or(EINVAL)?;
        let mut vkey = kernel::crypto::Secret::zeroed();
        vkey.copy_from_slice(&key[..16]);
        let mut vnonce = [0u8; 8];
        vnonce.copy_from_slice(&key[16..24]);
        drop(keys);
        let timing = self
            .last_timing
            .lock()
            .get(head)
            .copied()
            .flatten()
            .ok_or(ENODEV)?;
        let connector = head as u8;
        let geom = self.geometry();
        let stream = geom.stream_id(connector);
        let frame_sub = u16::from(geom.head_sub(connector));

        let mut buf = KVec::with_capacity(1600, GFP_KERNEL)?;
        for sub in [stream, stream | 0x0010] {
            let mut body = [0u8; 16];
            body[0..2].copy_from_slice(&sub.to_le_bytes());
            body[2..4].copy_from_slice(&0x0006u16.to_le_bytes());
            buf.extend_from_slice(&super::cp::video_arm_plain_frame(sub, &body), GFP_KERNEL)?;
        }

        let descriptor = super::cp::navarro_pipe_descriptor(connector)?;
        let seal_seq = self.take_seal_seq(head, descriptor.len().div_ceil(16) as u32);
        let mut sealed =
            super::cp::seal_video_arm(&vkey, &vnonce, stream, 0x0000, seal_seq, &descriptor)?;
        // Diagnostic: a dock that authenticates this record must behave differently when its tag
        // is wrong. Identical behaviour means the record is already being dropped.
        if *crate::module_parameters::break_mac.value() != 0 {
            if let Some(last) = sealed.last_mut() {
                *last ^= 0xff;
            }
        }
        buf.extend_from_slice(&sealed, GFP_KERNEL)?;

        let mut body = [0u8; 16];
        body[0..2].copy_from_slice(&frame_sub.to_le_bytes());
        buf.extend_from_slice(
            &super::cp::video_arm_plain_frame(frame_sub, &body),
            GFP_KERNEL,
        )?;

        // Unsealed type-4 record: the connector's first and fifth ring addresses.
        let mut ring = [0u8; 32];
        ring[2..4].copy_from_slice(&0x001cu16.to_le_bytes());
        ring[4..8].copy_from_slice(&4u32.to_le_bytes());
        ring[8..10].copy_from_slice(&frame_sub.to_le_bytes());
        ring[10..12].copy_from_slice(&0x0004u16.to_le_bytes());
        ring[16..19].copy_from_slice(&[0x0a, 0x00, 0x04]);
        ring[19] = frame_sub as u8;
        ring[22..26].copy_from_slice(&super::cp::navarro_pipe_ring(connector, 0).to_le_bytes());
        ring[26..30].copy_from_slice(&super::cp::navarro_pipe_ring(connector, 4).to_le_bytes());
        buf.extend_from_slice(&ring, GFP_KERNEL)?;

        let mut tail = [0u8; 14];
        super::rng::fill(&mut tail);
        let config = super::video_arm::build_with_layout_word(
            timing.hactive,
            timing.vactive,
            NAVARRO_LAYOUT_WORD,
            &tail,
        )?;
        let seal_seq = self.take_seal_seq(head, config.len().div_ceil(16) as u32);
        buf.extend_from_slice(
            &super::cp::seal_video_arm(&vkey, &vnonce, stream, 0x000e, seal_seq, &config)?,
            GFP_KERNEL,
        )?;
        Ok(buf)
    }

    fn build_arm_burst_buf(&self, head: usize) -> Result<KVec<u8>> {
        let keys = self.video_keys.lock();
        let key = keys.get(head).ok_or(EINVAL)?;
        let mut vkey = kernel::crypto::Secret::zeroed();
        vkey.copy_from_slice(&key[..16]);
        let mut vnonce = [0u8; 8];
        vnonce.copy_from_slice(&key[16..24]);
        drop(keys);
        let timing = self
            .last_timing
            .lock()
            .get(head)
            .copied()
            .flatten()
            .ok_or(ENODEV)?;
        let h = head as u16;
        // The sealed records share the video channel's running block counter:
        // #2 seq0(+1), #3 seq1(+1), #8 seq2(+69), and #9 seq71.
        let mut seal_seq: u32 = 0;
        let mut buf = KVec::with_capacity(2560, GFP_KERNEL)?;
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
                        super::cp::seal_video_arm(&vkey, &vnonce, sub, aux, seal_seq, &content)?;
                    seal_seq += 1;
                    buf.extend_from_slice(&frame, GFP_KERNEL)?;
                }
                6 | 7 => {
                    // type=4 but FIXED plaintext (not encrypted, no MAC): a 32-byte frame whose
                    // Its 16-byte body is fixed, with 0x10 at byte 11.
                    let mut f = [0u8; 32];
                    f[2..4].copy_from_slice(&0x001cu16.to_le_bytes());
                    f[4..8].copy_from_slice(&4u32.to_le_bytes());
                    f[8..10].copy_from_slice(&sub.to_le_bytes());
                    f[10..12].copy_from_slice(&aux.to_le_bytes());
                    f[16..32].copy_from_slice(&[
                        0x0a, 0x00, 0x04, 0x00, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0, 0,
                    ]);
                    buf.extend_from_slice(&f, GFP_KERNEL)?;
                }
                8 | 9 => {
                    debug_assert_eq!(body_len, 1104);
                    let mut nonce = [0u8; 14];
                    super::rng::fill(&mut nonce);
                    let content = super::video_arm::build(timing.hactive, timing.vactive, &nonce)?;
                    debug_assert_eq!(content.len(), body_len);
                    let frame =
                        super::cp::seal_video_arm(&vkey, &vnonce, sub, aux, seal_seq, &content)?;
                    seal_seq += (body_len / 16) as u32;
                    buf.extend_from_slice(&frame, GFP_KERNEL)?;
                }
                _ => {
                    // wire_type==2 plaintext records (#0/#1/#4/#5).
                    let body = super::cp::video_arm_plaintext_body(i, h);
                    let frame = super::cp::video_arm_plain_frame(sub, &body);
                    buf.extend_from_slice(&frame, GFP_KERNEL)?;
                }
            }
        }
        Ok(buf)
    }

    /// Seal and send one interactive CP message, advance the session, and pass its paired reply
    /// to `consume`.
    ///
    /// `build(counter)` produces the inner message for the dock-echoed counter. The `cp_link`
    /// mutex serialises the complete EP02/EP84 transaction with the KMS worker and keepalive.
    /// Callers are sleepable; atomic callbacks queue commands instead of invoking this path.
    fn send_cp_reply<T>(
        &self,
        dev: &BoundInterface<'_>,
        id: u16,
        tag_reserved: usize,
        reserved_counter: Option<u16>,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
        consume: impl FnOnce(&[u8; 16], &[u8; 8], &[u8]) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.cp_link.lock();
        let Some(link) = (&mut *guard).as_mut() else {
            return Err(ENODEV);
        };
        // DLM normally uses the next wire-order counter. Its cold Navarro mode transaction is the
        // exception: per-head workers reserve counters before their writes interleave, so the
        // inner counter order differs from the monotonically advancing AES block sequence.
        let request_counter = reserved_counter.unwrap_or(link.counter);
        let msg = build(request_counter)?;
        let inner_sub = if msg.len() >= 4 {
            u16::from_le_bytes([msg[2], msg[3]])
        } else {
            0
        };
        let content = &msg[..msg.len().saturating_sub(tag_reserved)];
        let frame = super::cp::seal_interactive(&link.ks, &link.riv, id, link.wire_seq, content)?;
        dev.ctrl_send(&frame, super::timeout(), GFP_KERNEL)?;
        link.wire_seq = link
            .wire_seq
            .wrapping_add(((content.len() + 15) / 16) as u32);
        // A normal message consumes the next counter now. A reserved message consumed its counter
        // when its logical worker queued the transaction, before independently queued EP02 writes
        // interleaved; consuming it twice here would skip a value after the cold transaction.
        if reserved_counter.is_none() {
            link.counter = link.counter.wrapping_add(1);
        }
        // DLM keeps reading EP84 until it sees the reply whose inner counter echoes this request.
        // Navarro also emits unprompted `id=2/sub=0x86` status pushes on the same endpoint; treating
        // the first such push as the paired reply advances EP02 before the dock has completed the
        // transaction.  The dock then NAKs that write until the real reply is reaped, which is the
        // exact 100-ms staircase visible in the failed captures.  Consume pushes here and stop only
        // at the echoed counter (or a bounded timeout for request classes which do not reply).
        //
        //
        // Use the validated 4096-byte request size so larger logical replies arrive intact.
        let mut reply = KVec::from_elem(0u8, 4096, GFP_KERNEL)?;
        let deadline = Instant::<Monotonic>::now() + Delta::from_millis(64);
        let mut matched = 0usize;
        let (mut reaped, mut undecodable) = (0u32, 0u32);
        let (mut seen_id, mut seen_sub, mut seen_counter) = (0u16, 0u16, 0u16);
        loop {
            let got = if let Some(q) = link.ep84_q.as_mut() {
                match q.recv(dev.io(), &mut reply, super::cp_reply_timeout()) {
                    Ok(Some(n)) => n,
                    Ok(None) => 0,
                    Err(_) => break,
                }
            } else {
                dev.ctrl_recv(&mut reply, super::cp_reply_timeout(), GFP_KERNEL)
                    .unwrap_or(0)
            };
            if got > 16 {
                // During re-engagement an EDID push can precede the paired acknowledgment. Keep
                // it for the waiting head while continuing to wait for the echoed counter.
                let target = self.edid_target.load(Ordering::Relaxed);
                if target != NO_EDID_TARGET {
                    if let Ok(Some(blob)) =
                        super::cp::parse_edid_from_reply(&link.ks, &link.riv, &reply[..got])
                    {
                        *self.edid_caught.lock() = Some(blob);
                    }
                }
                if let Some((reply_id, reply_sub, reply_counter)) =
                    super::cp::decode_in_lenient(&link.ks, &link.riv, &reply[..got])
                {
                    if reply_counter == request_counter {
                        matched = got;
                        break;
                    }
                    seen_id = reply_id;
                    seen_sub = reply_sub;
                    seen_counter = reply_counter;
                    if matches!(reply_id, 0x44 | 0x194) {
                        self.downstream_event.store(true, Ordering::Release);
                    }
                } else {
                    undecodable += 1;
                }
                reaped += 1;
            }
            if (Instant::<Monotonic>::now() - deadline).as_millis() >= 0 {
                break;
            }
        }
        // Name the message that went unanswered, and say whether the dock was silent or merely
        // unreadable. Without this a stalled control session can only be reported as ETIMEDOUT,
        // which cannot distinguish "the dock sent nothing" from "the dock replied and vino could
        // not decode it" -- and on the D6000 the wire shows 50 sealed replies arriving during an
        // attempt that ends in ETIMEDOUT.
        if matched == 0 {
            // Name the *inner* sub as well as the wire id. "id=0x16 went unanswered" covers the
            // EDID engage, the readiness kick and both stream/display markers, and which one the
            // dock ignored is the whole diagnosis.
            pr_info!(
                "vino: unanswered id={id:#06x} sub={inner_sub:#06x} ctr={request_counter}: reaped {reaped} reply/replies, {undecodable} undecodable, last decoded id={seen_id:#06x} sub={seen_sub:#06x} ctr={seen_counter}\n"
            );
        }
        consume(&link.ks, &link.riv, &reply[..matched])
    }

    /// Seal and send one interactive CP message on EP02, advancing the session keystream.
    pub(super) fn send_cp(
        &self,
        dev: &BoundInterface<'_>,
        id: u16,
        tag_reserved: usize,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
    ) -> Result {
        self.send_cp_reply(dev, id, tag_reserved, None, build, |_, _, _| Ok(()))
    }

    /// Send one CP message using a counter token previously consumed from the live allocator.
    fn send_cp_reserved(
        &self,
        dev: &BoundInterface<'_>,
        id: u16,
        inner_counter: u16,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
    ) -> Result {
        self.send_cp_reply(
            dev,
            id,
            0,
            Some(inner_counter),
            build,
            |_, _, _| Ok(()),
        )
    }

    /// Consume `N` consecutive counters from the live session and return them as reservation
    /// tokens. This models DLM's independently queued per-head workers: allocation order remains
    /// monotonic even when their actual EP02 writes interleave in a different order.
    fn reserve_cp_counters<const N: usize>(&self) -> Result<[u16; N]> {
        let mut guard = self.cp_link.lock();
        let Some(link) = (&mut *guard).as_mut() else {
            return Err(ENODEV);
        };
        let mut counters = [0u16; N];
        for counter in &mut counters {
            *counter = link.counter;
            link.counter = link.counter.wrapping_add(1);
        }
        Ok(counters)
    }

    /// Consume the dock's *unprompted* EP84 pushes, i.e. reads that are not the reply to any of our
    /// writes. Returns how many frames were drained.
    ///
    /// The dock also emits capability and heartbeat frames without a paired request. The bounded
    /// zero-timeout loop consumes those pushes without delaying keepalive or allowing a chatty dock
    /// to monopolise the worker.
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
                // Every queue slot is already posted. One millisecond is enough to reap a
                // completion without turning this best-effort reader into another control
                // deadline. A zero-jiffy completion wait does not observe an already-signalled
                // completion reliably, so it left pushes queued until the next EP02 write.
                Some(q) => q.recv(dev.io(), &mut reply, Delta::from_millis(1)),
                None => dev
                    .ctrl_recv(&mut reply, Delta::from_millis(1), GFP_KERNEL)
                    .map(Some),
            };
            // `Ok(None)` is the queue's timeout: nothing pending, so the dock has nothing more to
            // say right now. Any error is treated the same -- this is best-effort drainage.
            match got {
                Ok(Some(len)) if len > 0 => {
                    n += 1;
                    if let Some((id, _, _)) =
                        super::cp::decode_in_lenient(&link.ks, &link.riv, &reply[..len])
                    {
                        // An EDID-handler reply arriving with no probe outstanding is the dock
                        // reporting that a downstream sink changed -- it is the *only* thing it
                        // sent between a measured monitor replug and its own give-up reset. Treat
                        // it as "re-probe now" rather than waiting out the presence period.
                        if matches!(id, 0x44 | 0x194) {
                            self.downstream_event.store(true, Ordering::Release);
                        }
                    }
                }
                _ => break,
            }
        }
        n
    }

    /// Cache a head's downstream EDID (read during probe). Bring-up publishes all heads with one
    /// hotplug only after both presence and EDID state are complete; firing here exposed KWin to a
    /// transient no-EDID mode list (including synthetic 1920x1440) before the real EDID arrived.
    /// Out-of-range heads are ignored.
    pub(super) fn set_edid(&self, head: usize, blob: KVec<u8>) {
        let mut edids = self.cached_edids.lock();
        let Some(slot) = edids.get_mut(head) else {
            return;
        };
        *slot = Some(blob);
    }

    /// Mark a head connected from CP engagement alone (no raw EDID). Bring-up fires one hotplug
    /// after every head's EDID has also been cached, so the compositor never probes partial state.
    /// Called once the head's DISPLAY-CAP push confirms monitor presence.
    pub(super) fn set_connected(&self, head: usize) {
        if head >= HEADS {
            return;
        }
        self.heads_present
            .fetch_or(1 << head, core::sync::atomic::Ordering::Release);
    }

    /// Clear a head's presence bit and cached EDID after monitor removal.
    ///
    /// `detect()` reports connected when either exists, so both must be cleared together.
    pub(super) fn set_disconnected(&self, head: usize) {
        if head >= HEADS {
            return;
        }
        // Tagged with the dock's connector count (Navarro 4, Ridge 2). With both docks bound only
        // one ever ends up with a `connected` DRM connector, and vino logs
        // `head 0 monitor connected` for the loser on a path that does call `set_connected` --
        // so something clears it afterwards and this says which dock, and when.
        if self.heads_present.load(Ordering::Acquire) & (1u32 << head) != 0 {
            pr_info!(
                "vino: [{}conn] head {head} presence CLEARED (connector goes disconnected)\n",
                self.connector_count()
            );
        }
        self.heads_present
            .fetch_and(!(1u32 << head), core::sync::atomic::Ordering::Release);
        if let Some(slot) = self.cached_edids.lock().get_mut(head) {
            *slot = None;
        }
    }

    fn send_reengage_step(
        &self,
        io: &BoundInterface<'_>,
        id: u16,
        gap_ms: i64,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
    ) -> Result {
        self.send_cp(io, id, 0, build)?;
        fsleep(Delta::from_millis(gap_ms));
        Ok(())
    }

    /// Re-run one head's EDID probe, fetch, engage, and capability query on the live CP link.
    ///
    /// Monitor removal tears down the dock's downstream sink, so a replug requires the engage
    /// messages before a later mode set can start its pixel clock. The cached EDID is cleared on
    /// removal and must be repopulated here before reporting the connector as present. Returns true
    /// only when a valid EDID was received.
    pub(super) fn reengage_head(&self, io: &BoundInterface<'_>, head: u8) -> Result<bool> {
        self.set_self_blanked(head as usize, false);
        self.edid_target.store(head as u32, Ordering::Release);
        *self.edid_caught.lock() = None;
        let result = (|| -> Result {
            self.send_reengage_step(io, 0x15, 117, |c| {
                super::cp::get_edid_req_sub(c, 0x0020, head)
            })?;
            self.send_reengage_step(io, 0x15, 115, |c| {
                super::cp::get_edid_req_sub(c, 0x0020, head)
            })?;
            self.send_reengage_step(io, 0x16, 107, |c| super::cp::edid_readiness_kick(c, head))?;
            self.send_reengage_step(io, 0x15, 11, |c| super::cp::get_edid_req(c, head))?;
            self.send_reengage_step(io, 0x16, 118, |c| super::cp::edid_engage_req(c, head))?;
            self.send_reengage_step(io, 0x16, 107, |c| super::cp::edid_engage_req(c, head))?;
            self.send_reengage_step(io, 0x15, 11, |c| super::cp::post_edid_query(c, head))
        })();
        let caught = self.edid_caught.lock().take();
        self.edid_target.store(NO_EDID_TARGET, Ordering::Release);
        result?;
        match caught.or_else(|| self.drain_for_edid(io)) {
            Some(blob) => {
                let n = blob.len();
                // Say what the EDID claims to be, not just that one arrived. On unfamiliar
                // hardware this is what distinguishes a real monitor from a block the dock
                // synthesised for an empty port.
                if blob.len() >= 12 {
                    let m = u16::from_be_bytes([blob[8], blob[9]]);
                    let vendor = [
                        b'@' + ((m >> 10) & 0x1f) as u8,
                        b'@' + ((m >> 5) & 0x1f) as u8,
                        b'@' + (m & 0x1f) as u8,
                    ];
                    pr_info!(
                        "vino: head {head} EDID {n} B, vendor {}{}{} product {:#06x}\n",
                        vendor[0] as char,
                        vendor[1] as char,
                        vendor[2] as char,
                        u16::from_le_bytes([blob[10], blob[11]])
                    );
                }
                self.set_edid(head as usize, blob);
                Ok(true)
            }
            None => {
                vino_debug!(
                    "vino: head {head} re-engaged but no EDID came back -- no monitor, or it is \
                     not ready yet\n"
                );
                Ok(false)
            }
        }
    }

    /// Drain EP84 looking for the `id=0x194` EDID the fetch above asks for, and return it.
    ///
    /// The real EDID only ever arrives as that push (never inside `id=0x4c`/`0x78`), and it can
    /// land a few messages after the fetch, so this reads a bounded run of replies rather than just
    /// the next one. Bounded twice over -- attempt count and per-read timeout -- because it runs on
    /// the keepalive, which must not stall.
    fn drain_for_edid(&self, dev: &BoundInterface<'_>) -> Option<KVec<u8>> {
        let mut guard = self.cp_link.lock();
        let link = (&mut *guard).as_mut()?;
        let mut reply = KVec::from_elem(0u8, 4096, GFP_KERNEL).ok()?;
        for _ in 0..24 {
            let got = match link.ep84_q.as_mut() {
                Some(q) => match q.recv(dev.io(), &mut reply, Delta::from_millis(8)) {
                    Ok(Some(n)) if n > 16 => n,
                    _ => continue,
                },
                None => match dev.ctrl_recv(&mut reply, Delta::from_millis(8), GFP_KERNEL) {
                    Ok(n) if n > 16 => n,
                    _ => continue,
                },
            };
            if let Ok(Some(blob)) =
                super::cp::parse_edid_from_reply(&link.ks, &link.riv, &reply[..got])
            {
                return Some(blob);
            }
        }
        None
    }

    /// Stage 2 (runtime monitor hotplug): probe one physical connector.
    ///
    /// Navarro multiplexes two connectors per bulk endpoint, but its EDID selector and stream
    /// record subfield are still per socket. Never collapse sockets 0/2 or 1/3 here: doing so
    /// turns two independently connected monitors into one KMS connector.
    pub(super) fn probe_head_present(&self, dev: &BoundInterface<'_>, head: u8) -> Option<bool> {
        if usize::from(head) >= self.connector_count() {
            return Some(false);
        }
        self.probe_connector_present(dev, head, head)
    }

    /// Probe one downstream connector. `head` selects its own presence-change cell.
    ///
    /// Sends the EDID probe (`id=0x15 sub=0x20`, byte22 = connector selector -- the same selector
    /// that unblocked the whole EDID path) and decodes the dock's sealed `0x45` reply. Returns
    /// `Some(true/false)` on a decodable reply, `None` if CP is down or nothing decoded. Reuses the
    /// live session `ks/riv/counter` exactly like `send_cp`, so it stays in CP lockstep.
    fn probe_connector_present(&self, dev: &BoundInterface<'_>, sel: u8, head: u8) -> Option<bool> {
        let mut guard = self.cp_link.lock();
        let link = (&mut *guard).as_mut()?;
        let request_counter = link.counter;
        let msg = super::cp::get_edid_req_sub(request_counter, 0x0020, sel).ok()?;
        let frame =
            super::cp::seal_interactive(&link.ks, &link.riv, 0x15, link.wire_seq, &msg).ok()?;
        dev.ctrl_send(&frame, super::timeout(), GFP_KERNEL).ok()?;
        link.wire_seq = link.wire_seq.wrapping_add(((msg.len() + 15) / 16) as u32);
        link.counter = link.counter.wrapping_add(1);
        // Take the reply that answers this probe, not simply the next frame on EP84: the
        // connectors are probed back to back, so a late reply or an unprompted push would
        // otherwise be attributed to the wrong head. The inner counter echoes the request. A round
        // that never sees its own echo returns `None`, which the caller treats as "this poll
        // learned nothing" rather than as an unplug.
        let mut reply = KVec::from_elem(0u8, 4096, GFP_KERNEL).ok()?;
        let deadline = Instant::<Monotonic>::now() + Delta::from_millis(64);
        let got = loop {
            let n = match link.ep84_q.as_mut() {
                Some(q) => match q.recv(dev.io(), &mut reply, super::cp_reply_timeout()) {
                    Ok(Some(n)) => n,
                    Ok(None) => 0,
                    Err(_) => return None,
                },
                None => dev
                    .ctrl_recv(&mut reply, super::cp_reply_timeout(), GFP_KERNEL)
                    .unwrap_or(0),
            };
            if n > 16 {
                match super::cp::decode_in_lenient(
                    &link.ks,
                    &link.riv,
                    &reply[..n],
                ) {
                    Some((_, _, echoed)) if echoed == request_counter => break n,
                    // Undecodable frames are the dock's asynchronous pushes; keep draining.
                    _ => {}
                }
            }
            if (Instant::<Monotonic>::now() - deadline).as_millis() >= 0 {
                return None;
            }
        };
        // Decode the downstream status at inner bytes 22..26 as well as the handler ID.
        let (id, status, _) = super::cp::probe_reply_status(&link.ks, &link.riv, &reply[..got])?;
        // Presence is bit 0x10 of inner byte 23, which lands in bits 8..15 of the status word:
        // `05 11 27 00` for an occupied connector, `05 01 <20|21|60|61> 00` for an empty one.
        // Which handler answered says nothing about it -- both docks reply `id=0x44` either way.
        let present = status & 0x0000_1000 != 0;
        // One line per *changed* answer per head, so a steady link is silent and an unplug is
        // unmissable. Both fields are packed into the same cell: the id alone cannot distinguish a
        // dock that keeps saying `0x44` from one whose downstream state has moved underneath it.
        let cell = ((id as u32) << 16) | (status & 0xffff);
        let prev = self.presence_reply[head as usize].swap(cell, Ordering::Relaxed);
        if prev != cell {
            // The dock moves the *other* head's status word too when a sink appears or disappears,
            // and it does so sooner than it pushes anything. `prev == 0` is this head's first ever
            // reply, which is bring-up, not an event.
            if prev != 0 {
                self.downstream_event.store(true, Ordering::Release);
            }
            // The decoded answer itself, not just the verdict derived from it. Without this a
            // presence flap can only be read as "monitor disconnected", which says nothing about
            // whether the dock changed its mind or vino changed the question. It is one line per
            // *changed* reply per head, so a steady link prints nothing at all.
            // Tagged with the dock's connector count -- Navarro exposes four, Ridge two -- because
            // an untagged line cannot be attributed when both docks are bound, and reading these
            // as the wrong dock's is exactly how this session lost time.
            pr_info!(
                "vino: [{}conn] head {head} presence reply id={id:#06x} status={status:#010x} \
                 -> present={present} (was id={:#06x} status={:#06x})\n",
                self.connector_count(),
                prev >> 16,
                prev & 0xffff
            );
        }
        Some(present)
    }

    /// Whether vino itself took `head`'s sink down, so the presence watcher can tell its own
    /// blank apart from a real unplug. See [`VinoDrmData::self_blanked`].
    pub(super) fn is_self_blanked(&self, head: usize) -> bool {
        self.self_blanked.load(Ordering::Acquire) & (1u32 << head) != 0
    }

    fn set_self_blanked(&self, head: usize, on: bool) {
        if on {
            self.self_blanked.fetch_or(1u32 << head, Ordering::Release);
        } else {
            self.self_blanked
                .fetch_and(!(1u32 << head), Ordering::Release);
        }
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

    /// The dock's total pixel-rate budget shared across all heads.
    ///
    /// Zero means unknown and disables limiting.
    fn dock_budget(&self) -> u32 {
        self.dock_pixel_budget
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Record this dock's pixel-rate budget, refresh ceiling and pixel-clock ceiling.
    pub(super) fn set_mode_limits(
        &self,
        pixel_budget: u32,
        max_refresh_hz: u32,
        max_head_clock_khz: u32,
    ) {
        self.dock_pixel_budget
            .store(pixel_budget, core::sync::atomic::Ordering::Relaxed);
        self.max_refresh_hz.store(
            if max_refresh_hz == 0 {
                DEFAULT_MAX_REFRESH_HZ
            } else {
                max_refresh_hz
            },
            core::sync::atomic::Ordering::Relaxed,
        );
        self.max_head_clock_khz.store(
            if max_head_clock_khz == 0 {
                DEFAULT_MAX_HEAD_CLOCK_KHZ
            } else {
                max_head_clock_khz
            },
            core::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Highest per-mode pixel clock in kHz this dock is known to accept.
    fn max_head_clock_khz(&self) -> u32 {
        self.max_head_clock_khz
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Highest refresh rate this dock is known to drive.
    pub(super) fn max_refresh_hz(&self) -> u32 {
        self.max_refresh_hz.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Whether DRM's rounded refresh rate is within this dock's limit.
    pub(super) fn refresh_within_limit(&self, vrefresh: i32) -> bool {
        vrefresh <= 0 || (vrefresh as u32) <= self.max_refresh_hz()
    }

    /// Combined pixel rate of every head *except* `head` that currently has a mode driven onto it.
    ///
    /// Only active heads consume the shared limit. `last_timing` survives
    /// `atomic_disable`, so activity is read from `modeset_requested`, which
    /// is cleared on disable.
    fn other_heads_rate(&self, head: usize) -> u32 {
        let timings = *self.last_timing.lock();
        let mut total: u32 = 0;
        for (i, t) in timings.iter().enumerate() {
            if i == head || self.modeset_requested[i].load(Ordering::Acquire) == 0 {
                continue;
            }
            if let Some(t) = t {
                total = total.saturating_add(
                    u32::from(t.hactive)
                        .saturating_mul(u32::from(t.vactive))
                        .saturating_mul(u32::from(t.refresh_hz)),
                );
            }
        }
        total
    }

    /// Publish the latest desired operation for a head and wake the async worker.
    ///
    /// Each operation class has one fixed slot, so updates cannot fail allocation and obsolete
    /// cursor positions or stream states do not build a backlog.
    fn queue_cmd(&self, dev: &VinoDrmDevice, cmd: KmsCmd) {
        let mut pending = self.pending_kms.lock();
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        pending.update(cmd);
        // Enqueue while the queue lock still serializes us with `shutdown()`. Otherwise shutdown
        // could cancel an idle work item between this unlock and enqueue, leaving a late work-owned
        // device reference behind after teardown.
        //
        // `::<_, 0>` names `cmd_work`. The ID is only inferrable while a single `WorkItem` impl
        // exists; adding the per-head scanout items made every bare `enqueue` ambiguous, which is
        // exactly the failure mode you want here -- an unannotated enqueue would otherwise be free
        // to pick the wrong worker.
        let _ = self.kms_queue.enqueue::<_, 0>(ARef::from(dev));
        drop(pending);
    }

    /// Publish the latest framebuffer for one head and wake the same deferred worker used by the
    /// blocking runtime CP commands. Replacing an unsent flip is deliberate backpressure: the dock
    /// needs the newest desktop, not every historical compositor buffer. If damaged flips are
    /// coalesced, carry the unsent damage into the newest framebuffer so no intermediate update is
    /// lost without needlessly promoting every busy compositor interval to a full-screen refresh.
    fn queue_scanout(
        &self,
        dev: &VinoDrmDevice,
        fb: &kms::framebuffer::Framebuffer<VinoDrmDriver>,
        mut frame: PendingScanout,
    ) {
        let head = frame.head as usize;
        if head >= HEADS || self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let (source_w, source_h) = src_dims(frame.rotation, frame.w, frame.h);
        // Reserve a slot and lend its surface out, so the ~14.7 MB copy below runs with the pool
        // lock dropped. Holding it across the copy put this head's scanout worker into
        // `mutex_spin_on_owner` for the whole snapshot -- 5.4% of the machine, burnt spinning.
        let (mut surface, binding, idx) = {
            let mut pool = self.shadow[head].lock();
            // Rotate, rather than always taking the first free slot. `find` returned slot 0 on
            // every commit whenever nothing was inflight, so consecutive snapshots overwrote the
            // same slot and bumped its generation -- invalidating any frame the worker had already
            // selected from it, which showed up as a third of all frames being dropped at the
            // generation check. Alternating means a fresh snapshot lands clear of the frame the
            // worker is about to pick up.
            let start = self.shadow_rr[head].fetch_add(1, Ordering::Relaxed) as usize;
            let Some(idx) = (0..SHADOW_SLOTS)
                .map(|i| (start + i) % SHADOW_SLOTS)
                .find(|&idx| pool.inflight != Some(idx) && pool.writing != Some(idx))
            else {
                return;
            };
            let binding = match pool.source_bindings.get(fb) {
                Ok(binding) => binding,
                Err(e) => {
                    pr_warn!("vino: head {head} framebuffer binding failed ({e:?})\n");
                    return;
                }
            };
            pool.writing = Some(idx);
            (pool.slots[idx].surface.take(), binding, idx)
        };

        let r = snapshot_to_shadow(
            self.geometry(),
            &mut surface,
            &binding.mapping,
            source_w,
            source_h,
        );

        let snapshot = {
            let mut pool = self.shadow[head].lock();
            pool.writing = None;
            let slot = &mut pool.slots[idx];
            slot.surface = surface;
            // Bump unconditionally: the slot's contents have been rewritten either way, so any
            // frame still pointing at the old generation must not be encoded from it.
            slot.generation = slot.generation.wrapping_add(1);
            r.map(|()| (idx, slot.generation))
        };
        let (idx, generation) = match snapshot {
            Ok(snapshot) => snapshot,
            Err(e) => {
                pr_warn!("vino: head {head} framebuffer snapshot failed ({e:?})\n");
                return;
            }
        };
        frame.shadow_idx = idx;
        frame.shadow_generation = generation;

        // A real flip carries newer content than an armed repaint.
        self.settle_repaint.lock()[head] = None;

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
        self.enqueue_scanout(dev, head);
        drop(pending);
    }

    /// Wake `head`'s scanout worker. The work ID is a const generic, so the runtime head index has
    /// to be matched into it here. Enqueueing an already-pending item is a no-op, and enqueueing
    /// one that is currently running re-arms it, preserving a flip that arrives during
    /// encoding for the worker's next pass.
    fn enqueue_scanout(&self, dev: &VinoDrmDevice, head: usize) {
        match head {
            0 => {
                let _ = self.scanout_queue.enqueue::<_, 1>(ARef::from(dev));
            }
            1 => {
                let _ = self.scanout_queue.enqueue::<_, 2>(ARef::from(dev));
            }
            2 => {
                let _ = self.scanout_queue.enqueue::<_, 3>(ARef::from(dev));
            }
            3 => {
                let _ = self.scanout_queue.enqueue::<_, 4>(ARef::from(dev));
            }
            _ => {}
        }
    }

    /// Wait for any frame already in flight on a scanout worker to finish, after [`Self::cmd_busy`]
    /// has been published. A worker that has not yet started re-checks `cmd_busy` and backs off on
    /// its own; this only covers one that got past that check before the flag was set.
    ///
    /// Bounded, and it proceeds anyway on timeout: a mode-set that never reaches the dock is worse
    /// than one that races a frame, and this is the path cold activation depends on. The bound is
    /// generous against a worst-case frame (a ~3.19 MB keyframe: ~21 ms to encode plus its wire
    /// time), so exceeding it means something is genuinely wedged and the log line is the point.
    fn wait_for_video_idle(&self) {
        use core::sync::atomic::Ordering::SeqCst;
        for _ in 0..500 {
            if !self.video_inflight.iter().any(|f| f.load(SeqCst)) {
                return;
            }
            fsleep(Delta::from_millis(1));
        }
        pr_warn!("vino: timed out waiting for in-flight scanout before a mode-set; proceeding\n");
    }

    /// Wake every head's scanout worker. Used by `cmd_work` once its batch is done, since a command
    /// batch is exactly what makes the scanout workers bail (see [`run_scanout_worker`]).
    fn enqueue_scanout_all(&self, dev: &VinoDrmDevice) {
        for head in 0..HEADS {
            self.enqueue_scanout(dev, head);
        }
    }

    /// Record that `head` owes a full keyframe, and refill its settle-repaint budget.
    ///
    /// Mode sets, output enables, and gamma changes use this path. Training
    /// and settle repaints may re-raise the keyframe bit without refilling
    /// the budget, which bounds idle keyframe generation.
    fn owe_keyframe(&self, head: usize) {
        self.keyframe_pending
            .fetch_or(1u32 << head, Ordering::Release);
        self.settle_budget[head].store(SETTLE_REPAINTS, Ordering::Relaxed);
        // Whatever left the dock's framebuffer undefined left its cursor bitmap undefined too, so
        // the two invalidations are raised together. Keeping them in one place is deliberate:
        // their being separate is exactly how the cursor came to be dropped on a mode-set.
        self.cursor_epoch[head].fetch_add(1, Ordering::Release);
        self.cursor_geometry.lock()[head] = None;
    }

    /// Choose the next frame or delay for `head`.
    ///
    /// Neither means this head is idle and its worker can exit.
    fn select_scanout(&self, head: usize) -> (Option<PendingScanout>, Option<i64>) {
        // Keep a frame that arrived before the cadence deadline in the
        // coalescing slot. Userspace may stop committing after that flip, so
        // discarding it could leave the newest image unsent.
        let mut pending = self.pending_scanout.lock();
        let mut selected = None;
        let mut wait_us: Option<i64> = None;
        if self.modeset_requested[head].load(Ordering::Acquire) != 0 && pending[head].is_some() {
            let owes_keyframe = self.keyframe_pending.load(Ordering::Acquire) & (1u32 << head) != 0;
            let elapsed_us = self.last_frame.lock()[head]
                .map_or(FRAME_PERIOD_US, |t| t.elapsed().as_micros_ceil());
            if owes_keyframe || elapsed_us >= FRAME_PERIOD_US {
                selected = pending[head].take();
                // A busy compositor continuously replaces `settle_repaint`.
                // Force cadence-selected frames to be keyframes while training;
                // the elapsed check above still applies the cadence limit.
                let sustaining = self.sustain_until.lock()[head]
                    .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
                if sustaining {
                    self.keyframe_pending
                        .fetch_or(1u32 << head, Ordering::Release);
                }
            } else {
                wait_us = Some(FRAME_PERIOD_US - elapsed_us);
            }
        }
        // Nothing flipped in. Fall back to the one-shot settle repaint if one is due, so a
        // compositor that went idle straight after enabling the output still ends up with its real
        // desktop on the panel rather than the buffer that happened to be current when the
        // mode-set's keyframe went out.
        if selected.is_none() {
            let mut settle = self.settle_repaint.lock();
            if self.modeset_requested[head].load(Ordering::Acquire) == 0 {
                settle[head] = None;
            } else if let Some((due, _, _)) = settle[head].as_ref() {
                let remaining = *due - Instant::<Monotonic>::now();
                if remaining.as_millis() <= 0 {
                    let taken = settle[head].take();
                    let as_keyframe = taken.as_ref().is_some_and(|(_, _, kf)| *kf);
                    selected = taken.map(|(_, f, _)| f);
                    if as_keyframe {
                        self.keyframe_pending
                            .fetch_or(1u32 << head, Ordering::Release);
                    }
                    let kind = if as_keyframe {
                        "settle repaint (compositor idle after mode-set)"
                    } else {
                        "debt repaint (retransmissions owed, compositor idle)"
                    };
                    vino_debug!("vino: head {head} {kind}\n");
                } else {
                    let remaining = remaining.as_micros_ceil().max(1);
                    wait_us = Some(wait_us.map_or(remaining, |old| old.min(remaining)));
                }
            }
        }
        (selected, wait_us)
    }
}

impl_has_delayed_work! {
    impl HasDelayedWork<VinoDrmDevice> for VinoDrmData { self.cmd_work }
}

impl_has_work! {
    impl HasWork<VinoDrmDevice, 1> for VinoDrmData { self.scanout_work_h0 }
    impl HasWork<VinoDrmDevice, 2> for VinoDrmData { self.scanout_work_h1 }
    impl HasWork<VinoDrmDevice, 3> for VinoDrmData { self.scanout_work_h2 }
    impl HasWork<VinoDrmDevice, 4> for VinoDrmData { self.scanout_work_h3 }
}

/// One scanout work item exists per connector, and its work ID is a const generic. Keep this
/// assertion adjacent to the explicit fields/arms below: adding another connector without all
/// three would silently leave its frames in `pending_scanout`.
const _: () = assert!(
    HEADS == 4,
    "add a scanout_work_hN work item per head (see VinoDrmData::enqueue_scanout)"
);

impl WorkItem<1> for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;
    fn run(this: ARef<VinoDrmDevice>) {
        run_scanout_worker(this, 0);
    }
}

impl WorkItem<2> for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;
    fn run(this: ARef<VinoDrmDevice>) {
        run_scanout_worker(this, 1);
    }
}

impl WorkItem<3> for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;
    fn run(this: ARef<VinoDrmDevice>) {
        run_scanout_worker(this, 2);
    }
}

impl WorkItem<4> for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;
    fn run(this: ARef<VinoDrmDevice>) {
        run_scanout_worker(this, 3);
    }
}

/// One head's deferred scanout loop: pick this head's due frame, encode it, transmit it, repeat
/// until the head has nothing left to do. Both heads run concurrently on the per-device scanout
/// queue.
///
/// A queued `ModeSet` must reach the dock before video for that head, so this worker returns while
/// a stream command is pending or executing. `cmd_work` re-enqueues the scanout workers when the
/// command batch completes, and the pending framebuffer remains in its coalescing slot.
///
/// Two conditions, and both are needed: a stream operation in `pending_kms` (not yet drained), and
/// [`VinoDrmData::cmd_busy`] (drained and executing -- the window in which `pending_kms` is
/// misleadingly empty). The `video_inflight` store must be published *before* reading `cmd_busy`,
/// and both use `SeqCst`, so this and `wait_for_video_idle` cannot both conclude the other is idle.
fn run_scanout_worker(this: ARef<VinoDrmDevice>, head: usize) {
    use core::sync::atomic::Ordering::SeqCst;
    let data: &VinoDrmData = &this;
    // As in `cmd_work`: once the I/O window refuses a token, unplug has begun and there is no USB
    // left to do. `drm_dev_enter()` holds the parent interface Bound for the duration.
    let Ok(link) = super::UsbLink::open(&data.io, data.eps) else {
        return;
    };
    let dev = &link;
    loop {
        if data.shutting_down.load(Ordering::Acquire) {
            return;
        }
        // Claim the head's video endpoint first, then look for a reason not to use it.
        data.video_inflight[head].store(true, SeqCst);
        let blocked = data.cmd_busy.load(SeqCst) || data.pending_kms.lock().has_stream();
        if blocked {
            data.video_inflight[head].store(false, SeqCst);
            return;
        }
        let (frame, cadence_wait_us) = data.select_scanout(head);
        if let Some(frame) = frame {
            run_pending_scanout(dev, data, frame);
            data.video_inflight[head].store(false, SeqCst);
            continue;
        }
        data.video_inflight[head].store(false, SeqCst);
        if let Some(us) = cadence_wait_us {
            // Bound the sleep. The settle-repaint arm can ask for its full deadline
            // ([`SETTLE_REPAINT_MS`], 1.2 s); sleeping that long inside the work item makes the
            // head unreachable, because a flip arriving meanwhile finds the item already running
            // and its enqueue is dropped. Waking at the cadence window instead costs a few extra
            // wakeups while idle and keeps the head responsive to real frames.
            let us = us.min(FRAME_PERIOD_US);
            fsleep(Delta::from_micros(us));
            continue;
        }
        // Re-check before exiting. A frame published between `select_scanout` above and this point
        // finds the work item still running, so its `enqueue_scanout` is dropped and the frame
        // waits for some *later* flip to enqueue successfully -- a lost wakeup that showed up as
        // multi-second stalls on whichever head lost the race. The condition mirrors
        // `select_scanout`'s own guard so a head with no mode-set cannot spin here.
        if data.modeset_requested[head].load(Ordering::Acquire) != 0
            && data.pending_scanout.lock()[head].is_some()
        {
            continue;
        }
        return;
    }
}

impl WorkItem for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;

    /// Reconcile the latest desired stream and cursor state from the atomic callbacks.
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
        loop {
            if data.shutting_down.load(Ordering::Acquire) {
                return;
            }
            // A dual-head atomic commit runs `atomic_enable` once per head, and each of those
            // queues its own `ModeSet` and wakes this worker -- microseconds apart, but far less
            // than it takes to get scheduled. Taking the first one alone turns one dock-wide wake
            // into two single-head activations and skips the cold choreography that arms the video
            // endpoints, so wait, briefly and boundedly, for the siblings the compositor has
            // already published a timing for.
            {
                let started = Instant::<Monotonic>::now();
                let present = data.heads_present.load(Ordering::Acquire);
                loop {
                    let queued = data.pending_kms.lock().heads.iter().enumerate().fold(
                        0u32,
                        |acc, (h, p)| {
                            if matches!(p.stream, Some(KmsCmd::ModeSet { .. })) {
                                acc | (1u32 << h)
                            } else {
                                acc
                            }
                        },
                    );
                    // Nothing to wait for until at least one mode set has landed, and nothing
                    // left to wait for once every head with a monitor is either already active
                    // or represented in this batch.
                    let outstanding = (0..HEADS).any(|h| {
                        present & (1u32 << h) != 0
                            && queued & (1u32 << h) == 0
                            && data.modeset_active[h].load(Ordering::Acquire) == 0
                    });
                    if queued == 0
                        || !outstanding
                        || (Instant::<Monotonic>::now() - started).as_millis()
                            >= MODESET_BATCH_SETTLE_MS
                    {
                        break;
                    }
                    fsleep(Delta::from_millis(1));
                }
            }
            let pending = core::mem::replace(&mut *data.pending_kms.lock(), PendingKms::new());
            // A cold dual-head atomic commit is one dock-wide wake: both mode-sets precede either
            // head's video. Detect that shape before
            // consuming the owned state.
            let mut dual_timings: [Option<super::cp::Timing>; HEADS] = [None; HEADS];
            let mut cmd_heads = 0u32;
            for head in &pending.heads {
                if let Some(KmsCmd::ModeSet {
                    head: cmd_head,
                    timing,
                }) = &head.stream
                {
                    let head_i = *cmd_head as usize;
                    if head_i < HEADS {
                        cmd_heads |= 1u32 << head_i;
                        if data.modeset_active[head_i].load(Ordering::Acquire) == 0
                            && data.modeset_requested[head_i].load(Ordering::Acquire)
                                == timing_key(timing)
                        {
                            dual_timings[head_i] = Some(*timing);
                        }
                    }
                }
            }
            // Exclude the scanout workers for exactly as long as this batch can touch a video
            // endpoint. `activate_dual_wake` and the `ModeSet` arm both run
            // `submit_prompt_training`, which writes the activation carrier to the head's endpoint;
            // a concurrent scanout frame there would interleave records on the wire and would have
            // its `video_q` slot double-opened. Cursor-only batches deliberately skip this: they
            // never touch video, and a mouse in motion produces a continuous stream of them.
            // `Blank` writes the head's video endpoint for the same reason `ModeSet` does, so it
            // needs the same exclusion against the scanout workers -- otherwise a frame already in
            // flight interleaves its records with the blanking frames on the wire.
            let has_modeset = pending.has_stream();
            if has_modeset {
                data.cmd_busy
                    .store(true, core::sync::atomic::Ordering::SeqCst);
                data.wait_for_video_idle();
            }
            let dual_wake = dual_timings.iter().flatten().count() >= 2;
            if has_modeset {
                vino_debug!(
                    "vino: KMS batch -- stream cmds {}, dual timings {}, dual_wake {}, requested [{} {} {} {}], active [{} {} {} {}]\n",
                    (0..HEADS).filter(|&h| cmd_heads & (1u32 << h) != 0).count(),
                    dual_timings.iter().flatten().count(),
                    dual_wake,
                    data.modeset_requested[0].load(Ordering::Acquire),
                    data.modeset_requested[1].load(Ordering::Acquire),
                    data.modeset_requested[2].load(Ordering::Acquire),
                    data.modeset_requested[3].load(Ordering::Acquire),
                    data.modeset_active[0].load(Ordering::Acquire),
                    data.modeset_active[1].load(Ordering::Acquire),
                    data.modeset_active[2].load(Ordering::Acquire),
                    data.modeset_active[3].load(Ordering::Acquire),
                );
            }
            let dual_complete = dual_wake
                && match data.activate_dual_wake(dev, dual_timings) {
                    Ok(done) => done,
                    Err(e) => {
                        pr_warn!("vino: dual-head activation failed ({e:?})\n");
                        false
                    }
                };
            let mut cmds: [Option<KmsCmd>; HEADS * 4] = [const { None }; HEADS * 4];
            for (head, pending) in pending.heads.into_iter().enumerate() {
                cmds[head] = pending.stream;
                cmds[HEADS + head] = pending.cursor_create;
                cmds[HEADS * 2 + head] = pending.cursor_image;
                cmds[HEADS * 3 + head] = pending.cursor_move;
            }
            // Control-plane ordering comes first. An enabling atomic commit queues the plane flip
            // before its CRTC mode-set. Finish the mode transaction before
            // selecting a pending framebuffer.
            let mut cmds = cmds.into_iter().flatten();
            let mut retry = false;
            while let Some(cmd) = cmds.next() {
                let res = match &cmd {
                    KmsCmd::ModeSet { head, timing } => {
                        if dual_complete {
                            // `activate_dual_wake` consumed the current generation for both heads.
                            // A superseding generation published while it ran remains in
                            // `pending_kms` for the next outer iteration.
                            continue;
                        }
                        let head_i = *head as usize;
                        let key = timing_key(timing);
                        if head_i >= HEADS
                            || data.modeset_requested[head_i].load(Ordering::Acquire) != key
                        {
                            Ok(()) // superseded or disabled while queued
                        } else {
                            data.activate_head(dev, *head, timing, key).map(|_| ())
                        }
                    }
                    KmsCmd::CursorCreate { head, w, h } => data.send_cp(dev, 0x1b, 0, |ctr| {
                        super::cp::cursor_create(ctr, *head, *w, *h)
                    }),
                    KmsCmd::CursorImage { head, w, h, bgra } => data.send_cp(dev, 0x1c, 0, |ctr| {
                        super::cp::cursor_image(ctr, *head, *w, *h, bgra)
                    }),
                    KmsCmd::CursorMove {
                        head,
                        x,
                        y,
                        visible,
                    } => data.send_cp(dev, 0x1a, 0, |ctr| {
                        super::cp::cursor_move(ctr, *head, *x, *y, *visible)
                    }),
                    KmsCmd::Blank { head } => data.blank_head(dev, *head),
                };
                if let Err(e) = res {
                    if !kms_error_retryable(e) {
                        pr_warn!("vino: dropping invalid asynchronous KMS command ({e:?})\n");
                        continue;
                    }

                    // Preserve the failed command and everything ordered behind it. Concurrent
                    // atomic callbacks may already have published newer state into these slots;
                    // `retry` never replaces that newer state with this drained batch.
                    let mut pending = data.pending_kms.lock();
                    pending.retry(cmd);
                    for cmd in cmds {
                        pending.retry(cmd);
                    }
                    retry = true;
                    vino_debug!("vino: asynchronous KMS command deferred after {e:?}\n");
                    break;
                }
            }
            if has_modeset {
                data.cmd_busy
                    .store(false, core::sync::atomic::Ordering::SeqCst);
            }

            if retry {
                if !data.shutting_down.load(Ordering::Acquire) {
                    let delay = kernel::time::msecs_to_jiffies(KMS_RETRY_MS);
                    let _ = workqueue::system().enqueue_delayed::<_, 0>(ARef::from(&*this), delay);
                }
                return;
            }
            if data.pending_kms.lock().is_empty() {
                break;
            }
        }
        // Wake both scanout workers after the command batch. They stop while
        // a queued mode set must reach the dock before video and resume here.
        data.enqueue_scanout_all(&this);
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
            preferred_fourcc: Some(drm::fourcc::XRGB8888),
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
                plane::Rotation::ROTATE_0,
                plane::Rotation::ROTATE_0
                    | plane::Rotation::ROTATE_90
                    | plane::Rotation::ROTATE_180
                    | plane::Rotation::ROTATE_270
                    | plane::Rotation::REFLECT_X
                    | plane::Rotation::REFLECT_Y,
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
            // An alpha framebuffer requires a blend-mode property. The dock composites the cursor
            // from a premultiplied bitmap, so premultiplied is the only supported mode.
            cursor.create_blend_mode_property(plane::BlendModes::PREMULTIPLIED)?;
            let crtc_obj = crtc::UnregisteredCrtc::<VinoCrtc>::new(
                dev,
                primary,
                Some(&cursor),
                None,
                head as u8,
            )?;
            // Advertise CTM and a 256-entry GAMMA_LUT; the scanout applies both (cached via the
            // CRTC hooks). The dock has no colour hardware, so software application here is the
            // only place a compositor's correction can land -- KDE's Night Colour and GNOME's
            // Night Light drive these properties rather than rewriting the framebuffer.
            crtc_obj.enable_color_mgmt(0, true, super::color::LUT_LEN as u32);
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
                // DisplayPort connectors receive DRM's standard EDID property. A virtual connector
                // would not, and therefore could not publish the downstream monitor's modes.
                connector::Type::DisplayPort,
                head as u8,
            )?;
            conn.attach_encoder(&*enc)?;
        }
        Ok(())
    }
}
