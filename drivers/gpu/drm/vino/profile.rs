// SPDX-License-Identifier: GPL-2.0

//! What distinguishes one DisplayLink dock from another.
//!
//! Most of it is data -- endpoints, strip geometry, connector count, link limits -- which the rest
//! of the driver reads rather than branches on. [`Generation`] names the one split that is genuine
//! code: Ridge and Navarro differ in their initialisation sequence, per-head HDCP framing, stream
//! open and mode description.

use super::*;

/// Which protocol generation a dock speaks.
///
/// Ridge and Navarro differ in more than parameter values: the initialisation sequence, the
/// per-head HDCP framing, how a video stream is opened and how a mode is described are each
/// distinct code paths. This names that split once. Everything that is merely a different *value*
/// -- endpoints, strip geometry, connector count, link limits -- stays a field on [`DockProfile`]
/// and is shared code driven by data.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Generation {
    /// DL-6xxx silicon: the Dell D6000.
    Ridge,
    /// DL-7000 silicon: the DL-7400 quad dock.
    Navarro,
}

/// What differs between the DisplayLink docks this driver drives.
///
/// The control plane is identical across them -- bulk OUT `0x02`, bulk IN `0x84`, the same HDCP
/// and CP sequence -- but the video endpoints are not, so they cannot be a global constant. The
/// D6000 exposes four video bulk-OUT endpoints and drives its two heads from `0x08` and `0x0b`;
/// the DL7400 exposes only two, `0x08` and `0x0a`, so naming `0x0b` there fails endpoint
/// resolution outright and the device never comes up.
pub(crate) struct DockProfile {
    /// Human name, logged at probe so an unfamiliar unit identifies itself in dmesg.
    pub(crate) name: &'static str,
    /// Video bulk-OUT endpoint per physical connector. Navarro deliberately repeats its two
    /// endpoint addresses: connectors 0/2 share 0x08 and connectors 1/3 share 0x0a.
    pub(crate) video_eps: [u8; drm_sink::HEADS],
    /// Which protocol generation this dock speaks; see [`Generation`].
    pub(crate) generation: Generation,
    /// How the dock encodes a head in a video record's `sub` field, as a left shift.
    ///
    /// Ridge uses the bare connector number (shift 0). Navarro spaces connectors eight apart --
    /// records use `0x00`/`0x08`/`0x10`/`0x18` and stream-opens `0x07`/`0x0f`/`0x17`/`0x1f`.
    pub(crate) head_sub_shift: u8,
    /// The bits a head's content-stream id sets over its record `sub`.
    ///
    /// Ridge streams are `0x08 | head`, Navarro's `(connector << 3) | 7`. See
    /// [`video::wht::Geometry::stream_id_mask`], which this configures.
    pub(crate) stream_id_mask: u8,
    /// The connector-count marker byte the per-head `strm2` record carries at offset 24.
    pub(crate) strm2_marker: u8,
    /// Whether an image record's `sub` carries the y-band parity; see
    /// [`video::wht::Geometry::band_parity_bit`].
    pub(crate) band_parity_bit: bool,
    /// Blocks across one strip; see [`video::wht::Geometry`]. Ridge lays a strip's sixteen
    /// blocks 8 across x 2 down (64x16 px), the DL7400 16 across x 1 down (128x8 px).
    pub(crate) strip_blocks_x: usize,
    /// Whether image records interlace y bands; see [`video::wht::Geometry::interlaced_bands`].
    pub(crate) interlaced_bands: bool,
    /// Number of downstream connectors the dock answers a presence probe for.
    ///
    /// This is the range of the selector at probe byte 22, and it is **not** the head count: Ridge
    /// has two of each, Navarro has four connectors feeding two video endpoints (`0x08` carried
    /// connectors 0 then 2, `0x0a` carried 1 then 3, measured across cable moves). Connector index
    /// is the physical socket number minus one.
    pub(crate) connectors: u8,
    /// How many buffers the dock rotates through as it presents frames.
    ///
    /// Ridge is double buffered. The DL7400 rotates three slots -- `video::wht::ring_phase()`
    /// steps `seq0 % 3` and its pipe descriptor names three ring addresses. This drives both the
    /// keyframe presentation count and the per-strip retransmit debt, so getting it wrong leaves
    /// one slot holding stale pixels and the panel ghosts on anything detailed.
    pub(crate) dock_buffers: u8,
    /// Highest refresh rate this dock is known to drive, or `u32::MAX` for a dock that has no
    /// refresh limit beyond its link rate.
    ///
    /// This is a *rate* limit, not a bandwidth one -- `max_head_clock_khz` and `pixel_budget` carry
    /// bandwidth. DLM clamps Ridge here regardless of resolution: asked for 2560x1440@180 it puts
    /// 119.998 Hz on the wire, and asked for @85 it programs the 59.95 Hz CVT-RB timing. It applies
    /// no such clamp to the DL7400, which it drives at 2560x1440@164.96 on both heads, so nothing
    /// justifies capping that dock by refresh alone.
    pub(crate) max_refresh_hz: u32,
    /// Highest per-mode pixel clock in kHz this dock is known to carry.
    ///
    /// This is the constraint that actually bounds a mode: the DL7400 accepts 2560x1440@180 and
    /// then fails to deliver it, and what separates that mode from the 165 Hz one it does drive is
    /// 714.81 MHz against 699.50 MHz of link rate, not 180 against 165 of refresh.
    ///
    /// The set-mode message carries the clock at offsets 70..73 as a `u32` in 10 kHz units. Ridge
    /// is never driven above 497.75 MHz, so its captures only ever fill the low half. Take the
    /// ceiling from the mode's own clock rather than from DLM's copy of it, which is rounded to
    /// the wire's 10 kHz unit (`0x0001113d`, 699.49 MHz, for a 699.50 MHz mode).
    pub(crate) max_head_clock_khz: u32,
    /// Dock-wide pixel-rate budget in pixels per second, shared across all heads.
    ///
    /// Ridge's is DLM's declared `pixel_per_second_limit` for both heads. The DL7400's is the
    /// dual-head rate DLM was measured sustaining.
    pub(crate) pixel_budget: u32,
    /// Outstanding EP84 reads to keep posted.
    ///
    /// Navarro needs exactly one, as DLM keeps: a deeper queue delays an EDID reply behind an
    /// un-reaped slot and the dock then NAKs EP02. Ridge interleaves many more unsolicited pushes
    /// with the replies it waits for, and loses them at a depth of one.
    pub(crate) ep84_queue_depth: usize,
}

impl DockProfile {
    /// Whether this dock speaks the Navarro protocol.
    pub(crate) fn is_navarro(&self) -> bool {
        matches!(self.generation, Generation::Navarro)
    }

    /// Whether per-head HDCP records select a connector as a one-hot bit at byte `22 + head`.
    /// Ridge instead has a one-based head number at byte 23.
    pub(crate) fn perhead_onehot(&self) -> bool {
        self.is_navarro()
    }

    /// This dock's codec geometry, for the codec calls made before a DRM device exists.
    ///
    /// The steady-state path reads `VinoDrmData::geometry()` instead; both describe the same
    /// dock, and this exists because CP setup names stream ids before the sink is published.
    pub(crate) fn geometry(&self) -> video::wht::Geometry {
        video::wht::Geometry::new(
            self.strip_blocks_x,
            self.interlaced_bands,
            self.band_parity_bit,
            self.head_sub_shift,
            self.stream_id_mask,
            self.dock_buffers,
        )
    }
}

/// Dell D6000 and other Ridge-platform docks. HW-verified.
pub(crate) static PROFILE_D6000: DockProfile = DockProfile {
    name: "Dell D6000 (Ridge, DL-6xxx)",
    video_eps: [0x08, 0x0b, 0x08, 0x0b],
    generation: Generation::Ridge,
    head_sub_shift: 0,
    stream_id_mask: 0x08,
    strm2_marker: 0x06,
    band_parity_bit: true,
    strip_blocks_x: 8,
    interlaced_bands: false,
    connectors: 2,
    dock_buffers: 2,
    max_refresh_hz: 120,
    max_head_clock_khz: 655_350,
    pixel_budget: 884_736_000,
    ep84_queue_depth: 4,
};

/// DL-7400 quad-display docks (Navarro).
///
/// Four independent physical connectors multiplexed over two video endpoints. This is not tiling:
/// the Windows capture has a distinct stream-open and record `sub` for each socket.
pub(crate) static PROFILE_DL7400: DockProfile = DockProfile {
    name: "DL-7400 quad dock (Navarro, DL-7000)",
    video_eps: [0x08, 0x0a, 0x08, 0x0a],
    generation: Generation::Navarro,
    head_sub_shift: 3,
    // The shared-pipe stream-open is sent before pixels by `build_stream_open_buf()`.  Its
    // remaining per-session material is established by the connector authentication sequence.
    stream_id_mask: 0x07,
    strm2_marker: 0x0c,
    band_parity_bit: false,
    // The fixed-size black carriers happen to need no padding and therefore show zero here, but
    // ordinary compressed image records carry the actual 0..15-byte pad count just like Ridge.
    // This only became visible once a non-uniform live framebuffer was compared record-by-record.
    strip_blocks_x: 16,
    interlaced_bands: true,
    connectors: 4,
    dock_buffers: 3,
    max_refresh_hz: u32::MAX,
    max_head_clock_khz: 699_500,
    pixel_budget: 1_216_512_000,
    ep84_queue_depth: 1,
};

/// Control and per-head bulk endpoints.
pub(crate) const EP_CTRL_OUT: u8 = 0x02;
pub(crate) const EP_CTRL_IN: u8 = 0x84;
