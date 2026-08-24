// SPDX-License-Identifier: GPL-2.0

//! Measured bring-up and mode-set timelines.
//!
//! A cold dock will not light from a correct message sequence alone; the vendor's spacing is
//! part of the contract, and a dock driven faster than this stays dark or resets. Every delay
//! here was read off a capture of the vendor driver waking the same hardware.

/// Dual-connector cold-wake timeline relative to the connector-0 mode set.
///
/// EP02 must remain quiet between `H1_MODE` and `QUIET_END`; the dock also requires a connector-1
/// EDID probe and fetch before video starts.
pub(crate) mod cold {
    pub(crate) const H1_MODE: i64 = 29;
    /// End of the silent window. Nothing may be sent on EP02 between `H1_MODE` and here.
    pub(crate) const QUIET_END: i64 = 1016;
    pub(crate) const H0_VIDEO: i64 = 1159;
    pub(crate) const H1_VIDEO: i64 = 1233;
    /// `(offset_ms, connector, sub, state)` stream markers. `sub` 0x2f/0x2e as on the wire.
    pub(crate) const MARKERS: &[(i64, u8, u16, u8)] = &[
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
    pub(crate) const POLLS: &[i64] = &[
        5, 26, 1019, 1130, 1192, 1204, 1222, 1235, 1253, 1270, 1287, 1304,
    ];
    /// `(offset_ms, connector, is_fetch)` -- `false` is the `0x15/0x20` probe, `true` the
    /// `0x15/0x21` fetch.
    pub(crate) const EDID: &[(i64, u8, bool)] = &[(1033, 1, false), (1059, 1, true)];
    /// Keep both carriers active until downstream clock programming completes.
    pub(crate) const CARRIER_TAIL_MS: i64 = 800;
}

/// A dock's cold bring-up choreography, anchored on the first connector's mode set.
///
/// Ridge and Navarro differ in more than timing: Navarro opens each connector's bracket with a
/// state-0 pair before the state-1/3 pair, spaces its two mode sets 757 ms apart rather than 29 ms,
/// sets connector 0's mode a *second* time shortly before connector 0's video, and streams
/// connector 1 first. Replaying one dock's timeline at the other leaves the endpoint unarmed. Every
/// connector field below is a transcript slot, not a connector number: 0 is the first connector an
/// activation brings up and 1 the second, whichever sockets they occupy. Both timelines were
/// recorded with the panels in the first two sockets, where the two happen to coincide.
/// [`VinoDrmData::activate_dual_wake`] resolves them.
pub(super) struct ColdTimeline {
    /// Offset of the second slot's mode set.
    pub(crate) h1_mode: i64,
    /// End of any silent window on EP02 after the second mode set.
    pub(crate) quiet_end: i64,
    /// Slots in the order they start streaming, with the offset each starts at.
    pub(crate) video: &'static [(usize, i64)],
    /// Mode sets repeated after the initial pair, as `(offset, slot)`.
    pub(crate) remode: &'static [(i64, usize)],
    /// `(offset_ms, slot, sub, state)` stream markers. `sub` 0x2f/0x2e as on the wire.
    pub(crate) markers: &'static [(i64, u8, u16, u8)],
    /// `id=0x14 sub=0x0c` status polls.
    pub(crate) polls: &'static [i64],
    /// `(offset_ms, slot, is_fetch)` EDID re-reads inside the bracket.
    pub(crate) edid: &'static [(i64, u8, bool)],
}

/// Ridge's timeline, as replayed from a D6000 cold bring-up.
pub(super) static COLD_RIDGE: ColdTimeline = ColdTimeline {
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
pub(super) static COLD_NAVARRO: ColdTimeline = ColdTimeline {
    h1_mode: 10,
    // DLM polls continuously across this span; there is no silent window to preserve.
    quiet_end: 11,
    // Video is not a pair of one-shot events: DLM keeps connector 0's carrier alive throughout the
    // still-open control bracket, starts connector 1, and continues both through the closing
    // markers. A gap here makes the dock accept one frame and NAK the next forever. The activation
    // path uses its pre-encoded carrier; normal scanout replaces it as soon as activation returns.
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

/// One step of a DL-3x00 dock-wide activation.
///
/// Every connector field is an activation slot, not a connector number: slot 0 is the
/// lowest-numbered activating connector and slot 1 the next, resolved to real connectors at the
/// point of send, exactly as [`ColdTimeline`]'s are. Steps carry no offsets because the vendor's
/// are separated by its own frames rather than by a clock; [`DockWideStep::Stream`] states the ones
/// that matter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DockWideStep {
    /// `0x48/0x22`, this slot's mode.
    SetMode(u8),
    /// `0x16/0x2e` or `0x16/0x2f`, with the state in byte 23.
    Marker(u8, u16, u8),
    /// `id=0x14 sub=0x0c` device status.
    Poll,
    /// The ring descriptor and decoder configuration that open this slot's stream as one unit.
    ///
    /// Kept for the conservative runtime re-arm.  The cold Ella transcript separates these two
    /// records with control traffic and uses [`DockWideStep::Ring`] / [`DockWideStep::Config`]
    /// instead, so the table can state those producer boundaries exactly.
    Prologue(u8),
    /// The unsealed ring descriptor that starts this slot's stream generation.
    Ring(u8),
    /// The sealed decoder configuration that follows a slot's ring descriptor.
    Config(u8),
    /// This slot's activation carrier -- the flat surface its stream opens with.
    Carrier(u8),
    /// Frames the vendor presents on `slot` before its next control record.
    ///
    /// The second connector comes up behind a running stream, and this is that stream. A dock
    /// whose activation is spaced by wall clock rather than by frames has nothing to put here.
    Stream(u8, u32),
}

/// How the vendor brings both connectors of a DL-3x00 dock up, measured record for record.
///
/// Two rules distinguish it from a driver that simply repeats a per-connector bracket, and both are
/// load-bearing. A single dock-wide transaction configures *both* connectors before any pixels:
/// the mode sets are adjacent, and every marker between them addresses the first connector. And
/// the second connector's sink is held down (`0x2e` state 3) across that transaction, coming up
/// only once the first is streaming -- with no mode set of its own, because it already has one.
///
/// A second independent bracket instead is what the dock stops answering: it acknowledges
/// everything up to the second bracket's first marker and nothing afterwards, which reads as a
/// dead dock and ends with every scanout returning `ENODEV`.
pub(crate) static ELLA_DOCK_WIDE: &[DockWideStep] = &[
    DockWideStep::SetMode(0),
    DockWideStep::Marker(0, 0x2f, 1),
    DockWideStep::Marker(0, 0x2e, 3),
    DockWideStep::SetMode(1),
    DockWideStep::Marker(0, 0x2f, 1),
    DockWideStep::Marker(0, 0x2e, 0),
    DockWideStep::Marker(1, 0x2f, 1),
    DockWideStep::Ring(0),
    DockWideStep::Poll,
    DockWideStep::Config(0),
    DockWideStep::Marker(1, 0x2e, 3),
    DockWideStep::Marker(0, 0x2f, 1),
    DockWideStep::Marker(0, 0x2e, 0),
    DockWideStep::Carrier(0),
    DockWideStep::Stream(0, 2),
    DockWideStep::Marker(1, 0x2f, 1),
    DockWideStep::Marker(0, 0x2f, 0),
    DockWideStep::Stream(0, 2),
    DockWideStep::Marker(1, 0x2e, 0),
    DockWideStep::Stream(0, 3),
    DockWideStep::Poll,
    DockWideStep::Ring(1),
    DockWideStep::Marker(0, 0x2e, 0),
    DockWideStep::Stream(0, 1),
    DockWideStep::Config(1),
    DockWideStep::Poll,
    DockWideStep::Carrier(1),
    // The second connector's sink is completed the same way the first one's was, and only once its
    // own carrier is running: assert, bring up, release, bring up again. The first connector
    // receives exactly this four-marker tail earlier in the transaction, and leaving it off the
    // second one ends the transaction with that sink half raised.
    DockWideStep::Marker(1, 0x2f, 1),
    DockWideStep::Stream(1, 2),
    DockWideStep::Poll,
    DockWideStep::Marker(1, 0x2e, 0),
    DockWideStep::Poll,
    DockWideStep::Marker(1, 0x2f, 0),
    DockWideStep::Poll,
    DockWideStep::Marker(1, 0x2e, 0),
];

/// How the vendor reconfigures one connector of a DL-3x00 dock while the other one is lit.
///
/// This is not the dock-wide sequence with a connector left out. It is shorter, it takes the sink
/// down *before* the mode set rather than after it, and it touches nothing belonging to the other
/// connector -- which keeps streaming across the whole of it. Replaying the cold bracket here
/// instead is what silences the dock: it stops answering at the first marker and does not answer
/// again.
///
/// Measured twice, at two different resolutions, with identical shape both times.
///
/// The vendor also re-reads the sink's EDID between the sink-down and the mode set. That is left
/// out: vino reads EDID on its own schedule, and a fetch issued inside a transaction has nowhere
/// to deliver its reply.
pub(crate) static ELLA_RUNTIME_MODE: &[DockWideStep] = &[
    DockWideStep::Marker(0, 0x2f, 1),
    DockWideStep::Marker(0, 0x2e, 3),
    DockWideStep::SetMode(0),
    DockWideStep::Marker(0, 0x2f, 1),
    DockWideStep::Marker(0, 0x2e, 0),
    DockWideStep::Poll,
    DockWideStep::Prologue(0),
    DockWideStep::Carrier(0),
];

/// Reservation-token slots for [`COLD_NAVARRO::markers`] and [`COLD_NAVARRO::polls`].
///
/// DLM assigns counters in its per-connector workers before their EP02 writes interleave. Wire AES
/// sequence remains monotonic, but the echoed inner counters consequently do not: for example the
/// wire order begins `n, n+1, n+3, n+2, n+5, n+4`. Navarro starts NAKing at the first flattened
/// counter, so retain that allocation order. These numbers index live reservation tokens; they
/// are not protocol counters and are never added to a captured/base counter.
pub(super) static NAVARRO_MARKER_COUNTER_SLOTS: &[usize] =
    &[1, 2, 4, 6, 8, 7, 10, 12, 11, 14, 16, 17, 20, 21, 26, 27];
pub(super) static NAVARRO_POLL_COUNTER_SLOTS: &[usize] = &[5, 9, 13, 15, 18, 19, 22, 23, 24, 25];
pub(super) const NAVARRO_COLD_COUNTERS: usize = 28;

/// One operation in Navarro's cold sink-reset prelude. This is separate from [`ColdTimeline`]: it
/// runs before the first real mode set and changes downstream EDID/sink state, whereas
/// `ColdTimeline` brackets already-programmed streams.
#[derive(Clone, Copy)]
pub(super) enum NavarroColdOp {
    Poll,
    EdidState(u8, u8),
    Probe(u8),
    Fetch(u8),
    /// Tear the downstream sink down. Offset 23 is the literal state `0xff`, not a connector.
    SinkTeardown(u8),
    /// Engage the downstream sink.
    ///
    /// `id=0x16 sub=0x23` names the connector twice, at offset 22 and at offset 23, which is why it
    /// is [`crate::cp::edid_sink_state`]`(connector, connector)`. It is a distinct variant rather
    /// than a state so that the remap below cannot mistake the second selector for a constant: the
    /// dock acknowledges a mismatched pair and then never enables the sink.
    Engage(u8),
    PostEdid(u8),
    Clear(u8),
}

impl NavarroColdOp {
    /// Translate this op's transcript slot into the connector that slot stands for in this
    /// activation.
    ///
    /// The captured sequence names connectors 0 and 1 because that is where DLM's panels were. It
    /// describes the first and second connector being brought up, whichever sockets those are.
    pub(crate) fn remap_head(self, remap: &impl Fn(u8) -> u8) -> Self {
        match self {
            Self::Poll => Self::Poll,
            Self::EdidState(h, s) => Self::EdidState(remap(h), s),
            Self::Probe(h) => Self::Probe(remap(h)),
            Self::Fetch(h) => Self::Fetch(remap(h)),
            Self::SinkTeardown(h) => Self::SinkTeardown(remap(h)),
            Self::Engage(h) => Self::Engage(remap(h)),
            Self::PostEdid(h) => Self::PostEdid(remap(h)),
            Self::Clear(h) => Self::Clear(remap(h)),
        }
    }
}

/// Authenticated DLM transaction between Navarro's first clear pair and its first real mode.
///
/// Offsets are milliseconds from the first connector-0 clear in
/// `navarro-dlm-today-124144/wire.pcapng`. Equal offsets deliberately retain wire order. Most
/// importantly, DLM stops both EDID readers and sends sink state `0xff` immediately after the
/// first clears, then re-reads and re-engages each sink before its second clear. Omitting this
/// whole state transition left the video endpoints accepting one bulk transfer and NAKing every
/// subsequent transfer.
pub(super) static NAVARRO_COLD_PRELUDE: &[(i64, NavarroColdOp)] = &[
    (2, NavarroColdOp::EdidState(0, 0)),
    (3, NavarroColdOp::Probe(0)),
    (5, NavarroColdOp::EdidState(1, 0)),
    (5, NavarroColdOp::Probe(1)),
    (7, NavarroColdOp::SinkTeardown(0)),
    (7, NavarroColdOp::SinkTeardown(1)),
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
    (1810, NavarroColdOp::Engage(0)),
    (1822, NavarroColdOp::Clear(0)),
    (1827, NavarroColdOp::Poll),
    (1850, NavarroColdOp::Probe(1)),
    (1853, NavarroColdOp::EdidState(1, 1)),
    (1856, NavarroColdOp::Probe(1)),
    (1902, NavarroColdOp::PostEdid(0)),
    (1903, NavarroColdOp::Fetch(1)),
    (1907, NavarroColdOp::Engage(1)),
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
pub(super) const NAVARRO_PRIME_CLEAR_H1_MS: i64 = 2;

/// One status-poll interval, the gap the prelude's own trailing polls run at.
pub(super) const NAVARRO_REAL_MODE_SETTLE_MS: i64 = 20;

/// When the cold prelude's last operation is due.
pub(crate) const NAVARRO_COLD_PRELUDE_END_MS: i64 =
    NAVARRO_COLD_PRELUDE[NAVARRO_COLD_PRELUDE.len() - 1].0;

/// Where the first real mode goes, measured from the start of activation.
///
/// DLM's capture puts it at 2978 ms, but DLM's own prelude finishes at 2016 ms; the ~960 ms
/// between is dead air in that capture rather than something the dock asked for, and it is over
/// half a 5.59 s enumerate-to-pixels bring-up. Follow the end of the prelude plus one poll
/// interval, so the two cannot drift apart when the prelude changes.
pub(crate) const NAVARRO_REAL_MODE_H0_MS: i64 =
    NAVARRO_COLD_PRELUDE_END_MS + NAVARRO_REAL_MODE_SETTLE_MS;

/// How long the KMS worker waits for the rest of a multi-connector atomic commit's mode sets.
///
/// Bounded so a genuine single-connector commit costs at most this before proceeding.
pub(super) const MODESET_BATCH_SETTLE_MS: i64 = 20;

/// Number of back-to-back presentations of one already-encoded full frame while a newly-mode-set
/// downstream is training.
pub(crate) const COLD_TRAINING_PRESENTATIONS: u32 = 8;

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
use kernel::prelude::kunit_tests;

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
#[kunit_tests(vino_timeline)]
mod tests {
    use super::*;
    use crate::*;

    /// A mode is bounded by link rate, not by refresh, except where the dock itself clamps.
    ///
    /// The first real mode follows the cold prelude; it does not sit at a copied capture offset.
    ///
    /// DLM's capture put its first real mode at 2978 ms while its own prelude finished at
    /// 2016 ms. Deriving the offset keeps the two from drifting apart, and this pins the
    /// direction: the mode must come *after* the prelude's last op, and must not reintroduce the
    /// second of dead air that made it 53% of a bring-up.
    #[test]
    fn the_first_real_mode_follows_the_prelude_rather_than_the_capture() {
        let last = NAVARRO_COLD_PRELUDE_END_MS;
        assert_eq!(last, 2016);
        assert!(NAVARRO_REAL_MODE_H0_MS > last);
        assert!(NAVARRO_REAL_MODE_H0_MS - last <= 100);
    }

    /// The DL-3x00 dock-wide activation, step for step against the vendor's record stream.
    ///
    /// The order below is transcribed from a decrypted DLM bring-up; the vendor's frames between
    /// its control records are what the settles stand in for. Two properties of it are what the
    /// dock actually enforces, and both are checked separately afterwards, because a plausible
    /// reordering breaks them without changing any single step.
    #[test]
    fn ella_dock_wide_matches_the_dlm_capture() -> Result {
        use super::DockWideStep::*;
        let want = [
            SetMode(0),         // vendor #86
            Marker(0, 0x2f, 1), // #87
            Marker(0, 0x2e, 3), // #88
            SetMode(1),         // #89, inside the first connector's bracket
            Marker(0, 0x2f, 1), // #90
            Marker(0, 0x2e, 0), // #91
            Marker(1, 0x2f, 1), // #92
            Ring(0),            // #93 ring descriptor
            Poll,               // #94, between the two records of the prologue
            Config(0),          // #95 decoder configuration
            Marker(1, 0x2e, 3), // #96, the second connector's sink held down
            Marker(0, 0x2f, 1), // #97
            Marker(0, 0x2e, 0), // #98, the last record before pixels
            Carrier(0),         // #99
            Stream(0, 2),       // #99..#227, the first connector streaming
            Marker(1, 0x2f, 1), // #229
            Marker(0, 0x2f, 0), // #247
            Stream(0, 2),       // #248..#429
            Marker(1, 0x2e, 0), // #431, the second connector's sink up behind a running stream
            Stream(0, 3),       // #432..#629
            Poll,               // #632
            Ring(1),            // #633 ring descriptor, ahead of the marker
            Marker(0, 0x2e, 0), // #634
            Stream(0, 1),       // one first-connector frame, #635..#732
            Config(1),          // #733 decoder configuration
            Poll,               // #734, the last record before the second connector's pixels
            Carrier(1),         // #735
            // The second connector's sink is completed exactly as the first one's was, and only
            // behind its own running carrier. Leaving this off ends the transaction with that
            // sink half raised.
            Marker(1, 0x2f, 1), // #737
            Stream(1, 2),       // #738..#742, two more carrier frames
            Poll,               // #743
            Marker(1, 0x2e, 0), // #744
            Poll,               // #745
            Marker(1, 0x2f, 0), // #746
            Poll,               // #747
            Marker(1, 0x2e, 0), // #748
        ];
        assert_eq!(ELLA_DOCK_WIDE, &want[..]);
        Ok(())
    }

    /// Both connectors are configured before either sends a pixel, and only one mode set each.
    ///
    /// A second bracket around a second mode set is what the dock stops answering: it acknowledges
    /// every record up to that bracket's first marker and nothing afterwards.
    #[test]
    fn ella_dock_wide_sets_both_modes_before_any_pixels() -> Result {
        use super::DockWideStep::*;
        let mut modes = [0u32; 2];
        for step in ELLA_DOCK_WIDE {
            match *step {
                SetMode(slot) => modes[usize::from(slot)] += 1,
                // Every carrier finds both connectors already configured, and each exactly once.
                Carrier(_) => assert_eq!(modes, [1, 1]),
                _ => {}
            }
        }
        assert_eq!(modes, [1, 1]);
        Ok(())
    }

    /// The second connector's sink stays down across the first connector's carrier.
    ///
    /// The vendor takes it down inside the dock-wide bracket and brings it up only once the first
    /// connector is streaming. A sink brought up early is a connector the dock is scanning out
    /// while its stream has neither a ring descriptor nor a decoder configuration.
    #[test]
    fn ella_dock_wide_holds_the_second_sink_down_until_the_first_streams() -> Result {
        use super::DockWideStep::*;
        let mut down = false;
        let mut streaming = false;
        let mut up_after_streaming = false;
        for step in ELLA_DOCK_WIDE {
            match *step {
                Marker(1, 0x2e, 3) => down = true,
                Marker(1, 0x2e, 0) => {
                    // Up only after it was taken down, and only once the first is streaming.
                    assert!(down);
                    assert!(streaming);
                    up_after_streaming = true;
                }
                Carrier(0) => streaming = true,
                // Neither the second stream's opening records nor its pixels reach a downed sink.
                Carrier(1) => assert!(up_after_streaming),
                Ring(1) | Config(1) | Prologue(1) => assert!(up_after_streaming),
                _ => {}
            }
        }
        assert!(up_after_streaming);
        Ok(())
    }

    /// Reconfiguring one DL-3x00 connector while the other is lit, against the vendor's own.
    ///
    /// Transcribed from two vendor reconfigurations at different resolutions, whose records agree
    /// step for step. The sink-down before the mode set is the part a reader is most likely to
    /// think redundant with the cold table's, where it comes after.
    #[test]
    fn ella_runtime_mode_matches_the_dlm_capture() -> Result {
        use super::DockWideStep::*;
        let want = [
            Marker(0, 0x2f, 1), // vendor #38348 / #44725
            Marker(0, 0x2e, 3), // #38349 / #44726, sink down ahead of the mode set
            SetMode(0),         // #38353 / #44729
            Marker(0, 0x2f, 1), // #38354 / #44730
            Marker(0, 0x2e, 0), // #38355 / #44731, sink up
            Poll,               // #38356 / #44732
            Prologue(0),        // #38357 / #44733
            Carrier(0),         // #38358 / #44734 onward
        ];
        assert_eq!(ELLA_RUNTIME_MODE, &want[..]);
        Ok(())
    }

    /// A runtime reconfiguration names one connector and never the other.
    ///
    /// The connector that is already lit keeps streaming across the whole sequence; a marker
    /// addressed to it here would bracket a stream the dock is mid-frame on.
    #[test]
    fn ella_runtime_mode_touches_one_connector() -> Result {
        use super::DockWideStep::*;
        for step in ELLA_RUNTIME_MODE {
            let slot = match *step {
                SetMode(slot)
                | Marker(slot, _, _)
                | Prologue(slot)
                | Ring(slot)
                | Config(slot)
                | Carrier(slot) => slot,
                Poll | Stream(_, _) => continue,
            };
            assert_eq!(slot, 0);
        }
        Ok(())
    }
}
