// SPDX-License-Identifier: GPL-2.0

//! The runtime knobs a matched dock profile installs on the KMS device.
//!
//! The KMS side has one code path per operation and no per-dock branches; every difference
//! between generations arrives here as data, is stored in an atomic, and is read back by the
//! workers. Nothing in this module decides anything -- [`crate::profile`] does.

use super::*;

impl VinoDrmData {
    /// Record this device's codec geometry; see [`Self::geometry`].
    pub(crate) fn set_codec_geometry(
        &self,
        strip_blocks_x: usize,
        interlaced: bool,
        band_parity: bool,
        connector_selector_shift: u8,
        stream_id_mask: u8,
        dock_buffers: u8,
        coding: crate::video_arm::CodeTables,
        steady_sub_bit: u8,
    ) {
        let narrow = coding == crate::video_arm::CodeTables::Narrow;
        let packed = (strip_blocks_x as u32 & 0xff)
            | ((interlaced as u32) << 8)
            | ((band_parity as u32) << 9)
            | ((narrow as u32) << 10)
            | (((steady_sub_bit != 0) as u32) << 11)
            | ((connector_selector_shift as u32) << 16)
            | ((stream_id_mask as u32) << 20)
            | ((dock_buffers as u32) << 28);
        self.codec_geometry
            .store(packed | 0x8000, core::sync::atomic::Ordering::Release);
    }

    /// This device's codec geometry, to be passed into every codec call made on its behalf.
    ///
    /// Stored packed because the DRM device allocation is pin-initialised before `probe` knows
    /// which dock it matched. A device with no profile applied reads the Ridge layout.
    pub(crate) fn geometry(&self) -> crate::video::haar::Geometry {
        let p = self
            .codec_geometry
            .load(core::sync::atomic::Ordering::Acquire);
        if p & 0x8000 == 0 {
            return crate::video::haar::RIDGE_GEOMETRY;
        }
        crate::video::haar::Geometry::new(
            ((p & 0xff) as usize).max(1),
            p & (1 << 8) != 0,
            p & (1 << 9) != 0,
            ((p >> 16) & 0xf) as u8,
            ((p >> 20) & 0xff) as u8,
            ((p >> 28) & 0xf) as u8,
        )
        .with_coding(if p & (1 << 10) != 0 {
            crate::video_arm::CodeTables::Narrow
        } else {
            crate::video_arm::CodeTables::Wide
        })
        .with_steady_sub_bit(if p & (1 << 11) != 0 { 0x20 } else { 0 })
    }

    /// Record how this dock delivers logical framebuffer updates to its ring buffers.
    pub(crate) fn set_frame_delivery(&self, policy: crate::profile::FrameDelivery) {
        let packed = u32::from(policy.keyframe_presentations.max(1))
            | (u32::from(policy.delta_presentations.max(1)) << 8)
            | (u32::from(policy.damage_frames.max(1)) << 16);
        self.frame_delivery.store(packed, Ordering::Release);
    }

    /// Snapshot this dock's frame-delivery policy.
    pub(crate) fn frame_delivery(&self) -> crate::profile::FrameDelivery {
        let packed = self.frame_delivery.load(Ordering::Acquire);
        crate::profile::FrameDelivery::new(
            ((packed & 0xff) as u8).max(1),
            (((packed >> 8) & 0xff) as u8).max(1),
            (((packed >> 16) & 0xff) as u8).max(1),
        )
    }

    /// Record whether timed presence probes may reset a bracket beside a live connector.
    pub(crate) fn set_probe_bracket(&self, policy: crate::profile::ProbeBracket) {
        self.probe_bracket.store(policy as u8, Ordering::Release);
    }

    /// This dock's bracket-reset policy.
    pub(crate) fn probe_bracket(&self) -> crate::profile::ProbeBracket {
        match self.probe_bracket.load(Ordering::Acquire) {
            x if x == crate::profile::ProbeBracket::DeferWithActiveSibling as u8 => {
                crate::profile::ProbeBracket::DeferWithActiveSibling
            }
            _ => crate::profile::ProbeBracket::Always,
        }
    }

    /// Whether this dock can be driven at ten bits per channel; see [`DockProfile::hdr_capable`].
    pub(crate) fn hdr_capable(&self) -> bool {
        self.hdr_capable.load(Ordering::Acquire)
    }

    /// Whether this dock composites a cursor bitmap of its own; see [`DockProfile::hw_cursor`].
    pub(crate) fn hw_cursor(&self) -> bool {
        self.hw_cursor.load(Ordering::Acquire)
    }

    /// Record whether the dock's presence probe describes a connector; see
    /// [`DockProfile::reports_presence`].
    pub(crate) fn set_reports_presence(&self, on: bool) {
        self.reports_presence.store(on, Ordering::Release);
    }

    /// Whether the dock's presence probe describes a connector; see
    /// [`DockProfile::reports_presence`].
    pub(crate) fn reports_presence(&self) -> bool {
        self.reports_presence.load(Ordering::Acquire)
    }

    /// Record whether the connectors share one EDID handler; see
    /// [`DockProfile::shared_edid_handler`].
    pub(crate) fn set_shared_edid_handler(&self, on: bool) {
        self.shared_edid_handler.store(on, Ordering::Release);
    }

    /// Whether the connectors share one EDID handler; see [`DockProfile::shared_edid_handler`].
    pub(crate) fn shared_edid_handler(&self) -> bool {
        self.shared_edid_handler.load(Ordering::Acquire)
    }

    /// Record whether a frame ending on a full packet is split; see
    /// [`DockProfile::split_full_packet_frame`].
    pub(crate) fn set_split_full_packet_frame(&self, on: bool) {
        self.split_full_packet_frame.store(on, Ordering::Release);
    }

    /// Whether a frame ending on a full packet is split.
    pub(crate) fn split_full_packet_frame(&self) -> bool {
        self.split_full_packet_frame.load(Ordering::Acquire)
    }

    /// Whether `connector`'s committed framebuffer is 10 bits per channel.
    ///
    /// Read by the mode-set path, which must announce a depth to the dock that matches the one the
    /// plane will actually send: the dock sizes its buffer from the pair and mis-sizes it if they
    /// disagree.
    pub(super) fn connector_is_ten_bit(&self, connector: usize) -> bool {
        self.connector_ten_bit.load(Ordering::Acquire) & (1u32 << connector) != 0
    }

    /// Whether `connector`'s connector is being driven in PQ; see [`Self::set_connector_st2084`].
    pub(super) fn connector_is_st2084(&self, connector: usize) -> bool {
        self.head_st2084.load(Ordering::Acquire) & (1u32 << connector) != 0
    }

    /// Record the transfer function userspace has asked for on `connector`.
    ///
    /// Driven by the connector's `HDR_OUTPUT_METADATA` blob rather than by anything vino decides,
    /// for the same reason [`Self::set_connector_depth`] follows the framebuffer's fourcc: the dock
    /// must be told what the pixels actually are, and any state of our own could drift from them.
    pub(super) fn set_connector_st2084(&self, connector: u8, on: bool) {
        let bit = 1u32 << u32::from(connector);
        if on {
            self.head_st2084.fetch_or(bit, Ordering::Release);
        } else {
            self.head_st2084.fetch_and(!bit, Ordering::Release);
        }
    }

    /// This device's codec geometry at one connector's committed sample depth.
    ///
    /// Every path that touches pixels wants this rather than [`Self::geometry`]: the depth decides
    /// the entropy coder's escape ceiling, and getting it wrong desynchronises the dock's decoder
    /// rather than merely degrading the picture. See [`crate::video::haar::Depth`].
    pub(super) fn geometry_for_connector(&self, connector: u8) -> crate::video::haar::Geometry {
        let ten = self.connector_ten_bit.load(Ordering::Acquire) & (1 << u32::from(connector)) != 0;
        self.geometry().with_depth(if ten {
            crate::video::haar::Depth::Ten
        } else {
            crate::video::haar::Depth::Eight
        })
    }

    /// Record the sample depth of the framebuffer a connector is scanning out.
    ///
    /// Driven by the committed framebuffer's fourcc rather than by any state of our own, so it
    /// cannot drift from the pixels actually in hand. A format the codec does not know leaves the
    /// connector where it was; `atomic_check` is what rejects those, and a plane list that only
    /// offers `XRGB8888` means this never sees one.
    pub(super) fn set_connector_depth(&self, connector: u8, depth: crate::video::haar::Depth) {
        let bit = 1u32 << u32::from(connector);
        match depth {
            crate::video::haar::Depth::Ten => {
                self.connector_ten_bit.fetch_or(bit, Ordering::Release)
            }
            crate::video::haar::Depth::Eight => {
                self.connector_ten_bit.fetch_and(!bit, Ordering::Release)
            }
        };
    }

    /// Record the mode-programming and blanking behaviour this dock wants.
    pub(crate) fn set_mode_behaviour(&self, profile: &'static crate::profile::DockProfile) {
        self.dock_wide_modeset
            .store(profile.protocol.dock_wide_modeset, Ordering::Release);
        self.clear_mode_before_set
            .store(profile.protocol.clear_mode_before_set, Ordering::Release);
        self.video_keepalive
            .store(profile.protocol.video_keepalive, Ordering::Release);
        self.blank_markers_held.store(
            matches!(
                profile.protocol.blank_bracket,
                crate::profile::BlankBracket::MarkersHeld
            ),
            Ordering::Release,
        );
    }

    /// Whether a connector must keep being fed while its content is unchanged.
    pub(crate) fn video_keepalive(&self) -> bool {
        self.video_keepalive.load(Ordering::Acquire)
    }

    /// Whether programming any connector reconfigures the whole dock.
    pub(crate) fn dock_wide_modeset(&self) -> bool {
        self.dock_wide_modeset.load(Ordering::Acquire)
    }

    /// Whether a connector's pipe is torn down before a timing is programmed onto it.
    pub(crate) fn clear_mode_before_set(&self) -> bool {
        self.clear_mode_before_set.load(Ordering::Acquire)
    }

    /// How a connector blanks; see [`crate::profile::BlankBracket`].
    pub(crate) fn blank_bracket(&self) -> crate::profile::BlankBracket {
        if self.blank_markers_held.load(Ordering::Acquire) {
            crate::profile::BlankBracket::MarkersHeld
        } else {
            crate::profile::BlankBracket::BlackThenClose
        }
    }

    /// Record how this dock states its framebuffer allocation in a set-mode.
    pub(crate) fn set_allocation(&self, allocation: &'static crate::profile::Allocation) {
        let _ = self.allocation.populate(allocation);
    }

    /// How this dock states its framebuffer allocation; Ridge's device override until probe has
    /// matched a profile, as with every other value published there.
    pub(crate) fn allocation(&self) -> &'static crate::profile::Allocation {
        self.allocation
            .as_ref()
            .copied()
            .unwrap_or(&crate::profile::PROFILE_RIDGE.protocol.allocation)
    }

    /// Record whether this dock opens a stream with the ARM burst; see `DockProfile::arm_burst`.
    pub(crate) fn set_arm_burst(&self, on: bool) {
        self.arm_burst.store(on, Ordering::Release);
    }

    /// Whether the first frame after a mode set carries the cold ARM burst.
    pub(super) fn uses_arm_burst(&self) -> bool {
        self.arm_burst.load(Ordering::Acquire)
    }

    /// Length of a continuous-presentation window, for this dock.
    ///
    /// The activation carrier and the blank presentation both work by presenting one encoded frame
    /// back to back for a fixed wall-clock window, with no control transaction in between. That
    /// trains a downstream link on a dock with a video pipe of its own. On a dock that carries
    /// video on the control pipe it instead holds the endpoint for the whole window, and the dock
    /// is silenced at exactly the moment the mode set needs it to answer. Such a dock gets a
    /// single presentation instead, which is what `submit_prompt_training` does at zero.
    pub(super) fn carrier_ms(&self, base: i64) -> i64 {
        if self.video_on_ctrl_pipe() {
            0
        } else {
            base
        }
    }

    /// How many carrier frames a connector presents before its first content frame; see
    /// `DockProfile::carrier_frames`.
    pub(super) fn carrier_presentations(&self) -> u32 {
        self.carrier_frames
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Record how this dock's video stream describes itself.
    ///
    /// The three values travel together because they are read together, by the one builder that
    /// states a stream's mode and decoder tables.
    pub(crate) fn set_video_stream_desc(
        &self,
        layout_word: u16,
        marker_kind: u8,
        tables: crate::video_arm::CodeTables,
    ) {
        let narrow = matches!(tables, crate::video_arm::CodeTables::Narrow);
        let packed =
            u32::from(layout_word) | (u32::from(marker_kind) << 16) | ((narrow as u32) << 24);
        self.video_stream_desc.store(packed, Ordering::Release);
    }

    /// The word repeated beside the surface size in this dock's stream mode header.
    pub(crate) fn layout_word(&self) -> u16 {
        self.video_stream_desc.load(Ordering::Acquire) as u16
    }

    /// The byte naming this dock in a sealed stream's opening marker.
    pub(crate) fn stream_marker_kind(&self) -> u8 {
        (self.video_stream_desc.load(Ordering::Acquire) >> 16) as u8
    }

    /// Which form of decoder code tables this dock's stream configuration states.
    pub(crate) fn code_tables(&self) -> crate::video_arm::CodeTables {
        if self.video_stream_desc.load(Ordering::Acquire) & (1 << 24) != 0 {
            crate::video_arm::CodeTables::Narrow
        } else {
            crate::video_arm::CodeTables::Wide
        }
    }

    /// Record this dock's minimum frame interval; see `DockProfile::frame_period_ms`.
    pub(crate) fn set_frame_period_ms(&self, ms: i64) {
        let ms = if ms <= 0 { FRAME_PERIOD_MS } else { ms };
        self.frame_period_us
            .store(ms * 1000, core::sync::atomic::Ordering::Relaxed);
    }

    /// Record how many carrier frames open a stream; see `DockProfile::carrier_frames`.
    pub(crate) fn set_carrier_frames(&self, frames: u32) {
        self.carrier_frames
            .store(frames.max(1), core::sync::atomic::Ordering::Relaxed);
    }

    /// Record this dock's keepalive status interval; see `DockProfile::status_period_ms`.
    pub(crate) fn set_status_period_ms(&self, ms: i64) {
        let ms = if ms <= 0 { STATUS_PERIOD_MS } else { ms };
        self.status_period_ms
            .store(ms, core::sync::atomic::Ordering::Relaxed);
    }

    /// This dock's interval between keepalive status queries, in milliseconds.
    pub(crate) fn status_period_ms(&self) -> i64 {
        self.status_period_ms
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// This dock's minimum interval between frames on one connector, in milliseconds.
    pub(crate) fn frame_period_ms(&self) -> i64 {
        self.frame_period_us() / 1000
    }

    /// This dock's minimum interval between frames on one connector, in microseconds.
    pub(super) fn frame_period_us(&self) -> i64 {
        self.frame_period_us
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Record how much of this dock's endpoint may be occupied; see `DockProfile::stream_pacing`.
    pub(crate) fn set_stream_pacing(&self, pacing: crate::profile::StreamPacing) {
        self.stream_budget_bps.store(
            pacing.bytes_per_sec.max(1),
            core::sync::atomic::Ordering::Relaxed,
        );
        self.stream_burst_bytes.store(
            pacing.burst_bytes.max(1),
            core::sync::atomic::Ordering::Relaxed,
        );
        let mut credit = self.stream_credit.lock();
        *credit = StreamCredit::new();
    }

    /// How long a frame must wait for this dock's sustained budget, or `None` to send it now.
    ///
    /// Tops the ledger up for the time since it was last read, so an idle dock is always in credit
    /// and this costs a busy dock one spinlock per frame.
    pub(super) fn stream_budget_wait_us(&self) -> Option<i64> {
        let bps = self
            .stream_budget_bps
            .load(core::sync::atomic::Ordering::Relaxed);
        if bps == u32::MAX {
            return None;
        }
        let now = Instant::<Monotonic>::now();
        let mut credit = self.stream_credit.lock();
        let elapsed_us = credit
            .topped_up
            .map_or(1_000_000, |last| (now - last).as_micros_ceil());
        credit.topped_up = Some(now);
        // Cap the ledger at the burst allowance, not at a second of throughput. A dock idle for a
        // minute must not bank a minute of bytes; nor may it bank a whole second's worth, which is
        // more than this dock survives in one go.
        let ceiling = i64::from(
            self.stream_burst_bytes
                .load(core::sync::atomic::Ordering::Relaxed),
        );
        credit.bytes = credit
            .bytes
            .saturating_add(stream_credit_accrued(bps, elapsed_us))
            .min(ceiling);
        stream_credit_wait_us(bps, credit.bytes)
    }

    /// Charge a frame that reached the dock against the sustained budget.
    pub(super) fn charge_stream_budget(&self, bytes: usize) {
        if self
            .stream_budget_bps
            .load(core::sync::atomic::Ordering::Relaxed)
            == u32::MAX
        {
            return;
        }
        let mut credit = self.stream_credit.lock();
        credit.bytes = credit.bytes.saturating_sub(bytes as i64);
    }

    /// Record the state that takes this dock's sinks down; see `DockProfile::sink_down_state`.
    /// Record the `0x2e` state re-sent mid-bracket; see `DockProfile::bracket_reopen_state`.
    pub(crate) fn set_post_mode_sink_states(&self, states: [u8; 2]) {
        let packed = u16::from(states[0]) | (u16::from(states[1]) << 8);
        self.post_mode_sink_states
            .store(packed, core::sync::atomic::Ordering::Release);
    }

    /// Record the state this dock wants before a mode set; see
    /// `DockProfile::pre_mode_sink_state`.
    pub(crate) fn set_pre_mode_sink_state(&self, state: Option<u8>) {
        self.pre_mode_sink_state.store(
            state.map_or(u16::MAX, u16::from),
            core::sync::atomic::Ordering::Release,
        );
    }

    pub(crate) fn pre_mode_sink_state(&self) -> Option<u8> {
        match self
            .pre_mode_sink_state
            .load(core::sync::atomic::Ordering::Acquire)
        {
            u16::MAX => None,
            state => Some(state as u8),
        }
    }

    pub(super) fn post_mode_sink_state(&self, index: usize) -> u8 {
        let packed = self
            .post_mode_sink_states
            .load(core::sync::atomic::Ordering::Acquire);
        (packed >> (8 * index)) as u8
    }

    pub(crate) fn set_sink_down_state(&self, state: u8) {
        self.sink_down_state
            .store(state, core::sync::atomic::Ordering::Release);
    }

    /// The `0x16/0x2e` state that takes a downstream sink down on this dock.
    pub(crate) fn sink_down_state(&self) -> u8 {
        self.sink_down_state
            .load(core::sync::atomic::Ordering::Acquire)
    }

    /// Record whether video shares the control pipe; see `DockProfile::video_on_ctrl_pipe`.
    pub(crate) fn set_video_on_ctrl_pipe(&self, on: bool) {
        self.video_on_ctrl_pipe.store(on, Ordering::Release);
    }

    /// Whether video records travel on the control bulk-OUT pipe.
    pub(crate) fn video_on_ctrl_pipe(&self) -> bool {
        self.video_on_ctrl_pipe.load(Ordering::Acquire)
    }
}
