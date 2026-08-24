// SPDX-License-Identifier: GPL-2.0

//! Taking a connector's sink down and back up, and the brackets around a mode set.
//!
//! A dock does not simply accept a new timing. Its vendor drives the sink down, programs the
//! timing and brings it back up, and the states and the gaps between them are what retrains the
//! downstream link. Get the sequence wrong and the dock accepts every byte of every frame and
//! lights nothing, with nothing on the wire to say so.

use super::*;

impl VinoDrmData {
    /// Drive `connector` to black on the dock, then close its stream bracket.
    ///
    /// Runs on the command worker for [`KmsCmd::Blank`], i.e. after `atomic_disable` has already
    /// zeroed this connector's mode generation. That zero is what makes the write legal: every
    /// video path gates on `modeset_requested == modeset_active == want`, and passing `want = 0`
    /// matches exactly the disabled state and nothing else -- so a re-enable racing this blank
    /// flips both atomics to a real key and the submit loop drops out with `ENODEV` instead of
    /// painting black over the freshly enabled mode.
    ///
    /// The dock's stream itself is still configured (vino never told it otherwise), so the black
    /// frames are an ordinary accepted write, not a write onto a torn-down pipe.
    pub(super) fn blank_connector(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        let socket = connector + 1;
        let connector_index = connector as usize;
        // A dock that wants its markers held takes two of them and then silence: no video, no mode
        // set and no close bracket for as long as the output stays down. It is the same pair
        // `modeset_bracket_pre` opens with, so the stream is held rather than torn down, and
        // [`Self::close_blank_bracket`] owes the matching re-open before the connector is driven
        // again. The scanout has already stopped, because `atomic_disable` zeroed this connector's
        // mode generation before queueing this command.
        //
        // Sending such a dock the close bracket below instead re-enumerates it about two seconds
        // later, which takes the whole desktop down with it.
        if self.blank_bracket() == crate::profile::BlankBracket::MarkersHeld {
            self.stream_marker(dev, connector, 0x2f, 1)?;
            self.stream_marker(dev, connector, 0x2e, self.sink_down_state())?;
            self.set_self_blanked(connector_index, true);
            self.blank_bracket_open
                .fetch_or(1u32 << connector_index, Ordering::Release);
            vino_debug!("vino: socket {socket} blanked; bracket held open, stream idle\n");
            return Ok(());
        }
        let Some(timing) = self.last_timing.lock()[connector_index] else {
            // Never modeset, so there is nothing lit to blank.
            return Ok(());
        };
        let geometry = self.geometry();
        let padded_width =
            (timing.hactive as usize + geometry.strip_w() - 1) & !(geometry.strip_w() - 1);
        let padded_height =
            (timing.vactive as usize + geometry.strip_h() - 1) & !(geometry.strip_h() - 1);
        let frames =
            crate::video::haar::black_frame_ep08(geometry, padded_width, padded_height, connector)?;
        let ordinary_frames = crate::video::haar::black_frame_ep08_ordinary(
            geometry,
            padded_width,
            padded_height,
            connector,
        )?;
        // Present for long enough to reach every dock buffer. The dock is multi-buffered and a
        // single presentation lands in one buffer only -- the same reason damage debt exists --
        // so a one-shot blank leaves the other buffer holding the frozen desktop and the panel
        // alternates between black and stale content.
        let sent = self.submit_prompt_training(
            dev,
            connector,
            0,
            &frames,
            &ordinary_frames,
            BLANK_PRESENT_MS,
            false,
        )?;
        self.stream_marker(dev, connector, 0x2f, 0)?;
        self.stream_marker(dev, connector, 0x2e, 0)?;
        // Do not take the sink down for a connector whose monitor has already gone away.
        //
        // `atomic_disable` fires for both a DPMS-off and a monitor removal, and they need opposite
        // treatment. Sending the power-down marker at a sink that is already gone is pointless, and
        // setting `self_blanked` would make the presence watcher deliberately ignore that
        // connector's silence, preventing a later replug from being detected.
        let candidate = if self.connector_present(connector_index) {
            self.sink_down_state()
        } else {
            vino_debug!(
                "vino: socket {socket} blank skips the sink marker -- its monitor is already gone\n"
            );
            0
        };
        if candidate != 0 {
            // From here the dock will stop answering this connector's presence probe, exactly as it
            // does for a real unplug. Claim the silence before causing it.
            self.set_self_blanked(connector_index, true);
            if let Err(e) = self.power_down_sink(dev, connector) {
                self.set_self_blanked(connector_index, false);
                return Err(e);
            }
        }
        vino_debug!("vino: socket {socket} blanked on the dock ({sent} black presentation(s))\n");
        Ok(())
    }
    /// Take one connector's downstream sink out of power, leaving the monitor with no signal.
    ///
    /// The dock goes on scanning out whatever it last decoded, so a driver that simply stops
    /// sending pixels leaves the panel lit on a frozen image indefinitely. Only this sequence ends
    /// that.
    fn power_down_sink(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        let socket = connector + 1;
        self.stream_marker(dev, connector, 0x2f, 1)?;
        self.stream_marker(dev, connector, 0x2e, self.sink_down_state())?;
        self.poll_status(dev)?;
        self.stream_marker(dev, connector, 0x2f, 0)?;
        vino_debug!("vino: socket {socket} downstream sink powered down\n");
        Ok(())
    }
    /// Power down every sink this driver lit, before it stops being able to.
    ///
    /// Unbinding does not reach the dock: `atomic_disable` queues the blank on the command worker,
    /// and teardown discards that queue and cancels the worker before it runs. The dock therefore
    /// keeps scanning out the last frame it decoded and the monitors stay lit on a frozen desktop
    /// until something else drives them.
    ///
    /// Best effort by construction. It runs on the disconnect path, where the device may already be
    /// physically gone -- every transfer then fails immediately, which is the right outcome -- and
    /// it must not be the reason unbinding blocks, so it does no work at all when no connector is
    /// lit and its waits are the bounded ones the mode-set path already uses.
    pub(crate) fn park_sinks(&self) {
        let lit: u32 = self
            .modeset_active
            .iter()
            .enumerate()
            .fold(0, |mask, (h, active)| {
                mask | (u32::from(active.load(Ordering::Acquire) != 0) << h)
            });
        // Nothing lit, or nothing to say it to: an unplug arrives here with the session already
        // gone, and waiting for the video path to drain against a dead dock buys nothing.
        if lit == 0 || self.check_cp_session().is_err() {
            return;
        }
        // Stand the scanout workers down and let any frame already on the wire finish, so these
        // markers do not land in the middle of one.
        self.cmd_busy
            .store(true, core::sync::atomic::Ordering::SeqCst);
        self.wait_for_video_idle();
        let Ok(link) = crate::usb_link::UsbLink::open(&self.io, self.endpoints) else {
            return;
        };
        for connector in 0..MAX_CONNECTORS {
            if lit & (1u32 << connector) == 0 {
                continue;
            }
            if let Err(e) = self.power_down_sink(&link, connector as u8) {
                vino_debug!(
                    "vino: socket {} sink power-down on unbind failed ({e:?})\n",
                    connector + 1
                );
            }
        }
    }
    /// Drive one connector's stream bracket to the closed state.
    ///
    /// The dock holds a connector opened with `2e=3` until it is told otherwise: it stops driving
    /// the sink and disengages that connector's EDID handler, so the connector then reads exactly
    /// like an empty socket. Nothing but this sequence puts it back, and the dock keeps the state
    /// across a USB re-enumeration, so it must be sent rather than inferred.
    fn send_bracket_close(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        self.stream_marker(dev, connector, 0x2f, 1)?;
        self.stream_marker(dev, connector, 0x2e, 0)?;
        self.stream_marker(dev, connector, 0x2f, 0)?;
        self.stream_marker(dev, connector, 0x2e, 0)
    }
    /// Assert the closed bracket state on a connector that is about to be probed.
    ///
    /// The dock's bracket state outlives this driver's record of it, so dedicated-pipe docks assert
    /// it before probing. Ella is different: its vendor stream never performs a periodic reset, and
    /// the four markers share EP02 with pixels. Resetting an idle socket while its sibling is live
    /// drops the live sink, so its profile defers the close until no sibling is active.
    ///
    /// Best effort otherwise: a connector that is genuinely empty is not worth failing a probe
    /// over.
    pub(super) fn close_bracket_before_probe(&self, dev: &BoundInterface<'_>, connector: u8) {
        let bit = 1u32 << connector;
        let active_connectors = self
            .modeset_active
            .iter()
            .enumerate()
            .fold(0u32, |mask, (h, active)| {
                mask | (u32::from(active.load(Ordering::Acquire) != 0) << h)
            });
        if !self
            .probe_bracket()
            .should_close(connector, active_connectors)
        {
            vino_debug!(
                "vino: socket {} probe defers bracket reset beside active sibling(s) {:#x}\n",
                connector + 1,
                active_connectors & !bit
            );
            return;
        }
        if self.send_bracket_close(dev, connector).is_ok() {
            self.blank_bracket_open.fetch_and(!bit, Ordering::AcqRel);
        }
    }
    /// Put a connector back into the closed bracket state after a failed mode set.
    ///
    /// A mode set opens the bracket before it configures anything, so an error anywhere after that
    /// point leaves the connector open on the dock with no record of it here. Best effort: this
    /// runs on the error path, and the transport that just failed may fail again.
    pub(super) fn unwind_bracket(&self, dev: &BoundInterface<'_>, connector: u8) {
        // A dead session cannot carry the close, and every queued mode set fails against it, so
        // attempting one per failure floods the log with a consequence of the disconnect rather
        // than a cause. The dock is being re-established anyway; a fresh session closes the
        // bracket in `reengage_connector` before it probes.
        if !self.cp_link_alive() {
            return;
        }
        if self.send_bracket_close(dev, connector).is_err() {
            // The dock is still holding this connector. Say so once, rather than reporting only the
            // error that got us here, because the connector now reads as an empty socket.
            pr_warn!(
                "vino: socket {socket} left open after a failed mode set; it will read as empty until a re-engage closes it\n",
                socket = connector + 1
            );
        }
    }
    /// Close a blank bracket before this connector is driven again.
    ///
    /// Precedes the EDID probe and the mode set that follow. A connector that was never blanked
    /// costs nothing here.
    pub(super) fn close_blank_bracket(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        let socket = connector + 1;
        let bit = 1u32 << connector;
        if self.blank_bracket_open.load(Ordering::Acquire) & bit == 0 {
            return Ok(());
        }
        // Clear the record only once the dock has been told, so a send that fails partway leaves
        // the debt standing for the next attempt. Clearing first strands the connector open
        // forever: the driver stops believing anything is owed while the dock goes on holding it.
        self.send_bracket_close(dev, connector)?;
        self.blank_bracket_open.fetch_and(!bit, Ordering::AcqRel);
        // Closing the bracket restores the stream, not the sink. The blank powered the downstream
        // sink down and nothing else turns it back on, so the vendor follows the closing markers
        // with a full probe, EDID fetch and sink engage before it programs a timing. A dock woken
        // without them comes back slowly, or not at all once the sink has been down long enough
        // for the dock to have let go of it.
        //
        // Best effort: a connector whose monitor genuinely went away while it was blanked belongs
        // to the presence watcher, and the mode set below is what reports a wake that failed.
        if self.reengage_connector(dev, connector).is_err() {
            vino_debug!("vino: socket {socket} wake re-engage failed; the mode set will retry\n");
        }
        // A wake is still not a cold plug: the re-engage above restores the sink, so the
        // three-second keyframe window `sustain_window` grants a cold activation buys nothing here
        // and costs about a gigabyte per connector. Sustained bandwidth is also what destabilises
        // this dock, so it is never spent twice.
        self.repair_connectors.fetch_or(bit, Ordering::Release);
        vino_debug!("vino: socket {socket} blank bracket closed; wake runs as a repair\n");
        Ok(())
    }
    /// Open the per-connector stream bracket before changing an active mode.
    pub(super) fn modeset_bracket_pre(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        self.stream_marker(dev, connector, 0x2f, 1)?;
        // Docks that want the sink torn down before it is configured say so in their profile.
        // Where no state is carried the vendor sets the mode first and brackets behind it, and
        // downing a sink that is about to be programmed would leave it down for the whole bracket.
        if let Some(state) = self.pre_mode_sink_state() {
            self.stream_marker(dev, connector, 0x2e, state)?;
        }
        self.poll_status(dev)
    }
    /// Sleep until an absolute millisecond offset from the mode-set anchor.
    ///
    /// Absolute deadlines keep scheduler delay from accumulating across the activation sequence.
    pub(super) fn wait_mode_offset(anchor: Instant<Monotonic>, target_ms: i64) {
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
    pub(super) fn wait_video_offset(anchor: Instant<Monotonic>, target_us: i64) {
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
    pub(super) fn modeset_bracket_post_open(
        &self,
        dev: &BoundInterface<'_>,
        connector: u8,
        anchor: Instant<Monotonic>,
    ) -> Result {
        self.poll_status(dev)?;
        Self::wait_mode_offset(anchor, 5);
        self.stream_marker(dev, connector, 0x2f, 1)?;
        Self::wait_mode_offset(anchor, 9);
        self.stream_marker(dev, connector, 0x2e, self.post_mode_sink_state(0))?;
        Self::wait_mode_offset(anchor, 12);
        self.stream_marker(dev, connector, 0x2f, 1)?;
        Self::wait_mode_offset(anchor, 14);
        // `0x2e` state 3 takes the downstream sink down and 0 brings it back up. The vendor drives
        // it down once here and then straight back up; repeating the 3 leaves the sink down for
        // the rest of the bracket, which on DL-3x00 is a dock that accepts every byte of a frame
        // and displays none of it.
        self.stream_marker(dev, connector, 0x2e, self.post_mode_sink_state(1))?;
        // The vendor's ring descriptor and decoder configuration land here, between the fourth
        // marker and the fifth, so the closing `2e(connector, 0)` below is the last thing the dock
        // sees before pixels. A dock told to bring its sink up after a frame has already gone out
        // has been handed that frame with nothing scanning it out.
        self.send_stream_prologue(dev, connector)?;
        Self::wait_mode_offset(anchor, 20);
        self.stream_marker(dev, connector, 0x2f, 1)?;
        // The status poll shares the final `2f(1)` deadline.
        self.poll_status(dev)?;
        Self::wait_mode_offset(anchor, 26);
        self.stream_marker(dev, connector, 0x2e, 0)?;
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
    ///
    /// A dock without a video pipe has already closed its bracket: its last marker before the
    /// strips is the sink-up, and the vendor sends nothing at all between a frame and the next
    /// frame's opener. Closing again there puts two records into the one gap the vendor leaves
    /// empty, one of them a marker state it never uses on this generation.
    pub(super) fn modeset_bracket_post_close(
        &self,
        dev: &BoundInterface<'_>,
        connector: u8,
        anchor: Instant<Monotonic>,
    ) -> Result {
        if self.video_on_ctrl_pipe() {
            return Ok(());
        }
        Self::wait_mode_offset(anchor, PROMPT_CLOSE_2F_MS);
        self.stream_marker(dev, connector, 0x2f, 0)?;
        Self::wait_mode_offset(anchor, PROMPT_CLOSE_2E_MS);
        self.stream_marker(dev, connector, 0x2e, 0)
    }
}
