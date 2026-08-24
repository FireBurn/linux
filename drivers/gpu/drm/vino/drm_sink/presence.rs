// SPDX-License-Identifier: GPL-2.0

//! What is plugged into a connector, and keeping that answer honest.
//!
//! A dock reports presence through its EDID probe rather than through any hotplug interrupt, so
//! everything here is polled and everything here can be wrong for a moment. A monitor still waking
//! up, a connector the dock has stopped answering for, and a connector this driver blanked itself
//! all look alike on the wire; telling them apart is what the debounce and repair paths do.

use super::*;

impl VinoDrmData {
    /// Cache a connector's downstream EDID (read during probe). Bring-up publishes all connectors
    /// with one hotplug only after both presence and EDID state are complete; firing here exposed
    /// KWin to a transient no-EDID mode list (including synthetic 1920x1440) before the real EDID
    /// arrived. Out-of-range connectors are ignored.
    pub(crate) fn set_edid(&self, connector: usize, blob: KVec<u8>) {
        let mut edids = self.cached_edids.lock();
        let Some(slot) = edids.get_mut(connector) else {
            return;
        };
        *slot = Some(blob);
    }
    /// Whether userspace has taken responsibility for describing `connector`'s sink, because the
    /// dock cannot read it.
    ///
    /// A DP-to-HDMI converter that mangles or drops DDC leaves the dock unable to read the monitor
    /// at all: the presence probe reports the socket occupied, but no `id=0x194` ever arrives, so
    /// the connector stays disconnected and nothing is ever driven. The monitor is real, and its
    /// EDID is readable from a working port on another machine.
    ///
    /// Rather than carry a second EDID source, this hands the connector to DRM's own override,
    /// which already accepts a blob two ways -- `drm_kms_helper.edid_firmware=<connector>:<path>`
    /// or a write to the connector's debugfs `edid_override`. The core applies it only to a
    /// connector that is connected and whose `get_modes` returned none, so all this flag does is
    /// make the connector report no modes of its own while no EDID has been read.
    ///
    /// It deliberately does not report the connector connected. Connecting a modeless connector
    /// hands fbdev emulation a blank cheque, and its 640x480 default mode set puts this dock into a
    /// 25 s re-enumeration loop. Userspace supplies the description and then forces the connector
    /// on (`echo on > .../status`), in that order, which is also what makes the sequence race-free.
    ///
    /// An override describes the sink, not the link. A blob claiming a mode the converter cannot
    /// carry gives a black screen just the same: this substitutes for a broken read, it does not
    /// negotiate anything.
    pub(crate) fn edid_from_userspace(&self, connector: usize) -> bool {
        usize::from(*crate::module_parameters::edid_override.value()) & (1 << connector) != 0
    }
    /// Mark a connector connected from CP engagement alone (no raw EDID). Bring-up fires one
    /// hotplug after every connector's EDID has also been cached, so the compositor never probes
    /// partial state. Called once the connector's DISPLAY-CAP push confirms monitor presence.
    pub(crate) fn set_connected(&self, connector: usize) {
        if connector >= MAX_CONNECTORS {
            return;
        }
        self.connectors_present
            .fetch_or(1 << connector, core::sync::atomic::Ordering::Release);
    }
    /// Clear a connector's presence bit and cached EDID after monitor removal.
    ///
    /// `detect()` reports connected when either exists, so both must be cleared together.
    pub(crate) fn set_disconnected(&self, connector: usize) {
        let socket = connector + 1;
        if connector >= MAX_CONNECTORS {
            return;
        }
        // Tagged with the dock's connector count, which is what tells two bound docks apart in a
        // single log.
        if self.connectors_present.load(Ordering::Acquire) & (1u32 << connector) != 0 {
            vino_debug!(
                "vino: {}-connector dock: socket {socket} presence cleared\n",
                self.connector_count()
            );
        }
        self.connectors_present
            .fetch_and(!(1u32 << connector), core::sync::atomic::Ordering::Release);
        if let Some(slot) = self.cached_edids.lock().get_mut(connector) {
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
    /// Re-run one connector's EDID probe, fetch, engage, and capability query on the live CP link.
    ///
    /// Monitor removal tears down the dock's downstream sink, so a replug requires the engage
    /// messages before a later mode set can start its pixel clock. The cached EDID is cleared on
    /// removal and must be repopulated here before reporting the connector as present. Returns true
    /// only when a valid EDID was received.
    pub(crate) fn reengage_connector(
        &self,
        io: &BoundInterface<'_>,
        connector: u8,
    ) -> Result<bool> {
        let socket = connector + 1;
        self.set_self_blanked(connector as usize, false);
        // A connector that is not answering is exactly a connector that may be sitting in an open
        // bracket, where the dock has disengaged its EDID handler and every probe below would go
        // unanswered. The state is the dock's and it survives a re-enumeration, so a fresh session
        // cannot know it is owed: assert the closed state rather than infer it. A connector already
        // closed ignores this.
        self.close_bracket_before_probe(io, connector);
        self.edid_target.store(connector as u32, Ordering::Release);
        *self.edid_caught.lock() = None;
        let result = (|| -> Result {
            self.send_reengage_step(io, 0x15, 117, |c| {
                crate::cp::get_edid_req_sub(c, 0x0020, connector)
            })?;
            self.send_reengage_step(io, 0x15, 115, |c| {
                crate::cp::get_edid_req_sub(c, 0x0020, connector)
            })?;
            self.send_reengage_step(io, 0x16, 107, |c| {
                crate::cp::edid_readiness_kick(c, connector)
            })?;
            self.send_reengage_step(io, 0x15, 11, |c| crate::cp::get_edid_req(c, connector))?;
            self.send_reengage_step(io, 0x16, 118, |c| crate::cp::edid_engage_req(c, connector))?;
            self.send_reengage_step(io, 0x16, 107, |c| crate::cp::edid_engage_req(c, connector))?;
            self.send_reengage_step(io, 0x15, 11, |c| crate::cp::post_edid_query(c, connector))
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
                    vino_debug!(
                        "vino: socket {socket} EDID {n} B, vendor {}{}{} product {:#06x}\n",
                        vendor[0] as char,
                        vendor[1] as char,
                        vendor[2] as char,
                        u16::from_le_bytes([blob[10], blob[11]])
                    );
                }
                self.set_edid(connector as usize, blob);
                Ok(true)
            }
            None => {
                // Deliberately not published here, even under `edid_override`. A connector that
                // reports connected with no modes is immediately mode-set by fbdev emulation at
                // its own 640x480 default, and driving that at the dock resets it -- measured, in
                // a 25 s re-enumeration loop. The connector stays disconnected until userspace has
                // supplied the description AND forced the connector on; see `edid_from_userspace`.
                if self.edid_from_userspace(connector as usize) {
                    pr_warn!(
                        "vino: socket {socket} has no EDID from the dock and is waiting for one from \
                         userspace (edid_override); it stays disconnected until then\n"
                    );
                    return Ok(false);
                }
                vino_debug!(
                    "vino: socket {socket} re-engaged but no EDID came back -- no monitor, or it is \
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
        self.check_cp_session().ok()?;
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
                crate::cp::parse_edid_from_reply(&link.ks, &link.riv, &reply[..got])
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
    pub(crate) fn probe_connector_present(
        &self,
        dev: &BoundInterface<'_>,
        connector: u8,
    ) -> Option<bool> {
        if usize::from(connector) >= self.connector_count() {
            return Some(false);
        }
        self.send_presence_probe(dev, connector, connector)
    }
    /// Probe one downstream connector. `connector` selects its own presence-change cell.
    ///
    /// Sends the EDID probe (`id=0x15 sub=0x20`, byte22 = connector selector -- the same selector
    /// that unblocked the whole EDID path) and decodes the dock's sealed `0x45` reply. Returns
    /// `Some(true/false)` on a decodable reply, `None` if CP is down or nothing decoded. Reuses the
    /// live session `ks/riv/counter` exactly like `send_cp`, so it stays in CP lockstep.
    fn send_presence_probe(
        &self,
        dev: &BoundInterface<'_>,
        sel: u8,
        connector: u8,
    ) -> Option<bool> {
        let socket = connector + 1;
        self.check_cp_session().ok()?;
        let mut guard = self.cp_link.lock();
        let link = (&mut *guard).as_mut()?;
        let request_counter = link.counter;
        let msg = crate::cp::get_edid_req_sub(request_counter, 0x0020, sel).ok()?;
        let frame =
            crate::cp::seal_interactive(&link.ks, &link.riv, 0x15, link.wire_seq, &msg).ok()?;
        if dev.ctrl_send(&frame, crate::timeout(), GFP_KERNEL).is_err() {
            let silent_ms = self.cp_silent_for_ms();
            if silent_ms >= self.cp_silence_limit_ms() {
                self.abandon_cp_session(silent_ms);
                *guard = None;
            }
            return None;
        }
        link.wire_seq = link.wire_seq.wrapping_add(((msg.len() + 15) / 16) as u32);
        link.counter = link.counter.wrapping_add(1);
        // Take the reply that answers this probe, not simply the next frame on EP84: the connectors
        // are probed back to back, so a late reply or an unprompted push would otherwise be
        // attributed to the wrong connector. The inner counter echoes the request. A round that
        // never sees its own echo returns `None`, which the caller treats as "this poll learned
        // nothing" rather than as an unplug.
        let mut reply = KVec::from_elem(0u8, 4096, GFP_KERNEL).ok()?;
        let deadline = Instant::<Monotonic>::now() + Delta::from_millis(64);
        let got = loop {
            let n = match link.ep84_q.as_mut() {
                Some(q) => match q.recv(dev.io(), &mut reply, crate::cp_reply_timeout()) {
                    Ok(Some(n)) => n,
                    Ok(None) => 0,
                    Err(_) => return None,
                },
                None => dev
                    .ctrl_recv(&mut reply, crate::cp_reply_timeout(), GFP_KERNEL)
                    .unwrap_or(0),
            };
            if n > 16 {
                match crate::cp::decode_in_lenient(&link.ks, &link.riv, &reply[..n]) {
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
        let (id, status, ready) =
            crate::cp::probe_reply_status(&link.ks, &link.riv, &reply[..got])?;
        self.note_cp_reply();
        // Presence is bit 0x10 of inner byte 23, which lands in bits 8..15 of the status word:
        // `05 11 27 00` for an occupied connector, `05 01 <20|21|60|61> 00` for an empty one.
        // Which handler answered says nothing about it -- both docks reply `id=0x44` either way.
        let present = status & 0x0000_1000 != 0;
        // One line per *changed* answer per connector, so a steady link is silent and an unplug is
        // unmissable. Both fields are packed into the same cell: the id alone cannot distinguish a
        // dock that keeps saying `0x44` from one whose downstream state has moved underneath it.
        let cell = ((id as u32) << 16) | (status & 0xffff);
        let prev = self.presence_reply[connector as usize].swap(cell, Ordering::Relaxed);
        if prev != cell {
            // The dock moves the *other* connector's status word too when a sink appears or
            // disappears, and it does so sooner than it pushes anything. `prev == 0` is this
            // connector's first ever reply, which is bring-up, not an event.
            if prev != 0 {
                self.downstream_event.store(true, Ordering::Release);
            }
            // The decoded answer itself, not just the verdict derived from it. Without this a
            // presence flap can only be read as "monitor disconnected", which says nothing about
            // whether the dock changed its mind or vino changed the question. It is one line per
            // *changed* reply per connector, so a steady link prints nothing at all.
            // Tagged with this dock's video endpoints, which name the family uniquely: 08/0b is
            // a DL-6xxx, 08/0a a DL-7400, 02/02 a DL-3x00. A connector count does not -- two of
            // the three have two connectors, so once both are bound the tag distinguishes
            // nothing and the same reading reads as either dock.
            vino_debug!(
                "vino: [video {:02x}/{:02x}] socket {socket} presence reply id={id:#06x} \
                 status={status:#010x} -> present={present} ready={ready} \
                 (was id={:#06x} status={:#06x})\n",
                dev.endpoints.video[0].address(),
                dev.endpoints.video[1].address(),
                prev >> 16,
                prev & 0xffff
            );
        }
        Some(present)
    }
    /// Whether vino itself took `connector`'s sink down, so the presence watcher can tell its own
    /// blank apart from a real unplug. See [`VinoDrmData::self_blanked`].
    pub(crate) fn is_self_blanked(&self, connector: usize) -> bool {
        self.self_blanked.load(Ordering::Acquire) & (1u32 << connector) != 0
    }
    pub(crate) fn set_self_blanked(&self, connector: usize, on: bool) {
        if on {
            self.self_blanked
                .fetch_or(1u32 << connector, Ordering::Release);
        } else {
            self.self_blanked
                .fetch_and(!(1u32 << connector), Ordering::Release);
        }
    }
    /// How long this connector should hold the post-mode-set training cadence.
    ///
    /// A cold activation needs it: the dock will not program its downstream pixel clock without a
    /// sustained stream. A repair does not -- the link is already trained and the sink was dropped
    /// underneath us -- and running it there costs the dock three seconds of full keyframes at
    /// `FRAME_PERIOD_MS`. Consumes the repair flag, so the window returns for the next real
    /// bring-up.
    pub(crate) fn sustain_window(&self, connector: usize) -> Option<Instant<Monotonic>> {
        let bit = 1u32 << connector;
        if self.repair_connectors.fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
            return None;
        }
        // A dock whose video shares the control pipe cannot be given this window. It exists to
        // train a downstream link by presenting keyframes at frame cadence, which on a pipe of its
        // own is bandwidth well spent; here it is bandwidth taken directly from the control plane,
        // and the dock stops answering EP84 entirely rather than merely dropping frames.
        if self.video_on_ctrl_pipe() {
            return None;
        }
        Some(Instant::<Monotonic>::now() + Delta::from_millis(SUSTAIN_MS))
    }
    /// Re-drive every lit connector after the dock dropped a downstream sink underneath us.
    ///
    /// The presence flap is the dock really taking a sink down, and the only thing that used to
    /// repair it was letting the DRM connector disappear so the compositor would re-enable the
    /// output. That cure was worse than the disease: it re-lays-out userspace, and the mode set it
    /// produces names one connector while its sibling is lit, which re-enumerates the dock. So vino
    /// repairs the connector itself and leaves the connector alone.
    ///
    /// Every lit connector is re-queued, not just the one that flapped, and they go into one batch:
    /// a mode set this dock accepts is one that names every connector at once
    /// (`activate_dual_wake`). Zeroing the mode generation is what makes that path take them.
    ///
    /// Returns the number of connectors queued.
    #[expect(
        dead_code,
        reason = "kept for the flap-repair experiment; see its doc comment"
    )]
    pub(super) fn repair_flapped_connector(&self, dev: &VinoDrmDevice, flapped: usize) -> u32 {
        let mut queued = 0u32;
        let mut cmds: [Option<crate::cp::Timing>; MAX_CONNECTORS] = [None; MAX_CONNECTORS];
        for connector in 0..MAX_CONNECTORS {
            let active = self.modeset_active[connector].load(Ordering::Acquire);
            if active == 0 || self.modeset_requested[connector].load(Ordering::Acquire) != active {
                continue;
            }
            let Some(timing) = self.last_timing.lock()[connector] else {
                continue;
            };
            if timing_key(&timing) != active {
                continue;
            }
            cmds[connector] = Some(timing);
        }
        if cmds.iter().flatten().count() == 0 {
            return 0;
        }
        for connector in 0..MAX_CONNECTORS {
            let Some(timing) = cmds[connector] else {
                continue;
            };
            // The dock's copy of this connector is gone, so nothing it holds can be diffed against.
            self.repair_connectors
                .fetch_or(1u32 << connector, Ordering::Release);
            self.modeset_active[connector].store(0, Ordering::Release);
            self.owe_keyframe(connector);
            self.strip_hashes.lock()[connector] = None;
            self.dirty_ttl.lock()[connector] = None;
            self.queue_cmd(
                dev,
                KmsCmd::ModeSet {
                    connector: connector as u8,
                    timing,
                },
            );
            queued += 1;
        }
        pr_warn!("vino: connector {flapped} sink flap -- re-driving {queued} lit connector(s) together\n");
        queued
    }
    /// Whether connector `connector`'s presence bit is currently set (for the keepalive to seed its
    /// baseline before watching for runtime connect/remove transitions). Whether a monitor has
    /// described itself on this socket.
    ///
    /// The dock recovers an EDID for a socket with something plugged into it and nothing at all for
    /// an empty one, which on a family that cannot report downstream presence is the only presence
    /// signal there is.
    pub(crate) fn connector_has_edid(&self, connector: usize) -> bool {
        self.cached_edids
            .lock()
            .get(connector)
            .is_some_and(Option::is_some)
    }
    pub(crate) fn connector_present(&self, connector: usize) -> bool {
        connector < MAX_CONNECTORS
            && self
                .connectors_present
                .load(core::sync::atomic::Ordering::Acquire)
                & (1u32 << connector)
                != 0
    }
}
