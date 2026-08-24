// SPDX-License-Identifier: GPL-2.0

//! The control-plane session: sending sealed messages and keeping the link alive.
//!
//! The control plane is host-driven lockstep. Every message carries an authenticated counter that
//! both ends step together, so a message sent out of turn, or a reply left undrained,
//! desynchronises the session and every later message decrypts to nothing. That is why sending is
//! serialised here rather than at the call sites.

use super::*;

impl VinoDrmData {
    pub(crate) fn set_cp_engaged(&self, engaged: bool) {
        self.cp_engaged
            .store(engaged, core::sync::atomic::Ordering::SeqCst);
    }
    /// Whether the control session is still usable.
    ///
    /// Reads the flag, not the mutex: the caller is usually asking precisely because the link may
    /// be stuck, and the stuck thread is holding that mutex.
    pub(crate) fn cp_link_alive(&self) -> bool {
        self.cp_session_live.load(Ordering::Acquire)
    }
    /// Record that the dock answered. Resets the silence deadline.
    pub(super) fn note_cp_reply(&self) {
        *self.cp_last_reply.lock() = Instant::<Monotonic>::now();
    }
    /// How long the dock has answered nothing, in milliseconds.
    pub(super) fn cp_silent_for_ms(&self) -> i64 {
        (Instant::<Monotonic>::now() - *self.cp_last_reply.lock()).as_millis()
    }
    /// Give up on the control session. Idempotent, and logs once.
    ///
    /// Only the flag is cleared here. Callers that hold `cp_link` drop its contents themselves;
    /// the watchdog cannot, because the mutex is exactly what a wedged transfer is holding. Every
    /// path consults the flag before the mutex, so an orphaned [`CpLink`] is unreachable, and
    /// [`Self::shutdown`] frees it with the rest of the session state.
    pub(super) fn abandon_cp_session(&self, silent_ms: i64) {
        if self
            .cp_session_live
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            pr_warn!(
                "vino: dock has answered nothing for {silent_ms} ms; abandoning the session\n"
            );
        }
    }
    /// Gate every control transaction, without taking `cp_link`.
    ///
    /// Checking the deadline here rather than under the mutex is the difference between noticing
    /// the dock has gone and queueing another thread behind the transfer that proves it.
    pub(super) fn check_cp_session(&self) -> Result {
        // Teardown first. `disconnect()` waits for the workers, and a worker that starts a control
        // transfer to a device already being disconnected waits for a completion that will not
        // come; on a dock whose video shares this endpoint the scanout path issues those transfers
        // too, so the window is wide. The pair deadlocks `usb_hub_wq` inside `usb_disconnect()`,
        // which stops USB hotplug machine-wide and is only recoverable by rebooting.
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(ENODEV);
        }
        if !self.cp_link_alive() {
            return Err(ENODEV);
        }
        let silent_ms = self.cp_silent_for_ms();
        if silent_ms >= self.cp_silence_limit_ms() {
            self.abandon_cp_session(silent_ms);
            return Err(ETIMEDOUT);
        }
        Ok(())
    }
    /// How long this dock may say nothing before its session is abandoned.
    pub(super) fn cp_silence_limit_ms(&self) -> i64 {
        if self.video_on_ctrl_pipe() {
            CP_SILENCE_LIMIT_SHARED_MS
        } else {
            CP_SILENCE_LIMIT_MS
        }
    }
    /// Ask the USB core to reset the dock after the control session has been abandoned.
    ///
    /// Without this the outputs stay down until the user unplugs the dock, because a session can
    /// only be established through probe. It is the one recovery a stuck transfer cannot block:
    /// the USB core runs it from its own work item. What turns it into a fresh session is
    /// `post_reset` asking for the interface to be rebound -- a reset on its own leaves the
    /// driver bound with nothing to drive.
    pub(super) fn reset_after_wedge(&self) {
        if self
            .cp_reset_queued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        // A closed window means unplug is already under way, which needs no help from us.
        let Ok(io) = self.io.enter() else {
            return;
        };
        pr_warn!("vino: resetting the dock to recover the control session\n");
        io.interface().queue_reset_device();
    }
    /// Whether the current device session has engaged content protection.
    pub(crate) fn cp_engaged(&self) -> bool {
        self.cp_engaged.load(core::sync::atomic::Ordering::Acquire)
    }
    /// Pause the background CP loop and let any iteration which passed its check finish. The mode
    /// worker calls this before taking its timestamp anchor; the fixed delay therefore cannot move
    /// any event relative to the mode-set itself.
    pub(super) fn begin_cp_timeline(&self) {
        self.cp_timeline_exclusive.store(true, Ordering::Release);
        self.initial_modeset_quiet.store(false, Ordering::Release);
        fsleep(Delta::from_millis(PROMPT_KEEPALIVE_QUIESCE_MS));
    }
    pub(super) fn end_cp_timeline(&self) {
        self.cp_timeline_exclusive.store(false, Ordering::Release);
    }
    /// Used by `BringUp`'s long-lived keepalive worker. It deliberately remains cheap because that
    /// worker checks it every millisecond while an activation sequence is in progress.
    pub(crate) fn cp_timeline_exclusive(&self) -> bool {
        self.cp_timeline_exclusive.load(Ordering::Acquire)
    }
    /// Publish the engaged CP session so the KMS callbacks can send runtime CP messages.
    /// Called once by the bring-up work item after the dock acks (`acks > 0`). `wire_seq`/
    /// `counter` are the next free values past the bring-up CP setup.
    pub(crate) fn publish_session(
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
        self.note_cp_reply();
        self.cp_session_live.store(true, Ordering::Release);
    }
    /// Start the silence watchdog for the session just published.
    ///
    /// Separate from [`Self::publish_session`] because it needs an `ARef` to the device, and it
    /// re-arms itself for the lifetime of the session.
    pub(crate) fn start_cp_watchdog(&self, drm_dev: &VinoDrmDevice) {
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let delay = kernel::time::msecs_to_jiffies(CP_WATCHDOG_PERIOD_MS);
        let _ = workqueue::system().enqueue_delayed::<_, 5>(ARef::from(drm_dev), delay);
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
        self.check_cp_session()?;
        self.send_cp_locked(dev, id, tag_reserved, reserved_counter, build, consume)
    }
    #[allow(clippy::too_many_arguments)]
    fn send_cp_locked<T>(
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
        // exception: per-connector workers reserve counters before their writes interleave, so the
        // inner counter order differs from the monotonically advancing AES block sequence.
        let request_counter = reserved_counter.unwrap_or(link.counter);
        let msg = build(request_counter)?;
        let inner_sub = if msg.len() >= 4 {
            u16::from_le_bytes([msg[2], msg[3]])
        } else {
            0
        };
        let content = &msg[..msg.len().saturating_sub(tag_reserved)];
        let frame = crate::cp::seal_interactive(&link.ks, &link.riv, id, link.wire_seq, content)?;
        let pipe = self.own_pipe();
        // A shared video failure publishes the session dead while a CP writer may already be
        // waiting for this pipe behind the failed frame.  Recheck after acquiring it so that
        // waiter cannot issue one last control record into the terminal stream before reset.
        if !self.cp_link_alive() {
            return Err(ENODEV);
        }
        if let Err(e) = dev.ctrl_send(&frame, crate::timeout(), GFP_KERNEL) {
            let silent_ms = self.cp_silent_for_ms();
            if silent_ms >= self.cp_silence_limit_ms() {
                self.abandon_cp_session(silent_ms);
                *guard = None;
            }
            return Err(e);
        }
        // The reply arrives on the other endpoint, so the pipe is free again the moment the
        // request is out: holding it across the wait would stall a frame for the whole timeout.
        drop(pipe);
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
        // Navarro also emits unprompted `id=2/sub=0x86` status pushes on the same endpoint;
        // treating the first such push as the paired reply advances EP02 before the dock has
        // completed the transaction.  The dock then NAKs that write until the real reply is reaped,
        // which is the exact 100-ms staircase visible in the failed captures.  Consume pushes here
        // and stop only at the echoed counter (or a bounded timeout for request classes which do
        // not reply).
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
                match q.recv(dev.io(), &mut reply, crate::cp_reply_timeout()) {
                    Ok(Some(n)) => n,
                    Ok(None) => 0,
                    Err(_) => break,
                }
            } else {
                dev.ctrl_recv(&mut reply, crate::cp_reply_timeout(), GFP_KERNEL)
                    .unwrap_or(0)
            };
            if got > 16 {
                // During re-engagement an EDID push can precede the paired acknowledgment. Keep
                // it for the waiting connector while continuing to wait for the echoed counter.
                let target = self.edid_target.load(Ordering::Relaxed);
                if target != NO_EDID_TARGET {
                    if let Ok(Some(blob)) =
                        crate::cp::parse_edid_from_reply(&link.ks, &link.riv, &reply[..got])
                    {
                        *self.edid_caught.lock() = Some(blob);
                    }
                }
                if let Some((reply_id, reply_sub, reply_counter)) =
                    crate::cp::decode_in_lenient(&link.ks, &link.riv, &reply[..got])
                {
                    if reply_counter == request_counter {
                        matched = got;
                        break;
                    }
                    seen_id = reply_id;
                    seen_sub = reply_sub;
                    seen_counter = reply_counter;
                    if reply_id == 0x44 || crate::cp::edid_reply_len(reply_id).is_some() {
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
            vino_debug!(
                "vino: unanswered id={id:#06x} sub={inner_sub:#06x} ctr={request_counter}: reaped {reaped} reply/replies, {undecodable} undecodable, last decoded id={seen_id:#06x} sub={seen_sub:#06x} ctr={seen_counter}\n"
            );
            let silent_ms = self.cp_silent_for_ms();
            if silent_ms >= self.cp_silence_limit_ms() {
                self.abandon_cp_session(silent_ms);
                *guard = None;
                return Err(ETIMEDOUT);
            }
        } else {
            self.note_cp_reply();
        }
        consume(&link.ks, &link.riv, &reply[..matched])
    }
    /// Seal and send one interactive CP message on EP02, advancing the session keystream.
    pub(crate) fn send_cp(
        &self,
        dev: &BoundInterface<'_>,
        id: u16,
        tag_reserved: usize,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
    ) -> Result {
        self.send_cp_reply(dev, id, tag_reserved, None, build, |_, _, _| Ok(()))
    }
    /// Send one CP message using a counter token previously consumed from the live allocator.
    pub(super) fn send_cp_reserved(
        &self,
        dev: &BoundInterface<'_>,
        id: u16,
        inner_counter: u16,
        build: impl FnOnce(u16) -> Result<KVec<u8>>,
    ) -> Result {
        self.send_cp_reply(dev, id, 0, Some(inner_counter), build, |_, _, _| Ok(()))
    }
    /// Consume `N` consecutive counters from the live session and return them as reservation
    /// tokens. This models DLM's independently queued per-connector workers: allocation order
    /// remains monotonic even when their actual EP02 writes interleave in a different order.
    pub(super) fn reserve_cp_counters<const N: usize>(&self) -> Result<[u16; N]> {
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
    pub(crate) fn drain_cp_pushes(&self, dev: &BoundInterface<'_>, max: usize) -> usize {
        if !self.cp_link_alive() {
            return 0;
        }
        // Best-effort really does mean best-effort: if a transaction owns the link there is
        // nothing to reap that it is not already reaping, and blocking here would put the
        // keepalive behind a transfer that may never return.
        let Some(mut guard) = self.cp_link.try_lock() else {
            return 0;
        };
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
                    // An unprompted push is the dock answering. The silence deadline asks whether
                    // it is talking at all, not whether it is answering us in particular, and on
                    // an idle lit link the heartbeats are most of what it says.
                    self.note_cp_reply();
                    if let Some((id, _, _)) =
                        crate::cp::decode_in_lenient(&link.ks, &link.riv, &reply[..len])
                    {
                        // An EDID-handler reply arriving with no probe outstanding is the dock
                        // reporting that a downstream sink changed -- it is the *only* thing it
                        // sent between a measured monitor replug and its own give-up reset. Treat
                        // it as "re-probe now" rather than waiting out the presence period.
                        if id == 0x44 || crate::cp::edid_reply_len(id).is_some() {
                            self.downstream_event.store(true, Ordering::Release);
                        }
                    }
                }
                _ => break,
            }
        }
        n
    }
}
