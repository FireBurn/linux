// SPDX-License-Identifier: GPL-2.0

//! Publishing desired state to the asynchronous workers.
//!
//! Atomic callbacks run in contexts that may not sleep or touch USB, so they record what the
//! dock should be doing and wake a worker. Each operation class owns one slot, which makes an
//! update infallible and lets a stale cursor position or stream state be overwritten rather
//! than queued behind the state that replaced it.

use super::*;

impl VinoDrmData {
    /// Publish the latest desired operation for a connector and wake the async worker.
    ///
    /// Each operation class has one fixed slot, so updates cannot fail allocation and obsolete
    /// cursor positions or stream states do not build a backlog.
    pub(super) fn queue_cmd(&self, dev: &VinoDrmDevice, cmd: KmsCmd) {
        let mut pending = self.pending_kms.lock();
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        pending.update(cmd);
        // Registration precedes the blocking encrypted setup and platform readiness interval.
        // Retain and coalesce commands that arrive there, but do not let them touch the dock.
        if !self.kms_activation_ready() {
            return;
        }
        // Enqueue while the queue lock still serializes us with `shutdown()`. Otherwise shutdown
        // could cancel an idle work item between this unlock and enqueue, leaving a late work-owned
        // device reference behind after teardown.
        //
        // `::<_, 0>` names `cmd_work`. The ID is only inferrable while a single `WorkItem` impl
        // exists; adding the per-connector scanout items made every bare `enqueue` ambiguous, which
        // is exactly the failure mode you want here -- an unannotated enqueue would otherwise be
        // free to pick the wrong worker.
        let _ = self.kms_queue.enqueue::<_, 0>(ARef::from(dev));
        drop(pending);
    }

    /// Publish the end of bring-up and wake transport state retained while it ran.
    pub(crate) fn publish_kms_activation_ready(&self, dev: &VinoDrmDevice) {
        let pending = self.pending_kms.lock();
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        // Share `pending_kms` with `queue_cmd` as the readiness/wakeup handshake: either this sees
        // a retained command, or a later producer sees readiness and enqueues the work itself.
        self.kms_activation_ready.store(true, Ordering::Release);
        if !pending.is_empty() {
            let _ = self.kms_queue.enqueue::<_, 0>(ARef::from(dev));
        }
        drop(pending);
        // Plane state may also have arrived after CP engagement but before activation readiness.
        // Its worker gates on the same flag and leaves the coalesced frame in place until this
        // wake.
        self.enqueue_scanout_all(dev);
    }

    /// Publish the latest framebuffer for one connector and wake the same deferred worker used by
    /// the blocking runtime CP commands. Replacing an unsent flip is deliberate backpressure: the
    /// dock needs the newest desktop, not every historical compositor buffer. If damaged flips are
    /// coalesced, carry the unsent damage into the newest framebuffer so no intermediate update is
    /// lost without needlessly promoting every busy compositor interval to a full-screen refresh.
    pub(super) fn queue_scanout(
        &self,
        dev: &VinoDrmDevice,
        fb: &kms::framebuffer::Framebuffer<VinoDrmDriver>,
        mut frame: PendingScanout,
    ) {
        let connector = frame.connector as usize;
        let socket = connector + 1;
        if connector >= MAX_CONNECTORS || self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        // Do not snapshot faster than the encoder consumes. An unclaimed frame in the coalescing
        // slot means the worker has not caught up, so this snapshot would be overwritten before it
        // was ever read -- and it is not free: it runs on the compositor's atomic-commit thread and
        // reads the whole source to hash it. On a busy machine the encoder falls behind by many
        // commits, and paying that read for every one of them stalls the compositor itself, on
        // every output it drives rather than only this one.
        //
        // Nothing is lost by dropping this flip. Damage is decided by comparing strip hashes
        // against the last *encoded* baseline, never by the compositor's damage clips, so whatever
        // this commit changed is still described by the next snapshot that gets through. The frame
        // the worker eventually takes is at most one encode period old, which is the pacing the
        // hardware imposes anyway.
        //
        // Geometry changes and owed keyframes are never dropped: the first makes the pending
        // frame's damage coordinates meaningless, and the second is a mode set waiting on current
        // content rather than an ordinary repaint.
        let coalesce = {
            let pending = self.pending_scanout.lock();
            pending[connector].as_ref().is_some_and(|queued| {
                queued.w == frame.w
                    && queued.h == frame.h
                    && queued.rotation == frame.rotation
                    && self.keyframe_pending.load(Ordering::Acquire) & (1u32 << connector) == 0
            })
        };
        if coalesce {
            vino_debug!("vino: socket {socket} flip coalesced before snapshot\n");
            return;
        }
        // The snapshot below is format-agnostic -- both layouts are four bytes per pixel -- so the
        // depth only has to be recorded, not acted on, before the copy.
        if let Some(depth) = crate::video::haar::Depth::from_fourcc(fb.format()) {
            self.set_connector_depth(frame.connector, depth);
        }
        let (source_w, source_h) = src_dims(frame.rotation, frame.w, frame.h);
        // Reserve a slot and lend its surface out, so the ~14.7 MB copy below runs with the pool
        // lock dropped. Holding it across the copy put this connector's scanout worker into
        // `mutex_spin_on_owner` for the whole snapshot -- 5.4% of the machine, burnt spinning.
        let (mut surface, binding, idx) = {
            let mut pool = self.shadow[connector].lock();
            // Rotate, rather than always taking the first free slot. `find` returned slot 0 on
            // every commit whenever nothing was inflight, so consecutive snapshots overwrote the
            // same slot and bumped its generation -- invalidating any frame the worker had already
            // selected from it, which showed up as a third of all frames being dropped at the
            // generation check. Alternating means a fresh snapshot lands clear of the frame the
            // worker is about to pick up.
            let start = self.shadow_rr[connector].fetch_add(1, Ordering::Relaxed) as usize;
            let Some(idx) = (0..SHADOW_SLOTS)
                .map(|i| (start + i) % SHADOW_SLOTS)
                .find(|&idx| pool.inflight != Some(idx) && pool.writing != Some(idx))
            else {
                return;
            };
            let binding = match pool.source_bindings.get(fb) {
                Ok(binding) => binding,
                Err(e) => {
                    pr_warn!("vino: socket {socket} framebuffer binding failed ({e:?})\n");
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
            let mut pool = self.shadow[connector].lock();
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
                pr_warn!("vino: socket {socket} framebuffer snapshot failed ({e:?})\n");
                return;
            }
        };
        frame.shadow_idx = idx;
        frame.shadow_generation = generation;

        // A real flip carries newer content than an armed repaint.
        self.settle_repaint.lock()[connector] = None;

        let mut pending = self.pending_scanout.lock();
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        if let Some(old) = pending[connector].take() {
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
        pending[connector] = Some(frame);
        self.enqueue_scanout(dev, connector);
        drop(pending);
    }

    /// Wake `connector`'s scanout worker. The work ID is a const generic, so the runtime connector
    /// index has to be matched into it here. Enqueueing an already-pending item is a no-op, and
    /// enqueueing one that is currently running re-arms it, preserving a flip that arrives during
    /// encoding for the worker's next pass.
    pub(super) fn enqueue_scanout(&self, dev: &VinoDrmDevice, connector: usize) {
        match connector {
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
    pub(super) fn wait_for_video_idle(&self) {
        use core::sync::atomic::Ordering::SeqCst;
        for _ in 0..500 {
            if !self.video_inflight.iter().any(|f| f.load(SeqCst)) {
                return;
            }
            fsleep(Delta::from_millis(1));
        }
        pr_warn!("vino: timed out waiting for in-flight scanout before a mode-set; proceeding\n");
    }

    /// Wake every connector's scanout worker. Used by `cmd_work` once its batch is done, since a
    /// command batch is exactly what makes the scanout workers bail (see [`run_scanout_worker`]).
    pub(crate) fn enqueue_scanout_all(&self, dev: &VinoDrmDevice) {
        for connector in 0..MAX_CONNECTORS {
            self.enqueue_scanout(dev, connector);
        }
    }

    /// Record that `connector` owes a full keyframe, and refill its settle-repaint budget.
    ///
    /// Mode sets, output enables, and gamma changes use this path. Training
    /// and settle repaints may re-raise the keyframe bit without refilling
    /// the budget, which bounds idle keyframe generation.
    pub(super) fn owe_keyframe(&self, connector: usize) {
        self.keyframe_pending
            .fetch_or(1u32 << connector, Ordering::Release);
        self.settle_budget[connector].store(SETTLE_REPAINTS, Ordering::Relaxed);
        // Whatever left the dock's framebuffer undefined left its cursor bitmap undefined too, so
        // the two invalidations are raised together. Keeping them in one place is deliberate:
        // their being separate is exactly how the cursor came to be dropped on a mode-set.
        self.cursor_epoch[connector].fetch_add(1, Ordering::Release);
        self.cursor_geometry.lock()[connector] = None;
    }

    /// Note a cursor command the dock has just accepted, so [`Self::rearm_cursor`] can replay it.
    pub(super) fn record_cursor(&self, cmd: &KmsCmd) {
        let connector = cmd.connector();
        if connector >= MAX_CONNECTORS {
            return;
        }
        let mut slots = self.cursor_shot.lock();
        match cmd {
            KmsCmd::CursorImage { w, h, bgra, .. } => {
                let mut copy = KVec::new();
                if copy.extend_from_slice(bgra, GFP_KERNEL).is_err() {
                    // A cursor that cannot be cached is still on the dock; it just will not be
                    // restored across the next mode set. Nothing else depends on this.
                    return;
                }
                match &mut slots[connector] {
                    Some(shot) => {
                        shot.w = *w;
                        shot.h = *h;
                        shot.bgra = copy;
                    }
                    slot @ None => {
                        *slot = Some(CursorShot {
                            w: *w,
                            h: *h,
                            bgra: copy,
                            x: 0,
                            y: 0,
                            visible: false,
                        })
                    }
                }
            }
            KmsCmd::CursorMove { x, y, visible, .. } => {
                if let Some(shot) = &mut slots[connector] {
                    shot.x = *x;
                    shot.y = *y;
                    shot.visible = *visible;
                }
            }
            _ => {}
        }
    }

    /// Re-upload the cursor on every connector in `connectors` after a mode set discarded it.
    ///
    /// `owe_keyframe` marks the dock's cursor stale, but only a compositor commit on the cursor
    /// plane acted on that mark -- and a pointer that is not moving never produces one, so the
    /// cursor stayed missing until it was moved. Replaying the cached shot closes that window
    /// without waiting for userspace.
    pub(super) fn rearm_cursor(&self, dev: &VinoDrmDevice, connectors: u32) {
        for connector in 0..MAX_CONNECTORS {
            if connectors & (1u32 << connector) == 0 {
                continue;
            }
            let (w, h, bgra, x, y, visible) = {
                let slots = self.cursor_shot.lock();
                let Some(shot) = &slots[connector] else {
                    continue;
                };
                let mut copy = KVec::new();
                if copy.extend_from_slice(&shot.bgra, GFP_KERNEL).is_err() {
                    continue;
                }
                (shot.w, shot.h, copy, shot.x, shot.y, shot.visible)
            };
            let connector = connector as u8;
            // The same order the plane callback uses, and the same order `cmd_work` drains them
            // in: geometry, then bitmap, then position.
            self.queue_cmd(dev, KmsCmd::CursorCreate { connector, w, h });
            self.queue_cmd(
                dev,
                KmsCmd::CursorImage {
                    connector,
                    w,
                    h,
                    bgra,
                },
            );
            self.queue_cmd(
                dev,
                KmsCmd::CursorMove {
                    connector,
                    x,
                    y,
                    visible,
                },
            );
        }
    }

    /// Choose the next frame or delay for `connector`.
    ///
    /// Neither means this connector is idle and its worker can exit.
    pub(super) fn select_scanout(&self, connector: usize) -> (Option<PendingScanout>, Option<i64>) {
        let socket = connector + 1;
        // Keep a frame that arrived before the cadence deadline in the
        // coalescing slot. Userspace may stop committing after that flip, so
        // discarding it could leave the newest image unsent.
        // A dock with a sustained budget is held to it ahead of everything else, including an owed
        // keyframe: the frame that overruns it costs the session, not a repaint. The coalescing
        // slot keeps the newest image meanwhile, so waiting here drops intermediate frames rather
        // than delaying the desktop.
        if let Some(us) = self.stream_budget_wait_us() {
            return (None, Some(us));
        }
        let mut pending = self.pending_scanout.lock();
        let mut selected = None;
        let mut wait_us: Option<i64> = None;
        if self.modeset_requested[connector].load(Ordering::Acquire) != 0
            && pending[connector].is_some()
        {
            let owes_keyframe =
                self.keyframe_pending.load(Ordering::Acquire) & (1u32 << connector) != 0;
            let elapsed_us = self.last_frame.lock()[connector]
                .map_or(self.frame_period_us(), |t| t.elapsed().as_micros_ceil());
            // An owed keyframe normally jumps the cadence queue, because it is the frame that makes
            // the output correct and a compositor may not send another. A dock sharing the control
            // pipe cannot grant that: a keyframe is its largest frame, and anything that re-raises
            // the keyframe bit would let it bypass the interval repeatedly and hold the endpoint
            // for as long as it keeps being raised. There the interval binds every frame.
            let urgent = owes_keyframe && !self.video_on_ctrl_pipe();
            if urgent || elapsed_us >= self.frame_period_us() {
                selected = pending[connector].take();
                // A busy compositor continuously replaces `settle_repaint`.
                // Force cadence-selected frames to be keyframes while training;
                // the elapsed check above still applies the cadence limit.
                let sustaining = self.sustain_until.lock()[connector]
                    .is_some_and(|until| (until - Instant::<Monotonic>::now()).as_millis() > 0);
                if sustaining {
                    self.keyframe_pending
                        .fetch_or(1u32 << connector, Ordering::Release);
                }
            } else {
                wait_us = Some(self.frame_period_us() - elapsed_us);
            }
        }
        // Nothing flipped in. Fall back to the one-shot settle repaint if one is due, so a
        // compositor that went idle straight after enabling the output still ends up with its real
        // desktop on the panel rather than the buffer that happened to be current when the
        // mode-set's keyframe went out.
        if selected.is_none() {
            let mut settle = self.settle_repaint.lock();
            if self.modeset_requested[connector].load(Ordering::Acquire) == 0 {
                settle[connector] = None;
            } else if let Some((due, _, _)) = settle[connector].as_ref() {
                let remaining = *due - Instant::<Monotonic>::now();
                if remaining.as_millis() <= 0 {
                    let taken = settle[connector].take();
                    let as_keyframe = taken.as_ref().is_some_and(|(_, _, kf)| *kf);
                    selected = taken.map(|(_, f, _)| f);
                    if as_keyframe {
                        self.keyframe_pending
                            .fetch_or(1u32 << connector, Ordering::Release);
                    }
                    let kind = if as_keyframe {
                        "settle repaint (compositor idle after mode-set)"
                    } else {
                        "debt repaint (retransmissions owed, compositor idle)"
                    };
                    vino_debug!("vino: socket {socket} {kind}\n");
                } else {
                    let remaining = remaining.as_micros_ceil().max(1);
                    wait_us = Some(wait_us.map_or(remaining, |old| old.min(remaining)));
                }
            }
        }
        (selected, wait_us)
    }
}
