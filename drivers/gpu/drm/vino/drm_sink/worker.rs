// SPDX-License-Identifier: GPL-2.0

//! The workqueue items that carry out what the atomic callbacks asked for.
//!
//! One item reconciles KMS state, one per connector drives scanout, and one watches the
//! control plane for silence. They are the only contexts in the driver allowed to block on
//! USB.

use super::*;

impl_has_delayed_work! {
    impl HasDelayedWork<VinoDrmDevice> for VinoDrmData { self.cmd_work }
    impl HasDelayedWork<VinoDrmDevice, 5> for VinoDrmData { self.cp_watchdog }
}

impl WorkItem<5> for VinoDrmData {
    type Pointer = ARef<VinoDrmDevice>;
    fn run(this: ARef<VinoDrmDevice>) {
        run_cp_watchdog(this);
    }
}

/// Enforce the control session's silence deadline from off the control path.
///
/// Checking the deadline when a caller arrives to start a transaction is too late by
/// construction: the thread that would arrive is the keepalive, and the keepalive is the thread
/// that gets stuck -- a wedge then runs to 15 s against a 5 s limit, and ends only because the
/// dock re-enumerates. This runs on the system queue, so it is scheduled whatever vino's own
/// queues are doing, and it touches nothing a stuck transfer can be holding.
pub(super) fn run_cp_watchdog(this: ARef<VinoDrmDevice>) {
    let data: &VinoDrmData = &this;
    if data.shutting_down.load(Ordering::Acquire) {
        return;
    }
    // Vino must not call a dock silent while vino has chosen not to speak to it. Navarro's
    // setup-to-first-mode-set hold runs as long as the deadline itself, so holding the deadline
    // off here -- rather than at each of the two places the hold ends -- is what stops a healthy
    // cold bring-up from abandoning its own session at the boundary.
    if data.initial_modeset_quiet() {
        data.note_cp_reply();
    } else if data.cp_link_alive() {
        let silent_ms = data.cp_silent_for_ms();
        if silent_ms >= data.cp_silence_limit_ms() {
            data.abandon_cp_session(silent_ms);
            data.drop_connectors_with_session(&this);
            data.reset_after_wedge();
        }
    }
    data.start_cp_watchdog(&this);
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
    MAX_CONNECTORS == 4,
    "add a scanout_work_hN work item per connector (see VinoDrmData::enqueue_scanout)"
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

/// One connector's deferred scanout loop: pick this connector's due frame, encode it, transmit it,
/// repeat until the connector has nothing left to do. Both connectors run concurrently on the
/// per-device scanout queue.
///
/// A queued `ModeSet` must reach the dock before video for that connector, so this worker returns
/// while a stream command is pending or executing. `cmd_work` re-enqueues the scanout workers when
/// the command batch completes, and the pending framebuffer remains in its coalescing slot.
///
/// Two conditions, and both are needed: a stream operation in `pending_kms` (not yet drained), and
/// [`VinoDrmData::cmd_busy`] (drained and executing -- the window in which `pending_kms` is
/// misleadingly empty). The `video_inflight` store must be published *before* reading `cmd_busy`,
/// and both use `SeqCst`, so this and `wait_for_video_idle` cannot both conclude the other is idle.
pub(super) fn run_scanout_worker(this: ARef<VinoDrmDevice>, connector: usize) {
    use core::sync::atomic::Ordering::SeqCst;
    let data: &VinoDrmData = &this;
    // Plane updates can be accepted once CP engages, before the later platform readiness interval
    // finishes. Leave their coalesced frames untouched until bring-up publishes activation safety;
    // the publisher wakes every scanout worker after opening this gate.
    if !data.kms_activation_ready() {
        return;
    }
    // As in `cmd_work`: once the I/O window refuses a token, unplug has begun and there is no USB
    // left to do. `drm_dev_enter()` holds the parent interface Bound for the duration.
    let Ok(link) = crate::UsbLink::open(&data.io, data.endpoints) else {
        return;
    };
    let dev = &link;
    loop {
        if data.shutting_down.load(Ordering::Acquire) {
            return;
        }
        // Claim the connector's video endpoint first, then look for a reason not to use it.
        data.video_inflight[connector].store(true, SeqCst);
        let blocked = data.cmd_busy.load(SeqCst) || data.pending_kms.lock().has_stream();
        if blocked {
            data.video_inflight[connector].store(false, SeqCst);
            return;
        }
        let (frame, cadence_wait_us) = data.select_scanout(connector);
        if let Some(frame) = frame {
            run_pending_scanout(dev, data, frame);
            data.video_inflight[connector].store(false, SeqCst);
            continue;
        }
        data.video_inflight[connector].store(false, SeqCst);
        if let Some(us) = cadence_wait_us {
            // Bound the sleep. The settle-repaint arm can ask for its full deadline
            // ([`SETTLE_REPAINT_MS`], 1.2 s); sleeping that long inside the work item makes the
            // connector unreachable, because a flip arriving meanwhile finds the item already
            // running and its enqueue is dropped. Waking at the cadence window instead costs a few
            // extra wakeups while idle and keeps the connector responsive to real frames.
            let us = us.min(data.frame_period_us());
            fsleep(Delta::from_micros(us));
            continue;
        }
        // Re-check before exiting. A frame published between `select_scanout` above and this point
        // finds the work item still running, so its `enqueue_scanout` is dropped and the frame
        // waits for some *later* flip to enqueue successfully -- a lost wakeup that showed up as
        // multi-second stalls on whichever connector lost the race. The condition mirrors
        // `select_scanout`'s own guard so a connector with no mode-set cannot spin here.
        if data.modeset_requested[connector].load(Ordering::Acquire) != 0
            && data.pending_scanout.lock()[connector].is_some()
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
        let data: &VinoDrmData = &this;
        // Registration deliberately precedes the blocking CP/EDID setup, so userspace can publish a
        // complete atomic mode-set batch before the runtime session exists. Do not drain that batch
        // yet: attempting its dock-wide activation returns ENODEV, then the ordinary command loop
        // can split it into per-connector activations as setup becomes ready between those two
        // attempts. A readiness deferral is not a failed transport operation, so it neither
        // consumes `kms_retries` nor rewrites any pending slot; newer atomic state may continue to
        // coalesce there. The common worker gate applies to every dock generation.
        if !data.kms_activation_ready() {
            return;
        }
        // `drm_dev_enter()` holds the parent USB interface in Bound typestate until this worker
        // finishes. If unplug has begun, discard queued transport work without touching USB.
        // The I/O window is closed by `disconnect()` before it returns, so once it refuses a token
        // there is no USB left to do: discard the queued transport work.
        let Ok(link) = crate::UsbLink::open(&data.io, data.endpoints) else {
            return;
        };
        let dev = &link;
        loop {
            if data.shutting_down.load(Ordering::Acquire) {
                return;
            }
            // A dual-connector atomic commit runs `atomic_enable` once per connector, and each of
            // those queues its own `ModeSet` and wakes this worker -- microseconds apart, but far
            // less than it takes to get scheduled. Taking the first one alone turns one dock-wide
            // wake into two single-connector activations and skips the cold choreography that arms
            // the video endpoints, so wait, briefly and boundedly, for the siblings the compositor
            // has already published a timing for.
            {
                let started = Instant::<Monotonic>::now();
                let present = data.connectors_present.load(Ordering::Acquire);
                loop {
                    let queued = data.pending_kms.lock().connectors.iter().enumerate().fold(
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
                    // left to wait for once every connector with a monitor is either already active
                    // or represented in this batch.
                    let outstanding = (0..MAX_CONNECTORS).any(|h| {
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
            // A cold dual-connector atomic commit is one dock-wide wake: both mode-sets precede
            // either connector's video. Detect that shape before consuming the owned state.
            let mut dual_timings: [Option<crate::cp::Timing>; MAX_CONNECTORS] =
                [None; MAX_CONNECTORS];
            let mut cmd_connectors = 0u32;
            for connector in &pending.connectors {
                if let Some(KmsCmd::ModeSet {
                    connector: cmd_head,
                    timing,
                }) = &connector.stream
                {
                    let connector_index = *cmd_head as usize;
                    if connector_index < MAX_CONNECTORS {
                        cmd_connectors |= 1u32 << connector_index;
                        if data.modeset_active[connector_index].load(Ordering::Acquire) == 0
                            && data.modeset_requested[connector_index].load(Ordering::Acquire)
                                == timing_key(timing)
                        {
                            dual_timings[connector_index] = Some(*timing);
                        }
                    }
                }
            }
            // A dock that comes up as one transaction over every connector it has needs a timing
            // for each of them, and the compositor only describes the sockets it can see. A socket
            // with nothing plugged into it still has to be configured -- the scanout path already
            // declines to paint a connector with no EDID -- so let it join at its sibling's mode.
            //
            // The generation has to be published with the timing. A connector whose requested mode
            // is zero is a connector the activation waits on and never gets, so it defers on every
            // commit for as long as the dock is up, and that retry churn is what takes the shared
            // pipe down. Publishing it is also what makes this happen once: the connector then has
            // a request of its own and no longer looks unasked-for.
            if data.video_on_ctrl_pipe() && dual_timings.iter().flatten().count() == 1 {
                let sibling = dual_timings.iter().flatten().copied().next();
                if let Some(timing) = sibling {
                    for connector in 0..data.connector_count().min(MAX_CONNECTORS) {
                        if dual_timings[connector].is_some()
                            || data.modeset_requested[connector].load(Ordering::Acquire) != 0
                            || data.modeset_active[connector].load(Ordering::Acquire) != 0
                        {
                            continue;
                        }
                        data.last_timing.lock()[connector] = Some(timing);
                        data.modeset_requested[connector]
                            .store(timing_key(&timing), Ordering::Release);
                        dual_timings[connector] = Some(timing);
                        cmd_connectors |= 1u32 << connector;
                        vino_debug!(
                            "vino: socket {} has no monitor and is configured at its sibling's mode\n",
                            connector + 1
                        );
                    }
                }
            }
            // Exclude the scanout workers for exactly as long as this batch can touch a video
            // endpoint. `activate_dual_wake` and the `ModeSet` arm both run
            // `submit_prompt_training`, which writes the activation carrier to the connector's
            // endpoint; a concurrent scanout frame there would interleave records on the wire and
            // would have its `video_q` slot double-opened. Cursor-only batches deliberately skip
            // this: they never touch video, and a mouse in motion produces a continuous stream of
            // them. `Blank` writes the connector's video endpoint for the same reason `ModeSet`
            // does, so it needs the same exclusion against the scanout workers -- otherwise a frame
            // already in flight interleaves its records with the blanking frames on the wire.
            let has_modeset = pending.has_stream();
            // On the DL7400 a mode set is dock-wide, not per connector. Configuring one connector
            // while any other is lit makes the dock re-enumerate about 100 ms after the next video
            // write, on every shape of the change: 120 -> 165, 165 -> 120 and 120 -> 60 on a live
            // connector, waking a second connector a second after the first, and reconfiguring a
            // connector whose sibling has been lit and idle for minutes. The same changes with the
            // sibling disabled are clean, and so is the simultaneous `activate_dual_wake` path. DLM
            // behaves the same way: it logs `[Profile change] Recreating device` and re-runs a
            // bring-up-shaped burst rather than reconfiguring one connector in place.
            //
            // So fold every already-active connector into this batch. Zeroing its mode generation
            // makes `activate_dual_wake` treat it as a fresh wake, and the whole dock is then taken
            // through the one choreography the hardware accepts. The cost is that the sibling
            // blinks through a mode change on its neighbour; the alternative is a dock reset and
            // tens of seconds of dark panels on both. `cmd_connectors` rather than `has_modeset`: a
            // `Blank`-only batch also counts as a stream command, and it must not drag every lit
            // connector through a re-activation. Folding the lit connectors in makes
            // `activate_dual_wake` name every connector at once, which is the only shape of mode
            // set this dock accepts while more than one connector is lit. This replays the cold
            // choreography -- a dock-wide sink reset and pipe clears -- on a dock that is already
            // driving its sinks. That is the cost of the only mode set this dock will take.
            if cmd_connectors != 0 && data.dock_wide_modeset() {
                // Gather the whole dock's desired state first, and only commit to it if at least
                // two connectors end up in it. Below two there is nothing for the dual path to do
                // and the per-connector schedule is the proven one, so nothing is disturbed.
                let mut fold: [Option<crate::cp::Timing>; MAX_CONNECTORS] = [None; MAX_CONNECTORS];
                for connector in 0..MAX_CONNECTORS {
                    // A connector this batch is already mode-setting: take the requested timing,
                    // even if the connector is currently lit. A live reconfigure is exactly the
                    // case that must not go down the per-connector path.
                    if cmd_connectors & (1u32 << connector) != 0 {
                        if let Some(timing) = data.last_timing.lock()[connector] {
                            if data.modeset_requested[connector].load(Ordering::Acquire)
                                == timing_key(&timing)
                            {
                                fold[connector] = Some(timing);
                            }
                        }
                        continue;
                    }
                    // A connector this batch does not name, but which is lit and still wants the
                    // mode it is showing. One whose request has already moved on has its own
                    // `ModeSet` queued behind this batch and is left to it.
                    let active = data.modeset_active[connector].load(Ordering::Acquire);
                    if active != 0
                        && data.modeset_requested[connector].load(Ordering::Acquire) == active
                    {
                        fold[connector] = data.last_timing.lock()[connector];
                    }
                }
                if fold.iter().flatten().count() >= 2 {
                    for connector in 0..MAX_CONNECTORS {
                        let socket = connector + 1;
                        let Some(timing) = fold[connector] else {
                            continue;
                        };
                        // `activate_dual_wake` only accepts a connector whose generation is zero; a
                        // dock-wide transaction re-establishes every connector from scratch, so say
                        // so.
                        data.modeset_active[connector].store(0, Ordering::Release);
                        dual_timings[connector] = Some(timing);
                        cmd_connectors |= 1u32 << connector;
                        vino_debug!(
                            "vino: socket {socket} joins a dock-wide mode set ({}x{}@{})\n",
                            timing.hactive,
                            timing.vactive,
                            timing.refresh_hz
                        );
                    }
                }
            }
            if has_modeset {
                data.cmd_busy
                    .store(true, core::sync::atomic::Ordering::SeqCst);
                data.wait_for_video_idle();
            }
            // Both dock-wide schedules need two connectors coming up together; below that the
            // per-connector path is the proven one. Which schedule applies is a property of the
            // dock: the Ridge and DL7400 cold timeline consists of operations a dock carrying video
            // on its control pipe does not take, and that dock has its own measured choreography in
            // `ELLA_DOCK_WIDE`. Driving either one from the other's table fails every pass.
            let both_connectors = dual_timings.iter().flatten().count() >= 2;
            let dual_wake = both_connectors && !data.video_on_ctrl_pipe();
            let dock_wide = both_connectors && data.video_on_ctrl_pipe();
            if has_modeset {
                vino_debug!(
                    "vino: KMS batch -- stream cmds {}, dual timings {}, dual_wake {}, requested [{} {} {} {}], active [{} {} {} {}]\n",
                    (0..MAX_CONNECTORS).filter(|&h| cmd_connectors & (1u32 << h) != 0).count(),
                    dual_timings.iter().flatten().count(),
                    dual_wake || dock_wide,
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
            let multihead_attempted = dock_wide || dual_wake;
            let dual_complete = if dock_wide {
                match data.activate_dock_wide(dev, ELLA_DOCK_WIDE, dual_timings) {
                    Ok(done) => done,
                    Err(e) => {
                        pr_warn!("vino: dock-wide activation failed ({e:?})\n");
                        false
                    }
                }
            } else {
                dual_wake
                    && match data.activate_dual_wake(dev, dual_timings) {
                        Ok(done) => done,
                        Err(e) => {
                            pr_warn!("vino: dual-connector activation failed ({e:?})\n");
                            false
                        }
                    }
            };
            if multihead_attempted && !dual_complete {
                // A multihead activation is one indivisible transport transaction. In particular,
                // never let a failed Ella cold table fall through to the ordinary per-connector
                // loop: connector 0 can then light just as setup becomes ready and force connector
                // 1 down the live runtime table, so the cold two-connector choreography never
                // lands. Restore the entire owned batch; newer producer state already occupying a
                // slot still wins.
                if has_modeset {
                    data.cmd_busy
                        .store(false, core::sync::atomic::Ordering::SeqCst);
                }
                let mut retry_pending = data.pending_kms.lock();
                retry_pending.retry_batch(pending);
                let attempts = data.kms_retries.fetch_add(1, Ordering::Relaxed) + 1;
                if attempts >= KMS_RETRY_LIMIT {
                    pr_warn!(
                        "vino: dropping atomic multihead KMS batch after {} deferrals; the link is not coming back on its own\n",
                        KMS_RETRY_LIMIT
                    );
                    retry_pending.clear();
                    data.kms_retries.store(0, Ordering::Relaxed);
                    return;
                }
                drop(retry_pending);
                vino_debug!("vino: atomic multihead KMS batch deferred\n");
                if !data.shutting_down.load(Ordering::Acquire) {
                    let delay = kernel::time::msecs_to_jiffies(KMS_RETRY_MS);
                    let _ = workqueue::system().enqueue_delayed::<_, 0>(ARef::from(&*this), delay);
                }
                return;
            }
            // Heads whose mode this batch actually re-programmed, and whose dock-side cursor is
            // therefore gone. See `rearm_cursor`.
            let mut relit = if dual_complete { cmd_connectors } else { 0 };
            let mut cmds: [Option<KmsCmd>; MAX_CONNECTORS * 4] =
                [const { None }; MAX_CONNECTORS * 4];
            for (connector, pending) in pending.connectors.into_iter().enumerate() {
                cmds[connector] = pending.stream;
                cmds[MAX_CONNECTORS + connector] = pending.cursor_create;
                cmds[MAX_CONNECTORS * 2 + connector] = pending.cursor_image;
                cmds[MAX_CONNECTORS * 3 + connector] = pending.cursor_move;
            }
            // Control-plane ordering comes first. An enabling atomic commit queues the plane flip
            // before its CRTC mode-set. Finish the mode transaction before
            // selecting a pending framebuffer.
            let mut cmds = cmds.into_iter().flatten();
            let mut retry = false;
            while let Some(cmd) = cmds.next() {
                let mut mode_programmed = 0u32;
                let res = match &cmd {
                    KmsCmd::ModeSet { connector, timing } => {
                        if dual_complete {
                            // `activate_dual_wake` consumed the current generation for both
                            // connectors. A superseding generation published while it ran remains
                            // in `pending_kms` for the next outer iteration.
                            continue;
                        }
                        let connector_index = *connector as usize;
                        let key = timing_key(timing);
                        if connector_index >= MAX_CONNECTORS
                            || data.modeset_requested[connector_index].load(Ordering::Acquire)
                                != key
                        {
                            Ok(()) // superseded or disabled while queued
                        } else {
                            data.activate_head(dev, *connector, timing, key)
                                .map(|activated| {
                                    if activated {
                                        mode_programmed = 1u32 << connector;
                                    }
                                })
                        }
                    }
                    KmsCmd::CursorCreate { connector, w, h } => data.send_cp(dev, 0x1b, 0, |ctr| {
                        crate::cp::cursor_create(ctr, *connector, *w, *h)
                    }),
                    KmsCmd::CursorImage {
                        connector,
                        w,
                        h,
                        bgra,
                    } => data.send_cp(dev, 0x1c, 0, |ctr| {
                        crate::cp::cursor_image(ctr, *connector, *w, *h, bgra)
                    }),
                    KmsCmd::CursorMove {
                        connector,
                        x,
                        y,
                        visible,
                    } => data.send_cp(dev, 0x1a, 0, |ctr| {
                        crate::cp::cursor_move(ctr, *connector, *x, *y, *visible)
                    }),
                    KmsCmd::Blank { connector } => data.blank_connector(dev, *connector),
                };
                // Remember what the dock accepted, so a later mode set can put it back. Recorded
                // here rather than where the atomic callback queues it, because only a command
                // that actually went out describes the dock's state.
                if res.is_ok() {
                    data.record_cursor(&cmd);
                    relit |= mode_programmed;
                }
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
                    if data.kms_retries.fetch_add(1, Ordering::Relaxed) + 1 >= KMS_RETRY_LIMIT {
                        pr_warn!(
                            "vino: dropping asynchronous KMS work after {} deferrals ({e:?}); the link is not coming back on its own\n",
                            KMS_RETRY_LIMIT
                        );
                        pending.clear();
                        data.kms_retries.store(0, Ordering::Relaxed);
                        retry = false;
                    }
                    break;
                }
            }
            if has_modeset {
                data.cmd_busy
                    .store(false, core::sync::atomic::Ordering::SeqCst);
            }
            // Put each re-programmed connector's cursor back. Queued rather than sent inline so it
            // drains through the ordinary path on the next turn of this loop, behind anything the
            // compositor has published in the meantime -- a real cursor commit always wins.
            if relit != 0 && !retry {
                data.rearm_cursor(&this, relit);
            }

            if retry {
                if !data.shutting_down.load(Ordering::Acquire) {
                    let delay = kernel::time::msecs_to_jiffies(KMS_RETRY_MS);
                    let _ = workqueue::system().enqueue_delayed::<_, 0>(ARef::from(&*this), delay);
                }
                return;
            }
            // A batch that got through means whatever was wrong has cleared.
            data.kms_retries.store(0, Ordering::Relaxed);
            if data.pending_kms.lock().is_empty() {
                break;
            }
        }
        // Wake both scanout workers after the command batch. They stop while
        // a queued mode set must reach the dock before video and resume here.
        data.enqueue_scanout_all(&this);
    }
}
