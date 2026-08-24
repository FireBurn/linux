// SPDX-License-Identifier: GPL-2.0

//! Bringing a connector's downstream sink up.
//!
//! A sink that is merely programmed with a timing stays dark. The dock has to be walked through a
//! sequence of bracket states, carrier frames and mode sets, in an order and at intervals its own
//! vendor was measured using, so the constants and the ordering here are load-bearing.

use super::*;

impl VinoDrmData {
    /// Continuously present one already-encoded activation carrier for at least `duration_ms`.
    ///
    /// This deliberately performs no CP transaction between presentations. `BringUp` runs the
    /// status/heartbeat dialogue concurrently on another work item; doing it here would put a
    /// 10--15 ms control round-trip between tiny black frames and recreate the endpoint starvation
    /// this path exists to remove. A persistent eight-URB queue bounds how far submission can run
    /// ahead, so elapsed wall time closely follows actual endpoint progress rather than merely
    /// copying an arbitrary number of frames into unbounded memory.
    pub(super) fn submit_prompt_training(
        &self,
        dev: &BoundInterface<'_>,
        connector: u8,
        want: u64,
        frames: &[KVec<u8>],
        ordinary_frames: &[KVec<u8>],
        duration_ms: i64,
        with_arm: bool,
    ) -> Result<u32> {
        if frames.is_empty() {
            return Err(kernel::error::code::EINVAL);
        }
        let geometry = self.geometry();
        let xfer: usize = VIDEO_XFER;
        let connector_index = connector as usize;
        let pipe_i = dev.video_pipe_index(connector_index)?;
        let connector_bit = 1u32 << connector;
        // The DL7400's per-strip parameter map, ahead of the pixels it describes. This path --
        // the startup/prompt-training submit -- is the one the dock's first frames go out on, so
        // wiring the map only into the steady-state scanout left it absent from every frame that
        // matters. Ridge has no equivalent record and gets an empty slice.
        let params: KVec<u8> = if geometry.connector_selector_shift == 0 {
            KVec::new()
        } else {
            let t = self
                .last_timing
                .lock()
                .get(connector_index)
                .copied()
                .flatten();
            match t {
                Some(t) => {
                    let (sw, sh) = (geometry.strip_w(), geometry.strip_h());
                    let padded_width = (t.hactive as usize).div_ceil(sw) * sw;
                    let padded_height = (t.vactive as usize).div_ceil(sh) * sh;
                    crate::video::haar::navarro_strip_params(
                        geometry,
                        connector,
                        padded_width,
                        padded_height,
                        &frames,
                        &mut self.strip_classes.lock()[connector_index],
                    )?
                }
                None => KVec::new(),
            }
        };
        vino_debug!(
            "vino: connector={connector} prompt-training parameter map {} B\n",
            params.len()
        );
        let arm = if with_arm {
            // The bit is cleared only once this connector's carrier has gone out, so finding it
            // clear means the stream is already open and the carrier already presented. That is
            // this call's whole purpose, so report it done. Reporting it as a failure instead makes
            // the caller re-arm and present a second carrier, and a retried activation then walks
            // the dock's ring one slot further on every pass -- six flat frames where the vendor
            // sends one, with the picture in a slot the dock is not showing.
            if self.arm_prefix_pending.load(Ordering::Acquire) & connector_bit == 0 {
                return Ok(0);
            }
            Some(self.build_stream_prefix_buf(connector_index)?)
        } else {
            None
        };
        if arm.is_some() {
            self.send_stream_open(dev, connector_index)?;
            // A stream that is being opened here starts its ring and its frame counter from the
            // beginning, whatever the connector reached before. Arming the prologue already resets
            // this, but an activation that is retried arms once and presents several times, so a
            // connector could open a stream and immediately tell the dock it was filling a later
            // slot with a later frame number -- and the dock scans out a slot nothing wrote.
            self.scanout_seq.lock()[connector_index] = 0;
        }
        let startup = arm.is_some();
        let seq0 = self.scanout_seq.lock()[connector_index];
        let started = Instant::<Monotonic>::now();
        let mut repeat = 0u32;
        // Presentations that named a ring slot, which is what the frame counter counts. See
        // `names_ring_slot`.
        let mut named = 0u32;

        loop {
            if self.shutting_down.load(Ordering::Acquire)
                || self.modeset_requested[connector_index].load(Ordering::Acquire) != want
                || self.modeset_active[connector_index].load(Ordering::Acquire) != want
            {
                return Err(kernel::error::code::ENODEV);
            }

            let seq = seq0.wrapping_add(named);
            let trailer = self.build_frame_trailer(connector, seq);
            let arm_slice: &[u8] = if repeat == 0 {
                arm.as_ref().map_or(&[], |a| &a[..])
            } else {
                &[]
            };
            let prologue_frame = startup && repeat == 0;
            let opener = self.build_frame_opener(connector, seq, prologue_frame);
            let opener_slice: &[u8] = opener.as_ref().map_or(&[], |o| &o[..]);
            // Counted once this presentation has actually gone out; see the loop's tail. A pass
            // that defers because the endpoint is full sends nothing, and a ring slot the dock
            // never saw must not be spent.
            let names_slot = names_ring_slot(opener_slice, &trailer);
            // The report leads the frame it rides with, ahead of any pixels; see
            // `build_stream_report_buf` for which frames carry one. The frame bearing the prologue
            // never does: every generation goes from its decoder configuration into image records.
            //
            // The parameter map is not the report and has no such exception. The same-day DLM cold
            // capture carries the map part-way through frame zero, and Windows carries it after
            // the image records. Both put it before frame close. Omitting it made vino's first
            // frame exactly 5,984 bytes short and left the dock briefly scanning a partially
            // described framebuffer.
            let report = if prologue_frame {
                None
            } else {
                self.build_stream_report_buf(connector_index, seq)?
            };
            let report_slice: &[u8] = report.as_ref().map_or(&[], |r| &r[..]);
            let params_slice: &[u8] = &params[..];
            // The prologue and ordinary DLM carriers contain identical strips but different
            // producer record boundaries. Select by the presence of the one-shot arm rather than
            // by this helper's local `repeat`: the cold timeline invokes the helper once per
            // measured presentation, so every invocation starts with repeat zero.
            let frame_parts = if prologue_frame {
                frames
            } else {
                ordinary_frames
            };
            let image_len: usize = frame_parts.iter().map(|f| f.len()).sum();
            let image_parts = frame_parts.len();
            let wire_len = arm_slice.len()
                + opener_slice.len()
                + report_slice.len()
                + params_slice.len()
                + image_len
                + trailer.len();
            {
                // One writer owns a shared pipe for the whole frame; see `own_pipe`.
                let _pipe = self.own_pipe();
                let mut staging_slots = self.video_staging.lock();
                let staging_slot = &mut staging_slots[connector_index];
                if staging_slot.is_none() {
                    let mut staging = KVec::new();
                    staging.resize(xfer, 0, GFP_KERNEL)?;
                    *staging_slot = Some(staging);
                }
                let staging = staging_slot.as_mut().ok_or(kernel::error::code::ENOMEM)?;

                let mut queue_slot = self.video_q[pipe_i].lock();
                if queue_slot.is_none() {
                    // Navarro's required EP08/EP0a clears are sent together at their captured
                    // pre-commit point in `send_cp_setup`. Nothing clears here: doing so after
                    // stream setup has begun changes SuperSpeed endpoint sequence state at a
                    // point DLM does not.
                    *queue_slot = Some(dev.video_queue(connector_index, 8, xfer)?);
                    vino_debug!(
                        "vino: connector={} endpoint={:#04x} persistent video queue opened by prompt training\n",
                        connector,
                        dev.endpoints.video[connector_index].address()
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
                        .fetch_or(connector_bit, Ordering::AcqRel)
                        & connector_bit
                        == 0
                    {
                        match dev.video_endpoint_status(connector_index) {
                            Ok(status) => vino_debug!(
                                "vino: connector={} endpoint={:#04x} stopped accepting video: GET_STATUS={:#06x} halt={}\n",
                                connector,
                                dev.endpoints.video[connector_index].address(),
                                status,
                                status & 1
                            ),
                            Err(e) => pr_warn!(
                                "vino: connector={} endpoint={:#04x} stopped accepting video: GET_STATUS failed ({e:?})\n",
                                connector,
                                dev.endpoints.video[connector_index].address()
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
                // NAVARRO_PARAM_IMAGE_OFFSET bytes of image records in both the prologue and
                // ordinary carriers. Build a borrowed scatter list in the captured order. A
                // carrier's chunks all hold the same records, so the exact offset is usable here:
                // the insertion may fall inside one allocation chunk, but is always on a
                // wire-record boundary. A content frame's are not uniform and round to a chunk.
                let mut wire_parts: KVec<&[u8]> = KVec::with_capacity(image_parts + 6, GFP_KERNEL)?;
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
                // DLM's authenticated first connector-0 prologue does not put all four URBs on the
                // xHCI ring at once. It submits at +0/+806/+851/+873 us and receives completion
                // of the first at +104 us, so the dock gets a ~700-us ready interval before the
                // second transfer and then a three-URB pipeline. Submitting chunk two immediately
                // leaves Navarro NRDY forever after exactly one completed URB. Preserve this
                // producer boundary only for the one-shot, full-size prologue; ordinary frames
                // use the normal eight-deep queue, just as DLM does.
                const NAVARRO_PROLOGUE_SUBMIT_US: [i64; 4] = [0, 806, 851, 873];
                let pace_prologue = !arm_slice.is_empty()
                    && xfer == VIDEO_XFER
                    && geometry.connector_selector_shift != 0;
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
                    if pace_prologue {
                        let chunk = wire_off / xfer;
                        if let Some(anchor) = prologue_anchor {
                            if let Some(&target_us) = NAVARRO_PROLOGUE_SUBMIT_US.get(chunk) {
                                Self::wait_video_offset(anchor, target_us);
                            }
                        }
                    }
                    // DLM's mixed transport: prologue chunk zero is reaped below, then the rest
                    // and all ordinary frames are pipelined.
                    queue.send(dev.io(), dst, crate::timeout())?;
                    if pace_prologue && wire_off == 0 {
                        let anchor = Instant::<Monotonic>::now();
                        prologue_anchor = Some(anchor);
                        // Do not expose chunk two to xHCI before chunk zero completes. The capture
                        // has the first completion at +104 us and the next submit at +806 us.
                        queue.flush(dev.io(), crate::timeout())?;
                    }
                    self.last_video_at.lock()[connector_index] = Some(Instant::<Monotonic>::now());
                    wire_off += data_len;
                }
            }

            if repeat == 0 && startup {
                self.arm_prefix_pending
                    .fetch_and(!connector_bit, Ordering::Release);
                // The window runs from the first frame the dock actually receives, not from the
                // mode set, so it is re-armed here. Whether there is a window at all is
                // `sustain_window`'s decision and must not be second-guessed: a dock that shares
                // its control pipe is granted none, and re-arming one unconditionally spent three
                // seconds of full keyframes on the endpoint its control plane needs.
                let mut sustain = self.sustain_until.lock();
                if sustain[connector_index].is_some() {
                    sustain[connector_index] =
                        Some(Instant::<Monotonic>::now() + Delta::from_millis(SUSTAIN_MS));
                }
                drop(sustain);
                vino_debug!(
                    "vino: connector {} startup frame submitted after {} ms ({} bytes)\n",
                    connector,
                    (Instant::<Monotonic>::now() - started).as_millis(),
                    wire_len
                );
            }
            repeat = repeat.wrapping_add(1);
            if names_slot {
                named = named.wrapping_add(1);
            }
            self.scanout_seq.lock()[connector_index] = seq0.wrapping_add(named);

            if repeat >= self.carrier_presentations() {
                break;
            }
            // A zero window means this dock is bounded by the count above, not by wall clock.
            // Testing the elapsed time against it regardless ends the carrier after one frame,
            // whatever the count says.
            if duration_ms > 0 && (Instant::<Monotonic>::now() - started).as_millis() >= duration_ms
            {
                break;
            }
            // Leave the endpoint between carrier frames on a dock that shares it, at the same
            // interval its ordinary frames are paced to. A back-to-back carrier is what silenced
            // this dock before, and it is the reason its window was reduced to a single frame.
            if self.video_on_ctrl_pipe() {
                fsleep(Delta::from_millis(self.frame_period_ms()));
            }
        }

        vino_debug!(
            "vino: connector={} training complete ({} presentations, {} ms)\n",
            connector,
            repeat,
            (Instant::<Monotonic>::now() - started).as_millis()
        );
        Ok(repeat)
    }
    /// Apply one desired mode generation and its activation carrier.
    ///
    /// The control timeline is always released before returning. Any failed bracket, mode-set, arm,
    /// or carrier transfer clears `modeset_active`, allowing the desired generation to be retried.
    pub(super) fn activate_head(
        &self,
        dev: &BoundInterface<'_>,
        connector: u8,
        timing: &crate::cp::Timing,
        want: u64,
    ) -> Result<bool> {
        let connector_index = connector as usize;
        if connector_index >= MAX_CONNECTORS
            || self.modeset_requested[connector_index].load(Ordering::Acquire) != want
        {
            return Ok(false);
        }
        // Keep no-op adoption at the activation boundary rather than in one caller. The command
        // worker and the scanout worker can both arrive here, and either must avoid reopening an
        // exact mode that is already usable under a different raw request token.
        if self.adopt_programmed_mode(connector_index, timing, want) {
            vino_debug!(
                "vino: connector {connector_index} already active at this mode; no re-activation\n"
            );
            return Ok(false);
        }
        // Reconfiguring one connector of a dock that shares its control pipe, while another
        // connector is lit, is a sequence of its own: see `ELLA_RUNTIME_MODE`. The schedule below
        // is the cold one, and a dock already driving a sink stops answering partway through it,
        // taking the lit connector down with it.
        if self.video_on_ctrl_pipe()
            && (0..MAX_CONNECTORS).any(|h| {
                h != connector_index && self.modeset_active[h].load(Ordering::Acquire) != 0
            })
        {
            let mut timings: [Option<crate::cp::Timing>; MAX_CONNECTORS] = [None; MAX_CONNECTORS];
            timings[connector_index] = Some(*timing);
            return self.activate_dock_wide(dev, ELLA_RUNTIME_MODE, timings);
        }
        // `wake` describes the state on entry and therefore has to be captured before invalidating
        // it. From this point onward the transaction is going to touch the dock; do not leave the
        // old token adoptable while a bracket, clear-mode, or set-mode is only partly complete.
        let wake = self.modeset_active[connector_index].swap(0, Ordering::AcqRel) == 0;
        self.programmed_timing.lock()[connector_index] = None;
        // A superseding callback can land between the initial/adoption checks and the swap.
        // Clearing the old active token is conservative, but sending its stale transaction is not:
        // leave the newer request to its queued command or inline retry.
        if self.modeset_requested[connector_index].load(Ordering::Acquire) != want {
            return Ok(false);
        }
        // Same as the dual path: a connector coming back from a blank has a bracket owed before
        // anything else is sent to it.
        self.close_blank_bracket(dev, connector)?;
        // The caller's timing was built when this connector was enabled, possibly before its
        // endpoint partner existed. The request token remains the caller's generation; the exact
        // corrected timing sent below is recorded separately in `programmed_timing`.
        let timing = &self.effective_timing(connector_index, timing);

        let geometry = self.geometry();
        let padded_width =
            (timing.hactive as usize + geometry.strip_w() - 1) & !(geometry.strip_w() - 1);
        let padded_height =
            (timing.vactive as usize + geometry.strip_h() - 1) & !(geometry.strip_h() - 1);
        let prompt =
            crate::video::haar::black_frame_ep08(geometry, padded_width, padded_height, connector)?;
        let prompt_ordinary = crate::video::haar::black_frame_ep08_ordinary(
            geometry,
            padded_width,
            padded_height,
            connector,
        )?;

        self.begin_cp_timeline();
        let transaction = (|| -> Result<bool> {
            // The timing is what the sink has to be retrained onto, so the bracket belongs to the
            // mode program and not to how the connector got here. A connector arriving from a blank
            // needs it exactly as much as one being configured cold.
            self.modeset_bracket_pre(dev, connector)?;
            let mode_anchor = Instant::<Monotonic>::now();
            // Tear the connector's pipe down before configuring it, as DLM does. See
            // `cp::clear_mode`: the dock expects a connector to be torn down before it is
            // configured.
            if self.clear_mode_before_set() {
                self.send_cp(dev, 0x48, 0, |ctr| crate::cp::clear_mode(ctr, connector))?;
            }
            self.send_cp(dev, 0x48, 0, |ctr| {
                crate::cp::set_mode(ctr, connector, timing)
            })?;
            self.programmed_timing.lock()[connector_index] = Some(*timing);
            if self.modeset_requested[connector_index].load(Ordering::Acquire) != want {
                return Ok(false);
            }

            self.modeset_active[connector_index].store(want, Ordering::Release);
            self.sustain_until.lock()[connector_index] = self.sustain_window(connector_index);
            let bit = 1u32 << connector;
            self.arm_stream_prologue(connector_index);
            // A driven connector's stream opens with its pipe descriptor, not the idle open.
            self.stream_open_pending.fetch_and(!bit, Ordering::Release);
            self.owe_keyframe(connector_index);
            self.strip_hashes.lock()[connector_index] = None;
            self.dirty_ttl.lock()[connector_index] = None;

            self.modeset_bracket_post_open(dev, connector, mode_anchor)?;
            let opening = self.submit_prompt_training(
                dev,
                connector,
                want,
                &prompt,
                &prompt_ordinary,
                self.carrier_ms(PROMPT_TRAINING_OPEN_MS),
                true,
            );
            let closing = self.modeset_bracket_post_close(dev, connector, mode_anchor);
            opening?;
            closing?;
            Ok(true)
        })();
        self.end_cp_timeline();

        let activated = match transaction {
            Ok(activated) => activated,
            Err(e) => {
                self.modeset_active[connector_index].store(0, Ordering::Release);
                self.programmed_timing.lock()[connector_index] = None;
                self.unwind_bracket(dev, connector);
                return Err(e);
            }
        };
        if !activated {
            self.programmed_timing.lock()[connector_index] = None;
            self.unwind_bracket(dev, connector);
            return Ok(false);
        }
        // The tail continues a carrier that is bounded by wall clock, so a dock bounded by a frame
        // count has already presented all of it and this would add one more. The count is what the
        // vendor's stream opens with, and the frames it names walk the dock's ring: an extra one
        // puts every later frame a slot further on than the vendor puts it, and the dock presents
        // a slot holding the flat carrier rather than the one holding the picture.
        if self.carrier_ms(PROMPT_TRAINING_TAIL_MS) > 0 {
            if let Err(e) = self.submit_prompt_training(
                dev,
                connector,
                want,
                &prompt,
                &prompt_ordinary,
                self.carrier_ms(PROMPT_TRAINING_TAIL_MS),
                false,
            ) {
                self.modeset_active[connector_index].store(0, Ordering::Release);
                self.programmed_timing.lock()[connector_index] = None;
                self.unwind_bracket(dev, connector);
                return Err(e);
            }
        }

        vino_debug!(
            "vino: applied {} stream-enable sequence for connector {}\n",
            if wake { "wake" } else { "mode-change" },
            connector
        );
        Ok(true)
    }
    /// Activate both downstream connectors using the dock-wide cold-link schedule.
    ///
    /// Both mode sets precede either connector's video. Single-connector activation and live mode
    /// changes use the per-connector schedule.
    ///
    /// Every connector number in a [`ColdTimeline`] is a transcript slot, not a connector: slot 0
    /// is the lowest-numbered activating connector and slot 1 the next, and they are resolved to
    /// real connectors at the point of send. Taken literally they address connectors 0 and 1, whose
    /// bits are absent from `sent` for any other pair of sockets, so no marker and no video would
    /// go out at all.
    pub(super) fn activate_dual_wake(
        &self,
        dev: &BoundInterface<'_>,
        mut timings: [Option<crate::cp::Timing>; MAX_CONNECTORS],
    ) -> Result<bool> {
        let geometry = self.geometry();
        let mut prompts: [Option<KVec<KVec<u8>>>; MAX_CONNECTORS] = core::array::from_fn(|_| None);
        let mut ordinary_prompts: [Option<KVec<KVec<u8>>>; MAX_CONNECTORS] =
            core::array::from_fn(|_| None);
        let mut keys = [0u64; MAX_CONNECTORS];
        let mut valid = 0u32;
        // Snapshot topology once for the whole dock-wide transaction, so two partner modes cannot
        // disagree about whether they share an endpoint if another callback lands mid-loop.
        let requested_heads = self.requested_connector_mask();

        // Pre-encode both tiny carriers before excluding the keepalive or starting either
        // mode-set. Encoding work must not serialize the dock's back-to-back mode pair.
        for connector in 0..MAX_CONNECTORS {
            let Some(timing) = timings[connector] else {
                continue;
            };
            let key = timing_key(&timing);
            if self.modeset_requested[connector].load(Ordering::Acquire) != key
                || self.modeset_active[connector].load(Ordering::Acquire) != 0
            {
                continue;
            }
            timings[connector] =
                Some(self.effective_timing_in_mask(connector, &timing, requested_heads));
            vino_debug!(
                "vino: dual activation connector={} mode={}x{}@{}\n",
                connector,
                timing.hactive,
                timing.vactive,
                timing.refresh_hz
            );
            let padded_width =
                (timing.hactive as usize + geometry.strip_w() - 1) & !(geometry.strip_w() - 1);
            let padded_height =
                (timing.vactive as usize + geometry.strip_h() - 1) & !(geometry.strip_h() - 1);
            prompts[connector] = Some(crate::video::haar::black_frame_ep08(
                geometry,
                padded_width,
                padded_height,
                connector as u8,
            )?);
            ordinary_prompts[connector] = Some(crate::video::haar::black_frame_ep08_ordinary(
                geometry,
                padded_width,
                padded_height,
                connector as u8,
            )?);
            keys[connector] = key;
            valid |= 1u32 << connector;
        }
        if valid.count_ones() < 2 {
            return Ok(false);
        }

        // The two connectors this activation is about, in the order the timeline brings them up.
        // Both cold timelines describe exactly two connectors, so a third is reported rather than
        // silently left out of the choreography.
        let mut slots = [0u8; 2];
        let mut n = 0;
        for connector in 0..MAX_CONNECTORS {
            if valid & (1u32 << connector) != 0 {
                if n < slots.len() {
                    slots[n] = connector as u8;
                }
                n += 1;
            }
        }
        if n > slots.len() {
            pr_warn!(
                "vino: {n} connectors activating but the cold timeline describes {}; choreographing {} and {}\n",
                slots.len(),
                slots[0],
                slots[1]
            );
        }
        // Slot -> connector. Out-of-range slots cannot occur (both timelines only name 0 and 1) but
        // clamp rather than panic, because a timeline is data and this runs under the CP lock.
        let connector_of = |slot: u8| usize::from(slots[usize::from(slot).min(slots.len() - 1)]);

        // Keep the clear/settle phase and the real mode-set choreography in one exclusive control
        // transaction. Their deadlines have separate anchors because the Navarro cold timeline was
        // measured from the real connector-0 mode set, 1,156 ms after its pipe clear.
        // A blank is closed before the choreography, not during it: the bracket must be shut
        // before anything re-probes or re-sets the mode.
        for connector in 0..MAX_CONNECTORS {
            if valid & (1u32 << connector) != 0 {
                self.close_blank_bracket(dev, connector as u8)?;
            }
        }
        self.begin_cp_timeline();
        let activation_started = Instant::<Monotonic>::now();
        let mut anchor = activation_started;
        let mut sent = 0u32;
        let mut started = 0u32;
        let timeline = (|| -> Result<(u32, u32)> {
            if self.dock_wide_modeset() {
                let remap = |slot: u8| connector_of(slot) as u8;

                // DLM's first clear pair begins a dock-wide sink reset. The authenticated
                // transcript then stops/restarts each EDID reader, disengages/re-engages the
                // downstream sinks, and clears each pipe a second time before any real mode.
                for (slot, &connector) in slots.iter().enumerate() {
                    if slot == 1 {
                        Self::wait_mode_offset(activation_started, NAVARRO_PRIME_CLEAR_H1_MS);
                    }
                    self.send_cp(dev, 0x48, 0, |ctr| crate::cp::clear_mode(ctr, connector))?;
                }
                for &(at, op) in NAVARRO_COLD_PRELUDE {
                    Self::wait_mode_offset(activation_started, at);
                    self.navarro_cold_op(dev, op.remap_head(&remap))?;
                }
                Self::wait_mode_offset(activation_started, NAVARRO_REAL_MODE_H0_MS);
                anchor = Instant::<Monotonic>::now();
            }

            let dock_wide_counters = if self.dock_wide_modeset() {
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
                            let (_, slot, sub, state) = timeline.markers[mi];
                            let connector = connector_of(slot) as u8;
                            if sent & (1u32 << connector) != 0 {
                                if let Some(counters) = dock_wide_counters.as_ref() {
                                    let slot =
                                        *NAVARRO_MARKER_COUNTER_SLOTS.get(mi).ok_or(EINVAL)?;
                                    let ctr = *counters.get(slot).ok_or(EINVAL)?;
                                    self.send_cp_reserved(dev, 0x16, ctr, |ctr| {
                                        crate::cp::stream_marker(ctr, connector, sub, state)
                                    })?;
                                } else {
                                    self.stream_marker(dev, connector, sub, state)?;
                                }
                            }
                            mi += 1;
                        } else if np == Some(off) {
                            if let Some(counters) = dock_wide_counters.as_ref() {
                                let slot = *NAVARRO_POLL_COUNTER_SLOTS.get(pi).ok_or(EINVAL)?;
                                let ctr = *counters.get(slot).ok_or(EINVAL)?;
                                self.send_cp_reserved(dev, 0x14, ctr, |ctr| {
                                    crate::cp::device_query_req(ctr, 0x000c)
                                })?;
                            } else {
                                self.poll_status(dev)?;
                            }
                            pi += 1;
                        } else {
                            let (_, slot, fetch) = timeline.edid[ei];
                            let connector = connector_of(slot) as u8;
                            // Re-read the sink's EDID at its required place in the transaction.
                            // This dock-side DDC operation is not a source of new modes, so discard
                            // its reply rather than publishing a hotplug during a mode set.
                            self.send_cp(dev, 0x15, 0, |ctr| {
                                if fetch {
                                    crate::cp::get_edid_req(ctr, connector)
                                } else {
                                    crate::cp::get_edid_req_sub(ctr, 0x0020, connector)
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
            for slot in 0..slots.len() {
                let connector = connector_of(slot as u8);
                let bit = 1u32 << connector;
                if valid & bit == 0 {
                    continue;
                }
                let Some(timing) = timings[connector] else {
                    continue;
                };
                // The second connector's mode set is spaced from the first by this dock's measured
                // interval -- 757 ms on Navarro, 29 ms on Ridge. Gate on the slot, so the spacing
                // survives whichever sockets the monitors are in.
                if slot == 1 {
                    cp_until!(timeline.h1_mode - 1);
                    Self::wait_mode_offset(anchor, timeline.h1_mode);
                }
                // Retrain the downstream link onto the new timing. The sink goes down
                // immediately ahead of the timing and the post-mode bracket brings it back up; a
                // dock whose vendor does not bracket this way carries no state here.
                if let Some(state) = self.pre_mode_sink_state() {
                    self.stream_marker(dev, connector as u8, 0x2f, 1)?;
                    self.stream_marker(dev, connector as u8, 0x2e, state)?;
                }
                if let Some(counters) = dock_wide_counters.as_ref() {
                    // Reservation-token slots for the two mode sets, by activation order. Keyed on
                    // the connector number these collided for any pair but (0, 1), and Navarro NAKs
                    // from the first flattened counter onward.
                    let ctr_slot = if slot == 0 { 0 } else { 3 };
                    let ctr = *counters.get(ctr_slot).ok_or(EINVAL)?;
                    self.send_cp_reserved(dev, 0x48, ctr, |ctr| {
                        crate::cp::set_mode(ctr, connector as u8, &timing)
                    })?;
                } else {
                    self.send_cp(dev, 0x48, 0, |ctr| {
                        crate::cp::set_mode(ctr, connector as u8, &timing)
                    })?;
                }
                self.programmed_timing.lock()[connector] = Some(timing);
                if self.modeset_requested[connector].load(Ordering::Acquire) != keys[connector] {
                    continue;
                }
                self.modeset_active[connector].store(keys[connector], Ordering::Release);
                self.sustain_until.lock()[connector] = self.sustain_window(connector);
                self.arm_stream_prologue(connector);
                // A driven connector's stream opens with its pipe descriptor, not the idle open.
                self.stream_open_pending.fetch_and(!bit, Ordering::Release);
                self.owe_keyframe(connector);
                self.strip_hashes.lock()[connector] = None;
                self.dirty_ttl.lock()[connector] = None;
                sent |= bit;
            }

            // Preserve the required silent window on EP02 between the connector-1 mode set and
            // `cold::QUIET_END`. The exclusive control timeline already excludes keepalives.
            Self::wait_mode_offset(anchor, timeline.quiet_end);

            // Bracket, status polls and the mid-bracket EDID re-read, up to the first video.
            cp_until!(timeline.video[0].1 - 1);

            // DLM opens a short sealed stream on every connector without a monitor at this point.
            // Vino does not: this dock re-enumerates when a stream is driven at an empty connector,
            // and an empty socket's index is not stable across bring-ups.

            for &(vslot, at) in timeline.video {
                let connector = connector_of(vslot as u8);
                cp_until!(at - 1);
                // Some docks set a connector's mode a second time shortly before its video.
                for &(off, reslot) in timeline.remode {
                    let replay_connector = connector_of(reslot as u8);
                    if off >= at
                        || remoded & (1u32 << replay_connector) != 0
                        || sent & (1u32 << replay_connector) == 0
                    {
                        continue;
                    }
                    let Some(timing) = timings[replay_connector] else {
                        continue;
                    };
                    cp_until!(off - 1);
                    Self::wait_mode_offset(anchor, off);
                    self.send_cp(dev, 0x48, 0, |ctr| {
                        crate::cp::set_mode(ctr, replay_connector as u8, &timing)
                    })?;
                    self.programmed_timing.lock()[replay_connector] = Some(timing);
                    remoded |= 1u32 << replay_connector;
                }
                cp_until!(at - 1);
                Self::wait_mode_offset(anchor, at);
                let bit = 1u32 << connector;
                if sent & bit == 0 {
                    continue;
                }
                // Exactly one ARM+carrier presentation keeps the closing markers from being
                // delayed behind a blocking multi-frame submission.
                let frames = prompts[connector].as_ref().ok_or(EINVAL)?;
                let ordinary_frames = ordinary_prompts[connector].as_ref().ok_or(EINVAL)?;
                let t_sub = Instant::<Monotonic>::now();
                let first_for_head = started & bit == 0;
                self.submit_prompt_training(
                    dev,
                    connector as u8,
                    keys[connector],
                    frames,
                    ordinary_frames,
                    self.carrier_ms(PROMPT_TRAINING_OPEN_MS),
                    first_for_head,
                )?;
                vino_debug!(
                    "vino: connector {} video submit took {} ms (timeline offset {} ms, {} ms since anchor)\n",
                    connector,
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
                for connector in 0..MAX_CONNECTORS {
                    if sent & (1u32 << connector) != 0
                        && self.modeset_active[connector].load(Ordering::Acquire) == keys[connector]
                    {
                        self.modeset_active[connector].store(0, Ordering::Release);
                    }
                }
                // The choreography opens every activating connector's bracket well before it sets a
                // mode, so unwind on `valid` rather than `sent`: a connector that failed before its
                // mode set is still open on the dock.
                for connector in 0..MAX_CONNECTORS {
                    if valid & (1u32 << connector) != 0 {
                        self.unwind_bracket(dev, connector as u8);
                    }
                }
                return Err(e);
            }
        };
        if sent.count_ones() < 2 {
            for connector in 0..MAX_CONNECTORS {
                if sent & (1u32 << connector) != 0
                    && self.modeset_active[connector].load(Ordering::Acquire) == keys[connector]
                {
                    self.modeset_active[connector].store(0, Ordering::Release);
                }
            }
            return Ok(false);
        }

        // Keep both endpoints busy through downstream clock training, so the carrier outlives the
        // bracket rather than stopping with it.
        let tail_started = Instant::<Monotonic>::now();
        while (Instant::<Monotonic>::now() - tail_started).as_millis()
            < self.carrier_ms(cold::CARRIER_TAIL_MS)
        {
            for connector in 0..MAX_CONNECTORS {
                if started & (1u32 << connector) == 0 {
                    continue;
                }
                let frames = prompts[connector].as_ref().ok_or(EINVAL)?;
                let ordinary_frames = ordinary_prompts[connector].as_ref().ok_or(EINVAL)?;
                if let Err(e) = self.submit_prompt_training(
                    dev,
                    connector as u8,
                    keys[connector],
                    frames,
                    ordinary_frames,
                    self.carrier_ms(PROMPT_TRAINING_OPEN_MS),
                    false,
                ) {
                    for reset in 0..MAX_CONNECTORS {
                        if sent & (1u32 << reset) != 0
                            && self.modeset_active[reset].load(Ordering::Acquire) == keys[reset]
                        {
                            self.modeset_active[reset].store(0, Ordering::Release);
                        }
                    }
                    for reset in 0..MAX_CONNECTORS {
                        if valid & (1u32 << reset) != 0 {
                            self.unwind_bracket(dev, reset as u8);
                        }
                    }
                    return Err(e);
                }
            }
        }
        vino_debug!(
            "vino: dual-connector activation complete after {} ms (mode/started masks 0x{:x}/0x{:x})\n",
            (Instant::<Monotonic>::now() - anchor).as_millis(),
            sent,
            started
        );
        Ok(true)
    }
    /// Activate both connectors of a dock that carries its video on the control pipe.
    ///
    /// Replays [`ELLA_DOCK_WIDE`]: one transaction configures both connectors, the first streams,
    /// and the second's sink comes up behind it. The per-connector schedule cannot express this,
    /// because its second pass sets a mode the dock has already been given and opens a bracket the
    /// dock stops answering.
    pub(super) fn activate_dock_wide(
        &self,
        dev: &BoundInterface<'_>,
        steps: &[DockWideStep],
        mut timings: [Option<crate::cp::Timing>; MAX_CONNECTORS],
    ) -> Result<bool> {
        // Connectors the table drives, which is what decides how many the caller has to supply.
        let wanted = steps
            .iter()
            .filter_map(|step| match *step {
                DockWideStep::SetMode(slot) => Some(u32::from(slot) + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let geometry = self.geometry();
        let mut prompts: [Option<KVec<KVec<u8>>>; MAX_CONNECTORS] = core::array::from_fn(|_| None);
        let mut ordinary_prompts: [Option<KVec<KVec<u8>>>; MAX_CONNECTORS] =
            core::array::from_fn(|_| None);
        let mut keys = [0u64; MAX_CONNECTORS];
        let mut valid = 0u32;
        let requested_heads = self.requested_connector_mask();

        // Encode every carrier before the transaction opens. Where a table sets two modes they are
        // adjacent on the wire and must not be separated by this dock's encoder.
        for connector in 0..MAX_CONNECTORS {
            let Some(timing) = timings[connector] else {
                continue;
            };
            let key = timing_key(&timing);
            // Only that the connector still wants this mode. Whether it is already lit is the
            // caller's to decide: the cold table takes connectors whose generation is zero, and the
            // runtime table exists precisely to reconfigure one that is streaming.
            if self.modeset_requested[connector].load(Ordering::Acquire) != key {
                continue;
            }
            timings[connector] =
                Some(self.effective_timing_in_mask(connector, &timing, requested_heads));
            let padded_width =
                (timing.hactive as usize + geometry.strip_w() - 1) & !(geometry.strip_w() - 1);
            let padded_height =
                (timing.vactive as usize + geometry.strip_h() - 1) & !(geometry.strip_h() - 1);
            prompts[connector] = Some(crate::video::haar::black_frame_ep08(
                geometry,
                padded_width,
                padded_height,
                connector as u8,
            )?);
            ordinary_prompts[connector] = Some(crate::video::haar::black_frame_ep08_ordinary(
                geometry,
                padded_width,
                padded_height,
                connector as u8,
            )?);
            keys[connector] = key;
            valid |= 1u32 << connector;
        }
        if valid.count_ones() < wanted {
            return Ok(false);
        }

        // Slot -> connector, in activation order. Two is this dock's whole complement of
        // connectors.
        let mut slots = [0u8; 2];
        let mut n = 0;
        for connector in 0..MAX_CONNECTORS {
            if valid & (1u32 << connector) != 0 {
                if n < slots.len() {
                    slots[n] = connector as u8;
                }
                n += 1;
            }
        }
        if n > slots.len() {
            pr_warn!(
                "vino: {n} connectors activating but this dock has {}; choreographing {} and {}\n",
                slots.len(),
                slots[0],
                slots[1]
            );
        }
        let connector_of = |slot: u8| usize::from(slots[usize::from(slot).min(slots.len() - 1)]);

        // The runtime table can enter with a live target. Once any table step runs, its old stream
        // state is no longer safe to adopt: Ella's live table changes sink markers before SetMode.
        // Invalidate every participant before the first bracket/control write. A request that moved
        // while carriers were encoded is then left at zero for its newer command to establish.
        for connector in 0..MAX_CONNECTORS {
            if valid & (1u32 << connector) != 0 {
                self.modeset_active[connector].store(0, Ordering::Release);
                self.programmed_timing.lock()[connector] = None;
            }
        }
        if (0..MAX_CONNECTORS).any(|connector| {
            valid & (1u32 << connector) != 0
                && self.modeset_requested[connector].load(Ordering::Acquire) != keys[connector]
        }) {
            return Ok(false);
        }

        for connector in 0..MAX_CONNECTORS {
            if valid & (1u32 << connector) != 0 {
                self.close_blank_bracket(dev, connector as u8)?;
            }
        }

        self.begin_cp_timeline();
        let started = Instant::<Monotonic>::now();
        let mut sent = 0u32;
        let transaction = (|| -> Result<u32> {
            for step in steps {
                match *step {
                    DockWideStep::SetMode(slot) => {
                        let connector = connector_of(slot);
                        let timing = timings[connector].ok_or(EINVAL)?;
                        self.send_cp(dev, 0x48, 0, |ctr| {
                            crate::cp::set_mode(ctr, connector as u8, &timing)
                        })?;
                        self.programmed_timing.lock()[connector] = Some(timing);
                        if self.modeset_requested[connector].load(Ordering::Acquire)
                            != keys[connector]
                        {
                            return Ok(sent);
                        }
                        self.modeset_active[connector].store(keys[connector], Ordering::Release);
                        self.sustain_until.lock()[connector] = self.sustain_window(connector);
                        self.arm_stream_prologue(connector);
                        // A driven connector's stream opens with its pipe descriptor, not the idle
                        // open.
                        self.stream_open_pending
                            .fetch_and(!(1u32 << connector), Ordering::Release);
                        self.owe_keyframe(connector);
                        self.strip_hashes.lock()[connector] = None;
                        self.dirty_ttl.lock()[connector] = None;
                        sent |= 1u32 << connector;
                    }
                    DockWideStep::Marker(slot, sub, state) => {
                        self.stream_marker(dev, connector_of(slot) as u8, sub, state)?;
                    }
                    DockWideStep::Poll => self.poll_status(dev)?,
                    DockWideStep::Prologue(slot) => {
                        self.send_stream_prologue(dev, connector_of(slot) as u8)?;
                    }
                    DockWideStep::Ring(slot) => {
                        self.send_stream_ring(dev, connector_of(slot) as u8)?;
                    }
                    DockWideStep::Config(slot) => {
                        self.send_stream_config(dev, connector_of(slot) as u8)?;
                    }
                    DockWideStep::Carrier(slot) => {
                        let connector = connector_of(slot);
                        if sent & (1u32 << connector) == 0 {
                            continue;
                        }
                        self.submit_prompt_training(
                            dev,
                            connector as u8,
                            keys[connector],
                            prompts[connector].as_ref().ok_or(EINVAL)?,
                            ordinary_prompts[connector].as_ref().ok_or(EINVAL)?,
                            self.carrier_ms(PROMPT_TRAINING_OPEN_MS),
                            true,
                        )?;
                    }
                    DockWideStep::Stream(slot, frames) => {
                        // The vendor does not pause here, it keeps presenting. Its second connector
                        // is configured and its sink brought up while the first one's stream is
                        // running, so the records around this step reach a dock that is mid-frame
                        // -- which sleeping through reproduces on the wire and not at all in what
                        // the dock is doing. Present the connector's own flat surface: this
                        // transaction owns the endpoint exclusively and cannot take the
                        // compositor's live frame, but keeping the stream and its ring advancing is
                        // what the step is for.
                        let connector = connector_of(slot);
                        if sent & (1u32 << connector) == 0 {
                            continue;
                        }
                        self.submit_prompt_training(
                            dev,
                            connector as u8,
                            keys[connector],
                            prompts[connector].as_ref().ok_or(EINVAL)?,
                            ordinary_prompts[connector].as_ref().ok_or(EINVAL)?,
                            self.carrier_ms(
                                self.frame_period_ms().saturating_mul(i64::from(frames)),
                            ),
                            false,
                        )?;
                    }
                }
            }
            Ok(sent)
        })();
        self.end_cp_timeline();

        let sent = match transaction {
            Ok(sent) => sent,
            Err(e) => {
                for connector in 0..MAX_CONNECTORS {
                    if valid & (1u32 << connector) != 0 {
                        self.modeset_active[connector].store(0, Ordering::Release);
                        self.programmed_timing.lock()[connector] = None;
                    }
                }
                // Unwind on `valid`: the transaction opens every activating connector's bracket
                // before it reaches that connector's mode set, so a connector that failed early is
                // still open on the dock.
                for connector in 0..MAX_CONNECTORS {
                    if valid & (1u32 << connector) != 0 {
                        self.unwind_bracket(dev, connector as u8);
                    }
                }
                return Err(e);
            }
        };
        if sent.count_ones() < wanted {
            for connector in 0..MAX_CONNECTORS {
                if valid & (1u32 << connector) != 0 {
                    self.modeset_active[connector].store(0, Ordering::Release);
                    self.programmed_timing.lock()[connector] = None;
                    self.unwind_bracket(dev, connector as u8);
                }
            }
            return Ok(false);
        }
        vino_debug!(
            "vino: dock-wide activation complete after {} ms (connectors 0x{:x})\n",
            (Instant::<Monotonic>::now() - started).as_millis(),
            sent
        );
        Ok(true)
    }
}
