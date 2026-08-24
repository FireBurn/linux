// SPDX-License-Identifier: GPL-2.0

//! Opening a connector's video stream, and the records that frame it.
//!
//! Before any pixels reach a connector the dock has to be told what is coming: a ring of buffer
//! addresses, a decoder configuration, and an opening that differs by family. Every record here is
//! sealed with the connector's own video key, and the dock silently declines a stream whose
//! opening it did not expect.

use super::*;

impl VinoDrmData {
    /// Build one connector's cold video-arm burst, prepended to the first video frame after a mode
    /// set. Records #0/#1/#4/#5 are plaintext and #6/#7 contain a fixed `type=4` body. Records
    /// #2/#3/#8/#9 use this connector's video key and nonce, derived from the per-connector SKE
    /// with `riv_h ^ (0x08 | connector)` in byte 7, and share one block counter. Records #8/#9
    /// carry the decoder configuration and independent nonces. Build the per-connector video
    /// stream-open this platform wants in place of the cold ARM burst.
    ///
    /// Sealed with the connector's video key like every other video-endpoint message, and prefixed
    /// to the first frame after a mode set so it reaches the dock before any pixels. Build the
    /// prefix that opens a connector's video stream after a mode set.
    ///
    /// Ridge prefixes a cold ARM burst to the first frame; Navarro prefixes a short sealed
    /// stream-open. Both occupy the same slot ahead of any pixels, and every submission path must
    /// pick between them the same way: sending one platform's opening to the other's dock leaves
    /// the stream unopened, and the dock then watchdog-resets a few seconds later. Build the
    /// records that close a frame, in this dock's format.
    pub(super) fn build_frame_trailer(
        &self,
        connector: u8,
        seq0: u32,
    ) -> crate::video::haar::FrameTrailer {
        let geometry = self.geometry();
        if self.video_on_ctrl_pipe() {
            crate::video::haar::FrameTrailer::one(&crate::video::haar::ella_frame_close(
                geometry, connector, seq0,
            ))
        } else if self.uses_arm_burst() {
            crate::video::haar::frame_trailer(geometry, connector, seq0)
        } else {
            crate::video::haar::navarro_frame_trailer(geometry, connector, seq0)
        }
    }
    /// Build the record that starts a non-prologue DL7400 frame.
    ///
    /// Ridge carries its slot transition in the three-record trailer and the DL-3x00 in its single
    /// closing record, so neither has an opener. Navarro terminates the old frame after its close
    /// record and starts the next USB transfer with this opener instead.
    pub(super) fn build_frame_opener(
        &self,
        connector: u8,
        seq0: u32,
        prologue: bool,
    ) -> Option<KVec<u8>> {
        if self.uses_arm_burst() || self.video_on_ctrl_pipe() || prologue {
            return None;
        }
        let mut out = KVec::new();
        out.extend_from_slice(
            &crate::video::haar::navarro_frame_opener(self.geometry(), connector, seq0),
            GFP_KERNEL,
        )
        .ok()
        .map(|()| out)
    }
    /// Open a connector's video stream, once per mode generation, ahead of any pixels.
    ///
    /// Does nothing on a dock whose opening is the ARM burst carried with the first frame.
    pub(super) fn send_stream_open(&self, dev: &BoundInterface<'_>, connector: usize) -> Result {
        let bit = 1u32 << connector;
        if self.stream_open_pending.load(Ordering::Acquire) & bit == 0 {
            return Ok(());
        }
        let Some(open) = self.build_stream_open_buf(connector)? else {
            self.stream_open_pending.fetch_and(!bit, Ordering::Release);
            return Ok(());
        };
        let pipe_i = dev.video_pipe_index(connector)?;
        let mut queue_slot = self.video_q[pipe_i].lock();
        if queue_slot.is_none() {
            *queue_slot = Some(dev.video_queue(connector, 8, VIDEO_XFER)?);
        }
        let queue = queue_slot
            .as_mut()
            .get_mut()
            .as_mut()
            .ok_or(kernel::error::code::ENODEV)?;
        queue.send(dev.io(), &open, crate::timeout())?;
        self.stream_open_pending.fetch_and(!bit, Ordering::Release);
        vino_debug!("vino: connector {} video stream opened\n", connector);
        Ok(())
    }
    /// Keep every engaged connector's video endpoint from going quiet long enough for the dock to
    /// tear the link down.
    ///
    /// The DL7400 stops answering -- video *and* control -- about a second after the last byte on a
    /// video endpoint, whatever it was doing before. A compositor with nothing to redraw leaves
    /// vino silent well past that, so each connector that has not sent anything for
    /// [`NAVARRO_KEEPALIVE_MS`] sends the same sealed report DLM pairs with every frame. Called
    /// from the control keepalive, which already runs for the life of the session.
    ///
    /// Only connectors whose video queue is already open are fed: a connector that has never
    /// streamed has nothing to keep alive, and opening a queue here would start a stream nothing
    /// follows.
    pub(crate) fn send_video_keepalive(&self, dev: &BoundInterface<'_>) {
        if self.uses_arm_burst() || self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let now = Instant::<Monotonic>::now();
        for connector in 0..MAX_CONNECTORS {
            if self.modeset_active[connector].load(Ordering::Acquire) == 0 {
                continue;
            }
            let due = match self.last_video_at.lock()[connector] {
                Some(last) => (now - last).as_millis() >= NAVARRO_KEEPALIVE_MS,
                None => false,
            };
            if !due {
                continue;
            }
            let frame = self.scanout_seq.lock()[connector];
            let Ok(report) = self.build_stream_report_buf(connector, frame) else {
                continue;
            };
            let Some(report) = report else { continue };
            let Ok(pipe_i) = dev.video_pipe_index(connector) else {
                continue;
            };
            let mut queue_slot = self.video_q[pipe_i].lock();
            let Some(queue) = queue_slot.as_mut().get_mut().as_mut() else {
                continue;
            };
            if queue.send(dev.io(), &report, crate::timeout()).is_ok() {
                self.last_video_at.lock()[connector] = Some(Instant::<Monotonic>::now());
            }
        }
    }
    pub(super) fn build_stream_prefix_buf(&self, connector: usize) -> Result<KVec<u8>> {
        if self.uses_arm_burst() {
            return self.build_arm_burst_buf(connector);
        }
        if self.video_on_ctrl_pipe() {
            // Sent inside the mode-set bracket instead, where the vendor puts it, so nothing is
            // owed to the frame. The empty buffer still marks this as the frame that opens the
            // generation. See `send_stream_prologue`.
            return Ok(KVec::new());
        }
        self.build_navarro_prologue_buf(connector)
    }
    /// Write a connector's stream prologue on the control pipe, for a dock that has no video pipe.
    ///
    /// The ring descriptor and the decoder configuration are records like any other on such a
    /// dock, so they can be ordered against the mode-set markers rather than glued to the front of
    /// a frame. Docks with a pipe of their own carry theirs with the first frame, which is where
    /// their own captures put it, and get nothing here.
    pub(super) fn send_stream_prologue(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        self.send_stream_ring(dev, connector)?;
        self.send_stream_config(dev, connector)?;
        Ok(())
    }
    /// Whether `connector` still owes its shared-pipe stream prologue.
    fn stream_prologue_pending(&self, connector: u8) -> bool {
        self.video_on_ctrl_pipe()
            && usize::from(connector) < MAX_CONNECTORS
            && self.arm_prefix_pending.load(Ordering::Acquire) & (1u32 << connector) != 0
    }
    /// Send only the unsealed ring descriptor of an Ella stream prologue.
    ///
    /// DLM does not concatenate this record with the decoder configuration: status and marker
    /// records sit between them during cold activation.  Making the two records independent table
    /// actions preserves that ordering while the conservative runtime path may still invoke both
    /// through [`Self::send_stream_prologue`].
    pub(super) fn send_stream_ring(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        if !self.stream_prologue_pending(connector) {
            return Ok(());
        }
        let ring = crate::video::haar::ella_stream_open(self.geometry(), connector);
        dev.ctrl_send(&ring, crate::timeout(), GFP_KERNEL)?;
        vino_debug!(
            "vino: connector={} stream ring descriptor sent inside the bracket ({} B)\n",
            connector,
            ring.len()
        );
        Ok(())
    }
    /// Send only the sealed decoder configuration of an Ella stream prologue.
    pub(super) fn send_stream_config(&self, dev: &BoundInterface<'_>, connector: u8) -> Result {
        if !self.stream_prologue_pending(connector) {
            return Ok(());
        }
        let connector_index = usize::from(connector);
        let config = self.build_ella_config_buf(connector_index)?;
        dev.ctrl_send(&config, crate::timeout(), GFP_KERNEL)?;
        vino_debug!(
            "vino: connector={} decoder configuration sent inside the bracket ({} B)\n",
            connector,
            config.len()
        );
        Ok(())
    }
    /// The mode header a connector's stream states, built from the mode it was last given.
    ///
    /// The surface named here is the padded one the codec actually produces: a mode whose height
    /// is not a whole number of strips is encoded as the next whole one, and the dock has to be
    /// told the size it is going to be sent. Every captured mode on a dock with its own video pipe
    /// is already a whole number of strips, so this only ever rounds on the shared-pipe dock,
    /// whose 1080-line modes are stated as 1088.
    /// Whether this connector is being driven at 30 bpp.
    ///
    /// The decoder configuration and the set-mode both state the depth, and both read it here so
    /// that they cannot disagree.
    fn connector_ten_bit(&self, connector: usize) -> bool {
        self.last_timing
            .lock()
            .get(connector)
            .copied()
            .flatten()
            .is_some_and(|t| t.ten_bit)
    }

    fn stream_mode_header(&self, connector: usize) -> Result<[u8; 26]> {
        let timing = self
            .last_timing
            .lock()
            .get(connector)
            .copied()
            .flatten()
            .ok_or(ENODEV)?;
        let geometry = self.geometry();
        let pad = |value: u16, unit: usize| -> u16 {
            let unit = unit.max(1) as u16;
            value.div_ceil(unit).saturating_mul(unit)
        };
        Ok(crate::video_arm::mode_header(
            pad(timing.hactive, geometry.strip_w()),
            pad(timing.vactive, geometry.strip_h()),
            self.layout_word(),
            timing.ten_bit,
        ))
    }
    /// This connector's video sealing key and nonce, as [`set_video_keys`](Self::set_video_keys)
    /// stored them: the whitened SKE key at the front and the stream content nonce behind it.
    fn video_seal_key(&self, connector: usize) -> Result<(kernel::crypto::Secret<16>, [u8; 8])> {
        let keys = self.video_keys.lock();
        let key = keys.get(connector).ok_or(EINVAL)?;
        let mut vkey = kernel::crypto::Secret::zeroed();
        vkey.copy_from_slice(&key[..16]);
        let mut vnonce = [0u8; 8];
        vnonce.copy_from_slice(&key[16..24]);
        Ok((vkey, vnonce))
    }
    /// Build the sealed decoder configuration a connector owes ahead of its first frame on a dock
    /// that shares its control pipe.
    ///
    /// The stream itself was announced during CP setup, where the plaintext markers and the sealed
    /// open could be ordered against the rest of the sequence. What is left is what DLM sends after
    /// the mode set. The configuration continues the block counter the setup open started, which is
    /// why it must not be rebuilt from block zero.
    fn build_ella_config_buf(&self, connector: usize) -> Result<KVec<u8>> {
        let (vkey, vnonce) = self.video_seal_key(connector)?;
        let header = self.stream_mode_header(connector)?;
        let connector_selector = u8::try_from(connector).map_err(|_| EINVAL)?;
        let geometry = self.geometry();
        let stream = geometry.stream_id(connector_selector);
        let config = crate::video_arm::build_config(
            self.code_tables(),
            &header,
            &[],
            self.connector_ten_bit(connector),
        )?;
        let seq = self.take_seal_seq(connector, config.len().div_ceil(16) as u32);
        crate::cp::seal_video_arm(&vkey, &vnonce, stream, 0x0000, seq, &config)
    }
    /// Build the message a connector's video stream opens with, sent alone ahead of everything
    /// else.
    ///
    /// Ridge has none: its ARM burst opens the stream from within the first frame's transfer.
    ///
    /// Navarro has none either, for a connector it is about to drive. The short sealed open does
    /// exist on this dock, but both DLM captures send it only on the stream ids of the connectors
    /// with no monitor -- `0x17` and `0x1f` while pixels went to connectors 0 and 1 -- each as the
    /// first and only record on its own stream, sealed with that connector's own key at block 0. A
    /// driven connector's sealed chain instead opens with the pipe descriptor at block 0.
    ///
    /// It must not go out on `stream_id | 0x10`, which for connector 0 is connector 2's stream id,
    /// sealed with connector 0's key. That both signed another connector's stream with the wrong
    /// key and, because the prologue then also started at block 0, used the connector's first
    /// keystream block twice.
    fn build_stream_open_buf(&self, connector: usize) -> Result<Option<KVec<u8>>> {
        // A dock that carries video on the control pipe has already opened its streams: the same
        // record went out inside the CP setup burst, where it can be ordered against the rest of
        // setup. Sending a second one here would seal a block the dock has already accounted for.
        if self.uses_arm_burst() || self.video_on_ctrl_pipe() {
            return Ok(None);
        }
        let (vkey, vnonce) = self.video_seal_key(connector)?;
        let content = crate::cp::navarro_stream_open();
        let stream = self.geometry().stream_id(connector as u8);
        let seq = self.take_seal_seq(connector, content.len().div_ceil(16) as u32);
        Ok(Some(crate::cp::seal_video_arm(
            &vkey, &vnonce, stream, 0x0002, seq, &content,
        )?))
    }
    /// Build the sealed report a connector owes its stream for one frame.
    ///
    /// DLM pairs every frame on the frame sub with one of these on the stream sub, so a stream
    /// that sends pixels and then falls silent on its stream sub is a stream the dock stops
    /// believing in. Returns `None` on a dock whose frames carry no such record.
    ///
    /// The ordinary `aux=0x000c` form is what DLM sends for all but a handful of frames; the
    /// `aux=0x0002` form restates the mode and goes out with the frame that carries the prologue,
    /// which is the frame right after a mode set.
    pub(super) fn build_stream_report_buf(
        &self,
        connector: usize,
        frame: u32,
    ) -> Result<Option<KVec<u8>>> {
        if self.uses_arm_burst() {
            return Ok(None);
        }
        let owed = self.stream_reports_owed.get(connector).ok_or(EINVAL)?;
        if self.video_on_ctrl_pipe()
            && (frame < STREAM_REPORT_FRAME
                || owed
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                    .is_err())
        {
            return Ok(None);
        }
        let (vkey, vnonce) = self.video_seal_key(connector)?;
        let stream = self.geometry().stream_id(connector as u8);
        let with_mode = self.video_on_ctrl_pipe()
            || self.arm_prefix_pending.load(Ordering::Acquire) & (1u32 << connector) != 0;
        let (aux, content): (u16, KVec<u8>) = if with_mode {
            let header = self.stream_mode_header(connector)?;
            let mut v = KVec::new();
            if self.video_on_ctrl_pipe() {
                v.extend_from_slice(&crate::cp::stream_report_mode_only(&header), GFP_KERNEL)?;
                (0x0006, v)
            } else {
                v.extend_from_slice(&crate::cp::navarro_stream_report_mode(&header), GFP_KERNEL)?;
                (0x0002, v)
            }
        } else {
            let mut v = KVec::new();
            v.extend_from_slice(&crate::cp::navarro_stream_report(), GFP_KERNEL)?;
            (0x000c, v)
        };
        let seq = self.take_seal_seq(connector, content.len().div_ceil(16) as u32);
        Ok(Some(crate::cp::seal_video_arm(
            &vkey, &vnonce, stream, aux, seq, &content,
        )?))
    }
    /// Build the DL7400 records that precede a connector's first frame.
    ///
    /// In wire order: two plaintext stream markers, the sealed pipe descriptor, a plaintext frame
    /// marker, an unsealed record naming the connector's first and fifth ring addresses, and the
    /// sealed decoder configuration. Both sealed records draw from the stream's running block
    /// counter, so on a first arm the descriptor seals at block 0 and the configuration at block
    /// 19 -- the descriptor's 304 bytes in blocks -- exactly as DLM's `0 -> 19 -> 88` chain does.
    fn build_navarro_prologue_buf(&self, connector: usize) -> Result<KVec<u8>> {
        let (vkey, vnonce) = self.video_seal_key(connector)?;
        let connector_selector = connector as u8;
        let geometry = self.geometry();
        let stream = geometry.stream_id(connector_selector);
        let frame_sub = u16::from(geometry.connector_selector(connector_selector));

        let mut buf = KVec::with_capacity(1600, GFP_KERNEL)?;
        for sub in [stream, stream | 0x0010] {
            buf.extend_from_slice(
                &crate::cp::stream_announce(sub, crate::cp::STREAM_ANNOUNCE_MARKER),
                GFP_KERNEL,
            )?;
        }

        let descriptor = crate::cp::navarro_pipe_descriptor(connector_selector)?;
        let seal_seq = self.take_seal_seq(connector, descriptor.len().div_ceil(16) as u32);
        let sealed =
            crate::cp::seal_video_arm(&vkey, &vnonce, stream, 0x0000, seal_seq, &descriptor)?;
        buf.extend_from_slice(&sealed, GFP_KERNEL)?;

        buf.extend_from_slice(&crate::cp::stream_announce(frame_sub, 0), GFP_KERNEL)?;

        // Unsealed type-4 record: the connector_selector's first and fifth ring addresses.
        let mut ring = [0u8; 32];
        ring[2..4].copy_from_slice(&0x001cu16.to_le_bytes());
        ring[4..8].copy_from_slice(&4u32.to_le_bytes());
        ring[8..10].copy_from_slice(&frame_sub.to_le_bytes());
        ring[10..12].copy_from_slice(&0x0004u16.to_le_bytes());
        ring[16..19].copy_from_slice(&[0x0a, 0x00, 0x04]);
        ring[19] = frame_sub as u8;
        ring[22..26]
            .copy_from_slice(&crate::cp::navarro_pipe_ring(connector_selector, 0).to_le_bytes());
        ring[26..30]
            .copy_from_slice(&crate::cp::navarro_pipe_ring(connector_selector, 4).to_le_bytes());
        buf.extend_from_slice(&ring, GFP_KERNEL)?;

        let mut tail = [0u8; 14];
        crate::rng::fill(&mut tail);
        let config = crate::video_arm::build_config(
            self.code_tables(),
            &self.stream_mode_header(connector)?,
            &tail,
            self.connector_ten_bit(connector),
        )?;
        let seal_seq = self.take_seal_seq(connector, config.len().div_ceil(16) as u32);
        buf.extend_from_slice(
            &crate::cp::seal_video_arm(&vkey, &vnonce, stream, 0x000e, seal_seq, &config)?,
            GFP_KERNEL,
        )?;
        Ok(buf)
    }
    fn build_arm_burst_buf(&self, connector: usize) -> Result<KVec<u8>> {
        let (vkey, vnonce) = self.video_seal_key(connector)?;
        let h = connector as u16;
        // The sealed records share the video channel's running block counter:
        // #2 seq0(+1), #3 seq1(+1), #8 seq2(+69), and #9 seq71.
        let mut seal_seq: u32 = 0;
        let mut buf = KVec::with_capacity(2560, GFP_KERNEL)?;
        for (i, &(_wire_type, sub_base, aux, body_len)) in
            crate::cp::VIDEO_ARM_BURST.iter().enumerate()
        {
            let sub = sub_base.wrapping_add(h);
            match i {
                2 | 3 => {
                    // Sealed under the per-connector video key/nonce. Content = the six-byte stream
                    // marker + 10 host-random bytes; seq is the shared block counter (16 B = 1
                    // block each).
                    let content = crate::cp::stream_open(self.stream_marker_kind());
                    let frame =
                        crate::cp::seal_video_arm(&vkey, &vnonce, sub, aux, seal_seq, &content)?;
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
                    crate::rng::fill(&mut nonce);
                    let content = crate::video_arm::build_config(
                        self.code_tables(),
                        &self.stream_mode_header(connector)?,
                        &nonce,
                        self.connector_ten_bit(connector),
                    )?;
                    debug_assert_eq!(content.len(), body_len);
                    let frame =
                        crate::cp::seal_video_arm(&vkey, &vnonce, sub, aux, seal_seq, &content)?;
                    seal_seq += (body_len / 16) as u32;
                    buf.extend_from_slice(&frame, GFP_KERNEL)?;
                }
                _ => {
                    // wire_type==2 plaintext records (#0/#1/#4/#5).
                    let body = crate::cp::video_arm_plaintext_body(i, h);
                    let frame = crate::cp::video_arm_plain_frame(sub, &body);
                    buf.extend_from_slice(&frame, GFP_KERNEL)?;
                }
            }
        }
        Ok(buf)
    }
}
