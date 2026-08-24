// SPDX-License-Identifier: GPL-2.0

//! The post-authentication setup burst.
//!
//! Once the AKE completes the dock still knows nothing about how it is to be driven. This is the
//! sequence that tells it: the dock-wide records, the per-connector HDCP restatement, the video
//! engine transition and the connector-selecting records. Its ordering is measured from the
//! vendor, and a message sent out of turn desynchronises the authenticated counter for the rest of
//! the session.

use super::*;

impl VinoDriver {
    /// Configure the encrypted control plane after SKE.
    ///
    /// The sequence contains the plaintext arm marker, the first encrypted message, initialization,
    /// per-connector authentication and stream finalization. The returned counters continue the
    /// live session, and `video_keys` receives the key and nonce established for each connector.
    pub(crate) fn send_cp_setup(
        link: &UsbLink<'_>,
        profile: &DockProfile,
        session: &mut Session,
        // Scratch slot filled by reply drains and moved into the selected connector's EDID cache.
        edid_out: &mut Option<KVec<u8>>,
        edid_connectors: &mut [Option<KVec<u8>>; Self::CP_SETUP_CONNECTORS],
        video_keys: &mut [kernel::crypto::Secret<32>; Self::CP_SETUP_CONNECTORS],
        connectors_present: &mut [bool; Self::CP_SETUP_CONNECTORS],
        discovery_deferred: &mut [bool; Self::CP_SETUP_CONNECTORS],
        // Which connectors had their video stream opened in this burst, as a connector bitmask. A
        // connector with no sink yet is skipped here and owes its open to whatever drives it later.
        stream_opened: &mut u32,
    ) -> Result<(usize, u32, u16)> {
        let connector_count =
            usize::from(profile.topology.connectors).min(Self::CP_SETUP_CONNECTORS);
        // 16 KiB so the dock's ~5787 B capability block is read whole (see [`EP84_BUF`]).
        let mut resp = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL)?;
        let mut drained = 0usize;
        let mut acks = 0usize;
        let mut rejects = 0usize;
        let mut sent = 0usize;
        // Match each display-capability response to the stream-open counter of its connector.
        let mut stream_open_ctr: [Option<u16>; Self::CP_SETUP_CONNECTORS] =
            [None; Self::CP_SETUP_CONNECTORS];

        // Plaintext `type=2 sub=0x24`+`0x45` stream-open arm marker -- the mandatory gate
        // before the first encrypted frame.
        const STREAM_OPEN: [u8; 64] = [
            0x00, 0x00, 0x1c, 0x00, 0x02, 0x00, 0x00, 0x00, // pad, size, type
            0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // wsub=0x24, aux=0, seq=0
            0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, // payload
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // padding
            0x00, 0x00, 0x1c, 0x00, 0x02, 0x00, 0x00, 0x00, // pad, size, type
            0x45, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // wsub=0x45, aux=0, seq=0
            0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, // payload
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // padding
        ];

        // Post the persistent EP84 reader before arming so asynchronous replies cannot fill the
        // dock's IN FIFO while the host submits control traffic.
        let ep84_depth = profile.protocol.ep84_queue_depth;
        let mut ep84_q = match link.ctrl_in_queue(ep84_depth, EP84_BUF) {
            Ok(q) => {
                vino_debug!("vino: EP84 async IN queue opened (depth={ep84_depth})\n");
                Some(q)
            }
            Err(e) => {
                vino_debug!("vino: EP84 queue unavailable ({e:?}); using synchronous reads\n");
                None
            }
        };

        let mut out_q = match link.ctrl_out_queue(4, 1024) {
            Ok(q) => {
                vino_debug!("vino: EP02 async OUT queue opened (depth=4)\n");
                Some(q)
            }
            Err(e) => {
                vino_debug!("vino: EP02 async OUT queue open failed ({e:?})\n");
                None
            }
        };

        // Submit and flush the arm before sealing the first encrypted message.
        let arm_res = match out_q.as_mut() {
            Some(q) => q
                .send(link.io(), &STREAM_OPEN, timeout())
                .and_then(|()| q.flush(link.io(), timeout())),
            None => link
                .ctrl_send(&STREAM_OPEN, timeout(), GFP_KERNEL)
                .map(|_| ()),
        };
        arm_res?;
        // The first live message continues the AKE inner counter and starts the encrypted wire
        // block counter at zero. Every following message advances both counters from its true size.
        let mut cp_ctr: u16 = session.next_ctr;
        let mut wseq: u32 = 0;

        let content = cp::session_hello(cp_ctr);
        let body_len = content.len() + 16; // AES-CTR ciphertext + 16-byte Dl3Cmac
        let size = ((16 + body_len) - 4) as u16;
        let aux = cp::aux_for_id(0x14, body_len);
        let mut hdr = [0u8; 16];
        hdr[2..4].copy_from_slice(&size.to_le_bytes());
        hdr[4..8].copy_from_slice(&4u32.to_le_bytes()); // type=4
        hdr[8..10].copy_from_slice(&0x24u16.to_le_bytes()); // sub=0x24 (interactive CP)
        hdr[10..12].copy_from_slice(&aux.to_le_bytes());
        // Running AES-CTR block index, initially zero.
        hdr[12..16].copy_from_slice(&wseq.to_le_bytes());
        let frame = cp::seal_livemac(&session.ks, &session.riv, &hdr, &content)?;

        match out_q.as_mut() {
            Some(q) => {
                q.send(link.io(), &frame, timeout())?;
                q.flush(link.io(), timeout())?;
                if profile.protocol.reply_discipline == profile::ReplyDiscipline::Lockstep {
                    // DLM advances as soon as the dock authenticates msg0. The old eight-drain
                    // loop waited for seven empty 10-ms windows after that reply and moved every
                    // following setup transition roughly 90 ms later on the wire.
                    let d = Self::lockstep_reply(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        cp_ctr,
                        edid_out,
                    );
                    drained += d.reads;
                    acks += d.acks;
                    rejects += d.rejects;
                } else {
                    for _ in 0..8 {
                        let d = Self::drain_ep84(
                            link,
                            ep84_q.as_mut(),
                            &mut resp,
                            session,
                            edid_out,
                            Delta::from_millis(10),
                        );
                        drained += d.reads;
                        acks += d.acks;
                        rejects += d.rejects;
                    }
                }
            }
            None => {
                // A NAK transfers no bytes, so cancel and retry are safe.
                // Between attempts drain EP84 so the dock can push/drain its IN queue. Bounded.
                const TRIES: usize = 40;
                let mut last_err = ETIMEDOUT;
                let mut accepted = false;
                // Name what the dock is refusing. This path returns `control session failed ...
                // (ETIMEDOUT)`, which is otherwise indistinguishable from a reply that never
                // arrived; the distinction matters because this is the send side, where the dock
                // has stopped taking EP02 writes at all.
                let mut nak_reported = false;
                for _ in 0..TRIES {
                    match link.ctrl_send(&frame, Delta::from_millis(5), GFP_KERNEL) {
                        Ok(_) => {
                            accepted = true;
                            break;
                        }
                        // OUT NAK'd (nothing transferred) -- let the dock push on EP84, then retry.
                        Err(e) => {
                            last_err = e;
                            if !nak_reported {
                                nak_reported = true;
                                vino_debug!(
                                    "vino: EP02 NAKed a {} B control frame (cp_ctr={cp_ctr}, wseq={wseq}); retrying up to {TRIES}x\n",
                                    frame.len()
                                );
                            }
                            let d = Self::drain_ep84(
                                link,
                                ep84_q.as_mut(),
                                &mut resp,
                                session,
                                edid_out,
                                Delta::from_millis(10),
                            );
                            drained += d.reads;
                            acks += d.acks;
                            rejects += d.rejects;
                        }
                    }
                }
                if !accepted {
                    pr_err!(
                        "vino: EP02 refused {TRIES} submissions of a {} B frame (cp_ctr={cp_ctr}, wseq={wseq}) -- giving up\n",
                        frame.len()
                    );
                    return Err(last_err);
                }
            }
        }
        sent += 1;
        cp_ctr += 1; // past msg0
        wseq += 2; // msg0 content is 32 B = 2 AES blocks

        // Initialization continues with two dock-wide records and one `0x16/0x2a` record for
        // every physical connector. The authenticated Navarro transcript carries selectors
        // 0,1,2,3 here; emitting Ridge's historical pair left all later counters four AES blocks
        // behind DLM and never initialized Navarro's last two connector slots.
        macro_rules! send_init {
            ($id:expr, $sub:expr, $fixed_prefix:expr) => {{
                let id: u16 = $id;
                let sub: u16 = $sub;
                let fixed_prefix: &[u8] = $fixed_prefix;
                let mut c = [0u8; 32];
                c[0..2].copy_from_slice(&id.to_le_bytes());
                c[2..4].copy_from_slice(&sub.to_le_bytes());
                c[4..6].copy_from_slice(&cp_ctr.to_le_bytes());
                rng::fill(&mut c[22..32]);
                c[22..22 + fixed_prefix.len()].copy_from_slice(fixed_prefix);
                let frame = cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c)?;
                Self::submit_cp_frame(link, &mut out_q, &frame)?;
                sent += 1;
                // Navarro's reference transaction is reply-lockstep: the next operation follows
                // the matching authenticated counter immediately. A generic burst drain adds an
                // empty 10-ms read after every acknowledgment and changes the EP0/EP02 ordering.
                let d = if profile.protocol.reply_discipline == profile::ReplyDiscipline::Lockstep {
                    Self::lockstep_reply(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        cp_ctr,
                        edid_out,
                    )
                } else {
                    Self::drain_ep84(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        edid_out,
                        Delta::from_millis(10),
                    )
                };
                drained += d.reads;
                acks += d.acks;
                rejects += d.rejects;
                cp_ctr += 1;
                wseq += 2; // every initialization message is 32 B = 2 AES blocks
            }};
        }
        // See `DockProfile::dock_wide_init`: a dock that does not expect these is left every
        // later inner counter and AES block out of step by them.
        if profile.protocol.dock_wide_init {
            send_init!(0x0014, 0x0030, &[]);
            send_init!(0x0015, 0x000b, &[0x01]);
        }

        if profile.protocol.video_commit_point == profile::VideoCommitPoint::BeforeConnectorRecords
        {
            // The working DLM transaction places the video-engine transition at this exact
            // authenticated boundary: after the reply to 0x15/0x0b (counter 11), before the four
            // connector-selecting 0x16/0x2a records (counters 12..15). Measured submit times are
            // EP08 clear, +12.647 ms EP0a clear, +143 us vendor commit, +2.941 ms first 0x16/0x2a.
            // Performing the same requests after finalization moved them 53 messages later.
            link.clear_video_halt_wire(0)?;
            fsleep(Delta::from_millis(13));
            link.clear_video_halt_wire(1)?;
            link.control_send(
                0x24,
                0x40, /* VENDOR_OUT */
                0,
                0,
                &[],
                timeout(),
                GFP_KERNEL,
            )?;
            let mut state2 = [0u8; 28];
            link.control_recv(0x22, 0xc1, 1, 0, &mut state2, timeout(), GFP_KERNEL)?;
            fsleep(Delta::from_millis(3));
        }
        if profile.protocol.dock_wide_init {
            for connector in 0..connector_count {
                let prefix = [connector as u8, 0x01];
                send_init!(0x0016, 0x002a, &prefix);
            }
        }

        // Drain pending replies before starting the per-connector authentication blocks. Each block
        // mirrors the HDCP AKE layout and ends by opening that connector's stream.
        {
            let d = Self::drain_ep84(
                link,
                ep84_q.as_mut(),
                &mut resp,
                session,
                edid_out,
                Delta::from_millis(10),
            );
            drained += d.reads;
            acks += d.acks;
            rejects += d.rejects;
            if let Some(c) = d.display_cap_ctr {
                for h in 0..Self::CP_SETUP_CONNECTORS {
                    if stream_open_ctr[h] == Some(c) {
                        connectors_present[h] = true;
                    }
                }
            }
        }
        // Which connectors completed their downstream authentication. A connector with nothing
        // plugged into it never runs one, so this is not expected to be all of them.
        let mut connector_ok = [false; Self::CP_SETUP_CONNECTORS];
        let mut heads_authenticated = 0usize;
        'per_head: for connector in 0..connector_count {
            // The socket number printed on the dock's case; the wire counts connectors from zero.
            let socket = connector + 1;
            // Derive an independent HDCP 2.2 authentication chain for this downstream connector.
            let mut rtx_h = [0u8; drm_hdcp::RTX_LEN];
            rng::fill(&mut rtx_h);
            let mut km_h =
                kernel::crypto::Secret::<{ drm_hdcp::ENCRYPTED_SESSION_KEY_LEN }>::zeroed();
            rng::fill(&mut km_h[..]);
            let mut rn_h = [0u8; drm_hdcp::RN_LEN];
            rng::fill(&mut rn_h);
            let mut ske_ks_h =
                kernel::crypto::Secret::<{ drm_hdcp::ENCRYPTED_SESSION_KEY_LEN }>::zeroed();
            rng::fill(&mut ske_ks_h[..]);
            let mut riv_h = [0u8; drm_hdcp::RIV_LEN];
            rng::fill(&mut riv_h);
            let ekpub_h = hdcp::oaep_encrypt_km(&mut session.rsa, &km_h)?;
            let mut kd_h: Option<kernel::crypto::Secret<32>> = None;
            let mut edkey_h = None;
            let mut v_h = None;
            let mut fresh_rrx: Option<[u8; drm_hdcp::RRX_LEN]> = None;
            let mut per_connector_repeater: Option<bool> = None;
            let mut rrx_applied = false;
            // SKE_Send_Eks establishes this connector's video key. Store the whitened key and the
            // video nonce derived from the delivered RIV for the scanout arm burst. Layout: key(16)
            // || nonce(8) || pad(8).
            let stream_id = profile.geometry().stream_id(connector as u8);
            video_keys[connector] = kernel::crypto::Secret::zeroed();
            // Whether the dock applies the control-plane whitening constant to a per-connector SKE
            // key is proven only for the link stream; the per-connector rule is carried over from
            // Ridge, and both docks accept the sealed records it produces.
            let video_key = cp::cp_session_key(&ske_ks_h);
            video_keys[connector][..16].copy_from_slice(&video_key[..]);
            let vnonce = cp::stream_content_nonce(&riv_h, stream_id);
            video_keys[connector][16..24].copy_from_slice(&vnonce);
            for (i, (id, sub, content_len)) in cp::CP_SETUP_PER_HEAD.iter().copied().enumerate() {
                // The per-connector `rrx` arrives with the response to AKE_No_Stored_km. It is
                // mandatory for deriving this connector's kd and Edkey before the consuming
                // messages. V is not computed until the connector's own ReceiverID_List/V' has been
                // received and verified.
                if i >= 3 && !rrx_applied {
                    let Some(rrx_h) = fresh_rrx else {
                        // No `rrx` means this connector never began a downstream authentication,
                        // which is what an empty DisplayPort connector looks like -- DLM does not
                        // run a per-connector burst for a connector with no sink either, as a
                        // capture of it driving a monitorless dock shows: one AKE for the dock,
                        // none per connector.
                        //
                        // Skip the connector rather than failing the device. Aborting here took the
                        // whole dock down whenever a single connector was empty, so a two-connector
                        // dock with one monitor never came up at all, and a dock with none was
                        // unreachable even for EDID and hotplug.
                        vino_debug!(
                            "vino: socket {socket} has no downstream sink (no AKE_Send_Rrx); skipping its authentication\n"
                        );
                        Self::announce_stream(link, &mut out_q, profile, connector)?;
                        continue 'per_head;
                    };
                    let kd = hdcp::derive_kd(&km_h, &rtx_h, &rrx_h)?;
                    edkey_h = Some(hdcp::compute_eks(&km_h, &rtx_h, &rrx_h, &rn_h, &ske_ks_h)?);
                    // Ridge retains the older reply-drain path and has only the dock-wide list
                    // available here. Navarro replaces this after SKE with the verified list from
                    // this exact connector.
                    if !profile.protocol.per_connector_onehot {
                        let vf = hdcp::compute_v_full(&kd, &session.rxid_list);
                        let mut ack = [0u8; drm_hdcp::V_PRIME_HALF_LEN];
                        ack.copy_from_slice(&vf[drm_hdcp::V_PRIME_HALF_LEN..]);
                        v_h = Some(ack);
                    }
                    kd_h = Some(kd);
                    rrx_applied = true;
                }
                // id=0x26 (Stream_Manage restatement) is fully decoded -- deterministic content,
                // not the generic path below. See `cp::stream_manage_restatement`'s doc comment.
                if id == 0x0026 {
                    let content = cp::stream_manage_restatement(
                        cp_ctr,
                        connector as u8,
                        stream_id,
                        profile.protocol.per_connector_onehot,
                    )?;
                    let frame =
                        cp::seal_interactive(&session.ks, &session.riv, id, wseq, &content)?;
                    Self::submit_cp_frame(link, &mut out_q, &frame)?;
                    sent += 1;
                    let d = if profile.protocol.per_connector_onehot {
                        Self::wait_per_connector_push(
                            link,
                            ep84_q.as_mut(),
                            &mut resp,
                            session,
                            ake::id::REPEATERAUTH_STREAM_READY,
                            Delta::from_millis(30),
                        )
                    } else {
                        Self::drain_ep84(
                            link,
                            ep84_q.as_mut(),
                            &mut resp,
                            session,
                            edid_out,
                            Delta::from_millis(10),
                        )
                    };
                    drained += d.reads;
                    acks += d.acks;
                    rejects += d.rejects;
                    fresh_rrx = fresh_rrx.or(d.per_connector_rrx);
                    if profile.protocol.per_connector_onehot && d.per_connector_mprime.is_none() {
                        pr_err!(
                            "vino: socket {socket} downstream HDCP never returned M'/Stream_Ready\n"
                        );
                        return Err(EPROTO);
                    }
                    cp_ctr += 1;
                    wseq += ((content_len + 15) / 16) as u32;
                    // The vendor names the connector's content stream here, between the HDCP
                    // restatement and the stream-open control record. The position is part of the
                    // sequence: carried ahead of the first frame instead, this dock stops
                    // answering altogether.
                    Self::announce_stream(link, &mut out_q, profile, connector)?;
                    continue;
                }
                let mut c = KVec::from_elem(0u8, content_len, GFP_KERNEL)?;
                // Shared header (id / sub=0x10 / inner counter), identical to the plaintext AKE
                // body layout (`ake::body`). The buffer is already zeroed by `from_elem`.
                c[0..2].copy_from_slice(&id.to_le_bytes());
                c[2..4].copy_from_slice(&sub.to_le_bytes());
                c[4..6].copy_from_slice(&cp_ctr.to_le_bytes());
                // Per-connector AKE messages carry the platform-specific connector marker, HDCP
                // message id at offset 27 and the standard HDCP payload at offset 28.
                match i {
                    // AKE restatements: connector marker @23, HDCP msg-id tag @27, HDCP field @28..
                    0 | 1 | 2 | 3 | 4 | 5 => {
                        cp::connector_marker(
                            &mut c,
                            connector as u8,
                            profile.protocol.per_connector_onehot,
                        );
                        c[27] = match i {
                            0 => 0x02, // AKE_Init (rtx)
                            1 => 0x13, // AKE_Transmitter_Info
                            2 => 0x04, // AKE_No_Stored_km (Ekpub)
                            3 => 0x09, // LC_Init (rn)
                            4 => 0x0b, // SKE_Send_Eks (edkey+riv)
                            _ => 0x0f, // 5: RepeaterAuth_Send_Ack (V)
                        };
                        match i {
                            0 => {
                                // AKE_Init carries Rtx and a fresh proprietary suffix.
                                c[28..36].copy_from_slice(&rtx_h);
                                rng::fill(&mut c[36..48]);
                            }
                            1 => {
                                c[28..33].copy_from_slice(&[0x00, 0x06, 0x02, 0x00, 0x02]);
                                rng::fill(&mut c[33..48]);
                            }
                            2 => {
                                c[28..156].copy_from_slice(&ekpub_h);
                                rng::fill(&mut c[156..160]);
                            }
                            3 => {
                                c[28..36].copy_from_slice(&rn_h); // LC_Init: rn
                                rng::fill(&mut c[36..48]);
                            }
                            4 => {
                                let Some(ed) = edkey_h.as_ref() else {
                                    return Err(EPROTO);
                                };
                                c[28..44].copy_from_slice(ed);
                                c[44..52].copy_from_slice(&riv_h);
                                rng::fill(&mut c[52..64]);
                            }
                            _ => {
                                let Some(v) = v_h else {
                                    return Err(EPROTO);
                                };
                                c[28..44].copy_from_slice(&v); // RepeaterAuth_Send_Ack: V
                                rng::fill(&mut c[44..48]);
                            }
                        }
                    }
                    // Stream-open control: header + zero[8..22] + 10 host-random bytes[22..32]; no
                    // connector marker, no tag (confirmed genuinely fully random across both
                    // connectors). Record this connector's request counter. The display-capability
                    // reply echoes it only when this connector has a monitor.
                    7 => {
                        if connector < stream_open_ctr.len() {
                            stream_open_ctr[connector] = Some(cp_ctr);
                        }
                        rng::fill(&mut c[22..]);
                    }
                    // strm2: connector index @22, then the `<marker> [connector*4] 04` triple
                    // @24..27, then a fresh 5-byte host-random tail. The marker is a per-platform
                    // constant, not a connector count: Ridge sends 0x06 and Navarro 0x0c for two
                    // and four connectors, but DL-3x00 sends 0x10 with two.
                    8 => {
                        c[22] = connector as u8;
                        c[24] = profile.protocol.strm2_marker;
                        c[25] = (connector as u8) * 4;
                        c[26] = 0x04;
                        rng::fill(&mut c[27..]);
                    }
                    _ => {}
                }
                let send_at = Instant::<Monotonic>::now();
                let frame = cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c)?;
                Self::submit_cp_frame(link, &mut out_q, &frame)?;
                sent += 1;
                let mut d = if profile.protocol.per_connector_onehot && i <= 5 {
                    let want = match i {
                        0 => ake::id::AKE_SEND_CERT,
                        1 => 0x14, // DisplayLink AKE_Receiver_Info
                        2 => ake::id::AKE_SEND_RRX,
                        3 => ake::id::LC_SEND_L_PRIME,
                        4 => ake::id::REPEATERAUTH_SEND_RECEIVERID_LIST,
                        _ => ake::id::RECEIVER_AUTH_STATUS,
                    };
                    Self::wait_per_connector_push(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        want,
                        Delta::from_millis(30),
                    )
                } else if profile.protocol.per_connector_onehot {
                    Self::lockstep_reply(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        cp_ctr,
                        edid_out,
                    )
                } else {
                    Self::drain_ep84(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        edid_out,
                        Delta::from_millis(10),
                    )
                };
                per_connector_repeater = per_connector_repeater.or(d.per_connector_repeater);
                fresh_rrx = fresh_rrx.or(d.per_connector_rrx);

                // AKE_No_Stored_km starts the receiver's H' calculation. Rrx is the immediate
                // result; H' and pairing info are later milestones, and DLM sends no LC_Init
                // (i == 3) until both have crossed EP84. Both platforms need the wait and the
                // drain, or the connector authenticates on material that never arrived.
                if i == 2 && !profile.protocol.per_connector_onehot {
                    hold_until(send_at, HDCP_HPRIME_WAIT_US);
                    // Wait for the rrx rather than sampling for it. A fixed drain window returns
                    // whatever has already arrived, so a connector whose receiver answers a little
                    // late reads as a connector with no sink and loses the whole rest of its burst
                    // -- including the key exchange that gives the dock a key for its content
                    // stream. How long a receiver may take to answer is a property of the dock,
                    // so the bound comes from its profile.
                    let dh = Self::wait_per_connector_push(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        ake::id::AKE_SEND_RRX,
                        Delta::from_millis(profile.protocol.perhead_rrx_wait_ms),
                    );
                    fresh_rrx = fresh_rrx.or(dh.per_connector_rrx);
                    per_connector_repeater = per_connector_repeater.or(dh.per_connector_repeater);
                    d.add(dh);
                }
                if i == 2 && profile.protocol.per_connector_onehot {
                    let Some(rrx_h) = fresh_rrx else {
                        cp_ctr += 1;
                        wseq += ((content_len + 15) / 16) as u32;
                        vino_debug!(
                            "vino: socket {socket} has no downstream sink (no AKE_Send_Rrx); skipping its authentication\n"
                        );
                        Self::announce_stream(link, &mut out_q, profile, connector)?;
                        continue 'per_head;
                    };
                    hold_until(send_at, HDCP_HPRIME_WAIT_US);
                    let dh = Self::wait_per_connector_push(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        ake::id::AKE_SEND_H_PRIME,
                        Delta::from_millis(50),
                    );
                    d.add(dh);
                    let dp = Self::wait_per_connector_push(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        0x08, // AKE_Send_Pairing_Info
                        Delta::from_millis(30),
                    );
                    d.add(dp);
                    let Some(hprime) = d.per_connector_hprime else {
                        pr_err!(
                            "vino: socket {socket} downstream HDCP returned no H'\n",
                            socket = connector + 1
                        );
                        return Err(EPROTO);
                    };
                    let kd = hdcp::derive_kd(&km_h, &rtx_h, &rrx_h)?;
                    let want_h =
                        hdcp::compute_h(&kd, &rtx_h, per_connector_repeater.unwrap_or(true));
                    if want_h != hprime {
                        pr_err!(
                            "vino: socket {socket} downstream H' mismatch\n",
                            socket = connector + 1
                        );
                        return Err(EPROTO);
                    }
                    edkey_h = Some(hdcp::compute_eks(&km_h, &rtx_h, &rrx_h, &rn_h, &ske_ks_h)?);
                    kd_h = Some(kd);
                    rrx_applied = true;
                    vino_debug!(
                        "vino: socket {socket} downstream H' verified\n",
                        socket = connector + 1
                    );
                }

                if i == 3 && profile.protocol.per_connector_onehot {
                    let Some(lprime) = d.per_connector_lprime else {
                        pr_err!(
                            "vino: socket {socket} downstream HDCP returned no L'\n",
                            socket = connector + 1
                        );
                        return Err(EPROTO);
                    };
                    let (Some(kd), Some(rrx_h)) = (kd_h.as_ref(), fresh_rrx.as_ref()) else {
                        return Err(EPROTO);
                    };
                    if hdcp::compute_l(kd, rrx_h, &rn_h) != lprime {
                        pr_err!(
                            "vino: socket {socket} downstream L' mismatch\n",
                            socket = connector + 1
                        );
                        return Err(EPROTO);
                    }
                    vino_debug!(
                        "vino: socket {socket} downstream L' verified\n",
                        socket = connector + 1
                    );
                }

                if i == 4 && profile.protocol.per_connector_onehot {
                    let Some((list_header, vprime)) = d.per_connector_v else {
                        pr_err!(
                            "vino: socket {socket} downstream HDCP returned no ReceiverID_List/V'\n"
                        );
                        return Err(EPROTO);
                    };
                    let Some(kd) = kd_h.as_ref() else {
                        return Err(EPROTO);
                    };
                    let vf = hdcp::compute_v_full(kd, &list_header);
                    if vf[..drm_hdcp::V_PRIME_HALF_LEN] != vprime {
                        pr_err!(
                            "vino: socket {socket} downstream V' mismatch\n",
                            socket = connector + 1
                        );
                        return Err(EPROTO);
                    }
                    let mut ack = [0u8; drm_hdcp::V_PRIME_HALF_LEN];
                    ack.copy_from_slice(&vf[drm_hdcp::V_PRIME_HALF_LEN..]);
                    v_h = Some(ack);
                    if session.rxid_list.as_slice() != list_header {
                        vino_debug!(
                            "vino: socket {socket} ReceiverID list differs from dock-wide list\n",
                            socket = connector + 1
                        );
                    }
                    vino_debug!(
                        "vino: socket {socket} downstream V' verified\n",
                        socket = connector + 1
                    );
                }

                if i == 5 && profile.protocol.per_connector_onehot {
                    let Some(status) = d.per_connector_auth_status else {
                        pr_err!(
                            "vino: socket {socket} downstream HDCP returned no receiver-auth status\n"
                        );
                        return Err(EPROTO);
                    };
                    if status != 0x04 {
                        pr_err!(
                            "vino: socket {socket} downstream receiver-auth status {status:#x}, expected 0x04\n"
                        );
                        return Err(EPROTO);
                    }
                }
                drained += d.reads;
                acks += d.acks;
                rejects += d.rejects;
                // Attribute a display-capability reply by its echoed stream-open counter.
                if let Some(c) = d.display_cap_ctr {
                    for h in 0..Self::CP_SETUP_CONNECTORS {
                        if stream_open_ctr[h] == Some(c) {
                            connectors_present[h] = true;
                        }
                    }
                }
                cp_ctr += 1;
                wseq += ((content_len + 15) / 16) as u32;
            }
            // Collect replies before moving to the next connector without adding another phase
            // delay.
            {
                let d = Self::drain_ep84(
                    link,
                    ep84_q.as_mut(),
                    &mut resp,
                    session,
                    edid_out,
                    Delta::from_millis(10),
                );
                drained += d.reads;
                acks += d.acks;
                rejects += d.rejects;
                if let Some(c) = d.display_cap_ctr {
                    for h in 0..Self::CP_SETUP_CONNECTORS {
                        if stream_open_ctr[h] == Some(c) {
                            connectors_present[h] = true;
                        }
                    }
                }
            }

            if trace_crypto_enabled() {
                pr_info!(
                    "vino-crypto: video connector={connector} raw_key={:02x?} delivered_riv={riv_h:02x?} key={:02x?} nonce={:02x?}\n",
                    &ske_ks_h[..],
                    &video_keys[connector][..16],
                    &video_keys[connector][16..24]
                );
            }
            connector_ok[connector] = true;
            heads_authenticated += 1;
        }
        vino_debug!(
            "vino: {heads_authenticated}/{} connector(s) authenticated\n",
            connector_count
        );

        // Navarro performs one dock-wide state transition after the last per-connector AKE and
        // before any connector finalizer. This message was absent from vino even though its
        // authenticated DLM reply reports state 2. Keep it Navarro-only until a Ridge transcript
        // establishes that platform's behavior.
        if profile.protocol.per_connector_onehot {
            let state = cp::post_auth_state_req(cp_ctr)?;
            let frame = cp::seal_interactive(&session.ks, &session.riv, 0x15, wseq, &state)?;
            Self::submit_cp_frame(link, &mut out_q, &frame)?;
            sent += 1;
            let d = Self::drain_ep84(
                link,
                ep84_q.as_mut(),
                &mut resp,
                session,
                edid_out,
                Delta::from_millis(10),
            );
            drained += d.reads;
            acks += d.acks;
            rejects += d.rejects;
            cp_ctr += 1;
            wseq += ((state.len() + 15) / 16) as u32;
        }

        // Finalize the streams of the connectors that authenticated, before entering the
        // steady-state heartbeat.
        //
        // Only those connectors: finalizing a stream whose downstream authentication never ran
        // makes the dock hard-reset a few seconds later and re-enumerate, which reads as a
        // spontaneous dock reset rather than as a message it refused. The sequence is per
        // connector, so it follows the dock's connector count rather than a fixed pair: a
        // four-connector dock finalizes four, and DLM does the same. A dock that shares its control
        // pipe is never finalized: DLM sends neither `0x16/0x4c` nor `0x15/0x4a` to it in a whole
        // session, and every message it has not been asked to handle is one more it must answer on
        // the pipe it is about to take pixels on.
        let finalize = (0..connector_count)
            .filter(|_| !profile.topology.video_on_ctrl_pipe)
            .flat_map(|c| {
                cp::CP_SETUP_FINALIZE_STEPS
                    .iter()
                    .map(move |&(id, sub)| (id, sub, c as u8))
            });
        for (id, sub, off22) in finalize {
            if (off22 as usize) < Self::CP_SETUP_CONNECTORS && !connector_ok[off22 as usize] {
                continue;
            }
            // Offset 22 selects the connector or step; sub 0x4c also carries 1 at offset 23.
            let mut c = [0u8; 32];
            c[0..2].copy_from_slice(&id.to_le_bytes());
            c[2..4].copy_from_slice(&sub.to_le_bytes());
            c[4..6].copy_from_slice(&cp_ctr.to_le_bytes());
            c[22] = off22;
            if sub == 0x004c {
                c[23] = 0x01;
                rng::fill(&mut c[24..]);
            } else {
                rng::fill(&mut c[23..]);
            }
            let frame = cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c)?;
            Self::submit_cp_frame(link, &mut out_q, &frame)?;
            sent += 1;
            let d = Self::drain_ep84(
                link,
                ep84_q.as_mut(),
                &mut resp,
                session,
                edid_out,
                Delta::from_millis(10),
            );
            drained += d.reads;
            acks += d.acks;
            rejects += d.rejects;
            cp_ctr += 1;
            wseq += 2; // 32 B content = 2 AES blocks
        }
        if let Some(queue) = out_q.as_mut() {
            // A flush timeout is not fatal: `reap()` leaves a slow slot posted for a later call,
            // and the session is otherwise complete. A dock with an empty connector never drains
            // that connector's last EP02 write, which would otherwise fail the whole setup.
            if let Err(e) = queue.flush(link.io(), timeout()) {
                vino_debug!(
                    "vino: EP02 queue flush timed out ({e:?}); session already acknowledged, continuing\n"
                );
            }
        }

        // Ridge commits after finalization. Navarro has already committed at its measured
        // post-0x15/pre-0x16 boundary above.
        if profile.protocol.video_commit_point == profile::VideoCommitPoint::AfterFinalize {
            link.control_send(
                0x24,
                0x40, /* VENDOR_OUT */
                0,
                0,
                &[],
                timeout(),
                GFP_KERNEL,
            )?;
            // Refresh the interface state after the render/commit request.
            let mut state2 = [0u8; 28];
            link.control_recv(0x22, 0xc1, 1, 0, &mut state2, timeout(), GFP_KERNEL)?;
        }

        // Read the dock's reply: a VERIFIED `wsub=0x45` ack means the cipher engaged on our frame.
        let ls = Self::lockstep_reply(link, ep84_q.as_mut(), &mut resp, session, 0x08, edid_out);
        drained += ls.reads;
        acks += ls.acks;
        rejects += ls.rejects;

        const MAX_ROUNDS: usize = 16;
        for _ in 0..MAX_ROUNDS {
            let d = Self::drain_ep84(
                link,
                ep84_q.as_mut(),
                &mut resp,
                session,
                edid_out,
                Delta::from_millis(10),
            );
            drained += d.reads;
            acks += d.acks;
            rejects += d.rejects;
            if d.reads == 0 {
                break;
            }
        }

        if acks == 0 {
            pr_err!(
                "vino: encrypted session not acknowledged (reads={drained}, rejects={rejects})\n"
            );
            return Err(EPROTO);
        }

        // Complete downstream discovery on the authenticated counter stream.
        if profile.protocol.per_connector_onehot {
            // Navarro's authenticated DLM transcript is a compact, ordered transaction. Keep it
            // separate from Ridge's older retry-heavy discovery below: hundreds of extra status
            // polls delayed Navarro's first mode by ~23 seconds and changed every live counter.
            macro_rules! navarro_send {
                ($id:expr, $body:expr) => {{
                    let id: u16 = $id;
                    let body = $body;
                    let e = Self::send_live_cp(
                        link,
                        session,
                        ep84_q.as_mut(),
                        &mut resp,
                        edid_out,
                        id,
                        wseq,
                        &body,
                    )?;
                    drained += e.reads;
                    acks += e.acks;
                    rejects += e.rejects;
                    sent += 1;
                    wseq = wseq.wrapping_add(((body.len() + 15) / 16) as u32);
                    cp_ctr += 1;
                }};
            }

            // DLM first seeks every physical connector, including empty sockets.
            for connector in 0..connector_count {
                let probe = cp::get_edid_req_sub(cp_ctr, 0x20, connector as u8)?;
                navarro_send!(0x15, probe);
            }
            let devq = cp::device_query_req(cp_ctr, 0x0000)?;
            navarro_send!(0x14, devq);

            let mut active = [false; Self::CP_SETUP_CONNECTORS];
            for connector in 0..connector_count {
                // Navarro exposes four connector numbers on two physical video pipes, and the
                // reference DLM session opened only the first connector on each pipe -- but that
                // session had its two panels in sockets 1 and 2. Sockets 3 and 4 are not aliases
                // of them: they are separate physical connectors that happen to share a bulk
                // endpoint, with their own selector, EDID and stream. Skipping them by endpoint
                // made a monitor in socket 3 or 4 invisible, with the dock never even probed for
                // it. Read every socket; engagement below is what stays selective.
                active[connector] = true;
                *edid_out = None;
                let hu8 = connector as u8;
                let kick = cp::edid_readiness_kick(cp_ctr, hu8)?;
                navarro_send!(0x16, kick);
                let probe = cp::get_edid_req_sub(cp_ctr, 0x20, hu8)?;
                navarro_send!(0x15, probe);
                let fetch = cp::get_edid_req(cp_ctr, hu8)?;
                navarro_send!(0x15, fetch);
                // The fetch drain carries the asynchronous EDID in the working DLM cadence.
                edid_connectors[connector] = edid_out.take();
                discovery_deferred[connector] = edid_connectors[connector].is_none();
            }
            // Engage each connector that answered with an EDID, once, after all kick/probe/fetch
            // triplets. Gated on a recovered EDID because that is the presence signal on this
            // platform, and because driving setup at a socket with nothing in it is exactly what
            // makes this dock hard-reset a few seconds later.
            for connector in 0..connector_count {
                if active[connector] && edid_connectors[connector].is_some() {
                    let engage = cp::edid_engage_req(cp_ctr, connector as u8)?;
                    navarro_send!(0x16, engage);
                }
            }

            // Two status samples bracket the one-shot RTC synchronization, followed by thirteen
            // more samples in the same-day working capture.
            for _ in 0..2 {
                let status = cp::device_query_req(cp_ctr, 0x000c)?;
                navarro_send!(0x14, status);
            }
            let now = kernel::time::ktime_get_real_seconds();
            let rtc = cp::rtc_sync_req(
                cp_ctr,
                now,
                *crate::module_parameters::rtc_utc_offset_minutes.value(),
            )?;
            navarro_send!(0x1e, rtc);
            for _ in 0..13 {
                let status = cp::device_query_req(cp_ctr, 0x000c)?;
                navarro_send!(0x14, status);
            }
            // The last discovery records immediately preceding DLM's clear-mode pair re-seek
            // each active connector once.
            for connector in 0..connector_count {
                if active[connector] {
                    let probe = cp::get_edid_req_sub(cp_ctr, 0x20, connector as u8)?;
                    navarro_send!(0x15, probe);
                }
            }
        } else if profile.topology.video_on_ctrl_pipe {
            // Discovery on a dock that shares its control pipe walks all its connectors through
            // each phase before starting the next, rather than taking one connector end to end.
            // Running a connector to completion first leaves the connector that went first without
            // an EDID while the second streams, which is what a single-monitor dock looks like from
            // userspace.
            //
            // It is also short: no reader kick, one engage rather than two, and no readiness
            // polling. Every extra transaction here is one the dock has to answer on the pipe it
            // will shortly be asked to take pixels on.
            macro_rules! walk_send {
                ($id:expr, $body:expr) => {{
                    let id: u16 = $id;
                    let body = $body;
                    let e = Self::send_live_cp(
                        link,
                        session,
                        ep84_q.as_mut(),
                        &mut resp,
                        edid_out,
                        id,
                        wseq,
                        &body,
                    )?;
                    drained += e.reads;
                    acks += e.acks;
                    rejects += e.rejects;
                    sent += 1;
                    wseq = wseq.wrapping_add(((body.len() + 15) / 16) as u32);
                    cp_ctr += 1;
                }};
            }

            for connector in 0..connector_count {
                let hu8 = connector as u8;
                *edid_out = None;
                let probe = cp::get_edid_req_sub(cp_ctr, 0x20, hu8)?;
                walk_send!(0x15, probe);
                let fetch = cp::get_edid_req(cp_ctr, hu8)?;
                walk_send!(0x15, fetch);
                // The EDID arrives asynchronously, a few messages behind the fetch it answers, and
                // the push names no connector -- so a connector's answer has to be collected before
                // the next connector is asked, even though DLM issues both fetches back to back.
                let waited = Instant::<Monotonic>::now();
                while edid_out.is_none() && Instant::<Monotonic>::now() - waited < EDID_REPLY_WAIT {
                    let d = Self::drain_ep84(
                        link,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        edid_out,
                        Delta::from_millis(20),
                    );
                    drained += d.reads;
                    acks += d.acks;
                    rejects += d.rejects;
                }
                edid_connectors[connector] = edid_out.take();
                vino_debug!(
                    "vino: socket {socket} EDID fetch {}\n",
                    if edid_connectors[connector].is_some() {
                        "succeeded"
                    } else {
                        "returned no EDID"
                    },
                    socket = connector + 1
                );
            }
            // Both probes are answered before either sink is engaged, and the last message before
            // the engages restates the session hello.
            let hello = cp::session_hello(cp_ctr);
            walk_send!(0x14, hello);
            // Every connector, not only the ones that answered with an EDID. This dock reports no
            // EDID for a sink it is nonetheless driving, so an EDID is evidence of presence and
            // its absence is evidence of nothing; the vendor engages both connectors before any
            // pixels and leaves an unoccupied one engaged for the life of the session.
            for connector in 0..connector_count {
                let engage = cp::edid_engage_req(cp_ctr, connector as u8)?;
                walk_send!(0x16, engage);
            }

            // Open every connector's sealed video stream, and announce its video plane.
            //
            // This is where DLM does it: after the sinks are engaged, before the capability
            // queries, and long before any mode is set. The sealed record takes block zero of the
            // stream so the decoder configuration that follows it at prologue time takes block
            // one; `set_video_keys` is told as much.
            //
            // Every connector, not only the ones that answered with an EDID: a monitor that arrives
            // after setup would otherwise be driven on a stream the dock was never told about, and
            // this dock has no video pipe to open one on later. The vendor opens both.
            //
            // A dock with a video pipe of its own is opened by the scanout path instead,
            // immediately ahead of the frame that needs it -- it has a pipe to do that on.
            // The vendor spaces the stages of this burst with status polls; see `SetupPolls`.
            // They are round trips, so a stage that follows three of them is a stage the dock was
            // given three acknowledged messages' worth of time to reach. Opening both streams back
            // to back puts the same records on the wire and gives it none of that.
            macro_rules! settle_polls {
                ($n:expr) => {{
                    for _ in 0..$n {
                        let status = cp::device_query_req(cp_ctr, 0x000c)?;
                        walk_send!(0x14, status);
                    }
                }};
            }

            for connector in 0..connector_count {
                settle_polls!(profile.protocol.setup_polls.before_open(connector as u8));
                let stream_id = profile.geometry().stream_id(connector as u8);
                let mut vkey = kernel::crypto::Secret::<16>::zeroed();
                vkey.copy_from_slice(&video_keys[connector][..16]);
                let mut vnonce = [0u8; 8];
                vnonce.copy_from_slice(&video_keys[connector][16..24]);
                let content = cp::stream_open(profile.protocol.stream_marker_kind);
                let open = cp::seal_video_arm(&vkey, &vnonce, stream_id, 0x000a, 0, &content)?;
                link.ctrl_send(&open, timeout(), GFP_KERNEL)?;
                let announce = cp::stream_announce(
                    u16::from(profile.geometry().connector_selector(connector as u8)),
                    0,
                );
                link.ctrl_send(&announce, timeout(), GFP_KERNEL)?;
                sent += 2;
                *stream_opened |= 1u32 << connector;
                let d = Self::drain_ep84(
                    link,
                    ep84_q.as_mut(),
                    &mut resp,
                    session,
                    edid_out,
                    Delta::from_millis(10),
                );
                drained += d.reads;
                acks += d.acks;
                rejects += d.rejects;
            }

            settle_polls!(profile.protocol.setup_polls.after_stream_opens);

            // Every connector again, for the same reason the engages are.
            for connector in 0..connector_count {
                let query = cp::post_edid_query(cp_ctr, connector as u8)?;
                walk_send!(0x15, query);
            }
        } else {
            // Open discovery with a heartbeat and the one-shot device-capability query.
            let hb = cp::heartbeat(cp_ctr)?;
            let e = Self::send_live_cp(
                link,
                session,
                ep84_q.as_mut(),
                &mut resp,
                edid_out,
                0x16,
                wseq,
                &hb,
            )?;
            drained += e.reads;
            acks += e.acks;
            rejects += e.rejects;
            wseq = wseq.wrapping_add(((hb.len() + 15) / 16) as u32);
            cp_ctr += 1;

            let devq = cp::device_query_req(cp_ctr, 0x0000)?;
            let e = Self::send_live_cp(
                link,
                session,
                ep84_q.as_mut(),
                &mut resp,
                edid_out,
                0x14,
                wseq,
                &devq,
            )?;
            drained += e.reads;
            acks += e.acks;
            rejects += e.rejects;
            wseq = wseq.wrapping_add(((devq.len() + 15) / 16) as u32);
            cp_ctr += 1;

            // EDID discovery is a probe/kick/fetch sequence followed by two engage messages. A
            // cold receiver then needs bounded status polling until the readiness bit is set.
            const EDID_STEP_DELAY: Delta = Delta::from_millis(100);
            const EDID_EARLY_ROUNDS: usize = 1;
            // Bound both the poll count and wall-clock duration.
            const EDID_POLL_ITERS: usize = 250;
            const EDID_POLL_DELAY: Delta = Delta::from_millis(20);
            const EDID_POLL_PROBE_EVERY: usize = 8;
            // Offset 22 selects the downstream connector. A connector the display-capability
            // transaction reported empty need not be probed, but that inference only holds once
            // some connector has reported one: a dock that pushed no capability at all has said
            // nothing about any socket, and treating silence as "empty" leaves every connector past
            // the first undiscovered.
            let capabilities_reported = connectors_present.iter().any(|&present| present);
            for connector in 0..connector_count {
                if connector != 0 && capabilities_reported && !connectors_present[connector] {
                    continue;
                }
                let hu8 = connector as u8;
                let socket = connector + 1;
                *edid_out = None;
                let mut edid_ready = false;
                let mut transport_error = None;
                'discovery: {
                    macro_rules! edid_send {
                        ($ep:expr, $body:expr, $tag:expr) => {{
                            match Self::send_live_cp(
                                link,
                                session,
                                ep84_q.as_mut(),
                                &mut resp,
                                edid_out,
                                $ep,
                                wseq,
                                &$body,
                            ) {
                                Ok(e) => {
                                    drained += e.reads;
                                    acks += e.acks;
                                    rejects += e.rejects;
                                    wseq = wseq.wrapping_add((($body.len() + 15) / 16) as u32);
                                    cp_ctr += 1;
                                    edid_ready |= e.edid_ready;
                                    vino_debug!(
                                        "vino: live connector {} {} sent\n",
                                        connector,
                                        $tag
                                    );
                                }
                                Err(e) => {
                                    transport_error = Some(e);
                                    break 'discovery;
                                }
                            }
                        }};
                    }
                    // A block that arrives before the dock reports its downstream read complete
                    // describes the dock's own bridge rather than the monitor, and publishing it
                    // drives the panel at a timing it never advertised. Offset 26 bit 7 of the
                    // presence reply is that report; drop anything offered ahead of it and let the
                    // engage below produce the real one.
                    let gate_on_ready = profile.quirks.edid_ready_reported;
                    macro_rules! edid_settled {
                        () => {{
                            if gate_on_ready && !edid_ready && edid_out.is_some() {
                                vino_debug!(
                                    "vino: socket {socket} discarding an EDID offered before the \
                                     downstream read completed\n"
                                );
                                *edid_out = None;
                            }
                            edid_out.is_some()
                        }};
                    }
                    'early: for cycle in 0..EDID_EARLY_ROUNDS {
                        if edid_settled!() {
                            break;
                        }
                        vino_debug!(
                            "vino: live get-EDID socket {socket} early round {cycle}\n",
                            socket = connector + 1
                        );
                        for _ in 0..2 {
                            let probe = cp::get_edid_req_sub(cp_ctr, 0x20, hu8)?;
                            edid_send!(0x15, probe, "get-EDID probe (id=0x15 sub=0x20)");
                            fsleep(EDID_STEP_DELAY);
                        }
                        // Start or continue the selected connector's downstream DDC read.
                        let kick = cp::edid_readiness_kick(cp_ctr, hu8)?;
                        edid_send!(0x16, kick, "get-EDID kick (id=0x16 sub=0x4b)");
                        fsleep(EDID_STEP_DELAY);
                        let req = cp::get_edid_req(cp_ctr, hu8)?;
                        edid_send!(0x15, req, "get-EDID fetch (id=0x15 sub=0x21)");
                        if edid_settled!() {
                            break 'early;
                        }
                        fsleep(EDID_STEP_DELAY);
                        // EDID arrives asynchronously after the fetch acknowledgment.
                        let reply_wait = Instant::<Monotonic>::now();
                        while Instant::<Monotonic>::now() - reply_wait < Delta::from_secs(2) {
                            let d = Self::drain_ep84(
                                link,
                                ep84_q.as_mut(),
                                &mut resp,
                                session,
                                edid_out,
                                Delta::from_millis(20),
                            );
                            drained += d.reads;
                            acks += d.acks;
                            rejects += d.rejects;
                            edid_ready |= d.edid_ready;
                            if edid_settled!() {
                                break 'early;
                            }
                        }
                    }
                    // Engage is required twice even if the EDID push arrived early.
                    for _ in 0..2 {
                        let engage = cp::edid_engage_req(cp_ctr, hu8)?;
                        edid_send!(0x16, engage, "get-EDID engage (id=0x16 sub=0x0023)");
                        fsleep(EDID_STEP_DELAY);
                    }
                    if edid_out.is_none() {
                        // Bound wall-clock time independently of the iteration
                        // count because each failed send has its own USB timeout.
                        const EDID_POLL_MAX: Delta = Delta::from_secs(6);
                        let poll_start = Instant::<Monotonic>::now();
                        'poll: for i in 0..EDID_POLL_ITERS {
                            if edid_out.is_some() || edid_ready {
                                break 'poll;
                            }
                            if Instant::<Monotonic>::now() - poll_start > EDID_POLL_MAX {
                                vino_debug!(
                                "vino: get-EDID socket {socket} readiness poll hit wall-clock cap\n"
                            );
                                break 'poll;
                            }
                            let status = cp::device_query_req(cp_ctr, 0x000c)?;
                            edid_send!(0x14, status, "device-status poll (id=0x14 sub=0x000c)");
                            if i % EDID_POLL_PROBE_EVERY == EDID_POLL_PROBE_EVERY - 1 {
                                let probe = cp::get_edid_req_sub(cp_ctr, 0x20, hu8)?;
                                edid_send!(
                                    0x15,
                                    probe,
                                    "get-EDID readiness probe (id=0x15 sub=0x20)"
                                );
                            }
                            if edid_out.is_some() || edid_ready {
                                break 'poll;
                            }
                            fsleep(EDID_POLL_DELAY);
                        }
                        vino_debug!(
                        "vino: get-EDID socket {socket} readiness poll finished (ready={edid_ready})\n"
                    );
                        // The asynchronous `id=0x194` EDID can follow the fetch
                        // acknowledgment by several messages.
                        for _ in 0..24 {
                            if edid_settled!() {
                                break;
                            }
                            let req = cp::get_edid_req(cp_ctr, hu8)?;
                            edid_send!(0x15, req, "get-EDID retry (id=0x15 sub=0x21)");
                            let d = Self::drain_ep84(
                                link,
                                ep84_q.as_mut(),
                                &mut resp,
                                session,
                                edid_out,
                                Delta::from_millis(10),
                            );
                            drained += d.reads;
                            acks += d.acks;
                            rejects += d.rejects;
                            fsleep(EDID_POLL_DELAY);
                        }
                    }
                    // Complete this connector with its post-EDID capability query.
                    let query = cp::post_edid_query(cp_ctr, hu8)?;
                    edid_send!(0x15, query, "post-EDID capability query (id=0x15 sub=0x53)");
                    // `edid_send!` folds the drain's readiness bit into `edid_ready`. This is the
                    // last statement of the per-connector iteration and the next connector
                    // re-derives it, so that update is deliberately not read again here.
                    let _ = edid_ready;
                }

                discovery_deferred[connector] = transport_error.is_some();
                if let Some(e) = transport_error {
                    // The encrypted session is already authenticated. Keep it running and recover
                    // this connector independently after a connector-local discovery timeout.
                    *edid_out = None;
                    pr_warn!(
                        "vino: socket {socket} discovery timed out ({e:?}); deferring to runtime \
                         recovery\n",
                        socket = connector + 1
                    );
                }
                edid_connectors[connector] = edid_out.take();
                vino_debug!(
                    "vino: socket {socket} EDID fetch {}\n",
                    if edid_connectors[connector].is_some() {
                        "succeeded"
                    } else {
                        "returned no EDID"
                    },
                    socket = connector + 1
                );
            }

            // KMS is the sole owner of mode selection; setup only discovers connector state.
        }

        if rejects > 0 {
            pr_warn!("vino: dock returned {rejects} undecryptable control frame(s)\n");
        }
        vino_debug!("vino: control setup tx={sent} rx={drained} ack={acks} reject={rejects}\n");
        // Hand the caller the running counters: the next free AES-CTR block (`wseq`) and inner
        // message counter (`cp_ctr`), so runtime KMS sends (mode-set/cursor) continue the sequence.
        Ok((sent, wseq, cp_ctr))
    }
}
