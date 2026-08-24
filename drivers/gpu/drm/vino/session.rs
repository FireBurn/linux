// SPDX-License-Identifier: GPL-2.0

//! Bringing a dock's encrypted control session up.
//!
//! In order: the plaintext initialisation preamble, the HDCP 2.2 link AKE, then the sealed control
//! setup that authenticates each downstream connector and reads its EDID. Everything here runs
//! once per bind, on the bring-up worker; the steady-state keepalive and the KMS transactions live
//! in [`super::drm_sink`].

use super::*;

mod replies;
mod setup;

impl VinoDriver {
    /// Initialize the plaintext control transport.
    pub(super) fn bring_up(link: &UsbLink<'_>, profile: &DockProfile) -> Result {
        // Control-request preamble: dock identity, interface selection, then the
        // vendor-OUT 0x24 / vendor-IN 0x22 pair that starts the HDCP path.
        const VENDOR_OUT: u8 = 0x40; // host->link, vendor, device
        const VENDOR_IN_IFACE: u8 = 0xc1; // device-to-host, vendor, interface recipient

        // Individual vendor requests may stall. Only bulk initialization and
        // its acknowledgment are required.
        let mut identity_bytes = [0u8; 16];
        match link.control_recv(
            0xfe,
            VENDOR_IN_IFACE,
            0,
            1,
            &mut identity_bytes,
            timeout(),
            GFP_KERNEL,
        ) {
            // The raw blob is printed only when it cannot be read as an identity, which is the
            // case where someone has to decode it by hand. Info, not debug: on unfamiliar
            // hardware this is what places the device, and needing a debug build to see it costs
            // a whole test round trip.
            Ok(()) => match firmware::Identity::parse(&identity_bytes) {
                // The DFU interface names the dock at info level, so one that parses is already
                // reported once.
                Some(id) => vino_debug!("{id} running firmware {}\n", id.version),
                None => pr_info!("unrecognised device identity = {identity_bytes:02x?}\n"),
            },
            Err(e) => pr_info!("device identity unavailable ({e:?})\n"),
        }
        // A composite driver may only change its own interface.
        match link.set_alternate_setting(0) {
            Ok(()) => {}
            Err(e) => vino_debug!("vino: alternate setting unchanged ({e:?})\n"),
        }
        // The first vendor transition is platform-specific even though both platforms use the
        // same request number. Ridge uses wValue=3; both occurrences in the authenticated
        // Navarro/DLM USB transcript use wValue=0. Sending Ridge's value here still permits AKE
        // and an exact EP02 transcript, but leaves Navarro's video-side state machine different
        // before its later value-0 commit.
        let vendor_state = profile.protocol.initial_vendor_state;
        match link.control_send(
            0x24,
            VENDOR_OUT,
            vendor_state,
            0,
            &[],
            timeout(),
            GFP_KERNEL,
        ) {
            Ok(()) => {}
            Err(e) => vino_debug!("vino: vendor preamble request stalled ({e:?})\n"),
        }
        // Request interface 0 state using the vendor interface recipient.
        let mut state = [0u8; 28];
        match link.control_recv(
            0x22,
            VENDOR_IN_IFACE,
            1,
            0,
            &mut state,
            timeout(),
            GFP_KERNEL,
        ) {
            Ok(()) => vino_debug!("vino: interface state = {state:02x?}\n"),
            Err(e) => vino_debug!("vino: interface state unavailable ({e:?})\n"),
        }

        // The dock requires this exact plaintext initialization order. It acknowledges the
        // sequence only after `init_4` and the following probe. The interleaved descriptor reads
        // are best-effort because a short reply still completes the required control transfer.
        const STD_IN: u8 = 0x80; // link->host, standard, device
        let mut config_descriptor = KVec::from_elem(0u8, 618, GFP_KERNEL)?;
        let _ = link.control_recv(
            0x06,
            STD_IN,
            0x0200,
            0,
            &mut config_descriptor[..40],
            timeout(),
            GFP_KERNEL,
        ); // CONFIG, 40
        let _ = link.control_recv(
            0x06,
            STD_IN,
            0x0200,
            0,
            &mut config_descriptor,
            timeout(),
            GFP_KERNEL,
        );

        // Report EP02's maximum packet size because exact-multiple messages require an explicit
        // terminating short packet.
        {
            let total = ((config_descriptor[2] as usize) | ((config_descriptor[3] as usize) << 8))
                .min(config_descriptor.len());
            let mut i = 0usize;
            while i + 2 <= total {
                let blen = config_descriptor[i] as usize;
                if blen == 0 {
                    break;
                }
                if config_descriptor[i + 1] == 0x05
                    && i + 7 <= total
                    && config_descriptor[i + 2] == EP_CTRL_OUT
                {
                    let wmax = (config_descriptor[i + 4] as u16)
                        | ((config_descriptor[i + 5] as u16) << 8);
                    vino_debug!("vino: EP02 max packet size {wmax}\n");
                }
                i += blen;
            }
        }

        let send_required = |label: &str, msg: &[u8]| -> Result {
            match link.ctrl_send(msg, timeout(), GFP_KERNEL) {
                Ok(_) => Ok(vino_debug!("vino: sent {label} ({} bytes)\n", msg.len())),
                Err(e) => {
                    pr_err!("vino: {label} failed ({e:?})\n");
                    Err(e)
                }
            }
        };
        send_required("init_0", &proto::init_0()?)?;
        send_required("init_25", &proto::init_25()?)?;
        // Two required string reads between `init_25` and `init_4`.
        let _ = link.control_recv(
            0x06,
            STD_IN,
            0x0300,
            0x0000,
            &mut config_descriptor[..255],
            timeout(),
            GFP_KERNEL,
        ); // STRING #0
        let _ = link.control_recv(
            0x06,
            STD_IN,
            0x0303,
            0x0409,
            &mut config_descriptor[..255],
            timeout(),
            GFP_KERNEL,
        ); // STRING #3 en-US
        send_required("init_4+probe", &proto::init_4_probe()?)?;

        // Read the single ACK that follows init_4+probe.
        let mut ack = KVec::from_elem(0u8, 1024, GFP_KERNEL)?;
        match link.ctrl_recv(&mut ack, timeout(), GFP_KERNEL) {
            Ok(n) => vino_debug!(
                "vino: session-init ACK = {n} bytes: {:02x?}\n",
                &ack[..n.min(40)]
            ),
            Err(e) => {
                pr_err!("vino: session acknowledgment failed ({e:?})\n");
                return Err(e);
            }
        }

        Ok(())
    }

    /// Number of display connectors the post-msg0 CP setup burst re-states the AKE for. Tied to the
    /// single connector-count knob `drm_sink::MAX_CONNECTORS` so bumping the connector count is a
    /// one-line change (was a duplicated literal `2` that had to be kept in sync by hand).
    pub(super) const CP_SETUP_CONNECTORS: usize = drm_sink::MAX_CONNECTORS;

    /// Reads the next HDCP response (type=4 sub=0x25, sec 5.2) from EP `0x84`,
    /// skipping any non-HDCP frames (e.g. plain ACKs) in between, and returns the
    /// parsed `(msg_id, payload)`. Bounded retry so a chatty dock can't wedge us.
    fn recv_hdcp(link: &UsbLink<'_>) -> Result<(u8, KVec<u8>)> {
        const SUB_HDCP_RESP: u16 = 0x25;
        // The dock interleaves capability blocks up to ~5.8 KiB into the AKE reply
        // stream; size the buffer like the rest of the EP84 reads ([`EP84_BUF`]) so a
        // large frame is read whole rather than truncated/`-EOVERFLOW`'d.
        let mut buf = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL)?;
        for _ in 0..24 {
            // The dock interleaves status and capability pushes with the HDCP replies.
            let n = link.ctrl_recv(&mut buf, timeout(), GFP_KERNEL)?;
            if n < 16 {
                continue;
            }
            // Include interleaved capability and status frames in dynamic-debug output.
            {
                let wsub = u16::from_le_bytes([buf[8], buf[9]]);
                let iid = if n >= 18 {
                    u16::from_le_bytes([buf[16], buf[17]])
                } else {
                    0
                };
                let isub = if n >= 20 {
                    u16::from_le_bytes([buf[18], buf[19]])
                } else {
                    0
                };
                vino_debug!(
                    "vino: AKE-EP84 {n}B wsub={wsub:#x} inner_id={iid:#x} inner_sub={isub:#x}\n"
                );
            }
            if u16::from_le_bytes([buf[8], buf[9]]) != SUB_HDCP_RESP {
                continue; // non-HDCP frame -- skip
            }
            if let Some((id, payload)) = ake::parse_in(&buf[16..n]) {
                // Inner msg_id 0 is a status/ACK frame (the dock emits one as a
                // sub=0x25 frame after each OUT message, e.g. the `14 00 76 00...`
                // frame after AKE_Init) -- skip it and keep reading for the real
                // HDCP response, mirroring the oracle's recv_hdcp_msg.
                if id == 0 {
                    continue;
                }
                let mut pl = KVec::with_capacity(payload.len(), GFP_KERNEL)?;
                pl.extend_from_slice(payload, GFP_KERNEL)?;
                return Ok((id, pl));
            }
        }
        Err(EINVAL)
    }

    /// Drain one repeater-authentication acknowledgment before the next request.
    ///
    /// The dock enforces request/reply lockstep during this phase. The drain is bounded and
    /// best-effort so an idle dock cannot stall teardown.
    fn pace_cap_ack(link: &UsbLink<'_>, want_ctr: u16, saw_cap_complete: &mut bool) {
        // EP84 frames here can carry an interleaved capability block up to ~5.8 KiB;
        // size to [`EP84_BUF`] so a large frame isn't truncated mid-pacing.
        let Ok(mut buf) = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL) else {
            return;
        };
        for _ in 0..8 {
            match link.ctrl_recv(&mut buf, Delta::from_millis(30), GFP_KERNEL) {
                Ok(len) if len >= 22 => {
                    let wsub = u16::from_le_bytes([buf[8], buf[9]]);
                    let iid = u16::from_le_bytes([buf[16], buf[17]]);
                    let isub = u16::from_le_bytes([buf[18], buf[19]]);
                    let ictr = u16::from_le_bytes([buf[20], buf[21]]);
                    if iid == 0x0b && isub == 0x84 {
                        *saw_cap_complete = true;
                    }
                    // The per-frame cap-ack: wsub=0x25, inner id=0x14 sub=0x10 ctr=want.
                    // An interleaved cap push (sub=0x84) or earlier ack -- keep reading.
                    if wsub == 0x25 && iid == 0x14 && ictr == want_ctr {
                        return;
                    }
                }
                // A short frame (header-only ack/keepalive): not our cap-ack, but the
                // dock is still talking -- keep pacing rather than bailing out.
                Ok(_) => continue,
                // Nothing queued within the short window -- the dock is idle, don't block.
                Err(_) => return,
            }
        }
    }

    /// Drain the terminal capability burst before arming the encrypted control plane.
    ///
    /// The terminal markers are capability-complete (`id=0x0b sub=0x84`) and
    /// `RepeaterAuth_Stream_Ready`. If the latter is absent, a bounded quiet interval after the
    /// capability marker is accepted for firmware compatibility.
    fn wait_cap_complete(link: &UsbLink<'_>, kd: &[u8; 32], mut saw_0b: bool) {
        let Ok(mut buf) = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL) else {
            return;
        };
        // M' verification is diagnostic until every vendor stream-management field is decoded.
        let sha_kd = crypto::sha256(kd);
        let mut saw_ready = false;
        let mut quiet = 0usize;
        const QUIET_GAP: usize = 3; // ~3 consecutive empty short reads => dock done pushing
        const MAX_ROUNDS: usize = 48;
        for _ in 0..MAX_ROUNDS {
            match link.ctrl_recv(&mut buf, Delta::from_millis(5), GFP_KERNEL) {
                Ok(len) if len >= 20 => {
                    quiet = 0;
                    let iid = u16::from_le_bytes([buf[16], buf[17]]);
                    let isub = u16::from_le_bytes([buf[18], buf[19]]);
                    let mid = if len >= 26 { buf[25] } else { 0 }; // HDCP msg_id (body[9])
                    if isub == 0x84 && iid == 0x0b {
                        saw_0b = true;
                    }
                    if mid == ake::id::REPEATERAUTH_STREAM_READY && len >= 58 {
                        saw_ready = true;
                        let mprime = &buf[26..58];
                        vino_debug!("vino: AKE: Stream_Ready (0x11) M'={mprime:02x?}\n");
                        // The content-stream-management input contains two
                        // seven-byte stream entries and a three-byte sequence.
                        let m_data: [u8; 17] = [
                            0, 0, 0, 0x04, 0, 0, 0, // stream 0: StreamID_Type[0]
                            0, 0, 0, 0x05, 0, 0, 0, // stream 1: StreamID_Type[1]
                            0, 0, 0, // seq_num_M = 0 (first Stream_Manage, big-endian)
                        ];
                        let m = crypto::hmac_sha256(&sha_kd, &m_data);
                        let eq = if &m[..] == mprime { "==" } else { "!=" };
                        vino_debug!("vino: AKE:   M {} M' (CSM stream-entry layout)\n", eq);
                    } else if mid == ake::id::RECEIVER_AUTH_STATUS && len >= 27 {
                        vino_debug!("vino: AKE: RECEIVER_AUTH_STATUS=0x{:02x}\n", buf[26]);
                    }
                    // Both terminal markers complete the burst; do not add a quiet delay here.
                    if saw_0b && saw_ready {
                        vino_debug!("vino: repeater authentication complete\n");
                        return;
                    }
                }
                // Empty/short read = a quiet window. Fallback when Stream_Ready (0x11) never
                // arrives:
                // once id=0x0b has arrived AND the dock has been quiet for QUIET_GAP rounds, the
                // terminal burst is drained -- arm now.
                _ => {
                    if saw_0b {
                        quiet += 1;
                        if quiet >= QUIET_GAP {
                            vino_debug!("vino: repeater reply drained (ready={saw_ready})\n");
                            return;
                        }
                    }
                }
            }
        }
        vino_debug!("vino: repeater drain ended (complete={saw_0b}, ready={saw_ready})\n");
    }

    /// Run HDCP 2.2 AKE, locality check, session-key exchange and repeater authentication.
    ///
    /// `H'`, `L'` and `V'` are verified locally. Outbound messages use `type=4 sub=0x04` and the
    /// inner sequence is:
    ///
    /// * ctr=1 session-init ACK (id=0x14/0x76), ctr=2 AKE_Init, ctr=3 AKE_Transmitter_Info
    /// * ctr=4 AKE_No_Stored_km, ctr=5 LC_Init, ctr=6 SKE_Send_Eks
    /// * ctr=7 RepeaterAuth_Send_Ack, ctr=8 RepeaterAuth_Stream_Manage  (then msg0 at ctr=9)
    pub(super) fn run_ake(link: &UsbLink<'_>) -> Result<Session> {
        use ake::id;

        let mut saw_cap_complete = false;

        // A warm rebind can leave replies from the previous session queued on EP84.
        let flush_probe = Delta::from_millis(3);
        if let Ok(mut flush) = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL) {
            let mut flushed = 0usize;
            for _ in 0..32 {
                match link.ctrl_recv(&mut flush, flush_probe, GFP_KERNEL) {
                    Ok(n) if n > 0 => flushed += 1,
                    _ => break,
                }
            }
            if flushed > 0 {
                vino_debug!("vino: flushed {flushed} stale EP84 frame(s) before AKE\n");
            }
        }

        // The setup phase continues this counter through `Session::next_ctr`.
        let mut hseq: u32 = 1;

        // (1) session-init ACK (ctr=1, id=0x14/0x76).
        link.ctrl_send(&ake::session_init_ack(hseq, 0)?, timeout(), GFP_KERNEL)?;
        // The dock requires the counter-1 echo to be drained before AKE_Init.
        Self::pace_cap_ack(link, hseq as u16, &mut saw_cap_complete);
        hseq += 1;

        // (2) AKE_Init -- use a fresh rtx and the transmitter capability profile.
        let mut rtx = [0u8; drm_hdcp::RTX_LEN];
        rng::fill(&mut rtx);
        link.ctrl_send(
            &ake::ake_init(hseq, 0, &rtx, &[0; 3])?,
            timeout(),
            GFP_KERNEL,
        )?;
        hseq += 1;

        // (2) AKE_Send_Cert: payload = REPEATER(1) || cert_rx(522). Extract the
        // RSA-1024 public key (modulus[5..133], exponent[133..136]).
        let (cid, cert_msg) = Self::recv_hdcp(link)?;
        const CERT_KEY_END: usize = 5 + drm_hdcp::RSA_MODULUS_LEN + drm_hdcp::RSA_EXPONENT_LEN;
        if cid != id::AKE_SEND_CERT || cert_msg.len() < 1 + CERT_KEY_END {
            pr_err!(
                "vino: AKE: bad AKE_Send_Cert (id={cid:#x}, {} B)\n",
                cert_msg.len()
            );
            return Err(EINVAL);
        }
        let repeater = cert_msg[0] != 0;
        let cert = &cert_msg[1..];
        let mut modulus = [0u8; drm_hdcp::RSA_MODULUS_LEN];
        modulus.copy_from_slice(&cert[5..5 + drm_hdcp::RSA_MODULUS_LEN]);
        let mut exponent = [0u8; drm_hdcp::RSA_EXPONENT_LEN];
        exponent.copy_from_slice(&cert[5 + drm_hdcp::RSA_MODULUS_LEN..CERT_KEY_END]);

        // (3) AKE_Transmitter_Info (ctr=3), then read AKE_Receiver_Info (RxCaps unused).
        link.ctrl_send(&ake::ake_transmitter_info(hseq, 0)?, timeout(), GFP_KERNEL)?;
        hseq += 1;
        let _ = Self::recv_hdcp(link)?;

        // (5) AKE_No_Stored_km -- fresh km, RSA-OAEP-SHA256 to Ekpub(km).
        let mut km = kernel::crypto::Secret::<{ drm_hdcp::ENCRYPTED_SESSION_KEY_LEN }>::zeroed();
        rng::fill(&mut km[..]);
        let mut rsa = kernel::crypto::akcipher::RsaPublicKey::new(&modulus, &exponent, GFP_KERNEL)?;
        let ekpub = hdcp::oaep_encrypt_km(&mut rsa, &km)?;
        // (4) AKE_No_Stored_km (ctr=4). The dock authenticates its downstream link before it
        // answers, so the following receive naturally covers that interval.
        link.ctrl_send(
            &ake::ake_no_stored_km(hseq, 0, &ekpub)?,
            timeout(),
            GFP_KERNEL,
        )?;
        hseq += 1;

        // (6) AKE_Send_Rrx.
        let (rid, rrx_pl) = Self::recv_hdcp(link)?;
        if rid != id::AKE_SEND_RRX || rrx_pl.len() < drm_hdcp::RRX_LEN {
            pr_err!("vino: AKE: bad AKE_Send_Rrx (id={rid:#x})\n");
            return Err(EINVAL);
        }
        let mut rrx = [0u8; drm_hdcp::RRX_LEN];
        rrx.copy_from_slice(&rrx_pl[..drm_hdcp::RRX_LEN]);

        // (7)/(8) AKE_Send_H_prime -- verify H' = HMAC(kd, rtx^REPEATER).
        let (hid, hp) = Self::recv_hdcp(link)?;
        if hid != id::AKE_SEND_H_PRIME || hp.len() < drm_hdcp::H_PRIME_LEN {
            pr_err!("vino: AKE: bad H' (id={hid:#x})\n");
            return Err(EINVAL);
        }
        let kd = hdcp::derive_kd(&km, &rtx, &rrx)?;
        if hdcp::compute_h(&kd, &rtx, repeater)[..] != hp[..drm_hdcp::H_PRIME_LEN] {
            pr_err!("vino: AKE: H' mismatch -- authentication failed\n");
            return Err(EINVAL);
        }
        vino_debug!("vino: AKE: H' verified\n");

        // (9) AKE_Send_Pairing_Info (Ekh_km) -- read and discard (no-stored path).
        let _ = Self::recv_hdcp(link)?;

        // (10) Locality Check -- LC_Init(rn) then verify L'.
        let mut rn = [0u8; drm_hdcp::RN_LEN];
        rng::fill(&mut rn);
        // (5) LC_Init (ctr=5).
        link.ctrl_send(&ake::lc_init(hseq, 0, &rn)?, timeout(), GFP_KERNEL)?;
        hseq += 1;
        let (lid, lp) = Self::recv_hdcp(link)?;
        if lid != id::LC_SEND_L_PRIME || lp.len() < drm_hdcp::L_PRIME_LEN {
            pr_err!("vino: AKE: bad L' (id={lid:#x})\n");
            return Err(EINVAL);
        }
        if hdcp::compute_l(&kd, &rrx, &rn)[..] != lp[..drm_hdcp::L_PRIME_LEN] {
            pr_err!("vino: AKE: L' mismatch -- locality check failed\n");
            return Err(EINVAL);
        }
        vino_debug!("vino: AKE: L' verified\n");

        // (11) Session Key Exchange -- send Edkey(ske_ks) and the fresh RIV. The wrapped value is
        // the raw SKE key; both peers apply the control-plane whitening constant afterwards.
        let mut ske_ks =
            kernel::crypto::Secret::<{ drm_hdcp::ENCRYPTED_SESSION_KEY_LEN }>::zeroed();
        let mut riv = [0u8; drm_hdcp::RIV_LEN];
        rng::fill(&mut ske_ks[..]);
        rng::fill(&mut riv);
        let edkey = hdcp::compute_eks(&km, &rtx, &rrx, &rn, &ske_ks)?;
        let ks = cp::cp_session_key(&ske_ks);
        // SKE carries the full RIV. Control AES-CTR toggles byte 7 bit 2;
        // Dl3Cmac separately transforms byte 0 bit 7.
        let riv_ske = riv; // deliver the full random RIV before the control transform
        riv[7] ^= 0x04; // OUT CP AES-CTR nonce = delivered ^0x04@byte7 (byte0 UNCHANGED)
                        // (6) SKE_Send_Eks (ctr=6).
        link.ctrl_send(
            &ake::ske_send_eks(hseq, 0, &edkey, &riv_ske)?,
            timeout(),
            GFP_KERNEL,
        )?;
        hseq += 1;
        // (12) RepeaterAuth -- verify V' over the ReceiverID_List, ACK, then SM2. Retained (empty
        // on the non-repeater path) so `send_cp_setup`'s per-connector restatement can recompute a
        // fresh per-connector `V = HMAC(kd_h, rxid_list)` over the same list the dock sent.
        let mut rxid_list: KVec<u8> = KVec::new();
        if repeater {
            let (vid, list) = Self::recv_hdcp(link)?;
            if vid != id::REPEATERAUTH_SEND_RECEIVERID_LIST || list.len() < 16 {
                pr_err!("vino: AKE: bad ReceiverID_List (id={vid:#x})\n");
                return Err(EINVAL);
            }
            let split = list.len() - 16;
            // V' is the first 128 bits; RepeaterAuth_Send_Ack carries the second 128 bits.
            let v_full = hdcp::compute_v_full(&kd, &list[..split]);
            let mut v_ack = [0u8; 16];
            v_ack.copy_from_slice(&v_full[16..]);
            if v_full[..16] != list[split..] {
                pr_err!("vino: AKE: V' mismatch -- repeater verification failed\n");
                return Err(EINVAL);
            }
            vino_debug!("vino: AKE: V' verified\n");
            rxid_list.extend_from_slice(&list[..split], GFP_KERNEL)?;
            // (7) RepeaterAuth_Send_Ack (ctr=7).
            link.ctrl_send(
                &ake::repeater_auth_send_ack(hseq, 0, &v_ack)?,
                timeout(),
                GFP_KERNEL,
            )?;
            // Preserve repeater-authentication request/reply lockstep.
            Self::pace_cap_ack(link, hseq as u16, &mut saw_cap_complete);
            hseq += 1;
            // (8) RepeaterAuth_Stream_Manage (ctr=8).
            link.ctrl_send(
                &ake::repeater_auth_stream_manage(hseq, 0)?,
                timeout(),
                GFP_KERNEL,
            )?;
            Self::pace_cap_ack(link, hseq as u16, &mut saw_cap_complete);
            hseq += 1;
            // Drain capability-complete and Stream_Ready before arming the control plane.
            Self::wait_cap_complete(link, &kd, saw_cap_complete);
        }

        // `hseq` points past the last capability/AKE frame; `send_cp_setup` continues the inner
        // counter from here for msg0.
        if trace_crypto_enabled() {
            pr_info!(
                "vino-crypto: control key={:02x?} riv_out={riv:02x?} next_ctr={}\n",
                &ks[..],
                hseq
            );
        }
        Ok(Session {
            ks,
            riv,
            next_ctr: hseq as u16,
            rsa,
            rxid_list,
        })
    }

    /// Submit one encrypted control-plane frame without changing protocol counters on failure. Name
    /// a connector's content stream, on a dock where that is a control-plane record.
    ///
    /// A dock whose video shares the control pipe has no video endpoint to open a stream on later,
    /// so every connector it may ever drive has to be named during setup -- including one whose
    /// downstream authentication did not run, because a monitor plugged in afterwards comes up on
    /// exactly that connector and would then be driven on a stream the dock was never told about.
    /// The vendor names both connectors unconditionally. Docks with a video pipe of their own carry
    /// this ahead of the first frame instead, so they get nothing here.
    fn announce_stream(
        link: &UsbLink<'_>,
        out_q: &mut Option<usb::BulkOutQueue>,
        profile: &DockProfile,
        connector: usize,
    ) -> Result {
        if !profile.topology.video_on_ctrl_pipe {
            return Ok(());
        }
        let stream_id = profile.geometry().stream_id(connector as u8);
        Self::submit_cp_frame(
            link,
            out_q,
            &cp::stream_announce(stream_id, cp::STREAM_ANNOUNCE_MARKER),
        )
    }

    fn submit_cp_frame(
        link: &UsbLink<'_>,
        out_q: &mut Option<usb::BulkOutQueue>,
        frame: &[u8],
    ) -> Result {
        match out_q {
            // The queued path, which is the one both docks actually use. An error here is what
            // surfaces as `control session failed after N attempts (ETIMEDOUT)`; the 40-retry NAK
            // loop further down is the *unqueued* fallback and does not run, which is why
            // instrumenting it said nothing.
            Some(queue) => queue.send(link.io(), frame, timeout()).inspect_err(|e| {
                pr_err!(
                    "vino: EP02 queued submit of {} B failed ({e:?})\n",
                    frame.len()
                );
            }),
            None => link.ctrl_send(frame, timeout(), GFP_KERNEL).map(|_| ()),
        }
    }

    /// Seal and send one live interactive control message.
    ///
    /// The OUT is one long-lived URB, then EP84 is drained once after its completion. The returned
    /// tally distinguishes verified acknowledgments from rejected ciphertext.
    ///
    /// Do not implement a short-timeout retry here. A USB bulk timeout does not prove that no
    /// bytes reached the device: Navarro accepted several complete 64-byte frames just before
    /// Vino cancelled their 5 ms URBs. Re-submitting the same sealed frame consequently put
    /// duplicate inner counters on the wire (measured at counters 72, 75 and 95), while DLM sends
    /// each transaction once. Leaving one URB outstanding lets xHCI retry NRDY packets without
    /// replaying an already accepted application message.
    fn send_live_cp(
        link: &UsbLink<'_>,
        session: &Session,
        mut q: Option<&mut usb::BulkInQueue>,
        resp: &mut [u8],
        edid_out: &mut Option<KVec<u8>>,
        id: u16,
        wire_seq: u32,
        content: &[u8],
    ) -> Result<Ep84Drain> {
        let frame = cp::seal_interactive(&session.ks, &session.riv, id, wire_seq, content)?;

        let mut tally = Ep84Drain::default();
        link.ctrl_send(&frame, timeout(), GFP_KERNEL)?;
        // Collect the dock's reply, including a possible get-EDID id=0x194 frame.
        tally.add(Self::drain_ep84(
            link,
            q.as_deref_mut(),
            resp,
            session,
            edid_out,
            Delta::from_millis(10),
        ));
        Ok(tally)
    }
}
