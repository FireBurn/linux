// SPDX-License-Identifier: GPL-2.0

//! Reading the dock's interrupt endpoint.
//!
//! The dock answers on EP84 and also pushes unsolicited messages there. A reply left undrained
//! desynchronises the control plane, so the read paths here are what keep the session in lockstep
//! rather than merely convenient.

use super::*;

impl VinoDriver {
    /// Log one EP84 wire header and its decoded inner header when available.
    fn log_ep84(session: &Session, frame: &[u8]) {
        let len = frame.len();
        let wtype = if len >= 8 {
            u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]])
        } else {
            0
        };
        let wsub = if len >= 10 {
            u16::from_le_bytes([frame[8], frame[9]])
        } else {
            0
        };
        let aux = if len >= 12 {
            u16::from_le_bytes([frame[10], frame[11]])
        } else {
            0
        };
        let wseq = if len >= 16 {
            u32::from_le_bytes([frame[12], frame[13], frame[14], frame[15]])
        } else {
            0
        };
        {
            // Bound dynamic-debug output and split large frames below printk's line limit.
            let cap = len.min(768);
            if cap <= 64 {
                let raw = &frame[..cap];
                vino_debug!("vino: dock EP84 RAW {len}B {raw:02x?}\n");
            } else {
                vino_debug!("vino: dock EP84 RAW {len}B (first {cap} B in 128-B chunks):\n");
                let mut o = 0usize;
                while o < cap {
                    let e = (o + 128).min(cap);
                    let chunk = &frame[o..e];
                    vino_debug!("vino:   ep84[{o:#06x}] {chunk:02x?}\n");
                    o = e;
                }
            }
        }
        match cp::decode_any(&session.ks, &session.riv, frame) {
            Some((rivtag, rid, rsub, rictr, _)) => {
                vino_debug!("vino: EP84 {rivtag} id={rid:#x} sub={rsub:#x} ctr={rictr:#x}\n");
            }
            None => vino_debug!(
                "vino: EP84 type={wtype} sub={wsub:#x} aux={aux:#x} seq={wseq:#x} len={len}\n"
            ),
        }
    }
    /// Read one EP84 frame from the persistent queue or the synchronous fallback.
    pub(super) fn read_ep84(
        link: &UsbLink<'_>,
        q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        to: Delta,
    ) -> Result<usize> {
        match q {
            Some(queue) => match queue.recv(link.io(), buf, to) {
                Ok(Some(n)) => Ok(n),
                Ok(None) => Err(ETIMEDOUT),
                Err(e) => Err(e),
            },
            None => link.ctrl_recv(buf, to, GFP_KERNEL),
        }
    }
    pub(super) fn drain_ep84(
        link: &UsbLink<'_>,
        mut q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        session: &Session,
        edid_out: &mut Option<KVec<u8>>,
        // The first read may cover an HDCP computation interval; subsequent reads only drain a
        // contiguous reply burst.
        first_wait: Delta,
    ) -> Ep84Drain {
        const MAX_READS: usize = 16;
        let mut out = Ep84Drain::default();
        // Read EP84 before doing any unrelated work: the control plane is lockstep.
        for i in 0..MAX_READS {
            let wait = if i == 0 {
                first_wait
            } else {
                Delta::from_millis(10)
            };
            match Self::read_ep84(link, q.as_deref_mut(), buf, wait) {
                Ok(len) if len > 0 => {
                    out.reads += 1;
                    Self::log_ep84(session, &buf[..len]);
                    // Decode every downstream-HDCP milestone, not just Rrx.  A generic encrypted
                    // envelope acknowledgment does not mean the downstream authentication has
                    // advanced; L', V' and M' are the state-machine gates.
                    if let Some(push) =
                        cp::per_connector_hdcp_push(&session.ks, &session.riv, &buf[..len])
                    {
                        out.observe_perhead(push);
                    }
                    if len >= 10 && u16::from_le_bytes([buf[8], buf[9]]) == 0x45 {
                        // The 0x45 wire tag is shared by status traffic. Only a valid decrypted
                        // inner header proves that this session's cipher is engaged.
                        match cp::verify_in_ack(&session.ks, &session.riv, &buf[..len]) {
                            Some((id, sub, ctr)) => {
                                out.acks += 1;
                                vino_debug!(
                                    "vino: CP acknowledgment id={id:#x} sub={sub:#x} ctr={ctr}\n"
                                );
                                // A display-capability reply identifies a
                                // present monitor and echoes the request counter.
                                if cp::is_display_cap_reply(id, sub) {
                                    out.display_cap_ctr = Some(ctr);
                                }
                                // Capture the first `id=0x194 sub=0x21` EDID
                                // reply for the standard DRM mode helpers.
                                if edid_out.is_none() {
                                    if let Ok(Some(e)) = cp::parse_edid_from_reply(
                                        &session.ks,
                                        &session.riv,
                                        &buf[..len],
                                    ) {
                                        vino_debug!(
                                            "vino: EDID read from dock ({} bytes)\n",
                                            e.len()
                                        );
                                        *edid_out = Some(e);
                                    }
                                }
                                // Track the downstream-DDC readiness bit so
                                // the EDID loop can distinguish pending work.
                                if let Some(true) =
                                    cp::edid_poll_ready(&session.ks, &session.riv, &buf[..len])
                                {
                                    out.edid_ready = true;
                                }
                            }
                            None => {
                                match cp::decode_in_lenient(&session.ks, &session.riv, &buf[..len])
                                {
                                    // A structurally valid header with an uncatalogued sub-id still
                                    // proves possession of the session key.
                                    Some((id, sub, ctr)) => {
                                        out.acks += 1;
                                        vino_debug!(
                                            "vino: CP reply id={id:#x} sub={sub:#x} ctr={ctr}\n"
                                        );
                                        if sub == 0x000c {
                                            if let Some(inner) = cp::inner_plaintext(
                                                &session.ks,
                                                &session.riv,
                                                &buf[..len],
                                            ) {
                                                if let Some(line) = cp::dock_trace_line(&inner) {
                                                    vino_debug!(
                                                        "vino: dock: {}\n",
                                                        core::str::from_utf8(&line).unwrap_or("?")
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    // Not a sealed reply. Navarro pushes plaintext-framed
                                    // messages the dock originates rather than answers, so a
                                    // frame vino cannot open is only a rejection once that
                                    // framing has been ruled out too.
                                    None => match cp::inner_plaintext(
                                        &session.ks,
                                        &session.riv,
                                        &buf[..len],
                                    ) {
                                        Some(inner) => {
                                            out.acks += 1;
                                            if let Some(line) = cp::dock_trace_line(&inner) {
                                                vino_debug!(
                                                    "vino: dock: {}\n",
                                                    core::str::from_utf8(&line).unwrap_or("?")
                                                );
                                            }
                                        }
                                        None => {
                                            out.rejects += 1;
                                            pr_warn!("vino: invalid encrypted CP reply\n");
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
                _ => break,
            }
        }
        out
    }
    /// Read through the downstream-HDCP reply stream until one specific protocol milestone.
    ///
    /// Navarro acknowledges every encrypted envelope before it emits the corresponding HDCP
    /// result.  Advancing on the generic ACK alone allowed Vino to send SKE/V-ACK/Stream-Manage
    /// without ever receiving L', V' or M'.  The dock accepted the envelopes but never enabled
    /// the video consumer.  This routine makes the HDCP result, rather than the wrapper ACK, the
    /// sequencing condition.
    pub(super) fn wait_per_connector_push(
        link: &UsbLink<'_>,
        mut q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        session: &Session,
        want_msg_id: u8,
        wait: Delta,
    ) -> Ep84Drain {
        const MAX_READS: usize = 16;
        let mut out = Ep84Drain::default();
        for _ in 0..MAX_READS {
            match Self::read_ep84(link, q.as_deref_mut(), buf, wait) {
                Ok(len) if len > 16 => {
                    out.reads += 1;
                    Self::log_ep84(session, &buf[..len]);
                    if let Some(push) =
                        cp::per_connector_hdcp_push(&session.ks, &session.riv, &buf[..len])
                    {
                        out.observe_perhead(push);
                        if out.saw_perhead(want_msg_id) {
                            return out;
                        }
                        continue;
                    }
                    if u16::from_le_bytes([buf[8], buf[9]]) != 0x45 {
                        continue;
                    }
                    match cp::verify_in_ack(&session.ks, &session.riv, &buf[..len]) {
                        Some((id, sub, ctr)) => {
                            out.acks += 1;
                            vino_debug!(
                                "vino: CP acknowledgment while waiting for HDCP {want_msg_id:#x}: id={id:#x} sub={sub:#x} ctr={ctr}\n"
                            );
                        }
                        None if cp::decode_in_lenient(&session.ks, &session.riv, &buf[..len])
                            .is_some() =>
                        {
                            out.acks += 1;
                        }
                        None => {
                            out.rejects += 1;
                            pr_warn!("vino: invalid encrypted CP reply\n");
                        }
                    }
                }
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        out
    }
    /// Drain replies until a verified inner counter echoes the submitted request.
    ///
    /// Asynchronous pushes are processed while waiting, and the operation remains bounded.
    pub(super) fn lockstep_reply(
        link: &UsbLink<'_>,
        mut q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        session: &Session,
        ictr: u16,
        edid_out: &mut Option<KVec<u8>>,
    ) -> Ep84Drain {
        const MAX_READS: usize = 8;
        let mut out = Ep84Drain::default();
        for _ in 0..MAX_READS {
            match Self::read_ep84(link, q.as_deref_mut(), buf, Delta::from_millis(30)) {
                Ok(len) if len > 16 => {
                    out.reads += 1;
                    Self::log_ep84(session, &buf[..len]);
                    if u16::from_le_bytes([buf[8], buf[9]]) != 0x45 {
                        continue;
                    }
                    if let Some(push) =
                        cp::per_connector_hdcp_push(&session.ks, &session.riv, &buf[..len])
                    {
                        out.observe_perhead(push);
                    }
                    match cp::verify_in_ack(&session.ks, &session.riv, &buf[..len]) {
                        Some((id, sub, ctr)) => {
                            out.acks += 1;
                            let echo = if ctr == ictr {
                                " (echoes our ictr)"
                            } else {
                                ""
                            };
                            vino_debug!("vino: CP reply id={id:#x} sub={sub:#x} ctr={ctr}{echo}\n");
                            // Opportunistically extract an EDID from an id=0x194 reply.
                            if edid_out.is_none() {
                                if let Ok(Some(e)) = cp::parse_edid_from_reply(
                                    &session.ks,
                                    &session.riv,
                                    &buf[..len],
                                ) {
                                    vino_debug!("vino: EDID read from dock ({} bytes)\n", e.len());
                                    *edid_out = Some(e);
                                }
                            }
                            // Stop early once the dock acknowledges the counter we sent.
                            if ctr == ictr {
                                break;
                            }
                        }
                        None => match cp::decode_in_lenient(&session.ks, &session.riv, &buf[..len])
                        {
                            // Decrypts to a plausible CP header, just an unlisted `sub` -- a valid
                            // ack (cipher engaged), not a rejection. See the drain_ep84 branch.
                            Some((id, sub, ctr)) => {
                                out.acks += 1;
                                vino_debug!("vino: CP reply id={id:#x} sub={sub:#x} ctr={ctr}\n");
                            }
                            None => {
                                out.rejects += 1;
                                pr_warn!("vino: invalid encrypted CP reply\n");
                            }
                        },
                    }
                }
                // A short, header-only frame (bare ack/keepalive, len <= 16): not a CP
                // reply, but the dock is still talking -- keep reading for the 0x45 rather
                // than dropping the rest of the lockstep window.
                Ok(_) => continue,
                // Read error / nothing queued within the window: the dock is idle, stop.
                Err(_) => break,
            }
        }
        out
    }
}
