// SPDX-License-Identifier: GPL-2.0

//! The DL3 "universal" wire framing and the plaintext session-init messages (sec 3/sec 4).

use super::*;

/// Append a sec 3-framed message to `out` with an explicit `sub_len_dw`: a 16-byte
/// little-endian header (`pad(2) | size(2)=total-4 | type(4) | sub_id(2) |
/// sub_len_dw(2) | seq(4)`) followed by `body`.
///
/// HDCP OUT messages (sec 5.1) carry DLM-fixed `sub_len_dw` values that are *not*
/// `body.len() / 4`, so the framer cannot derive it -- the caller passes it.
pub(super) fn push_frame_with(
    out: &mut KVec<u8>,
    msg_type: u32,
    sub_id: u16,
    sub_len_dw: u16,
    seq: u32,
    body: &[u8],
) -> Result {
    let size = ((16 + body.len()) - 4) as u16;
    out.extend_from_slice(&[0, 0], GFP_KERNEL)?;
    out.extend_from_slice(&size.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&msg_type.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&sub_id.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&sub_len_dw.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&seq.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(body, GFP_KERNEL)?;
    Ok(())
}

/// `init_25` body (sec 4, verified 2026-05-27). Framed with `sub_len_dw=0` -- the
/// DLM-fixed value, NOT `body.len()/4` (the dock ignores/rejects otherwise).
pub(super) const INIT_25: [u8; 16] = [0x05, 0, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// `init_4` (Part A) body (sec 4), also framed with `sub_len_dw=0`.
pub(super) const INIT_4: [u8; 16] = [0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// The first HDCP-channel probe **body** (Part B of the init_4+probe transfer,
/// sec 4): a 32-byte body leading with `14 00 90 00`, the rest zero. It is wrapped
/// in its own type=4 sub=0x04 frame (`sub_len_dw=0x0a`) -- see [`init_4_probe`].
/// The dock only ACKs once this framed probe arrives.
///
/// `p[2]` **CORRECTED 2026-07-07** (was `0x76`): a byte-exact diff of this
/// literal builder's output against a live DLM 3.4.26 libusb trace
/// (`captures/libusb-timing-comparison-20260707/dlm-libusb-trace.json`, decoded
/// with `vino-chimera`'s own code, not hand-parsed hex -- see memory
/// `project_dlm_hdcp_seq_offbyone_fixed_20260707`) found DLM's real COMBINED
/// init_4+probe write carries `0x90` here, not `0x76`. Confusingly, DLM's own
/// SEPARATE standalone resend of what looks like the same probe body (see
/// [`probe_resend`]) carries `0x76` at this position instead -- i.e. DLM's own
/// two transmissions of "the same" content disagree with each other at this
/// byte, strong evidence it's uninitialised/non-deterministic host memory
/// rather than a meaningful field the dock validates. Matched to the COMBINED
/// write specifically since that's what this constant serves; not proven
/// stable across DLM sessions.
pub(super) const PROBE_BODY: [u8; 32] = {
    let mut p = [0u8; 32];
    p[0] = 0x14;
    p[2] = 0x90;
    p
};

/// The body of DLM's SEPARATE standalone resend of the probe body, sent right
/// after the combined [`init_4_probe`] write (same wire framing: type=4
/// sub=0x04 `sub_len_dw=0x0a`, 48 bytes total) -- see [`probe_resend`]. Found
/// 2026-07-07 via the same live DLM 3.4.26 trace; matches [`PROBE_BODY`] at
/// `p[0]`/`p[2]` conceptually (`0x14`/`0x76`) but ALSO sets `p[4]=0x01`, which
/// `PROBE_BODY` does not. See `PROBE_BODY`'s doc comment for why this is
/// believed non-deterministic rather than meaningful.
pub(super) const PROBE_BODY_RESEND: [u8; 32] = {
    let mut p = [0u8; 32];
    p[0] = 0x14;
    p[2] = 0x76;
    p[4] = 0x01;
    p
};

/// `init_0`: 16-byte framing header only, empty body (sec 4).
pub(super) fn init_0() -> Result<KVec<u8>> {
    let mut buf = KVec::with_capacity(16, GFP_KERNEL)?;
    push_frame_with(&mut buf, 0x01, 0x00, 0, 0, &[])?;
    Ok(buf)
}

/// `init_25`: type=2 sub=0x25, `sub_len_dw=0`, 32 bytes total (sec 4).
pub(super) fn init_25() -> Result<KVec<u8>> {
    let mut buf = KVec::with_capacity(32, GFP_KERNEL)?;
    push_frame_with(&mut buf, 0x02, 0x25, 0, 0, &INIT_25)?;
    Ok(buf)
}

/// `init_4` + HDCP probe as one 80-byte transfer (sec 4): Part A (type=2 sub=0x04,
/// `sub_len_dw=0`, 32 B) concatenated with Part B -- the probe framed as type=4
/// sub=0x04 with `sub_len_dw=0x0a` over the 32-byte [`PROBE_BODY`] (48 B). This
/// is the message the dock ACKs.
pub(super) fn init_4_probe() -> Result<KVec<u8>> {
    let mut buf = KVec::with_capacity(80, GFP_KERNEL)?;
    push_frame_with(&mut buf, 0x02, 0x04, 0, 0, &INIT_4)?; // Part A
    push_frame_with(&mut buf, 0x04, 0x04, 0x0a, 0, &PROBE_BODY)?; // Part B (framed probe)
    Ok(buf)
}

/// DLM's observed SEPARATE standalone resend of the probe body (sec 4), sent as
/// its own 48-byte transfer immediately after [`init_4_probe`]'s combined 80-byte
/// write -- found 2026-07-07 via a live DLM 3.4.26 libusb trace (not previously
/// documented; `init_4_probe` alone was the whole story until then). Same wire
/// framing as Part B (type=4 sub=0x04 `sub_len_dw=0x0a`) over
/// [`PROBE_BODY_RESEND`]. See `PROBE_BODY`'s doc comment for the content caveat.
pub(super) fn probe_resend() -> Result<KVec<u8>> {
    let mut buf = KVec::with_capacity(48, GFP_KERNEL)?;
    push_frame_with(&mut buf, 0x04, 0x04, 0x0a, 0, &PROBE_BODY_RESEND)?;
    Ok(buf)
}
