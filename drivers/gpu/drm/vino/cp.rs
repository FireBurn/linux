// SPDX-License-Identifier: GPL-2.0
//! Encrypted-control-plane message builders (the inner plaintext of the type=4
//! sub=0x24 AES-CTR frames) plus the AES-CTR `seal` that encrypts and frames them.
//! Layouts are from the reverse-engineered protocol; offsets cite the guide and
//! should be re-checked against a capture before they drive real hardware.
#![allow(dead_code)] // some seal/handler paths run only after the dock engages CP (open blocker)
use super::*;

/// **CRACKED 2026-07-16 (rr reverse-execution of DLM's live seal derivation).**
/// The CP session key that keys BOTH the AES-CTR content cipher and the Dl3Cmac is NOT the raw
/// SKE key the dock unwraps from `Edkey` -- it is that key XORed with a fixed DisplayLink device
/// constant [`CP_KEY_WHITEN`]:
/// ```text
///   cp_session_key = ske_ks XOR CP_KEY_WHITEN
/// ```
/// DLM picks a random `ske_ks`, wraps it in `Edkey` (the dock unwraps it), then keys CP with
/// `ske_ks XOR CP_KEY_WHITEN`. Both sides bake in the same constant. vino had keyed CP with the raw
/// random `ks`, so the dock -- deriving `ks XOR const` -- failed every OUT Dl3Cmac and silently
/// dropped the CP stream after the arm (0x sub=0x45), which was the whole engagement wall.
///
/// Ground truth: the derivation is a single 16-byte `PXOR` (DLM+0x1d118e) of the SKE random `A`
/// (a per-session CTR_DRBG draw, wrapped in Edkey, never used directly) with this constant `B`.
/// `B` was byte-identical across two independent cold recordings and is NOT embedded in the DLM
/// binary (computed at runtime) -- it is a device-global constant shared by host and dock firmware.
/// See memory `project_cp_seal_key_is_derived_not_random_20260715` (rounds 31-35) and the reproducible
/// rr harness in `scripts/rr-*`.
pub(super) const CP_KEY_WHITEN: [u8; 16] = [
    0x26, 0xab, 0xee, 0x38, 0x93, 0xd0, 0xc4, 0x32, 0x61, 0x43, 0xa4, 0xbf, 0x5b, 0x45, 0xd6, 0xec,
];

/// Derive the live CP session key from the raw SKE key: `ske_ks XOR `[`CP_KEY_WHITEN`]. The result
/// keys the AES-CTR content cipher and the Dl3Cmac in [`seal_livemac`]; `ske_ks` (the input) is what
/// gets wrapped into `Edkey` and delivered to the dock, which applies the same XOR on its side.
pub(super) fn cp_session_key(ske_ks: &[u8; 16]) -> [u8; 16] {
    let mut key = *ske_ks;
    for i in 0..16 {
        key[i] ^= CP_KEY_WHITEN[i];
    }
    key
}

/// Derive the AES-CTR content nonce for one video head from the RIV carried by that head's
/// `SKE_Send_Eks` restatement (`id=0x32`).  This is *not* the main CP channel's `byte7 ^= 0x04`
/// transform: a same-session rr correlation of the clear `id=0x32` plaintext, the low-frequency
/// video-key setter, and the AES counter blocks proved `byte7 ^= 0x08 | head`:
///
/// - head 0: delivered `a12baab70e0b0274` -> EP08 CTR `a12baab70e0b027c`;
/// - head 1: delivered `d02ac083b6427257` -> EP0b CTR `d02ac083b642725e`.
///
/// The Dl3Cmac layer subsequently applies its independent `byte0 ^= 0x80` transform.
pub(super) fn video_content_nonce(riv: &[u8; 8], head: u8) -> [u8; 8] {
    let mut nonce = *riv;
    nonce[7] ^= 0x08 | head;
    nonce
}

/// Common CP inner header: `[id u16][sub u16][counter u16][00 00]` (sec 6.1/sec 8.6.4).
fn header(out: &mut KVec<u8>, id: u16, sub: u16, counter: u16) -> Result {
    out.extend_from_slice(&id.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&sub.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&counter.to_le_bytes(), GFP_KERNEL)?;
    out.extend_from_slice(&[0, 0], GFP_KERNEL)?;
    Ok(())
}
fn pad_to(out: &mut KVec<u8>, len: usize) -> Result {
    while out.len() < len {
        out.push(0, GFP_KERNEL)?;
    }
    Ok(())
}
/// OUT session heartbeat: `id=0x16 sub=0x75`, two AES blocks.
///
/// **Corrected 2026-07-22** against the two decrypted DLM captures that contain one
/// (`captures/dlm-wake-ab-20260722-150209/cp-decrypted.json`, 5 messages, and
/// `captures/max-cold-20260721-235609/cp-decrypted.json`, 8 messages). All 13 are byte-identical
/// in shape:
///
/// ```text
/// 16 00 75 00 [ctr:2] 00 00   14x 00   e0 2e   [8-byte host-random token]
/// ```
///
/// The 16-bit value at off22 is `0x2ee0` (12000) -- confirmed on every live heartbeat plaintext
/// (2026-07-23 Frida trace), not the `0x2710` (10000) this built before.
///
/// **off24..32 is STALE SCRATCH-BUFFER MEMORY, settled 2026-07-23 by the same live method that
/// settled the cold video-arm tail (`docs/VIDEO-ARM-RANDOM-FIELDS.md`).** Hooking DLM's mbedTLS
/// CTR-DRBG output and its pre-encryption CP plaintext in one live session
/// (`vino-re/frida/hook-cp-token.js`, capture `captures/cp-token-coldplug-20260723-*.ndjson`)
/// showed: (1) **zero** CTR-DRBG draws within 200 ms of any of the captured heartbeats -- the only
/// 8-byte draws in the whole session were the AKE session material during a DLM restart, none
/// equal to a heartbeat tail; and (2) the tails **repeat** (counter 366 and 382 both
/// `00000002257c5c6c`) and carry fragments of other CP messages (`0000000aa0003000` embeds the
/// mode-set word `a0 00 30 00`; the `00000002/04/09/0a` leaders are the BE32 head/index fields of
/// the CP setup/restatement burst). DLM assembles the 32-byte heartbeat in a recycled buffer,
/// writes through off22 (`e0 2e`), and leaves off24..32 as whatever the previous message left. It
/// is neither random nor derived; the dock ignores it (vino has had this field zero, `10 27` and
/// CSPRNG-random accepted). We therefore emit zeros: deterministic, no RNG draw, and honest about
/// the field being don't-care rather than dressing it up as a token.
///
/// DLM sends this every **3.000 s for the whole streaming session** (measured interval across
/// both captures: 3.000-3.001 s), not just during bring-up. vino previously sent exactly one, from
/// inside the EDID loop, and then went heartbeat-silent -- see `BringUp::run`'s keepalive.
pub(super) fn heartbeat(counter: u16) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x16, 0x75, counter)?;
    pad_to(&mut b, 22)?; // block0 tail + block1[0..6]
    b.extend_from_slice(&[0xe0, 0x2e], GFP_KERNEL)?; // off22: 0x2ee0 (live-confirmed)
    pad_to(&mut b, 32)?; // off24..32: proven don't-care stale-buffer bytes -- emit zeros
    Ok(b)
}
/// Post-video-arm-burst "stream commit" (recovered 2026-07-17 by re-mining the SAME rr recording
/// used for `cp::VIDEO_ARM_BURST`, this time past where earlier mining passes stopped -- see
/// `project_post_burst_stream_commit_found_20260717` memory): `id=0x16 sub=0x4c`, DLM sends this
/// FOUR times (twice per head) on the MAIN EP02 channel immediately after its video-endpoint
/// arm-burst, before real video starts -- a step vino's own burst never replicated. Same
/// `[id][sub][counter][00 00]` header as `heartbeat`, 14 zero bytes, then a per-head flag byte at
/// offset 22 (`0x00` for head 0, `0x01` for head 1 -- confirmed against 2 messages each), a zero
/// at offset 23, then a fresh 8-byte host-random trailing token (same "dock doesn't validate it"
/// shape as `msg0`/`get_edid_req`). Distinct from the 16-byte-content `id=0x16 sub=0x4c` entries
/// in `CP_SETUP_FINALIZE` (same id/sub, different context/length -- this is NOT a duplicate).
/// Stream/display **enable markers** that DLM brackets every mode-set with: `id=0x16 sub=0x2e` and
/// `sub=0x2f`. Decoded 2026-07-20 from the arm-window capture
/// (`captures/archive-pre-3426/ep08-framing-20260628/armburst-modeset-20260630.json`), whose ordered
/// sequence per head is:
///   `2f(1) 2e(3)` -> **mode-set** -> `2f(1) 2e(0) 2f(1) 2e(0) 2f(0) 2e(0)`
/// (4x each per head, 8x each across the D6000's two heads -- matching the capture's counts).
/// Same 32-byte frame as [`stream_commit`]/[`heartbeat`]: 8-byte header, zeros to off22, the
/// downstream head selector at off22, the state byte at off23, then an 8-byte host-random token.
/// A dual-head DLM trace proves the first bracket uses off22=0 and the second uses off22=1; the
/// earlier "constant 1" conclusion came from a capture that changed head 1 only.
///
/// vino never sent these, which is the leading explanation for the dock ACCEPTING EP08 video
/// (`frame ok`, no ESHUTDOWN) while the downstream monitor never lit up: the stream is written but
/// never enabled. `sub` must be 0x2e or 0x2f; `state` is the off23 value from the sequence above.
pub(super) fn stream_marker(counter: u16, head: u8, sub: u16, state: u8) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x16, sub, counter)?;
    pad_to(&mut b, 22)?;
    b.push(head, GFP_KERNEL)?; // off22: downstream head selector
    b.push(state, GFP_KERNEL)?; // off23: state byte
    let mut token = [0u8; 8];
    rng::fill(&mut token);
    b.extend_from_slice(&token, GFP_KERNEL)?;
    Ok(b)
}

pub(super) fn stream_commit(counter: u16, head: u8) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x16, 0x4c, counter)?;
    pad_to(&mut b, 22)?;
    b.push(if head == 0 { 0 } else { 1 }, GFP_KERNEL)?; // off22: per-head flag
    b.push(0, GFP_KERNEL)?; // off23
    let mut token = [0u8; 8];
    rng::fill(&mut token);
    b.extend_from_slice(&token, GFP_KERNEL)?;
    Ok(b)
}
/// OUT device-status/capability query: `id=0x14`, sub-selectable. **Found 2026-07-17** by
/// decrypting DLM's raw (not just flagged-"interesting") OUT/IN sequence around both of its real
/// EDID successes (`captures/rr-out-sequence-20260716/full-session-trace1/raw-out-in-192msg.txt`)
/// -- a message class vino never sent at all before this. Same 32-byte shape as `get_edid_req`
/// (8B header + 14 zero + 10B host-random tail).
///
/// `sub=0x0000`: sent by DLM exactly ONCE, immediately after the CP-setup finalize burst and
/// before it starts probing for EDID at all (ictr 39 in the trace, right after a heartbeat and
/// right before the first `sub=0x0020` probe at ictr 40). The dock's reply is `id=0x4c sub=0x0`,
/// an 80-byte structured blob (readable-ish device/firmware info, e.g. a `0c 02 19` firmware-
/// version-shaped triplet at a fixed offset) -- not a generic ack. Read as a one-shot "who are
/// you" / device-capability query that establishes some dock-side state before display-related
/// queries are entertained.
///
/// `sub=0x000c`: sent repeatedly by DLM throughout steady state (seen ictr 115-118, 123 in the
/// same trace, clustered around -- but not exclusively tied to -- its SECOND EDID success). The
/// dock's reply is EITHER a generic `id=0x14 sub=0xc` ack (nothing new) OR, when it has new data,
/// an `id` that varies per call (`0x5e`/`0x49`/`0x82` seen) carrying ASCII-ish `|`-delimited
/// device debug-log lines. Looks like an incremental device-log/diagnostic-buffer poll, not
/// EDID-specific -- included here for completeness and because vino currently sends nothing
/// resembling background device-status traffic at all.
pub(super) fn device_query_req(counter: u16, sub: u16) -> Result<KVec<u8>> {
    random_tail_msg(0x14, sub, counter)
}
/// Shared builder for the many CP messages that share one wire shape: the standard 8-byte
/// `[id][sub][counter][00 00]` header, 14 zero bytes, then a fresh 10-byte host-random tail the
/// dock does not validate (see [`get_edid_req`]'s doc comment for the evidence this tail is
/// session noise, not content the dock checks).
fn random_tail_msg(id: u16, sub: u16, counter: u16) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, id, sub, counter)?;
    pad_to(&mut b, 22)?;
    let mut tail = [0u8; 10];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}
/// OUT `id=0x16 sub=0x0023` EDID-engage. **Found 2026-07-17 (2nd pass)**, re-decoding the SAME
/// full-session trace used for [`device_query_req`]/[`edid_readiness_kick`] but this time
/// tracing every message between the trace's two placeholder `id=0x114` fetches (ictr 44, 47)
/// and its first real `id=0x194` fetch (ictr 120) instead of stopping at the first fetch retry.
/// DLM sends this TWICE (ictr 48, 49), back-to-back, right after the SECOND placeholder fetch
/// and immediately before it starts the long [`device_query_req`]`(.., 0x000c)` status-poll run
/// that precedes the real EDID becoming available -- a message class vino never sent before this.
/// Same 32-byte shape as [`get_edid_req`] (host-random tail), but the head selector is TWO bytes,
/// not one -- see below. The dock's CP reply is a plain generic ack (`id=0x14 sub=0x0023`, no
/// payload) whether or not the message is well-formed, which is exactly why the malformed version
/// survived so long: the ack says nothing.
///
/// **byte23 corrected 2026-07-26.** The dock's own firmware trace (`scripts/dock-trace-live.py`)
/// shows this message's handler as `388cf`, and DLM's cold plug continues
/// `388cf / 27525 <h> <h> / 2a887 <h> <h> / 3a7b3 <h> 0 0` -- the downstream sink enable. vino got
/// `388cf / 30c5a <byte23>` instead: a bad-parameter reject that echoes the offending byte. vino
/// filled byte23 with the random tail, so the dock never enabled the downstream sink, its
/// `2bf21 <h>` state machine ran 0 -> 2 -> **4** instead of 0 -> 2 -> **3**, and `2807a` (hence the
/// pixel clock `2990d`) never fired on a link vino had to train itself. A warm handback from a lit
/// DLM hid the bug: DLM had already run the successful engage.
///
/// DLM cold capture, the only two `id=0x16 sub=0x23` messages in the session:
/// ```text
/// head 0: ...  off22 = 00  off23 = 00   then dock: 275250 0 0 / 2a8870 0 0
/// head 1: ...  off22 = 01  off23 = 01   then dock: 275251 1 1 / 2a8871 1 1
/// ```
/// The dock echoes both bytes as the two arguments of `27525`/`2a887`, so both are the head.
pub(super) fn edid_engage_req(counter: u16, head: u8) -> Result<KVec<u8>> {
    // byte22 AND byte23 = head selector (the dock reads both; a random byte23 is rejected with
    // `30c5a <byte23>` and the downstream sink is never enabled); byte24.. host-random.
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x16, 0x0023, counter)?;
    pad_to(&mut b, 22)?;
    b.push(head, GFP_KERNEL)?;
    b.push(head, GFP_KERNEL)?;
    let mut tail = [0u8; 8];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}
/// OUT `id=0x15 sub=0x0053` post-EDID capability query. **Found 2026-07-23** by inventorying the
/// decrypted COLD capture (`captures/edid-cold-decrypt-20260719/`) rather than the warm ones --
/// this type does not occur in either warm capture at all, which is why every previous comparison
/// missed it. DLM sends it TWICE, late in the cold sequence (block-seq 1289 and 1293, after the
/// real `id=0x194` EDID has arrived), with an index that runs 1 then 2 (read as `head + 1`,
/// matching every other per-head selector's shape) and a host-random tail.
///
/// **Offset corrected 2026-07-26:** the index sits at **off22**, the same slot every other per-head
/// selector uses -- not off23. The original reading came from a hand-counted hex dump; re-derived
/// from `captures/dlm-coldplug-withmon-160457` with a decoder, DLM's two messages are
/// `..off22=01, off23=d9(random)..` and `..off22=02, off23=2d(random)..`. vino was sending
/// `off22=00, off23=head+1`, i.e. the index one byte late and a zero where the selector belongs.
///
/// The dock's reply is a structured `id=0x1c sub=0x53` carrying `07 10 10 00 01 00 01 00` at
/// off24 in both cases -- a fixed capability/word block, not an echo, so this is a real query
/// whose answer the dock has ready only once the downstream EDID read has completed.
pub(super) fn post_edid_query(counter: u16, head: u8) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x15, 0x0053, counter)?;
    pad_to(&mut b, 22)?;
    b.push(head + 1, GFP_KERNEL)?; // off22: 1-based call/head index (DLM: 1 then 2)
    let mut tail = [0u8; 9];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}
/// OUT `id=0x16 sub=0x004b` "kick". **Found 2026-07-17**, same mining pass as
/// [`device_query_req`]: sent once by DLM partway through its `sub=0x0020` EDID-readiness polling
/// (ictr 42, between two `sub=0x0020` polls at ictr 41 and 43). The dock's reply is a plain
/// generic ack (`id=0x14 sub=0x4b`, no payload) -- but the VERY NEXT `sub=0x0020` poll reply
/// (ictr 43) is visibly richer/more-filled-in than the two polls before the kick (see
/// `edid_poll_progress`), and the fetch right after that succeeds. Read as "start/continue the
/// downstream DDC-EDID read", with its effect only observable via the next status poll, not its
/// own reply. Same 32-byte shape as `heartbeat`/`stream_commit` (8B header + 14 zero + 2-byte
/// tail `[0x10, 0x27]` + 8 zero, matching `heartbeat`'s block-1 tail byte-for-byte -- unverified
/// whether that specific tail matters here or is incidental; kept for a byte-exact replication).
pub(super) fn edid_readiness_kick(counter: u16, head: u8) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x16, 0x4b, counter)?;
    pad_to(&mut b, 22)?;
    // CORRECTED 2026-07-19 from the fully-decrypted cold EDID fetch
    // (`captures/edid-cold-decrypt-20260719/`): DLM's real kick body is `00 01` @off22-23 then an
    // 8-byte host-random tail. The old `[0x10, 0x27]` was the HEARTBEAT marker copied by mistake --
    // that malformed kick is why the dock REJECTED it and the kick was removed from the sequence.
    // 2026-07-20: off22 is the head selector (0 in the single-monitor capture); kick the head's DDC.
    b.extend_from_slice(&[head, 0x01], GFP_KERNEL)?;
    let mut tail = [0u8; 8];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}
/// Decode an `id=0x0044 sub=0x0020` EDID-readiness poll reply (the real response to
/// [`get_edid_req_sub`]'s `sub=0x20` probe -- NOT a generic `id=0x14` ack, see that function's doc
/// comment) and return a coarse "how filled in is it" progress score: the count of non-zero bytes
/// in the reply body. **Found 2026-07-17** from the same trace: DLM's own two probes before its
/// `edid_readiness_kick` return a body with only ~3 non-zero bytes; the probe right after the kick
/// (and every later one, cross-session) returns a body with the same 3 bytes PLUS a distinct
/// 8-byte tail block filled in (~11 non-zero bytes) -- immediately followed by a successful
/// `sub=0x0021` fetch both times this was observed. The exact bit meaning is not decoded; this is
/// a threshold heuristic, not a real parse.
pub(super) fn edid_poll_progress(wire: &[u8], ks: &[u8; 16], out_riv: &[u8; 8]) -> Option<usize> {
    if wire.len() <= 16 || u16::from_le_bytes([wire[8], wire[9]]) != 0x45 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let inner = open_in(ks, &in_riv(out_riv), seq, &wire[16..]).ok()?;
    if inner.len() < 8 {
        return None;
    }
    let id = u16::from_le_bytes([inner[0], inner[1]]);
    let sub = u16::from_le_bytes([inner[2], inner[3]]);
    if id != 0x44 || sub != 0x20 {
        return None;
    }
    Some(inner[8..].iter().filter(|&&b| b != 0).count())
}
/// OUT get-EDID request (CP-HANDSHAKE.md sec 4f): `id=0x15 sub=0x21`, the message that asks
/// the dock to return the downstream monitor's EDID. **Ground-truthed 2026-07-16** by mining a
/// full DLM session (rr replay of `rr-traces/`, `mineinout.py`, no reverse-execution needed) far
/// past where earlier sessions stopped: DLM's real request is **32 bytes** -- the standard 8-byte
/// `[id][sub][counter][00 00]` header, 14 zero bytes, then a fresh 10-byte host-random tail at
/// offset 22 (the same shape as `msg0` and the `CP_SETUP_PER_HEAD` messages) -- NOT the bare
/// 16-byte header this function used to emit. That mismatch is the likely reason vino's own
/// get-EDID attempts only ever got a generic ack instead of the dock's real
/// `id=0x194 sub=0x21` EDID push (see `captures/rr-out-sequence-20260716/full-session-trace1/`).
/// DLM also does not get the real EDID on its FIRST attempt either -- its early requests
/// (inner ctr 44, 47) get back a dock-internal placeholder (`id=0x114`, a generic "NOVATEK"
/// descriptor, not the monitor's), and only a later retry (ctr 120+, well into steady-state
/// heartbeat traffic) gets the real `id=0x194` reply with the monitor's actual EDID. So the
/// caller should retry this request rather than give up after one attempt; see
/// `VinoDriver::send_cp_setup`'s get-EDID retry loop.
///
/// **2026-07-17: a `sub=0x0020` companion request precedes every successful `sub=0x0021` pull
/// in the same trace.** Re-examining the raw OUT sequence (not just the flagged "interesting"
/// lines) around both real EDID successes shows DLM sends one or more `id=0x15 sub=0x0020`
/// requests immediately before each `sub=0x0021` that actually gets an EDID-shaped reply back
/// (ictr 40/41/43 → the 44 success; ictr 119 → the 120 success; ictr 121 → the 122 success) --
/// every occurrence, not a coincidence of adjacent counters. `docs/CONTROL-PLANE.md`'s older
/// note calling `sub=0x0020` "a different query" was written from a single matched-session
/// example that never actually tested omitting it. Use [`get_edid_req_sub`] with `sub=0x20` to
/// send this probe/seek request immediately before each `sub=0x21` pull attempt -- same 32-byte
/// wire shape (see `VinoDriver::send_cp_setup`).
pub(super) fn get_edid_req(counter: u16, head: u8) -> Result<KVec<u8>> {
    get_edid_req_sub(counter, 0x21, head)
}
/// Build an `id=0x15` EDID-family request with an explicit `sub` (`0x20` = probe/seek,
/// `0x21` = fetch -- see [`get_edid_req`]'s doc comment). Same 32-byte wire shape for both.
pub(super) fn get_edid_req_sub(counter: u16, sub: u16, head: u8) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x15, sub, counter)?;
    pad_to(&mut b, 22)?;
    // byte22 is the downstream-PORT / HEAD SELECTOR in DLM's id=0x15 EDID probes/fetches (decoded
    // 2026-07-20: DLM alternates 0,1 across consecutive messages -- it probes head 0 then head 1;
    // the real EDID came back on the byte22=0 fetch, the head with the monitor). An earlier pass
    // wrongly dismissed off22..31 as an all-random tail; vino randomized byte22, which is almost
    // never 0/1, so the dock rejected every probe (generic id=0x14/status-04) instead of routing to
    // the EDID handler (rich id=0x44). Set it to the head being fetched; byte23.. stays host-random.
    b.push(head, GFP_KERNEL)?;
    let mut tail = [0u8; 9];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}
/// A video timing in DisplayID-Type-I terms (sec 8.6.4), as carried by the
/// `0x48/0x22` set-mode message. Field meanings and offsets are verified
/// byte-exact against the golden 3840x2160@60 capture (see [`set_mode`]).
#[derive(Clone, Copy)]
pub(super) struct Timing {
    pub hactive: u16,
    pub hblank: u16,
    pub hsync_front: u16,
    pub hsync_width: u16,
    pub vactive: u16,
    pub vblank: u16,
    pub vsync_front: u16,
    pub vsync_width: u16,
    pub refresh_hz: u16,
    /// Pixel clock in 10 kHz units (e.g. 0xd040 = 533.12 MHz for 4K@60).
    ///
    /// **Held as a u32, but only the low 16 bits are proven on the wire.** Every DLM capture in the
    /// corpus is under 655.35 MHz (the highest is 4K@60 at 533.12 MHz), so off70's u16 always
    /// sufficed and off72 was always zero -- which means no capture can distinguish "the field is
    /// 16-bit" from "the field is the 24-bit DisplayID Type-I clock and its high byte happens to be
    /// zero". Above 655.35 MHz the two differ, and this used to `clamp()` silently: a 1440p165 mode
    /// (699.5 MHz) went out as 655.35 MHz with no error anywhere, so the dock was told the wrong
    /// clock. It now writes the high byte to off72 (see [`set_mode`]) and refuses to encode a clock
    /// that does not fit 24 bits. See `docs/HIGH-REFRESH.md`.
    pub pixel_clock_10khz: u32,
    /// DisplayID field at off42 -- **undecoded, mode-dependent, filled from [`mode_words`]**.
    ///
    /// Measured across the DLM corpus it tracks the *resolution* and not the refresh rate:
    /// 1080p ⇒ `0x0400` at both 60 and 120 Hz, 1440p ⇒ `0x0600`. `docs/VIDEO.md` reads it instead as
    /// the DP link-bandwidth tier (1024 = HBR, 1536 = HBR2), which fits the same data; see
    /// [`mode_words`], where the two readings diverge and what to do about it.
    /// Constructors set it from [`mode_words`]; do not hardcode it.
    pub field42: u16,
    /// The undecoded mode-dependent word at off66 -- also filled from [`mode_words`].
    ///
    /// Unlike [`Self::field42`] this is **not** a function of resolution alone: at 1080p it is
    /// `0x2810` at 60 Hz and `0x083f` at 120 Hz. See [`mode_words`] for the measured corpus and for
    /// why that matters to the pending 1920x1080@165 experiment.
    pub off66: u16,
}
impl Timing {
    /// 3840x2160@60 (CVT-RB) -- the mode the non-HDCP dongle advertises, kept as a
    /// known-good reference whose `set_mode` output is byte-exact vs the golden capture.
    pub(super) const UHD_60: Timing = Timing {
        hactive: 3840,
        hblank: 160,
        hsync_front: 48,
        hsync_width: 32,
        vactive: 2160,
        vblank: 62,
        vsync_front: 3,
        vsync_width: 5,
        refresh_hz: 60,
        pixel_clock_10khz: 0xd040,
        // The golden 4K@60 capture is the non-HDCP *dongle*, not the D6000, and its off66 is not
        // legible in the archived transcript -- so this keeps the `0x0800` vino has always sent for
        // it. `mode_words` deliberately does not claim a 3840x2160 entry for the same reason.
        field42: 0x0604,
        off66: 0x0800,
    };
}
/// The two undecoded mode-dependent words of the set-mode message, `off42` and `off66`, as
/// **measured** from decrypted DLM `id=0x48 sub=0x22` traffic.
///
/// | DLM mode | off42 | off66 | source |
/// |---|---|---|---|
/// | 1920x1080@60 | `0x0400` | `0x2810` | `captures/max-cold-20260721-235609` ctr=244 |
/// | 1920x1080@120 | `0x0400` | `0x083f` | `captures/dlm-180hz-cp-20260726-2300` ctr=929/1072 |
/// | 2560x1440@120 | `0x0600` | `0x0800` | `dlm-180hz` ctr=999/1143, `max-cold` ctr=318 |
///
/// Six DLM messages over three modes and both heads; `off68` is `0x0200` and `off72` is `0` in every
/// one of them, and off44 is the refresh rate in every one of them (60/120 tracking the derived rate
/// exactly), which is why those three are not parameterised here.
///
/// ⚠ **off42 tracks the resolution; off66 does NOT.** At 1080p, off66 changes with *refresh*
/// (`0x2810` at 60, `0x083f` at 120) while off42 stays `0x0400`.
/// `captures/dlm-180hz-cp-20260726-2300/SUMMARY.md` concluded both were resolution-dependent and
/// eliminated off66 as a dark-panel suspect, but that capture contained only 120 Hz samples of
/// 1080p; the 1080p**60** message above refutes it. So off66 is **not** eliminated, and vino's
/// long-standing hardcoded `0x0800` was a 1440p value being sent for every 1080p mode too.
///
/// ⚠ Do not add a 2560x1440@59.95 row from `captures/vino-prompt-timing-20260725-2339` -- that
/// message is **vino's own** CP output, so it merely echoes the hardcoded values back. Only DLM
/// traffic is evidence here. Consequently 1440p is sampled at exactly one refresh, and off66 is
/// **not** known to be refresh-independent there either.
///
/// For a mode with no exact measurement this falls back to the nearest measured refresh at the same
/// resolution, else to `0x0600`/`0x0800` (the 1440p120 pair vino has always sent, and the pair
/// proven to light both panels). `exact` reports which happened so the caller can say so out loud:
/// the pending 1920x1080@165 discriminator has **no** measured off66, and a dark result there must
/// not be read as "the dock caps refresh" while an unproven word is on the wire.
///
/// ★ **The A/B to run before blaming the dock at 1080p165.** `docs/VIDEO.md` reads off42 not as a
/// resolution tag but as the **DP link-bandwidth tier** required by the mode (`0x0400` = 1024 = HBR,
/// `0x0600` = 1536 = HBR2). That model fits all four captures *better*, because it explains why the
/// value tracks resolution: at 24 bpp, 1080p60 (3.56 Gbps) and 1080p120 (7.13 Gbps) both fit 4-lane
/// HBR's ~8.6 Gbps, while 1440p120 (11.95 Gbps) and 4K60 (12.79 Gbps) both need HBR2. The two models
/// agree on every mode ever captured and disagree **exactly** on 1920x1080@165: ~381 MHz CVT-RB is
/// 9.14 Gbps, which is above 4-lane HBR, so the tier model wants `0x0600` where the table gives
/// `0x0400`. The corpus cannot settle it -- the real threshold is only bounded to (7.13, 11.95) Gbps.
/// So if 1080p165 comes up dark, **retry it with `0x0400` → `0x0600` here before concluding anything
/// about the dock's refresh ceiling.**
pub(super) fn mode_words(hactive: u16, vactive: u16, refresh_hz: u16) -> (u16, u16, bool) {
    // (hactive, vactive, refresh, off42, off66) -- DLM captures only; see the table above.
    const MEASURED: &[(u16, u16, u16, u16, u16)] = &[
        (1920, 1080, 60, 0x0400, 0x2810),
        (1920, 1080, 120, 0x0400, 0x083f),
        (2560, 1440, 120, 0x0600, 0x0800),
    ];
    let mut best: Option<(u16, u16, u16)> = None; // (off42, off66, |refresh delta|)
    for &(hact, vact, refresh, off42, off66) in MEASURED {
        if hact != hactive || vact != vactive {
            continue;
        }
        if refresh == refresh_hz {
            return (off42, off66, true);
        }
        let delta = refresh.abs_diff(refresh_hz);
        if best.is_none_or(|(_, _, d)| delta < d) {
            best = Some((off42, off66, delta));
        }
    }
    match best {
        Some((off42, off66, _)) => (off42, off66, false),
        None => (0x0600, 0x0800, false),
    }
}
/// Set-mode (`id=0x48 sub=0x22`): an 80-byte inner message carrying the target head and a
/// DisplayID-Type-I u16 timing record.
///
/// Fresh DLM 3.4.26 cold-plug and live mode-change captures (2026-07-21) establish that off22 is
/// the head selector, off23 is the fixed value 2, and off74..79 is a fresh six-byte random tail.
/// Earlier notes called this a 96-byte plaintext because they accidentally treated the 16-byte
/// Dl3Cmac appended on the wire as encrypted content. Layout (inner offsets): off20..21 zero;
/// off22 head; off23 generation=2; off24..25 zero; off26 begins the LE u16
/// record `hactive,hblank,hsync_front,hsync_width,vactive,vblank,vsync_front,
/// vsync_width,field42,refresh,flags(0x4000)`; off48/off58/off60 carry observed constants; off66 is
/// an undecoded mode-dependent word (see [`mode_words`], which supplies it and off42 from the
/// measured corpus); off70 is the pixel clock (10 kHz units). DLM 3.4.26 writes `0x0200` at off68 in
/// fresh live mode changes. Vino writes the observed value too; a USB ACK alone was not evidence
/// that zero was equivalent because the panel still remained dark.
///
/// off44 is the **refresh rate**, confirmed against four modes (60/120 tracking the derived rate
/// exactly). `captures/dlm-180hz-cp-20260726-2300/SUMMARY.md` records it as "120 (constant, NOT
/// refresh)"; that capture happened to contain only 120 Hz modes. The 1080p**60** message in
/// `captures/max-cold-20260721-235609` carries 60 there, so it is refresh and vino is right to
/// write it.
pub(super) fn set_mode(counter: u16, head: u8, t: &Timing) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(80, GFP_KERNEL)?;
    header(&mut b, 0x48, 0x22, counter)?;
    pad_to(&mut b, 22)?;
    b.push(head, GFP_KERNEL)?; // off22: downstream head selector
    b.push(2, GFP_KERNEL)?; // off23: fixed generation/type value
    pad_to(&mut b, 26)?; // off24..25 zero; timing begins at off26
    for v in [
        t.hactive,
        t.hblank,
        t.hsync_front,
        t.hsync_width,
        t.vactive,
        t.vblank,
        t.vsync_front,
        t.vsync_width,
        t.field42,
        t.refresh_hz,
        0x4000,
        /* off46 flags */ 0x6000, /* off48 */
    ] {
        b.extend_from_slice(&v.to_le_bytes(), GFP_KERNEL)?;
    }
    pad_to(&mut b, 58)?;
    b.extend_from_slice(&0x0080u16.to_le_bytes(), GFP_KERNEL)?; // off58 (observed const)
    b.extend_from_slice(&0x00ffu16.to_le_bytes(), GFP_KERNEL)?; // off60 (observed const)
    pad_to(&mut b, 66)?;
    b.extend_from_slice(&t.off66.to_le_bytes(), GFP_KERNEL)?; // off66: see `mode_words`
    b.extend_from_slice(&0x0200u16.to_le_bytes(), GFP_KERNEL)?; // off68: DLM 3.4.26 constant
    // off70..72 pixel clock. The low u16 at off70 is proven byte-exact against every capture; off72
    // is the DisplayID Type-I 24-bit clock's high byte ON A HYPOTHESIS -- it is zero for every mode
    // under 655.35 MHz, i.e. for every mode any capture covers and every mode vino shipped before
    // 2026-07-26, so writing it changes nothing below that line and is the only way anything above
    // it can be expressed at all. off73 stays zero (the following `pad_to`).
    b.extend_from_slice(&(t.pixel_clock_10khz as u16).to_le_bytes(), GFP_KERNEL)?; // off70
    b.push((t.pixel_clock_10khz >> 16) as u8, GFP_KERNEL)?; // off72
    pad_to(&mut b, 74)?;
    let mut tail = [0u8; 6];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?; // off74..79: fresh per-message token
    Ok(b)
}
/// Standard VESA MCCS (Monitor Control Command Set 2.2) VCP feature codes, driven over
/// DDC/CI. The macOS DisplayLink agent exposes these as per-display brightness/contrast
/// ("Popover did show -- starting DDC/CI communication", `setBrightness`/`setContrast`); the
/// dock bridges the DDC/CI transaction to the downstream monitor's I2C slave 0x37 -- the same
/// monitor-I2C path the EDID read ([`get_edid_req`]) uses for the 0x50 EDID slave.
pub(super) const VCP_BRIGHTNESS: u8 = 0x10;
pub(super) const VCP_CONTRAST: u8 = 0x12;
/// VCP 0xD6 "Power mode": value 0x01 = on, 0x04 = off (DPMS-off / hard standby). Lets DPMS
/// blank the panel backlight instead of freezing the last frame (see [`crtc_atomic_disable`]).
pub(super) const VCP_POWER_MODE: u8 = 0xd6;
pub(super) const POWER_ON: u16 = 0x01;
pub(super) const POWER_OFF: u16 = 0x04;
/// Build a DDC/CI "Set VCP Feature" request: the 7 bytes a DDC/CI host writes to the
/// monitor's I2C slave 0x37, after the 0x6e (= 0x37<<1) write address (VESA DDC/CI 1.1
/// sec 4.4). Layout: source 0x51, length `0x80 | 4`, opcode 0x03 (Set VCP), VCP code,
/// value-hi, value-lo, then an XOR checksum seeded with the destination address 0x6e. Pure
/// and fully standard, so it is unit-tested byte-exact against the spec
/// ([`super::tests::ddc_ci_set_vcp_checksum`]).
pub(super) fn ddc_ci_set_vcp(vcp: u8, value: u16) -> [u8; 7] {
    let body = [0x51u8, 0x84, 0x03, vcp, (value >> 8) as u8, value as u8];
    let mut chk = 0x6eu8; // checksum seed = destination slave-write address (0x37 << 1)
    for &x in &body {
        chk ^= x;
    }
    [body[0], body[1], body[2], body[3], body[4], body[5], chk]
}
/// CP message that tunnels a DDC/CI Set-VCP write to one downstream monitor.
pub(super) fn ddc_set_vcp(counter: u16, head: u8, vcp: u8, value: u16) -> Result<KVec<u8>> {
    ddc_forward(counter, head, &ddc_ci_set_vcp(vcp, value))
}
/// The DDC/CI I2C slave address on the monitor bus.
pub(super) const DDCCI_I2C_ADDR: u8 = 0x37;
/// Tunnel a raw DDC/CI write to `head`. This layout is decrypted wire ground truth from DLM
/// 3.4.26 while setting and reading brightness on both D6000 outputs (2026-07-21): `id=0x36
/// sub=0x26`, head at off22, payload length at off23, raw DDC bytes at off24, zero padding through
/// off55, and a fresh eight-byte token at off56..63. The I2C slave address is implicit; it is not
/// carried in the CP plaintext.
pub(super) fn ddc_forward(counter: u16, head: u8, payload: &[u8]) -> Result<KVec<u8>> {
    if payload.len() > 32 {
        return Err(EINVAL);
    }
    let mut b = KVec::with_capacity(64, GFP_KERNEL)?;
    header(&mut b, 0x36, 0x26, counter)?;
    pad_to(&mut b, 22)?;
    b.push(head, GFP_KERNEL)?;
    b.push(payload.len() as u8, GFP_KERNEL)?;
    b.extend_from_slice(payload, GFP_KERNEL)?;
    pad_to(&mut b, 56)?;
    let mut tail = [0u8; 8];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}

/// Request the result of the preceding DDC/CI read command for `head`. DLM sends this 32-byte
/// `id=0x15 sub=0x25` message after forwarding a Get-VCP write; the dock responds with
/// `id=0x20 sub=0x25`, response length at inner off22 and raw DDC bytes at off23.
pub(super) fn ddc_read_req(counter: u16, head: u8) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    header(&mut b, 0x15, 0x25, counter)?;
    pad_to(&mut b, 22)?;
    b.push(head, GFP_KERNEL)?;
    let mut tail = [0u8; 9];
    rng::fill(&mut tail);
    b.extend_from_slice(&tail, GFP_KERNEL)?;
    Ok(b)
}
/// EDID base-block sanity check: length, the `00 FF..FF 00` magic, and the 1-byte
/// checksum (all 128 base bytes sum to 0 mod 256). A corrupt blob must never drive a
/// mode-set, so [`timing_from_edid`] rejects anything that fails this.
fn edid_valid(edid: &[u8]) -> bool {
    const MAGIC: [u8; 8] = [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00];
    edid.len() >= 128
        && edid[..8] == MAGIC
        && edid[..128].iter().fold(0u8, |a, &b| a.wrapping_add(b)) == 0
}
/// Parse one 18-byte EDID detailed timing descriptor into a [`Timing`], or `None` if it
/// is too short or not a timing (pixel clock 0 marks a monitor descriptor). The two undecoded
/// mode-dependent words come from [`mode_words`].
fn parse_dtd(d: &[u8]) -> Option<Timing> {
    if d.len() < 18 {
        return None;
    }
    let pclk = u16::from_le_bytes([d[0], d[1]]);
    if pclk == 0 {
        return None; // monitor descriptor, not a detailed timing
    }
    let hi = |v: u8, lo: u8| -> u16 { ((v as u16) << 8) | lo as u16 };
    let hactive = hi((d[4] >> 4) & 0xf, d[2]);
    let hblank = hi(d[4] & 0xf, d[3]);
    let vactive = hi((d[7] >> 4) & 0xf, d[5]);
    let vblank = hi(d[7] & 0xf, d[6]);
    let hsync_front = (((d[11] >> 6) & 0x3) as u16) << 8 | d[8] as u16;
    let hsync_width = (((d[11] >> 4) & 0x3) as u16) << 8 | d[9] as u16;
    let vsync_front = (((d[11] >> 2) & 0x3) as u16) << 4 | ((d[10] >> 4) & 0xf) as u16;
    let vsync_width = ((d[11] & 0x3) as u16) << 4 | (d[10] & 0xf) as u16;
    let htotal = hactive.wrapping_add(hblank) as u32;
    let vtotal = vactive.wrapping_add(vblank) as u32;
    let refresh_hz = if htotal != 0 && vtotal != 0 {
        ((pclk as u32 * 10_000 + (htotal * vtotal) / 2) / (htotal * vtotal)) as u16
    } else {
        0
    };
    let (field42, off66, _) = mode_words(hactive, vactive, refresh_hz);
    Some(Timing {
        hactive,
        hblank,
        hsync_front,
        hsync_width,
        vactive,
        vblank,
        vsync_front,
        vsync_width,
        refresh_hz,
        pixel_clock_10khz: u32::from(pclk),
        field42,
        off66,
    })
}
/// Extract the monitor's **preferred** detailed timing from an EDID for the live mode-set
/// (CP-HANDSHAKE.md sec 4e). The first DTD in the base block is the preferred timing per the
/// EDID spec; scan all four base descriptor slots (off 54/72/90/108) so a leading monitor
/// descriptor (name/range/serial) doesn't hide it, and if the base block carries no DTD at
/// all, fall back to the first DTD in the CTA-861 extension block. The blob is validated
/// first; an invalid or timing-less EDID returns `None` so the caller keeps its known-good
/// fallback timing rather than driving the dock with garbage.
pub(super) fn timing_from_edid(edid: &[u8]) -> Option<Timing> {
    if !edid_valid(edid) {
        return None;
    }
    // Base-block descriptors: the first valid DTD is the preferred timing.
    for off in [54usize, 72, 90, 108] {
        if off + 18 <= edid.len() {
            if let Some(t) = parse_dtd(&edid[off..off + 18]) {
                return Some(t);
            }
        }
    }
    // No DTD in the base block: try the first CTA-861 extension's DTD area. CTA-861 blocks
    // have tag 0x02 at byte 0 and a DTD-area byte offset at byte 2 (>= 4 when DTDs follow);
    // descriptors run in 18-byte records up to the extension's checksum byte (127).
    if edid[126] as usize >= 1 && edid.len() >= 256 {
        let ext = &edid[128..256];
        if ext[0] == 0x02 {
            let start = ext[2] as usize;
            if start >= 4 {
                let mut off = start;
                while off + 18 <= 127 {
                    if let Some(t) = parse_dtd(&ext[off..off + 18]) {
                        return Some(t);
                    }
                    off += 18;
                }
            }
        }
    }
    None
}
/// Overwrite the geometry + clock fields of an in-place set-mode inner message
/// (`id=0x48 sub=0x22`) with `t` (CP-HANDSHAKE.md sec 4e). Offsets mirror [`set_mode`]:
/// the LE u16 timing record at off26 and the pixel clock at off70. The encrypted trailer is
/// intentionally left as captured; only the EDID-derived values change, so the wire length (hence
/// `wire_seq`) is unchanged. No-op if `plain` is too short.
///
/// off42/off66 are patched too, from [`mode_words`]: this rewrites a captured message's *geometry*,
/// and both of those words are mode-dependent, so leaving them at the captured mode's values
/// contradicts the new timing. (This path currently has no callers -- the live mode-set builds a
/// fresh message with [`set_mode`] -- but a stale-word bug here would be invisible if it gained one.)
pub(super) fn apply_edid_timing(plain: &mut [u8], t: &Timing) {
    if plain.len() < 73 {
        return;
    }
    let put = |b: &mut [u8], off: usize, v: u16| {
        b[off] = v as u8;
        b[off + 1] = (v >> 8) as u8;
    };
    put(plain, 26, t.hactive);
    put(plain, 28, t.hblank);
    put(plain, 30, t.hsync_front);
    put(plain, 32, t.hsync_width);
    put(plain, 34, t.vactive);
    put(plain, 36, t.vblank);
    put(plain, 38, t.vsync_front);
    put(plain, 40, t.vsync_width);
    put(plain, 42, t.field42);
    put(plain, 44, t.refresh_hz);
    put(plain, 66, t.off66);
    put(plain, 70, t.pixel_clock_10khz as u16);
    plain[72] = (t.pixel_clock_10khz >> 16) as u8; // see `set_mode`: 24-bit clock hypothesis
}
/// Convert a DRM display mode (the timing the *compositor* selected from the connector's
/// EDID-derived mode list) into a set-mode [`Timing`]. This is what makes the dock
/// multi-mode: `drm_edid_connector_add_modes` already advertises every base+extension mode
/// from the dock's EDID, and when userspace sets any one of them the resulting
/// `drm_display_mode` lands here verbatim -- no re-parsing of EDID offsets. The blanking
/// fields map straight across (CVT/DMT/DisplayID all use the same front-porch/sync model),
/// and the refresh rate comes from DRM's own `drm_mode_vrefresh` helper rather than a
/// hand-rolled divide. The two undecoded mode-dependent words come from [`mode_words`], which also
/// says whether the mode was actually measured -- a mode that was not gets one log line, because a
/// dark panel on a mode carrying an unproven off66 is not evidence about the dock.
///
/// SAFETY: `mode` must point to a valid `drm_display_mode` for the duration of the call.
pub(super) fn timing_from_drm_mode(mode: &kernel::drm::kms::modes::DisplayMode) -> Timing {
    let refresh = mode.vrefresh() as u16;
    let sub = |a: u16, b: u16| a.saturating_sub(b);
    let (field42, off66, exact) = mode_words(mode.hdisplay(), mode.vdisplay(), refresh);
    if !exact {
        pr_info!(
            "vino: {}x{}@{} has no measured set-mode mode-words; sending off42=0x{:04x} off66=0x{:04x} (UNPROVEN)\n",
            mode.hdisplay(),
            mode.vdisplay(),
            refresh,
            field42,
            off66
        );
    }
    Timing {
        hactive: mode.hdisplay(),
        hblank: sub(mode.htotal(), mode.hdisplay()),
        hsync_front: sub(mode.hsync_start(), mode.hdisplay()),
        hsync_width: sub(mode.hsync_end(), mode.hsync_start()),
        vactive: mode.vdisplay(),
        vblank: sub(mode.vtotal(), mode.vdisplay()),
        vsync_front: sub(mode.vsync_start(), mode.vdisplay()),
        vsync_width: sub(mode.vsync_end(), mode.vsync_start()),
        refresh_hz: refresh,
        // `clock` is in kHz; the set-mode field is in 10 kHz units. This used to `clamp()` to
        // `u16::MAX`, which SILENTLY sent the wrong clock for anything above 655.35 MHz -- e.g.
        // 2560x1440@165 (699.5 MHz) went out as 655.35 MHz and nothing logged it. The 24-bit
        // ceiling below is unreachable in practice (`MAX_HEAD_CLOCK_KHZ` prunes long before it), so
        // it is a hard error rather than a clamp: better a failed mode-set than a wrong one.
        pixel_clock_10khz: {
            let k = (mode.clock().max(0) as u32) / 10;
            if k > 0x00ff_ffff {
                pr_warn!("vino: pixel clock {k}0 kHz exceeds the 24-bit set-mode field\n");
                0x00ff_ffff
            } else {
                k
            }
        },
        field42,
        off66,
    }
}
/// Decode the inner header of a dock->host CP frame: returns `(id, sub, ictr)` from
/// the first decrypted block (CP-HANDSHAKE.md sec 3), or `None` if `wire` is not a
/// decryptable CP frame. Used by the live loop to log what the dock is replying.
pub(super) fn reply_info(ks: &[u8; 16], out_riv: &[u8; 8], wire: &[u8]) -> Option<(u16, u16, u16)> {
    if wire.len() <= 16 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let head = &wire[16..wire.len().min(32)];
    let inner = open_in(ks, &in_riv(out_riv), seq, head).ok()?;
    if inner.len() < 6 {
        return None;
    }
    Some((
        u16::from_le_bytes([inner[0], inner[1]]),
        u16::from_le_bytes([inner[2], inner[3]]),
        u16::from_le_bytes([inner[4], inner[5]]),
    ))
}
/// CP `sub` ids seen on the wire (CP-HANDSHAKE.md). Used to score a candidate
/// decrypt: a plaintext whose `sub` is one of these (and whose post-counter pad is
/// zero) is almost certainly the correct key/riv.
fn is_known_sub(sub: u16) -> bool {
    matches!(
        sub,
        0x00 | 0x04 | 0x0b | 0x0c | 0x10 | 0x20 | 0x21 | 0x22 | 0x24 | 0x25 | 0x2a | 0x30
        | 0x41 | 0x42 | 0x43 | 0x45 | 0x75 | 0x84 | 0x86
        // The dock's replies to the per-head stream setup: `id=0x14 sub=0x31` (strm2 status),
        // `id=0x26 sub=0x4a` (RepeaterAuth), `id=0x14 sub=0x4c` (finalize). These decrypt to valid
        // CP headers but were missing from this allow-list, so `verify_in_ack` returned `None` and
        // `send_cp_setup` logged them as ~9 "rejects" per session (`sub=0x45_acks` undercounted by
        // that many). They are valid acks, not rejections -- identified offline from the decoded CP
        // dialogue (`captures/rr-out-sequence-20260716/cp-dialogue-decoded.txt`), 2026-07-19.
        | 0x31 | 0x4a | 0x4c
        // `id=0x14 sub=0x4b`: the dock's ack to our EDID-readiness kick (`id=0x16 sub=0x4b`,
        // restored 2026-07-19). Decodes to a valid CP header; catalogued here so it counts as an
        // ack, not an "unlisted sub" (and never the old false "the wall") -- identified on HW
        // 2026-07-20 once the reject-classifier stopped conflating it with genuine garbage.
        | 0x4b
    )
}
/// Diagnostic decode: try a dock->host frame under every plausible riv variant and
/// return the best-scoring inner `(riv_tag, id, sub, ictr)`. The interactive
/// `wsub=0x45` replies decrypt under `in_riv` (byte7^1), but the **cap-phase**
/// `wsub=0x25` frames decrypt under the session ks with **byte7 unchanged** (the OUT
/// value) -- see the cold-ref transcript. `byte0^0x80` selects the head. This mirrors
/// `decode-handshake.py`'s scoring so a live trace shows what the dock is actually
/// asking for during the capability exchange we currently skip.
pub(super) fn decode_any(
    ks: &[u8; 16],
    out_riv: &[u8; 8],
    wire: &[u8],
) -> Option<(&'static str, u16, u16, u16, [u8; 24])> {
    if wire.len() <= 16 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let head = &wire[16..wire.len().min(48)];
    let out0 = *out_riv;
    let in0 = in_riv(out_riv);
    let mut out1 = out0;
    out1[0] ^= 0x80;
    let mut in1 = in0;
    in1[0] ^= 0x80;
    let variants: [(&'static str, [u8; 8]); 4] = [
        ("out/h0", out0),
        ("in/h0", in0),
        ("out/h1", out1),
        ("in/h1", in1),
    ];
    let mut best: Option<(i32, &'static str, u16, u16, u16, [u8; 24])> = None;
    for (tag, riv) in variants.iter() {
        let Ok(pt) = open_in(ks, riv, seq, head) else {
            continue;
        };
        if pt.len() < 8 {
            continue;
        }
        let id = u16::from_le_bytes([pt[0], pt[1]]);
        let sub = u16::from_le_bytes([pt[2], pt[3]]);
        let ctr = u16::from_le_bytes([pt[4], pt[5]]);
        let pad = u16::from_le_bytes([pt[6], pt[7]]);
        let mut sc = 0i32;
        if is_known_sub(sub) {
            sc += 50;
        }
        if pad == 0 {
            sc += 10;
        }
        if ctr < 0x400 {
            sc += 5;
        }
        if best.map_or(true, |b| sc > b.0) {
            // Keep the first 24 plaintext bytes so the live trace shows the decoded
            // structure (e.g. the `..4c..de..` cap-descriptor template that, in the
            // capture, is session-independent -- its absence flags a ks/riv mismatch).
            let mut sample = [0u8; 24];
            let n = pt.len().min(24);
            sample[..n].copy_from_slice(&pt[..n]);
            best = Some((sc, tag, id, sub, ctr, sample));
        }
    }
    best.map(|(_, tag, id, sub, ctr, sample)| (tag, id, sub, ctr, sample))
}
/// Classify a dock->host `wsub=0x45` frame: is it a GENUINE control-plane ack sealed to our
/// live session, or the dock's constant heartbeat/rejection traffic that merely carries the
/// `0x45` wire tag? A tag-only test (`wire[8..10] == 0x45`) cannot tell them apart -- the dock
/// emits `0x45`-tagged status frames even when it has NOT engaged our cipher, and those are
/// byte-identical across sessions (keystream-independent), so counting them as acks is the
/// long-standing false positive. We therefore DECRYPT: a real ack decrypts under the session
/// key to a small `id`, a known `sub`, and zero post-counter pad; heartbeat/garbage decrypts
/// to high-entropy noise (`id`~0x9b.., random `sub`, nonzero pad).
///
/// Returns `Some((id, sub, ctr))` only for a verified ack; `None` for a non-0x45 frame OR a
/// `0x45` frame that does not decrypt to a valid CP header (i.e. the dock is rejecting us).
///
/// The IN-riv model has drifted across dock firmware -- old cold-ref/dl3cmac captures decrypt
/// replies under `IN == OUT` (both `delivered ^ 0x04`); the current `0c 02 19` firmware
/// (`live-3426`) uses `OUT == delivered ^ 0x01`, `IN == delivered` (raw). Recognition must not
/// bet on one model, so we try every plausible IN riv (the raw session value and its byte7^1
/// sibling, each with the byte0^0x80 head flag) and accept the first that yields a valid
/// header. The conjunction (small id AND known sub AND zero pad) is ~1/2^30, so trying four
/// riv candidates does not manufacture a false positive out of the dock's heartbeat noise.
pub(super) fn verify_in_ack(
    ks: &[u8; 16],
    out_riv: &[u8; 8],
    wire: &[u8],
) -> Option<(u16, u16, u16)> {
    if wire.len() <= 16 || u16::from_le_bytes([wire[8], wire[9]]) != 0x45 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let head = &wire[16..wire.len().min(32)];
    let raw = in_riv(out_riv);
    let mut flipped = raw;
    flipped[7] ^= 0x01;
    for base in [raw, flipped] {
        for h in 0..2u8 {
            let mut riv = base;
            if h == 1 {
                riv[0] ^= 0x80;
            }
            let Ok(pt) = open_in(ks, &riv, seq, head) else {
                continue;
            };
            if pt.len() < 8 {
                continue;
            }
            let id = u16::from_le_bytes([pt[0], pt[1]]);
            let sub = u16::from_le_bytes([pt[2], pt[3]]);
            let ctr = u16::from_le_bytes([pt[4], pt[5]]);
            let pad = u16::from_le_bytes([pt[6], pt[7]]);
            if id < 0x400 && is_known_sub(sub) && pad == 0 {
                return Some((id, sub, ctr));
            }
        }
    }
    None
}

/// Lenient sibling of [`verify_in_ack`]: decrypt a `wsub=0x45` reply and return its inner
/// `(id, sub, ctr)` if it decodes to a **plausible CP header** -- `id < 0x400` and `pad == 0` -- but
/// WITHOUT requiring the `sub` to be in [`is_known_sub`]. Used to tell "decrypts fine under our key,
/// just an uncatalogued sub" apart from "genuine garbage": the former means the cipher IS engaged (a
/// valid ack), so it must NOT be logged as `dock REJECTING our CP -- the wall` (a scary false alarm).
/// `id < 0x400 && pad == 0` alone is a ~1/4M filter, so garbage does not slip through. Returns `None`
/// only when the frame genuinely fails to decrypt to a sane header under every riv variant.
pub(super) fn decode_in_lenient(
    ks: &[u8; 16],
    out_riv: &[u8; 8],
    wire: &[u8],
) -> Option<(u16, u16, u16)> {
    if wire.len() <= 16 || u16::from_le_bytes([wire[8], wire[9]]) != 0x45 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let head = &wire[16..wire.len().min(32)];
    let raw = in_riv(out_riv);
    let mut flipped = raw;
    flipped[7] ^= 0x01;
    for base in [raw, flipped] {
        for h in 0..2u8 {
            let mut riv = base;
            if h == 1 {
                riv[0] ^= 0x80;
            }
            let Ok(pt) = open_in(ks, &riv, seq, head) else {
                continue;
            };
            if pt.len() < 8 {
                continue;
            }
            let id = u16::from_le_bytes([pt[0], pt[1]]);
            let sub = u16::from_le_bytes([pt[2], pt[3]]);
            let ctr = u16::from_le_bytes([pt[4], pt[5]]);
            let pad = u16::from_le_bytes([pt[6], pt[7]]);
            if id < 0x400 && pad == 0 {
                return Some((id, sub, ctr));
            }
        }
    }
    None
}
/// Extract the dock's **fresh per-head `rrx`** from its `RepeaterAuth`/AKE `AKE_Send_rrx` push.
///
/// During each per-head AKE restatement the dock sends a NEW `rrx` (8 bytes) -- decoded 2026-07-20
/// from the cold DLM capture: it arrives as an `id=0x10 sub=0x84` push whose inner HDCP msg-id
/// (plaintext byte 9) is `AKE_SEND_RRX` (0x06), with the `rrx` at plaintext bytes 10..18 (head0
/// `6911402772ec2aef`, head1 `1575fbc22078f35c` -- distinct per head). The host MUST derive that
/// head's `kd = derive_kd(km, rtx, rrx)` (and thus `edkey`/`V`) from THIS rrx, not the stale main-AKE
/// one, or the dock computes a different `V'` and silently fails the repeater auth -- which is what
/// gates the trusted DDC/EDID path (dock replies bare `id=0x14`, never rich `id=0x44`). Returns the
/// rrx if this frame is that push, else `None`. Same multi-riv-variant decrypt as [`verify_in_ack`],
/// but decrypts enough of the payload (bytes 0..18) to reach the rrx.
pub(super) fn perhead_rrx(ks: &[u8; 16], out_riv: &[u8; 8], wire: &[u8]) -> Option<[u8; 8]> {
    if wire.len() <= 16 || u16::from_le_bytes([wire[8], wire[9]]) != 0x45 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let head = &wire[16..wire.len().min(48)];
    let raw = in_riv(out_riv);
    let mut flipped = raw;
    flipped[7] ^= 0x01;
    for base in [raw, flipped] {
        for h in 0..2u8 {
            let mut riv = base;
            if h == 1 {
                riv[0] ^= 0x80;
            }
            let Ok(pt) = open_in(ks, &riv, seq, head) else {
                continue;
            };
            if pt.len() < 18 {
                continue;
            }
            let id = u16::from_le_bytes([pt[0], pt[1]]);
            let sub = u16::from_le_bytes([pt[2], pt[3]]);
            // id=0x10 sub=0x84 push, inner HDCP msg-id (byte 9) == AKE_SEND_RRX (0x06).
            if id == 0x10 && sub == 0x84 && pt[9] == 0x06 {
                let mut rrx = [0u8; 8];
                rrx.copy_from_slice(&pt[10..18]);
                return Some(rrx);
            }
        }
    }
    None
}
// All three cursor messages share one 32-byte inner layout, recovered byte-exact from the
// cold-ref session by `scripts/verify-cp-seal.py` (a 64x64 cursor, t~=41.3s):
// off0..7 id/sub/counter header
// off8..21 zero
// off22 0x02 constant marker
// off23 head_id (0 / 1 across the cold-ref's two monitors)
// off24..25 field1 LE u16 (create: width / move: X / image: 0)
// off26..27 field2 LE u16 (create: height / move: Y / image: 0)
// off28..31 trailing 4 bytes (DLM leaks UNINITIALISED struct padding here -- the bytes vary
// non-monotonically every message and match no checksum, so the
// dock cannot validate them; we send zero, which is correct)
// The image then appends its w*h*4 BGRA bitmap starting at off32, and its inner id carries a
// 0x40 high-byte flag (0x401c, vs the 0x1c the aux/send_cp path uses). Prior layouts put the
// marker/fields two bytes early and omitted the head byte (cursor_move) -- a latent bug, since
// cursor traffic is post-CP-engagement. See captures/cp-seal-differential-20260622.md.
fn cursor_header(b: &mut KVec<u8>, id: u16, sub: u16, counter: u16, head: u8) -> Result {
    header(b, id, sub, counter)?;
    pad_to(b, 22)?;
    b.push(0x02, GFP_KERNEL)?; // off22 marker
    b.push(head, GFP_KERNEL)?; // off23 head id
    Ok(())
}
/// cursor create (sec 8.6.1): `id=0x1b sub=0x42`, advertises `w x h` for `head`.
pub(super) fn cursor_create(counter: u16, head: u8, w: u16, h: u16) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    cursor_header(&mut b, 0x1b, 0x42, counter, head)?;
    b.extend_from_slice(&w.to_le_bytes(), GFP_KERNEL)?; // off24..25
    b.extend_from_slice(&h.to_le_bytes(), GFP_KERNEL)?; // off26..27
    pad_to(&mut b, 32)?; // off28..31 (DLM uninit; we zero)
    Ok(b)
}
/// cursor move (sec 8.6.1): `id=0x1a sub=0x43`, head id @23, X @24, Y @26 (LE).
pub(super) fn cursor_move(counter: u16, head: u8, x: u16, y: u16) -> Result<KVec<u8>> {
    let mut b = KVec::with_capacity(32, GFP_KERNEL)?;
    cursor_header(&mut b, 0x1a, 0x43, counter, head)?;
    b.extend_from_slice(&x.to_le_bytes(), GFP_KERNEL)?; // off24..25
    b.extend_from_slice(&y.to_le_bytes(), GFP_KERNEL)?; // off26..27
    pad_to(&mut b, 32)?; // off28..31 (DLM uninit; we zero)
    Ok(b)
}
/// cursor image (sec 8.6.1): inner `id=0x401c sub=0x41` (the 0x40 high-byte flag marks the
/// bitmap-bearing message). A 32-byte header (shared marker + head, no w/h -- those come from
/// [`cursor_create`]) followed by the `w*h*4` BGRA bitmap at off32. `bgra` must be `w*h*4` bytes
/// -- DRM hands the driver a 64x64 ARGB8888 cursor buffer and the caller swaps it to BGRA.
pub(super) fn cursor_image(
    counter: u16,
    head: u8,
    w: u16,
    h: u16,
    bgra: &[u8],
) -> Result<KVec<u8>> {
    // `w*h*4` can wrap a 32-bit `usize` (max ~1.7e10 > u32::MAX), which would let an
    // undersized `bgra` pass the check; compute it with checked arithmetic so an
    // overflow is rejected as a mismatch rather than silently bypassing validation.
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4));
    if expected != Some(bgra.len()) {
        return Err(EINVAL);
    }
    let mut b = KVec::with_capacity(32 + bgra.len(), GFP_KERNEL)?;
    cursor_header(&mut b, 0x401c, 0x41, counter, head)?;
    pad_to(&mut b, 32)?; // off24..31 zero (no w/h here)
    b.extend_from_slice(bgra, GFP_KERNEL)?; // bitmap @ off32
    Ok(b)
}
/// DisplayLink "Dl3Cmac" CP-message integrity tag (16 bytes) -- **FULLY SOLVED + CROSS-SESSION
/// VERIFIED 2026-06-11** (`captures/DL3CMAC-FULLY-SOLVED-20260611.md`):
/// `tag = AES-CMAC(ks, mac_nonce(8) || BE64(wire_seq) || ciphertext)` where
/// - `mac_nonce` = the AES-CTR content nonce (`riv`) with `byte0 ^= 0x80`. Pass the CTR `riv`
/// (delivered riv with `byte7 ^= 0x04`, byte0 unchanged); the byte0 flip is applied here. The CTR
/// and CMAC nonces are DISTINCT values (they differ by `byte0 ^ 0x80`) -- proven by rr reverse-
/// execution of DLM 3.4.26's real msg0: CTR nonce `f621..af`, CMAC nonce `7621..af`.
/// - `wire_seq` = the AES-CTR block counter (frame header off-12), zero-extended to BE64,
/// - `ciphertext` = the AES-CTR ciphertext content (encrypt-then-MAC), tag appended IN CLEAR.
/// `K_dl3 = ks`. Reproduces cold-ref msg0 AND both rr-trace msg0s byte-exact.
pub(super) fn dl3cmac_tag(
    ks: &[u8; 16],
    riv: &[u8; 8],
    wire_seq: u64,
    ciphertext: &[u8],
) -> Result<[u8; 16]> {
    // The Dl3Cmac nonce = the AES-CTR content nonce with `byte0 ^= 0x80`. rr reverse-execution of
    // DLM 3.4.26's REAL msg0 (2026-07-16) caught the encrypt: the CTR keystream keys on
    // `riv` = delivered `^0x04@byte7` (byte0 UNCHANGED, e.g. f621..af), and this CMAC keys on that
    // SAME `riv` with `byte0 ^= 0x80` (e.g. 7621..af). `riv` passed in is the CTR nonce; the byte0
    // flip is applied here. (The CTR and CMAC nonces DIFFER by byte0 -- they are NOT the same value.)
    let mut mac_nonce = *riv;
    mac_nonce[0] ^= 0x80;
    let mut buf = KVec::with_capacity(16 + ciphertext.len(), GFP_KERNEL)?;
    buf.extend_from_slice(&mac_nonce, GFP_KERNEL)?;
    buf.extend_from_slice(&wire_seq.to_be_bytes(), GFP_KERNEL)?;
    buf.extend_from_slice(ciphertext, GFP_KERNEL)?;
    Ok(crypto::aes_cmac(ks, &buf))
}
/// Seal a CP message with a **freshly computed live Dl3Cmac**, reusing DLM's captured wire
/// `header` (so `seq`/`aux` are byte-identical) but recomputing the tail tag for THIS session.
/// `content_pt` is the real inner plaintext WITHOUT the 16-byte tag region. Wire body =
/// `AES-CTR(ks, riv, content_pt)` || `dl3cmac_tag(...)`. This is the live-generation path. See
/// `captures/DL3CMAC-FULLY-SOLVED-20260611.md`.
pub(super) fn seal_livemac(
    ks: &[u8; 16],
    riv: &[u8; 8],
    header: &[u8],
    content_pt: &[u8],
) -> Result<KVec<u8>> {
    let seq = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let cipher = crypto::Aes128::new(ks)?;
    let mut ct = KVec::with_capacity(content_pt.len(), GFP_KERNEL)?;
    for (i, chunk) in content_pt.chunks(16).enumerate() {
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(riv);
        iv[12..].copy_from_slice(&seq.wrapping_add(i as u32).to_be_bytes());
        let ksb = cipher.encrypt_block(&iv);
        for (j, &p) in chunk.iter().enumerate() {
            ct.push(p ^ ksb[j], GFP_KERNEL)?;
        }
    }
    let tag = dl3cmac_tag(ks, riv, seq as u64, &ct)?;
    let mut frame = KVec::with_capacity(16 + ct.len() + 16, GFP_KERNEL)?;
    frame.extend_from_slice(&header[..16], GFP_KERNEL)?;
    frame.extend_from_slice(&ct, GFP_KERNEL)?;
    frame.extend_from_slice(&tag, GFP_KERNEL)?;
    Ok(frame)
}
/// Seal an inner CP message into a wire frame (type=4 sub=0x24, `seq`). DisplayLink
/// CP is **encrypt-then-MAC**: the message content is AES-CTR-encrypted, then a
/// 16-byte Dl3Cmac tag (`AES-CMAC(ks, riv || BE64(seq) || ciphertext)`) is appended.
/// The keystream is `AES_ECB(ks, riv(8) || u32(0) || u32_be(seq + block))` (sec 6.1).
///
/// `inner` is the captured golden plaintext `[content || stale-tag-region(16)]`; we
/// encrypt only `content = inner[..len-16]` and append a **fresh** tag keyed by our
/// live session, so the dock's Dl3Cmac verification passes (the stale replayed tag is
/// why the dock previously dropped our CP). VERIFIED construction (sec 8.6.7).
pub(super) fn seal(ks: &[u8; 16], riv: &[u8; 8], seq: u32, inner: &[u8]) -> Result<KVec<u8>> {
    // The interactive CP stream: session ks, wire sub `0x24`.
    seal_stream(ks, riv, 0x24, seq, inner)
}
/// Build a fully sealed interactive CP frame (`type=4 sub=0x24`) at `wire_seq` over `content`
/// (the inner plaintext, WITHOUT any appended 16-byte tag placeholder): the 16-byte wire
/// header -- size, `type=4`, `sub=0x24`, the per-`id` [`aux_for_id`] field, and `wire_seq` --
/// followed by [`seal_livemac`] (AES-CTR ciphertext + appended live Dl3Cmac). Shared by the
/// bring-up live loop ([`VinoDriver::send_live_cp`]) and the runtime KMS senders
/// ([`drm_sink::VinoDrmData::send_cp`]) so both produce a byte-identical wire frame.
pub(super) fn seal_interactive(
    ks: &[u8; 16],
    riv: &[u8; 8],
    id: u16,
    wire_seq: u32,
    content: &[u8],
) -> Result<KVec<u8>> {
    let body_len = content.len() + 16; // AES-CTR ciphertext + 16-byte Dl3Cmac
    let size = ((16 + body_len) - 4) as u16;
    let aux = aux_for_id(id, body_len);
    let mut hdr = [0u8; 16];
    hdr[2..4].copy_from_slice(&size.to_le_bytes());
    hdr[4..8].copy_from_slice(&4u32.to_le_bytes()); // type=4
    hdr[8..10].copy_from_slice(&0x24u16.to_le_bytes()); // sub=0x24 (interactive CP)
    hdr[10..12].copy_from_slice(&aux.to_le_bytes());
    hdr[12..16].copy_from_slice(&wire_seq.to_le_bytes());
    seal_livemac(ks, riv, &hdr, content)
}
/// The CP wire-header `aux`@10 (`sub_len_dw`) field is a **strict per-inner-message-id
/// constant** in DLM's CP stream -- verified byte-exact across all 94 captured 1080p CP
/// frames (`cp-hdrwire-1080p.bin`) -- **not** `body.len()/4`, which is what `push_frame`
/// derives. Reproducing it makes a generated CP frame's header byte-identical to DLM, the
/// leading hypothesis for the dock engaging its CP cipher (the dock acks our plaintext cap
/// but emits 0 encrypted replies with the wrong `aux`). See docs/BLOCKER.md and memory
/// `project_cp_aux_field_per_id_constant`. Unknown ids fall back to the dword count so an
/// unrecognised message is still well-formed.
///
/// The cursor ids (0x1a/0x1b/0x1c) are absent from the 1080p set (cursor messages fire only
/// once a cursor is set, post-engagement) and were recovered from the cold-ref session by the
/// `scripts/verify-cp-seal.py` differential: every other OUT CP frame regenerated byte-exact,
/// but the three cursor messages diverged in `aux` alone (DLM 0x04/0x03/0x02 vs the body/4
/// fallback) -- a latent bug that would have made the dock misparse every cursor update. This
/// makes the generated `seal`/`seal_stream` path match DLM without a captured-header blob -- the
/// basis for **live** CP generation.
pub(super) fn aux_for_id(id: u16, body_len: usize) -> u16 {
    match id {
        0x14 => 0x0a,
        0x15 => 0x09,
        0x16 => 0x08,
        0x19 => 0x05,
        0x1a => 0x04, // cursor move
        0x1b => 0x03, // cursor create
        0x1c => 0x02, // cursor image
        0x1f => 0x0f,
        0x22 => 0x0c,
        0x26 => 0x08,
        0x2a => 0x04,
        0x32 => 0x0c,
        0x36 => 0x08, // DDC/CI write (captured wire aux, not body_len/4)
        0x48 => 0x06,
        0x9a => 0x04,
        _ => (body_len / 4) as u16,
    }
}
/// Per-head encrypted AKE re-statement + stream-open table for the post-msg0 CP setup burst:
/// `(id, sub, content_len)`, where `content_len` is the inner plaintext length in bytes BEFORE
/// the 16-byte Dl3Cmac tag [`seal_interactive`] appends. Originally decoded from a real DLM
/// 3.4.26 cold plug via rr reverse-execution
/// (`captures/rr-out-sequence-20260716/cp-dialogue-decoded.txt`, ictr 14..22 for head 0 / 23..31
/// for head 1).
///
/// **2026-07-17: framing FULLY resolved by decryption (`docs/CP-PERHEAD-RESTATEMENT.md`).**
/// Earlier attempts read only the 24-byte truncation in `cp-dialogue-decoded.txt` and guessed at
/// the layout (placing real AKE values at off24 with no HDCP msg-id, `Session::ake` restatement),
/// which HW-tested as a disconnect regression (`project_per_head_burst_length_regression_20260717`)
/// and is now known to be the leading suspect for the head-1 NAK -> dock-reset loop. Decrypting
/// the *complete* encrypted burst (trace1, via `scripts/decrypt-perhead-restatement.py`) shows
/// each restatement is byte-for-byte the plaintext AKE body (`ake::body`: HDCP msg-id at off27,
/// payload at off28) with the `0x30`@22 marker swapped for `off22=0, off23=head+1`. The lengths
/// here are correct (measured from raw ciphertext block counts -- `id=0x9a` is 160B, i.e. 10 AES
/// blocks, matching the seq gap 16->26). **The payloads are DERIVED standard-HDCP-2.2 crypto, not
/// random** (rr-confirmed 2026-07-17: the restatement `Ekpub` is a real RSA-1024 output, `V` a
/// RepeaterAuth_Send_Ack). This is a per-head **downstream repeater AKE**: `VinoDriver::send_cp_setup`
/// computes a fresh self-consistent chain per head (`Ekpub(km_h)`, `edkey`, `V=HMAC(kd_h,list)`)
/// from the same proven primitives `run_ake` uses. The CP-encrypted framing and a few non-standard
/// trailing bytes past each HDCP field are DisplayLink-proprietary (left best-effort/zero, RE
/// pending). `id=0x26` is built by [`stream_manage_restatement`]. See that loop for the exact bytes.
///
/// [`VinoDriver::send_cp_setup`]: super::VinoDriver::send_cp_setup
pub(super) const CP_SETUP_PER_HEAD: [(u16, u16, usize); 9] = [
    (0x0022, 0x0010, 48),  // AKE_Init -- msg-id 0x02 @off27, 20B random payload
    (0x001f, 0x0010, 48),  // AKE_Transmitter_Info -- msg-id 0x13, fixed 00 06 02 00 02 prefix
    (0x009a, 0x0010, 160), // AKE_No_Stored_km -- msg-id 0x04, 132B payload (10 AES blocks)
    (0x0022, 0x0010, 48),  // LC_Init -- msg-id 0x09 @off27, 20B random payload
    (0x0032, 0x0010, 64),  // per-head VIDEO KEY -- msg-id 0x0b, fresh 32B key @off28, stashed
    (0x002a, 0x0010, 48),  // LC_Send_L_prime -- msg-id 0x0f @off27, 20B random payload
    (0x0026, 0x0010, 48),  // RepeaterAuth Stream_Manage -- built by stream_manage_restatement
    (0x0014, 0x0030, 32),  // per-head stream-open ctl -- no marker/tag, 10B random @off22
    (0x0019, 0x0031, 32),  // per-head strm2 -- head @off22, fixed 06 [head*4] 04 @off24
];
/// Build the `id=0x0026 sub=0x0010` (RepeaterAuth Stream_Manage restatement) content for one
/// head: 48 bytes. **Framing corrected 2026-07-17** against a FULL AES-CTR decryption of the real
/// burst (`docs/CP-PERHEAD-RESTATEMENT.md`) -- the earlier reconstruction, from the 24-byte
/// truncation in `cp-dialogue-decoded.txt`, wrongly placed the head marker as a u32 at off20 and
/// `0x10` at off24. The real deterministic layout (everything but the 8-byte random tail is
/// head-invariant across sessions): `[8B header][14 zero][head+1 @off23][0 @off24..27]
/// [0x10 msg-id @off27][0 @off28..32][1:u32 @off32][seq:u32 = 8/9 @off36][8B host-random tail]`.
/// The `seq` field's meaning past "8 then 9" is not understood -- hardcoded to match the wire.
pub(super) fn stream_manage_restatement(counter: u16, head: u8) -> Result<KVec<u8>> {
    // Layout ground-truthed 2026-07-17 by FULL AES-CTR decryption of trace1's encrypted per-head
    // burst (`docs/CP-PERHEAD-RESTATEMENT.md`, `scripts/decrypt-perhead-restatement.py`),
    // superseding the earlier truncated-decode guess that placed the head marker at off20 and the
    // 0x10 tag at off24. The real wire, like every other restatement entry, carries the head
    // marker at off23 and the HDCP msg-id (0x10) at off27, then the `0 / 1 / (head+8)` u32 fields
    // and an 8-byte host-random tail the dock does not validate.
    let mut b = KVec::from_elem(0u8, 48, GFP_KERNEL)?;
    b[0..2].copy_from_slice(&0x0026u16.to_le_bytes());
    b[2..4].copy_from_slice(&0x0010u16.to_le_bytes());
    b[4..6].copy_from_slice(&counter.to_le_bytes());
    b[23] = head + 1; // head marker
    b[27] = 0x10; // HDCP msg-id (RepeaterAuth_Stream_Manage)
    b[32..36].copy_from_slice(&1u32.to_le_bytes());
    b[36..40].copy_from_slice(&(head as u32 + 8).to_le_bytes()); // seq: 8 head0 / 9 head1
    let mut tail = [0u8; 8];
    rng::fill(&mut tail);
    b[40..48].copy_from_slice(&tail);
    Ok(b)
}
/// Stream-finalize tail sent once, after both heads' [`CP_SETUP_PER_HEAD`] blocks: `(id, sub,
/// off22)`. **2026-07-17: corrected against the full decryption** (`docs/CP-PERHEAD-RESTATEMENT.md`,
/// ictr 32..36) -- the content is **32 bytes** (the old 16 was half the real length), with
/// `[8..22]` zero, a per-message index flag at off22 (the observed `0/0/0/1/1` sequence), and for
/// the `sub=0x4c` messages a constant `0x01` at off23; the rest of the tail is host-random (varies
/// per message, dock does not validate it).
pub(super) const CP_SETUP_FINALIZE: [(u16, u16, u8); 6] = [
    (0x0016, 0x004c, 0),
    (0x0015, 0x004a, 0),
    (0x0016, 0x004c, 0),
    (0x0016, 0x004c, 1),
    (0x0015, 0x004a, 1),
    // ADDED 2026-07-19: DLM sends a SIXTH finalize (ictr 37, id=0x16/0x4c, off22=0x01) right before
    // the EDID probe -- decrypted cold fetch (captures/edid-cold-decrypt-20260719/) shows ictr 32..37,
    // not 32..36. vino was truncating the burst at 5; the dock only starts the DDC/EDID read (rich
    // id=0x44 replies) for DLM, and this dropped message is the strongest remaining suspect.
    (0x0016, 0x004c, 1),
];

/// Structured burst DLM sends directly on each head's **video** bulk-OUT endpoint (EP08 head 0 /
/// EP0b head 1) immediately before real video data -- found 2026-07-16 by re-mining the SAME rr
/// recording used for [`CP_SETUP_PER_HEAD`], this time also tracking the video endpoints (every
/// earlier mining pass this session only ever filtered EP02). `(wire_type, sub_base, aux,
/// body_len)`; `sub_base` is head 0's sub value -- head `h`'s sub is `sub_base + h as u16` (the
/// head index sits in the sub LSB, same convention as `CP_SETUP_PER_HEAD`). `wire_type==2`
/// entries are PLAINTEXT and fully known byte-for-byte (built by
/// [`video_arm_plaintext_body`]/[`video_arm_plain_frame`]). `wire_type==4` entries are CP-sealed
/// and their true content/key is **NOT** reverse-engineered: brute-forcing every single-byte XOR
/// variant of the main CP channel's `riv` against the captured ciphertext (all 8 byte positions x
/// 256 values) found no plausible decrypted message id -- this channel's nonce/key context
/// differs from the EP02 channel's.
///
/// **2026-07-17 re-derived from an independent, better source
/// (`captures/dlm-cold-3426-20260714-140216-allbus/dlm.pcapng`, a real full-bus DLM session with
/// monitors attached, NOT the rr-reversed partial trace above) using a proper script (not manual
/// hex reading -- an earlier manual pass this same session got confused and wrongly flagged part
/// of this as a parsing artifact; the scripted walk resolved it cleanly). Confirmed entries 0/1/4/5
/// byte-exact again (three-way match now) and found entries 2/3's true wire `size` field gives
/// `body_len = 16`, not 32. **Entries 6-9 are now REPLACED, not just corrected** -- the real
/// structure at that wire position is completely different from the old guess:
/// - **6/7**: `wire_type=4, aux=0x0004, content_len=0` (just a bare 16-byte Dl3Cmac tag over EMPTY
///   content, `seq=0` each) -- not the old guessed plaintext `sub=0x0020/0x0030` entries. The two
///   tags are byte-identical, which first looked like a bug but is exactly what the established
///   Dl3Cmac formula (`AES_CMAC(key, nonce ‖ BE64(seq) ‖ ciphertext)`) predicts for two messages
///   with the same key, same nonce (session `riv` doesn't change), `seq=0`, and empty ciphertext --
///   not a parsing error after all.
/// - **8/9**: `wire_type=4, sub=0x0008/0x0018 (same subs as 2/3, not new ones), aux=0x000e,
///   content_len=1104`, whose `seq` (2, then 71) is the EXACT continuation of entries 2/3's block
///   counter (2/3 consume blocks 0 and 1; 8 consumes 1104/16=69 blocks landing at 71; 9 starts at
///   71) -- i.e. **entries 2, 3, 8, 9 all share ONE running counter (and therefore almost
///   certainly one key)**. This is very likely real per-channel video-stream **init/header**
///   data (one per sub-channel: 0x0008 and 0x0018), CP-sealed, sent once before the bulk pixel
///   payload -- content still unrecoverable (sealed, key unknown, same as 2/3 always were).
/// - Immediately after entry 9, later "messages" in a structured walk turn out to be an artifact
///   of continuing to greedily parse a `type=4` header shape into what is actually **unstructured,
///   unencrypted codec bitstream** (low-entropy, highly repetitive content, `seq=0` reused across
///   many distinct "messages" -- a real crypto nonce-reuse bug is implausible for a shipped
///   HDCP link, so this is coincidental header-shaped noise, not real framing). **The true boundary
///   between structured CP messages and raw bulk video data is right after entry 9** -- confirms
///   vino's existing one-shot arm design is right in spirit, just missing these two init chunks.
/// See `project_video_arm_burst_init_chunks_found_20260717` memory for the full scripted derivation
/// and the entropy check that located the structured/unstructured boundary.
///
/// **2026-07-17 (later): entries 6-9 CONFIRMED, 4-way cross-validated -- the earlier "two
/// captures disagree" open question is resolved.** A separate rr-replay mining pass of
/// `rr-traces/DisplayLinkManager-0` had reported a completely different structure at this same
/// wire position (`type=2` plaintext `sub=0x0020/0x0030` for 6/7, tiny empty `sub=0x0020/0x0030`
/// pings for 8/9 -- see `project_video_burst_two_captures_disagree_20260717`). Re-parsed the
/// FIRST 65536B EP08 URB from three MORE independent real DLM sessions
/// (`captures/dlm-cold-3426-20260713-{225656,225759,225857}/dlm.pcapng`, different dates/times
/// from both prior sources) with a clean, from-scratch wire-header walker (`type`/`sub`/`aux` are
/// cleartext -- no decryption needed to check this). **All three agree byte-for-byte with the
/// table above** on every field for entries 0-9, including the identical bare Dl3Cmac tag content
/// at 6/7. That makes it 4 independent real sessions (07-13 x3, 07-14 x1) unanimous for this
/// structure, against 1 (the rr-traces recording) for the other. The rr-traces decode is the
/// outlier -- plausibly a re-arm or a different DLM-lifecycle event captured by that recording,
/// not the initial cold arm every pcap capture's first URB shows -- and should not be used to
/// second-guess this table without new, similarly-corroborated evidence.
///
/// **The COLD arm burst** — DLM's real first-video bring-up, 4-way confirmed across the cold-plug
/// pcaps (`dlm-cold-3426-20260714…` frame 1818, `…20260713-225656` frame 1603, +2), re-verified
/// 2026-07-19. See `docs/EP08-ARM-BURST.md`. This is what vino needs: it brings up video on a fresh
/// CP session, so the pipe is unarmed for it. (An earlier 2026-07-18 rr trace captured a *warm
/// re-arm* — 352 B, lighter #6-9 — which is a DIFFERENT, insufficient burst; live HW confirmed the
/// dock accepts it but then ESHUTDOWNs the video frame.)
///
/// **Delivery:** the whole 2560-byte burst is **PREPENDED to frame 0's video in ONE URB** (records
/// #0-9 = arm, #10+ = video), NOT a separate transfer. Layout `(wire_type, sub_base, aux,
/// content_len)`; actual `sub = sub_base + head`.
/// - #0/#1/#4/#5: `type=2` plaintext (byte-known bodies).
/// - #6/#7: `type=4` but FIXED plaintext `0a 00 04 00 …00 10 00 00 00 00` (no seal), sub 0x00/0x10.
/// - #2/#3/#8/#9: `type=4` SEALED under this head's VIDEO channel key (`ske_ks_h XOR CP_KEY_WHITEN`)
///   + content nonce (`riv_h ^ (0x08 | head) @ byte7`), sharing ONE block counter
///   (seq 0,1,2,71). #2/#3 = 16 B plaintext (`04 00 08 04 03 00` + 10 fresh bytes). #8/#9 are
///   1104 B each and their decrypted DLM plaintext is fresh DRBG output, not a hidden structured
///   decoder table. Vino now generates the same 10/10/14/14-byte random requests and expands the
///   14-byte seeds to the 1104-byte bodies. Hardware accepted the complete ARM followed by
///   repeated arbitrary 2.60 MB framebuffer frames; the ARM content is no longer a blocker.
pub(super) const VIDEO_ARM_BURST: [(u32, u16, u16, usize); 10] = [
    (2, 0x0008, 0x0000, 16),   // #0 plaintext: body 08 00 06
    (2, 0x0018, 0x0000, 16),   // #1 plaintext: body 08 00 16
    (4, 0x0008, 0x000a, 16),   // #2 SEALED 16B, per-head video key, seq 0
    (4, 0x0018, 0x000a, 16),   // #3 SEALED 16B, per-head video key, seq 1
    (2, 0x0000, 0x0000, 16),   // #4 plaintext: body 00
    (2, 0x0010, 0x0000, 16),   // #5 plaintext: body 00 00 10
    (4, 0x0000, 0x0004, 16),   // #6 type=4 FIXED plaintext 0a 00 04 … (sub 0x00, unsealed)
    (4, 0x0010, 0x0004, 16),   // #7 type=4 FIXED plaintext 0a 00 04 … (sub 0x10, unsealed)
    (4, 0x0008, 0x000e, 1104), // #8 SEALED 1104B, per-head video key, seq 2 — DRBG-derived body
    (4, 0x0018, 0x000e, 1104), // #9 SEALED 1104B, per-head video key, seq 71 — DRBG-derived body
];

/// Build the fully-known 16-byte plaintext body for one of [`VIDEO_ARM_BURST`]'s `wire_type==2`
/// entries at table index `i`, for head `h`. Byte-exact vs the capture -- see the table's doc
/// comment.
pub(super) fn video_arm_plaintext_body(i: usize, h: u16) -> [u8; 16] {
    let mut b = [0u8; 16];
    match i {
        0 => {
            b[0..2].copy_from_slice(&(0x0008u16 + h).to_le_bytes());
            b[2..4].copy_from_slice(&0x0006u16.to_le_bytes());
        }
        1 => {
            b[0..2].copy_from_slice(&(0x0008u16 + h).to_le_bytes());
            b[2..4].copy_from_slice(&0x0016u16.to_le_bytes());
        }
        4 => {
            b[0..2].copy_from_slice(&h.to_le_bytes());
        }
        5 => {
            b[0..2].copy_from_slice(&h.to_le_bytes());
            b[2..4].copy_from_slice(&0x0010u16.to_le_bytes());
        }
        // In the COLD arm #6/#7 are NOT type-2 plaintext — they are type-4 fixed `0a 00 04 …`
        // records built directly in `build_arm_burst_buf`, so this fn is never called for them.
        _ => {}
    }
    b
}

/// Build a fixed 32-byte `wire_type=2` (plaintext) video-arm-burst frame: 16-byte header
/// (`size=0x1c`, `type=2`, `sub`, `aux=0`, `seq=0`) + the 16-byte `body`. Matches
/// [`VIDEO_ARM_BURST`]'s plaintext entries byte-exact.
pub(super) fn video_arm_plain_frame(sub: u16, body: &[u8; 16]) -> [u8; 32] {
    let mut f = [0u8; 32];
    f[2..4].copy_from_slice(&0x001cu16.to_le_bytes());
    f[4..8].copy_from_slice(&2u32.to_le_bytes());
    f[8..10].copy_from_slice(&sub.to_le_bytes());
    f[16..32].copy_from_slice(body);
    f
}

/// Build a sealed `wire_type=4` video-arm-burst frame: a 16-byte header (`sub`/`aux`/`seq` as
/// given) followed by [`seal_livemac`]'s AES-CTR ciphertext + live Dl3Cmac over `content`. See
/// [`VIDEO_ARM_BURST`]'s doc comment for the (unverified) key/content this is sealed under.
pub(super) fn seal_video_arm(
    key: &[u8; 16],
    riv: &[u8; 8],
    sub: u16,
    aux: u16,
    seq: u32,
    content: &[u8],
) -> Result<KVec<u8>> {
    let body_len = content.len() + 16; // AES-CTR ciphertext + 16-byte Dl3Cmac
    let size = ((16 + body_len) - 4) as u16;
    let mut hdr = [0u8; 16];
    hdr[2..4].copy_from_slice(&size.to_le_bytes());
    hdr[4..8].copy_from_slice(&4u32.to_le_bytes()); // type=4
    hdr[8..10].copy_from_slice(&sub.to_le_bytes());
    hdr[10..12].copy_from_slice(&aux.to_le_bytes());
    hdr[12..16].copy_from_slice(&seq.to_le_bytes());
    seal_livemac(key, riv, &hdr, content)
}
/// General AES-CTR seal under an arbitrary stream `key`/`riv` and wire sub. `seal`
/// is the session-CP case (`wsub=0x24`); the **cap phase** (CP-HANDSHAKE.md sec 4b)
/// needs `wsub=0x04` sealed under the dock's `id=0x32`-delivered per-head stream key,
/// not the session ks -- which `seal` cannot express. Body construction is identical:
/// AES-CTR(key, riv || 0x00000000 || BE32(seq+block)) over the **whole** inner message
/// (no appended MAC; the inner carries its own encrypted trailer -- verified byte-exact
/// vs DLM, 30/30 wire frames).
pub(super) fn seal_stream(
    key: &[u8; 16],
    riv: &[u8; 8],
    wsub: u16,
    seq: u32,
    inner: &[u8],
) -> Result<KVec<u8>> {
    let cipher = crypto::Aes128::new(key)?;
    let mut ct = KVec::with_capacity(inner.len(), GFP_KERNEL)?;
    for (i, chunk) in inner.chunks(16).enumerate() {
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(riv);
        iv[12..].copy_from_slice(&seq.wrapping_add(i as u32).to_be_bytes());
        let ksb = cipher.encrypt_block(&iv);
        for (j, &p) in chunk.iter().enumerate() {
            ct.push(p ^ ksb[j], GFP_KERNEL)?;
        }
    }
    let mut frame = KVec::with_capacity(16 + ct.len(), GFP_KERNEL)?;
    // DLM-exact `aux`@10: a per-inner-id constant (see `aux_for_id`), not `body/4`. The
    // id is read from the *plaintext* inner (off 0); `push_frame` would derive the wrong
    // value and is the suspected reason the dock won't engage its CP cipher.
    let id = if inner.len() >= 2 {
        u16::from_le_bytes([inner[0], inner[1]])
    } else {
        0
    };
    super::proto::push_frame_with(&mut frame, 0x04, wsub, aux_for_id(id, ct.len()), seq, &ct)?;
    Ok(frame)
}
/// Derive the dock->host (IN) CP riv from the host->dock (OUT) `riv` by flipping bit 0 of
/// byte 7. **Ground-truthed against a real ENGAGED DLM session on the current `0c 02 19`
/// firmware** (`captures/live-3426-20260707/reauth.pcap`, ks=`127d51f9..`): the SKE delivers
/// base riv `bf1ed14be51a945a`; the host's real msg0 (OUT) decrypts ONLY under `base^0x01`
/// (`...945b` -> `id=0x14 sub=0 ctr=9`), and the dock's real first ACK (IN) decrypts ONLY under
/// `base` raw (`...945a` -> `id=0x4c sub=0 ctr=9`). Since [`run_ake`] stores the OUT value
/// (`base^0x01`) as the session riv, recovering the IN keystream means flipping byte7 bit0 back
/// to `base`.
///
/// This supersedes the 2026-06-12 "identity / no transform" note, which was measured on an
/// OLDER firmware capture (`dlm-coldkeys-20260611`) and does NOT hold for the dock in front of
/// us -- under identity, a genuine ACK decrypts to garbage (`id=0x8a2d`), which would make vino
/// misread every engaged reply and log the dock as "rejecting" even after it engaged. Verified
/// by decoding both real frames in this tree; see the reauth README.
///
/// [`run_ake`]: super::VinoDriver::run_ake
pub(super) fn in_riv(out_riv: &[u8; 8]) -> [u8; 8] {
    let mut riv = *out_riv;
    riv[7] ^= 0x01;
    riv
}
/// Decrypt a dock->host CP frame body (AES-CTR, the same keystream as [`seal`] but
/// keyed with the IN `riv`). `ct` is the ciphertext (wire bytes after the 16-byte
/// cleartext header); `seq` is the wire counter at wire offset 12.
pub(super) fn open_in(ks: &[u8; 16], in_riv: &[u8; 8], seq: u32, ct: &[u8]) -> Result<KVec<u8>> {
    let cipher = crypto::Aes128::new(ks)?;
    let mut pt = KVec::with_capacity(ct.len(), GFP_KERNEL)?;
    for (i, chunk) in ct.chunks(16).enumerate() {
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(in_riv);
        iv[12..].copy_from_slice(&seq.wrapping_add(i as u32).to_be_bytes());
        let ksb = cipher.encrypt_block(&iv);
        for (j, &c) in chunk.iter().enumerate() {
            pt.push(c ^ ksb[j], GFP_KERNEL)?;
        }
    }
    Ok(pt)
}
/// If `wire` is an EDID reply (dock->host EP84, `type=4 sub=0x45`, inner
/// `id=0x194 sub=0x21`), decrypt it and return the embedded EDID blob (base block +
/// extensions). The EDID begins at inner offset 22; its total length is
/// `128 * (1 + extension_count)`, where the extension count is base-block byte 126.
/// Returns `None` for any other frame. See docs/CONTROL-PLANE.md.
///
/// **2026-07-17: try every IN riv variant, not just the head-0 raw one.** This used to call
/// `open_in` with a single fixed riv (`in_riv(out_riv)`), while the rest of this file's dock->
/// host decoders (`verify_in_ack`, `decode_any`) already try FOUR variants -- the IN-riv model
/// has drifted across dock firmware (old captures decrypt IN under `delivered^0x04` unchanged;
/// current `live-3426` firmware uses `delivered^0x01`) and multi-head replies additionally flip
/// `byte0^0x80`. A live test (50 retries, DLM-shaped requests, generous heartbeat-interleaved
/// spacing) got zero `id=0x194`/`id=0x114` hits under the single old variant while every dock
/// reply logged by `decode_any` (which DOES try all four) also came back generic -- consistent
/// with either a genuinely mute dock OR a real EDID/placeholder reply landing under a riv
/// variant this function never checked. Try all four so a real reply can no longer be missed
/// purely by riv-variant mismatch.
pub(super) fn parse_edid_from_reply(
    ks: &[u8; 16],
    out_riv: &[u8; 8],
    wire: &[u8],
) -> Result<Option<KVec<u8>>> {
    // Wire header: [.. type@4 u32 .. sub@8 u16 .. seq@12 u32]; body at off16.
    if wire.len() <= 16 || u16::from_le_bytes([wire[8], wire[9]]) != 0x45 {
        return Ok(None);
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let body = &wire[16..];
    let raw = in_riv(out_riv);
    let mut flipped = raw;
    flipped[7] ^= 0x01;
    for base in [raw, flipped] {
        for h in 0..2u8 {
            let mut riv = base;
            if h == 1 {
                riv[0] ^= 0x80;
            }
            let Ok(inner) = open_in(ks, &riv, seq, body) else {
                continue;
            };
            // Inner header: [id u16][sub u16][counter u16][00 00]; EDID payload at off22.
            const EDID_OFF: usize = 22;
            if inner.len() < EDID_OFF + 128 {
                continue;
            }
            let id = u16::from_le_bytes([inner[0], inner[1]]);
            let sub = u16::from_le_bytes([inner[2], inner[3]]);
            // The get-EDID reply id is `0x194` on the wire (CP-HANDSHAKE.md sec 4f,
            // ground-truthed against the cold-ref capture); older notes wrote the low byte
            // `0x94` alone. Accept both so a real `0x194` reply is not silently dropped.
            if (id != 0x94 && id != 0x194) || sub != 0x21 {
                continue;
            }
            let edid = &inner[EDID_OFF..];
            // Validate the EDID base-block magic `00 FF FF FF FF FF FF 00`.
            const MAGIC: [u8; 8] = [0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00];
            if edid[..8] != MAGIC {
                continue;
            }
            let total = ((1 + edid[126] as usize) * 128).min(edid.len());
            let mut out = KVec::with_capacity(total, GFP_KERNEL)?;
            out.extend_from_slice(&edid[..total], GFP_KERNEL)?;
            return Ok(Some(out));
        }
    }
    Ok(None)
}
/// Decode an `id=0x0044 sub=0x0020` EDID-readiness probe reply (dock's answer to
/// [`get_edid_req_sub`]'s `sub=0x20`) and report whether the dock's downstream DDC/EDID read has
/// actually finished. **Found 2026-07-17 (2nd pass)**, superseding [`edid_poll_progress`]'s
/// cruder non-zero-byte-count heuristic with an exact bit: re-diffing every `sub=0x0020` reply in
/// `full-session-trace1` against which probe immediately precedes a real vs placeholder fetch
/// shows inner content byte offset 26 (i.e. `inner[26]`) is `0x00` while the dock is still
/// working and flips to `0x80` in the probe immediately before BOTH of the trace's real
/// `id=0x194` fetches (ictr 43->44 stays `0x00` -> placeholder; ictr 119->120 and 121->122 are
/// both `0x80` -> real EDID) -- and reverts to `0x00` in later steady-state probes once the dock
/// has nothing new pending (ictr 180, 182). Bytes 22..26 (`05 11/01 27/61 00`) also toggle
/// between two 2-byte codes (`0x6101` = idle/no-attempt-yet, `0x2711` = actively negotiating)
/// depending on session phase, but offset 26 alone is the readiness bit that matters here.
/// Returns `None` for any frame that isn't a decryptable `sub=0x0020` reply, so the caller can
/// tell "not this message class" apart from "this message class, not ready yet" (`Some(false)`).
pub(super) fn edid_poll_ready(ks: &[u8; 16], out_riv: &[u8; 8], wire: &[u8]) -> Option<bool> {
    if wire.len() <= 16 || u16::from_le_bytes([wire[8], wire[9]]) != 0x45 {
        return None;
    }
    let seq = u32::from_le_bytes([wire[12], wire[13], wire[14], wire[15]]);
    let body = &wire[16..];
    let raw = in_riv(out_riv);
    let mut flipped = raw;
    flipped[7] ^= 0x01;
    for base in [raw, flipped] {
        for h in 0..2u8 {
            let mut riv = base;
            if h == 1 {
                riv[0] ^= 0x80;
            }
            let Ok(inner) = open_in(ks, &riv, seq, body) else {
                continue;
            };
            if inner.len() < 27 {
                continue;
            }
            let id = u16::from_le_bytes([inner[0], inner[1]]);
            let sub = u16::from_le_bytes([inner[2], inner[3]]);
            if id != 0x44 || sub != 0x20 {
                continue;
            }
            return Some(inner[26] & 0x80 != 0);
        }
    }
    None
}
