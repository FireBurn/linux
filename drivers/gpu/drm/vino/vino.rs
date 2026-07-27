// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (C) 2026 Mike Lothian

//! Vino -- open in-kernel Rust driver for DisplayLink DL3 docks (Dell D6000, ...).
//!
//! This is an `[RFC]` work-in-progress, posted to ask for help. It is a clean-room
//! reverse-engineered replacement for the proprietary DisplayLinkManager userspace
//! daemon + the EVDI kernel module, written natively in Rust against the in-tree USB,
//! crypto and DRM/KMS bindings (the prerequisite binding patches are posted as their
//! own series).
//!
//! # What works
//!
//! On probe the driver runs, all on real hardware (Dell Universal Dock D6000):
//! - the plaintext connect handshake over the Rust USB bulk + control transfer API;
//! - the clean-room HDCP 2.2 AKE / LC / SKE -- H', L' and V' all verify against the
//!   dock, so the session key `ks` is established and shared;
//! - the AES-CTR + AES-CMAC ("Dl3Cmac") control-plane seal, byte-exact against the
//!   reference daemon's captured wire;
//! - the plaintext `type=2 sub=0x24` stream-open arm marker; and
//! - registration of a real `struct drm_device` (see [`drm_sink`]) via the simple
//!   display pipe, so the dock appears to userspace as a mode-settable GEM/dumb DRM
//!   card, with a live EP08 framebuffer-scanout hook on every page-flip.
//!
//! # What does NOT work -- the wall (help wanted)
//!
//! After the arm marker the driver sends the first encrypted control-plane frame
//! (msg0) and the dock **never acknowledges it** (`wsub=0x45` ack count stays 0), so
//! the CP cipher never engages and no pixels ever flow. Every host-observable channel
//! has been matched to the reference daemon -- the bulk wire is byte-identical through
//! the arm + msg0, the AKE verifies, the seal/MAC/IV are byte-exact, the full EP0
//! control-transfer set matches, the endpoint set matches, the arm timing is tighter
//! than the daemon's -- and the dock still silently drops our encrypted CP while it
//! engages the daemon's. The gate appears to be something not visible on the host wire
//! (dock-internal session state, or a whole-bus timing/ordering property a per-channel
//! diff cannot see). **If you know the DL3 / DisplayLink control-plane engagement
//! sequence, or have ideas for the remaining paired full-bus diff, please help.**
//!
//! Note: the plaintext capability-announce IS the HDCP AKE. `run_ake` emits it once as the
//! ctr=1..8 cap frames (hello + AKE_Init..Stream_Manage, all `type=4 sub=0x04`), interleaved with
//! the dock's cert/H'/L'/V' replies, exactly as DLM's wire does; `send_cp_setup` then sends only
//! the encrypted arm + msg0 + burst. Every field/counter is ground-truthed against a real engaged
//! capture -- no captured skeleton is replayed and the AKE is never sent twice.
//!
//! Device: VID 0x17e9 (DisplayLink) / PID 0x6006 (Dell Universal Dock D6000).

use kernel::{
    alloc::flags::GFP_KERNEL,
    alloc::Flags,
    bindings,
    device::{self, Core},
    drm,
    error::code::{EINVAL, ENODEV},
    prelude::*,
    sync::{aref::ARef, new_mutex, Arc, Mutex},
    time::{
        delay::{fsleep, udelay},
        Delta, Instant, Monotonic,
    },
    usb,
    workqueue::{self, impl_has_work, new_work, Work, WorkItem},
};

/// DisplayLink vendor id.
const VID_DISPLAYLINK: u16 = 0x17e9;
/// Dell Universal Dock D6000 (DL3 family) product id.
const PID_D6000: u16 = 0x6006;

// Refcounted vino-bound USB interfaces, indexed by `bInterfaceNumber`. The sysfs `remove_all`
// path takes these references out of the registry before calling `device_release_driver()`, so a
// concurrent physical disconnect cannot invalidate the target and the disconnect callback can
// safely lock the now-empty registry without recursion.
kernel::sync::global_lock! {
    // SAFETY: initialized exactly once in `VinoModule::init`, before any `probe()` can run.
    unsafe(uninit) static VINO_IFACES: Mutex<[Option<ARef<usb::Interface>>; 8]> =
        [const { None }; 8];
}

/// Bitmask (bit N = `bInterfaceNumber` N) of every USB interface `probe()` has been offered, reset
/// when interface 0 (re)probes -- i.e. at the start of a fresh bring-up. `bring_up` polls this AFTER
/// the plaintext session-init so the HDCP AKE does not start until the dock's audio/HID interfaces
/// (5 & 6) have been enumerated/claimed: the session-init triggers the dock's billboard->function
/// switch, and while the kernel enumerates those interfaces the dock stalls its CP/EP84 channel
/// ~100 ms mid-AKE (measured vs the DLM cold refs). DLM never hits this -- as a userspace daemon it
/// only opens the device after the kernel has claimed everything. See `bring_up`.
static PROBED_IFACES: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Sticky "the dock's `id=0x0b sub=0x84` terminal cap-complete push was seen this session" flag. Set
/// by whichever EP84 reader encounters it -- `pace_cap_ack` (draining the ctr7/ctr8 acks) OR
/// `wait_cap_complete` -- and reset when interface 0 (re)probes. FIX 2026-07-14: the dock races the
/// `0x0b` push against the ctr8 echo; when `0x0b` arrives first, `pace_cap_ack(8)` consumed+discarded
/// it (it only matches `id=0x14 ctr=8`), so `wait_cap_complete` never saw it, drained its full ~808 ms
/// budget, and fired the arm ~800 ms late. This shared flag stops the terminal push from being lost
/// to whichever reader happens to pull it off the wire first.
static SAW_0B: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Consecutive-reconnect threshold at which [`VinoDriver::probe`] gives up on interface 0 and
/// refuses to start a fresh bring-up -- see [`PROBE_LOOP`]. Added 2026-07-17 at the user's
/// request while away from the machine and unable to pull the dock's power to break a loop by
/// hand: the 2026-07-16 incident (`CLAUDE.md`) showed the dock can disconnect/reconnect right as
/// vino touches the video endpoints, and `probe()` re-running the whole AKE/CP dance on
/// re-enumeration re-triggers the same fault -- a "clean probe()-rerun loop (not a freeze)" that
/// previously only stopped when the user physically unplugged the dock.
const LOOP_THRESHOLD: u32 = 4;
/// Interface-0 reprobes spaced less than this apart count toward [`LOOP_THRESHOLD`]; spaced
/// further apart (e.g. the user manually replugging during normal testing) reset the count to 1,
/// so this does not fire on ordinary unplug/replug cycles.
const LOOP_WINDOW: Delta = Delta::from_secs(15);

kernel::sync::global_lock! {
    // SAFETY: initialized exactly once in `VinoModule::init`, before any `probe()` can run.
    unsafe(uninit) static PROBE_LOOP: Mutex<ProbeLoopState> = ProbeLoopState {
        last: None,
        count: 0,
        tripped: false,
    };
}

/// State behind the interface-0 reconnect-loop breaker; see [`PROBE_LOOP`] and
/// [`record_probe_and_check_loop`]. Once `tripped`, it stays tripped (surviving further probes)
/// until cleared via `/sys/devices/vino/probe_loop_tripped` (`VinoControl`) or a module reload.
struct ProbeLoopState {
    last: Option<Instant<Monotonic>>,
    count: u32,
    tripped: bool,
}

/// Record one interface-0 probe against [`PROBE_LOOP`] and return whether the loop breaker is
/// (now, or already) tripped. Called from [`VinoDriver::probe`] before any USB I/O for this plug
/// is attempted, so a tripped result can skip bring-up entirely -- no AKE, no CP, no video writes,
/// nothing left for vino to re-trigger the dock's disconnect with.
fn record_probe_and_check_loop() -> bool {
    let mut st = PROBE_LOOP.lock();
    let now = Instant::<Monotonic>::now();
    let rapid = st.last.is_some_and(|last| now - last < LOOP_WINDOW);
    st.last = Some(now);
    st.count = if rapid { st.count + 1 } else { 1 };
    if st.count >= LOOP_THRESHOLD {
        st.tripped = true;
    }
    st.tripped
}

/// Control + per-head bulk endpoints (guide sec 2).
const EP_CTRL_OUT: u8 = 0x02;
pub(crate) const EP_CTRL_IN: u8 = 0x84;
/// Interrupt IN status endpoint, polled once during bring-up.
const EP_STATUS_IN: u8 = 0x83;

/// The dock's endpoints, resolved once against interface 0's descriptor and validated for
/// direction and transfer type by [`usb::Interface::endpoint`].
///
/// Resolving up front means the rest of the driver names an endpoint by what it *is* rather than
/// by a bare address, so a bulk-OUT transfer cannot be aimed at the interrupt-IN status endpoint.
/// The whole set is [`Copy`] (each entry is an address plus a max-packet size), so it can be
/// carried by value in a [`UsbLink`] rather than borrowed.
#[derive(Clone, Copy)]
pub(crate) struct Endpoints {
    /// EP02: host->dock control-plane bulk writes.
    pub(crate) ctrl_out: usb::Endpoint<usb::BulkOut>,
    /// EP84: dock->host control-plane bulk replies.
    pub(crate) ctrl_in: usb::Endpoint<usb::BulkIn>,
    /// EP83: dock->host interrupt status. Absent on docks that do not advertise it.
    pub(crate) status_in: Option<usb::Endpoint<usb::InterruptIn>>,
    /// Per-head video bulk-OUT endpoints ([`drm_sink::VIDEO_EPS`]).
    pub(crate) video: [usb::Endpoint<usb::BulkOut>; drm_sink::HEADS],
}

impl Endpoints {
    /// Resolves every endpoint the driver uses against `intf`'s active alternate setting.
    ///
    /// The control endpoints and the video endpoints are required; the interrupt status endpoint
    /// is optional because only the bring-up probe reads it.
    pub(crate) fn resolve<Ctx: device::DeviceContext>(intf: &usb::Interface<Ctx>) -> Result<Self> {
        let mut video = [intf.endpoint::<usb::BulkOut>(drm_sink::VIDEO_EPS[0])?; drm_sink::HEADS];
        for (slot, addr) in video.iter_mut().zip(drm_sink::VIDEO_EPS).skip(1) {
            *slot = intf.endpoint::<usb::BulkOut>(addr)?;
        }

        Ok(Self {
            ctrl_out: intf.endpoint::<usb::BulkOut>(EP_CTRL_OUT)?,
            ctrl_in: intf.endpoint::<usb::BulkIn>(EP_CTRL_IN)?,
            status_in: intf.endpoint::<usb::InterruptIn>(EP_STATUS_IN).ok(),
            video,
        })
    }
}

/// A live USB transfer handle: an [`usb::Io`] token proving I/O is currently permitted, plus the
/// resolved [`Endpoints`].
///
/// This is what the driver passes around in place of the old `&Interface<Bound>`. Obtaining one
/// requires the device's [`usb::IoWindow`] to still be open, so a transfer cannot be issued after
/// `disconnect()` has closed it; and because the endpoints travel with it, the control and video
/// paths read as `link.ctrl_send(..)` / `link.video_send(head, ..)` rather than repeating a raw
/// endpoint address at each call.
pub(crate) struct UsbLink<'a> {
    window: &'a Arc<usb::IoWindow>,
    io: usb::Io<'a>,
    eps: Endpoints,
}

impl<'a> UsbLink<'a> {
    /// Opens a link on `window`, failing with `ENODEV` once the window has been closed.
    pub(crate) fn open(window: &'a Arc<usb::IoWindow>, eps: Endpoints) -> Result<Self> {
        Ok(Self {
            io: window.enter()?,
            window,
            eps,
        })
    }

    /// Opens the persistent EP84 control-plane reader.
    pub(crate) fn ctrl_in_queue(&self, depth: usize, buf_len: usize) -> Result<usb::BulkInQueue> {
        usb::BulkInQueue::new(self.window, &self.io, &self.eps.ctrl_in, depth, buf_len)
    }

    /// Opens the pipelined EP02 control-plane writer.
    pub(crate) fn ctrl_out_queue(&self, depth: usize, buf_len: usize) -> Result<usb::BulkOutQueue> {
        usb::BulkOutQueue::new(self.window, &self.io, &self.eps.ctrl_out, depth, buf_len)
    }

    /// Opens the pipelined video writer for `head`.
    pub(crate) fn video_queue(
        &self,
        head: usize,
        depth: usize,
        buf_len: usize,
    ) -> Result<usb::BulkOutQueue> {
        let ep = self.eps.video.get(head).ok_or(EINVAL)?;
        usb::BulkOutQueue::new(self.window, &self.io, ep, depth, buf_len)
    }

    /// The underlying I/O token, for the paths that open a persistent queue.
    pub(crate) fn io(&self) -> &usb::Io<'a> {
        &self.io
    }

    /// Writes a control-plane message to EP02.
    pub(crate) fn ctrl_send(&self, data: &[u8], timeout: Delta, gfp: Flags) -> Result<usize> {
        self.io.bulk_send(&self.eps.ctrl_out, data, timeout, gfp)
    }

    /// Reads a control-plane reply from EP84.
    pub(crate) fn ctrl_recv(&self, data: &mut [u8], timeout: Delta, gfp: Flags) -> Result<usize> {
        self.io.bulk_recv(&self.eps.ctrl_in, data, timeout, gfp)
    }

    /// Clears a stall on EP02.
    pub(crate) fn clear_ctrl_halt(&self) -> Result {
        self.io.clear_halt(&self.eps.ctrl_out)
    }

    /// Clears a stall on `head`'s video endpoint.
    pub(crate) fn clear_video_halt(&self, head: usize) -> Result {
        self.io.clear_halt(self.eps.video.get(head).ok_or(EINVAL)?)
    }

    /// Reads the interrupt status endpoint, if the dock advertises one.
    pub(crate) fn status_recv(&self, data: &mut [u8], timeout: Delta, gfp: Flags) -> Result<usize> {
        let ep = self.eps.status_in.as_ref().ok_or(ENODEV)?;
        self.io.interrupt_recv(ep, data, timeout, gfp)
    }

    /// Issues a control OUT transfer on EP0.
    pub(crate) fn control_send(
        &self,
        request: u8,
        request_type: u8,
        value: u16,
        index: u16,
        data: &[u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result {
        self.io
            .control_send(request, request_type, value, index, data, timeout, gfp)
    }

    /// Issues a control IN transfer on EP0.
    pub(crate) fn control_recv(
        &self,
        request: u8,
        request_type: u8,
        value: u16,
        index: u16,
        data: &mut [u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result {
        self.io
            .control_recv(request, request_type, value, index, data, timeout, gfp)
    }

    /// Selects an alternate setting on the driver's own interface.
    pub(crate) fn set_alternate_setting(&self, alternate: u8) -> Result {
        self.io.set_alternate_setting(alternate)
    }
}
/// EP84 (dock->host) URB size. DLM consistently posts one 4096-byte read; replies larger than one
/// URB are delivered as consecutive fragments rather than by changing the request length.
const EP84_BUF: usize = 4096;
/// Number of IN URBs kept perpetually posted on EP84 by the async reader
/// ([`usb::Interface::bulk_in_queue`]); `depth - 1` stay outstanding while one is serviced.
///
/// MEASURED 2026-06-27 (`WinCap` IRP pairing + usbmon S/C pairing): **both** reference drivers run
/// EP84 at an outstanding depth of exactly **1** -- Windows DLM (USBPcap `max_outstanding_depth=1`)
/// and Linux DLM (usbmon `max_depth=1`) post one IN read, wait for the dock's reply, then re-post.
/// It is an "always one posted" reader, not a deep queue. vino was the only implementation running
/// depth=4, so its EP84 IN-token/NAK cadence differed from every engaging driver. Match them
/// exactly (1) -- still always-posted (re-armed on completion) but with no extra concurrent reads.
///
/// (A speculative bump to 16 -- "absorb a burst of cap-phase ACKs" -- was reverted: it contradicts
/// the measured DLM cadence above, and with the cap-announce now PACED per frame the dock never
/// bursts more ACKs than one always-posted reader drains.)
/// Depth of the persistent EP84 IN reader.
///
/// **Raised 1 -> 4 on 2026-07-23 from a cadence survey of the whole capture corpus**
/// (`scripts/cadence-survey.py`). Depth alone was never the difference -- DLM's EP84 max-in-flight
/// is also 1 -- but the DUTY CYCLE was: measuring what fraction of wall time an EP84 IN URB is
/// actually outstanding gives DLM 36-100% (median ~80%) and vino 3-22%. DLM posts a read and
/// leaves it pending for the whole ~16 ms cycle; vino posted one with an 8 ms timeout only around
/// each send, so for most of the session the dock had NOWHERE to push an asynchronous event.
/// With depth 4 one URB stays posted while the others are reaped and resubmitted.
const EP84_QUEUE_DEPTH: usize = 4;

/// USB transfer timeout used during bring-up.
fn timeout() -> Delta {
    Delta::from_millis(1000)
}

/// Short timeout for draining the dock's per-message CP reply on EP84 after a runtime `send_cp`
/// (see `drm_sink::VinoDrmData::send_cp`). DLM keeps EP84 read in lockstep with EP02; not every
/// message elicits a reply, so this is deliberately short -- a NAK/timeout just means "nothing to
/// drain this time" and must not stall the scanout/keepalive path.
pub(crate) fn cp_reply_timeout() -> Delta {
    Delta::from_millis(8)
}

/// Ceiling for the REACTIVE HDCP per-phase wait, applied at each CP-setup phase boundary (after the
/// plaintext init group, and after each per-head AKE block). The all-bus timing diff showed DLM is
/// The **downstream-HDCP H' compute wait** the dock enforces inside each per-head AKE restatement.
///
/// **MEASURED 2026-07-20** from the fully-decoded cold DLM capture (`edid-cold-decrypt-20260719`,
/// `scratchpad/decode_edid.py` + EP02 frame timestamps): across the whole 668 ms CP setup there is
/// exactly ONE hold per head, and it sits at a specific spot -- **160.1/160.2 ms between
/// `AKE_No_Stored_km` (`id=0x9a`, msg-id 0x04) and `LC_Init` (`id=0x22`, msg-id 0x09)**. That is
/// textbook HDCP 2.2: after receiving `AKE_No_Stored_km` the receiver takes up to ~200 ms to compute
/// and return `AKE_Send_H_prime`, and the transmitter must not send `LC_Init` before it. Everywhere
/// else DLM sends tight (1-30 ms). There is NO hold before the per-head loop and NO hold between
/// heads (those transitions are 7-8 ms) -- an earlier "155 ms at every phase boundary" reading was
/// wrong, and a reactive drain-until-response is worse still: the dock ACKs each message immediately
/// and only *then* computes H' silently, so "wait for the dock to break silence" returns on the
/// instant ack and skips the compute window entirely (HW 2026-07-20: 0 rich `id=0x44`, vs ~40% with a
/// blind hold). vino sends `LC_Init` only ~11 ms after `No_Stored_km`, so the dock's downstream
/// repeater auth never completes -> it engages the CP cipher (needs only the seal key) but withholds
/// the EDID/DDC path (bare `id=0x14`, never rich `id=0x44`). This blind hold, measured send-to-send
/// after `No_Stored_km`, reproduces DLM's exact cadence. 165 ms = DLM's 160 ms + a small margin.
const HDCP_HPRIME_WAIT_US: i64 = 165_000;

/// Impersonate DLM's **fixed-timer** bring-up fingerprint instead of vino's reactive pacing.
///
/// The 2026-06-25 step-timing survey (`captures/step-timing-survey-20260625.md`) across 9 DLM
/// and 29 vino plugs showed DLM does NOT react to the wire -- it uses hardcoded sleeps, so its
/// pre-arm milestones are tight constants: `cp_first->cert` ~1.3 ms, `cert->arm` ~156 ms (current
/// 3.4.26 firmware; ~59 ms on pre-3.4.26), `arm->msg0` ~0.17 ms. vino's reactive settle scatters
/// those (cert->arm 57-292 ms) and, at
/// the one step we can measure precisely, fires msg0 ~0.07 ms after the arm -- ~2x FASTER than
/// DLM, the only consistent timing *inversion* in the whole corpus and a never-tested variable.
///
/// With this on, vino reproduces DLM's fingerprint as closely as the host allows:
///   - the pre-AKE stale-EP84 flush probes at 1 ms (front gap ~1 ms, like DLM) -- see `run_ake`;
///   - it holds the arm marker to a *fixed* [`CERT_TO_ARM_US`] after the cert (~156 ms, like
///     DLM) instead of arming reactively the instant the AKE settles -- see `send_cp_setup`;
///   - it holds [`ARM_TO_MSG0`] between the arm marker and msg0 (~0.17 ms, like DLM);
///   - it logs the realised `cp_start->arm`, `cert->arm` and the hold so the next cold plug's
///     dmesg reports the actual fingerprint for an A/B against DLM.
///
/// The cert->arm hold is the key one: the 2026-07-13 cold references showed DLM's `cert->arm` is
/// ~156 ms +-0.1 ms across 3 plugs on the current firmware (a hard sleep spanning the ~152.6 ms
/// downstream-HDCP pause) while vino's reactive settle can arm as early as ~57.9 ms -- deep inside
/// that pause, before the dock has sent ctr5..8. The fixed hold closes that and makes the step
/// deterministic. (Note the reactive [`wait_cap_complete`] normally lands the arm at ~cert+155 ms
/// on its own; the fixed hold is the floor that catches its failure modes.)
///
/// **2026-07-17: flipped to `false` for re-test.** The whole premise of this flag was "maybe
/// engagement is gated on matching DLM's exact wire timing" -- a hypothesis from the pre-msg0-fix
/// era when every host-observable channel had been exhausted and timing was one of the few
/// remaining untried variables. It is now known the actual gate was a single wrong AES-CTR nonce
/// byte (`cp::dl3cmac_tag`, `CLAUDE.md` 2026-07-16), unrelated to timing entirely -- the dock
/// engages (`sub=0x45_acks>0`) with vino's reactive pacing exactly as readily as with this fixed
/// fingerprint. Left in the tree (default OFF) as a documented A/B knob rather than deleted, in
/// case a future firmware revision turns out to care about timing after all -- but reactive
/// pacing is simpler, arms sooner (~58ms vs ~156ms), and is now the default.
const DLM_FIXED_TIMERS: bool = false;

/// Mimic the Windows DisplayLink driver's pre-arm control choreography instead of the Linux/libusb
/// DLM one (2026-06-27, `WinCap/WINCAP-ANALYSIS.md`). The USBPcap traces of three engaging Windows
/// sessions on this exact dock (`bcdDevice=0x3159`) showed a *leaner* device-open than DLM's libusb
/// stack, on three concrete, observable axes:
///   1. ONE device-open vendor-IN read -- only `0xc1 0xfe wIdx=1` (the 16 B "RidgeDock" blob).
///      Windows never issues the `0xfc`/`0xfd`/`0xfb` DFU reads vino picked up from the DLM oracle.
///   2. NO libusb descriptor-burst at open (the CONFIG 618x3/40x3 + STRING 255x22 replay): Windows
///      runs over the already-enumerated device and uses cached descriptors, exactly like a native
///      kernel driver -- so [`CP_LIBUSB_OPEN_ENUM`] is forced OFF.
///   3. The pre-session-init `0x40 bReq=0x24` vendor-OUT uses **wValue=0** (Windows), not wValue=3
///      (Linux DLM). Both DLM and Windows send just ONE 0x24 here; DLM's *other* 0x24 (wValue=0) is
///      a post-engagement command issued after the first dock ack (see the post-msg0 0x24 site).
/// The analysis already proved none of these is the CP gate (Linux DLM engages WITH the libusb burst
/// and wValue=3; Windows engages WITHOUT them) -- so this is a "just in case" A/B, not a fix. It is
/// the smallest set of changes that makes vino's EP0 pre-arm stream resemble Windows'. The cap-
/// announce / cert-req framing is left alone (changing it risks the byte-exact seal, and both DLM's
/// 7-descriptor form -- which vino matches -- and Windows' 6-descriptor form engage). Flip to `false`
/// to restore the DLM/libusb behaviour for a clean paired-vs-DLM diff.
///
/// **2026-07-17: flipped to `false` for re-test.** This was already documented as "not a fix, just
/// in case" when written -- a wall-chasing A/B from before the real gate (a nonce byte, unrelated
/// to any of this) was found. With msg0 fixed, there's no open question left for it to answer;
/// reverting to the DLM/libusb pre-arm shape, which is the one that's actually been HW-proven to
/// engage across dozens of reloads this project, rather than carrying the Windows deviation
/// forward as an untested silent default.
const WINDOWS_MIMIC: bool = false;

/// DLM's `cert->arm` hold on the CURRENT firmware (3.4.26): **~156 ms**, [156.8..157.0] over the
/// three 2026-07-13 cold + engaged references (`captures/dlm-cold-3426-20260713-*`, cross-validated
/// on usbmon and the xHCI TRB trace -- see `gemini.md`). This spans the dock's ~152.6 ms
/// downstream-HDCP pause (which runs cert+0.8 ms .. cert+153.6 ms) plus the trailing ctr5..8 cap
/// exchange; DLM arms *after* both. The old 59.1 ms value was a **pre-3.4.26-firmware** (2026-06-26)
/// measurement -- 95 ms too early for this firmware, i.e. mid-pause, before the dock has sent ctr5..8.
/// Under [`DLM_FIXED_TIMERS`], `send_cp_setup` holds the arm until this long after `Session::cert_at`.
/// This is a *floor*: [`wait_cap_complete`] normally gates the arm reactively on the dock's id=0x0b
/// terminal push (~cert+155 ms) and lands first; the floor only binds if that reactive path fails
/// (a `saw_0b` miss, or the quiet fallback firing early), in which case ~156 ms is the correct
/// fallback -- 59 ms would have armed into the downstream-HDCP pause.
const CERT_TO_ARM_US: i64 = 156_000;

/// DLM's `ctr2->ctr3` gap (AKE_Transmitter_Info -> AKE_No_Stored_km): consistently **~1.65 ms**
/// over 6 cold plugs [1.55..1.89], vs vino's **~0.30 ms** (5x faster, and rock-steady across both
/// corpora -- 2026-06-26 per-message latency analysis vs the same-day engaging DLM baseline). This
/// is the window where a real HDCP transmitter **verifies the receiver's DCP-signed certificate**
/// (RSA-1024 signature check + revocation) before wrapping `km`; vino skips it -- it only
/// RSA-OAEP-encrypts `km`, which is why it is so much faster. It is the first consistent,
/// host-reachable behavioural divergence from DLM found since the host was declared "exhausted",
/// and it fits the wall's evidence box (invisible to a passive byte diff; DLM satisfies it for
/// free; a repeater could time it as a locality-style "did you actually validate me?" check).
/// Under [`DLM_FIXED_TIMERS`], hold `ctr2->ctr3` to this so vino spends a realistic cert-verify
/// time instead of answering impossibly fast.
const CERT_VERIFY_HOLD_US: i64 = 1650;

/// DLM's median `arm->msg0` gap (the survey: 0.152 / 0.188 ms on the two clean DLM cold plugs).
/// vino naturally fires msg0 ~0.07 ms after the arm; hold to match DLM when [`DLM_FIXED_TIMERS`].
const ARM_TO_MSG0: Delta = Delta::from_micros(170);

/// EXPERIMENT (2026-07-15), SETTLED: the msg0 inner `content[22..32]` is a 10-byte token vino
/// fills with the kernel CSPRNG. It is host-arbitrary/dock-ignored (never echoed; a fresh
/// decrypted byte-diff vs DLM's engaged msg0 showed content[0..22] already byte-identical and
/// only this token differs -- and it also differs between vino's own two sessions). This flag let
/// the token be pinned to a real DLM-captured value to test whether the dock content-validates
/// it. **2026-07-16: settled by the real fix** -- the actual engagement gate was an unrelated
/// AES-CTR nonce byte (`cp::dl3cmac_tag`), and live cold plugs since then engage every time with
/// this back at `None` (fresh CSPRNG token, never the hardcoded replay) -- conclusively confirming
/// the token was never the gate. Reverted to `None`; do not pin this to a stale captured value
/// again outside a deliberate one-off A/B (a fixed token is also just bad practice to leave as a
/// standing default -- predictable content on every session for no benefit).
const MSG0_TOKEN_OVERRIDE: Option<[u8; 10]> = None;

/// Per-frame send pads that reproduce DLM's `cp_first->cert` cadence (2026-06-26 frame-by-frame
/// diff of the plaintext session-init + AKE_Init burst). With the cold-plug flush removed, vino's
/// sync `bulk_send`s fire back-to-back ~0.37 ms QUICKER than DLM's pipelined libusb URBs, so
/// `cp_first->cert` was 0.64 ms vs DLM's 1.07 ms. Each pad is the measured per-gap deficit
/// (vino gap -> DLM gap): init_0->init_25 0.065->0.144, init_25->init_4 0.194->0.372,
/// session-init-ACK->AKE_Init 0.043->0.159. Applied as `udelay` (calibrated busy-wait, us-precise,
/// unlike `fsleep`'s slack) only under [`DLM_FIXED_TIMERS`]. Sum 0.373 ms lands `cp_first->cert`
/// on DLM's ~1.07 ms.
const PAD_INIT0_TO_INIT25_US: i64 = 79;
const PAD_INIT25_TO_INIT4_US: i64 = 120;
const PAD_ACK_TO_AKEINIT_US: i64 = 90;

/// FALLBACK timeout (5 s) for `bring_up`'s reactive wait on the dock's composite enumeration. The
/// wait itself is on [`PROBED_IFACES`] -- it blocks the HDCP AKE until interfaces 5 & 6 have been
/// enumerated/claimed; this value only caps how long it waits if 5/6 never appear (then it proceeds
/// and logs the timeout). Background (2026-07-14, quantified vs the fresh DLM cold refs): vino is an
/// in-kernel interface-0 driver whose `bring_up` starts the AKE while the kernel is still enumerating
/// the dock's audio/HID interfaces (5 & 6), which the dock's session-init-triggered billboard->function
/// switch presents; during that ~100 ms enumeration the dock stalls the CP/EP84 channel mid-AKE
/// (AKE_Init->first dock reply = 100 ms vs DLM's 0.1 ms). DLM never hits it -- as a userspace daemon
/// it only opens the device after the kernel has claimed all 7 interfaces. Two blind timing holds
/// (150 ms before and after session-init) did NOT collapse the stall, so this is the reactive fix the
/// data pointed to. See gemini.md.
const ENUM_RESPONSE_HOLD_US: i64 = 5_000_000;

/// Hold until `anchor` is at least `target_us` old, to microsecond precision. A plain `fsleep`
/// overshoots a wall-clock target by its timer slack (~0.3 ms observed on the cert->arm hold); so
/// `fsleep` the bulk of the wait (cheap -- it must not
/// busy-burn ~1 ms of CPU) leaving a margin, then re-measure and `udelay` the exact residual to
/// hit `target_us` on the nose. Never returns before `target_us` of `anchor` has elapsed; returns
/// immediately if it already has. Used to realise DLM's fixed pre-arm timer (`DLM_FIXED_TIMERS`).
fn hold_until(anchor: Instant<Monotonic>, target_us: i64) {
    /// Leave this much for the precise `udelay` tail; `fsleep`'s slack stays under it.
    const SPIN_MARGIN_US: i64 = 400;
    let now = anchor.elapsed().as_micros_ceil();
    if now >= target_us {
        return;
    }
    if target_us - now > SPIN_MARGIN_US {
        fsleep(Delta::from_micros(target_us - now - SPIN_MARGIN_US));
    }
    let now = anchor.elapsed().as_micros_ceil();
    if now < target_us {
        udelay(Delta::from_micros(target_us - now));
    }
}

/// Set once the dock has actually engaged the CP cipher (`wsub=0x45` acks > 0). EP08 video is
/// gated on it: pushing frames at a dock whose CP channel is dead makes it fault and USB-reset.
/// NOTE: with the current CP-engagement wall (see the file header) this is never set on real
/// hardware -- the dock runs the whole plaintext handshake but never engages the encrypted CP.
static CP_ENGAGED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Set once the bring-up work item finishes (AKE/CP attempt done). `detect` only connects the
/// live-scanout connector AFTER this, so a compositor enabling the output cannot start EP08
/// scanout on top of the still-running AKE on the same USB device.
/// Runs the continuous CP keepalive (see the loop in `BringUp::run`). Set when bring-up completes,
/// cleared by `disconnect()` so the loop exits promptly and cannot hang module unload. DLM keeps its
/// CP dialogue alive for the WHOLE session (~74 msgs/s, measured); a bounded keepalive lets the dock
/// time out and hard-reset the moment it stops, so this must run for the life of the device.
static KEEPALIVE_RUN: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Sticky "this binding is going away" flag: set by `disconnect()` on the control interface,
/// cleared only by a fresh `probe()` of it.
///
/// `KEEPALIVE_RUN` on its own cannot stop the keepalive, because `BringUp::run` *sets* it -- twice,
/// and both stores happen near the end of a bring-up that takes ~15 s. A disconnect landing while
/// bring-up is still running therefore cleared a flag the work item then put straight back. That is
/// exactly what wedged the machine on 2026-07-27: the dock reset mid-EDID-fetch, `disconnect()`
/// cleared `KEEPALIVE_RUN` at t=45.7 s, `BringUp::run` re-armed it at t=53.2 s and looped forever,
/// and `IoWindow::close()` -- which waits for the `Io` token that loop holds for its whole body --
/// never returned. `usb_hub_wq` stayed blocked in `hub_event` for the rest of the boot, so the
/// dock's USB3 half could never re-enumerate.
///
/// Being sticky is the whole point: nothing the outgoing session runs can un-set it.
static SESSION_TEARDOWN: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Arms the CP keepalive, unless this binding has already begun tearing down.
fn keepalive_arm() {
    use core::sync::atomic::Ordering::SeqCst;
    if !SESSION_TEARDOWN.load(SeqCst) {
        KEEPALIVE_RUN.store(true, SeqCst);
    }
}

/// The condition every keepalive/readiness loop spins on. Re-reading `SESSION_TEARDOWN` here, and
/// not only in [`keepalive_arm`], is what closes the race for good: an arm that slips past the
/// check there still drops out on the loop's very next iteration.
fn keepalive_running() -> bool {
    use core::sync::atomic::Ordering::Relaxed;
    KEEPALIVE_RUN.load(Relaxed) && !SESSION_TEARDOWN.load(Relaxed)
}

static BRINGUP_COMPLETE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

mod ake;
mod cp;
mod crypto;
mod hdcp;
mod proto;
mod rng;
mod video;
mod video_arm_content;

/// The shared secrets a completed HDCP 2.2 AKE leaves behind: the SKE session key
/// `ks` and content IV `riv` key the AES-CTR control plane (sec 6), and `kd` is kept
/// for any further repeater verification. Consumed by the Phase 2b/2c CP + video.
#[allow(dead_code)] // ks/riv/kd are consumed by the post-engagement CP stream (open blocker)
struct Session {
    ks: [u8; 16],
    riv: [u8; 8],
    kd: [u8; 32],
    /// Monotonic timestamp of the first CP frame (~`cp_first` in the timing survey), taken at
    /// the top of [`run_ake`]. Used by [`send_cp_setup`] to realise DLM's fixed pre-arm timer
    /// and to log the achieved `cp_start->arm` fingerprint. See [`DLM_FIXED_TIMERS`].
    cp_start: Instant<Monotonic>,
    /// Monotonic timestamp of the dock's `AKE_Send_Cert` push (`cert` in the timing survey),
    /// taken the instant [`run_ake`] receives it. DLM arms a *fixed* [`CERT_TO_ARM_US`] after
    /// this point (~156 ms on the current 3.4.26 firmware, +-0.1 ms over the 2026-07-13 cold refs --
    /// a hardcoded sleep spanning the downstream-HDCP pause), whereas vino's reactive settle can arm
    /// the moment the AKE completes (~57.9 ms -- deep inside that pause). [`send_cp_setup`] holds the arm to this offset under
    /// [`DLM_FIXED_TIMERS`] so vino never arms ahead of DLM's window. See [`DLM_FIXED_TIMERS`].
    cert_at: Instant<Monotonic>,
    /// The next inner `hdcp_seq` counter after the cap/AKE frames [`run_ake`] sent, so
    /// [`send_cp_setup`]'s msg0 continues the sequence without a hardcoded constant (hardcoded
    /// values caused past off-by-one regressions). DLM's current 3.4.26 cold plug sends the
    /// session-init ACK (ctr=1) then AKE_Init..Stream_Manage (ctr 2..8), leaving this = 9.
    next_ctr: u16,
    /// AKE inputs retained for the per-head **downstream repeater AKE** restatement
    /// ([`send_cp_setup`]'s per-head loop). The crypto is standard HDCP 2.2 (rr-confirmed 2026-07-17:
    /// the restatement `Ekpub` is a real RSA-1024 output, `V` a RepeaterAuth_Send_Ack) so vino
    /// recomputes a fresh, self-consistent chain per head from its proven primitives, reusing the
    /// dock's RSA public key (`modulus`/`exponent`, from its cert), the dock's `rrx`, and the
    /// receiver-ID list header the dock sent. See `docs/CP-PERHEAD-RESTATEMENT.md`. `rxid_list` is
    /// empty on the non-repeater path.
    modulus: [u8; 128],
    exponent: [u8; 3],
    rrx: [u8; 8],
    rxid_list: KVec<u8>,
}

/// Tally of one [`drain_ep84`](VinoDriver::drain_ep84) sweep. `reads` is every dock->host EP84
/// frame seen; `acks` counts only frames that DECRYPT to a valid CP header (genuine engagement);
/// `rejects` counts `wsub=0x45`-tagged frames that fail to decrypt (the dock talking but ignoring
/// our cipher). Separating the last two is what makes the summary honest: `reads>0, acks==0,
/// rejects>0` is the wall; `reads==0` is the dock gone silent.
#[derive(Default, Clone, Copy)]
struct Ep84Drain {
    reads: usize,
    acks: usize,
    rejects: usize,
    /// Set once a `sub=0x0020` EDID-readiness probe reply's ready bit
    /// (`cp::edid_poll_ready`) is seen `true`. Sticky within a sweep; the caller ORs
    /// sweeps together via `add` so it survives across the whole get-EDID retry loop.
    edid_ready: bool,
    /// The inner ctr of a per-head DISPLAY-CAP reply (`id=0x78 sub=0x30`) seen in this
    /// sweep, if any. The dock sends one such reply per head that has a monitor attached,
    /// echoing the ctr of that head's stream-open (`id=0x14 sub=0x30`) request -- so the
    /// caller matches this against each head's recorded stream-open ctr to know which
    /// heads have a display. Used for dual-monitor connector detection (see `send_cp_setup`).
    display_cap_ctr: Option<u16>,
    /// The dock's fresh per-head `rrx` (from its `id=0x10 sub=0x84` / inner msg-id `AKE_SEND_RRX`
    /// push, see [`cp::perhead_rrx`]) seen in this sweep, if any. The per-head AKE restatement must
    /// derive that head's `kd`/`edkey`/`V` from THIS rrx, not the stale main-AKE one -- otherwise the
    /// repeater auth silently fails and the dock withholds EDID. Captured across the per-head loop's
    /// drains and applied before the head's `SKE_Send_Eks`/`RepeaterAuth_Send_Ack` are built.
    perhead_rrx: Option<[u8; 8]>,
}

impl Ep84Drain {
    /// Fold another sweep's counts into this running total.
    fn add(&mut self, o: Ep84Drain) {
        self.reads += o.reads;
        self.acks += o.acks;
        self.rejects += o.rejects;
        self.edid_ready |= o.edid_ready;
        self.display_cap_ctr = self.display_cap_ctr.or(o.display_cap_ctr);
        self.perhead_rrx = self.perhead_rrx.or(o.perhead_rrx);
    }
}

mod drm_sink;

/// The USB driver itself. Stateless: everything per-binding lives in [`VinoBoundData`].
struct VinoDriver;

/// Per-bound-interface driver state.
///
/// Carries the DRM [`Registration`](drm::Registration), whose lifetime is tied to this bound
/// device, so unbinding unregisters the card through the accepted registration teardown rather
/// than a driver-local force-unplug.
struct VinoBoundData<'bound> {
    _intf: ARef<usb::Interface>,
    /// The USB I/O-permitted window, closed by `disconnect()` before it returns. Cloned into the
    /// DRM device data so the transport paths share it. `None` on the idle non-control
    /// interfaces, which never issue a transfer.
    io: Option<Arc<usb::IoWindow>>,
    /// The registered DRM card, dropped on unbind. `None` when DRM registration failed, or on the
    /// idle non-control interfaces. Also `disconnect()`'s handle on the DRM device data, so
    /// `shutdown()` runs whenever a card exists -- it must not be reached through `bringup`, which
    /// is additionally `None` if the work item could not be allocated.
    registration: Option<drm::Registration<'bound, drm_sink::VinoDrmDriver>>,
    /// Owned handle to the deferred bring-up work (control interface only). `disconnect()` takes
    /// the option under the mutex before synchronously cancelling the work and unplugging DRM.
    /// The mutex itself is heap-pinned because kernel locks must not move after initialization.
    bringup: Pin<KBox<Mutex<Option<Arc<BringUp>>>>>,
    /// The DDC/CI virtual I2C adapters (control interface only), dropped -> `i2c_del_adapter` on
    /// disconnect. Each adapter context owns an `ARef<VinoDrmDevice>`, so prompt driver-data drop
    /// is also required to release the DRM minor. The USB binding takes and drops driver data after
    /// this driver's synchronous `disconnect()` callback returns.
    _i2c: [Option<Pin<KBox<kernel::i2c::BusAdapter<drm_sink::VinoI2c>>>>; drm_sink::HEADS],
}

/// Deferred bring-up work item: the bring-up sequence run on the system workqueue instead
/// of inline in `probe()` (which would pin the driver-model probe thread on blocking USB
/// I/O while the card node is live). Holds a refcounted handle to the bound interface (and,
/// once the DRM sink exists, the DRM device), so they outlive `probe()`.
#[pin_data]
struct BringUp {
    ddev: ARef<drm_sink::VinoDrmDevice>,
    #[pin]
    work: Work<BringUp>,
}

impl_has_work! {
    impl HasWork<Self> for BringUp { self.work }
}

impl BringUp {
    fn new(ddev: ARef<drm_sink::VinoDrmDevice>) -> Result<Arc<Self>> {
        Arc::pin_init(
            pin_init!(BringUp {
                ddev,
                work <- new_work!("vino::bring_up"),
            }),
            GFP_KERNEL,
        )
    }
}

impl WorkItem for BringUp {
    type Pointer = Arc<BringUp>;

    fn run(this: Arc<BringUp>) {
        let data: &drm_sink::VinoDrmData = &this.ddev;
        let Ok(link) = UsbLink::open(&data.io, data.eps) else {
            return;
        };
        let dev = &link;
        let cdev: &device::Device = dev.io().interface().as_ref();
        let ddev = &this.ddev;
        // WIP scaffold: attempt the plaintext bring-up, then the clean-room HDCP 2.2
        // AKE/LC/SKE, then the post-SKE CP setup. Bind regardless of the outcome -- there
        // is no display path until the dock engages the encrypted control plane, which it
        // currently never does (see the "help wanted" note at the top of the file).
        match VinoDriver::bring_up(dev) {
            Ok(()) => {
                dev_info!(cdev, "vino: plaintext session init OK\n");
                match VinoDriver::run_ake(dev) {
                    Ok(session) => {
                        dev_info!(cdev, "vino: HDCP AKE + LC + SKE complete (session keyed)\n");
                        // Dev diagnostic: the live session key/riv, so the dock's encrypted
                        // EP84 replies can be decoded offline from a usbmon capture. Behind
                        // pr_debug, so compiled out unless dynamic debug is enabled.
                        pr_debug!(
                            "vino: SESSION ks={:02x?} riv={:02x?}\n",
                            &session.ks,
                            &session.riv
                        );

                        // Phase 2c: drive the post-SKE CP setup. send_cp_setup re-seals
                        // DLM's captured setup template under THIS session's live ks/riv and
                        // sends it; `acks` counts the dock's encrypted wsub=0x45 replies.
                        // THIS IS THE WALL: on a cold dock `acks` stays 0 -- the dock runs the
                        // entire plaintext handshake but never engages the encrypted CP.
                        let mut edid_out: Option<KVec<u8>> = None;
                        let mut edid_heads: [Option<KVec<u8>>; VinoDriver::CP_SETUP_HEADS] =
                            core::array::from_fn(|_| None);
                        let mut video_keys = [[0u8; 32]; VinoDriver::CP_SETUP_HEADS];
                        let mut heads_present = [false; VinoDriver::CP_SETUP_HEADS];
                        match VinoDriver::send_cp_setup(
                            dev,
                            &session,
                            &mut edid_out,
                            &mut edid_heads,
                            &mut video_keys,
                            &mut heads_present,
                        ) {
                            Ok((n, acks, wseq_end, ctr_end)) => {
                                dev_info!(cdev,
                                    "vino: CP setup sent -- {n} messages, {acks} dock CP acks (wsub=0x45)\n");
                                // CP engagement gates EP08 video: until the dock acks, pushing
                                // pixels at it wedges the hub.
                                CP_ENGAGED.store(acks > 0, core::sync::atomic::Ordering::SeqCst);
                                // Publish the engaged session to the DRM device so the KMS
                                // callbacks
                                // can send runtime CP (mode-set on a modeset, cursor on motion),
                                // continuing this keystream. Only when the dock actually engaged.
                                if acks > 0 {
                                    let drm_dev: &drm_sink::VinoDrmDevice = ddev;
                                    let data: &drm_sink::VinoDrmData = drm_dev;
                                    // Flip the DRM device's own CP-engaged gate (read by the
                                    // scanout path in `drm_sink`) so page-flips start pushing
                                    // EP08 pixels, and hand the session to the KMS callbacks.
                                    data.set_cp_engaged(true);
                                    data.publish_session(
                                        dev,
                                        &session.ks,
                                        &session.riv,
                                        wseq_end,
                                        ctr_end,
                                    );
                                    // Stash the per-head video keys for the EP08 encoder --
                                    // see the doc comment on `set_video_keys` for what is and
                                    // isn't proven about how they're actually used.
                                    data.set_video_keys(video_keys);
                                    // HW-CONFIRMED 2026-07-16: this firmware never delivers a
                                    // raw EDID over CP, so waiting on `set_edid` below to mark
                                    // the connector connected left a fully-engaged dock stuck
                                    // "Disconnected" forever ("[drm] Cannot find any crtc or
                                    // sizes"). CP engagement + this head's DISPLAY-CAP push
                                    // (`id=0x78 sub=0x30`, already required to reach here) IS
                                    // proof a real monitor answered -- mark it connected now;
                                    // `set_edid` below still upgrades to the real descriptor
                                    // if a future firmware ever sends one.
                                    // Mark each head's connector connected. Head 0 is always
                                    // connected once CP engages (the proven single-monitor path).
                                    // Head 1+ is connected ONLY when the dock answered that head's
                                    // stream-open with a DISPLAY-CAP (`heads_present[h]`, i.e. a
                                    // real monitor) -- never unconditionally, or a head with no
                                    // monitor becomes a phantom output the compositor extends onto.
                                    // (This firmware delivers no raw EDID over CP, so DISPLAY-CAP
                                    // presence is the monitor signal; `set_edid` still upgrades a
                                    // connector to the real descriptor if a future firmware sends
                                    // one.) `set_connected` is per-head-generic.
                                    for head in 0..VinoDriver::CP_SETUP_HEADS {
                                        if head == 0 || heads_present[head] {
                                            data.set_connected(drm_dev, head);
                                            dev_info!(
                                                    cdev,
                                                    "vino: head {} connector marked connected (CP engaged, monitor present)\n",
                                                    head
                                                );
                                        }
                                    }
                                }
                            }
                            Err(e) => dev_info!(cdev, "vino: CP setup incomplete ({e:?}) -- WIP\n"),
                        }
                        // Cache each head's dock EDID on the DRM device (when the CP channel
                        // delivered it) so that connector's get_modes installs the real monitor
                        // descriptor via the standard DRM EDID helpers. `send_cp_setup` fetches EDID
                        // per head (byte22 = head selector), so a dual-monitor dock brings both
                        // connectors up with their own native modes.
                        let drm_dev: &drm_sink::VinoDrmDevice = ddev;
                        let data: &drm_sink::VinoDrmData = drm_dev;
                        for (head, slot) in edid_heads.into_iter().enumerate() {
                            if let Some(blob) = slot {
                                let n = blob.len();
                                data.set_edid(drm_dev, head, blob);
                                dev_info!(
                                    cdev,
                                    "vino: cached dock EDID for head {head} connector ({n} bytes)\n"
                                );
                            }
                        }
                    }
                    Err(e) => dev_info!(cdev, "vino: HDCP AKE incomplete ({e:?}) -- WIP\n"),
                }
            }
            Err(e) => dev_info!(cdev, "vino: session init incomplete ({e:?}) -- WIP\n"),
        }
        // Bring-up attempt finished: allow the live-scanout connector to report connected
        // and let a compositor drive EP08 frames, without racing the handshake.
        BRINGUP_COMPLETE.store(true, core::sync::atomic::Ordering::SeqCst);
        {
            let drm_dev: &drm_sink::VinoDrmDevice = ddev;
            // ★ Downstream readiness wait (2026-07-25, confirmed against the DLM hotplug capture
            // `captures/dlm-hotplug-sequence-20260725-143903`, see docs/DLM-COLD-LIT-CHOREOGRAPHY.md):
            // on a monitor/dock connect DLM reads EDID, ENGAGEs, then polls `id=0x14 sub=0x0c` for
            // ~1.2 s while the dock brings the downstream LINK up -- its sealed `0x45` replies carry
            // downstream-progress status and shrink from ~190 B to ~64 B as the link settles -- and
            // ONLY THEN sends the mode-set. vino used to fire `hotplug_event()` (-> KWin mode-set)
            // immediately after engage, so a COLD downstream was mode-set mid-training and never lit
            // its panel; the WARM case worked only because the link was already up. Run the same
            // readiness poll here, BEFORE notifying userspace, so KWin's mode-set lands on a
            // downstream the dock has finished bringing up. Harmless on a warm plug (link already
            // ready -> the poll just idles). Bounded, and `keepalive_running()` also breaks it the
            // instant `disconnect()` latches `SESSION_TEARDOWN`, so `cancel_sync` can never hang
            // unload for the full window -- not even when the disconnect beat this poll to it.
            if CP_ENGAGED.load(core::sync::atomic::Ordering::SeqCst) {
                let data: &drm_sink::VinoDrmData = drm_dev;
                keepalive_arm();
                let start = Instant::<Monotonic>::now();
                let window = Delta::from_millis(1300);
                let mut polls = 0u32;
                while Instant::<Monotonic>::now() - start < window && keepalive_running() {
                    let _ = data.send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c));
                    polls += 1;
                    fsleep(Delta::from_millis(15));
                }
                dev_info!(
                    cdev,
                    "vino: downstream readiness poll window done ({polls} polls, ~1300 ms) before hotplug\n"
                );
            }
            drm_dev.hotplug_event();
            dev_info!(
                cdev,
                "vino: bring-up complete -- live-scanout connector now connected\n"
            );

            // ★ 2026-07-20 RE: DLM keeps the CP dialogue alive for the ENTIRE session -- measured in
            // the cold pcap at ~74 `id=0x14 sub=0x0c` messages/sec on EP02, running continuously
            // alongside (and between) video frames. vino previously went CP-SILENT the instant
            // bring-up finished, so the dock's session timed out and it hard-reset a few seconds
            // later -- exactly the observed DELAYED reset after an otherwise-accepted EP08 write.
            // A one-shot status burst is insufficient: the dialogue must remain continuous.
            // Bounded so a `disconnect()` flush of this work can never hang unload for long.
            let data: &drm_sink::VinoDrmData = drm_dev;
            dev_info!(
                cdev,
                "vino: starting continuous CP keepalive (~74/s, DLM cadence)\n"
            );
            let mut sent = 0u32;
            // ★ 2026-07-22: the status poll is NOT the whole steady-state dialogue. Both decrypted
            // DLM captures that cover a live session show `id=0x16 sub=0x75` going out every
            // 3.000 s for as long as the stream is up (see `cp::heartbeat`). vino sent exactly one,
            // from inside the EDID loop, and none afterwards -- the leading suspect for the dock
            // accepting complete multi-megabyte video frames while the downstream panels never
            // leave "no signal". Interleave it here on its own 3 s deadline; the poll cadence
            // below is deliberately left untouched so this is a single-variable change.
            const HEARTBEAT_PERIOD: Delta = Delta::from_secs(3);
            let mut next_heartbeat = Instant::<Monotonic>::now() + HEARTBEAT_PERIOD;
            let mut beats = 0u32;
            let mut pushes = 0usize;
            // Stage 2: runtime monitor connect/remove. vino latched each head's presence ONCE at
            // bring-up; a monitor plugged in or pulled out afterwards was never noticed. Re-probe
            // each head on a slow cadence and reflect changes to its connector (with the same
            // readiness wait on a fresh connect, and a full teardown on removal). Debounced so a
            // single mis-decoded reply cannot flap a connector.
            const PRESENCE_PERIOD: Delta = Delta::from_millis(1000);
            let mut next_presence = Instant::<Monotonic>::now() + PRESENCE_PERIOD;
            let mut head_known = [false; VinoDriver::CP_SETUP_HEADS];
            let mut head_debounce = [0u8; VinoDriver::CP_SETUP_HEADS];
            /// Consecutive undecodable presence replies that count as "this head has no monitor".
            ///
            /// The probe used to treat a missing reply as "no change", which makes a head that
            /// stops answering entirely indistinguishable from a healthy one. **That is exactly
            /// what a lost monitor looks like** -- measured 2026-07-27 on a physical unplug: the
            /// dock does not switch to a "no sink" reply, it simply stops answering this head's
            /// probe while the *other* head's status word moves. Silence is the signal, so it only
            /// has to outlast a dropped reply, not stand in for missing evidence.
            ///
            /// Three cycles at a 1 s period is ~3 s, against the ~16 s the first measurement (8 x
            /// 2 s) took to disconnect the output. A probe landing during a mode-set cannot produce
            /// a false positive: the CP timeline is exclusive there and this loop is paused, so no
            /// cycle elapses at all.
            const PRESENCE_SILENT_LIMIT: u8 = 3;
            /// How often to retry the sink re-engage on a head believed to have no monitor.
            ///
            /// Reacting to the dock's replug announcement is not enough on its own, because the
            /// announcement is not reliably identifiable. Across two physical replugs the dock sent
            /// completely different things: once an unprompted `id=0x44 sub=0x20`, once
            /// `id=0x88 sub=0xc` followed by `id=0x3 sub=0x82` -- and the id field of those 176/192
            /// byte pushes is not even stable enough to key on (`0x83`, `0x88`, `0x8b`, `0x91`,
            /// `0x92` all appear for what is evidently one message class).
            ///
            /// What *is* reliable is that the dock only answers a head's presence probe once its
            /// sink is re-engaged. So poll with re-engage attempts instead of waiting to be told.
            /// A monitor that is genuinely absent costs seven CP messages every few seconds, which
            /// is far below the ordinary keepalive rate; a monitor that came back is picked up
            /// within one period.
            const REENGAGE_RETRY: Delta = Delta::from_millis(4000);
            let mut next_reengage = [Instant::<Monotonic>::now(); VinoDriver::CP_SETUP_HEADS];
            /// How long after a head is brought back to ignore a negative presence probe.
            ///
            /// The dock answers `id=0x14` -- which this code reads as "no monitor" -- for a head
            /// whose EDID handler is not engaged, and there is a window right after a re-engage
            /// where that is still true even though the monitor is plainly there and its EDID has
            /// just been read. Without a grace period the two-cycle debounce would tear the head
            /// straight back down and the whole thing would oscillate.
            const PRESENCE_GRACE: Delta = Delta::from_millis(10_000);
            let mut presence_grace =
                [Instant::<Monotonic>::now(); VinoDriver::CP_SETUP_HEADS];
            let mut head_silent = [0u8; VinoDriver::CP_SETUP_HEADS];
            for h in 0..VinoDriver::CP_SETUP_HEADS {
                head_known[h] = data.head_present(h);
            }
            keepalive_arm();
            while keepalive_running() {
                // The modeset worker reproduces one short DLM control/video timeline from an
                // absolute mode-set anchor. Do not race an independent poll/heartbeat/presence
                // transaction into that sequence; resume immediately after its closing markers.
                if data.cp_timeline_exclusive() {
                    fsleep(Delta::from_millis(1));
                    continue;
                }
                if data
                    .send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c))
                    .is_ok()
                {
                    sent += 1;
                }
                // `Instant` implements neither `PartialOrd` nor `AddAssign` here, so compare via the
                // `Delta` that subtracting two instants yields (negative until the deadline passes).
                let now = Instant::<Monotonic>::now();
                if (now - next_heartbeat).as_millis() >= 0 {
                    if data.send_cp(dev, 0x16, 0, cp::heartbeat).is_ok() {
                        beats += 1;
                    }
                    // Fixed cadence, not "3 s after the last one finished": a slow send must not
                    // let the interval drift away from DLM's metronomic 3.000 s.
                    next_heartbeat = next_heartbeat + HEARTBEAT_PERIOD;
                    if (now - next_heartbeat).as_millis() > 0 {
                        next_heartbeat = now + HEARTBEAT_PERIOD; // fell far behind; resynchronise
                    }
                }
                // Unpaired drain: consume anything the dock pushed on its own initiative since the
                // last cycle, rather than leaving it until the next write's paired read. This is
                // what puts vino's EP84/EP02 submit ratio above 1.0, where every DLM cold plug
                // sits (1.116-1.157). See `VinoDrmData::drain_cp_pushes`.
                const MAX_UNPAIRED_DRAIN: usize = 4;
                pushes += data.drain_cp_pushes(dev, MAX_UNPAIRED_DRAIN);
                // Userspace-requested sink re-engage (`/sys/devices/vino/reengage`). Runs here
                // rather than in the sysfs write because this loop already owns a live `Io` token
                // and the CP dialogue; see `drm_sink::REENGAGE_REQUEST` for why the path needs a
                // trigger that does not depend on presence detection working first.
                let simulated = drm_sink::take_simulated_unplugs();
                if simulated != 0 {
                    for h in 0..VinoDriver::CP_SETUP_HEADS {
                        if simulated & (1 << h) != 0 {
                            dev_info!(cdev, "vino: head {h} simulated unplug -- dropping the sink\n");
                            data.drop_sink(dev, h as u8);
                        }
                    }
                }
                let requested = drm_sink::take_reengage_requests();
                if requested != 0 {
                    for h in 0..VinoDriver::CP_SETUP_HEADS {
                        if requested & (1 << h) == 0 {
                            continue;
                        }
                        dev_info!(cdev, "vino: head {h} sink re-engage requested via sysfs\n");
                        data.set_connected(drm_dev, h);
                        data.reengage_head(drm_dev, dev, h as u8);
                        // The same downstream readiness wait a fresh bring-up uses, so a mode-set
                        // following the hotplug lands on a settled link rather than a
                        // mid-negotiation one. Bounded, and it drops out at once if the session is
                        // going away.
                        let rs = Instant::<Monotonic>::now();
                        while (Instant::<Monotonic>::now() - rs).as_millis() < 1300
                            && keepalive_running()
                        {
                            let _ =
                                data.send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c));
                            fsleep(Delta::from_millis(15));
                        }
                        head_known[h] = true;
                        head_silent[h] = 0;
                        drm_dev.hotplug_event();
                    }
                    // The wait above consumed time the other deadlines were counting on.
                    next_presence = Instant::<Monotonic>::now() + PRESENCE_PERIOD;
                    next_heartbeat = Instant::<Monotonic>::now() + HEARTBEAT_PERIOD;
                }
                // Keep trying to bring an absent head's sink back. See `REENGAGE_RETRY`.
                {
                    let now_r = Instant::<Monotonic>::now();
                    for h in 0..VinoDriver::CP_SETUP_HEADS {
                        if head_known[h] || (now_r - next_reengage[h]).as_millis() < 0 {
                            continue;
                        }
                        next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                        dev_info!(cdev, "vino: head {h} absent -- retrying the sink re-engage\n");
                        // ★ A recovered EDID IS the presence signal (measured 2026-07-27, third
                        // physical replug).
                        //
                        // The probe's id is not. With the monitor physically back and its EDID
                        // reading out perfectly -- `EDID re-cached after re-engage (384 bytes)`,
                        // valid magic, right length -- the dock still answered the presence probe
                        // with the generic `id=0x14`, which this code read as "no monitor". So the
                        // head was re-engaged successfully over and over and never marked present,
                        // the retry never stopped, and ~20 s later the dock gave up and
                        // re-enumerated, taking the other screen with it. `0x44` versus `0x14` is
                        // not about whether a monitor is attached; it is about whether the dock's
                        // EDID handler is currently engaged for that head.
                        //
                        // A 384-byte blob with a correct EDID header cannot come from an empty
                        // port. Trust it, and stop.
                        if data.reengage_head(drm_dev, dev, h as u8) {
                            data.set_connected(drm_dev, h);
                            head_known[h] = true;
                            head_debounce[h] = 0;
                            head_silent[h] = 0;
                            presence_grace[h] = Instant::<Monotonic>::now() + PRESENCE_GRACE;
                            dev_info!(
                                cdev,
                                "vino: head {h} monitor CONNECTED -- its EDID read back after the \
                                 re-engage\n"
                            );
                            drm_dev.hotplug_event();
                        }
                        head_silent[h] = 0;
                        next_presence = Instant::<Monotonic>::now();
                    }
                }
                // Stage 2: periodic per-head monitor presence re-check -- brought forward the
                // moment the dock says the downstream topology moved. Waiting out the period after
                // an explicit announcement is what let a replug sit unhandled until the dock gave
                // up and re-enumerated itself; see `drm_sink::DOWNSTREAM_EVENT`.
                if drm_sink::take_downstream_event() {
                    next_presence = Instant::<Monotonic>::now();
                    // Bring the retry deadline forward rather than running a second, unbounded
                    // re-engage path. Two independent triggers hammered the dock with a seven
                    // message sequence roughly every second, which is very likely what pushed it
                    // into the give-up reset this was meant to prevent.
                    for h in 0..VinoDriver::CP_SETUP_HEADS {
                        if !head_known[h] {
                            next_reengage[h] = Instant::<Monotonic>::now();
                        }
                    }
                    // ★ Break the replug deadlock (measured 2026-07-27, second unplug test).
                    //
                    // A head whose monitor came back stays silent, because the dock will not answer
                    // its presence probe until the downstream sink is re-engaged -- and vino would
                    // not re-engage until the probe said the monitor was back. Neither side moves,
                    // and ~17 s later the dock gives up and re-enumerates the whole device, which
                    // is why a replug tore down the *other* screen too and restarted both.
                    //
                    // So when the dock reports a topology change, optimistically re-engage every
                    // head currently believed absent, then let the probe below settle it. If the
                    // monitor really is gone the sequence is harmless -- the dock stays silent and
                    // the head stays disconnected -- and the trigger is edge-driven, so this cannot
                    // spin.
                }
                let now_p = Instant::<Monotonic>::now();
                if (now_p - next_presence).as_millis() >= 0 {
                    next_presence = now_p + PRESENCE_PERIOD;
                    for h in 0..VinoDriver::CP_SETUP_HEADS {
                        let present = match data.probe_head_present(dev, h as u8) {
                            Some(p) => {
                                head_silent[h] = 0;
                                p
                            }
                            None => {
                                // A head vino blanked is silent because vino asked it to be. Only
                                // the dock's own silence is evidence of an unplug.
                                if data.is_self_blanked(h) {
                                    continue;
                                }
                                head_silent[h] = head_silent[h].saturating_add(1);
                                if head_silent[h] < PRESENCE_SILENT_LIMIT {
                                    continue; // one dropped reply proves nothing
                                }
                                if head_known[h] {
                                    dev_info!(
                                        cdev,
                                        "vino: head {h} presence probe silent for {} cycles -- \
                                         treating as REMOVED\n",
                                        head_silent[h]
                                    );
                                }
                                false
                            }
                        };
                        if present == head_known[h] {
                            head_debounce[h] = 0;
                            continue;
                        }
                        // Inside the settling window after a recovery, a negative answer is not
                        // evidence -- see `PRESENCE_GRACE`.
                        if !present
                            && (Instant::<Monotonic>::now() - presence_grace[h]).as_millis() < 0
                        {
                            head_debounce[h] = 0;
                            continue;
                        }
                        // Require two consecutive contrary reads before acting.
                        head_debounce[h] = head_debounce[h].saturating_add(1);
                        if head_debounce[h] < 2 {
                            continue;
                        }
                        head_debounce[h] = 0;
                        head_known[h] = present;
                        next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                        if present {
                            data.set_connected(drm_dev, h);
                            dev_info!(
                                cdev,
                                "vino: head {h} monitor CONNECTED at runtime -- re-engage + readiness wait + hotplug\n"
                            );
                            // Re-enable the dock's downstream sink for this head before anything
                            // else. The monitor's removal tore down the sink state that the EDID
                            // engage (`id=0x16 sub=0x23`) establishes, and without re-running it the
                            // dock accepts the replug's mode-set and every video byte while leaving
                            // the panel dark -- see `VinoDrmData::reengage_head`.
                            data.reengage_head(drm_dev, dev, h as u8);
                            // Same downstream readiness wait as a fresh bring-up before notifying
                            // userspace, so KWin's mode-set lands on a settled downstream link.
                            let rs = Instant::<Monotonic>::now();
                            while (Instant::<Monotonic>::now() - rs).as_millis() < 1300
                                && keepalive_running()
                            {
                                let _ = data
                                    .send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c));
                                fsleep(Delta::from_millis(15));
                            }
                        } else {
                            data.set_disconnected(h);
                            dev_info!(cdev, "vino: head {h} monitor REMOVED at runtime -- hotplug\n");
                        }
                        drm_dev.hotplug_event();
                        // Re-baseline the heartbeat/presence deadlines skipped during the wait.
                        next_presence = Instant::<Monotonic>::now() + PRESENCE_PERIOD;
                    }
                }
                fsleep(Delta::from_millis(13));
            }
            dev_info!(
                cdev,
                "vino: CP keepalive finished ({sent} polls, {beats} heartbeats, {pushes} unprompted dock pushes)\n"
            );
        }
    }
}

/// On-device crypto known-answer self-test. Confirms the IN-KERNEL crypto path (which the CP seal
/// depends on) is byte-correct -- something only ever checked offline (Python `verify-kdf.py`)
/// before.
/// Runs three checks and logs PASS/FAIL:
///   1. AES-128-ECB vs the FIPS-197 test vector.
///   2. AES-CMAC vs the RFC 4493 test vector (subkey + full-block path).
///   3. The full `cp::seal_livemac` vs cold-ref's REAL msg0: known plaintext + known `ks`/`riv`
///      must reproduce the captured wire ciphertext+tag byte-for-byte. A FAIL here (with 1+2
///      passing) would localize a bug in our seal framing; a FAIL in 1/2 means the kernel
///      primitive itself is wrong. If all PASS, the crypto we send is correct and the
///      CP-engagement wall is NOT our crypto.
fn crypto_selftest() {
    use core::sync::atomic::{AtomicBool, Ordering};
    static RAN: AtomicBool = AtomicBool::new(false);
    if RAN.swap(true, Ordering::Relaxed) {
        return;
    }

    // 1. AES-128-ECB KAT (FIPS-197 Appendix B / C.1).
    let ecb_key = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];
    let ecb_pt = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    let ecb_expect = [
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
        0x5a,
    ];
    match crypto::aes128_ecb(&ecb_key, &ecb_pt) {
        Ok(out) if out == ecb_expect => pr_info!("vino: selftest AES-128-ECB PASS\n"),
        Ok(out) => pr_err!("vino: selftest AES-128-ECB FAIL got={out:02x?}\n"),
        Err(e) => pr_err!("vino: selftest AES-128-ECB ERR ({e:?})\n"),
    }

    // 2. AES-CMAC KAT (RFC 4493 sec 4 example 2: a single 16-byte block).
    let cmac_key = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
        0x3c,
    ];
    let cmac_msg = [
        0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17,
        0x2a,
    ];
    let cmac_expect = [
        0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44, 0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28,
        0x7c,
    ];
    match crypto::aes_cmac(&cmac_key, &cmac_msg) {
        out if out == cmac_expect => pr_info!("vino: selftest AES-CMAC PASS\n"),
        out => pr_err!("vino: selftest AES-CMAC FAIL got={out:02x?}\n"),
    }

    // 3. Full seal_livemac vs cold-ref's REAL msg0 (capture t=36.813765). ks/riv are the cold-ref
    // session's; content is msg0's 32-byte plaintext; the expected frame is the captured wire.
    // `riv` here is the AES-CTR nonce; dl3cmac_tag keys the CMAC on `riv` with byte0^0x80. This same
    // CTR-nonce / CMAC-nonce(=CTR^0x80@byte0) split was reconfirmed byte-exact against DLM 3.4.26's
    // two rr-trace msg0s (2026-07-16) -- cold-ref and 3.4.26 use the identical scheme.
    let ks = [
        0xd8, 0xb2, 0x48, 0x12, 0x44, 0x1d, 0x50, 0x82, 0x0d, 0xa3, 0xc2, 0x71, 0xc7, 0xa3, 0x6e,
        0xc2,
    ];
    let riv = [0xfb, 0xa7, 0xc3, 0x5f, 0xe6, 0xce, 0x40, 0xec];
    let header = [
        0x00, 0x00, 0x3c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x24, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];
    let content = [
        0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0x48, 0xec, 0x9c, 0xec, 0xc3, 0x89, 0x23,
        0x5d, 0x69,
    ];
    let expect = [
        0x00, 0x00, 0x3c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x24, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xcb, 0x4c, 0x80, 0xde, 0xf0, 0xd0, 0xfd, 0x56, 0x22, 0x5f, 0x43, 0xbd, 0x55, 0x0d,
        0x8e, 0xc5, 0x7a, 0x1c, 0x35, 0x12, 0x81, 0x35, 0x31, 0x1a, 0x45, 0x13, 0x91, 0x41, 0x25,
        0x87, 0xe9, 0xf7, 0xe5, 0x5b, 0xb5, 0xbc, 0x76, 0x5b, 0x2f, 0x1e, 0x79, 0xf2, 0x8b, 0xd5,
        0x5b, 0x2c, 0x3c, 0xe7,
    ];
    match cp::seal_livemac(&ks, &riv, &header, &content) {
        Ok(frame) if frame.as_slice() == expect.as_slice() => {
            pr_info!(
                "vino: selftest seal_livemac(msg0) PASS -- CP crypto reproduces cold-ref wire\n"
            )
        }
        Ok(frame) => {
            // Show where it first diverges so a framing/order bug is localizable.
            let mut at = frame.len().min(expect.len());
            for i in 0..at {
                if frame[i] != expect[i] {
                    at = i;
                    break;
                }
            }
            pr_err!(
                "vino: selftest seal_livemac(msg0) FAIL at byte {at} (len {} vs {})\n",
                frame.len(),
                expect.len()
            );
            let s = at.saturating_sub(0);
            let e = (at + 16).min(frame.len());
            pr_err!("vino:   got[{s}..]={:02x?}\n", &frame[s..e]);
            let e2 = (at + 16).min(expect.len());
            pr_err!("vino:   exp[{s}..]={:02x?}\n", &expect[s..e2]);
        }
        Err(e) => pr_err!("vino: selftest seal_livemac(msg0) ERR ({e:?})\n"),
    }

    // 4. IN-reply decode vs a REAL ENGAGED dock ACK on the current 0c0219 firmware
    // (captures/live-3426-20260707/reauth.pcap, EP84 wire_seq=0 @ t=12.399789 -- the dock's
    // first ack of DLM's real msg0). `verify_in_ack` must recognise it as `id=0x4c sub=0 ctr=9`.
    // `ack_out_riv` is the value `run_ake` stores (SKE base `bf1ed14b..945a` with byte7^0x01);
    // `verify_in_ack`/`in_riv` must flip byte7 back to `base` to decrypt it. This is the
    // regression guard for the decode path the user asked to verify: under the old identity
    // `in_riv` this ACK decodes to garbage (`id=0x8a2d`) and vino would misreport a genuinely
    // engaged dock as "rejecting". Proven by decoding the frame offline in this tree.
    let ack_ks = [
        0x12, 0x7d, 0x51, 0xf9, 0xc6, 0xbe, 0xe7, 0x7b, 0xea, 0x39, 0x2c, 0xfe, 0x1a, 0x8f, 0x66,
        0x5e,
    ];
    let ack_out_riv = [0xbf, 0x1e, 0xd1, 0x4b, 0xe5, 0x1a, 0x94, 0x5b];
    let ack_wire = [
        0x00, 0x00, 0x6c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x45, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x18, 0x4f, 0x0d, 0x21, 0x4d, 0x3b, 0xb4, 0x5d, 0xcb, 0x0e, 0x88, 0xe2, 0xc8, 0x8b,
        0x5d, 0xc9, 0x9c, 0xac, 0x10, 0x3c, 0x3d, 0x89, 0x4a, 0x73, 0x41, 0xaa, 0x5e, 0x87, 0x1d,
        0xc5, 0x2b, 0x1e, 0xf2, 0xae, 0xc1, 0xee, 0x13, 0x5a, 0xf6, 0x91, 0xf7, 0x03, 0x95, 0x36,
        0x76, 0x91, 0xa4, 0xc0, 0x6c, 0x45, 0x89, 0x09, 0xf0, 0x1c, 0x00, 0xbe, 0x83, 0x35, 0x8e,
        0x50, 0x1e, 0xfd, 0x8d, 0x8a, 0x4f, 0x7c, 0x08, 0x25, 0x4e, 0xde, 0x99, 0xb4, 0xe9, 0x35,
        0xdf, 0xe1, 0x6b, 0x0f, 0x80, 0xc0, 0xd5, 0x28, 0x2b, 0xd8, 0xb3, 0x8e, 0x19, 0x7d, 0xb6,
        0xad, 0xb5, 0x27, 0x26, 0x08, 0xa1, 0xfa,
    ];
    match cp::verify_in_ack(&ack_ks, &ack_out_riv, &ack_wire) {
        Some((0x4c, 0x00, 9)) => pr_info!(
            "vino: selftest in-reply decode PASS -- real engaged dock ACK decodes to id=0x4c sub=0 ctr=9\n"
        ),
        other => pr_err!(
            "vino: selftest in-reply decode FAIL -- got {other:?} (want Some((0x4c, 0, 9)))\n"
        ),
    }
}

impl VinoDriver {
    /// Plaintext session bring-up (sec 4): control-request preamble then the three
    /// bulk init messages, reading the single ACK. Best-effort during scaffold
    /// bring-up -- errors are logged, not fatal.
    fn bring_up(dev: &UsbLink<'_>) -> Result {
        // BEFORE ANYTHING (2026-07-14, user's model): mimic DLM's userspace-daemon behaviour -- let the
        // OTHER USB drivers (audio/HID on interfaces 5 & 6) claim their interfaces at plug time, then
        // settle 5 s, and only THEN start the whole session (preamble + session-init + AKE). vino, being
        // an in-kernel interface-0 driver, otherwise reacts the instant it is probed -- long before the
        // composite device has finished coming up. `probe()` records every interface it is offered in
        // PROBED_IFACES; poll until 5 & 6 are seen (bounded by [`ENUM_RESPONSE_HOLD_US`] = 5 s), then
        // hold a further 5 s. The log says whether 5/6 were claimed here (plug-time enum -> the model is
        // right) or the poll timed out (5/6 are session-init-triggered -> they come later regardless).
        if DLM_FIXED_TIMERS {
            const WANT: u32 = (1u32 << 5) | (1u32 << 6);
            let start = Instant::<Monotonic>::now();
            loop {
                let mask = PROBED_IFACES.load(core::sync::atomic::Ordering::Acquire);
                if mask & WANT == WANT {
                    pr_info!(
                        "vino: interfaces 5/6 claimed at plug after {} us (mask={mask:#x}) -- settling then starting\n",
                        start.elapsed().as_micros_ceil()
                    );
                    break;
                }
                if start.elapsed().as_micros_ceil() >= ENUM_RESPONSE_HOLD_US {
                    pr_info!(
                        "vino: interfaces 5/6 NOT claimed at plug within {} us (mask={mask:#x}) -- starting anyway (session-init-triggered?)\n",
                        ENUM_RESPONSE_HOLD_US
                    );
                    break;
                }
                fsleep(Delta::from_millis(5));
            }
            // Then settle a further 1.5 s before touching the CP channel. This was 5 s ("wait then
            // wait 5 seconds then start"), but the 2026-07-14 all-bus diff measured DLM's own
            // enumeration->CP-setup idle gap at only ~1.8 s (last SET_CONFIGURATION 20.22 s -> first
            // 0x24 21.99 s), while vino's 5 s settle put it at ~5.3 s. 1.5 s keeps vino comfortably
            // under DLM's proven-safe gap so the dock's CP state machine is exercised on the same
            // fresh-plug timeline DLM engages on, without re-racing the plug-time enum storm.
            fsleep(Delta::from_millis(1500));
        }

        // Verify the KERNEL crypto path is byte-correct before we rely on it for CP. The KDF was
        // only ever checked offline (Python); this confirms the in-kernel AES-ECB, AES-CMAC and the
        // full `seal_livemac` reproduce ground-truth vectors on THIS device. Logs PASS/FAIL once.
        crypto_selftest();

        // Control-request preamble (sec 4): dock-id read, interface selection, then the
        // vendor_out 0x24 / vendor_in 0x22 pairs that kick off the HDCP path. (The
        // GET_DESCRIPTOR string reads DLM also issues look cosmetic and are omitted.)
        const VENDOR_OUT: u8 = 0x40; // host->dev, vendor, device
        const VENDOR_IN_IFACE: u8 = 0xc1; // dev->host, vendor, INTERFACE recipient (DLM's choice)

        // The DLM-style vendor preamble (sec 4). Per the userspace oracle, every
        // control request here is **best-effort**: the dock legitimately STALLs
        // some of them (e.g. the cosmetic dock-id read) yet still advances its
        // host-identification state. The oracle tolerates each error and relies
        // on DLM's inter-request timing gaps -- without those gaps the dock may
        // not advance. So we log-and-continue on every control step and insert
        // the same delays; only the bulk init + ACK is treated as load-bearing.
        // GROUND-TRUTH 2026-06-13: at device-open DLM issues two vendor-IN reads on interface 1,
        // recipient 0xc1, BEFORE the SET_INTERFACE / 0x24 / 0x22 sequence (dlm-cold-20260611-123347
        // f708 `0xc1 0xfe wIdx=1` -> 16 B "RidgeDock" blob; f710 `0xc1 0xfc wIdx=1` -> 0 B). vino
        // skipped them; the earlier attempt used recipient 0xc0 (device) and STALLed, which was
        // misread as "the dock rejects 0xfe / DLM never sends it". Issue them here with the correct
        // 0xc1 recipient. Best-effort: log and continue (the dock may still short/stall 0xfc).
        let mut dock_id = [0u8; 16];
        match dev.control_recv(
            0xfe,
            VENDOR_IN_IFACE,
            0,
            1,
            &mut dock_id,
            timeout(),
            GFP_KERNEL,
        ) {
            Ok(()) => pr_info!(
                "vino: step device-open 0xfe(iface1) OK = {:02x?}\n",
                dock_id
            ),
            Err(e) => pr_info!("vino: step device-open 0xfe(iface1) non-fatal ({e:?})\n"),
        }
        // Windows issues ONLY the single `0xfe` device-open read above; the `0xfc`/`0xfd`/`0xfb`
        // DFU reads are a DLM-oracle addition vino picked up. Skip them under [`WINDOWS_MIMIC`] to
        // match the Windows EP0 stream (they are diagnostic-only and CP-irrelevant either way).
        if !WINDOWS_MIMIC {
            let mut probe3 = [0u8; 3];
            match dev.control_recv(
                0xfc,
                VENDOR_IN_IFACE,
                0,
                1,
                &mut probe3,
                timeout(),
                GFP_KERNEL,
            ) {
                Ok(()) => pr_info!("vino: step device-open 0xfc(iface1) OK = {:02x?}\n", probe3),
                Err(e) => pr_info!("vino: step device-open 0xfc(iface1) non-fatal ({e:?})\n"),
            }
            // DFU firmware-version query, matching DLM / the macOS+Windows drivers'
            // DfuGetVmmDeviceFirmwareVersion: vendor IN bmRequestType=0xc1 bRequest=0xfd wIndex=1,
            // a 6-byte version blob (the reference driver's request-size table: 0xfb=4
            // customer/board, 0xfc=3 device-type, 0xfd=6 firmware-version, 0xfe=16 descriptor). This
            // is a device-level DFU read, independent of the CP channel, so it works regardless of
            // CP engagement -- handy for diagnostics and confirming the dock firmware revision.
            let mut fw_ver = [0u8; 6];
            match dev.control_recv(
                0xfd,
                VENDOR_IN_IFACE,
                0,
                1,
                &mut fw_ver,
                timeout(),
                GFP_KERNEL,
            ) {
                Ok(()) => pr_info!("vino: dock DFU firmware version = {:02x?}\n", fw_ver),
                Err(e) => pr_info!("vino: device-open 0xfd(firmware-version) non-fatal ({e:?})\n"),
            }
            // DFU customer/board id (DfuGetVmmDeviceCustomerAndBoardId): bRequest=0xfb, 4-byte blob.
            let mut cust_board = [0u8; 4];
            match dev.control_recv(
                0xfb,
                VENDOR_IN_IFACE,
                0,
                1,
                &mut cust_board,
                timeout(),
                GFP_KERNEL,
            ) {
                Ok(()) => pr_info!("vino: dock DFU customer/board id = {:02x?}\n", cust_board),
                Err(e) => pr_info!("vino: device-open 0xfb(customer/board) non-fatal ({e:?})\n"),
            }
        }

        // EXPERIMENT (2026-06-16): replay DLM's repeated STRING-descriptor reads at device-open.
        // Timing analysis of the paired cold capture (captures/paired-coldbus-20260615-220311)
        // shows DLM, beyond the distinct descriptor SET vino already issues, re-reads STRING idx0
        // (language-ID list) and idx3 (en-US product, langid 0x0409), 255 B each, at ~2/sec for the
        // ENTIRE 175 s session -- a 1 Hz host string-poll heartbeat. Engagement happens in the
        // first
        // second, so this is almost certainly NOT a pre-AKE gate (the distinct set already
        // matches),
        // but the repetition was never A/B-tested by replay the way the 0xfe/0xfc reads were. Issue
        // a
        // small burst here, BEFORE the AKE, to test whether the dock conditions CP engagement on
        // seeing the host poll its strings. Best-effort: the kernel reports EREMOTEIO on the
        // expected
        // short reply, but the GET_DESCRIPTOR still reaches the wire, which is all the experiment
        // needs.
        // RESULT 2026-06-16 (paired-coldbus-20260616-162650): the pre-arm GET_DESCRIPTOR delta is
        // USB ENUMERATION, not application protocol. Both captures contain an identical 3x 8-byte +
        // 7x 18-byte DEVICE-descriptor read sequence -- which no kernel driver issues (it is the
        // enumeration handshake the USB core runs each time the dock re-enumerates on the cold
        // plug, plus DisplayLink's leftover /opt/displaylink/udev.sh hook firing per uevent).
        // Proven to be enumeration, not the DLM daemon: the vino capture reproduces the SAME reads
        // with displaylink-driver.service masked and no DisplayLinkManager process running. It is
        // symmetric across both runs, so it is neither a DLM-vs-vino difference nor the engagement
        // gate. This speculative burst only ADDED vino-issued reads on top, so disable it.
        // -- LIBUSB-STYLE DEVICE-OPEN ENUMERATION (2026-06-17)
        // ----------------------------------
        // The clean paired capture (paired-coldbus-20260616-180401) isolated the LAST pre-AKE
        // divergence from DLM to ONE thing: DLM (libusb) re-reads the dock's full descriptor set
        // when it opens the device -- DEVICE(18), CONFIG(9 then full ~618), STRING langid(idx0),
        // then every STRING index the descriptors reference (~22x 255B) -- right before the AKE.
        // A
        // kernel driver normally skips this (the USB core cached it at enumeration), which is why
        // vino's pre-arm control stream was missing it (the "DLM-ONLY 255x22 / 618 / 40"
        // residual).
        // These reads are CP-irrelevant descriptor boilerplate. The cold-plug A/B proved the dock
        // does NOT gate CP on them (replaying them byte-for-byte still gave 0x wsub=0x45 -- see
        // project_get_descriptor_burst_experiment / the firmware-wall verdict), and the in-kernel
        // Windows (WDF) and macOS (IOUSBLib) drivers DON'T issue this burst either -- like vino
        // they run over an already-enumerated device and use the USB core's cached descriptors.
        // The burst is therefore a libusb-userspace artifact, not something the dock expects.
        // Default OFF so vino behaves like a native kernel driver; flip to `true` only to reproduce
        // DLM's libusb wire for a paired A/B diff. Best-effort throughout: a STALL/EREMOTEIO on an
        // absent index is fine -- EP0 auto-recovers and the SETUP still reaches the wire (all the
        // A/B diff needs). Reproduces (histogram diff DLM vs vino, paired-coldbus-20260616-180401):
        // DLM's libusb open adds CONFIG-full(618)x3, CONFIG-partial(40)x3, STRING(255)x22, with
        // no
        // extra DEVICE(18)/CONFIG(9).
        // Windows (like a native kernel driver) does NOT replay this libusb descriptor burst, so
        // [`WINDOWS_MIMIC`] forces it off; otherwise default ON to reproduce DLM's libusb open.
        const CP_LIBUSB_OPEN_ENUM: bool = !WINDOWS_MIMIC;
        if CP_LIBUSB_OPEN_ENUM {
            let mut tmp = [0u8; 255];
            let mut cfg = KVec::from_elem(0u8, 618, GFP_KERNEL)?;
            // CONFIG full (618) x3 -- parse the first to find real string indices so the STRING
            // reads
            // below return data (matching DLM's byte counts), not just the SETUP counts.
            for _ in 0..3 {
                let _ = dev.control_recv(0x06, 0x80, 0x0200, 0, &mut cfg, timeout(), GFP_KERNEL);
            }
            // CONFIG partial (40) x3.
            for _ in 0..3 {
                let _ =
                    dev.control_recv(0x06, 0x80, 0x0200, 0, &mut tmp[..40], timeout(), GFP_KERNEL);
            }
            // STRING idx0 = language-ID list (1st of the 22x 255 reads); adopt the dock's REAL
            // langid.
            let mut langid = 0x0409u16;
            if dev
                .control_recv(0x06, 0x80, 0x0300, 0, &mut tmp, timeout(), GFP_KERNEL)
                .is_ok()
                && tmp[0] >= 4
            {
                langid = (tmp[2] as u16) | ((tmp[3] as u16) << 8);
            }
            // String indices referenced by the config (iConfiguration @off6, iInterface @off8).
            let mut idxs = [0u8; 64];
            let mut ni = 0usize;
            let mut p = 0usize;
            while p + 2 <= cfg.len() {
                let blen = cfg[p] as usize;
                if blen == 0 {
                    break;
                }
                let btype = cfg[p + 1];
                if btype == 0x02 && p + 7 <= cfg.len() && cfg[p + 6] != 0 && ni < idxs.len() {
                    idxs[ni] = cfg[p + 6];
                    ni += 1;
                }
                if btype == 0x04 && p + 9 <= cfg.len() && cfg[p + 8] != 0 && ni < idxs.len() {
                    idxs[ni] = cfg[p + 8];
                    ni += 1;
                }
                p += blen;
            }
            // 21 more STRING(255) reads (idx0 above makes 22 total = DLM's count). Cycle the real
            // referenced indices so each returns data; DLM likewise re-reads indices.
            let mut nok = 0usize;
            for k in 0..21usize {
                let i = if ni > 0 {
                    idxs[k % ni] as u16
                } else {
                    1 + k as u16
                };
                if dev
                    .control_recv(
                        0x06,
                        0x80,
                        0x0300 | i,
                        langid,
                        &mut tmp,
                        timeout(),
                        GFP_KERNEL,
                    )
                    .is_ok()
                {
                    nok += 1;
                }
            }
            pr_info!(
                "vino: libusb-open enum: config 618x3 + 40x3, langid={langid:#06x}, strings 22 ({nok} ok of {ni} refs)\n"
            );
        }

        // SET_INTERFACE: DLM's two handshake SET_INTERFACEs target iface 1 (alt 0,
        // app-specific/DFU) then iface 0 (alt 0, vendor) -- confirmed by a clean cold
        // DLM usbmon capture (captures/dlm-cold-20260611-123347, t=52.079/52.085).
        // The old code set iface 4 (the microphone) which DLM NEVER touches in the
        // handshake (the 58 audio SET_INTERFACEs in a session are snd-usb-audio's, not
        // DLM's -- see project_cp_setinterface_is_audio_binding_fix).
        //
        // BEHAVIOUR CHANGE, not yet re-verified on hardware (2026-07-22 v3 port): only the
        // iface-0 SET_INTERFACE is issued now. The USB bindings deliberately no longer expose a
        // device-wide `set_interface(interface, alt)` -- a driver bound to one interface of a
        // composite device must not retarget a sibling -- so the iface-1 poke cannot be expressed
        // from here. Both calls were already logged as non-fatal and the dock engaged without
        // either being checked, so this is expected to be inert; if a cold plug regresses, the
        // iface-1 SET_INTERFACE belongs in the driver instance that actually binds interface 1.
        match dev.set_alternate_setting(0) {
            Ok(()) => pr_info!("vino: step set_interface(0,0) OK\n"),
            Err(e) => pr_info!("vino: step set_interface(0,0) non-fatal ({e:?})\n"),
        }
        // vendor_out 0x24 then vendor_in 0x22 (state read, wValue=1 -- DLM's exact value;
        // wValue=0 STALLs). DLM's cold all-bus capture (dlm-cold-3426-20260714) sends exactly ONE
        // 0x24 here pre-session-init: wValue=3, then its 0x22 read. DLM's *only* 0x24 wValue=0
        // comes far later -- 20 ms AFTER the dock's first 0x45 ack (t=22.156 ack -> t=22.176 w=0),
        // i.e. it is a POST-engagement command, issued at the post-msg0 site below to match DLM.
        // (A 2026-07-14 loop that fired [3, 0] here was a regression: it sent wValue=0
        // pre-session-init, which DLM never does; the earlier all-bus diff had misread DLM's later
        // w=0 as part of this pre-arm pair. HW-retested with the corrected hdcp_seq counters: the
        // 0x24 wValue is not the CP gate either way -- 0 acks with w=3, w=0, or [3, 0].) Windows'
        // leaner device-open uses wValue=0 for this single call (WINCAP), so [`WINDOWS_MIMIC`]
        // picks it; all best-effort, the dock advances state regardless.
        let w24: u16 = if WINDOWS_MIMIC { 0 } else { 3 };
        match dev.control_send(0x24, VENDOR_OUT, w24, 0, &[], timeout(), GFP_KERNEL) {
            Ok(()) => pr_info!("vino: step 0x24(wValue={w24}) OK\n"),
            Err(e) => pr_info!("vino: step 0x24(wValue={w24}) non-fatal ({e:?})\n"),
        }
        // 0xc1 = IN|vendor|INTERFACE recipient (NOT 0xc0, device recipient): DLM's cold capture
        // uses bmRequestType=0xc1, wIndex=0 (interface 0). wValue=1 (DLM's value; 0 stalls). Uses
        // the function-scope `VENDOR_IN_IFACE` from the device-open preamble.
        let mut state = [0u8; 28];
        match dev.control_recv(
            0x22,
            VENDOR_IN_IFACE,
            1,
            0,
            &mut state,
            timeout(),
            GFP_KERNEL,
        ) {
            Ok(()) => pr_info!("vino: step 0x22(wValue=1) OK = {:02x?}\n", state),
            Err(e) => pr_info!("vino: step 0x22(wValue=1) non-fatal ({e:?})\n"),
        }

        // Plaintext session init (sec 4) in DLM's exact wire order. The dock only
        // ACKs once init_4+probe arrives, and it gates on DLM's fingerprint -- the
        // interleaved GET_DESCRIPTOR reads (CONFIGURATION before init_0, two STRING
        // reads between init_25 and init_4). Those reads are best-effort: the
        // kernel reports EREMOTEIO on the short reply but the request still hits the
        // wire (all we need). init_0/init_25/init_4+probe are separate transfers.
        const STD_IN: u8 = 0x80; // dev->host, standard, device
        let mut desc = KVec::from_elem(0u8, 618, GFP_KERNEL)?;
        let _ = dev.control_recv(
            0x06,
            STD_IN,
            0x0200,
            0,
            &mut desc[..40],
            timeout(),
            GFP_KERNEL,
        ); // CONFIG, 40
        let _ = dev.control_recv(0x06, STD_IN, 0x0200, 0, &mut desc, timeout(), GFP_KERNEL); // CONFIG, 618

        // Log EP02's bulk wMaxPacketSize from the config descriptor. If it is 64 then a 64-byte
        // msg0/arm is an exact multiple and the in-kernel `usb_bulk_msg` path (unlike libusb's
        // LIBUSB_TRANSFER_ADD_ZERO_PACKET) won't auto-append the terminating ZLP -- the dock's SIE
        // would then wait for more data and never hand the frame to firmware. Rules the ZLP-trap
        // hypothesis in or out from data we already capture. Walk the standard descriptor chain
        // (bLength/bDescriptorType), find the ENDPOINT (0x05) descriptor for bEndpointAddress 0x02.
        {
            let total = ((desc[2] as usize) | ((desc[3] as usize) << 8)).min(desc.len());
            let mut i = 0usize;
            while i + 2 <= total {
                let blen = desc[i] as usize;
                if blen == 0 {
                    break;
                }
                if desc[i + 1] == 0x05 && i + 7 <= total && desc[i + 2] == EP_CTRL_OUT {
                    let wmax = (desc[i + 4] as u16) | ((desc[i + 5] as u16) << 8);
                    pr_info!("vino: EP02 bulk wMaxPacketSize = {wmax} (ZLP needed if msg0 is a multiple)\n");
                }
                i += blen;
            }
        }

        let load_bearing = |label: &str, msg: &[u8]| -> Result {
            match dev.ctrl_send(msg, timeout(), GFP_KERNEL) {
                Ok(_) => Ok(pr_info!("vino: step {label} OK ({} B)\n", msg.len())),
                Err(e) => {
                    pr_err!("vino: step {label} FAILED ({e:?})\n");
                    Err(e)
                }
            }
        };
        load_bearing("init_0", &proto::init_0()?)?;
        // Pad init_0->init_25 to DLM's cadence (sync bulk_send fires ~0.08 ms quicker than DLM's
        // libusb URB). See PAD_* docs. udelay = us-precise busy-wait.
        if DLM_FIXED_TIMERS {
            udelay(Delta::from_micros(PAD_INIT0_TO_INIT25_US));
        }
        load_bearing("init_25", &proto::init_25()?)?;
        // DLM's two interleaved STRING reads between init_25 and init_4+probe.
        let _ = dev.control_recv(
            0x06,
            STD_IN,
            0x0300,
            0x0000,
            &mut desc[..255],
            timeout(),
            GFP_KERNEL,
        ); // STRING #0
        let _ = dev.control_recv(
            0x06,
            STD_IN,
            0x0303,
            0x0409,
            &mut desc[..255],
            timeout(),
            GFP_KERNEL,
        ); // STRING #3 en-US
           // Pad init_25->init_4 to DLM's cadence (~0.18 ms; vino's STRING reads return quicker too).
        if DLM_FIXED_TIMERS {
            udelay(Delta::from_micros(PAD_INIT25_TO_INIT4_US));
        }
        load_bearing("init_4+probe", &proto::init_4_probe()?)?;
        // probe_resend: a standalone re-send of the probe body (inner id=0x14/0x76) right after
        // init_4+probe. MEASURED HARMFUL + non-DLM (2026-07-12 wire diff): DLM's real cold-ref
        // (`cold-ref-20260608`) sends NOTHING between init_4+probe and the session-init ACK -- it
        // goes straight init_4+probe -> ACK (0.15 ms) -> AKE_Init. vino's probe_resend made the dock
        // take ~100 ms to ACK AND reply with id=0x14/0x76 instead of DLM's id=0x15/0x90 -- i.e. it
        // pushed the dock down a different path before the AKE even started. Gated OFF so the pre-arm
        // wire matches DLM; flip to A/B if a future capture shows the current firmware wants it.
        const SEND_PROBE_RESEND: bool = false;
        if SEND_PROBE_RESEND {
            if let Ok(f) = proto::probe_resend() {
                match dev.ctrl_send(&f, timeout(), GFP_KERNEL) {
                    Ok(_) => pr_info!("vino: step probe_resend OK\n"),
                    Err(e) => pr_info!("vino: step probe_resend non-fatal ({e:?})\n"),
                }
            }
        }

        // Read the single ACK that follows init_4+probe.
        let mut ack = KVec::from_elem(0u8, 1024, GFP_KERNEL)?;
        match dev.ctrl_recv(&mut ack, timeout(), GFP_KERNEL) {
            Ok(n) => pr_info!(
                "vino: session-init ACK = {n} bytes: {:02x?}\n",
                &ack[..n.min(40)]
            ),
            Err(e) => {
                pr_err!("vino: session-init ACK read FAILED ({e:?})\n");
                return Err(e);
            }
        }

        Ok(())
    }

    /// Whether to service EP83 (interrupt-IN status) during bring-up. Measured 2026-06-16
    /// (paired-coldbus-20260616-162650): DLM polls EP83 0x in the pre-arm window (14x total, all
    /// post-engagement) while vino polled it 5x pre-arm -- injecting interrupt-IN traffic into the
    /// critical arm/msg0 window that DLM never generates. Disabled so the pre-arm wire matches DLM;
    /// re-enable if a post-engagement status channel is ever needed (DLM only services it once the
    /// dock has already acked).
    const POLL_EP83_DURING_BRINGUP: bool = false;

    /// CP_STREAM_TYPE0 experiment (2026-06-23, check.md panel Gemini #4 / Grok #3): send a single
    /// Type-0 (unrestricted) stream in `RepeaterAuth_Stream_Manage` instead of the DLM-replicated
    /// 0x04/0x05 stream-type bytes, to test whether the dock engages CP as a terminal Type-0 sink
    /// (vs an HDCP-2.2 repeater). Speculative: vino's Stream_Manage already matches DLM byte-exact
    /// and DLM engages, so this DIVERGES from the proven-good default -- keep `false` for normal
    /// runs and the paired diff; flip only for the A/B cold plug. M (`wait_cap_complete`) is
    /// host-verify-only so its value never gates, but its `m_data` tracks this flag for a clean log.
    const CP_STREAM_TYPE0: bool = false;

    /// Number of display heads the post-msg0 CP setup burst re-states the AKE for. Tied to the
    /// single head-count knob `drm_sink::HEADS` so bumping the head count is a one-line change
    /// (was a duplicated literal `2` that had to be kept in sync by hand).
    const CP_SETUP_HEADS: usize = drm_sink::HEADS;

    /// Reads the next HDCP response (type=4 sub=0x25, sec 5.2) from EP `0x84`,
    /// skipping any non-HDCP frames (e.g. plain ACKs) in between, and returns the
    /// parsed `(msg_id, payload)`. Bounded retry so a chatty dock can't wedge us.
    fn recv_hdcp(dev: &UsbLink<'_>) -> Result<(u8, KVec<u8>)> {
        const SUB_HDCP_RESP: u16 = 0x25;
        // The dock interleaves capability blocks up to ~5.8 KiB into the AKE reply
        // stream; size the buffer like the rest of the EP84 reads ([`EP84_BUF`]) so a
        // large frame is read whole rather than truncated/`-EOVERFLOW`'d.
        let mut buf = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL)?;
        for _ in 0..24 {
            // Read EP84 FIRST. The dock replies to AKE messages sub-millisecond (DLM cold capture:
            // ~0.1-0.7 ms between EP84 IN frames), but it interleaves status/cap pushes that we
            // skip. Polling EP83 (a ~2 ms idle wait) BEFORE every read added ~2 ms x
            // N-skipped-frames
            // of latency per reply -- making vino's AKE ~400 ms vs DLM's ~62 ms, slow enough that
            // the
            // dock starts downstream HDCP and NAKs our arm/Stream_Manage. So only service EP83 when
            // EP84 came back empty (same reorder as `drain_ep84`). See the cold wire diff.
            let n = dev.ctrl_recv(&mut buf, timeout(), GFP_KERNEL)?;
            if n < 16 {
                if Self::POLL_EP83_DURING_BRINGUP {
                    Self::poll_ep83(dev);
                }
                continue;
            }
            // DIAGNOSTIC (2026-06-11): log EVERY frame the dock returns during the AKE --
            // including
            // wsub!=0x25 and cap-block (sub=0x84) pushes we'd otherwise skip -- so we can see
            // whether
            // the dock interleaves its capability blocks with the HDCP replies (the suspected
            // reason
            // its cap phase never completes / it won't engage CP). Inner id/sub at off 16/18.
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
                pr_debug!(
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

    /// Pace like DLM after a RepeaterAuth OUT (ctr6 Send_Ack / ctr7 Stream_Manage):
    /// read the dock's per-frame `id=0x14 sub=0x10` ack off EP84 BEFORE the next OUT,
    /// so vino never transmits while the dock is mid-NAK.
    ///
    /// Ground truth (cold wire diff, captures/dlm-cold-20260611-123347 vs vino-cold):
    /// DLM reads that ack after EVERY cap/AKE OUT --
    /// ctr4->ack->ctr5->ack->ctr6->ack->ctr7->
    /// ack->arm, ~0.2 ms apart, whole ctr7->arm gap 0.46 ms. Commit d74a4d7 dropped the
    /// drain for ctr6/ctr7, so `run_ake` sent ctr6->ctr7 back-to-back with no read; the
    /// dock (busy with downstream HDCP after SKE) then NAK'd each OUT ~100 ms (vino's
    /// V'->arm gap measured ~200 ms), and the arm landed after the dock had left its
    /// freshly-keyed CP window -> CP never engaged (0 `wsub=0x45`). Restoring the read
    /// re-paces vino to DLM and lets the arm land tight. Best-effort: returns as soon as
    /// the matching ack arrives, or immediately if nothing is queued (dock idle).
    fn pace_cap_ack(dev: &UsbLink<'_>, want_ctr: u16) {
        // EP84 frames here can carry an interleaved capability block up to ~5.8 KiB;
        // size to [`EP84_BUF`] so a large frame isn't truncated mid-pacing.
        let Ok(mut buf) = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL) else {
            return;
        };
        for _ in 0..8 {
            match dev.ctrl_recv(&mut buf, Delta::from_millis(30), GFP_KERNEL) {
                Ok(len) if len >= 22 => {
                    let wsub = u16::from_le_bytes([buf[8], buf[9]]);
                    let iid = u16::from_le_bytes([buf[16], buf[17]]);
                    let isub = u16::from_le_bytes([buf[18], buf[19]]);
                    let ictr = u16::from_le_bytes([buf[20], buf[21]]);
                    // If this drain happens to pull the dock's terminal cap-complete push
                    // (id=0x0b sub=0x84) off the wire ahead of the ctr echo, record it in the
                    // shared flag so `wait_cap_complete` doesn't miss it (see [`SAW_0B`]).
                    if iid == 0x0b && isub == 0x84 {
                        SAW_0B.store(true, core::sync::atomic::Ordering::Release);
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

    /// After ctr7 (Stream_Manage) and its ack, WAIT for the dock's terminal capability block
    /// `id=0x0b sub=0x84` before letting the caller arm. This is the dock's "cap-complete"
    /// signal: DLM receives it and only then arms (cold-ref: `id=0x21` @52.1465 -> `id=0x0b`
    /// @52.1469 -> arm @52.1474). vino's lockstep ([`pace_cap_ack`]) only consumed the `id=0x14`
    /// ctr acks, so it armed right after ctr7's ack -- BEFORE the dock had emitted `id=0x0b`
    /// (vino received every other cap block id=0x213/0x0d/0x10/0x28/0x18/0x21 but armed one push
    /// early). The dock then NAK'd msg0 ~100 ms and dumped a 16 KB error block
    /// (`type=0x1003 wsub=0x37`) that DLM never produces, instead of engaging CP -- the true
    /// gate, found on cold plug `vino-cold-20260612-080549`. The dock emits `id=0x0b` a few ms
    /// after `id=0x21` once it settles downstream HDCP, so draining EP84 until it arrives keeps
    /// the arm tight (DLM ~ 0.5 ms after ctr7) yet correctly ordered. Best-effort, bounded.
    fn wait_cap_complete(dev: &UsbLink<'_>, kd: &[u8; 32]) {
        let Ok(mut buf) = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL) else {
            return;
        };
        // Drain EP84 until the dock goes QUIET, not merely until id=0x0b. Cold plug #2
        // (vino-cold-20260612-082707) showed DLM's LAST pre-arm push is the id=0x28 that
        // follows id=0x0b (cold-ref: id=0x0b@52.1469 -> ack ctr7 -> id=0x28@52.1472 ->
        // arm@52.1474),
        // whereas vino stopped at id=0x0b and armed -- leaving id=0x28 (and the rest of the dock's
        // terminal cap burst) un-drained in the dock's EP84 queue. With its IN queue backed up the
        // dock NAK'd vino's msg0 ~100 ms (it can't accept the OUT while it still owes IN data) and
        // then dumped the 16 KB error block. So after id=0x0b, keep reading until a read times out
        // (the dock has sent everything), then return so the caller arms into a clean dock -- like
        // DLM. Bounded: id=0x0b is the marker; QUIET_GAP short reads of silence end the drain.
        //
        // * 2026-06-12 (HDCP 2.3 Adaptation sec RepeaterAuth, pdfs/): one of the frames drained
        // here is
        // the dock's `RepeaterAuth_Stream_Ready` (HDCP msg 0x11) -- the 3rd `id=0x28` DLM receives
        // and
        // vino historically did not. The spec requires the transmitter to RECEIVE it within 100 ms
        // of
        // `Stream_Manage` and verify `M == M'` before transmitting content; the dock's exactly-100
        // ms
        // msg0 NAK on a cold plug is that window. We now RECOGNISE it in this same drain (no added
        // latency vs the old broken 10x1 s poll) and log `M'` plus candidate `M`s so the next
        // capture
        // pins the exact `STREAMID_TYPE || seq_num_M` the dock hashes. The HDCP msg_id rides at
        // `body[9]` = `buf[25]` in an EP84 reply (`ake::parse_in`); `M'[32]` follows at
        // `buf[26..58]`.
        // Verification is logged-only for now (the DisplayLink field offsets in `Stream_Manage` are
        // not yet confirmed, so a wrong guess must not block the arm); the arm is gated on
        // receiving
        // Stream_Ready when it arrives, else on the existing id=0x0b + quiet fallback. `M` key is
        // `SHA256(kd)`; `M = HMAC-SHA256(STREAMID_TYPE || seq_num_M, SHA256(kd))`, seq_num_M = 0.
        let sha_kd = crypto::sha256(kd);
        // Seed from the shared flag: `pace_cap_ack` (ctr7/ctr8 drain) may have already pulled the
        // dock's `id=0x0b` terminal push off the wire ahead of us (see [`SAW_0B`]).
        let mut saw_0b = SAW_0B.load(core::sync::atomic::Ordering::Acquire);
        let mut saw_ready = false;
        let mut quiet = 0usize;
        const QUIET_GAP: usize = 3; // ~3 consecutive empty short reads => dock done pushing
        const MAX_ROUNDS: usize = 48;
        for _ in 0..MAX_ROUNDS {
            match dev.ctrl_recv(&mut buf, Delta::from_millis(5), GFP_KERNEL) {
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
                        pr_info!("vino: AKE: Stream_Ready (0x11) M'={mprime:02x?}\n");
                        // M = HMAC-SHA256(SHA256(kd), data) where data is the Content Stream
                        // Management input the dock hashes: `k` 7-byte stream entries followed by
                        // the 3-byte `seq_num_M` (=0 on the first Stream_Manage). Cracked from the
                        // DLM aarch64 decompile (`FUN_0057be04`: data = memcpy(streams, k*7) ||
                        // BE16(field) || field, keyed by the 32-byte SHA256(kd) at session+0x37);
                        // reproduces DLM's captured M' byte-exact (captures/.../FINDINGS.md).
                        // vino's
                        // two streams carry the same StreamID_Type bytes its Stream_Manage sends
                        // (`repeater_auth_stream_manage`: type 0x04 and 0x05), so the dock computes
                        // the same M. (Earlier code guessed a 5-byte STREAMID_TYPE||seq layout and
                        // so
                        // always mismatched -- host-side only, never gated the dock.)
                        // Stream-type bytes track CP_STREAM_TYPE0 so the logged M matches what
                        // Stream_Manage actually sent (M is host-verify-only; never gates the dock).
                        let (s0, s1) = if Self::CP_STREAM_TYPE0 {
                            (0x00, 0x00)
                        } else {
                            (0x04, 0x05)
                        };
                        let m_data: [u8; 17] = [
                            0, 0, 0, s0, 0, 0, 0, // stream 0: StreamID_Type[0]
                            0, 0, 0, s1, 0, 0, 0, // stream 1: StreamID_Type[1]
                            0, 0, 0, // seq_num_M = 0 (first Stream_Manage, big-endian)
                        ];
                        let m = crypto::hmac_sha256(&sha_kd, &m_data);
                        let eq = if &m[..] == mprime { "==" } else { "!=" };
                        pr_info!("vino: AKE:   M {} M' (CSM stream-entry layout)\n", eq);
                    } else if mid == ake::id::RECEIVER_AUTH_STATUS && len >= 27 {
                        pr_info!("vino: AKE: RECEIVER_AUTH_STATUS=0x{:02x}\n", buf[26]);
                    }
                    // * 2026-06-12: arm the INSTANT both terminal markers have arrived -- the
                    // cap-complete
                    // id=0x0b AND the Stream_Ready (the trailing id=0x28 / HDCP 0x11). DLM arms
                    // 0.46 ms
                    // after its last cap block; a cold-plug cadence diff
                    // (vino-cold-20260612-113706) showed
                    // vino was instead waiting QUIET_GAP x 5 ms of EMPTY reads AFTER already
                    // seeing both
                    // markers, landing the arm ~68 ms late -- outside the dock's freshly-keyed CP
                    // window, so
                    // the dock errored on the arm (27 KB type=0x1001 dump) instead of engaging.
                    // Once both
                    // markers are in, the terminal burst is complete; arm now, like DLM. (The
                    // empty-read
                    // quiet path below remains the fallback when Stream_Ready never arrives.)
                    if saw_0b && saw_ready {
                        pr_info!(
                            "vino: cap-complete (id=0x0b + Stream_Ready 0x11) -- arming now\n"
                        );
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
                            pr_info!(
                                "vino: cap-complete drained (id=0x0b{}+ quiet) -- arming now\n",
                                if saw_ready {
                                    ", Stream_Ready 0x11, "
                                } else {
                                    " (no 0x11) "
                                }
                            );
                            return;
                        }
                    }
                }
            }
        }
        pr_info!(
            "vino: cap-complete drain budget hit (saw_0b={saw_0b} saw_ready={saw_ready}) -- arming anyway\n"
        );
    }

    /// Drives a full clean-room HDCP 2.2 AKE + LC + SKE (and RepeaterAuth for a
    /// repeater sink) over EP `0x02`/`0x84`, verifying `H'`, `L'` and `V'` against
    /// our own KDF (sec 5). On success returns the [`Session`] keys.
    ///
    /// **This send stream IS the plaintext capability-announce** (unified 2026-07-12). All OUT
    /// messages are `type=4 sub=0x04` frames whose inner counter `hdcp_seq` runs 1..8, interleaved
    /// with the dock's cert/`H'`/`L'`/`V'`/Stream_Ready replies:
    ///
    /// * ctr=1 session-init ACK (id=0x14/0x76), ctr=2 AKE_Init, ctr=3 AKE_Transmitter_Info
    /// * ctr=4 AKE_No_Stored_km, ctr=5 LC_Init, ctr=6 SKE_Send_Eks
    /// * ctr=7 RepeaterAuth_Send_Ack, ctr=8 RepeaterAuth_Stream_Manage  (then msg0 at ctr=9)
    ///
    /// This replaces the old scheme where the AKE was sent here AND a duplicate "restatement"
    /// (`build_cap_announce`) was re-sent in `send_cp_setup` -- the dock saw the AKE twice. The
    /// restatement is gone; `send_cp_setup` now sends only the encrypted arm + msg0 + burst.
    ///
    /// NOTE on the ctr=1 session-init ACK (id=0x14/0x76): CORRECTED 2026-07-14. A prior note held
    /// that this frame was a RE-AUTH-only artifact and that sending it on a cold plug made the dock
    /// reply id=0x3 instead of the cert -- but that was measured on the PRE-3.4.26 cold-ref. The
    /// fresh current-firmware cold references (`captures/dlm-cold-3426-20260713-*`) show DLM DOES
    /// send it on a cold plug, with the AKE at ctr 2..8 and msg0 at ctr=9. vino now matches.
    fn run_ake(dev: &UsbLink<'_>) -> Result<Session> {
        use ake::id;

        // Anchor the CP-start instant (~`cp_first` in the timing survey) before the first frame,
        // so `send_cp_setup` can realise DLM's fixed pre-arm timer and log the fingerprint.
        let cp_start = Instant::<Monotonic>::now();

        // Flush any STALE EP84 frames the dock still has queued from a PRIOR session before
        // starting a fresh AKE. On a warm rmmod/insmod re-probe the dock is not power-cycled, so
        // its previous CP/cap replies (including a multi-KB residual block) sit in its EP84 queue;
        // if we don't drain them, the first `recv_hdcp` picks up a stale frame and the whole AKE
        // reply stream is shifted.
        //
        // TIMING (2026-06-23 paired cold-plug diff): a stale frame is ALREADY queued in the dock,
        // so it returns sub-millisecond; only the trailing empty read pays the timeout. A 20 ms
        // probe therefore cost a dead 20 ms before AKE_Init on a *cold* plug (queue empty) -- the
        // sole pre-arm gap where vino diverged from DLM (DLM emits ctr1 ~0.3 ms after session-init,
        // vino was ~21.8 ms). Drop the probe to 3 ms: still ample to drain a warm-reprobe backlog
        // (each present frame returns immediately; the loop only stops on the empty read), but the
        // cold-plug cost collapses 20 ms -> 3 ms, matching DLM's cadence into the AKE.
        //
        // FIXED-TIMER (2026-06-26, `DLM_FIXED_TIMERS`): DLM does NOT flush here at all -- the
        // cp_first->cert frame dump shows it emits AKE_Init ~0.14 ms after the session-init ACK,
        // whereas vino blocked ~1.9 ms in this probe. The cause: on a COLD plug the EP84 queue is
        // empty (the session-init ACK was already consumed), so the first `bulk_recv` waits out its
        // full timeout -- and because a sub-ms `Delta` truncates via `as_millis()` to 0 = "wait
        // forever", the practical floor is ~1.9 ms wall even at a nominal 1 ms. That dead wait was
        // the ENTIRE residual `cp_first->cert` gap vs DLM (vino 2.6 ms, DLM 1.07 ms). The flush only
        // matters on a WARM rmmod/insmod re-probe, where the un-power-cycled dock still has a stale
        // CP/cap backlog queued that would shift the first `recv_hdcp`. So under DLM_FIXED_TIMERS
        // (the cold-plug DLM-impersonation mode) skip it entirely and let AKE_Init follow the ACK as
        // tightly as DLM; keep the drain on the reactive/warm path where stale frames are possible.
        if !DLM_FIXED_TIMERS {
            let flush_probe = Delta::from_millis(3);
            if let Ok(mut flush) = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL) {
                let mut flushed = 0usize;
                for _ in 0..32 {
                    match dev.ctrl_recv(&mut flush, flush_probe, GFP_KERNEL) {
                        Ok(n) if n > 0 => flushed += 1,
                        _ => break,
                    }
                }
                if flushed > 0 {
                    pr_info!("vino: flushed {flushed} stale EP84 frame(s) before AKE\n");
                }
            }
        }

        // Pad the dock's hello-ACK -> first cap frame (the session-init ACK sent just below) to
        // DLM's 0.159 ms cadence. With the cold-plug flush gone, vino fires it ~0.043 ms after the
        // ACK; DLM spaces it 0.159 ms. See PAD_* docs.
        if DLM_FIXED_TIMERS {
            udelay(Delta::from_micros(PAD_ACK_TO_AKEINIT_US));
        }

        // Running inner counter (`hdcp_seq`), incremented per OUT frame and carried into
        // `Session::next_ctr` so nothing downstream hardcodes a value (hardcoded ctrs caused past
        // off-by-one regressions). CORRECTED 2026-07-14: DLM's current-firmware (3.4.26) cold plug
        // opens the cap/AKE stream with a session-init ACK at ctr=1 (id=0x14/0x76), then AKE_Init..
        // Stream_Manage at ctr 2..8, and msg0 at ctr=9 -- byte-verified against the fresh cold
        // references (captures/dlm-cold-3426-20260713-*). The old "ctr=1 hello is a RE-AUTH-only
        // artifact, sending it on a cold plug makes the dock reply id=0x3" note was based on the
        // PRE-3.4.26 cold-ref; the firmware changed the cold sequence and vino had not adapted.
        let mut hseq: u32 = 1;

        // (1) session-init ACK (ctr=1, id=0x14/0x76).
        dev.ctrl_send(&ake::session_init_ack(hseq, 0)?, timeout(), GFP_KERNEL)?;
        // Drain the dock's ctr1 echo BEFORE sending AKE_Init. The dock enforces OUT->IN lockstep in
        // the cap phase: xHCI shows the AKE_Init OUT is NAK'd for ~100 ms (vs ~60 us for every other
        // cap-phase OUT) when the host sends it before reading the ctr1 IN echo. vino's synchronous
        // `bulk_send(AKE_Init)` blocks on that OUT while the dock waits for the host to read the echo
        // -- a deadlock the dock only breaks with a ~100 ms firmware timeout, which then corrupts its
        // CP state (all later encrypted CP dropped). DLM drains this echo between the two OUTs (~0.2 ms
        // after ctr1); vino was blasting ctr1->ctr2 back-to-back. `pace_cap_ack` reads EP84 until the
        // dock's `id=0x14 ctr=1` frame arrives. (Found via xHCI URB-completion timing, 2026-07-14.)
        Self::pace_cap_ack(dev, hseq as u16);
        hseq += 1;

        // (2) AKE_Init -- fresh rtx, TxCaps = 00 00 00 (DLM-exact).
        let mut rtx = [0u8; 8];
        rng::fill(&mut rtx);
        dev.ctrl_send(
            &ake::ake_init(hseq, 0, &rtx, &[0; 3])?,
            timeout(),
            GFP_KERNEL,
        )?;
        hseq += 1;

        // (2) AKE_Send_Cert: payload = REPEATER(1) || cert_rx(522). Extract the
        // RSA-1024 public key (modulus[5..133], exponent[133..136]).
        let (cid, cert_msg) = Self::recv_hdcp(dev)?;
        // Anchor the `cert` milestone the instant the dock's cert lands: DLM arms a fixed
        // CERT_TO_ARM_US after this point (see `Session::cert_at` / `DLM_FIXED_TIMERS`).
        let cert_at = Instant::<Monotonic>::now();
        if cid != id::AKE_SEND_CERT || cert_msg.len() < 1 + 136 {
            pr_err!(
                "vino: AKE: bad AKE_Send_Cert (id={cid:#x}, {} B)\n",
                cert_msg.len()
            );
            return Err(EINVAL);
        }
        let repeater = cert_msg[0] != 0;
        let cert = &cert_msg[1..];
        let mut modulus = [0u8; 128];
        modulus.copy_from_slice(&cert[5..133]);
        let mut exponent = [0u8; 3];
        exponent.copy_from_slice(&cert[133..136]);

        // (3) AKE_Transmitter_Info (ctr=3), then read AKE_Receiver_Info (RxCaps unused).
        dev.ctrl_send(&ake::ake_transmitter_info(hseq, 0)?, timeout(), GFP_KERNEL)?;
        hseq += 1;
        let xmit_info_at = Instant::<Monotonic>::now();
        let _ = Self::recv_hdcp(dev)?;

        // (5) AKE_No_Stored_km -- fresh km, RSA-OAEP-SHA256 to Ekpub(km).
        let mut km = [0u8; 16];
        rng::fill(&mut km);
        let ekpub = hdcp::oaep_encrypt_km(&modulus, &exponent, &km)?;
        // Spend a REALISTIC cert-verification time before No_Stored_km. vino reaches this point
        // ~0.3 ms after AKE_Transmitter_Info; DLM takes ~1.65 ms (it verifies the receiver's
        // DCP-signed cert). Hold to DLM's cadence so vino doesn't answer impossibly fast -- the one
        // consistent host-reachable divergence found vs the same-day engaging DLM. See
        // [`CERT_VERIFY_HOLD_US`].
        if DLM_FIXED_TIMERS {
            hold_until(xmit_info_at, CERT_VERIFY_HOLD_US);
        }
        // (4) AKE_No_Stored_km (ctr=4). After this the dock runs downstream HDCP (~154 ms in DLM's
        // wire) before it answers, so the following `recv_hdcp` blocks out that pause naturally.
        dev.ctrl_send(
            &ake::ake_no_stored_km(hseq, 0, &ekpub)?,
            timeout(),
            GFP_KERNEL,
        )?;
        hseq += 1;

        // (6) AKE_Send_Rrx.
        let (rid, rrx_pl) = Self::recv_hdcp(dev)?;
        if rid != id::AKE_SEND_RRX || rrx_pl.len() < 8 {
            pr_err!("vino: AKE: bad AKE_Send_Rrx (id={rid:#x})\n");
            return Err(EINVAL);
        }
        let mut rrx = [0u8; 8];
        rrx.copy_from_slice(&rrx_pl[..8]);

        // (7)/(8) AKE_Send_H_prime -- verify H' = HMAC(kd, rtx^REPEATER).
        let (hid, hp) = Self::recv_hdcp(dev)?;
        if hid != id::AKE_SEND_H_PRIME || hp.len() < 32 {
            pr_err!("vino: AKE: bad H' (id={hid:#x})\n");
            return Err(EINVAL);
        }
        let kd = hdcp::derive_kd(&km, &rtx, &rrx)?;
        if hdcp::compute_h(&kd, &rtx, repeater)[..] != hp[..32] {
            pr_err!("vino: AKE: H' mismatch -- authentication failed\n");
            return Err(EINVAL);
        }
        pr_info!("vino: AKE: H' verified\n");

        // (9) AKE_Send_Pairing_Info (Ekh_km) -- read and discard (no-stored path).
        let _ = Self::recv_hdcp(dev)?;

        // (10) Locality Check -- LC_Init(rn) then verify L'.
        let mut rn = [0u8; 8];
        rng::fill(&mut rn);
        // (5) LC_Init (ctr=5).
        dev.ctrl_send(&ake::lc_init(hseq, 0, &rn)?, timeout(), GFP_KERNEL)?;
        hseq += 1;
        let (lid, lp) = Self::recv_hdcp(dev)?;
        if lid != id::LC_SEND_L_PRIME || lp.len() < 32 {
            pr_err!("vino: AKE: bad L' (id={lid:#x})\n");
            return Err(EINVAL);
        }
        if hdcp::compute_l(&kd, &rrx, &rn)[..] != lp[..32] {
            pr_err!("vino: AKE: L' mismatch -- locality check failed\n");
            return Err(EINVAL);
        }
        pr_info!("vino: AKE: L' verified\n");

        // (11) Session Key Exchange -- send Edkey(ske_ks) || riv. `ske_ks` is the fresh-random SKE
        // key the dock unwraps from Edkey; the LIVE CP session key is `ske_ks XOR cp::CP_KEY_WHITEN`
        // (reverse-engineered 2026-07-16 -- see cp::cp_session_key). Wrapping delivers the raw
        // `ske_ks`; both host and dock then whiten it with the same device constant.
        let mut ske_ks = [0u8; 16];
        let mut riv = [0u8; 8];
        rng::fill(&mut ske_ks);
        rng::fill(&mut riv);
        let edkey = hdcp::compute_eks(&km, &rtx, &rrx, &rn, &ske_ks)?;
        // The CP AES-CTR + Dl3Cmac key: NOT the raw SKE key, but `ske_ks XOR CP_KEY_WHITEN`. vino had
        // keyed CP with the raw random, so the dock rejected every OUT Dl3Cmac (the engagement wall).
        let ks = cp::cp_session_key(&ske_ks);
        // Dev diagnostic: the full SKE secrets, so the live SKE wrap/whiten chain can be verified
        // OFFLINE against the rr-proven model -- edkey MUST equal `ske_ks XOR derive_dkey(km,rtx,rrx,
        // rn,2)` (so the dock unwraps exactly `ske_ks`), and cp_key MUST equal `ske_ks XOR B`. If
        // either fails, vino's live msg0 is sealed under a key the dock can't reconstruct (a plumbing
        // bug), independent of the proven crypto model. **pr_debug** (compiled out unless dynamic
        // debug is enabled): these leak per-session secrets to the kernel log, so they must NOT be
        // pr_info in normal operation. Enable via `dyndbg` when the SKE chain needs verifying.
        pr_debug!("vino: SKE-SECRETS km={km:02x?} rtx={rtx:02x?} rrx={rrx:02x?} rn={rn:02x?}\n");
        pr_debug!("vino: SKE-SECRETS ske_ks={ske_ks:02x?} edkey={edkey:02x?} cp_key={ks:02x?}\n");
        // * riv DERIVATION -- ground-truthed by rr reverse-execution of DLM 3.4.26's REAL msg0
        // ENCRYPT (2026-07-16): the SKE delivers the FULL random riv as-is (DLM masks nothing), and
        // the AES-CTR content nonce = delivered with `byte7 ^= 0x04` and **byte0 UNCHANGED**. For
        // DLM's session (delivered f621dc0d227ef4ab) the trace showed the CTR counter block =
        // f621dc0d227ef4af (AES_dec(seal, keystream)) -- byte0 stays f6. The Dl3Cmac then keys on
        // that CTR nonce with a further byte0^0x80 (7621dc0d227ef4af), applied inside dl3cmac_tag.
        // ⇒ the CTR and CMAC nonces DIFFER by byte0^0x80; they are NOT one value.
        // History: the original code used `^0x01@byte7` (wrong bit). A 2026-07-16 "unification" then
        // wrongly moved byte0^0x80 INTO the CTR nonce and dropped it from the CMAC -- so vino
        // encrypted msg0 under keystream AES(seal, 7621..) instead of AES(seal, f621..); the dock
        // decrypted vino's msg0 to garbage (no id=0x14), silently dropping it (0 sub=0x45) while the
        // CMAC still self-verified. Correct: CTR nonce = delivered ^0x04@byte7 (this `session.riv`);
        // dl3cmac_tag adds byte0^0x80.
        let riv_ske = riv; // deliver the full random riv, unmasked, exactly like DLM
        riv[7] ^= 0x04; // OUT CP AES-CTR nonce = delivered ^0x04@byte7 (byte0 UNCHANGED)
                        // (6) SKE_Send_Eks (ctr=6).
        dev.ctrl_send(
            &ake::ske_send_eks(hseq, 0, &edkey, &riv_ske)?,
            timeout(),
            GFP_KERNEL,
        )?;
        hseq += 1;
        // Dev diagnostic (pr_debug -- session secrets, compiled out unless dynamic debug is on): the
        // live CP session key (ske_ks XOR CP_KEY_WHITEN) and out-riv the dock must hold to decrypt/
        // verify our CP.
        pr_debug!("vino: SESSION cp_key={ks:02x?} out_riv={riv:02x?}\n");

        // (12) RepeaterAuth -- verify V' over the ReceiverID_List, ACK, then SM2.
        // Retained (empty on the non-repeater path) so `send_cp_setup`'s per-head restatement can
        // recompute a fresh per-head `V = HMAC(kd_h, rxid_list)` over the same list the dock sent.
        let mut rxid_list: KVec<u8> = KVec::new();
        if repeater {
            let (vid, list) = Self::recv_hdcp(dev)?;
            if vid != id::REPEATERAUTH_SEND_RECEIVERID_LIST || list.len() < 16 {
                pr_err!("vino: AKE: bad ReceiverID_List (id={vid:#x})\n");
                return Err(EINVAL);
            }
            let split = list.len() - 16;
            // V = HMAC(kd, list_header): MSB-128 = V' (verify vs the list trailer);
            // LSB-128 = the RepeaterAuth_Send_Ack value (NOT the MSB -- that was THE bug).
            let v_full = hdcp::compute_v_full(&kd, &list[..split]);
            let mut v_ack = [0u8; 16];
            v_ack.copy_from_slice(&v_full[16..]);
            if v_full[..16] != list[split..] {
                pr_err!("vino: AKE: V' mismatch -- repeater verification failed\n");
                return Err(EINVAL);
            }
            pr_info!("vino: AKE: V' verified\n");
            rxid_list.extend_from_slice(&list[..split], GFP_KERNEL)?; // retain for per-head V
                                                                      // (7) RepeaterAuth_Send_Ack (ctr=7).
            dev.ctrl_send(
                &ake::repeater_auth_send_ack(hseq, 0, &v_ack)?,
                timeout(),
                GFP_KERNEL,
            )?;
            // Read the dock's ack for THIS frame before sending the next -- DLM's lockstep pacing,
            // without which the dock NAKs the back-to-back OUTs ~100 ms each (see `pace_cap_ack`).
            Self::pace_cap_ack(dev, hseq as u16);
            hseq += 1;
            // (8) RepeaterAuth_Stream_Manage (ctr=8).
            dev.ctrl_send(
                &ake::repeater_auth_stream_manage(hseq, 0, Self::CP_STREAM_TYPE0)?,
                timeout(),
                GFP_KERNEL,
            )?;
            // Read the dock's ack before returning, so the caller's arm marker lands tight after
            // this frame (DLM: 0.46 ms) instead of while the dock is still NAKing.
            Self::pace_cap_ack(dev, hseq as u16);
            hseq += 1;
            // Then drain the dock's terminal cap burst -- id=0x0b (cap-complete) AND the dock's
            // `RepeaterAuth_Stream_Ready` (HDCP 0x11, the 3rd id=0x28) -- before the caller arms.
            // DLM arms only after this burst (cold-ref: id=0x21 -> id=0x0b -> id=0x28/0x11 ->
            // arm);
            // arming early makes the dock NAK msg0 ~100 ms and dump a 16 KB error block instead of
            // engaging. `wait_cap_complete` recognises + verifies the Stream_Ready in place (HDCP
            // 2.3 Adaptation sec RepeaterAuth). `kd` is needed to check `M == M'`.
            Self::wait_cap_complete(dev, &kd);
        }

        // `hseq` now points past the last cap/AKE frame sent (9 on the repeater path: session-init
        // ACK ctr=1 + AKE ctr 2..8); `send_cp_setup` continues the inner counter from here for msg0.
        Ok(Session {
            ks,
            riv,
            kd,
            cp_start,
            cert_at,
            next_ctr: hseq as u16,
            modulus,
            exponent,
            rrx,
            rxid_list,
        })
    }

    /// Poll EP 0x83 (interrupt-IN status endpoint). DLM submits URBs here CONTINUOUSLY and the dock
    /// pushes 6-byte status events; the dock may gate CP/downstream-HDCP engagement on the host
    /// servicing this endpoint (flagged in `vino-driver/src/bin/bringup.rs`). vino never polled it
    /// --
    /// invisible in the EP02/EP84 bulk-wire comparison. Reads up to a few events (short timeout so
    /// a
    /// URB is pending when the dock pushes). `usb_bulk_msg` auto-routes the interrupt endpoint.
    fn poll_ep83(dev: &UsbLink<'_>) -> usize {
        // EP83 (interrupt-IN) transfers need DMA-capable memory -- allocate on the HEAP.
        // A stack array trips usb_hcd_map_urb_for_dma's "transfer buffer is on stack"
        // WARNING (VMAP_STACK can't be DMA-mapped) and the broken submit also stalls the
        // bring-up (poll_ep83 runs inside every drain round). Best-effort: bail on OOM.
        let mut buf = match KVec::from_elem(0u8, 64, GFP_KERNEL) {
            Ok(b) => b,
            Err(_) => return 0,
        };
        let mut n = 0usize;
        // Short timeout: a pending URB gives the dock a window to push, but a 30 ms block on the
        // (normally idle) EP83 stalls the bring-up loop (see drain_ep84). 2 ms is enough to catch a
        // ready event without serializing the handshake.
        for _ in 0..4 {
            match dev.status_recv(&mut buf, Delta::from_millis(2), GFP_KERNEL) {
                Ok(len) if len > 0 => {
                    n += 1;
                    let s = &buf[..len.min(8)];
                    pr_info!("vino: EP83 status event {len}B {s:02x?}\n");
                }
                _ => break,
            }
        }
        n
    }

    /// Reap any completed transfers from the persistent EP83 interrupt-IN queue (opened in
    /// [`send_cp_setup`]) without blocking, re-posting each as it is read so a URB always stays
    /// pending. Returns the number of status events drained (logged for the cold-plug A/B vs DLM).
    ///
    /// [`send_cp_setup`]: Self::send_cp_setup
    fn drain_ep83_queue(dev: &UsbLink<'_>, q: Option<&mut usb::BulkInQueue>) -> usize {
        let Some(q) = q else { return 0 };
        let mut buf = [0u8; 64];
        let mut n = 0usize;
        // Non-blocking sweep: a 0 ms wait returns Ok(None) immediately if nothing has completed,
        // so this never serialises the arm/msg0 burst -- it only harvests what the dock pushed.
        for _ in 0..4 {
            match q.recv(dev.io(), &mut buf, Delta::from_millis(0)) {
                Ok(Some(len)) if len > 0 => {
                    n += 1;
                    let s = &buf[..len.min(8)];
                    pr_info!("vino: EP83 (async) status event {len}B {s:02x?}\n");
                }
                _ => break,
            }
        }
        n
    }

    /// Drives the post-SKE CP setup: opens the async EP84 reader, sends the plaintext
    /// stream-open arm marker, the first live encrypted CP frame (msg0), the full post-msg0
    /// setup burst (init group + per-head AKE re-statement + stream finalize -- see
    /// `captures/rr-out-sequence-20260716/cp-dialogue-decoded.txt`), and counts the dock's
    /// encrypted `wsub=0x45` acks. `video_keys` is filled with the per-head video key generated
    /// for each `id=0x32` message in the burst (see the loop below for what is and isn't proven
    /// about its role).
    fn send_cp_setup(
        dev: &UsbLink<'_>,
        session: &Session,
        // Scratch slot the drain/send helpers fill when an `id=0x194` EDID reply arrives; the EDID
        // phase runs once per head (byte22 = head selector) and moves each capture into `edid_heads`.
        edid_out: &mut Option<KVec<u8>>,
        edid_heads: &mut [Option<KVec<u8>>; Self::CP_SETUP_HEADS],
        video_keys: &mut [[u8; 32]; Self::CP_SETUP_HEADS],
        heads_present: &mut [bool; Self::CP_SETUP_HEADS],
    ) -> Result<(usize, usize, u32, u16)> {
        // 16 KiB so the dock's ~5787 B capability block is read whole (see [`EP84_BUF`]).
        let mut resp = KVec::from_elem(0u8, EP84_BUF, GFP_KERNEL)?;
        let mut drained = 0usize;
        let mut acks = 0usize;
        let mut rejects = 0usize;
        let mut sent = 0usize;
        let mut ep83_events = 0usize;
        // Dual-monitor detection: each head's stream-open (`id=0x14 sub=0x30`) request ctr. The
        // out-param `heads_present[h]` is set true when the dock answers head h's stream-open with a
        // per-head DISPLAY-CAP (`id=0x78 sub=0x30`), i.e. a monitor is present on that head. The
        // probe uses it to mark each head's connector connected (see the caller).
        let mut stream_open_ctr: [Option<u16>; Self::CP_SETUP_HEADS] = [None; Self::CP_SETUP_HEADS];

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

        // Open the persistent async EP84 IN reader BEFORE the arm marker and msg0, so
        // `EP84_QUEUE_DEPTH` IN transfers are already posted when the dock pushes its post-arm
        // reply (DLM's libusb always-pending-IN behaviour). Draining EP84 concurrently stops the
        // dock's IN FIFO filling and NAKing our OUT (the sync-bulk deadlock that produced a 100 ms
        // msg0 NAK). RAII: dropping the queue at function exit kills+frees the URBs.
        let mut ep84_q = match dev.ctrl_in_queue(EP84_QUEUE_DEPTH, EP84_BUF) {
            Ok(q) => {
                pr_info!("vino: EP84 async IN queue opened (depth={EP84_QUEUE_DEPTH})\n");
                Some(q)
            }
            Err(e) => {
                pr_info!("vino: EP84 async queue open failed ({e:?}) -- falling back to sync bulk_recv\n");
                None
            }
        };

        // A persistent async interrupt-IN queue on EP83 (mirroring `BulkInQueue` but for an
        // interrupt endpoint) was tried here to see whether FIFO backpressure on the dock's
        // status endpoint was blocking CP engagement. It wasn't: a HW cold plug measured
        // `EP83_events=0` and the dock still never acked (`wsub=0x45`). Dropped rather than
        // carried forward on an ad hoc binding outside the usb series -- `poll_ep83` (sync,
        // called elsewhere in bring-up) is sufficient and already matches DLM's own cadence.
        let mut ep83_q: Option<usb::BulkInQueue> = None;

        // A/B (2026-06-16): route the engagement-critical arm marker + msg0 through an async,
        // pipelined OUT queue (`usb::Interface::bulk_out_queue`) instead of the synchronous
        // `bulk_send`. This mirrors DLM's libusb execution model exactly: each OUT URB is
        // submitted and returns immediately (the HCD auto-retries NAKs until the URB's
        // teardown), so the arm and msg0 are queued back-to-back and reaped afterwards rather
        // than each blocking for its device-ACK round-trip before the next is submitted. The
        // 2026-06-15 measurement showed the *wire* (lengths + submit->complete latency) is
        // already identical, so this is not expected to change what the dock receives -- it is
        // the last structural host difference (sync `usb_bulk_msg` vs async submit/reap) made
        // identical so a cold plug can rule it in or out.
        // A/B RESOLVED 2026-07-14 (paired cold plugs w/ xHCI TRB traces, captures/vino-cold-
        // 20260713-{233429 async, 235745 sync}): sync and async are IDENTICAL on the wire and to
        // the dock. Both: all CP-OUT complete xHCI 'Success' in 57-97 us, max concurrent outstanding
        // OUT URB = 1 (the dock ACKs each OUT fast enough that URBs never overlap even at async
        // depth 4), dock SILENT, 0x45=0. The execution model is NOT the gate; the "async queue-depth
        // choke" theory is refuted. Left at the async default (mirrors DLM's libusb model); flip to
        // false is a proven no-op.
        const CP_ASYNC_OUT: bool = true;
        let mut out_q = if CP_ASYNC_OUT {
            match dev.ctrl_out_queue(4, 1024) {
                Ok(q) => {
                    pr_info!(
                        "vino: EP02 async OUT queue opened (depth=4) -- libusb-style submit/reap\n"
                    );
                    Some(q)
                }
                Err(e) => {
                    pr_info!(
                        "vino: EP02 async OUT queue open failed ({e:?}) -- using sync bulk_send\n"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Pin the EP02 DATA0/DATA1 toggle to DATA0 immediately before the arm. This is the one
        // host lever invisible to every "host exhausted" test: usbmon logs payloads, not the
        // toggle bit, and the crypto/timing work never touches it. DLM (libusb async URBs) and
        // vino (in-kernel blocking bulk_send) can reach the arm with EP02 at *different* parity
        // after the ~9 preceding OUT transfers (7 cap-announce + arm) -- a mismatch makes the
        // dock's SIE ACK the packet at the link layer (byte-identical on the wire) yet discard
        // the payload as a duplicate, i.e. "arms clean, silently drops msg0". clear_halt issues
        // CLEAR_FEATURE(ENDPOINT_HALT), which resets both sides' toggle to DATA0. Every earlier
        // reset (reset_configuration at the top of bring_up, HARD_RESET, VBUS cycle) reset the
        // toggle *before* those preceding transfers, so msg0's parity was never pinned. A/B:
        // flip to `reset_configuration()` to test the heavier reset at the same call site.
        // RESULT 2026-06-16 (cold plug vino-cold-20260616-000552): TESTED NEGATIVE.
        // clear_halt(EP02)
        // fired (wire shows CLEAR_FEATURE on EP2, dmesg "toggle -> DATA0") yet the dock still gave
        // sub=0x45_acks=0. The toggle was NOT the gate. Left default-OFF so vino doesn't carry an
        // EP02 CLEAR_FEATURE that DLM never sends (would pollute future paired diffs); flip to
        // test.
        // Sibling result: EP02 wMaxPacketSize logged = 1024, so a 64-byte msg0/arm always
        // terminates
        // as a natural short packet -- the ZLP-trap hypothesis is moot too.
        const CLEAR_HALT_BEFORE_ARM: bool = false;
        if CLEAR_HALT_BEFORE_ARM {
            match dev.clear_ctrl_halt() {
                Ok(()) => pr_info!("vino: EP02 clear_halt before arm OK (toggle -> DATA0)\n"),
                Err(e) => pr_info!("vino: EP02 clear_halt before arm non-fatal ({e:?})\n"),
            }
        }

        // cert->arm fixed hold (the key `DLM_FIXED_TIMERS` lever). DLM arms a hardcoded
        // CERT_TO_ARM_US (~156 ms on current firmware) after the dock's cert; vino's reactive settle
        // can arm the instant the AKE completes (~57.9 ms -- deep inside the ~152.6 ms downstream-HDCP
        // pause, ~95 ms too early). `hold_until` pads to exactly CERT_TO_ARM_US after the cert with
        // udelay-grade precision. In the normal case wait_cap_complete (reactive on the id=0x0b
        // terminal push, ~cert+155 ms) already ran past the target and this returns at once -- we
        // never arm *before* the window, only pad up to it; the floor catches wait_cap_complete's
        // failure modes (saw_0b miss / early quiet fallback).
        if DLM_FIXED_TIMERS {
            hold_until(session.cert_at, CERT_TO_ARM_US);
        }

        // NOTE: the plaintext capability-announce (the 8 ctr=1..8 cap frames -- hello + the AKE
        // restatement) is NO LONGER sent here. It IS the AKE, now emitted once in `run_ake` (hello
        // ctr=1, AKE_Init..Stream_Manage ctr 2..8), exactly as DLM's wire does it -- see the
        // unification note on `run_ake`. This function now does only the encrypted arm + msg0 + the
        // post-msg0 burst, which follow the AKE with no second restatement (the old duplicate send
        // is what made the dock NAK our arm).

        // Submit the arm marker. Async path: queue it and DO NOT flush -- leave it in flight so
        // msg0 follows after the fixed arm->msg0 hold below. Async path: submit AND flush so the
        // arm is fully on the wire (reaped) before we time and send msg0 -- DLM sends the two
        // sequentially with a fixed gap (`ARM_TO_MSG0`), not pipelined back-to-back. Sync path: the
        // original blocking send.
        let arm_res = match out_q.as_mut() {
            Some(q) => q
                .send(dev.io(), &STREAM_OPEN, timeout())
                .and_then(|()| q.flush(dev.io(), timeout())),
            None => dev
                .ctrl_send(&STREAM_OPEN, timeout(), GFP_KERNEL)
                .map(|_| ()),
        };
        // A NAK'd arm is informative (the dock rejected the stream-open) but must NOT abort the
        // whole CP setup -- we still send msg0 so the run produces a full ack/reject verdict rather
        // than bailing early. (Before the AKE/restatement unification the dock NAK'd the arm because
        // it had just seen the AKE sent twice; that duplicate is now gone.)
        if let Err(e) = arm_res {
            pr_warn!(
                "vino: CP stream-open arm marker NAK'd ({e:?}) -- continuing to msg0 anyway\n"
            );
        }
        // Report the realised `cp_start->arm` AND `cert->arm` so the cold-plug dmesg carries vino's
        // pre-arm fingerprint for an A/B against DLM (cert->arm fixed ~156 ms under DLM_FIXED_TIMERS,
        // matching the 2026-07-13 cold refs). Microsecond precision.
        let arm_at = session.cp_start.elapsed();
        let cert_to_arm = session.cert_at.elapsed().as_micros_ceil();
        pr_info!(
            "vino: CP stream-open arm marker sent (cp_start->arm = {} us, cert->arm = {} us, target {} us)\n",
            arm_at.as_micros_ceil(), cert_to_arm, CERT_TO_ARM_US
        );

        // arm->msg0 hold. The 2026-06-25 timing survey found this is the ONE step where vino is
        // consistently faster than DLM: vino fires msg0 ~0.07 ms after the arm, DLM ~0.17 ms -- the
        // only timing inversion in the corpus, and never tested as a variable. Earlier this gap was
        // left unpadded on the "engine is event-driven, sub-ms lead is immaterial" reasoning; the
        // survey shows DLM's gap is a *fixed* ~0.17 ms (0.152/0.188 ms, 0.036 ms spread = a hard
        // sleep, not a reaction), so under `DLM_FIXED_TIMERS` we hold [`ARM_TO_MSG0`] to match it
        // exactly. If the dock keys engagement on msg0 not arriving before the arm has settled, this
        // is the lever; if it engages, "msg0 too soon" was the gate. Cheap to A/B on one cold plug.
        if DLM_FIXED_TIMERS {
            fsleep(ARM_TO_MSG0);
        }
        // LIVE CP msg0: protocol-fixed header `id=0x14 sub=0x00 ctr=0x09`, 14 zero bytes, then a
        // fresh host-random 10-byte token (the dock does not validate or echo it), sealed under
        // THIS session's ks/riv with a live Dl3Cmac. This is the decisive engagement probe: a
        // `wsub=0x45` reply would mean the cipher engaged on a live session.
        // Running CP counters: the inner message counter and the AES-CTR wire-seq (block index),
        // advanced on every CP frame we send instead of hardcoding each value -- so the sequence
        // can never desync from a constant (hardcoded values caused past off-by-one regressions).
        // msg0 continues the inner counter from where `run_ake` left off (`session.next_ctr`): the
        // current-firmware cold plug sends session-init-ACK ctr=1 + AKE ctr 2..8, so msg0 is ctr=9 /
        // wire_seq=0 -- exactly the reauth flow's numbering (both carry the leading id=0x14/0x76).
        let mut cp_ctr: u16 = session.next_ctr;
        let mut wseq: u32 = 0;

        // msg0 CONTENT is STRUCTURED, not random. rr reverse-execution of DLM 3.4.26's real msg0
        // ENCRYPT (2026-07-16) read the plaintext straight out of DLM's XOR input register:
        // `14000000 09000000 00.. [10-byte token]` -- id=0x14, inner ctr=9, zeros, then a random
        // token at [22..32]. (The earlier "content is 32 random bytes" reading was an artifact of
        // decrypting under the WRONG CTR nonce -- byte0 was toggled; corrected above. With the right
        // f621..-style CTR nonce the plaintext decodes to this structure.) So build it exactly as
        // DLM does: id + inner counter + a host-random token the dock does not validate or echo.
        let mut content = [0u8; 32];
        content[0..2].copy_from_slice(&0x0014u16.to_le_bytes()); // id=0x14
        content[4..6].copy_from_slice(&cp_ctr.to_le_bytes()); // inner counter (sub=0x00, pad=0)
        match MSG0_TOKEN_OVERRIDE {
            Some(tok) => content[22..32].copy_from_slice(&tok), // replay a real DLM msg0 token
            None => rng::fill(&mut content[22..32]),            // host-random token (default)
        }
        let body_len = content.len() + 16; // AES-CTR ciphertext + 16-byte Dl3Cmac
        let size = ((16 + body_len) - 4) as u16;
        let aux = cp::aux_for_id(0x14, body_len);
        let mut hdr = [0u8; 16];
        hdr[2..4].copy_from_slice(&size.to_le_bytes());
        hdr[4..8].copy_from_slice(&4u32.to_le_bytes()); // type=4
        hdr[8..10].copy_from_slice(&0x24u16.to_le_bytes()); // sub=0x24 (interactive CP)
        hdr[10..12].copy_from_slice(&aux.to_le_bytes());
        hdr[12..16].copy_from_slice(&wseq.to_le_bytes()); // running AES-CTR block index (0 for msg0)
        let frame = cp::seal_livemac(&session.ks, &session.riv, &hdr, &content)?;

        let mut ok = false;
        // The per-path send Result is not inspected further (engagement is judged from the dock's
        // EP84 reply, not msg0's OUT status); bind it to satisfy `must_use` without a warning.
        let _msg0_res = match out_q.as_mut() {
            Some(q) => {
                // Arm already reaped + the fixed hold elapsed; submit msg0 now.
                let sent = q.send(dev.io(), &frame, timeout());
                pr_info!("vino: live CP msg0 submitted async (after arm settled)\n");
                if sent.is_ok() {
                    ok = true;
                    // Flush FIRST so msg0 is actually on the wire before we read for replies -- else
                    // the drain below reads the dock's traffic while msg0 is still queued (the
                    // 2026-07-12 log showed ~88 ms between "submitted" and "flushed OK", with dock
                    // frames arriving in between, i.e. before msg0 had even left the host).
                    match q.flush(dev.io(), timeout()) {
                        Ok(_) => pr_info!("vino: async msg0 flushed OK (on the wire)\n"),
                        Err(e) => {
                            pr_info!("vino: async msg0 flush incomplete ({e:?}) -- dock NAK'd\n")
                        }
                    }
                    // Now drain the dock's reply to msg0.
                    for _ in 0..8 {
                        let d = Self::drain_ep84(
                            dev,
                            ep84_q.as_mut(),
                            &mut resp,
                            session,
                            edid_out,
                            Delta::from_millis(10),
                        );
                        drained += d.reads;
                        acks += d.acks;
                        rejects += d.rejects;
                        ep83_events += Self::drain_ep83_queue(dev, ep83_q.as_mut());
                    }
                } else {
                    pr_info!("vino: live CP msg0 async submit failed ({sent:?})\n");
                }
                sent
            }
            None => {
                // Sync path: single-packet msg0 => a NAK transfers nothing, so cancel+retry is safe.
                // Between attempts drain EP84 so the dock can push/drain its IN queue. Bounded.
                const TRIES: usize = 40;
                let mut sync_res = Err(Error::from_errno(bindings::ETIMEDOUT.try_into().unwrap()));
                for t in 0..TRIES {
                    match dev.ctrl_send(&frame, Delta::from_millis(5), GFP_KERNEL) {
                        Ok(_) => {
                            ok = true;
                            pr_info!("vino: live CP msg0 ACCEPTED after {t} interleaved tries\n");
                            sync_res = Ok(());
                            break;
                        }
                        // OUT NAK'd (nothing transferred) -- let the dock push on EP84, then retry.
                        Err(_) => {
                            let d = Self::drain_ep84(
                                dev,
                                ep84_q.as_mut(),
                                &mut resp,
                                session,
                                edid_out,
                                Delta::from_millis(10),
                            );
                            drained += d.reads;
                            acks += d.acks;
                            rejects += d.rejects;
                            ep83_events += Self::drain_ep83_queue(dev, ep83_q.as_mut());
                        }
                    }
                }
                sync_res
            }
        };
        if ok {
            sent += 1;
            pr_info!("vino: live CP msg0 sent (id=0x14 ctr={cp_ctr}, random token, live seal)\n");
        } else {
            pr_info!("vino: live CP msg0 still NAK'd (no transfer accepted)\n");
        }
        cp_ctr += 1; // past msg0
        wseq += 2; // msg0 content is 32 B = 2 AES blocks

        // Post-msg0 setup burst -- DLM 3.4.26 follows msg0 with four more 32-byte messages of the
        // same shape before the control dance (verified byte-exact vs `live-3426-20260707/reauth`):
        // id=0x14 sub=0x30, id=0x15 sub=0x0b, id=0x16 sub=0x2a (x2). The counters come from the
        // running `cp_ctr`/`wseq` so they stay in lockstep with msg0: with msg0=ctr9 that is
        // ctr 10/11/12/13, wseq 2/4/6/8 -- identical to the reauth flow now that the cold path also
        // sends the leading id=0x14/0x76 session-init ACK. The per-id trailer prefix bytes match the
        // same capture.
        for (id, sub, fixed_prefix) in [
            (0x0014u16, 0x0030u16, &[][..]),
            (0x0015, 0x000b, &[0x01][..]),
            (0x0016, 0x002a, &[0x00, 0x01][..]),
            (0x0016, 0x002a, &[0x01, 0x01][..]),
        ] {
            let mut c = [0u8; 32];
            c[0..2].copy_from_slice(&id.to_le_bytes());
            c[2..4].copy_from_slice(&sub.to_le_bytes());
            c[4..6].copy_from_slice(&cp_ctr.to_le_bytes());
            rng::fill(&mut c[22..32]);
            c[22..22 + fixed_prefix.len()].copy_from_slice(fixed_prefix);
            match cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c) {
                Ok(f) => {
                    let burst_ok = match out_q.as_mut() {
                        Some(q) => q.send(dev.io(), &f, timeout()).is_ok(),
                        None => dev.ctrl_send(&f, timeout(), GFP_KERNEL).is_ok(),
                    };
                    if burst_ok {
                        sent += 1;
                    }
                    pr_info!("vino: live CP burst id={id:#06x} sub={sub:#06x} ctr={cp_ctr} wseq={wseq} {}\n",
                        if burst_ok { "sent" } else { "NAK'd" });
                }
                Err(e) => pr_info!("vino: live CP burst id={id:#06x} seal failed ({e:?})\n"),
            }
            let d = Self::drain_ep84(
                dev,
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
            wseq += 2; // each burst message is 32 B = 2 AES blocks
        }

        // Per-head encrypted AKE re-statement + stream-open (decoded dump ictr 14..22 for head 0,
        // ictr 23..31 for head 1 -- `CP_SETUP_HEADS` re-runs this same 9-message block once per
        // head). **2026-07-17: framing corrected against a FULL decryption of the real burst**
        // (`docs/CP-PERHEAD-RESTATEMENT.md`); each message mirrors the plaintext AKE body byte
        // layout with the `0x30`@22 marker swapped for `off23=head+1`, HDCP msg-id at off27, and a
        // fresh host-random payload at off28 (the dock does not validate it -- DLM itself sends
        // per-head-varying values). `id=0x32` carries a freshly generated per-head VIDEO KEY at
        // off28, stashed for the EP08 encoder; its cryptographic role in video is not yet RE'd.
        // Light drain to collect any pending acks before head 0's block. DLM does NOT hold here --
        // the measured cold capture flows from the init group straight into head 0's AKE with only a
        // few ms gap; the single per-head hold is the H' wait INSIDE each block (after No_Stored_km,
        // see HDCP_HPRIME_WAIT_US), not at this boundary.
        {
            let d = Self::drain_ep84(
                dev,
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
                for h in 0..Self::CP_SETUP_HEADS {
                    if stream_open_ctr[h] == Some(c) {
                        heads_present[h] = true;
                    }
                }
            }
        }
        for head in 0..Self::CP_SETUP_HEADS {
            // Fresh per-head downstream-repeater AKE. The crypto is **standard HDCP 2.2** (rr-
            // confirmed 2026-07-17: the restatement Ekpub is a real RSA-1024 output, V a
            // RepeaterAuth_Send_Ack) computed with the SAME proven primitives `run_ake` uses,
            // reusing the retained dock RSA pubkey / rrx / receiver-ID list. The chain is self-
            // consistent per head, so the dock re-derives it: Ekpub->km_h, edkey->ks_h,
            // kd_h=derive_kd(km_h,rtx_h,rrx), V=HMAC(kd_h,list). The CP-encrypted framing + a few
            // non-standard trailing bytes past each field are DisplayLink-proprietary and left
            // best-effort (zero) here -- RE pending, see `docs/CP-PERHEAD-RESTATEMENT.md`. On any
            // primitive error the affected entry falls back to random. NOT yet HW-verified.
            let mut rtx_h = [0u8; 8];
            rng::fill(&mut rtx_h);
            let mut km_h = [0u8; 16];
            rng::fill(&mut km_h);
            let mut rn_h = [0u8; 8];
            rng::fill(&mut rn_h);
            let mut ske_ks_h = [0u8; 16];
            rng::fill(&mut ske_ks_h);
            let mut riv_h = [0u8; 8];
            rng::fill(&mut riv_h);
            let ekpub_h = hdcp::oaep_encrypt_km(&session.modulus, &session.exponent, &km_h);
            // `rrx_h` is the rrx used to derive THIS head's kd/edkey/V. It starts as the main-AKE rrx
            // but is REPLACED by the dock's fresh per-head rrx (its `AKE_SEND_RRX` push, captured into
            // `fresh_rrx` during the loop's drains) before SKE_Send_Eks/RepeaterAuth_Send_Ack are
            // built -- see the recompute block in the loop. Using the stale main rrx here was the
            // 2026-07-20 EDID root cause: the dock derives kd from its fresh per-head rrx, so vino's V
            // did not match its V' and the repeater auth silently failed (dock withheld EDID).
            let mut rrx_h = session.rrx;
            let mut kd_h = hdcp::derive_kd(&km_h, &rtx_h, &rrx_h);
            let mut edkey_h = hdcp::compute_eks(&km_h, &rtx_h, &rrx_h, &rn_h, &ske_ks_h);
            let recompute_v = |kd_h: &Result<[u8; 32]>| {
                kd_h.as_ref().ok().map(|kd| {
                    let vf = hdcp::compute_v_full(kd, &session.rxid_list);
                    let mut v = [0u8; 16];
                    v.copy_from_slice(&vf[16..]);
                    v
                })
            };
            let mut v_h = recompute_v(&kd_h);
            let mut fresh_rrx: Option<[u8; 8]> = None;
            let mut rrx_applied = false;
            // Entry 4 (SKE_Send_Eks) establishes THIS head's VIDEO channel key. The video AES-CTR
            // nonce is NOT the main CP transform: same-session rr correlation proves delivered
            // riv_h with byte7 ^ (0x08 | head). Stash the SKE key and that content nonce so the
            // scanout ARM burst seals with what the dock reconstructs from this SKE.
            // Layout: key(16) || nonce(8) || pad(8).
            video_keys[head] = [0u8; 32];
            // The dock applies the same device whitening to the unwrapped per-head SKE key as it
            // does on the main CP channel. A raw-ske_ks hardware A/B caused an immediate dock reset
            // on the first ARM transfer; the whitened key is accepted without a reset.
            video_keys[head][..16].copy_from_slice(&cp::cp_session_key(&ske_ks_h));
            let vnonce = cp::video_content_nonce(&riv_h, head as u8);
            video_keys[head][16..24].copy_from_slice(&vnonce);
            for (i, (id, sub, content_len)) in cp::CP_SETUP_PER_HEAD.iter().copied().enumerate() {
                // EDID repeater-auth fix (2026-07-20): the dock sends a FRESH per-head rrx (its
                // `AKE_SEND_RRX` push, captured into `fresh_rrx` by the drains below). Re-derive this
                // head's kd/edkey/V from it before the messages that consume them -- SKE_Send_Eks
                // (i==4, needs edkey) and RepeaterAuth_Send_Ack (i==5, needs V). The rrx has arrived
                // by i==3 (it comes with the cert/H' burst the dock sends after No_Stored_km at i==2,
                // whose drain also holds ~160ms for H'). Applied once. Using the stale main-AKE rrx
                // made vino's V disagree with the dock's V' -> silent repeater-auth failure -> dock
                // withheld EDID (bare id=0x14 instead of rich id=0x44). See `cp::perhead_rrx`.
                if i >= 3 && !rrx_applied {
                    if let Some(rrx) = fresh_rrx {
                        rrx_h = rrx;
                        kd_h = hdcp::derive_kd(&km_h, &rtx_h, &rrx_h);
                        edkey_h = hdcp::compute_eks(&km_h, &rtx_h, &rrx_h, &rn_h, &ske_ks_h);
                        v_h = recompute_v(&kd_h);
                        pr_info!(
                            "vino: per-head[{head}] applied dock rrx -> re-derived kd/edkey/V\n"
                        );
                    } else {
                        pr_info!(
                            "vino: per-head[{head}] WARNING: no fresh rrx before SKE -- using main rrx (repeater auth will fail)\n"
                        );
                    }
                    rrx_applied = true;
                }
                // id=0x26 (Stream_Manage restatement) is fully decoded -- deterministic content,
                // not the generic path below. See `cp::stream_manage_restatement`'s doc comment.
                if id == 0x0026 {
                    match cp::stream_manage_restatement(cp_ctr, head as u8) {
                        Ok(c) => {
                            match cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c) {
                                Ok(f) => {
                                    let ok = match out_q.as_mut() {
                                        Some(q) => q.send(dev.io(), &f, timeout()).is_ok(),
                                        None => dev.ctrl_send(&f, timeout(), GFP_KERNEL).is_ok(),
                                    };
                                    if ok {
                                        sent += 1;
                                    }
                                    pr_info!(
                                    "vino: live CP per-head[{head}] id={id:#06x} sub={sub:#06x} ctr={cp_ctr} wseq={wseq} {}\n",
                                    if ok { "sent" } else { "NAK'd" }
                                );
                                }
                                Err(e) => pr_info!(
                                "vino: live CP per-head[{head}] id={id:#06x} seal failed ({e:?})\n"
                            ),
                            }
                        }
                        Err(e) => {
                            pr_info!("vino: live CP per-head id={id:#06x} build failed ({e:?})\n")
                        }
                    }
                    let d = Self::drain_ep84(
                        dev,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        edid_out,
                        Delta::from_millis(10),
                    );
                    drained += d.reads;
                    acks += d.acks;
                    rejects += d.rejects;
                    fresh_rrx = fresh_rrx.or(d.perhead_rrx);
                    cp_ctr += 1;
                    wseq += ((content_len + 15) / 16) as u32;
                    continue;
                }
                let mut c = match KVec::from_elem(0u8, content_len, GFP_KERNEL) {
                    Ok(v) => v,
                    Err(e) => {
                        pr_info!("vino: live CP per-head id={id:#06x} alloc failed ({e:?})\n");
                        cp_ctr += 1;
                        wseq += ((content_len + 15) / 16) as u32;
                        continue;
                    }
                };
                // Shared header (id / sub=0x10 / inner counter), identical to the plaintext AKE
                // body layout (`ake::body`). The buffer is already zeroed by `from_elem`.
                c[0..2].copy_from_slice(&id.to_le_bytes());
                c[2..4].copy_from_slice(&sub.to_le_bytes());
                c[4..6].copy_from_slice(&cp_ctr.to_le_bytes());
                // Per-head restatement framing, ground-truthed 2026-07-17 by FULL AES-CTR
                // decryption of trace1's encrypted burst (`docs/CP-PERHEAD-RESTATEMENT.md`). The
                // frame is the plaintext AKE body layout (`ake::body`: HDCP msg-id at off27, payload
                // at off28) with the `0x30` marker at off22 replaced by `off22=0, off23=head+1`.
                // The payloads are **standard HDCP 2.2 derived crypto** (rr-confirmed), placed at
                // their spec offsets from the fresh per-head chain computed above; the proprietary
                // trailing bytes past each field are left zero (RE pending).
                match i {
                    // AKE restatements: head marker @23, HDCP msg-id tag @27, HDCP field @28..
                    0 | 1 | 2 | 3 | 4 | 5 => {
                        c[23] = head as u8 + 1;
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
                                c[28..36].copy_from_slice(&rtx_h); // AKE_Init: rtx
                                                                   // DLM fills the proprietary bytes after the standard-HDCP field
                                                                   // with a fresh random draw.  They are not padding: leaving them
                                                                   // zero is a wire-visible difference in every captured per-head
                                                                   // restatement, including the one that establishes the video key.
                                rng::fill(&mut c[36..48]);
                            }
                            1 => {
                                c[28..33].copy_from_slice(&[0x00, 0x06, 0x02, 0x00, 0x02]);
                                rng::fill(&mut c[33..48]);
                            }
                            2 => match &ekpub_h {
                                Ok(ek) if content_len >= 160 => {
                                    c[28..156].copy_from_slice(ek);
                                    rng::fill(&mut c[156..160]);
                                }
                                _ => {
                                    pr_err!("vino: per-head[{head}] Ekpub unavailable -- random\n");
                                    rng::fill(&mut c[28..]);
                                }
                            },
                            3 => {
                                c[28..36].copy_from_slice(&rn_h); // LC_Init: rn
                                rng::fill(&mut c[36..48]);
                            }
                            4 => match &edkey_h {
                                Ok(ed) if content_len >= 64 => {
                                    c[28..44].copy_from_slice(ed);
                                    c[44..52].copy_from_slice(&riv_h);
                                    // Real id=0x32 messages carry twelve fresh random bytes after
                                    // Edkey||riv (e.g. ictr 18/27 in the fully decrypted trace).
                                    // This is the last clear setup input before EP08's sealed ARM,
                                    // so match it exactly instead of assuming zero padding.
                                    rng::fill(&mut c[52..64]);
                                }
                                _ => {
                                    pr_err!("vino: per-head[{head}] edkey unavailable -- random\n");
                                    rng::fill(&mut c[28..]);
                                }
                            },
                            _ => match v_h {
                                Some(v) => {
                                    c[28..44].copy_from_slice(&v); // RepeaterAuth_Send_Ack: V
                                    rng::fill(&mut c[44..48]);
                                }
                                None => {
                                    pr_err!("vino: per-head[{head}] V unavailable -- random\n");
                                    rng::fill(&mut c[28..]);
                                }
                            },
                        }
                    }
                    // Stream-open control: header + zero[8..22] + 10 host-random bytes[22..32];
                    // no head marker, no tag (confirmed genuinely fully random across both heads).
                    // Record this head's request ctr: the dock's DISPLAY-CAP reply (id=0x78 sub=0x30)
                    // echoes it iff a monitor is present on this head (dual-monitor detection).
                    7 => {
                        if head < stream_open_ctr.len() {
                            stream_open_ctr[head] = Some(cp_ctr);
                        }
                        rng::fill(&mut c[22..]);
                    }
                    // strm2: head index @22, then the fixed `06 [head*4] 04` triple @24..27, then
                    // a fresh 5-byte host-random tail.
                    8 => {
                        c[22] = head as u8;
                        c[24] = 0x06;
                        c[25] = (head as u8) * 4;
                        c[26] = 0x04;
                        rng::fill(&mut c[27..]);
                    }
                    _ => {}
                }
                let send_at = Instant::<Monotonic>::now();
                match cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c) {
                    Ok(f) => {
                        let ok = match out_q.as_mut() {
                            Some(q) => q.send(dev.io(), &f, timeout()).is_ok(),
                            None => dev.ctrl_send(&f, timeout(), GFP_KERNEL).is_ok(),
                        };
                        if ok {
                            sent += 1;
                        }
                        pr_info!(
                            "vino: live CP per-head[{head}] id={id:#06x} sub={sub:#06x} ctr={cp_ctr} wseq={wseq} {}\n",
                            if ok { "sent" } else { "NAK'd" }
                        );
                    }
                    Err(e) => pr_info!(
                        "vino: live CP per-head[{head}] id={id:#06x} seal failed ({e:?})\n"
                    ),
                }
                let d = Self::drain_ep84(
                    dev,
                    ep84_q.as_mut(),
                    &mut resp,
                    session,
                    edid_out,
                    Delta::from_millis(10),
                );
                drained += d.reads;
                acks += d.acks;
                rejects += d.rejects;
                fresh_rrx = fresh_rrx.or(d.perhead_rrx);
                // The dock's downstream-HDCP H' compute wait. `i == 2` is AKE_No_Stored_km
                // (msg-id 0x04); the dock now takes ~160 ms to compute AKE_Send_H_prime and will not
                // accept LC_Init (`i == 3`) before that. Hold send-to-send from this message so
                // LC_Init lands ~165 ms later, matching DLM's measured 160.1/160.2 ms cadence -- the
                // gate that flips the dock from bare id=0x14 to rich id=0x44 EDID replies. See
                // HDCP_HPRIME_WAIT_US. Precise `hold_until` (not fsleep) -- fsleep slack made the
                // earlier blind attempt intermittent.
                if i == 2 {
                    hold_until(send_at, HDCP_HPRIME_WAIT_US);
                    // The cert / fresh per-head rrx / H' burst arrives DURING this hold. Drain it now
                    // so `fresh_rrx` is populated before the i==3 recompute that re-derives kd/edkey/V
                    // (the EDID repeater-auth fix) -- the plain per-message drain above ran BEFORE the
                    // hold and would miss it.
                    let dh = Self::drain_ep84(
                        dev,
                        ep84_q.as_mut(),
                        &mut resp,
                        session,
                        edid_out,
                        Delta::from_millis(10),
                    );
                    drained += dh.reads;
                    acks += dh.acks;
                    rejects += dh.rejects;
                    fresh_rrx = fresh_rrx.or(dh.perhead_rrx);
                }
                // Attribute a DISPLAY-CAP reply to the head whose stream-open ctr it echoes -> that
                // head has a monitor. Matching by ctr (not loop position) is robust to a reply that
                // arrives a message or two late, even in the other head's block.
                if let Some(c) = d.display_cap_ctr {
                    for h in 0..Self::CP_SETUP_HEADS {
                        if stream_open_ctr[h] == Some(c) {
                            heads_present[h] = true;
                        }
                    }
                }
                cp_ctr += 1;
                wseq += ((content_len + 15) / 16) as u32;
            }
            // Light drain after each head's block. DLM does NOT hold between heads (the measured
            // head0->head1 transition is ~7 ms) -- the only per-head hold is the H' wait inside the
            // block (after No_Stored_km, HDCP_HPRIME_WAIT_US). Just collect pending acks / the
            // DISPLAY-CAP reply here.
            {
                let d = Self::drain_ep84(
                    dev,
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
                    for h in 0..Self::CP_SETUP_HEADS {
                        if stream_open_ctr[h] == Some(c) {
                            heads_present[h] = true;
                        }
                    }
                }
            }
        }

        // Stream finalize (decoded dump ictr 32..36): a fixed 5-message tail sent once, after
        // both heads' per-head blocks, before the dock settles into its steady-state heartbeat
        // (`id=0x16 sub=0x75`, unrelated to setup).
        for (id, sub, off22) in cp::CP_SETUP_FINALIZE {
            // 32-byte content (the old 16 was half the real length): header + zero[6..22], a
            // per-message index flag at off22, and -- for the sub=0x4c messages -- a constant 0x01
            // at off23. The remaining tail is host-random (the dock does not validate it). See
            // `docs/CP-PERHEAD-RESTATEMENT.md`.
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
            match cp::seal_interactive(&session.ks, &session.riv, id, wseq, &c) {
                Ok(f) => {
                    let ok = match out_q.as_mut() {
                        Some(q) => q.send(dev.io(), &f, timeout()).is_ok(),
                        None => dev.ctrl_send(&f, timeout(), GFP_KERNEL).is_ok(),
                    };
                    if ok {
                        sent += 1;
                    }
                    pr_info!(
                        "vino: live CP finalize id={id:#06x} sub={sub:#06x} ctr={cp_ctr} wseq={wseq} {}\n",
                        if ok { "sent" } else { "NAK'd" }
                    );
                }
                Err(e) => pr_info!("vino: live CP finalize id={id:#06x} seal failed ({e:?})\n"),
            }
            let d = Self::drain_ep84(
                dev,
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

        // DLM sends the `0x24 wValue=0` render/commit vendor request right after msg0.
        match dev.control_send(
            0x24,
            0x40, /* VENDOR_OUT */
            0,
            0,
            &[],
            timeout(),
            GFP_KERNEL,
        ) {
            Ok(()) => pr_info!("vino: post-msg0 0x24(wValue=0) OK\n"),
            Err(e) => pr_info!("vino: post-msg0 0x24(wValue=0) non-fatal ({e:?})\n"),
        }
        // DLM then re-reads the 0x22 vendor state (0xc1, wValue=1, wIndex=0, 28 B) -- its SECOND
        // 0x22 of the session, immediately after the post-msg0 0x24. vino issued the first 0x22
        // pre-arm but stopped here, leaving "DLM-ONLY 0x22" in the paired diff. Issue it
        // unconditionally so the wire matches DLM regardless of whether the dock acks; it is a
        // harmless vendor IN read. (0xc1 = IN|vendor|INTERFACE recipient, matching the first 0x22.)
        let mut state2 = [0u8; 28];
        match dev.control_recv(0x22, 0xc1, 1, 0, &mut state2, timeout(), GFP_KERNEL) {
            Ok(()) => pr_info!("vino: post-msg0 0x22(wValue=1) OK = {:02x?}\n", state2),
            Err(e) => pr_info!("vino: post-msg0 0x22(wValue=1) non-fatal ({e:?})\n"),
        }

        // Read the dock's reply: a VERIFIED `wsub=0x45` ack means the cipher engaged on our frame.
        let ls = Self::lockstep_reply(dev, ep84_q.as_mut(), &mut resp, session, 0x08, edid_out);
        drained += ls.reads;
        acks += ls.acks;
        rejects += ls.rejects;

        const MAX_ROUNDS: usize = 16;
        for _ in 0..MAX_ROUNDS {
            let d = Self::drain_ep84(
                dev,
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

        // ---- Post-engagement live setup (CP-HANDSHAKE.md sec 4f/sec 4e) ------------------------
        // Only meaningful once the dock has acked msg0: ask the dock for the downstream EDID,
        // then build the mode-set from its preferred timing and send that -- the live path that
        // replaces the static 1080p modeset and the opportunistic-only EDID capture. On a cold
        // dock `acks` stays 0 (the wall), so this does not run on current hardware; it completes
        // the standalone live-generation flow for when the engagement gate is solved.
        // The next free AES-CTR block index past this setup, handed to the DRM device so runtime
        // KMS sends (mode-set/cursor) continue the same keystream. Defaults to msg0's end (2) when
        // the live block below doesn't run (no acks) -- irrelevant then, since we only publish the
        // session when `acks > 0`.
        // The running `cp_ctr`/`wseq` continue here (after the burst: cp_ctr=14, wseq=10). get-EDID
        // and mode-set take the next values in sequence, so they can never collide with msg0/burst.
        // Only runs once the dock acks (the wall keeps acks=0 today); keeping it on the same running
        // counters means it is correct the instant engagement lands.
        if acks > 0 {
            // (1) Live get-EDID request -> the dock replies id=0x194; `drain_ep84` (called inside
            // `send_live_cp`) decodes it and fills `edid_out` via `parse_edid_from_reply`.
            //
            // GROUND-TRUTHED 2026-07-16 (mined a full DLM session well past where earlier work
            // stopped -- `captures/rr-out-sequence-20260716/full-session-trace1/`): DLM does NOT
            // get the real EDID on its first ask either. Its early get-EDID requests (inner ctr
            // 44, 47) get back a dock-internal placeholder (`id=0x114`, a generic "NOVATEK"
            // descriptor -- not any real monitor), and only a LATER retry (ctr 120+, roughly 76
            // inner messages / 156 USB-transfer events later, well into steady-state heartbeat
            // traffic) gets the real `id=0x194` reply with the monitor's actual EDID. **HW-TESTED
            // 2026-07-16: 8 back-to-back retries (no interleaved traffic) reproduced only DLM's
            // early failing window -- every attempt got a bare `id=0x14 sub=0x21` ack, never
            // `id=0x194`.** So this now interleaves a heartbeat (`cp::heartbeat`, `id=0x16
            // sub=0x75` -- harmless, already-proven-safe CP traffic) between get-EDID attempts and
            // spans many more rounds, giving the dock the same kind of settled, heartbeat-
            // interleaved session DLM's successful retry happened inside, rather than a burst of
            // identical requests in the same narrow window as DLM's own unsuccessful attempts.
            // Bounded so a dock that genuinely never delivers one doesn't stall bring-up long.
            // TEMPORARILY cut from 40 to 10 (2026-07-16, post-lockup investigation): a 40-round
            // run once froze the whole machine with no oops/backtrace captured (pstore only
            // preserved one truncated fragment, mid-round-38, before everything went silent --
            // see `project_bringup_lockup_20260716` memory). The checkpoints added around this
            // loop and the mode-set send below localise exactly where a repeat freeze happens.
            //
            // RESTORED to 50 (2026-07-17): at ROUNDS=10, a live test never got so much as DLM's
            // own early placeholder reply (`id=0x114`) -- every attempt only drew the generic
            // `id=0x14 sub=0x21` ack, confirmed from the raw decrypted bytes (byte1==0x00, not a
            // 0x114-vs-0x14 decode confusion). DLM's own captured session needed a ~76-inner-
            // message gap between its first (placeholder) and second (real) `id=0x194` attempt,
            // well into steady-state heartbeat traffic -- 10 rounds (20 messages incl. the
            // interleaved heartbeats) never gets remotely close to that window. 50 rounds (100
            // messages, ~5s at 100ms/round) comfortably spans it with margin. The original
            // whole-machine-lockup theory pinned this on the id=0x48 mode-set call reached only
            // once EDID succeeds -- that call has SINCE fired successfully multiple times this
            // session via a different call site (`VinoCrtc::atomic_enable`'s runtime mode-set
            // resend, same `send_live_cp(..., 0x48, ...)` primitive) with no incident, so that
            // theory is now doubted rather than confirmed. Restoring with the checkpoints from
            // the original investigation still in place so a repeat freeze is still localisable.
            // Device-capability query, one-shot: **found 2026-07-17** decrypting DLM's raw OUT/IN
            // sequence -- DLM sends `id=0x14 sub=0x0000` exactly ONCE, right after a heartbeat and
            // right before it starts `sub=0x0020`-polling for EDID readiness (ictr 38/39/40 in the
            // trace). Never sent by vino before now. See `cp::device_query_req`'s doc comment.
            if let Ok(hb) = cp::heartbeat(cp_ctr) {
                if let Ok((_, e)) = Self::send_live_cp(
                    dev,
                    session,
                    ep84_q.as_mut(),
                    &mut resp,
                    edid_out,
                    0x16,
                    wseq,
                    &hb,
                ) {
                    drained += e.reads;
                    acks += e.acks;
                    rejects += e.rejects;
                    wseq = wseq.wrapping_add(((hb.len() + 15) / 16) as u32);
                    cp_ctr += 1;
                }
            }
            if let Ok(devq) = cp::device_query_req(cp_ctr, 0x0000) {
                match Self::send_live_cp(
                    dev,
                    session,
                    ep84_q.as_mut(),
                    &mut resp,
                    edid_out,
                    0x14,
                    wseq,
                    &devq,
                ) {
                    Ok((ok, e)) => {
                        drained += e.reads;
                        acks += e.acks;
                        rejects += e.rejects;
                        wseq = wseq.wrapping_add(((devq.len() + 15) / 16) as u32);
                        cp_ctr += 1;
                        pr_info!(
                            "vino: live device-capability query {} (id=0x14 sub=0x0)\n",
                            if ok { "sent" } else { "NAK'd" }
                        );
                    }
                    Err(e) => pr_info!("vino: device-capability query failed ({e:?})\n"),
                }
            }

            // get-EDID cadence, restructured 2026-07-17 to replicate DLM's exact observed
            // step-by-step sequence rather than blind probe+fetch pairs (which never got so much
            // as the dock-internal placeholder in 50 rounds -- see
            // `project_edid_fetch_still_unsolved_20260717` memory). DLM's raw trace shows every
            // real EDID success preceded by probe(0x20) x2-3, then fetch(0x21).
            //
            // **The `id=0x16 sub=0x4b` "kick" step originally tried here stays REMOVED.** HW-tested
            // 2026-07-17: every kick sent (10/10) got `dock REJECTING our CP` -- a clean,
            // reproducible negative signal unlike every probe/fetch in the same run. Left out.
            //
            // **RESTRUCTURED 2026-07-17 (3rd pass) around a byte-exact readiness signal + two
            // never-sent message classes**, found by decoding `full-session-trace1`'s full inner
            // ctr 39..122 window (not just the previously-flagged "interesting" lines) and
            // independently cross-checked against a SECOND, unrelated capture
            // (`dlm-cold-3426-20260714-140216-allbus`, a warm/EDID-cached session): after a
            // couple of early probe+fetch rounds keep returning only the dock-internal
            // placeholder, DLM sends TWO `id=0x16 sub=0x0023` "engage" messages
            // (`cp::edid_engage_req`, cross-validated present in both captures, never sent by
            // vino before now) and then spins a long `id=0x14 sub=0x000c` device-status poll
            // (`cp::device_query_req(.., 0x000c)`, documented since the previous pass but never
            // actually called until now -- 66 of them in trace1) while the dock's downstream
            // DDC/EDID read genuinely completes in real time. The exact completion signal
            // (`cp::edid_poll_ready`, inner byte offset 26 of the `sub=0x0020` probe reply,
            // 0x00->0x80) was not known during the previous pass's cruder non-zero-byte-count
            // heuristic (`edid_poll_progress`) -- this pass gates every retry on it instead of a
            // blind round count, so a dock that's already ready (the warm-session capture got
            // its real EDID on the FIRST fetch, no placeholder at all) doesn't wait needlessly,
            // and a genuinely cold one gets the same long poll DLM itself needs. Still fully
            // bounded so a dock that never delivers one doesn't stall bring-up. NOT YET
            // HW-VERIFIED past what the superseded loop already tested -- see
            // docs/CP-HANDSHAKE.md sec 4f.
            const EDID_STEP_DELAY: Delta = Delta::from_millis(100);
            const EDID_EARLY_ROUNDS: usize = 1;
            const EDID_POLL_ITERS: usize = 250; // bounded; readiness comes fast once HDCP auth completes (reactive phase wait)
            const EDID_POLL_DELAY: Delta = Delta::from_millis(20);
            const EDID_POLL_PROBE_EVERY: usize = 8;
            // PER-HEAD EDID fetch. `byte22` of the id=0x15 probe/fetch (and id=0x16 engage/kick) is the
            // downstream PORT/HEAD selector (decoded 2026-07-20: DLM alternates it 0,1 -- it fetches
            // head 0 then head 1; the real EDID returned on the head with the monitor). Run the whole
            // readiness+fetch cadence once per head so a dual-monitor dock brings up BOTH connectors
            // with their own native EDID. Head 0 is always tried; other heads only when the dock's
            // DISPLAY-CAP (id=0x78) reported a monitor (`heads_present`), to skip a pointless
            // multi-second poll on an empty port. Each head's capture moves from the `edid_out`
            // scratch into `edid_heads[head]` for the caller to install on that head's connector.
            for head in 0..Self::CP_SETUP_HEADS {
                if head != 0 && !heads_present[head] {
                    continue;
                }
                let hu8 = head as u8;
                *edid_out = None;
                let mut edid_ready = false;
                macro_rules! edid_send {
                    ($ep:expr, $body:expr, $tag:expr) => {{
                        if let Ok((ok, e)) = Self::send_live_cp(
                            dev,
                            session,
                            ep84_q.as_mut(),
                            &mut resp,
                            edid_out,
                            $ep,
                            wseq,
                            &$body,
                        ) {
                            drained += e.reads;
                            acks += e.acks;
                            rejects += e.rejects;
                            wseq = wseq.wrapping_add((($body.len() + 15) / 16) as u32);
                            cp_ctr += 1;
                            edid_ready |= e.edid_ready;
                            pr_info!(
                                "vino: live head {} {} {}\n",
                                head,
                                $tag,
                                if ok { "sent" } else { "NAK'd" }
                            );
                        }
                    }};
                }
                'early: for cycle in 0..EDID_EARLY_ROUNDS {
                    if edid_out.is_some() {
                        break;
                    }
                    pr_info!("vino: live get-EDID head {head} early round {cycle}\n");
                    for _ in 0..2 {
                        if let Ok(probe) = cp::get_edid_req_sub(cp_ctr, 0x20, hu8) {
                            edid_send!(0x15, probe, "get-EDID probe (id=0x15 sub=0x20)");
                        }
                        fsleep(EDID_STEP_DELAY);
                    }
                    // The `id=0x16 sub=0x4b` kick (byte22 = head) starts/continues the head's
                    // downstream DDC-EDID read; its effect shows in the next readiness probe.
                    //
                    // Briefly removed on 2026-07-23 because a message inventory of the two WARM
                    // DLM captures showed no `0x16/0x4b` -- but the warm captures were the wrong
                    // reference. The decrypted COLD capture
                    // (`captures/edid-cold-decrypt-20260719/`) has DLM sending it twice, once per
                    // head, and its body already matches DLM byte-for-byte. Compare cold against
                    // cold: a message class can be entirely absent from a warm re-engage and still
                    // be mandatory on a cold plug.
                    if let Ok(kick) = cp::edid_readiness_kick(cp_ctr, hu8) {
                        edid_send!(0x16, kick, "get-EDID kick (id=0x16 sub=0x4b)");
                    }
                    fsleep(EDID_STEP_DELAY);
                    if let Ok(req) = cp::get_edid_req(cp_ctr, hu8) {
                        edid_send!(0x15, req, "get-EDID fetch (id=0x15 sub=0x21)");
                    }
                    if edid_out.is_some() {
                        break 'early;
                    }
                    fsleep(EDID_STEP_DELAY);
                    // The real EDID is an asynchronous push which follows the fetch's immediate
                    // ACK. Passively reap for up to two seconds instead of sending a second full
                    // request round across the just-established per-head video channels. The
                    // known-good session needed exactly one round per head (59 setup ACKs total).
                    let reply_wait = Instant::<Monotonic>::now();
                    while Instant::<Monotonic>::now() - reply_wait < Delta::from_secs(2) {
                        let d = Self::drain_ep84(
                            dev,
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
                        if edid_out.is_some() {
                            break 'early;
                        }
                    }
                }
                {
                    // 2026-07-23: send the engage pair UNCONDITIONALLY. It used to be gated on
                    // `edid_out.is_none()`, so on any run where the EDID arrived from the probe
                    // alone -- which is every run now that the byte22 head selector is fixed --
                    // vino never sent `id=0x16 sub=0x23` at all. A full message inventory of
                    // `captures/dlm-wake-ab-20260722-150209` shows DLM sends it exactly twice on
                    // every session, after the `0x15/0x20`+`0x15/0x21` probe pair has already
                    // succeeded. See `scripts/cp-inventory.py` and `docs/BLOCKER.md`.
                    for _ in 0..2 {
                        if let Ok(engage) = cp::edid_engage_req(cp_ctr, hu8) {
                            edid_send!(0x16, engage, "get-EDID engage (id=0x16 sub=0x0023)");
                        }
                        fsleep(EDID_STEP_DELAY);
                    }
                }
                if edid_out.is_none() {
                    // Wall-clock ceiling on the readiness poll, independent of iteration count, so a
                    // deaf/NAKing dock never wedges module removal (each NAK'd send can block on its
                    // USB timeout; a pure iteration bound could still pin the work item for minutes).
                    const EDID_POLL_MAX: Delta = Delta::from_secs(6);
                    let poll_start = Instant::<Monotonic>::now();
                    'poll: for i in 0..EDID_POLL_ITERS {
                        if edid_out.is_some() || edid_ready {
                            break 'poll;
                        }
                        if Instant::<Monotonic>::now() - poll_start > EDID_POLL_MAX {
                            pr_info!(
                                "vino: get-EDID head {head} readiness poll hit wall-clock cap\n"
                            );
                            break 'poll;
                        }
                        if let Ok(sp) = cp::device_query_req(cp_ctr, 0x000c) {
                            edid_send!(0x14, sp, "device-status poll (id=0x14 sub=0x000c)");
                        }
                        if i % EDID_POLL_PROBE_EVERY == EDID_POLL_PROBE_EVERY - 1 {
                            if let Ok(probe) = cp::get_edid_req_sub(cp_ctr, 0x20, hu8) {
                                edid_send!(
                                    0x15,
                                    probe,
                                    "get-EDID readiness probe (id=0x15 sub=0x20)"
                                );
                            }
                        }
                        if edid_out.is_some() || edid_ready {
                            break 'poll;
                        }
                        fsleep(EDID_POLL_DELAY);
                    }
                    pr_info!(
                        "vino: get-EDID head {head} readiness poll finished (ready={edid_ready})\n"
                    );
                    // Post-readiness fetch loop: the real id=0x194 EDID can land a few messages LATER
                    // than the fetch's sync ack (trailing async push). Fetch + drain repeatedly.
                    for _ in 0..24 {
                        if edid_out.is_some() {
                            break;
                        }
                        if let Ok(req) = cp::get_edid_req(cp_ctr, hu8) {
                            if let Ok((_ok, e)) = Self::send_live_cp(
                                dev,
                                session,
                                ep84_q.as_mut(),
                                &mut resp,
                                edid_out,
                                0x15,
                                wseq,
                                &req,
                            ) {
                                drained += e.reads;
                                acks += e.acks;
                                rejects += e.rejects;
                                wseq = wseq.wrapping_add(((req.len() + 15) / 16) as u32);
                                cp_ctr += 1;
                            }
                        }
                        let d = Self::drain_ep84(
                            dev,
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
                edid_heads[head] = edid_out.take();
                pr_info!(
                    "vino: head {head} EDID fetch {}\n",
                    if edid_heads[head].is_some() {
                        "SUCCEEDED"
                    } else {
                        "no EDID"
                    }
                );
                // Post-EDID capability query, once per head, immediately after this head's EDID has
                // landed -- DLM's cold position (see `cp::post_edid_query`). Absent from both warm
                // captures, so it was invisible until the cold capture was inventoried.
                if let Ok(q) = cp::post_edid_query(cp_ctr, hu8) {
                    edid_send!(0x15, q, "post-EDID capability query (id=0x15 sub=0x53)");
                }
                // `edid_send!` folds the drain's readiness bit into `edid_ready`. This is the last
                // statement of the per-head iteration and the next head re-derives it, so that
                // update is deliberately not read again here.
                let _ = edid_ready;
            }

            // Checkpoint (2026-07-16, post-lockup investigation): the last thing the previous
            // frozen run ever logged was inside this get-EDID loop (round 38) -- `send_cp_setup`
            // never reached its own "CP setup sent" summary log in the caller, so whatever hung
            // did so somewhere between here and this function's return. This line pins down
            // whether the loop itself exits normally.
            pr_info!(
                "vino: get-EDID retry loop finished (edid={})\n",
                edid_out.is_some()
            );

            // Do not mode-set during CP/EDID setup. The physical DLM cold capture performs both
            // EDID/ENGAGE rounds, completes its downstream-readiness interval, and only then sends
            // the modes negotiated independently for head 0 and head 1. Vino instead sent head 0's
            // preferred EDID timing here (1440p60 in the failing capture), before readiness, then
            // KWin replaced it with both real 1440p120 modes. That leaves three mode generations in
            // one cold activation and can poison both heads' shared display-engine state.
            //
            // `VinoCrtc::atomic_enable` already sends the exact negotiated timing for each head and
            // gates its matching framebuffer on that send. It is therefore the only authoritative
            // mode-set path, matching DLM and eliminating this speculative early head-0 mode.
            pr_info!("vino: EDID setup complete -- deferring all mode-sets to KMS\n");
        }

        // Final sweep of the EP83 interrupt queue so a status byte the dock pushed late (after the
        // arm/msg0 burst settled) is still counted before the queue is dropped.
        for _ in 0..4 {
            let e = Self::drain_ep83_queue(dev, ep83_q.as_mut());
            ep83_events += e;
            if e == 0 {
                break;
            }
        }
        // Honest verdict, distinguishing the three failure modes the dock exhibits so a run is
        // never again misread (see the check.md false-positive writeup):
        //   * acks>0            -- a 0x45 reply DECRYPTED to a valid CP header: genuine engagement.
        //   * rejects>0         -- the dock sent 0x45-tagged frames that DON'T decrypt: it is
        //                          talking but ignoring our cipher (the classic wall).
        //   * drained==0        -- the dock never replied on EP84 at all: it fell SILENT.
        //   * else              -- only non-0x45 traffic (cap-phase 0x25 etc.), no CP reply.
        let verdict = if acks > 0 {
            "dock ENGAGED (verified CP ack decrypts under our session key)"
        } else if rejects > 0 {
            "dock REJECTING our CP -- 0x45 replies do not decrypt (the wall)"
        } else if drained == 0 {
            "dock SILENT -- no EP84 reply to our CP at all"
        } else {
            "dock ignoring our CP -- only non-0x45 traffic seen (the wall)"
        };
        if rejects > 0 {
            pr_warn!("vino: dock returned {rejects} undecryptable 0x45 frame(s) -- rejecting our cipher\n");
        }
        pr_info!(
            "vino: CP setup sent={sent} EP84_resp={drained} sub=0x45_acks={acks} rejects={rejects} EP83_events={ep83_events} ({verdict})\n"
        );
        // Hand the caller the running counters: the next free AES-CTR block (`wseq`) and inner
        // message counter (`cp_ctr`), so runtime KMS sends (mode-set/cursor) continue the sequence.
        pr_info!("vino: per-head monitor presence (DISPLAY-CAP id=0x78): {heads_present:?}\n");
        Ok((sent, acks, wseq, cp_ctr))
    }

    /// Seal `content` (inner CP plaintext, WITHOUT the 16-byte tag region) into a live
    /// `type=4 sub=0x24` frame at `wire_seq`, send it on EP02 with EP84 drained between NAK
    /// retries (the single-packet interleave discipline msg0 uses), then drain once more to
    /// collect the dock's reply. `id` selects the DLM-exact `aux` header field
    /// ([`cp::aux_for_id`]). Returns `(sent_ok, ep84_drain)` where the drain tally separates
    /// verified acks from rejects. Used for the post-engagement live messages (get-EDID,
    /// mode-set) once the dock has acked msg0.
    fn send_live_cp(
        dev: &UsbLink<'_>,
        session: &Session,
        mut q: Option<&mut usb::BulkInQueue>,
        resp: &mut [u8],
        edid_out: &mut Option<KVec<u8>>,
        id: u16,
        wire_seq: u32,
        content: &[u8],
    ) -> Result<(bool, Ep84Drain)> {
        let frame = cp::seal_interactive(&session.ks, &session.riv, id, wire_seq, content)?;

        // Single-packet OUT: a NAK transfers nothing, so cancel+retry is safe. Between attempts
        // drain EP84 so the dock can push/drain its IN queue (matches msg0's behaviour).
        const TRIES: usize = 40;
        let mut ok = false;
        let mut tally = Ep84Drain::default();
        for _ in 0..TRIES {
            match dev.ctrl_send(&frame, Delta::from_millis(5), GFP_KERNEL) {
                Ok(_) => {
                    ok = true;
                    break;
                }
                Err(_) => {
                    tally.add(Self::drain_ep84(
                        dev,
                        q.as_deref_mut(),
                        resp,
                        session,
                        edid_out,
                        Delta::from_millis(10),
                    ));
                }
            }
        }
        // Collect the dock's reply (the get-EDID id=0x194 frame is captured here via drain_ep84).
        tally.add(Self::drain_ep84(
            dev,
            q.as_deref_mut(),
            resp,
            session,
            edid_out,
            Delta::from_millis(10),
        ));
        Ok((ok, tally))
    }

    /// sec 5 read-only diagnostic: log one dock->host EP84 frame's wire header
    /// (`type`@4, `sub`@8, `aux`@10, `seq`@12) and, when the body decrypts under the IN
    /// keystream, its inner `(id, sub, ictr)`. Surfaces EVERY frame the dock returns --
    /// not just `sub=0x45` -- so a hardware run reveals whether the dock is mute, NAKing,
    /// or replying with an unexpected sub. Pure logging; no state change.
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
            // Dev diagnostic (pr_debug, compiled out unless dynamic debug is enabled): the raw
            // wire, so the dock's pushes can be offline-decoded. The dock's large capability block
            // (~5787 B) must be dumped in 128-byte CHUNKS, because a single hex print of a
            // >~250-byte
            // array exceeds printk's per-line limit. Capped at 768 B (6 lines) to avoid flooding.
            let cap = len.min(768);
            if cap <= 64 {
                let raw = &frame[..cap];
                pr_debug!("vino: dock EP84 RAW {len}B {raw:02x?}\n");
            } else {
                pr_debug!("vino: dock EP84 RAW {len}B (first {cap} B in 128-B chunks):\n");
                let mut o = 0usize;
                while o < cap {
                    let e = (o + 128).min(cap);
                    let chunk = &frame[o..e];
                    pr_debug!("vino:   ep84[{o:#06x}] {chunk:02x?}\n");
                    o = e;
                }
            }
        }
        match cp::decode_any(&session.ks, &session.riv, frame) {
            Some((rivtag, rid, rsub, rictr, sample)) => {
                pr_info!(
                    "vino: dock EP84 type={wtype} wsub={wsub:#x} aux={aux:#x} seq={wseq:#x} {len}B -> [{rivtag}] id={rid:#x} sub={rsub:#x} ictr={rictr:#x} pt={sample:02x?}\n"
                );
            }
            None => {
                pr_info!(
                    "vino: dock EP84 type={wtype} wsub={wsub:#x} aux={aux:#x} seq={wseq:#x} {len}B (no inner decode)\n"
                );
            }
        }
    }

    /// Read one EP84 frame: from the persistent async queue `q` when [`CP_ASYNC_EP84`] has opened
    /// one, else a synchronous `bulk_recv`. The queue's timeout (`Ok(None)`) is mapped to
    /// `Err(ETIMEDOUT)` so the callers' existing match arms (which treat any `Err`/empty as
    /// "no more data right now") work unchanged across both paths.
    fn read_ep84(
        dev: &UsbLink<'_>,
        q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        to: Delta,
    ) -> Result<usize> {
        match q {
            Some(queue) => match queue.recv(dev.io(), buf, to) {
                Ok(Some(n)) => Ok(n),
                Ok(None) => Err(ETIMEDOUT),
                Err(e) => Err(e),
            },
            None => dev.ctrl_recv(buf, to, GFP_KERNEL),
        }
    }

    fn drain_ep84(
        dev: &UsbLink<'_>,
        mut q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        session: &Session,
        edid_out: &mut Option<KVec<u8>>,
        // Timeout for the FIRST read. Normal per-message drains pass a short value (the dock acks in
        // ~0.14 ms); the HDCP phase-boundary drains pass a long one to REACTIVELY wait out the dock's
        // silent compute window (RSA/H'/L', ~155 ms) for its response, instead of a blind timer.
        first_wait: Delta,
    ) -> Ep84Drain {
        const MAX_READS: usize = 16;
        let mut out = Ep84Drain::default();
        // Read EP84 FIRST (the dock answers in ~0.14 ms, same as it does for DLM). The EP83 status
        // poll is serviced AFTER -- polling it before the EP84 read blocked the critical path for
        // up
        // to 30 ms PER cap frame (timeline diff 2026-06-11: vino's cap phase was 446 ms / ~32 ms
        // per
        // frame vs DLM's 60 ms / 0.14 ms, purely from this ordering), arming the dock ~1 s late.
        for i in 0..MAX_READS {
            let wait = if i == 0 {
                first_wait
            } else {
                Delta::from_millis(10)
            };
            match Self::read_ep84(dev, q.as_deref_mut(), buf, wait) {
                Ok(len) if len > 0 => {
                    out.reads += 1;
                    // sec 5 diagnostic: surface EVERY dock->host frame, not just `sub=0x45`,
                    // so a hardware run shows what the dock actually returns (a different
                    // sub, a NAK, or plaintext) instead of a bare `EP84_resp=N` count.
                    Self::log_ep84(session, &buf[..len]);
                    // The dock's fresh per-head rrx rides an `id=0x10 sub=0x84` push (inner msg-id
                    // AKE_SEND_RRX). Capture it so the per-head AKE derives kd/edkey/V from the right
                    // rrx (see `cp::perhead_rrx` -- this is the EDID repeater-auth gate).
                    if out.perhead_rrx.is_none() {
                        out.perhead_rrx = cp::perhead_rrx(&session.ks, &session.riv, &buf[..len]);
                    }
                    if len >= 10 && u16::from_le_bytes([buf[8], buf[9]]) == 0x45 {
                        // A `0x45` wire tag is NECESSARY but not SUFFICIENT: the dock emits
                        // `0x45`-tagged heartbeat/status frames even when it has NOT engaged our
                        // cipher. Only count it as an ack if it actually DECRYPTS to a valid CP
                        // header under the session key (`cp::verify_in_ack`). A tag that fails to
                        // decrypt is the dock REJECTING us -- surface it loudly so a future run
                        // cannot mistake this traffic for engagement (the historical false positive).
                        match cp::verify_in_ack(&session.ks, &session.riv, &buf[..len]) {
                            Some((id, sub, ctr)) => {
                                out.acks += 1;
                                pr_info!(
                                    "vino: dock CP ACK VERIFIED -- 0x45 reply decrypts to id={id:#x} sub={sub:#x} ctr={ctr} under the session key (cipher ENGAGED)\n"
                                );
                                // Per-head DISPLAY-CAP: the dock answers each head's stream-open
                                // (`id=0x14 sub=0x30`) with `id=0x78 sub=0x30` ONLY when that head
                                // has a monitor, echoing the request ctr. Record it so the caller
                                // can mark the matching head's connector connected (dual-monitor).
                                if id == 0x78 && sub == 0x30 {
                                    out.display_cap_ctr = Some(ctr);
                                }
                                // Capture the dock's EDID the first time it appears (id=0x194
                                // sub=0x21 reply to the replayed get-EDID request). Reuses the
                                // standard DRM EDID infra in get_modes. See CONTROL-PLANE.md.
                                if edid_out.is_none() {
                                    if let Ok(Some(e)) = cp::parse_edid_from_reply(
                                        &session.ks,
                                        &session.riv,
                                        &buf[..len],
                                    ) {
                                        pr_info!("vino: EDID read from dock ({} bytes)\n", e.len());
                                        *edid_out = Some(e);
                                    }
                                }
                                // Track the dock's downstream-DDC readiness bit (a `sub=0x0020`
                                // probe reply's inner offset 26) so the get-EDID loop can tell
                                // "still working" apart from "genuinely never going to answer"
                                // instead of just retrying blind. See `cp::edid_poll_ready`.
                                if let Some(true) =
                                    cp::edid_poll_ready(&session.ks, &session.riv, &buf[..len])
                                {
                                    out.edid_ready = true;
                                }
                            }
                            None => {
                                match cp::decode_in_lenient(&session.ks, &session.riv, &buf[..len])
                                {
                                    // Decrypts fine to a plausible CP header, just a `sub` not yet in
                                    // `cp::is_known_sub`. The cipher IS engaged -- this is a valid ack,
                                    // not a rejection. Count it and log softly (so the sub can be
                                    // catalogued), NOT the alarming "the wall" false alarm. (Diagnosed
                                    // 2026-07-19: decrypting every reply against its wire seq, all 173 of
                                    // a session decode cleanly; the old branch mis-flagged unlisted-sub
                                    // acks as rejects.)
                                    Some((id, sub, ctr)) => {
                                        out.acks += 1;
                                        pr_info!(
                                        "vino: dock CP reply decodes under our key (unlisted sub) -- id={id:#x} sub={sub:#x} ctr={ctr}; valid ack, add sub to cp::is_known_sub\n"
                                    );
                                    }
                                    // Genuinely fails to decrypt to a sane header under any riv variant.
                                    None => {
                                        out.rejects += 1;
                                        pr_warn!(
                                        "vino: dock 0x45 reply does NOT decrypt under the session key (garbage inner header, even lenient) -- genuine reject\n"
                                    );
                                    }
                                }
                            }
                        }
                    }
                }
                _ => break,
            }
        }
        // Service EP83 AFTER draining EP84, so it never delays reading the dock's CP reply.
        if Self::POLL_EP83_DURING_BRINGUP {
            Self::poll_ep83(dev);
        }
        out
    }

    /// Lockstep counterpart to [`drain_ep84`]: after one CP OUT, drain EP84 until the
    /// `sub=0x45` reply whose **inner counter echoes** `ictr` arrives (DLM's 1:1 handshake) or
    /// the short read budget elapses. Like `drain_ep84`, only a frame that DECRYPTS to a valid
    /// CP header counts as an ack (`cp::verify_in_ack`); a `0x45` tag that fails to decrypt is a
    /// reject and is surfaced as such. Any async pushes seen meanwhile are still counted and
    /// scanned for the EDID. Returns the drain tally (the ictr-echo match only shortens the
    /// wait; it is not itself the engagement signal).
    fn lockstep_reply(
        dev: &UsbLink<'_>,
        mut q: Option<&mut usb::BulkInQueue>,
        buf: &mut [u8],
        session: &Session,
        ictr: u16,
        edid_out: &mut Option<KVec<u8>>,
    ) -> Ep84Drain {
        const MAX_READS: usize = 8;
        let mut out = Ep84Drain::default();
        for _ in 0..MAX_READS {
            match Self::read_ep84(dev, q.as_deref_mut(), buf, Delta::from_millis(30)) {
                Ok(len) if len > 16 => {
                    out.reads += 1;
                    // sec 5 diagnostic: log every frame the dock returns in the lockstep
                    // window -- including the non-`0x45` frames we otherwise skip -- so the
                    // divergence point is paired with the dock's actual reply on the wire.
                    Self::log_ep84(session, &buf[..len]);
                    if u16::from_le_bytes([buf[8], buf[9]]) != 0x45 {
                        continue;
                    }
                    match cp::verify_in_ack(&session.ks, &session.riv, &buf[..len]) {
                        Some((id, sub, ctr)) => {
                            out.acks += 1;
                            let echo = if ctr == ictr {
                                " (echoes our ictr)"
                            } else {
                                ""
                            };
                            pr_info!(
                                "vino: dock CP ACK VERIFIED (lockstep) -- id={id:#x} sub={sub:#x} ctr={ctr}{echo}\n"
                            );
                            // Opportunistically capture the EDID (id=0x194 reply, off 22).
                            if edid_out.is_none() {
                                if let Ok(Some(e)) = cp::parse_edid_from_reply(
                                    &session.ks,
                                    &session.riv,
                                    &buf[..len],
                                ) {
                                    pr_info!("vino: EDID read from dock ({} bytes)\n", e.len());
                                    *edid_out = Some(e);
                                }
                            }
                            // Stop early once the dock acks the counter we sent (DLM's 1:1 echo).
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
                                pr_info!(
                                    "vino: dock CP reply decodes (lockstep, unlisted sub) -- id={id:#x} sub={sub:#x} ctr={ctr}; valid ack, add sub to cp::is_known_sub\n"
                                );
                            }
                            None => {
                                out.rejects += 1;
                                pr_warn!(
                                    "vino: dock 0x45 reply does NOT decrypt (lockstep, garbage even lenient) -- genuine reject\n"
                                );
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

kernel::usb_device_table!(
    USB_TABLE,
    MODULE_USB_TABLE,
    <VinoDriver as usb::Driver>::IdInfo,
    [(usb::DeviceId::from_id(VID_DISPLAYLINK, PID_D6000), ())]
);

impl usb::Driver for VinoDriver {
    type IdInfo = ();
    type Data<'bound> = VinoBoundData<'bound>;
    const ID_TABLE: usb::IdTable<Self::IdInfo> = &USB_TABLE;

    fn probe<'bound>(
        intf: &'bound usb::Interface<Core<'_>>,
        _id: &usb::DeviceId,
        _info: &'bound Self::IdInfo,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
        let cdev: &device::Device<Core<'_>> = intf.as_ref();
        // The D6000 exposes several interfaces (0/1/5/6 match us; 2-4 are audio).
        // The control endpoints (0x02/0x84) and the whole HDCP session live on
        // interface 0 -- drive bring-up only there so we don't run the preamble and
        // AKE four times and pollute the dock's state machine. Other interfaces
        // bind (so usbcore doesn't hand them to another driver) but stay idle.
        // An interface with no active alternate setting has no endpoints to drive.
        let ifnum = intf.number().ok_or(ENODEV)?;
        // Track composite-interface enumeration for `bring_up`'s post-session-init claim wait (see
        // PROBED_IFACES). Interface 0 probes first and (re)starts a bring-up, so reset the mask there;
        // then record every interface vino is offered (bound OR declined) -- being offered interface N
        // means the kernel has enumerated it.
        if ifnum == 0 {
            PROBED_IFACES.store(0, core::sync::atomic::Ordering::Release);
            SAW_0B.store(false, core::sync::atomic::Ordering::Release);
            // Loop breaker: if interface 0 has reprobed LOOP_THRESHOLD times within LOOP_WINDOW of
            // each other, something (most likely our own bring-up re-triggering a dock-firmware
            // fault on re-enumeration) is looping. Stop here, before any USB I/O for this plug --
            // bind idle with no DRM card node, no AKE, no CP, nothing left for vino to keep
            // re-triggering the dock's disconnect with. See `record_probe_and_check_loop`.
            if record_probe_and_check_loop() {
                dev_info!(
                    cdev,
                    "vino: LOOP DETECTED -- interface 0 has reprobed {LOOP_THRESHOLD}+ times \
                     within {}s of each other. Refusing to start bring-up so vino stops \
                     re-triggering whatever is making the dock disconnect. No DRM card node, no \
                     AKE, no CP, no video writes will be attempted on this or further plugs. \
                     Clear via `echo 0 > /sys/devices/vino/probe_loop_tripped` (or a module \
                     reload) once the dock is confirmed stable, then replug.\n",
                    LOOP_WINDOW.as_millis() / 1000
                );
                // Still record this interface for `remove_all`/`modprobe -r` -- it's bound (just
                // idle), so it must stay reachable for teardown like any other bound interface.
                VINO_IFACES.lock()[0] = Some(intf.into());
                return Ok(VinoBoundData {
                    _intf: intf.into(),
                    io: None,
                    registration: None,
                    bringup: KBox::pin_init(new_mutex!(None), GFP_KERNEL)?,
                    _i2c: core::array::from_fn(|_| None),
                });
            }
        }
        PROBED_IFACES.fetch_or(
            1u32 << (u32::from(ifnum) & 31),
            core::sync::atomic::Ordering::Release,
        );
        // Record ONLY the interfaces vino actually keeps (0 = control, 1 = idle) for the `remove_all`
        // sysfs teardown -- never the declined ones (audio/ethernet), or remove_all would unbind them
        // from their own drivers.
        if ifnum == 0 || ifnum == 1 {
            VINO_IFACES.lock()[ifnum as usize] = Some(intf.into());
        }
        if ifnum != 0 {
            // Interface 1 (app-specific/DFU) is the only other one DLM claims; let everything else
            // (audio 2-4, Ethernet 5-6) fall through to its proper kernel driver. Returning ENODEV
            // tells usbcore this driver doesn't handle the interface, so it tries the next match.
            if ifnum != 1 {
                dev_info!(
                    cdev,
                    "vino: declining D6000 interface {ifnum} (left to its class driver)\n"
                );
                return Err(ENODEV);
            }
            dev_info!(
                cdev,
                "vino: bound D6000 interface {ifnum} (idle -- control is iface 0)\n"
            );
            return Ok(VinoBoundData {
                _intf: intf.into(),
                io: None,
                registration: None,
                bringup: KBox::pin_init(new_mutex!(None), GFP_KERNEL)?,
                _i2c: core::array::from_fn(|_| None),
            });
        }
        dev_info!(
            cdev,
            "vino: bound DisplayLink D6000 -- plaintext session bring-up\n"
        );
        // A fresh binding of the control interface is the ONLY thing that lifts the teardown latch;
        // see `SESSION_TEARDOWN`. Cleared here, before anything can arm the keepalive.
        SESSION_TEARDOWN.store(false, core::sync::atomic::Ordering::SeqCst);

        // Phase 3: register a real DRM/KMS device on the control interface so the dock
        // shows up as a mode-settable `card`/`renderD` node (atomic KMS via the simple
        // display pipe, one 1080p virtual connector, GEM-shmem dumb buffers). Non-fatal:
        // bring-up still proceeds (and the interface still binds) if any step fails, so
        // a DRM-core hiccup can't regress the USB session work.
        // Hold a refcounted handle to the bound interface; one copy goes into the DRM
        // device-private (for the EP08 scanout path), one stays in `VinoDriver`.
        let intf_ref: ARef<usb::Interface> = intf.into();

        // Resolve the dock's endpoints against interface 0's descriptor once, so every later
        // transfer names a direction/type-checked endpoint instead of a bare address.
        let eps = Endpoints::resolve(intf)?;

        // Open the USB I/O window. SAFETY: this is a successful `probe()` of the interface, so
        // I/O on it is permitted, and `disconnect()` below closes the window before it returns.
        let io = Arc::pin_init(unsafe { usb::IoWindow::new(intf_ref.clone()) }, GFP_KERNEL)?;

        // DRM device lifecycle: allocate an `UnregisteredDevice`, wire up the KMS pipeline on it
        // while still unregistered, then register it. The `Registration` is stored in the bound
        // data below, so the card is unregistered by the ordered unbind rather than by a
        // driver-local force-unplug.
        let mut registration = None;
        let ddev: Option<ARef<drm_sink::VinoDrmDevice>> = match drm::UnregisteredDevice::<
            drm_sink::VinoDrmDriver,
        >::new(
            intf,
            drm_sink::VinoDrmData::new(io.clone(), eps),
            &THIS_MODULE,
        ) {
            // `create_objects()` (which builds the CRTC/plane/connector/encoder) runs
            // automatically inside `UnregisteredDevice::new` above -- there is no separate
            // KMS init step to call.
            Ok(unreg) => {
                // `Core` derefs to `Bound`; name the context explicitly so `as_ref()` resolves
                // to the bound `struct device` the registration wants.
                let bound_intf: &usb::Interface<device::Bound> = intf;
                let parent: &device::Device<device::Bound> = bound_intf.as_ref();
                // SAFETY: the registration is stored in the returned `VinoBoundData` and is
                // therefore dropped during this driver's ordered unbind; it is never leaked.
                match unsafe { drm::Registration::new(parent, unreg, (), 0) } {
                    Ok(reg) => {
                        dev_info!(cdev, "vino: DRM+KMS device registered (card node live)\n");
                        let ddev: ARef<drm_sink::VinoDrmDevice> = reg.device().into();
                        registration = Some(reg);
                        Some(ddev)
                    }
                    Err(e) => {
                        dev_info!(cdev, "vino: DRM registration failed ({e:?}) -- continuing without card node\n");
                        None
                    }
                }
            }
            Err(e) => {
                dev_info!(
                    cdev,
                    "vino: drm::UnregisteredDevice::new failed ({e:?}) -- continuing\n"
                );
                None
            }
        };

        // Bring-up (preamble + HDCP AKE + ~6 s of lockstep CP replay) is all blocking
        // synchronous USB I/O. Running it inline here pins the USB driver-model probe
        // thread while the DRM card node is already registered and live, which stalled
        // the compositor (KWin) on first plug until the dock was physically yanked. Hand
        // it to the system workqueue so `probe()` returns immediately and userspace KMS
        // stays responsive. The work item holds refcounted handles to the interface (for
        // the bulk endpoints) and the DRM device (for EDID caching), so they outlive
        // `probe()`; USB I/O after an intervening disconnect simply errors and is logged,
        // exactly like any other failed bring-up step.
        // Retain the bring-up handle so `disconnect()` can flush it; enqueue a clone.
        let bringup = ddev.as_ref().and_then(|d| match BringUp::new(d.clone()) {
            Ok(work) => {
                let _ = workqueue::system().enqueue(work.clone());
                dev_info!(cdev, "vino: bring-up queued on system workqueue\n");
                Some(work)
            }
            Err(e) => {
                dev_info!(cdev, "vino: failed to queue bring-up ({e:?}) -- WIP\n");
                None
            }
        });

        // Register one DDC/CI I2C adapter per display head, matching DLM/EVDI. The adapter context
        // supplies the head selector carried at CP plaintext offset 22.
        let mut i2c = core::array::from_fn(|_| None);
        if let Some(d) = ddev.as_ref() {
            let names = [c"DisplayLink DDC/CI head 0", c"DisplayLink DDC/CI head 1"];
            for head in 0..drm_sink::HEADS {
                match kernel::i2c::BusAdapter::<drm_sink::VinoI2c>::new(
                    names[head],
                    cdev,
                    drm_sink::VinoI2cContext {
                        dev: d.clone(),
                        head: head as u8,
                    },
                ) {
                    Ok(a) => i2c[head] = Some(a),
                    Err(e) => {
                        dev_info!(
                            cdev,
                            "vino: DDC/CI head {head} I2C adapter registration failed ({e:?})\n"
                        );
                    }
                }
            }
        }

        Ok(VinoBoundData {
            _intf: intf_ref,
            io: Some(io),
            registration,
            bringup: KBox::pin_init(new_mutex!(bringup), GFP_KERNEL)?,
            _i2c: i2c,
        })
    }

    /// Stop asynchronous bring-up and unplug DRM before the USB interface loses its bound
    /// typestate. Generic driver teardown drops `VinoDriver` after this callback returns; taking
    /// the owned work handle here is about ordering, not compensating for missing drvdata cleanup.
    /// Stop all USB I/O before the interface is suspended.
    ///
    /// Closing the window cancels the persistent queue URBs and waits for any in-flight transfer,
    /// so no I/O is outstanding when this returns -- which is what the USB core requires and what
    /// `IoWindow`'s contract promises.
    fn suspend<'bound>(
        _intf: &'bound usb::Interface<Core<'_>>,
        data: Pin<&VinoBoundData<'bound>>,
    ) -> Result {
        if let Some(io) = data.io.as_ref() {
            io.close();
        }
        Ok(())
    }

    /// I/O is permitted again after a suspend.
    fn resume<'bound>(
        _intf: &'bound usb::Interface<Core<'_>>,
        data: Pin<&VinoBoundData<'bound>>,
    ) -> Result {
        if let Some(io) = data.io.as_ref() {
            io.reopen();
        }
        Ok(())
    }

    /// The device was reset while suspended, so the dock has lost the whole CP session.
    ///
    /// Reopen the window -- transfers are permitted again -- but the control plane must be
    /// re-established from scratch, which only a fresh bring-up does. Nothing re-arms it here yet.
    fn reset_resume<'bound>(
        intf: &'bound usb::Interface<Core<'_>>,
        data: Pin<&VinoBoundData<'bound>>,
    ) -> Result {
        Self::resume(intf, data)
    }

    /// Stop all USB I/O before the device is reset.
    fn pre_reset<'bound>(
        _intf: &'bound usb::Interface<Core<'_>>,
        data: Pin<&VinoBoundData<'bound>>,
    ) -> Result {
        if let Some(io) = data.io.as_ref() {
            io.close();
        }
        Ok(())
    }

    /// I/O is permitted again after the reset.
    fn post_reset<'bound>(
        _intf: &'bound usb::Interface<Core<'_>>,
        data: Pin<&VinoBoundData<'bound>>,
    ) -> Result {
        if let Some(io) = data.io.as_ref() {
            io.reopen();
        }
        Ok(())
    }

    fn disconnect<'bound>(
        intf: &'bound usb::Interface<Core<'_>>,
        data: Pin<&VinoBoundData<'bound>>,
    ) {
        // The order below is the one that survives a disconnect arriving *during* bring-up, which
        // is what wedged `usb_hub_wq` for a whole boot on 2026-07-27 (see `SESSION_TEARDOWN`):
        //
        //   1. latch teardown, so nothing the outgoing session runs can re-arm the keepalive;
        //   2. stop the looping DRM producers (they poll `shutting_down` every iteration);
        //   3. `cancel_sync` the bring-up work, which now drops out of its loops promptly;
        //   4. only then `close()` the I/O window.
        //
        // Doing (4) before (3) is precisely the deadlock: `BringUp::run` holds one `Io` token for
        // its entire body, `close()` waits for that token to be dropped, and the body was looping
        // forever. Closing first was meant to release anything blocked in `recv()`, but every
        // `recv`/`flush` in this driver is bounded by a timeout, so (3) terminates without it.
        //
        // Only the control interface owns the bring-up work and the keepalive; latching on any
        // other interface's unbind would leave a live session with no CP heartbeat, which makes the
        // dock time out and hard-reset.
        let control = intf.number() == Some(0);
        if control {
            SESSION_TEARDOWN.store(true, core::sync::atomic::Ordering::SeqCst);
            KEEPALIVE_RUN.store(false, core::sync::atomic::Ordering::SeqCst);
        }
        let dev: &device::Device<Core<'_>> = intf.as_ref();
        // Drop this interface from the `remove_all` table (idempotent with a concurrent remove_all).
        if let Some(number) = intf.number() {
            VINO_IFACES.lock()[(number as usize) & 7] = None;
        }

        // Publish the producers' stop flag before waiting on anything. This is only the flag: the
        // teardown that must not run until USB I/O is quiesced (vblank timers, the device's
        // self-reference cycles) still happens in `shutdown()` further down.
        if let Some(reg) = data.registration.as_ref() {
            let drm_data: &drm_sink::VinoDrmData = reg.device();
            drm_data.begin_shutdown();
        }

        // Take the sole driver-owned bring-up handle. The queued work holds its own Arc until it
        // runs or is cancelled, while this local Arc keeps the embedded Work pinned and live
        // throughout the `cancel_sync` below.
        let bringup = data.bringup.lock().take();

        // Flush the deferred bring-up before the interface is unbound: `cancel_sync` dequeues it
        // if pending and blocks until it returns if already running, so no USB I/O races the
        // unbind. Safe when the work already finished or never ran -- it then simply reports that
        // nothing was pending. The reclaimed `Arc<BringUp>` (returned only if the work was still
        // queued) is dropped here.
        if let Some(work) = bringup.as_ref() {
            drop(work.work.cancel_sync());
        }

        // Close the USB I/O window: new transfers are refused immediately, the persistent queues'
        // URBs are cancelled, and this blocks until every in-flight transfer has finished.
        // Everything below therefore runs with no USB I/O outstanding, which is what makes
        // `IoWindow::new`'s safety contract hold.
        if let Some(io) = data.io.as_ref() {
            io.close();
        }

        // Stop every producer. The DRM device itself is unregistered by `Registration`'s `Drop`
        // when the bound data is released -- the accepted registration teardown already calls
        // `drm_dev_unplug()`, so there is no driver-local force-unplug here any more.
        //
        // Taken from the registration rather than from `bringup`: `shutdown()` is also what breaks
        // the device's self-references (the vblank timer's published `CrtcRef` and the pinned
        // vblank ref), so skipping it leaks the whole `drm_device` and its DRM minor. A card can
        // exist with `bringup == None` -- `BringUp::new()` is a fallible allocation and probe
        // deliberately continues without it -- and that path used to skip teardown entirely.
        if let Some(reg) = data.registration.as_ref() {
            let drm_data: &drm_sink::VinoDrmData = reg.device();
            drm_data.shutdown();
        }
        dev_info!(dev, "vino: D6000 disconnected\n");
        // `bringup` drops here, releasing its DRM reference before generic driver-data teardown.
    }
}

/// Backs `/sys/devices/vino/remove_all`. Writing any value to `remove_all` unbinds every
/// vino-held interface via `device_release_driver()` (each triggers vino's clean disconnect),
/// after which `modprobe -r vino` unloads the module. The USB-driver analogue of revdi's
/// `/sys/devices/evdi/remove_all`.
#[pin_data]
struct VinoControl {}

impl kernel::sysfs::DeviceAttributes for VinoControl {
    const ATTRS: &'static [kernel::sysfs::Attr] = &[
        kernel::sysfs::Attr::wo(c"remove_all"),
        kernel::sysfs::Attr::rw(c"probe_loop_tripped"),
        kernel::sysfs::Attr::rw(c"dock_pixel_budget"),
        kernel::sysfs::Attr::rw(c"head_max_refresh"),
        kernel::sysfs::Attr::wo(c"reengage"),
        kernel::sysfs::Attr::rw(c"blank_marker"),
        kernel::sysfs::Attr::rw(c"record_sub_parity"),
        kernel::sysfs::Attr::wo(c"simulate_unplug"),
        kernel::sysfs::Attr::rw(c"record_band_order"),
        kernel::sysfs::Attr::rw(c"test_probe_absent"),
    ];

    fn show(&self, name: &CStr, out: &mut kernel::sysfs::Writer<'_>) -> Result {
        if name == c"probe_loop_tripped" {
            let tripped = PROBE_LOOP.lock().tripped;
            return out.write(if tripped { b"1\n" } else { b"0\n" });
        }
        if name == c"dock_pixel_budget" {
            let v = drm_sink::pixel_budget_override();
            let s = kernel::str::CString::try_from_fmt(kernel::prelude::fmt!("{v}\n"))?;
            return out.write(s.to_bytes());
        }
        // The refresh ceiling actually in force, not the override -- unlike the dock budget this
        // one prunes the advertised mode list, so "why is 165 Hz missing" wants the effective
        // figure, and reading back a bare `0` would answer the wrong question.
        if name == c"head_max_refresh" {
            let v = drm_sink::max_refresh_hz();
            let s = kernel::str::CString::try_from_fmt(kernel::prelude::fmt!("{v}\n"))?;
            return out.write(s.to_bytes());
        }
        if name == c"record_sub_parity" {
            let on = drm_sink::band_parity_bit();
            return out.write(if on { b"1\n" } else { b"0\n" });
        }
        if name == c"record_band_order" {
            let on = drm_sink::interlaced_bands();
            return out.write(if on { b"1\n" } else { b"0\n" });
        }
        if name == c"test_probe_absent" {
            let v = drm_sink::probe_forced_absent();
            let s = kernel::str::CString::try_from_fmt(kernel::prelude::fmt!("{v}\n"))?;
            return out.write(s.to_bytes());
        }
        if name == c"blank_marker" {
            let v = drm_sink::blank_marker_state();
            let s = kernel::str::CString::try_from_fmt(kernel::prelude::fmt!("{v}\n"))?;
            return out.write(s.to_bytes());
        }
        Err(EINVAL)
    }

    fn store(&self, name: &CStr, buf: &[u8]) -> Result {
        // Total dock pixel-rate budget in pixels/sec; `0` restores the compiled-in default. See
        // `drm_sink::PIXEL_BUDGET_OVERRIDE` for why this is writable rather than a fixed constant.
        // Takes effect on the next `mode_valid`/`atomic_check`, so re-probe the connector (or
        // toggle the output) for a changed value to show up in the advertised mode list.
        //
        // `head_max_refresh` is the same knob for the per-head REFRESH ceiling in Hz
        // (`drm_sink::DOCK_MAX_REFRESH_HZ`, 120 -- the highest rate this dock has ever been shown to
        // display; DLM clamps 180 Hz to 120 on the wire and vino's own 165/180 attempts went dark).
        // That one prunes the advertised mode list, so a change needs a connector re-probe to show
        // up in `kscreen-doctor`/`xrandr`; `echo 165` re-runs the open high-refresh experiment.
        if name == c"dock_pixel_budget" || name == c"head_max_refresh" {
            let mut v: u32 = 0;
            let mut digits = 0;
            for &b in buf {
                if b.is_ascii_digit() {
                    v = v
                        .checked_mul(10)
                        .and_then(|v| v.checked_add((b - b'0') as u32))
                        .ok_or(EINVAL)?;
                    digits += 1;
                } else if b == b'\n' || b == b'\r' || b == b' ' || b == b'\t' {
                    break;
                } else {
                    return Err(EINVAL);
                }
            }
            if digits == 0 {
                return Err(EINVAL);
            }
            if name == c"head_max_refresh" {
                drm_sink::set_max_refresh_override(v);
                let eff = drm_sink::max_refresh_hz();
                pr_info!(
                    "vino: head_max_refresh override set to {v} Hz via sysfs (in force: {eff} Hz)\n"
                );
                return Ok(());
            }
            drm_sink::set_pixel_budget_override(v);
            pr_info!("vino: dock_pixel_budget override set to {v} px/s via sysfs\n");
            return Ok(());
        }
        // Re-run one head's downstream sink enable (`echo 0 > /sys/devices/vino/reengage`).
        //
        // `reengage_head` is the fix for a monitor replug and it had never once run: its only
        // trigger was a runtime presence transition the watcher never reported, so the fix and the
        // detection it depends on could only ever be tested together. This separates them -- the
        // sequence can be exercised on demand, and it doubles as the manual recovery for a
        // replugged monitor that stayed dark. The work happens on the keepalive loop, which already
        // owns a live I/O token; see `drm_sink::REENGAGE_REQUEST`.
        // Fault injection: pin a head's presence probe negative, reproducing the `id=0x14` the
        // dock really answers after a re-engage. See `drm_sink::PROBE_FORCED_ABSENT`.
        if name == c"test_probe_absent" {
            let mask = match buf.first() {
                Some(&b) if b.is_ascii_digit() => (b - b'0') as u32,
                _ => return Err(EINVAL),
            };
            drm_sink::set_probe_forced_absent(mask);
            pr_info!("vino: presence probe forced absent for head mask 0x{mask:x} via sysfs\n");
            return Ok(());
        }
        // Band order: 0 = raster (vino's shape), 1 = all even bands then all odd (DLM's
        // `ep0b_frame1` shape). Like the parity bit, no pixel decoder can see the difference.
        if name == c"record_band_order" {
            let on = buf.first() == Some(&b'1');
            drm_sink::set_interlaced_bands(on);
            pr_info!(
                "vino: record band order set to {} via sysfs -- next frame is a keyframe\n",
                if on { "INTERLACED (even bands, then odd)" } else { "raster" }
            );
            return Ok(());
        }
        // The `sub` bit-4 y-parity flag on image records. The only oracle for this is a person
        // looking at the panel, so it has to be flippable live; see `drm_sink::BAND_PARITY_BIT`.
        if name == c"record_sub_parity" {
            let on = buf.first() == Some(&b'1');
            drm_sink::set_band_parity_bit(on);
            pr_info!(
                "vino: record sub bit4 y-parity {} via sysfs -- next frame is a keyframe\n",
                if on { "ENABLED" } else { "disabled" }
            );
            return Ok(());
        }
        // Candidate `id=0x16 sub=0x2e` off23 state for the blank path, or 0 to disable. The whole
        // point is to sweep it without a rebuild; see `drm_sink::BLANK_MARKER_STATE`.
        if name == c"blank_marker" {
            let v = match buf.first() {
                Some(&b) if b.is_ascii_digit() => (b - b'0') as u32,
                _ => return Err(EINVAL),
            };
            if v > 7 {
                return Err(EINVAL);
            }
            drm_sink::set_blank_marker_state(v);
            pr_info!("vino: blank_marker candidate set to 2e({v}) via sysfs\n");
            return Ok(());
        }
        // Drop a head's sink without claiming the silence, so the removal path sees exactly what a
        // physical unplug produces. See `drm_sink::request_simulated_unplug`.
        if name == c"simulate_unplug" {
            let head = match buf.first() {
                Some(&b) if b.is_ascii_digit() => (b - b'0') as u32,
                _ => return Err(EINVAL),
            };
            if head as usize >= VinoDriver::CP_SETUP_HEADS {
                return Err(EINVAL);
            }
            drm_sink::request_simulated_unplug(head);
            pr_info!("vino: head {head} simulated unplug queued via sysfs\n");
            return Ok(());
        }
        if name == c"reengage" {
            let head = match buf.first() {
                Some(&b) if b.is_ascii_digit() => (b - b'0') as u32,
                _ => return Err(EINVAL),
            };
            if head as usize >= VinoDriver::CP_SETUP_HEADS {
                return Err(EINVAL);
            }
            drm_sink::request_reengage(head);
            pr_info!("vino: head {head} sink re-engage queued via sysfs\n");
            return Ok(());
        }
        if name == c"probe_loop_tripped" {
            let on = buf.first() == Some(&b'1');
            let mut st = PROBE_LOOP.lock();
            st.tripped = on;
            st.count = 0;
            st.last = None;
            pr_info!(
                "vino: probe_loop_tripped -- {} via sysfs\n",
                if on { "force-TRIPPED" } else { "cleared" }
            );
            return Ok(());
        }
        if name == c"remove_all" {
            // Take every owned reference before unbinding. `disconnect()` also locks the registry,
            // so no driver-core call may happen while this guard is held.
            let interfaces = core::mem::replace(&mut *VINO_IFACES.lock(), [const { None }; 8]);
            let mut n = 0;
            for intf in interfaces.into_iter().flatten() {
                // The owned `ARef<Interface>` keeps the device live across the call, which
                // performs the standard synchronous unbind and invokes vino's disconnect callback
                // from process context. This is the narrow reviewed operation that replaced the
                // raw `dev_ptr()` getter.
                intf.release_driver();
                n += 1;
            }
            pr_info!("vino: remove_all -- released {n} interface(s); module can now be unloaded\n");
        }
        Ok(())
    }
}

/// Module root: owns the USB driver registration and the `/sys/devices/vino` control group.
#[pin_data]
struct VinoModule {
    _sysfs: Option<KBox<kernel::sysfs::AttributeGroup<VinoControl>>>,
    #[pin]
    _driver: kernel::driver::Registration<kernel::usb::Adapter<VinoDriver>>,
}

impl kernel::InPlaceModule for VinoModule {
    fn init(module: &'static kernel::ThisModule) -> impl PinInit<Self, Error> {
        // SAFETY: called exactly once, here, before any `probe()` (which is the only other user
        // of these global locks) can possibly run.
        unsafe { VINO_IFACES.init() };
        unsafe { PROBE_LOOP.init() };
        try_pin_init!(Self {
            // Best-effort: if the control group fails to register the driver still loads and the
            // standard `/sys/bus/usb/drivers/vino/unbind` path (and `scripts/vino-remove.sh`) remain.
            _sysfs: kernel::sysfs::AttributeGroup::register_root(
                c"vino", module, try_pin_init!(VinoControl {}))
                .inspect_err(|e| pr_err!("vino: remove_all sysfs registration failed ({e:?})\n"))
                .ok(),
            _driver <- kernel::driver::Registration::new(c"vino", module),
        })
    }
}

module! {
    type: VinoModule,
    name: "vino",
    authors: ["Mike Lothian"],
    description: "DisplayLink DL3 (Vino) open driver",
    license: "GPL v2",
}

/// Build a minimal valid 128-byte EDID with a 1920x1080@60 detailed timing at base-block
/// offset `dtd_at` (54 = preferred slot), a correct checksum, and the standard magic.
#[cfg(CONFIG_KUNIT)]
fn mk_test_edid(dtd_at: usize) -> [u8; 128] {
    let mut e = [0u8; 128];
    e[..8].copy_from_slice(&[0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00]);
    // 1920x1080@60: pclk 14850 (148.5 MHz, 10 kHz units); hblank 280, vblank 45;
    // hsync_front 88, hsync_width 44, vsync_front 4, vsync_width 5.
    let dtd: [u8; 18] = [
        0x02, 0x3a, // pixel clock 0x3a02 LE
        0x80, 0x18, 0x71, // hactive 1920 / hblank 280 (high nibbles in byte 4)
        0x38, 0x2d, 0x40, // vactive 1080 / vblank 45 (high nibbles in byte 7)
        0x58, 0x2c, 0x45, 0x00, // hsync/vsync front+width
        0, 0, 0, 0, 0, 0, // trailing flags (DTD is 18 bytes total)
    ];
    e[dtd_at..dtd_at + 18].copy_from_slice(&dtd);
    let s = e[..127].iter().fold(0u8, |a, &b| a.wrapping_add(b));
    e[127] = 0u8.wrapping_sub(s); // base-block checksum: all 128 bytes sum to 0
    e
}

/// Offline self-tests for the pure protocol builders/parsers and the crypto bindings the
/// control plane relies on. Gated behind `CONFIG_KUNIT` (the macro adds the cfg), so they
/// have zero effect on a production build; run with a KUnit-enabled kernel. The crypto cases
/// are published known-answer vectors (FIPS-197 AES-128, RFC 4493 AES-CMAC); the seal case is
/// a live round-trip; the rest pin wire layout and EDID parsing that have no hardware oracle.
#[kunit_tests(vino_protocol)]
mod tests {
    use super::*;
    use kernel::error::code::EINVAL;

    #[test]
    fn aes128_ecb_fips197_kat() -> Result {
        // FIPS-197 / NIST SP800-38A F.1.1 AES-128 ECB known-answer vector.
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let pt = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
            0x17, 0x2a,
        ];
        assert_eq!(
            crypto::aes128_ecb(&key, &pt)?,
            [
                0x3a, 0xd7, 0x7b, 0xb4, 0x0d, 0x7a, 0x36, 0x60, 0xa8, 0x9e, 0xca, 0xf3, 0x24, 0x66,
                0xef, 0x97,
            ]
        );
        Ok(())
    }

    #[test]
    fn colour_frame_ep08_damage_selects_changed_strips() -> Result {
        // Deterministic gradient source (a plain fn item so it's Copy/reusable across calls).
        fn g(x: usize, y: usize) -> (u8, u8, u8) {
            (
                ((x * 7) & 0xff) as u8,
                ((y * 5) & 0xff) as u8,
                (((x + y) * 3) & 0xff) as u8,
            )
        }
        let total = |fs: &KVec<KVec<u8>>| fs.iter().map(|f| f.len()).sum::<usize>();
        let flat = |fs: &KVec<KVec<u8>>| -> Result<KVec<u8>> {
            let mut v = KVec::new();
            for f in fs.iter() {
                v.extend_from_slice(f, GFP_KERNEL)?;
            }
            Ok(v)
        };
        // Damage granularity is the 256x64 MACRO-TILE (`MACRO_W`/`MACRO_H`), not the 64x16 strip:
        // DLM re-sends every strip of a touched macro-tile, and vino matches it (2026-07-25, see
        // `docs/DLM-DAMAGE-TILING.md`). The surface must therefore be several macro-tiles across or
        // *every* clip selects the whole frame and the "smaller than full" assertions below cannot
        // hold. The original 128x32 was smaller than a single macro-tile, which is exactly why this
        // test began failing when the tiling landed.
        //
        // 512x128 = 8 strips wide (512/64) x 8 bands (128/16) = 64 strips
        //         = 2 x 2 macro-tiles, each 4 strips wide x 4 bands = 16 strips.
        let (w, h) = (512usize, 128usize);
        const STRIPS_PER_MACRO: usize = 16;
        let (full, _) = video::wht::colour_frame_ep08(w, h, 0, 0, true, false, g)?;

        // A damage clip covering the WHOLE surface selects every strip in the same raster order as
        // the full-frame path, so the wire bytes are identical.
        let (dfull, _) = video::wht::colour_frame_ep08_damage(w, h, 0, 0, &[(0, 0, w, h)], true, false, g)?;
        assert_eq!(flat(&full)?.as_slice(), flat(&dfull)?.as_slice());

        // No damage -> no strips -> empty frame list (caller must skip the USB write).
        let (empty, _) = video::wht::colour_frame_ep08_damage(w, h, 0, 0, &[], true, false, g)?;
        assert!(empty.is_empty());

        // Selection is exact and macro-tile-quantised. Assert the strip COUNT directly (the shared
        // selector both encoders use) as well as the byte totals -- a count is a far sharper
        // statement than "smaller than full", and it is what actually pins the tiling behaviour.
        let coords = |clips: &[(usize, usize, usize, usize)]| -> Result<usize> {
            Ok(video::wht::damage_strip_coords(w, h, clips)?.len())
        };
        assert_eq!(coords(&[])?, 0);
        assert_eq!(coords(&[(0, 0, w, h)])?, 4 * STRIPS_PER_MACRO); // all four macro-tiles

        // A 1-pixel clip lands in ONE macro-tile and selects all 16 of its strips -- not 1.
        assert_eq!(coords(&[(1, 1, 2, 2)])?, STRIPS_PER_MACRO);
        let (d1, _) = video::wht::colour_frame_ep08_damage(w, h, 0, 0, &[(1, 1, 2, 2)], true, false, g)?;
        assert!(!d1.is_empty());
        assert!(total(&d1) < total(&full));

        // A 1-pixel-wide clip down the whole left edge spans the left macro-tile COLUMN: 2 tiles.
        assert_eq!(coords(&[(0, 0, 1, h)])?, 2 * STRIPS_PER_MACRO);
        let (d2, _) = video::wht::colour_frame_ep08_damage(w, h, 0, 0, &[(0, 0, 1, h)], true, false, g)?;
        assert!(total(&d1) < total(&d2) && total(&d2) < total(&full));

        // Non-aligned geometry is rejected (same contract as colour_frame_ep08).
        assert!(video::wht::colour_frame_ep08_damage(100, 32, 0, 0, &[(0, 0, 1, 1)], true, false, g).is_err());
        Ok(())
    }

    #[test]
    fn black_training_frame_matches_captured_1440p_size() -> Result {
        // Corrected Vino captures repeatedly measured a 205,696-byte first write:
        // 2,560-byte ARM + 203,040-byte black image + 96-byte frame trailer.
        let frame = video::wht::black_frame_ep08(2560, 1440, 0, true, false)?;
        let image_len = frame.iter().map(|part| part.len()).sum::<usize>();
        assert_eq!(image_len, 203_040);
        assert_eq!(2_560 + image_len + video::wht::frame_trailer(0, 0).len(), 205_696);
        Ok(())
    }

    #[test]
    fn aes_cmac_rfc4493_kat() -> Result {
        // RFC 4493 sec 4 AES-CMAC test vectors (same key as above).
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        assert_eq!(
            crypto::aes_cmac(&key, &[]),
            [
                0xbb, 0x1d, 0x69, 0x29, 0xe9, 0x59, 0x37, 0x28, 0x7f, 0xa3, 0x7d, 0x12, 0x9b, 0x75,
                0x67, 0x46,
            ]
        );
        let msg = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
            0x17, 0x2a,
        ];
        assert_eq!(
            crypto::aes_cmac(&key, &msg),
            [
                0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44, 0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a,
                0x28, 0x7c,
            ]
        );
        Ok(())
    }

    #[test]
    fn seal_livemac_roundtrip() -> Result {
        // A sealed CP frame must decrypt back to its content under the IN riv, and its
        // appended tag must equal a fresh Dl3Cmac over the ciphertext (encrypt-then-MAC).
        let ks = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let riv = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
        let content = [0xa5u8; 32];
        let mut hdr = [0u8; 16];
        hdr[12..16].copy_from_slice(&4u32.to_le_bytes()); // wire_seq = 4
        let frame = cp::seal_livemac(&ks, &riv, &hdr, &content)?;
        assert_eq!(frame.len(), 16 + 32 + 16);
        let ct = &frame[16..16 + 32];
        // Decrypt with the SAME riv it was sealed under. `open_in` is plain AES-CTR over whatever
        // nonce it is handed; its name describes its usual caller (dock->host), not a transform it
        // applies. This assertion previously passed `in_riv(&riv)` and so could never hold: the IN
        // nonce differs from the OUT nonce by `byte7 ^= 0x01` *by design*, which is precisely the
        // proven CP contract -- the two directions do not share a keystream.
        assert_eq!(&cp::open_in(&ks, &riv, 4, ct)?[..], &content[..]);
        // And pin that contract rather than leaving it implicit: the IN nonce really is different,
        // so opening with it must NOT recover the plaintext.
        assert_ne!(cp::in_riv(&riv), riv);
        assert_ne!(
            &cp::open_in(&ks, &cp::in_riv(&riv), 4, ct)?[..],
            &content[..]
        );
        assert_eq!(&frame[16 + 32..], &cp::dl3cmac_tag(&ks, &riv, 4, ct)?[..]);
        Ok(())
    }

    #[test]
    fn video_content_nonce_matches_same_session_dlm_rr_vectors() {
        // `scripts/rr-correlate-video-key.gdb` captured each clear id=0x32 RIV and the AES counter
        // actually consumed by the corresponding video seal channel in one deterministic replay.
        let h0 = cp::video_content_nonce(&[0xa1, 0x2b, 0xaa, 0xb7, 0x0e, 0x0b, 0x02, 0x74], 0);
        assert_eq!(h0, [0xa1, 0x2b, 0xaa, 0xb7, 0x0e, 0x0b, 0x02, 0x7c]);

        let h1 = cp::video_content_nonce(&[0xd0, 0x2a, 0xc0, 0x83, 0xb6, 0x42, 0x72, 0x57], 1);
        assert_eq!(h1, [0xd0, 0x2a, 0xc0, 0x83, 0xb6, 0x42, 0x72, 0x5e]);
    }

    #[test]
    fn aux_for_id_constants() {
        // The CP header `aux` field is a per-inner-id constant, not body_len/4.
        assert_eq!(cp::aux_for_id(0x14, 48), 0x0a);
        assert_eq!(cp::aux_for_id(0x15, 32), 0x09);
        assert_eq!(cp::aux_for_id(0x36, 80), 0x08);
        assert_eq!(cp::aux_for_id(0x48, 96), 0x06);
        // Cursor ids recovered from the cold-ref differential (scripts/verify-cp-seal.py):
        // body_len/4 (here 0x0c) would be wrong for all three.
        assert_eq!(cp::aux_for_id(0x1a, 48), 0x04); // cursor move
        assert_eq!(cp::aux_for_id(0x1b, 48), 0x03); // cursor create
        assert_eq!(cp::aux_for_id(0x1c, 48), 0x02); // cursor image
        assert_eq!(cp::aux_for_id(0x99, 40), 10); // unknown id falls back to body_len/4
    }

    #[test]
    fn cp_setup_burst_table_framing() -> Result {
        // Every entry in the post-msg0 CP setup burst tables (`cp::CP_SETUP_PER_HEAD`,
        // `cp::CP_SETUP_FINALIZE`) must produce a wire frame whose (aux, body_len) matches the
        // exact fingerprint decoded from real DLM 3.4.26 sessions. `body_len` is the wire size
        // minus the 16-byte cleartext header (AES-CTR ciphertext + 16-byte Dl3Cmac tag).
        //
        // **2026-07-17: corrected -- see `cp::CP_SETUP_PER_HEAD`'s doc comment.** Every
        // `content_len` is 16 bytes more than the original table had, re-measured directly from
        // raw wire ciphertext lengths (no decryption ambiguity) across TWO independent real
        // sessions.
        const PER_HEAD_FINGERPRINT: [(u16, usize); 9] = [
            (0x0c, 64),
            (0x0f, 64),
            (0x04, 176),
            (0x0c, 64),
            (0x0c, 80),
            (0x04, 64),
            (0x08, 64),
            (0x0a, 48),
            (0x05, 48),
        ];
        // body_len = 32B content + 16B Dl3Cmac = 48 (the content is 32B, corrected 2026-07-17
        // from the earlier 16B; see `cp::CP_SETUP_FINALIZE`).
        // One entry per `cp::CP_SETUP_FINALIZE` entry. The sixth (`id=0x16 sub=0x4c off22=1`,
        // added 2026-07-19) was never added here, so this indexed `[i]` panicked with
        // "index out of bounds: the len is 5 but the index is 5" on **every module load** --
        // KUnit runs at init, so each load left the kernel tainted `D` with an Oops. Recovered
        // from the pstore record of the 2026-07-22 crash. `id=0x16` frames are `(0x08, 48)`,
        // `id=0x15` frames `(0x09, 48)`; the sixth is another `id=0x16`.
        const FINALIZE_FINGERPRINT: [(u16, usize); 6] = [
            (0x08, 48),
            (0x09, 48),
            (0x08, 48),
            (0x08, 48),
            (0x09, 48),
            (0x08, 48),
        ];
        // Keep the fingerprint table and the burst table in lockstep: growing one without the
        // other is exactly the defect above.
        build_assert!(FINALIZE_FINGERPRINT.len() == cp::CP_SETUP_FINALIZE.len());

        let ks = [0x5au8; 16];
        let riv = [0x11u8; 8];
        for (i, &(id, _sub, content_len)) in cp::CP_SETUP_PER_HEAD.iter().enumerate() {
            let content = KVec::from_elem(0u8, content_len, GFP_KERNEL)?;
            let frame = cp::seal_interactive(&ks, &riv, id, 0, &content)?;
            let (want_aux, want_body) = PER_HEAD_FINGERPRINT[i];
            assert_eq!(frame.len(), 16 + want_body);
            assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), want_aux);
        }
        for (i, &(id, _sub, _off22)) in cp::CP_SETUP_FINALIZE.iter().enumerate() {
            let frame = cp::seal_interactive(&ks, &riv, id, 0, &[0u8; 32])?;
            let (want_aux, want_body) = FINALIZE_FINGERPRINT[i];
            assert_eq!(frame.len(), 16 + want_body);
            assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), want_aux);
        }
        Ok(())
    }

    #[test]
    fn stream_manage_restatement_matches_dlm() -> Result {
        // `cp::stream_manage_restatement`'s deterministic fields (everything except the trailing
        // 8-byte random tail) must match real DLM byte-for-byte for both heads. **2026-07-17:
        // corrected against a FULL AES-CTR decryption of trace1's encrypted burst
        // (`docs/CP-PERHEAD-RESTATEMENT.md`).** The head marker sits at off23 (not the off20 u32
        // the old truncated-decode guess used) and the 0x10 HDCP msg-id at off27 (not off24); the
        // `0 / 1 / (head+8)` u32 fields at off28/32/36 were already right.
        const WANT: [[u8; 40]; 2] = [
            [
                0x26, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x08, 0x00, 0x00, 0x00,
            ],
            [
                0x26, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00,
                0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x09, 0x00, 0x00, 0x00,
            ],
        ];
        for head in 0..2u8 {
            let c = cp::stream_manage_restatement(0, head)?;
            assert_eq!(c.len(), 48);
            // Bytes 4..6 are the live counter (passed as 0 here, so already covered); the last
            // 8 bytes (offset 40..48) are host-random.
            assert_eq!(&c[..40], &WANT[head as usize][..]);
        }
        Ok(())
    }

    #[test]
    fn video_arm_burst_table_framing() -> Result {
        // `cp::VIDEO_ARM_BURST`'s wire framing (type/sub/aux/body_len per entry) must match the
        // literal fingerprint decoded from the SAME rr recording used for
        // `cp_setup_burst_table_framing`, this time re-mined for the video endpoints (EP08/EP0b)
        // -- see `project_video_endpoint_arm_burst_found_20260716` memory + the raw hex in
        // `/tmp/claude-.../scratchpad/decode_video_ep.py` for the derivation. Head 0's sub values
        // (table entries) are asserted directly; head 1's (+1 per entry) are asserted via
        // `video_arm_plain_frame`/`video_arm_plaintext_body`'s `h` parameter.
        // **Corrected 2026-07-23** to the current HW-verified `cp::VIDEO_ARM_BURST`. This fingerprint
        // was the pre-2026-07-19 table copy and drifted when the ARM burst was re-RE'd: #2/#3 became
        // 16-byte sealed (not 32), #6/#7 became type-4 fixed-plaintext `(4, sub, 0x0004, 16)` (not
        // type-2 `0x20/0x30`), and #8/#9 became the 1104-byte DRBG-derived sealed bodies. Boot KUnit
        // logged `ASSERTION FAILED at vino.rs:...` (soft fail, kernel tainted) every load until this
        // caught up. `build_assert!` below ties the two table lengths together so they cannot drift
        // silently again.
        const FINGERPRINT_H0: [(u32, u16, u16, usize); 10] = [
            (2, 0x0008, 0x0000, 16),
            (2, 0x0018, 0x0000, 16),
            (4, 0x0008, 0x000a, 16),
            (4, 0x0018, 0x000a, 16),
            (2, 0x0000, 0x0000, 16),
            (2, 0x0010, 0x0000, 16),
            (4, 0x0000, 0x0004, 16),
            (4, 0x0010, 0x0004, 16),
            (4, 0x0008, 0x000e, 1104),
            (4, 0x0018, 0x000e, 1104),
        ];
        build_assert!(FINGERPRINT_H0.len() == cp::VIDEO_ARM_BURST.len());
        let ks = [0x5au8; 16];
        let riv = [0x11u8; 8];
        for (i, &(wire_type, sub_base, aux, body_len)) in cp::VIDEO_ARM_BURST.iter().enumerate() {
            let (want_type, want_sub, want_aux, want_body) = FINGERPRINT_H0[i];
            assert_eq!(
                (wire_type, sub_base, aux, body_len),
                (want_type, want_sub, want_aux, want_body)
            );
            if wire_type == 2 {
                let body = cp::video_arm_plaintext_body(i, 0);
                let frame = cp::video_arm_plain_frame(sub_base, &body);
                assert_eq!(frame.len(), 32);
                assert_eq!(u16::from_le_bytes([frame[8], frame[9]]), want_sub);
            } else {
                let content = KVec::from_elem(0u8, body_len, GFP_KERNEL)?;
                let frame = cp::seal_video_arm(&ks, &riv, sub_base, aux, 0, &content)?;
                assert_eq!(frame.len(), 16 + body_len + 16);
                assert_eq!(u16::from_le_bytes([frame[8], frame[9]]), want_sub);
                assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), want_aux);
            }
        }
        Ok(())
    }

    #[test]
    fn edid_timing_parse_and_validate() {
        // A well-formed EDID yields the DTD timing; a bad checksum is rejected; a leading
        // monitor descriptor (pclk 0) does not hide the preferred timing in a later slot.
        let edid = mk_test_edid(54);
        let t = cp::timing_from_edid(&edid).expect("valid EDID parses");
        assert_eq!(t.hactive, 1920);
        assert_eq!(t.vactive, 1080);
        assert_eq!(t.refresh_hz, 60);
        assert_eq!(t.pixel_clock_10khz, 14850);

        let mut bad = edid;
        bad[127] ^= 0xff;
        // bad checksum must be rejected
        assert!(cp::timing_from_edid(&bad).is_none());

        let scanned = mk_test_edid(72); // off54 left as a zero (monitor) descriptor
        assert_eq!(
            cp::timing_from_edid(&scanned)
                .expect("scans past off54")
                .hactive,
            1920
        );
    }

    #[test]
    fn edid_reply_guards() -> Result {
        // The pre-decrypt guards reject non-EDID frames without touching the cipher.
        let ks = [0u8; 16];
        let riv = [0u8; 8];
        assert!(cp::parse_edid_from_reply(&ks, &riv, &[0u8; 10])?.is_none());
        let mut wrong_sub = [0u8; 20];
        wrong_sub[8] = 0x44; // wire sub != 0x45
        assert!(cp::parse_edid_from_reply(&ks, &riv, &wrong_sub)?.is_none());
        Ok(())
    }

    #[test]
    fn get_edid_req_matches_dlm_wire_shape() -> Result {
        // Ground-truthed 2026-07-16 against a real DLM session (captures/rr-out-sequence-
        // 20260716/full-session-trace1/): the request is 32 bytes -- the standard 8-byte header,
        // 14 zero bytes, then a 10-byte random tail at offset 22 -- NOT the bare 16-byte header
        // this used to emit (the likely reason vino's own attempts never got a real EDID back).
        let req = cp::get_edid_req(0x2c, 0)?;
        assert_eq!(req.len(), 32);
        assert_eq!(
            &req[0..8],
            &[0x15, 0x00, 0x21, 0x00, 0x2c, 0x00, 0x00, 0x00]
        );
        assert_eq!(&req[8..22], &[0u8; 14]);
        // The wire framing this feeds into must match DLM's real request fingerprint too:
        // aux=0x09 (cp::aux_for_id(0x15, ..)), body = 32 + 16 (tag) = 48 bytes.
        let frame = cp::seal_interactive(&[0x5au8; 16], &[0x11u8; 8], 0x15, 0, &req)?;
        assert_eq!(frame.len(), 16 + 32 + 16);
        assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), 0x09);
        Ok(())
    }

    #[test]
    fn edid_engage_req_matches_dlm_wire_shape() -> Result {
        // Ground-truthed 2026-07-17 (2nd pass), cross-validated against TWO independent DLM
        // sessions (full-session-trace1 AND dlm-cold-3426-20260714-140216-allbus): `id=0x16
        // sub=0x0023`, same 32-byte shape as `get_edid_req` (8B header + 14 zero + 10B random
        // tail). Sent twice, right after the second placeholder EDID fetch and right before the
        // long device-status poll that precedes the dock's real EDID becoming available.
        let req = cp::edid_engage_req(0x30, 0)?;
        assert_eq!(req.len(), 32);
        assert_eq!(
            &req[0..8],
            &[0x16, 0x00, 0x23, 0x00, 0x30, 0x00, 0x00, 0x00]
        );
        assert_eq!(&req[8..22], &[0u8; 14]);
        let frame = cp::seal_interactive(&[0x5au8; 16], &[0x11u8; 8], 0x16, 0, &req)?;
        assert_eq!(frame.len(), 16 + 32 + 16);
        assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), 0x08); // cp::aux_for_id(0x16, ..)
        Ok(())
    }

    #[test]
    fn edid_poll_ready_byte_exact_vs_dlm_capture() -> Result {
        // Two real dock->host `id=0x0044 sub=0x0020` EDID-readiness probe replies, taken
        // verbatim (wire seq 205 and 857) from
        // captures/rr-out-sequence-20260716/full-session-trace1/raw-out-in-192msg.txt: the
        // first is DLM's probe right before its FIRST placeholder `id=0x114` fetch (ictr 43 ->
        // 44), the second is the probe right before its FIRST real `id=0x194` fetch (ictr 119
        // -> 120). Ground-truths `cp::edid_poll_ready`'s inner-offset-26 readiness bit against
        // actual dock traffic, not a synthetic fixture.
        const KS: [u8; 16] = [
            0xd9, 0xec, 0x1f, 0xbc, 0x8b, 0x5a, 0xb3, 0xd8, 0x71, 0x0f, 0xd3, 0xbd, 0x42, 0x04,
            0x06, 0x55,
        ];
        const OUT_RIV: [u8; 8] = [0xf6, 0x21, 0xdc, 0x0d, 0x22, 0x7e, 0xf4, 0xaf];
        #[rustfmt::skip]
        const NOT_READY: [u8; 112] = [
            0x00, 0x00, 0x6c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x45, 0x00, 0x0a, 0x00, 0xcd, 0x00,
            0x00, 0x00, 0xa5, 0xea, 0x5d, 0x51, 0xf6, 0xa8, 0x6b, 0xb6, 0x89, 0x88, 0x01, 0xa2,
            0x47, 0x30, 0xbd, 0x6c, 0x84, 0xb8, 0xaf, 0x9f, 0x85, 0xf2, 0x8a, 0x20, 0xc8, 0xec,
            0x51, 0x9e, 0x8d, 0xeb, 0xef, 0x5a, 0x3a, 0x1d, 0xb5, 0xc7, 0x80, 0x02, 0xfe, 0x1e,
            0xed, 0x07, 0xdd, 0x71, 0x00, 0x7f, 0x45, 0x77, 0x6c, 0x82, 0xf6, 0xe9, 0xc3, 0x0d,
            0xdf, 0x67, 0x82, 0xac, 0xa8, 0x23, 0xd5, 0x5a, 0x1c, 0xce, 0xcb, 0x89, 0xb5, 0x98,
            0x65, 0xba, 0xbb, 0xb6, 0x2d, 0x0e, 0x9b, 0x55, 0xee, 0xfd, 0x46, 0x0c, 0x22, 0x35,
            0x6f, 0x84, 0xe5, 0x36, 0x95, 0xd0, 0xdc, 0xfc, 0x6f, 0x8a, 0x57, 0xda, 0xa2, 0xae,
        ];
        #[rustfmt::skip]
        const READY: [u8; 112] = [
            0x00, 0x00, 0x6c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x45, 0x00, 0x0a, 0x00, 0x59, 0x03,
            0x00, 0x00, 0xf7, 0x8e, 0x70, 0xb2, 0xa3, 0x24, 0xe2, 0x6f, 0x9f, 0xb6, 0xe9, 0x8e,
            0x32, 0x55, 0x11, 0x21, 0x99, 0x74, 0xf6, 0xfb, 0xea, 0x97, 0xd5, 0x7f, 0xa6, 0x45,
            0x9d, 0x35, 0xf0, 0xa7, 0xbe, 0xd3, 0x9b, 0x19, 0x24, 0x8c, 0x98, 0xa6, 0x0c, 0xa2,
            0x4d, 0x8e, 0x83, 0xaa, 0x74, 0xd5, 0x8b, 0xe0, 0x6f, 0xb1, 0x9f, 0xa4, 0xb9, 0xae,
            0x39, 0xc6, 0x0a, 0x9c, 0x63, 0x70, 0xdb, 0x49, 0x74, 0xe5, 0x85, 0x42, 0x07, 0x7e,
            0xc2, 0x49, 0xfb, 0x67, 0x54, 0xd5, 0x47, 0x72, 0xb7, 0x19, 0x24, 0x8f, 0xb1, 0xb0,
            0xb2, 0x83, 0x89, 0x62, 0x4b, 0xcb, 0x59, 0x15, 0x1f, 0x8f, 0x85, 0xc3, 0xa5, 0x9d,
        ];
        assert_eq!(cp::edid_poll_ready(&KS, &OUT_RIV, &NOT_READY), Some(false));
        assert_eq!(cp::edid_poll_ready(&KS, &OUT_RIV, &READY), Some(true));
        Ok(())
    }

    #[test]
    fn rgb565_packing() {
        assert_eq!(video::rgb565(0xff, 0x00, 0x00), 0xf800);
        assert_eq!(video::rgb565(0x00, 0xff, 0x00), 0x07e0);
        assert_eq!(video::rgb565(0x00, 0x00, 0xff), 0x001f);
        let _ = EINVAL; // silence unused import on configs without the assert paths
    }

    #[test]
    fn cursor_messages_structure() -> Result {
        // The shared 32-byte cursor layout (recovered byte-exact from the cold-ref session by
        // scripts/verify-cp-seal.py): marker 0x02 @22, head_id @23, two LE u16 fields @24/@26.
        // Create (head 0): id=0x1b sub=0x42, fields = w,h.
        let c = cp::cursor_create(7, 0, 64, 64)?;
        assert_eq!(c.len(), 32);
        assert_eq!(&c[0..6], &[0x1b, 0x00, 0x42, 0x00, 0x07, 0x00]); // id, sub, counter (LE)
        assert_eq!(c[22], 0x02); // marker
        assert_eq!(c[23], 0); // head id
        assert_eq!(u16::from_le_bytes([c[24], c[25]]), 64); // width
        assert_eq!(u16::from_le_bytes([c[26], c[27]]), 64); // height

        // Move (head 1): id=0x1a sub=0x43, marker@22, head@23, X@24, Y@26 (LE).
        let m = cp::cursor_move(9, 1, 0x0140, 0x00f0)?;
        assert_eq!(m.len(), 32);
        assert_eq!(&m[0..4], &[0x1a, 0x00, 0x43, 0x00]); // id, sub
        assert_eq!(m[22], 0x02); // marker
        assert_eq!(m[23], 1); // head id
        assert_eq!(u16::from_le_bytes([m[24], m[25]]), 0x0140); // X
        assert_eq!(u16::from_le_bytes([m[26], m[27]]), 0x00f0); // Y

        // Image: 32-byte header (inner id 0x401c, the 0x40 bitmap flag) + w*h*4 BGRA at off32;
        // wrong-size input rejected.
        let bitmap = KVec::from_elem(0xabu8, 64 * 64 * 4, GFP_KERNEL)?;
        let img = cp::cursor_image(3, 0, 64, 64, &bitmap)?;
        assert_eq!(img.len(), 32 + 64 * 64 * 4);
        assert_eq!(&img[0..4], &[0x1c, 0x40, 0x41, 0x00]); // inner id 0x401c, sub 0x41
        assert_eq!(img[22], 0x02); // marker
        assert_eq!(img[32], 0xab); // bitmap begins at off32
        assert!(cp::cursor_image(3, 0, 64, 64, &[0u8; 16]).is_err()); // wrong bitmap length
        Ok(())
    }

    #[test]
    fn timing_from_drm_mode_1080p60() {
        // CEA 1920x1080@60: clock 148.5 MHz, h 2008/2052/2200, v 1084/1089/1125.
        let mut m = bindings::drm_display_mode::default();
        m.clock = 148_500; // kHz
        m.hdisplay = 1920;
        m.hsync_start = 2008;
        m.hsync_end = 2052;
        m.htotal = 2200;
        m.vdisplay = 1080;
        m.vsync_start = 1084;
        m.vsync_end = 1089;
        m.vtotal = 1125;
        // SAFETY: `m` is a fully-initialised local `drm_display_mode`, and `DisplayMode` is a
        // `#[repr(transparent)]` wrapper over it, so the cast only changes the Rust type.
        let t = unsafe {
            cp::timing_from_drm_mode(
                &*(&raw const m).cast::<kernel::drm::kms::modes::DisplayMode>(),
            )
        };
        assert_eq!(t.hactive, 1920);
        assert_eq!(t.hblank, 280); // htotal - hdisplay
        assert_eq!(t.hsync_front, 88); // hsync_start - hdisplay
        assert_eq!(t.hsync_width, 44); // hsync_end - hsync_start
        assert_eq!(t.vactive, 1080);
        assert_eq!(t.vblank, 45); // vtotal - vdisplay
        assert_eq!(t.vsync_front, 4);
        assert_eq!(t.vsync_width, 5);
        assert_eq!(t.pixel_clock_10khz, 14_850); // clock(kHz) / 10
        assert_eq!(t.refresh_hz, 60); // via drm_mode_vrefresh
    }

    /// Regression test for the silent pixel-clock clamp (2026-07-26). `timing_from_drm_mode` used to
    /// `clamp()` the 10-kHz clock to `u16::MAX`, so any mode above 655.35 MHz was sent to the dock
    /// with a WRONG clock and nothing logged it -- this monitor's 2560x1440@165 (699.5 MHz) would
    /// have gone out as 655.35 MHz. The clock is now 24-bit: low u16 at off70, high byte at off72.
    ///
    /// Both halves are asserted, and the 1440p120 case pins the part that is byte-exact against the
    /// captures: below 655.35 MHz off72 must stay zero, so widening the field cannot have perturbed
    /// any mode vino already drove correctly.
    #[test]
    fn set_mode_carries_a_24bit_pixel_clock() -> Result {
        let mut m = bindings::drm_display_mode::default();
        // 2560x1440@165 from the MSI MAG 27CQ6F DisplayID block: 699.5 MHz, htotal 2720.
        m.clock = 699_500; // kHz
        m.hdisplay = 2560;
        m.hsync_start = 2608;
        m.hsync_end = 2640;
        m.htotal = 2720;
        m.vdisplay = 1440;
        m.vsync_start = 1443;
        m.vsync_end = 1451;
        m.vtotal = 1559;
        // SAFETY: as in `timing_from_drm_mode_1080p60` -- `m` is a fully-initialised local
        // `drm_display_mode` and `DisplayMode` is a `#[repr(transparent)]` wrapper over it.
        let t = unsafe {
            cp::timing_from_drm_mode(
                &*(&raw const m).cast::<kernel::drm::kms::modes::DisplayMode>(),
            )
        };
        // 699_500 kHz / 10 = 69_950 -- above u16::MAX (65_535), which is what used to be lost.
        assert_eq!(t.pixel_clock_10khz, 69_950);
        assert_eq!(t.refresh_hz, 165);
        let w = cp::set_mode(1, 0, &t)?;
        assert_eq!(w.len(), 80);
        assert_eq!(u16::from_le_bytes([w[70], w[71]]), 69_950u32 as u16); // low 16 bits
        assert_eq!(w[72], (69_950u32 >> 16) as u8); // high byte = 1
        assert_eq!(w[73], 0); // off73 stays zero

        // 2560x1440@120 (497.75 MHz): the mode the captures cover. off72 must be zero.
        m.clock = 497_750;
        m.vtotal = 1524;
        // SAFETY: as above.
        let t120 = unsafe {
            cp::timing_from_drm_mode(
                &*(&raw const m).cast::<kernel::drm::kms::modes::DisplayMode>(),
            )
        };
        assert_eq!(t120.pixel_clock_10khz, 49_775);
        let w120 = cp::set_mode(1, 0, &t120)?;
        assert_eq!(u16::from_le_bytes([w120[70], w120[71]]), 49_775);
        assert_eq!(w120[72], 0);
        Ok(())
    }

    /// The dock's refresh ceiling prunes exactly the rates no stack has ever displayed.
    ///
    /// The monitor's EDID offers 2560x1440 at 120, 165 and 180 Hz. DLM takes the 180 Hz mode from
    /// the compositor and puts **119.998 Hz** on the wire (`captures/dlm-180hz-cp-20260726-2300`);
    /// vino sent the real 165 and 180 Hz timings and both panels went dark with the dock acking
    /// every byte (`docs/HIGH-REFRESH.md`). 120 Hz is the last rate this hardware is known to
    /// display, so the mode list stops there.
    ///
    /// The 120 Hz case is the one that matters most: it sits on the boundary and must pass by
    /// **equality**. A `<` here would prune the working configuration and dark both panels.
    #[test]
    fn refresh_ceiling_prunes_only_the_unproven_rates() {
        assert_eq!(drm_sink::max_refresh_hz(), 120); // no override in force under KUnit
        let ok = drm_sink::refresh_within_limit;

        // Allowed: the proven configuration, on the boundary, and everything under it. 164.96 Hz
        // and 119.998 Hz both arrive here already rounded by `drm_mode_vrefresh`.
        assert!(ok(120));
        assert!(ok(60));
        assert!(ok(119));
        // Pruned: the two rates that were tried on hardware and left the panel dark.
        assert!(!ok(165));
        assert!(!ok(180));
        // A degenerate mode reports 0 Hz; that carries no rate information, so it is not this
        // check's business to reject it (`MAX_HEAD_CLOCK_KHZ` and the dock budget still bound it),
        // and a signed refresh must never be read as a huge unsigned one.
        assert!(ok(0));
        assert!(ok(-1));

        // The ceiling is refresh, deliberately NOT pixel rate: 3840x2160@60 is 497,664,000 px/s --
        // above DLM's declared per-connector 442,368,000 -- and this dock is documented to drive it,
        // so a rate cap set at DLM's figure would prune a mode that works. The active-rate helper
        // still backs the dock-wide budget checks, and must saturate rather than wrap.
        let rate = drm_sink::active_pixel_rate;
        assert!(ok(60) && rate(3840, 2160, 60) == 497_664_000);
        assert_eq!(rate(2560, 1440, 120), 442_368_000); // DLM's declared limit, exactly
        assert_eq!(rate(65535, 65535, 65535), u32::MAX); // saturates, never wraps small
        assert_eq!(rate(2560, 1440, -1), 0);
    }

    /// The whole set-mode message vino builds, byte-checked against every **DLM** `id=0x48 sub=0x22`
    /// plaintext in the corpus (`cp::mode_words` lists the captures).
    ///
    /// Two of the fields here were got wrong by reading only the 2026-07-26 23:00 capture, which
    /// happened to contain nothing but 120 Hz modes:
    ///
    /// * **off66 is not a function of resolution.** At 1080p it is `0x2810` at 60 Hz and `0x083f` at
    ///   120 Hz. vino sent the 1440p `0x0800` for every mode, so every 1080p mode-set carried a wrong
    ///   word -- and off66 is therefore **not** eliminated as a suspect for the dark panel above
    ///   120 Hz, contrary to that capture's summary.
    /// * **off44 is the refresh rate**, not the constant 120 the same summary recorded. The 1080p60
    ///   message pins it at 60. vino already wrote refresh; this stops anyone "fixing" it to 120.
    #[test]
    fn set_mode_matches_every_dlm_capture() -> Result {
        // (hact, htotal, hsync_start, hsync_end, vact, vtotal, vsync_start, vsync_end, clock kHz,
        //  refresh, off42, off66)
        let cases: [(u16, u16, u16, u16, u16, u16, u16, u16, i32, u16, u16, u16); 3] = [
            // max-cold-20260721-235609 ctr=244: 1920x1080@60, 148.50 MHz.
            (1920, 2200, 2008, 2052, 1080, 1125, 1084, 1089, 148_500, 60, 0x0400, 0x2810),
            // dlm-180hz-cp-20260726-2300 ctr=929 (head 1) and ctr=1072 (head 0): 1080p120, 297 MHz.
            (1920, 2200, 2008, 2052, 1080, 1125, 1084, 1089, 297_000, 120, 0x0400, 0x083f),
            // dlm-180hz ctr=999/1143 and max-cold ctr=318: 2560x1440@120, 497.75 MHz, vtotal 1525.
            (2560, 2720, 2608, 2640, 1440, 1525, 1443, 1448, 497_750, 120, 0x0600, 0x0800),
        ];
        for (hact, htotal, hss, hse, vact, vtotal, vss, vse, clock, refresh, off42, off66) in cases {
            let mut m = bindings::drm_display_mode::default();
            m.clock = clock;
            m.hdisplay = hact;
            m.hsync_start = hss;
            m.hsync_end = hse;
            m.htotal = htotal;
            m.vdisplay = vact;
            m.vsync_start = vss;
            m.vsync_end = vse;
            m.vtotal = vtotal;
            // SAFETY: as in `timing_from_drm_mode_1080p60` -- `m` is a fully-initialised local
            // `drm_display_mode` and `DisplayMode` is a `#[repr(transparent)]` wrapper over it.
            let t = unsafe {
                cp::timing_from_drm_mode(
                    &*(&raw const m).cast::<kernel::drm::kms::modes::DisplayMode>(),
                )
            };
            let w = cp::set_mode(7, 1, &t)?;
            assert_eq!(w.len(), 80);
            let u16_at = |off: usize| u16::from_le_bytes([w[off], w[off + 1]]);
            assert_eq!(u16_at(26), hact); // hactive
            assert_eq!(u16_at(28), htotal - hact); // hblank
            assert_eq!(u16_at(30), hss - hact); // hsync front porch
            assert_eq!(u16_at(32), hse - hss); // hsync width
            assert_eq!(u16_at(34), vact); // vactive
            assert_eq!(u16_at(36), vtotal - vact); // vblank
            assert_eq!(u16_at(38), vss - vact); // vsync front porch
            assert_eq!(u16_at(40), vse - vss); // vsync width
            assert_eq!(u16_at(42), off42); // the resolution-dependent word
            assert_eq!(u16_at(44), refresh); // off44 is REFRESH, not a constant 120
            assert_eq!(u16_at(66), off66); // the timing-dependent word
            assert_eq!(u16_at(68), 0x0200); // constant across all six DLM messages
            assert_eq!(u16_at(70), (clock as u32 / 10) as u16); // pixel clock / 10 kHz
            assert_eq!(w[72], 0); // no DLM capture ever needed the high byte
        }
        Ok(())
    }

    /// A mode with no measured mode-words falls back within its own resolution and reports that it
    /// is a guess. This is what keeps the pending 1920x1080@165 discriminator honest: the fallback
    /// picks 1080p's nearest measured refresh rather than a 1440p word, and `exact == false` makes
    /// `timing_from_drm_mode` log that off66 is unproven -- so a dark panel there cannot be read as
    /// "the dock caps refresh" without accounting for it.
    #[test]
    fn mode_words_fall_back_within_the_resolution() {
        assert_eq!(cp::mode_words(1920, 1080, 60), (0x0400, 0x2810, true));
        assert_eq!(cp::mode_words(2560, 1440, 120), (0x0600, 0x0800, true));
        // 165 Hz: same resolution, nearest measured refresh is 120 -- not the 1440p word.
        assert_eq!(cp::mode_words(1920, 1080, 165), (0x0400, 0x083f, false));
        // 1440p is sampled at one refresh only, so every other 1440p rate is that row, inexact.
        assert_eq!(cp::mode_words(2560, 1440, 60), (0x0600, 0x0800, false));
        // An unmeasured resolution keeps the pair vino has always sent.
        assert_eq!(cp::mode_words(3840, 2160, 60), (0x0600, 0x0800, false));
    }

    #[test]
    fn rotation_pixel_mapping() {
        use bindings::{
            DRM_MODE_REFLECT_X, DRM_MODE_ROTATE_0, DRM_MODE_ROTATE_180, DRM_MODE_ROTATE_270,
            DRM_MODE_ROTATE_90,
        };
        // Source 2x3 (sw=2, sh=3). 0deg is identity; 180deg mirrors both axes.
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_0, 0, 0, 2, 3), (0, 0));
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_0, 1, 2, 2, 3), (1, 2));
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_180, 0, 0, 2, 3), (1, 2));
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_180, 1, 2, 2, 3), (0, 0));
        // 90deg: output dims are (sh,sw)=(3,2); (dx,dy) -> (dy, sh-1-dx).
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_90, 0, 0, 2, 3), (0, 2));
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_90, 2, 1, 2, 3), (1, 0));
        // 270deg: (dx,dy) -> (sw-1-dy, dx).
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_270, 0, 0, 2, 3), (1, 0));
        assert_eq!(drm_sink::rot_src(DRM_MODE_ROTATE_270, 2, 1, 2, 3), (0, 2));
        // Reflect-X composes on top of the rotation (here identity): sx -> sw-1-sx.
        assert_eq!(
            drm_sink::rot_src(DRM_MODE_ROTATE_0 | DRM_MODE_REFLECT_X, 0, 0, 2, 3),
            (1, 0)
        );
    }

    #[test]
    fn wht_colour_and_quantize() {
        use video::wht;
        // Colour transform vs DLM's transform-DC ground truth (validate-transform-encoderio):
        // white -> Y=16320, achromatic -> Cb=Cr=0, and the reversible luma Y = 64G +
        // 64*((Cb+Cr)>>2) reproduces every measured colour. Red's Y is 4032 (= 64*((255)>>2)
        // = 64*63), NOT the old un-floored 16*255 = 4080 (which ran 48 high; see `colour`).
        assert_eq!(wht::colour(255, 255, 255), (16320, 0, 0));
        assert_eq!(wht::colour(128, 128, 128), (128 * 64, 0, 0)); // gray: chroma zero
        assert_eq!(wht::colour(255, 0, 0), (4032, 64 * 255, 0)); // red: Y floored, Cb>0, Cr=0
        assert_eq!(wht::colour(0, 255, 0), (8128, -64 * 255, -64 * 255)); // green (signed chroma)
        assert_eq!(wht::colour(0, 0, 255), (4032, 0, 64 * 255)); // blue
                                                                 // The documented ground-truth vector: white Y_DC=16320 quantizes (DC, position 0) to 1020.
        assert_eq!(wht::quantize(16320, 0), 1020);
        // AC clamps to the 12-bit signed long-token range.
        assert_eq!(wht::quantize(1_000_000, 16), 2047);
        assert_eq!(wht::quantize(-1_000_000, 16), -2048);
    }

    #[test]
    fn wht_transform_uniform() {
        use video::wht;
        // A uniform block: DC = the per-pixel value, every AC coefficient = 0 (VIDEO.md invariant).
        let block = [16320i32; wht::BLOCK];
        let c = wht::transform(&block);
        assert_eq!(c[0], 16320); // DC = mean = the uniform value
        assert!(c[1..].iter().all(|&x| x == 0)); // AC all zero
                                                 // End-to-end: white pixel -> Y plane -> WHT DC -> quantize -> 1020.
        let (y, _, _) = wht::colour(255, 255, 255);
        assert_eq!(wht::quantize(wht::transform(&[y; wht::BLOCK])[0], 0), 1020);
    }

    /// The quantisers divide by powers of two, and as of 2026-07-27 they do it with an arithmetic
    /// shift rather than `div_euclid`/`/` -- because `i` is a runtime loop variable, so the divisor
    /// was a runtime value and every one of the 64 coefficients per block per plane compiled to a
    /// real `idiv`. Measured offline, that was 163 ns of the 256 ns `transform+quant` block cost,
    /// more than the wavelet itself.
    ///
    /// The rewrite is only valid because floor division by `2^k` IS an arithmetic right shift, for
    /// negative operands as well as positive. That identity is easy to state and easy to get wrong
    /// (a *truncating* `/` is not the same thing on negatives), and a coefficient off by one is a
    /// wire-visible codec change. So assert it directly, over the full coefficient range the
    /// transform can produce, against the division the encoder used to perform.
    #[test]
    fn quantiser_shifts_match_division() {
        use video::wht::{quantize, COEFFS};
        // 8-bit input in the codec's x64 fixed point, summed over an 8x8 block and floor-divided by
        // 64 by `transform`, bounds |coeff| well inside this; step past it on both signs anyway.
        const LIMIT: i32 = 200_000;
        // The luma table, restated here as DIVISORS so the test is not written against the same
        // shift constants it is checking.
        let step_bias = |i: usize| -> (i32, i32) {
            match i {
                0 | 1 | 2 => (16, 8),
                3 => (32, 16),
                4..=11 => (4, 2),
                12..=15 => (8, 4),
                16..=47 => (2, 0),
                _ => (4, 2),
            }
        };
        for i in 0..COEFFS {
            let (step, bias) = step_bias(i);
            for coeff in (-LIMIT..=LIMIT).step_by(37) {
                let want = if bias == 0 {
                    let q = coeff.abs() / step;
                    if coeff < 0 {
                        -q
                    } else {
                        q
                    }
                } else {
                    (coeff + bias).div_euclid(step)
                }
                .clamp(-2048, 2047);
                assert_eq!(quantize(coeff, i), want);
            }
        }
        // Boundary cases the stride above can step over: every exact multiple and half-step of the
        // coarsest divisor, on both signs, is where floor-vs-truncate actually differs.
        for i in 0..COEFFS {
            let (step, bias) = step_bias(i);
            for m in -4i32..=4 {
                for d in [-1, 0, 1, step / 2, -step / 2] {
                    let coeff = m * step + d;
                    let want = if bias == 0 {
                        let q = coeff.abs() / step;
                        if coeff < 0 {
                            -q
                        } else {
                            q
                        }
                    } else {
                        (coeff + bias).div_euclid(step)
                    }
                    .clamp(-2048, 2047);
                    assert_eq!(quantize(coeff, i), want);
                }
            }
        }
    }

    #[test]
    fn wht_transform_haar_vectors() {
        // The 8x8 2-D Haar (Mallat) wavelet, byte-exact-verified against DLM (2026-06-23):
        // `scripts/wht-transform.py` reproduces these on 320/320 real gradient blocks. Each vector
        // here is computed by that reference; `Y = 64*gray`. (See the breakthrough writeup.)
        use video::wht::{transform, DIM, PIXELS};
        // Build an 8x8 Y block by evaluating a gray-per-(row,col) selector.
        fn build(gray: impl Fn(usize, usize) -> i32) -> [i32; PIXELS] {
            let mut b = [0i32; PIXELS];
            for r in 0..DIM {
                for c in 0..DIM {
                    b[r * DIM + c] = 64 * gray(r, c);
                }
            }
            b
        }
        // vstripe2 (period-2 vertical, full contrast 0/255) -> level-2 HL band c[4..8] = -2040.
        let c = transform(&build(|_, c| if (c / 2) & 1 != 0 { 255 } else { 0 }));
        assert_eq!(&c[4..8], &[-2040, -2040, -2040, -2040]);
        assert!(c[1..4].iter().all(|&x| x == 0) && c[8..].iter().all(|&x| x == 0));
        // vstripe4 (period-4 vertical) -> coarse HL c[1] = -8160 (4x the fine band, energy-conserving).
        let c = transform(&build(|_, c| if (c / 4) & 1 != 0 { 255 } else { 0 }));
        assert_eq!(c[1], -8160);
        // hstripe2 (period-2 horizontal) -> level-2 LH band c[8..12] = -2040 (H/V swap of vstripe2).
        let c = transform(&build(|r, _| if (r / 2) & 1 != 0 { 255 } else { 0 }));
        assert_eq!(&c[8..12], &[-2040, -2040, -2040, -2040]);
        // A per-column gradient (gray = 36*col) exercises the DC, coarse-HL and finest band at once.
        let c = transform(&build(|_, col| 36 * col as i32));
        assert_eq!(c[0], 8064); // DC = mean(36*0..36*7)*64/64 = 8064
        assert_eq!(&c[4..8], &[-576, -576, -576, -576]);
        // The level-1 tail is THREE bands, not one: c[16..32] = HL1, c[32..48] = LH1,
        // c[48..64] = HH1 (each 4x4, Morton-scanned). This assertion used to read `c[16..]` as a
        // single band, which was true only under the old 32-coefficient layout; since the full-64
        // codec landed (2026-07-22) it also swept up LH1/HH1.
        //
        // A per-COLUMN ramp is constant down every column, so it has no vertical detail at all:
        // HL1 is uniformly -72 and LH1/HH1 are identically zero. Asserting the zeros is strictly
        // stronger than the original -- it pins the band layout, not just one band's value.
        assert!(c[16..32].iter().all(|&x| x == -72)); // finest HL: horizontal detail only
        assert!(c[32..].iter().all(|&x| x == 0)); // LH1 + HH1: no vertical detail
    }

    #[test]
    fn wht_vlc_codebook_byte_exact() -> Result {
        // ★ The recovered LSB-first entropy VLC (dumped from DLM 0x5e68b0), verified byte-exact
        // against DLM's own captured output (scripts/wht-block-codec.py). Symbol 7 is the AC code
        // 0b1110000 (LSB-first); four of them pack to the wire's per-block AC unit bytes, and the
        // final byte is padded with 1-bits (a truncated all-ones code), exactly as the dock emits.
        use video::wht::Vlc;
        let mut w = Vlc::new();
        for _ in 0..4 {
            w.symbol(7)?;
        }
        assert_eq!(&w.finish()?[..], &[0x87, 0xc3, 0xe1, 0xf0]);
        // The full per-block AC unit `0 0 0 7 7 7 7` (idx1-3 zero, idx4-7 AC) -- matches the live
        // wire bytes `38 1c 0e ...` captured for vstripe2.
        let mut w = Vlc::new();
        for s in [0usize, 0, 0, 7, 7, 7, 7] {
            w.symbol(s)?;
        }
        assert_eq!(&w.finish()?[..4], &[0x38, 0x1c, 0x0e, 0x87]);
        // Symbol 0 (the 1-bit `0` code) alone -> one byte padded with seven 1-bits.
        let mut w = Vlc::new();
        w.symbol(0)?;
        assert_eq!(&w.finish()?[..], &[0xfe]);
        Ok(())
    }

    #[test]
    fn wht_coeff_magnitude_code() -> Result {
        // The AC magnitude-code emitter, verified byte-exact vs DLM's per-coefficient wire bits
        // (q-4/q-8/q-16; scripts/wht-strip-encoder.py reproduces the full q-4 vstripe2 strip).
        use video::wht::Vlc;
        // Four q-4 coefficients (category 3, zero offset) == four sym7 -- the per-block AC unit.
        let mut w = Vlc::new();
        for _ in 0..4 {
            w.coeff(-4)?;
        }
        assert_eq!(&w.finish()?[..], &[0x87, 0xc3, 0xe1, 0xf0]);
        // A zero coefficient is the 1-bit symbol 0 -> one byte padded with seven 1-bits.
        let mut w = Vlc::new();
        w.coeff(0)?;
        assert_eq!(&w.finish()?[..], &[0xfe]);
        // Within-category offset (q-6 = category 3, offset 2) and sign polarity (negative vs +).
        let mut w = Vlc::new();
        w.coeff(-6)?;
        assert_eq!(&w.finish()?[..], &[0x97]);
        let mut w = Vlc::new();
        w.coeff(6)?;
        assert_eq!(&w.finish()?[..], &[0xd7]); // same magnitude, sign bit flipped
                                               // Category 5 with offset (q-16) spans two bytes.
        let mut w = Vlc::new();
        w.coeff(-16)?;
        assert_eq!(&w.finish()?[..], &[0x1f, 0xf8]);
        // Category >= 9 (|q| >= 256) is the unrecovered escape long form -> rejected, not mis-coded.
        let mut w = Vlc::new();
        assert!(w.coeff(-256).is_err());
        Ok(())
    }

    // ── Removed 2026-07-27: the LEGACY LUMA ENCODERS and their tests.
    //
    // `solid_strip`, `strip`, `ac_strip` and `encode_frame` were the achromatic codec -- the first
    // half of the RE, cracked on greyscale content before colour was understood. `colour_strip`
    // superseded them completely: it codes uniform strips too, and the live scanout path called
    // none of the four. What remained was ~300 lines of dead code plus tests whose DLM-captured
    // ground truth no longer described any executing path.
    //
    // The shared machinery they exercised is still covered, and by better tests: `transform` by
    // `wht_transform_uniform`/`wht_transform_haar_vectors`, the escape coder by
    // `wht_vlc_codebook_byte_exact`/`wht_coeff_magnitude_code`, the significance extent by
    // `wht_chroma_last_is_exact`. Stronger than any of them, `scripts/codec-re/colour_decode.py`
    // now decodes the live grammar end to end and is validated against a real DLM capture
    // (3597/3600 strips) -- a round-trip check the byte-vector tests could never be.

    #[test]
    fn wht_magnitude_category() {
        // Magnitude category = bit_length(|coeff|), verified across the 2026-06-23 value sweep
        // (q-4 -> cat 3 -> sym7, q-8 -> 4, ... q-128 -> 8).
        use video::wht::mag_category;
        assert_eq!(mag_category(0), 0);
        assert_eq!(mag_category(1), 1);
        assert_eq!(mag_category(-4), 3);
        assert_eq!(mag_category(7), 3);
        assert_eq!(mag_category(-8), 4);
        assert_eq!(mag_category(16), 5);
        assert_eq!(mag_category(-128), 8);
        assert_eq!(mag_category(255), 8);
    }

    #[test]
    fn wht_chroma_last_is_exact() {
        use video::wht::{chroma_last, COEFFS};
        let mut q = [0i32; COEFFS];
        assert_eq!(chroma_last(&q), 0);
        for exact in [1usize, 2, 3, 4, 7, 8, 11, 15, 16, 27, 31, 32, 48, 62, 63] {
            q.fill(0);
            q[exact] = 1;
            assert_eq!(chroma_last(&q), exact);
        }
    }

    #[test]
    fn ddc_ci_set_vcp_checksum() {
        // VESA DDC/CI 1.1 sec 4.4 worked example: Set brightness (VCP 0x10) to 50 (0x0032).
        // Bytes after the 0x6e write address: 51 84 03 10 00 32, checksum = XOR incl. 0x6e.
        let p = cp::ddc_ci_set_vcp(cp::VCP_BRIGHTNESS, 50);
        assert_eq!(&p[..6], &[0x51, 0x84, 0x03, 0x10, 0x00, 0x32]);
        let want = 0x6e ^ 0x51 ^ 0x84 ^ 0x03 ^ 0x10 ^ 0x00 ^ 0x32;
        assert_eq!(p[6], want);
        // The checksum makes the XOR of {dest, source, len, opcode, vcp, hi, lo, chk} zero.
        assert_eq!(0x6eu8 ^ p.iter().fold(0u8, |a, &b| a ^ b), 0);
        // Contrast (0x12) and the power VCP (0xd6 = off) carry their codes/values verbatim.
        assert_eq!(
            cp::ddc_ci_set_vcp(cp::VCP_CONTRAST, 0x0140)[3..6],
            [0x12, 0x01, 0x40]
        );
        assert_eq!(
            cp::ddc_ci_set_vcp(cp::VCP_POWER_MODE, cp::POWER_OFF)[3..6],
            [0xd6, 0x00, 0x04]
        );
    }

    #[test]
    fn ddc_set_vcp_message_structure() -> Result {
        // Decrypted DLM wrapper: id=0x36 sub=0x26, head + len at off22, DDC bytes at off24,
        // zeros through off55 and a fresh eight-byte token at off56.
        let m = cp::ddc_set_vcp(0x11, 1, cp::VCP_BRIGHTNESS, 75)?;
        assert_eq!(m.len(), 64);
        assert_eq!(&m[0..6], &[0x36, 0x00, 0x26, 0x00, 0x11, 0x00]);
        assert_eq!(&m[22..24], &[1, 7]);
        assert_eq!(&m[24..31], &cp::ddc_ci_set_vcp(cp::VCP_BRIGHTNESS, 75));
        assert!(m[31..56].iter().all(|&x| x == 0));
        Ok(())
    }

    #[test]
    fn ddc_read_request_message_structure() -> Result {
        let m = cp::ddc_read_req(0x1234, 1)?;
        assert_eq!(m.len(), 32);
        assert_eq!(&m[0..6], &[0x15, 0x00, 0x25, 0x00, 0x34, 0x12]);
        assert!(m[6..22].iter().all(|&x| x == 0));
        assert_eq!(m[22], 1);
        Ok(())
    }

    #[test]
    fn set_mode_has_head_and_exact_dlm_plaintext_length() -> Result {
        let m = cp::set_mode(0x1234, 1, &cp::Timing::UHD_60)?;
        assert_eq!(m.len(), 80);
        assert_eq!(&m[0..6], &[0x48, 0x00, 0x22, 0x00, 0x34, 0x12]);
        assert!(m[6..22].iter().all(|&x| x == 0));
        assert_eq!(&m[22..26], &[1, 2, 0, 0]);
        assert_eq!(u16::from_le_bytes([m[26], m[27]]), 3840);
        assert_eq!(u16::from_le_bytes([m[34], m[35]]), 2160);
        assert_eq!(u16::from_le_bytes([m[70], m[71]]), 0xd040);
        assert_eq!(u16::from_le_bytes([m[68], m[69]]), 0x0200);
        assert_eq!(&m[72..74], &[0, 0]);
        Ok(())
    }

    #[test]
    fn stream_marker_routes_the_selected_head() -> Result {
        let h0 = cp::stream_marker(0x1234, 0, 0x2f, 1)?;
        let h1 = cp::stream_marker(0x1235, 1, 0x2e, 3)?;
        assert_eq!(&h0[0..6], &[0x16, 0, 0x2f, 0, 0x34, 0x12]);
        assert_eq!(&h0[22..24], &[0, 1]);
        assert_eq!(&h1[0..6], &[0x16, 0, 0x2e, 0, 0x35, 0x12]);
        assert_eq!(&h1[22..24], &[1, 3]);
        Ok(())
    }

    #[test]
    fn video_frame_trailer_matches_dlm_cycle_and_head() {
        let t0 = video::wht::frame_trailer(0, 0);
        assert_eq!(
            &t0[..32],
            &[
                0, 0, 0x1c, 0, 4, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 5, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(
            &t0[32..64],
            &[
                0, 0, 0x1c, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0x0a, 0, 4, 2, 0, 0, 0, 8, 0, 0,
                0, 0, 0, 0, 0, 0,
            ]
        );
        // Record C carries the head selector OR'd with 0x10. `sub` is the little-endian u16 at
        // bytes 8..10, so the selector belongs in byte **8**, not byte 9. Records A and B hide the
        // difference for head 0 (both encodings are zero), which is why only this third record
        // catches it -- and why the original bug shipped: writing byte 9 produced 0x1100 instead of
        // DLM's 0x0011 and head 1 never presented a completed frame. The code is the HW-validated
        // side of this; the expectation below was the stale one.
        assert_eq!(
            &t0[64..],
            &[
                0, 0, 0x1c, 0, 4, 0, 0, 0, 0x10, 0, 4, 0, 0, 0, 0, 0, 0x0a, 0, 4, 2, 0, 0, 0, 8, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );

        let t1 = video::wht::frame_trailer(1, 1);
        assert_eq!(u16::from_le_bytes([t1[8], t1[9]]), 0x0001);
        assert_eq!(u16::from_le_bytes([t1[32 + 8], t1[32 + 9]]), 0x0001);
        assert_eq!(u16::from_le_bytes([t1[64 + 8], t1[64 + 9]]), 0x0011);
        assert_eq!(t1[19], 2);
        assert_eq!(t1[23], 8);
        assert_eq!(t1[25], 2);
        assert_eq!(t1[32 + 19], 4);
        assert_eq!(t1[32 + 23], 16);
        assert_eq!(t1[32 + 27], 8);
    }

    #[test]
    fn video_arm_templates_share_the_fixed_1090_byte_prefix() {
        use video_arm_content::{
            VIDEO_ARM_8_CONTENT, VIDEO_ARM_9_CONTENT, VIDEO_ARM_RANDOM_TAIL_OFFSET,
        };

        // Three independently decrypted sessions (both heads in the second and third) prove this
        // boundary: only the final 14 bytes vary. The builder replaces those captured sample tails
        // with a fresh RNG request for every #8/#9 record.
        assert_eq!(VIDEO_ARM_RANDOM_TAIL_OFFSET, 1090);
        assert_eq!(VIDEO_ARM_8_CONTENT.len() - VIDEO_ARM_RANDOM_TAIL_OFFSET, 14);
        assert_eq!(VIDEO_ARM_9_CONTENT.len() - VIDEO_ARM_RANDOM_TAIL_OFFSET, 14);
        assert_eq!(
            &VIDEO_ARM_8_CONTENT[..VIDEO_ARM_RANDOM_TAIL_OFFSET],
            &VIDEO_ARM_9_CONTENT[..VIDEO_ARM_RANDOM_TAIL_OFFSET]
        );
        assert_ne!(
            &VIDEO_ARM_8_CONTENT[VIDEO_ARM_RANDOM_TAIL_OFFSET..],
            &VIDEO_ARM_9_CONTENT[VIDEO_ARM_RANDOM_TAIL_OFFSET..]
        );
    }
}
