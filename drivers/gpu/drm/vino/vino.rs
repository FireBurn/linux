// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (C) 2026 Mike Lothian

//! DRM/KMS driver for DisplayLink DL3 docks.
//!
//! Vino drives the Dell Universal Dock D6000 using a clean-room implementation of its USB control,
//! HDCP authentication and compressed video protocols. Each device owns its control session and
//! exposes two atomic KMS pipelines backed by shmem GEM objects.

use kernel::{
    alloc::flags::GFP_KERNEL,
    alloc::Flags,
    device::{self, Core},
    drm,
    drm::display::hdcp as drm_hdcp,
    error::code::{EBUSY, EINVAL, ENODEV, EPROTO, ETIMEDOUT},
    prelude::*,
    sync::{aref::ARef, new_mutex, Arc, Mutex},
    time::{
        delay::{fsleep, udelay},
        Delta, Instant, Monotonic,
    },
    usb,
    workqueue::{impl_has_work, new_work, Work, WorkItem},
};

/// Whether the load-time `debug` parameter requested verbose protocol and scanout diagnostics.
pub(crate) fn debug_enabled() -> bool {
    *crate::module_parameters::debug.value() != 0
}

/// Whether this one module load may disclose its ephemeral session material for a wire capture.
///
/// This is deliberately separate from ordinary debug logging: the values make a usbmon trace
/// decryptable and must never appear during a normal load.
fn trace_crypto_enabled() -> bool {
    *crate::module_parameters::trace_crypto.value() != 0
}

/// Emit a driver diagnostic only when the load-time `debug` parameter is nonzero.
macro_rules! vino_debug {
    ($($arg:tt)*) => {
        if crate::debug_enabled() {
            kernel::pr_info!($($arg)*);
        }
    };
}

/// Device-prefixed counterpart to [`vino_debug`].
macro_rules! vino_dev_debug {
    ($dev:expr, $($arg:tt)*) => {
        if crate::debug_enabled() {
            kernel::dev_info!($dev, $($arg)*);
        }
    };
}

/// A byte slice on one line, as `08 0a 08 0a`.
///
/// `{:#04x?}` renders an array one element per line: the alternate flag asks the derived `Debug`
/// for pretty output, which in a log line is unreadable.
pub(crate) struct HexList<'a>(pub(crate) &'a [u8]);

impl kernel::fmt::Display for HexList<'_> {
    fn fmt(&self, f: &mut kernel::fmt::Formatter<'_>) -> kernel::fmt::Result {
        for (i, byte) in self.0.iter().enumerate() {
            write!(f, "{}{byte:02x}", if i == 0 { "" } else { " " })?;
        }
        Ok(())
    }
}

/// DisplayLink vendor id.
const VID_DISPLAYLINK: u16 = 0x17e9;
/// Dell Universal Dock D6000 (DL3 family) product id.
const PID_D6000: u16 = 0x6006;
/// WAVLINK DL7400 and relatives: "Universal DP Quad Display Docking 16G", identity tail
/// `NavaDock`, i.e. the Navarro platform on DL-7000 silicon.
const PID_DL7400: u16 = 0x7000;

/// Dock identification and the per-dock parameters the rest of the driver reads.
mod profile;
/// USB endpoint resolution and the I/O handle transfers go through.
mod usb_link;

pub(crate) use profile::{DockProfile, EP_CTRL_IN, EP_CTRL_OUT, PROFILE_D6000, PROFILE_DL7400};
pub(crate) use usb_link::{Endpoints, UsbLink, EP84_BUF};

/// USB transfer timeout used during session setup.
fn timeout() -> Delta {
    Delta::from_millis(1000)
}

/// Short timeout for draining a per-message control reply after a runtime `send_cp`.
///
/// EP84 remains in lockstep with EP02, but not every message elicits a reply. A NAK or timeout
/// therefore means that there is nothing to drain and must not stall scanout or keepalive work.
pub(crate) fn cp_reply_timeout() -> Delta {
    Delta::from_millis(8)
}

/// Time allowed for the downstream receiver to calculate H' during repeater authentication.
///
/// The dock acknowledges `AKE_No_Stored_km` before that calculation is complete, so an
/// acknowledgment cannot be used as the readiness signal.
// The DL7400's downstream receiver produces H' about 235--240 ms after AKE_No_Stored_km in the
// working DLM transaction.  Wake just before that result instead of advancing after an arbitrary
// shorter quiet window; `wait_perhead_push(0x07)` below remains the actual completion gate.
const HDCP_HPRIME_WAIT_US: i64 = 220_000;

/// Wait until `anchor` is at least `target_us` old.
fn hold_until(anchor: Instant<Monotonic>, target_us: i64) {
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

mod ake;
mod color;
mod cp;
mod crypto;
mod firmware;
mod hdcp;
mod proto;
mod rng;
mod video;
mod video_arm;

/// The state a completed HDCP 2.2 AKE leaves for control-plane setup.
struct Session {
    ks: kernel::crypto::Secret<{ drm_hdcp::ENCRYPTED_SESSION_KEY_LEN }>,
    riv: [u8; drm_hdcp::RIV_LEN],
    /// Next inner sequence counter after the AKE messages sent by [`run_ake`].
    next_ctr: u16,
    /// Receiver key retained for each downstream repeater authentication.
    rsa: kernel::crypto::akcipher::RsaPublicKey,
    rxid_list: KVec<u8>,
}

/// Tally of one [`drain_ep84`](VinoDriver::drain_ep84) sweep.
///
/// An acknowledgment is counted only after its inner header decrypts successfully. A tagged
/// frame which does not decrypt is counted separately as a rejection.
#[derive(Default, Clone, Copy)]
struct Ep84Drain {
    reads: usize,
    acks: usize,
    rejects: usize,
    /// Sticky EDID-readiness result across combined sweeps.
    edid_ready: bool,
    /// Inner counter echoed by a per-head display-capability reply.
    display_cap_ctr: Option<u16>,
    /// Fresh per-head `rrx` used by downstream repeater authentication.
    perhead_rrx: Option<[u8; drm_hdcp::RRX_LEN]>,
    /// Bit `msg_id` is set for every downstream-HDCP push observed in this sweep.
    perhead_seen: u32,
    perhead_repeater: Option<bool>,
    perhead_hprime: Option<[u8; drm_hdcp::H_PRIME_LEN]>,
    perhead_lprime: Option<[u8; drm_hdcp::L_PRIME_LEN]>,
    /// Navarro's receiver-list payload is nine authenticated list-header bytes followed by V'.
    perhead_v: Option<([u8; 9], [u8; drm_hdcp::V_PRIME_HALF_LEN])>,
    perhead_auth_status: Option<u8>,
    perhead_mprime: Option<[u8; drm_hdcp::H_PRIME_LEN]>,
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
        self.perhead_seen |= o.perhead_seen;
        self.perhead_repeater = self.perhead_repeater.or(o.perhead_repeater);
        self.perhead_hprime = self.perhead_hprime.or(o.perhead_hprime);
        self.perhead_lprime = self.perhead_lprime.or(o.perhead_lprime);
        self.perhead_v = self.perhead_v.or(o.perhead_v);
        self.perhead_auth_status = self.perhead_auth_status.or(o.perhead_auth_status);
        self.perhead_mprime = self.perhead_mprime.or(o.perhead_mprime);
    }

    fn observe_perhead(&mut self, push: cp::PerheadHdcpPush) {
        if push.msg_id < 32 {
            self.perhead_seen |= 1u32 << push.msg_id;
        }
        match push.msg_id {
            // AKE_Send_Cert: the first vendor payload byte is the repeater flag.
            0x03 if push.payload_len >= 1 => {
                self.perhead_repeater = Some(push.payload[0] != 0);
            }
            0x06 if push.payload_len >= drm_hdcp::RRX_LEN => {
                let mut v = [0u8; drm_hdcp::RRX_LEN];
                v.copy_from_slice(&push.payload[..drm_hdcp::RRX_LEN]);
                self.perhead_rrx = Some(v);
            }
            0x07 if push.payload_len >= drm_hdcp::H_PRIME_LEN => {
                let mut v = [0u8; drm_hdcp::H_PRIME_LEN];
                v.copy_from_slice(&push.payload[..drm_hdcp::H_PRIME_LEN]);
                self.perhead_hprime = Some(v);
            }
            0x0a if push.payload_len >= drm_hdcp::L_PRIME_LEN => {
                let mut v = [0u8; drm_hdcp::L_PRIME_LEN];
                v.copy_from_slice(&push.payload[..drm_hdcp::L_PRIME_LEN]);
                self.perhead_lprime = Some(v);
            }
            // ReceiverID_List: RxInfo/seq/list header (9 bytes), V' (16 bytes), padding.
            0x0c if push.payload_len >= 9 + drm_hdcp::V_PRIME_HALF_LEN => {
                let mut list = [0u8; 9];
                let mut vprime = [0u8; drm_hdcp::V_PRIME_HALF_LEN];
                list.copy_from_slice(&push.payload[..9]);
                vprime.copy_from_slice(&push.payload[9..9 + drm_hdcp::V_PRIME_HALF_LEN]);
                self.perhead_v = Some((list, vprime));
            }
            // DisplayLink prefixes ReceiverAuthStatus with one vendor status byte. The HDCP
            // value is payload[1] (`00 04` in all four working DLM per-head exchanges).
            0x12 if push.payload_len >= 2 => {
                self.perhead_auth_status = Some(push.payload[1]);
            }
            0x11 if push.payload_len >= drm_hdcp::H_PRIME_LEN => {
                let mut v = [0u8; drm_hdcp::H_PRIME_LEN];
                v.copy_from_slice(&push.payload[..drm_hdcp::H_PRIME_LEN]);
                self.perhead_mprime = Some(v);
            }
            _ => {}
        }
    }

    fn saw_perhead(&self, msg_id: u8) -> bool {
        msg_id < 32 && self.perhead_seen & (1u32 << msg_id) != 0
    }
}

mod drm_sink;

/// The USB driver itself. Stateless: everything per-binding lives in [`VinoBoundData`].
/// Log what this device is, and what it exposes, before any protocol runs.
///
/// A DisplayLink generation is not identifiable from the USB IDs alone -- the DL3 protocol vino
/// speaks does not apply to a DL-1x5 part, and the first sign of that is a control session timing
/// out long after bind succeeded. Printing the descriptor and the endpoint inventory up front means
/// a report from unfamiliar hardware carries what is needed to place it, without a debug build:
/// `bcdDevice` is the vendor's revision, and the endpoint list distinguishes a full DL3 control
/// device (bulk OUT 0x02 + bulk IN 0x84 + video 0x08) from a part that only has one bulk pipe.
fn log_device_identity(
    cdev: &device::Device<Core<'_>>,
    intf: &usb::Interface<Core<'_>>,
    ifnum: u8,
) {
    // The descriptor describes the whole device, so print it once rather than per interface.
    if ifnum == 0 {
        let dev: &usb::Device<Core<'_>> = intf.as_ref();
        let vid = dev.vendor_id();
        let pid = dev.product_id();
        let bcd = dev.bcd_device();
        let usb_bcd = dev.bcd_usb();
        vino_dev_debug!(
            cdev,
            "USB {vid:04x}:{pid:04x} bcdDevice {:x}.{:02x} bcdUSB {:x}.{:02x} speed {}\n",
            bcd >> 8,
            bcd & 0xff,
            usb_bcd >> 8,
            usb_bcd & 0xff,
            dev.speed_str()
        );
        // The USB core only caches these when the device answered the string requests.
        match (dev.manufacturer(), dev.product()) {
            (Some(m), Some(p)) => vino_dev_debug!(cdev, "{m} {p}\n"),
            (None, Some(p)) => vino_dev_debug!(cdev, "{p}\n"),
            (Some(m), None) => vino_dev_debug!(cdev, "{m} (no product string)\n"),
            (None, None) => vino_dev_debug!(cdev, "no manufacturer/product strings\n"),
        }
    }
    for ep in intf.cur_altsetting().endpoints() {
        let dir = match ep.endpoint_dir() {
            kernel::usb::ch9::Direction::In => "in",
            kernel::usb::ch9::Direction::Out => "out",
        };
        let kind = match ep.endpoint_type() {
            usb::EndpointType::Control => "control",
            usb::EndpointType::Isoc => "isoc",
            usb::EndpointType::Bulk => "bulk",
            usb::EndpointType::Int => "int",
        };
        // bEndpointAddress as the descriptor carries it: number plus the direction bit.
        let addr = ep.endpoint_number()
            | match ep.endpoint_dir() {
                kernel::usb::ch9::Direction::In => 0x80,
                kernel::usb::ch9::Direction::Out => 0,
            };
        vino_dev_debug!(cdev, "  ep {addr:#04x} {kind}-{dir} maxp {}\n", ep.maxp());
    }
}

struct VinoDriver;

/// Per-bound-interface driver state.
///
/// Carries the DRM [`Registration`](drm::Registration), whose lifetime is tied to this bound
/// device, so unbinding unregisters the card through the accepted registration teardown rather
/// than a driver-local force-unplug.
struct VinoBoundData {
    _intf: ARef<usb::Interface>,
    /// The registered DRM card, dropped on unbind.
    ///
    /// `None` only on idle non-control interfaces. On the control interface it owns the DRM
    /// registration and provides `disconnect()` access to the device state.
    registration: Option<drm::Registration<'static, drm_sink::VinoDrmDriver>>,
    /// Owned handle to the deferred bring-up work (control interface only). `disconnect()` takes
    /// the option under the mutex before synchronously cancelling the work and unplugging DRM.
    /// The mutex itself is heap-pinned because kernel locks must not move after initialization.
    bringup: Pin<KBox<Mutex<Option<Arc<BringUp>>>>>,
}

/// Deferred bring-up work item.
///
/// The device's dedicated session queue keeps blocking authentication and steady-state control I/O
/// out of the USB probe path and the shared system workqueues.
#[pin_data]
struct BringUp {
    ddev: ARef<drm_sink::VinoDrmDevice>,
    /// Which dock this is. The bring-up sequence differs by platform (see [`DockProfile`]), and
    /// the work item runs long after `probe` has returned, so it carries the profile itself.
    profile: &'static DockProfile,
    #[pin]
    work: Work<BringUp>,
}

impl_has_work! {
    impl HasWork<Self> for BringUp { self.work }
}

impl BringUp {
    fn new(
        ddev: ARef<drm_sink::VinoDrmDevice>,
        profile: &'static DockProfile,
    ) -> Result<Arc<Self>> {
        Arc::pin_init(
            pin_init!(BringUp {
                ddev,
                profile,
                work <- new_work!("vino::bring_up"),
            }),
            GFP_KERNEL,
        )
    }
}

impl WorkItem for BringUp {
    type Pointer = Arc<BringUp>;

    fn run(this: Arc<BringUp>) {
        let profile = this.profile;
        let data: &drm_sink::VinoDrmData = &this.ddev;
        let Ok(link) = UsbLink::open(&data.io, data.eps) else {
            return;
        };
        let dev = &link;
        let cdev: &device::Device = dev.io().interface().as_ref();
        let ddev = &this.ddev;
        // Establish the transport, authenticate the link and configure the encrypted control
        // session before publishing the connectors. A transient failure must not leave an
        // otherwise bound device inert until it is physically replugged.
        // A dock can refuse a session outright: it answers every control request while NAKing the
        // first EP02 bulk write until it times out. Back off to about half a minute before giving
        // the device up.
        const SESSION_ATTEMPTS: usize = 8;
        let mut established = false;
        for attempt in 1..=SESSION_ATTEMPTS {
            if data.is_shutting_down() {
                return;
            }
            let result = (|| -> Result {
                VinoDriver::bring_up(dev, profile)?;
                vino_dev_debug!(cdev, "plaintext session initialized\n");
                let mut session = VinoDriver::run_ake(dev)?;
                vino_dev_debug!(cdev, "HDCP AKE + LC + SKE complete\n");

                let mut edid_out: Option<KVec<u8>> = None;
                let mut edid_heads: [Option<KVec<u8>>; VinoDriver::CP_SETUP_HEADS] =
                    core::array::from_fn(|_| None);
                let mut video_keys = core::array::from_fn(|_| kernel::crypto::Secret::zeroed());
                let mut heads_present = [false; VinoDriver::CP_SETUP_HEADS];
                let mut discovery_deferred = [false; VinoDriver::CP_SETUP_HEADS];
                let (n, wseq_end, ctr_end) = VinoDriver::send_cp_setup(
                    dev,
                    profile,
                    &mut session,
                    &mut edid_out,
                    &mut edid_heads,
                    &mut video_keys,
                    &mut heads_present,
                    &mut discovery_deferred,
                )?;
                vino_dev_debug!(cdev, "encrypted control setup complete ({n} messages)\n");

                // `send_cp_setup` only returns after an authenticated reply proves that the dock
                // engaged the session. Publish it before connector state so runtime recovery can
                // immediately finish any per-head discovery transaction that was deferred.
                let drm_dev: &drm_sink::VinoDrmDevice = ddev;
                let data: &drm_sink::VinoDrmData = drm_dev;
                data.set_cp_engaged(true);
                data.publish_session(
                    dev,
                    &session.ks,
                    &session.riv,
                    wseq_end,
                    ctr_end,
                    profile.ep84_queue_depth,
                );
                data.set_video_keys(video_keys);

                // One line naming what setup found on every physical socket, including the ones
                // this dock does not drive as distinct streams. Which socket a monitor is in is
                // otherwise invisible from dmesg, and it decides whether a dark output is a sink
                // problem at all: a head vino never drives cannot light whatever is plugged into
                // it. `cap` is the socket's DISPLAY-CAP push, `edid` its raw EDID; the pair
                // distinguishes an empty socket from one whose sink cannot be read.
                for head in 0..usize::from(profile.connectors) {
                    vino_dev_debug!(
                        cdev,
                        "socket {} -- cap:{} edid:{} deferred:{} driven:{}\n",
                        head + 1,
                        if heads_present[head] { "yes" } else { "no " },
                        if edid_heads[head].is_some() {
                            "yes"
                        } else {
                            "no "
                        },
                        if discovery_deferred[head] {
                            "yes"
                        } else {
                            "no "
                        },
                        if data.runtime_connector(head) {
                            "yes"
                        } else {
                            "no "
                        },
                    );
                }

                // Cache complete per-head discovery results before emitting the single initial
                // hotplug event. A timed-out head remains absent and the keepalive's existing
                // bounded re-engagement path retries it without discarding the live session.
                for (head, slot) in edid_heads
                    .into_iter()
                    .enumerate()
                    .take(usize::from(profile.connectors))
                {
                    if discovery_deferred[head] {
                        continue;
                    }
                    let have_edid = slot.is_some();
                    if let Some(blob) = slot {
                        let n = blob.len();
                        data.set_edid(head, blob);
                        vino_dev_debug!(
                            cdev,
                            "cached socket {socket} EDID ({n} bytes)\n",
                            socket = head + 1
                        );
                    }
                    // A recovered EDID is the presence signal on both platforms. Publishing a
                    // connector without one puts a fallback mode into an empty socket and makes
                    // the dock lay out buffers for an output that does not exist.
                    if have_edid {
                        data.set_connected(head);
                        dev_info!(
                            cdev,
                            "socket {socket} monitor connected\n",
                            socket = head + 1
                        );
                    }
                }

                // Navarro normally receives each EDID on the fetch drain, exactly as DLM does, but
                // after a dock re-enumeration a response can arrive seconds late. Publishing that
                // partial topology lets userspace mode-set one head while its sibling is still
                // arriving, which resets this dock, so retry the deferred heads before the single
                // initial hotplug. They are interleaved: a head that never answers must not hold
                // up one that would. A normal setup sends no additional control messages.
                if profile.perhead_onehot() {
                    /// How long the deferred heads are retried before the topology is published.
                    const INITIAL_RECOVERY_MS: i64 = 6000;
                    /// Extra time granted after a head answers, for a sibling close behind it.
                    const SIBLING_GRACE_MS: i64 = 1500;

                    let mut pending: [bool; VinoDriver::CP_SETUP_HEADS] =
                        core::array::from_fn(|head| {
                            head < usize::from(profile.connectors)
                                && discovery_deferred[head]
                                && data.runtime_connector(head)
                                && !data.head_present(head)
                        });
                    // One probe answers "is this socket empty?"; a re-engage is seven messages
                    // carrying ~575 ms of mandated delay. A probe that cannot answer is not
                    // evidence of absence, so only a definite negative stands a head down.
                    for head in 0..VinoDriver::CP_SETUP_HEADS {
                        if pending[head] && data.probe_head_present(dev, head as u8) == Some(false)
                        {
                            pending[head] = false;
                        }
                    }

                    let started = Instant::<Monotonic>::now();
                    let mut give_up = started + Delta::from_millis(INITIAL_RECOVERY_MS);
                    let expired = |give_up: Instant<Monotonic>| {
                        (Instant::<Monotonic>::now() - give_up).as_millis() >= 0
                    };
                    let mut pass = 0u32;
                    while pending.iter().any(|p| *p)
                        && !data.is_shutting_down()
                        && !expired(give_up)
                    {
                        pass += 1;
                        for head in 0..VinoDriver::CP_SETUP_HEADS {
                            // Tested per head, not per pass: a re-engage the dock ignores costs
                            // seconds, so a pass across four of them would run well past the
                            // window before anything looked at it.
                            if !pending[head] || expired(give_up) {
                                continue;
                            }
                            if let Ok(true) = data.reengage_head(dev, head as u8) {
                                data.set_connected(head);
                                pending[head] = false;
                                vino_dev_debug!(
                                    cdev,
                                    "socket {socket} monitor connected during initial \
                                     recovery (pass {pass})\n",
                                    socket = head + 1
                                );
                                let grace = Instant::<Monotonic>::now()
                                    + Delta::from_millis(SIBLING_GRACE_MS);
                                if (grace - give_up).as_millis() > 0 {
                                    give_up = grace;
                                }
                            }
                        }
                        fsleep(Delta::from_millis(250));
                    }
                    let waited = (Instant::<Monotonic>::now() - started).as_millis();
                    for head in 0..VinoDriver::CP_SETUP_HEADS {
                        if pending[head] {
                            dev_warn!(
                                cdev,
                                "socket {socket} never answered its EDID fetch \
                                 ({pass} passes over {waited} ms); publishing without it\n",
                                socket = head + 1
                            );
                        }
                    }
                }
                Ok(())
            })();

            match result {
                Ok(()) => {
                    established = true;
                    break;
                }
                Err(e) if attempt < SESSION_ATTEMPTS => {
                    let backoff = 250i64 << (attempt - 1).min(5);
                    dev_warn!(
                        cdev,
                        "control-session attempt {attempt}/{SESSION_ATTEMPTS} failed \
                         ({e:?}); retrying in {backoff} ms\n"
                    );
                    fsleep(Delta::from_millis(backoff));
                }
                Err(e) => dev_err!(
                    cdev,
                    "control session failed after {SESSION_ATTEMPTS} attempts ({e:?})\n"
                ),
            }
        }
        if !established {
            return;
        }
        {
            let drm_dev: &drm_sink::VinoDrmDevice = ddev;
            // Ridge needs a bounded training interval before userspace can submit a mode set.
            // Navarro's working transcript has already performed its fixed status sequence in
            // `send_cp_setup`; another 1.3 seconds inserted ~84 messages before its first clear.
            if data.cp_engaged() && !profile.perhead_onehot() {
                let data: &drm_sink::VinoDrmData = drm_dev;
                let start = Instant::<Monotonic>::now();
                let window = Delta::from_millis(1300);
                let mut polls = 0u32;
                while Instant::<Monotonic>::now() - start < window && !data.is_shutting_down() {
                    let _ = data.send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c));
                    polls += 1;
                    fsleep(Delta::from_millis(15));
                }
                vino_dev_debug!(cdev, "link ready after {polls} status polls\n");
            }
            if data.is_navarro() {
                data.hold_cp_for_initial_modeset();
            }
            drm_dev.hotplug_event();
            vino_dev_debug!(cdev, "encrypted control session ready\n");

            // The dock requires a continuous control dialogue for the lifetime of the session.
            let data: &drm_sink::VinoDrmData = drm_dev;
            vino_dev_debug!(cdev, "starting control keepalive\n");
            let mut sent = 0u32;
            // Heartbeats have an independent fixed cadence alongside the status queries.
            const HEARTBEAT_PERIOD: Delta = Delta::from_secs(3);
            let mut next_heartbeat = Instant::<Monotonic>::now() + HEARTBEAT_PERIOD;
            // Probe downstream presence slowly and debounce transitions.
            const PRESENCE_PERIOD: Delta = Delta::from_millis(1000);
            let mut next_presence = Instant::<Monotonic>::now() + PRESENCE_PERIOD;
            let mut head_known = [false; VinoDriver::CP_SETUP_HEADS];
            // The presence probe's last verdict per connector, or `None` until it has answered
            // once. Distinguishes "no monitor here" from "not asked yet", which the blind
            // re-engage retry below needs and `head_known` cannot express.
            let mut head_probed: [Option<bool>; VinoDriver::CP_SETUP_HEADS] =
                [None; VinoDriver::CP_SETUP_HEADS];
            let mut head_debounce = [0u8; VinoDriver::CP_SETUP_HEADS];
            // Floor on the gap between presence probes. A downstream event brings the probe
            // forward; without a floor it would run once per loop iteration for as long as the
            // dock keeps talking.
            const PRESENCE_MIN_GAP: Delta = Delta::from_millis(50);
            /// How long a connector must read absent before its monitor is called removed.
            ///
            /// A removal has to be debounced in TIME, not in probes: every `id=0x44` reply sets the
            /// downstream-event flag, and a presence probe's own reply *is* an `id=0x44`, so the
            /// watcher kept pulling itself forward to `PRESENCE_MIN_GAP` and "two consecutive
            /// contrary reads" fired 132 ms after the first negative.
            ///
            /// Measured on a lit, idle DL-7400: the absent runs are 0.11 s to 2.29 s, twenty-nine
            /// of them over three minutes, reaching 2.46 s around a mode change.
            ///
            /// The debounce is only half of it. Those blips are the dock really dropping the sink,
            /// and letting the connector disappear is what repairs them: the compositor re-enables
            /// the output and the resulting mode set relights the panel. Debouncing alone leaves
            /// the head dark for good, so this works only together with `repair_flapped_head`.
            const PRESENCE_REMOVE_MS: i64 = 5000;
            let mut head_absent_since: [Option<Instant<Monotonic>>; VinoDriver::CP_SETUP_HEADS] =
                [None; VinoDriver::CP_SETUP_HEADS];
            // Whether a head's current run of negative probes has lasted long enough to be acted
            // on, starting the run if this is its first answer.
            //
            // Both the removal path and the EDID-recovery stand-down need this and for the same
            // reason, so they share one notion of it.
            let sustained_absent = |run: &mut Option<Instant<Monotonic>>| -> bool {
                let since = *run.get_or_insert_with(Instant::<Monotonic>::now);
                (Instant::<Monotonic>::now() - since).as_millis() >= PRESENCE_REMOVE_MS
            };
            // A recovered sink need not emit a uniquely identifiable event, so a head whose
            // discovery was deferred is retried at a bounded cadence until the probe answers.
            const REENGAGE_RETRY: Delta = Delta::from_millis(4000);
            let mut next_reengage = [Instant::<Monotonic>::now(); VinoDriver::CP_SETUP_HEADS];
            /// Settling period after re-engagement during which a negative probe is ignored.
            const PRESENCE_GRACE: Delta = Delta::from_millis(10_000);
            let mut presence_grace = [Instant::<Monotonic>::now(); VinoDriver::CP_SETUP_HEADS];
            /// Quiet window a runtime arrival waits out before userspace is told.
            ///
            /// A mode set is dock-wide, so two heads announced separately make the compositor
            /// reconfigure the dock twice and it re-enumerates. Each arrival restarts the window
            /// and one event covers the burst. A removal is announced immediately.
            const HOTPLUG_COALESCE: Delta = Delta::from_millis(1500);
            let mut hotplug_due: Option<Instant<Monotonic>> = None;
            // When the current run of silent probes started; only read while `head_silent > 0`.
            for h in 0..data.connector_count() {
                if !data.runtime_connector(h) {
                    continue;
                }
                head_known[h] = data.head_present(h);
            }
            // A normal hotplug commit claims this hold almost immediately. Keep a bounded escape
            // for a userspace session which elects not to light either connector at all.
            const INITIAL_MODESET_QUIET_LIMIT: u32 = 5000;
            let mut initial_quiet_ms = 0u32;
            // The cold activation owns EP02 for several seconds. Deadlines which expire while it
            // owns the link must be re-based when it releases it; otherwise the first post-close
            // loop sends an overdue heartbeat and presence probes ahead of the status dialogue.
            // DLM instead continues with status counters 184, 185, ... immediately after its
            // closing markers.
            let mut timeline_was_exclusive = false;
            while !data.is_shutting_down() {
                // The dock stopped answering and the session was abandoned. Take the outputs down
                // rather than poll a link that cannot carry anything: userspace can move its
                // windows off a connector that has disappeared, but not off one that is merely
                // frozen. Recovery is a replug, which rebinds and starts a fresh session.
                if !data.cp_link_alive() {
                    for h in 0..data.connector_count() {
                        if data.runtime_connector(h) && head_known[h] {
                            head_known[h] = false;
                            data.set_disconnected(h);
                            dev_info!(
                                cdev,
                                "vino: socket {socket} dropped with the control session\n",
                                socket = h + 1
                            );
                        }
                    }
                    drm_dev.hotplug_event();
                    break;
                }
                if data.initial_modeset_quiet() {
                    // Quiet means no unsolicited EP02 writes; DLM still has its one EP84 reader
                    // continuously posted and reaped. Keep draining pushes while userspace is
                    // preparing the first mode set so that transaction does not begin behind a
                    // multi-second status backlog.
                    data.drain_cp_pushes(dev, 8);
                    if initial_quiet_ms >= INITIAL_MODESET_QUIET_LIMIT {
                        data.release_initial_modeset_quiet();
                        vino_dev_debug!(
                            cdev,
                            "no initial mode set after {INITIAL_MODESET_QUIET_LIMIT} ms; \
                             releasing control keepalive\n"
                        );
                    } else {
                        initial_quiet_ms += 1;
                        fsleep(Delta::from_millis(1));
                        continue;
                    }
                }
                // Mode-set markers and video activation form one exclusive transaction.
                if data.cp_timeline_exclusive() {
                    timeline_was_exclusive = true;
                    // The KMS worker owns EP02, but it releases `cp_link` between scheduled
                    // writes. Reap asynchronous EP84 traffic in those gaps just as DLM's reader
                    // thread does; request replies remain protected because `send_cp_reply`
                    // holds the mutex until it sees the matching counter.
                    data.drain_cp_pushes(dev, 8);
                    fsleep(Delta::from_millis(1));
                    continue;
                }
                if timeline_was_exclusive {
                    let resumed = Instant::<Monotonic>::now();
                    next_heartbeat = resumed + HEARTBEAT_PERIOD;
                    next_presence = resumed + PRESENCE_PERIOD;
                    timeline_was_exclusive = false;
                }
                if data
                    .send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c))
                    .is_ok()
                {
                    sent += 1;
                }
                // Compare through the signed `Delta` returned by subtracting
                // two instants.
                let now = Instant::<Monotonic>::now();
                if (now - next_heartbeat).as_millis() >= 0 {
                    let _ = data.send_cp(dev, 0x16, 0, cp::heartbeat);
                    // Advance from the previous deadline so a slow send does not cause drift.
                    next_heartbeat = next_heartbeat + HEARTBEAT_PERIOD;
                    if (now - next_heartbeat).as_millis() > 0 {
                        next_heartbeat = now + HEARTBEAT_PERIOD; // fell far behind; resynchronise
                    }
                }
                // Consume asynchronous pushes instead of leaving them for the next paired read.
                const MAX_UNPAIRED_DRAIN: usize = 4;
                data.drain_cp_pushes(dev, MAX_UNPAIRED_DRAIN);
                // Recover a head whose setup-time discovery was deferred or timed out. This is a
                // recovery, not a poll: once the presence probe has answered for a connector that
                // answer is authoritative and this stands down, or an empty socket costs seven
                // unanswered CP messages every `REENGAGE_RETRY` for the life of the session.
                {
                    let now_r = Instant::<Monotonic>::now();
                    for h in 0..data.connector_count() {
                        let socket = h + 1;
                        if !data.runtime_connector(h) {
                            continue;
                        }
                        if head_probed[h] == Some(false) {
                            continue;
                        }
                        if head_known[h] || (now_r - next_reengage[h]).as_millis() < 0 {
                            continue;
                        }
                        // A blanked head's sink is idle because vino asked for it. Re-engaging it
                        // here would also clear `self_blanked`, since `reengage_head` does so on
                        // entry, and the connector would then be torn down mid-blank.
                        if data.is_self_blanked(h) {
                            continue;
                        }
                        next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                        // Same trade as the initial recovery: one cheap probe instead of seven
                        // paced messages. It also keeps an empty socket's retry from interleaving
                        // ~575 ms of engage traffic into a mode-set transaction on another head,
                        // which is measurable as delayed activation, not merely as noise.
                        if data.probe_head_present(dev, h as u8) == Some(false) {
                            // Do not stand down on the first negative. A recovered EDID is this
                            // dock's presence signal; the probe is a weaker one that reports a lit
                            // sink absent for up to 2.5 s at a time. Latching here costs a monitor
                            // slow to answer at bring-up its whole session: it is never asked for
                            // an EDID again, and only re-enumerating the dock brings it back.
                            //
                            // So hold a negative to the same evidence a removal needs. An empty
                            // socket still stands down, just `PRESENCE_REMOVE_MS` later, having
                            // cost one probe message per `REENGAGE_RETRY` in the meantime.
                            if sustained_absent(&mut head_absent_since[h]) {
                                head_probed[h] = Some(false);
                            }
                            continue;
                        }
                        head_absent_since[h] = None;
                        vino_dev_debug!(
                            cdev,
                            "socket {socket} absent -- retrying the sink re-engage\n"
                        );
                        // A valid EDID proves presence even while the generic status reply still
                        // reflects an unengaged EDID handler.
                        match data.reengage_head(dev, h as u8) {
                            Ok(true) => {
                                data.set_connected(h);
                                head_known[h] = true;
                                head_debounce[h] = 0;
                                presence_grace[h] = Instant::<Monotonic>::now() + PRESENCE_GRACE;
                                vino_dev_debug!(
                                    cdev,
                                    "socket {socket} monitor connected after sink re-engagement\n"
                                );
                                hotplug_due = Some(Instant::<Monotonic>::now() + HOTPLUG_COALESCE);
                            }
                            Ok(false) => {}
                            Err(e) => {
                                vino_dev_debug!(
                                    cdev,
                                    "socket {socket} sink re-engagement failed ({e:?})\n"
                                )
                            }
                        }
                        next_presence = Instant::<Monotonic>::now();
                    }
                }
                // A topology push brings presence probing forward.  Do not also cancel the
                // absent-head re-engage backoff here: Navarro emits an `id=0x44` reply for every
                // ordinary presence probe, and `drain_cp_pushes` deliberately reports that as a
                // downstream event.  Resetting `next_reengage` on each such reply turned two
                // empty sockets into a continuous engage/EDID loop instead of the documented
                // four-second retry cadence.  The probe below observes an actual arrival and
                // then re-engages that specific connector immediately.
                if data.take_downstream_event() {
                    // Bring the probe forward, but never below `PRESENCE_MIN_GAP`.
                    let soonest = Instant::<Monotonic>::now() + PRESENCE_MIN_GAP;
                    if (next_presence - soonest).as_millis() > 0 {
                        next_presence = soonest;
                    }
                }
                let now_p = Instant::<Monotonic>::now();
                if (now_p - next_presence).as_millis() >= 0 {
                    next_presence = now_p + PRESENCE_PERIOD;
                    for h in 0..data.connector_count() {
                        let socket = h + 1;
                        if !data.runtime_connector(h) {
                            continue;
                        }
                        // A missing reply carries no status bit, so it is not evidence that this
                        // monitor disappeared; wait for a decodable negative instead of tearing
                        // down a live connector.
                        let Some(present) = data.probe_head_present(dev, h as u8) else {
                            continue;
                        };
                        // Nothing this probe says about a head vino blanked is news, in either
                        // direction: the absence is vino's own doing, and this dock also flaps a
                        // blanked sink back to *present*, which would re-engage it and clear
                        // `self_blanked` -- leaving the next sustained negative free to tear the
                        // connector down mid-blank. The flag is cleared by the wake, in
                        // `atomic_enable`.
                        if data.is_self_blanked(h) {
                            head_debounce[h] = 0;
                            head_absent_since[h] = None;
                            continue;
                        }
                        // The probe has spoken for this connector, so the blind re-engage retry
                        // above stands down for it -- but only a *positive* answer is authoritative
                        // straight away. A negative one has to outlast `PRESENCE_REMOVE_MS` for the
                        // same reason it does above: this probe reports a lit sink absent
                        // routinely, and standing the EDID recovery down on one of those blips is
                        // what left a head dark for a whole session.
                        if present {
                            head_probed[h] = Some(true);
                            // The absent run is cleared below, not here: the "flap healed on its
                            // own" line reads it with `take()` and would never fire again.
                        } else {
                            if sustained_absent(&mut head_absent_since[h]) {
                                head_probed[h] = Some(false);
                            }
                        }
                        if present == head_known[h] {
                            head_debounce[h] = 0;
                            // The sink came back before the removal deadline, so the connector was
                            // never dropped and nothing downstream will re-drive this head. The
                            // dock has forgotten it, so vino has to put it back itself.
                            // Do not re-drive the head here. Most of these blips heal on their own
                            // -- the dock brings the sink back within a second or two -- and a
                            // repair costs a full dock-wide re-activation, four seconds of cold
                            // choreography for both panels. Firing one per flap puts the dock into
                            // a permanent re-activation loop, one every five to fifteen seconds,
                            // and neither panel stays lit. Absorbing the blip is
                            // the whole point; a drop that does *not* heal still falls through to
                            // the timed removal below.
                            if present && head_absent_since[h].take().is_some() {
                                vino_dev_debug!(
                                    cdev,
                                    "socket {socket} sink flap healed on its own\n"
                                );
                            }
                            continue;
                        }
                        // Inside the settling window after a recovery, a negative answer is not
                        // evidence -- see `PRESENCE_GRACE`.
                        if !present
                            && (Instant::<Monotonic>::now() - presence_grace[h]).as_millis() < 0
                        {
                            head_debounce[h] = 0;
                            head_absent_since[h] = None;
                            continue;
                        }
                        if present {
                            // Two consecutive contrary reads before announcing an arrival.
                            head_absent_since[h] = None;
                            head_debounce[h] = head_debounce[h].saturating_add(1);
                            if head_debounce[h] < 2 {
                                continue;
                            }
                        } else {
                            // A removal must be sustained: the dock reports a lit sink absent for
                            // seconds at a time. Counting probes instead of time does not work --
                            // every `id=0x44` reply sets the downstream-event flag, and a probe's
                            // own reply is one, so the watcher pulls itself forward and "two
                            // contrary reads" fires 132 ms after the first negative.
                            if !sustained_absent(&mut head_absent_since[h]) {
                                continue;
                            }
                            head_absent_since[h] = None;
                        }
                        head_debounce[h] = 0;
                        if present {
                            // Re-engage the downstream sink before accepting another mode set.
                            match data.reengage_head(dev, h as u8) {
                                Ok(true) => {}
                                Ok(false) => {
                                    next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                                    continue;
                                }
                                Err(e) => {
                                    vino_dev_debug!(
                                        cdev,
                                        "socket {socket} sink re-engagement failed ({e:?})\n"
                                    );
                                    next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                                    continue;
                                }
                            }
                            data.set_connected(h);
                            head_known[h] = true;
                            next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                            presence_grace[h] = Instant::<Monotonic>::now() + PRESENCE_GRACE;
                            dev_info!(cdev, "socket {socket} monitor connected\n");
                            // Same downstream readiness wait as a fresh bring-up before notifying
                            // userspace, so KWin's mode-set lands on a settled downstream link.
                            let rs = Instant::<Monotonic>::now();
                            while (Instant::<Monotonic>::now() - rs).as_millis() < 1300
                                && !data.is_shutting_down()
                            {
                                let _ = data
                                    .send_cp(dev, 0x14, 0, |ctr| cp::device_query_req(ctr, 0x000c));
                                fsleep(Delta::from_millis(15));
                            }
                            hotplug_due = Some(Instant::<Monotonic>::now() + HOTPLUG_COALESCE);
                        } else {
                            head_known[h] = false;
                            next_reengage[h] = Instant::<Monotonic>::now() + REENGAGE_RETRY;
                            data.set_disconnected(h);
                            dev_info!(cdev, "socket {socket} monitor disconnected\n");
                            // This event also covers any arrival still waiting out its window.
                            hotplug_due = None;
                            drm_dev.hotplug_event();
                        }
                        // Re-baseline the heartbeat/presence deadlines skipped during the wait.
                        next_presence = Instant::<Monotonic>::now() + PRESENCE_PERIOD;
                    }
                }
                // Announce a settled burst of arrivals as one topology change.
                if let Some(due) = hotplug_due {
                    if (Instant::<Monotonic>::now() - due).as_millis() >= 0 {
                        hotplug_due = None;
                        drm_dev.hotplug_event();
                    }
                }
                // A dock that tears the link down over a silent video endpoint needs feeding even
                // when the compositor has nothing to redraw.
                data.send_video_keepalive(dev);
                fsleep(Delta::from_millis(13));
            }
            vino_dev_debug!(cdev, "CP keepalive finished ({sent} polls)\n");
        }
    }
}

/// Control-session bring-up: plaintext init, link AKE, and the sealed per-connector setup.
mod session;

kernel::usb_device_table!(
    USB_TABLE,
    MODULE_USB_TABLE,
    <VinoDriver as usb::Driver>::IdInfo,
    [
        (
            usb::DeviceId::from_id(VID_DISPLAYLINK, PID_D6000),
            &PROFILE_D6000
        ),
        (
            usb::DeviceId::from_id(VID_DISPLAYLINK, PID_DL7400),
            &PROFILE_DL7400
        ),
    ]
);

impl usb::Driver for VinoDriver {
    type IdInfo = &'static DockProfile;
    type Data<'bound> = VinoBoundData;
    const ID_TABLE: usb::IdTable<Self::IdInfo> = &USB_TABLE;

    fn probe<'bound>(
        intf: &'bound usb::Interface<Core<'_>>,
        _id: &usb::DeviceId,
        info: &'bound Self::IdInfo,
        io: Arc<usb::IoWindow>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
        let cdev: &device::Device<Core<'_>> = intf.as_ref();
        // The D6000 exposes several interfaces (0/1/5/6 match us; 2-4 are audio).
        // The control endpoints (0x02/0x84) and the whole HDCP session live on
        // interface 0 -- drive bring-up only there so we don't run the preamble and
        // AKE four times and pollute the dock's state machine. Other interfaces
        // bind (so usbcore doesn't hand them to another driver) but stay idle.
        // An interface with no active alternate setting has no endpoints to drive.
        let ifnum = intf.number().ok_or(ENODEV)?;
        log_device_identity(cdev, intf, ifnum);
        if ifnum == 0 {
            // One line per bind, naming the hardware the driver decided it is holding. On
            // unfamiliar hardware this is what says whether the dock was recognised or fell back
            // to a stranger's profile, so it stays out of the debug gate. The endpoint map that
            // follows from it is a debug detail.
            dev_info!(cdev, "{}\n", info.name);
            vino_dev_debug!(
                cdev,
                "video endpoints {}, 10-bit capable {}\n",
                HexList(&info.video_eps),
                info.hdr_capable
            );
        }
        // Writing firmware the dock does not need is a deliberate act: its DFU interface does
        // not support upload, so there is no way to read the running image back and nothing to
        // restore from if the write goes wrong.
        let force = *crate::module_parameters::force_flash.value() != 0;
        if ifnum == firmware::DFU_INTERFACE {
            // The dock's DFU interface, which is also the one every DFU request is addressed to.
            // Reading the identity here costs one standard control transfer and is what says
            // whether an update is due at all; a failure is not fatal, because a dock runs
            // perfectly well on the firmware it shipped with.
            match io.enter().and_then(|link| {
                let id = firmware::read_identity(&link)?;
                dev_info!(cdev, "{id} running firmware {}\n", id.version);
                firmware::update_if_newer(&link, cdev, &id, u16::from(ifnum), force)
            }) {
                Ok(()) => {}
                Err(e) => dev_info!(cdev, "dock firmware check skipped ({e:?})\n"),
            }
        }
        if ifnum != 0 {
            // Keep the app-specific interface paired with the display function. Let the audio
            // (2-4) and Ethernet (5-6) interfaces fall through to their class drivers. Returning
            // ENODEV tells usbcore that this driver does not handle the interface.
            if ifnum != 1 {
                vino_dev_debug!(cdev, "declining interface {ifnum} (left to its class driver)\n");
                return Err(ENODEV);
            }
            vino_dev_debug!(cdev, "bound interface {ifnum} (idle -- control is iface 0)\n");
            return Ok(VinoBoundData {
                _intf: intf.into(),
                registration: None,
                bringup: KBox::pin_init(new_mutex!(None), GFP_KERNEL)?,
            });
        }
        vino_dev_debug!(cdev, "bound DisplayLink control interface\n");
        // Register the DRM/KMS device on the control interface. Keep a refcounted interface handle
        // in the bound data while the DRM device retains the I/O window used by its workers.
        let intf_ref: ARef<usb::Interface> = intf.into();

        // Resolve the dock's endpoints against interface 0's descriptor once, so every later
        // transfer names a direction/type-checked endpoint instead of a bare address.
        let eps = Endpoints::resolve(intf, info)?;

        // DRM device lifecycle: allocate an `UnregisteredDevice`, wire up the KMS pipeline on it
        // while still unregistered, then register it. The `Registration` is stored in the bound
        // data below, so the card is unregistered by the ordered unbind rather than by a
        // driver-local force-unplug.
        let unreg = drm::UnregisteredDevice::<drm_sink::VinoDrmDriver>::new(
            intf,
            // The ten-bit flag has to arrive here, not in the profile block below: the KMS objects
            // are built during this call, and they decide then whether to offer a 10-bit format
            // and the HDR connector properties.
            drm_sink::VinoDrmData::new(io.clone(), eps, info.hdr_capable),
            &THIS_MODULE,
        )?;
        // `Core` derefs to `Bound`; name the context explicitly so `as_ref()`
        // resolves to the bound parent required by DRM registration.
        let bound_intf: &usb::Interface<device::Bound> = intf;
        let parent: &device::Device<device::Bound> = bound_intf.as_ref();
        let registration = drm::Registration::new_static(parent, unreg, (), 0)?;
        let ddev: ARef<drm_sink::VinoDrmDevice> = registration.device().into();
        vino_dev_debug!(cdev, "DRM/KMS device registered\n");

        // The session preamble, HDCP authentication and control setup use blocking USB transfers.
        // Run them on the device's ordered session queue so probe can return immediately. The work
        // item owns the DRM device, and the bound data retains a handle so quiesce can cancel or
        // flush it before the I/O window closes.
        // Gate video on what this platform's video path is known to accept.
        {
            let d: &drm_sink::VinoDrmData = &ddev;
            // `force_video` exists to answer one question on a dock whose profile disables video:
            // whether the platform actually requires its sealed stream-open, or whether correct
            // record framing alone is enough. It is off by default because the way a dock rejects
            // a malformed video write is to reset itself, taking the control session with it.
            // This device's codec geometry, passed into every codec call made on its behalf.
            // It is per device because two docks of different generations lay a strip's sixteen
            // blocks over different pixels; see `video::wht::Geometry`.
            d.set_codec_geometry(
                info.strip_blocks_x,
                info.interlaced_bands,
                info.band_parity_bit,
                info.head_sub_shift,
                info.stream_id_mask,
                info.dock_buffers,
            );
            d.set_mode_limits(
                info.pixel_budget,
                info.max_refresh_hz,
                info.max_head_clock_khz,
            );
            d.set_navarro(info.is_navarro());
            d.set_connectors(info.connectors);
        }
        let bringup = BringUp::new(ddev.clone(), info)?;
        let bringup_slot = KBox::pin_init(new_mutex!(Some(bringup.clone())), GFP_KERNEL)?;

        let data: &drm_sink::VinoDrmData = &ddev;
        data.session_queue().enqueue(bringup).map_err(|_| EBUSY)?;

        Ok(VinoBoundData {
            _intf: intf_ref,
            registration: Some(registration),
            bringup: bringup_slot,
        })
    }

    fn quiesce<'bound>(_intf: &'bound usb::Interface<Core<'_>>, data: Pin<&VinoBoundData>) {
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

        // `bringup` drops here, releasing its DRM reference before I/O is revoked.
    }

    fn disconnect<'bound>(intf: &'bound usb::Interface<Core<'_>>, data: Pin<&VinoBoundData>) {
        let dev: &device::Device<Core<'_>> = intf.as_ref();

        // Stop every producer. The DRM device itself is unregistered by `Registration`'s `Drop`
        // when the bound data is released -- the accepted registration teardown already calls
        // `drm_dev_unplug()`, so there is no driver-local force-unplug here any more.
        //
        // Take the device through the registration. `shutdown()` also breaks
        // vblank self-references before registration teardown.
        if let Some(reg) = data.registration.as_ref() {
            let drm_data: &drm_sink::VinoDrmData = reg.device();
            drm_data.shutdown();
        }
        dev_info!(dev, "disconnected\n");
    }
}

kernel::module_usb_driver! {
    type: VinoDriver,
    name: "vino",
    authors: ["Mike Lothian"],
    description: "DisplayLink DL3 (Vino) open driver",
    license: "GPL v2",
    params: {
        debug: u8 {
            default: 0,
            description: "Enable verbose Vino protocol and scanout diagnostics",
        },
        trace_crypto: u8 {
            default: 0,
            description: "Diagnostic: disclose ephemeral control/video keys and nonces for one decryptable usbmon capture",
        },
        rtc_utc_offset_minutes: i32 {
            default: 0,
            description: "Local UTC offset used by Navarro RTC synchronization (minutes east of UTC)",
        },
        force_flash: u8 {
            default: 0,
            description: "Write the packaged dock firmware even when the dock already runs that version or newer. The DFU interface does not support upload, so there is no host-side restore path and an interrupted write leaves a partial image. For recovering a damaged dock, not for normal use",
        },
        edid_override: u8 {
            default: 0,
            description: "Bitmask of heads whose sink is described by DRM's EDID override (drm_kms_helper.edid_firmware=<connector>:<path>, or a debugfs edid_override write) because the dock cannot read its EDID -- typically a DP-to-HDMI converter that mangles DDC",
        },
    },
}

/// Offline self-tests for the pure protocol builders/parsers and crypto bindings the control plane
/// relies on. `CONFIG_DRM_VINO_KUNIT_TEST` keeps them out of an ordinary driver build, while
/// allowing a KUnit test kernel to run the published known-answer vectors and byte-exact wire
/// checks when the module is loaded.
#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
mod tests;
