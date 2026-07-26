// SPDX-License-Identifier: GPL-2.0
//
// The EVDI DRM/KMS device: registers a `struct drm_device` presenting one virtual
// display head (CRTC + primary plane + virtual encoder + virtual connector) with
// GEM-shmem dumb buffers, built on the safe KMS mode-object layer (`kernel::drm::kms`).
//
// Unlike a real display driver, EVDI's scanout is *pulled* by userspace: the
// DisplayLinkManager daemon grabs framebuffer pixels via the GRABPIX ioctl and is
// told when to do so through `drm_event`s (see `painter.rs`). The KMS callbacks here
// therefore translate atomic commits into those events rather than programming any
// hardware.

use core::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU32, Ordering};
use kernel::{
    bindings, drm,
    drm::event::{EventChannel, EventConnection},
    drm::kms::{
        connector::{self, AsRawConnector as _, ConnectorGuard, ModeStatus},
        crtc::{self, AsRawCrtc as _, CrtcAtomicCommit, RawCrtc as _, RawCrtcState as _},
        encoder,
        framebuffer::Framebuffer,
        modes::DisplayMode,
        plane::{self, PlaneAtomicCommit, RawPlaneState as _},
        vblank::{RawVblankCrtcState as _, VblankGuard, VblankSupport, VblankTimestamp},
        KmsDriver, ModeConfigGuard, ModeConfigInfo, ModeObject as _, UnregisteredKmsDevice,
    },
    impl_has_hr_timer,
    interrupt::LocalInterruptDisabled,
    prelude::*,
    sync::{
        aref::ARef, new_condvar, new_mutex, new_spinlock, Arc, ArcBorrow, CondVar,
        CondVarTimeoutResult, Mutex, SpinLock,
    },
    time::{
        hrtimer::{
            ArcHrTimerHandle, HasHrTimer, HrTimer, HrTimerCallback, HrTimerCallbackContext,
            HrTimerPointer, HrTimerRestart, RelativeMode,
        },
        Delta, Monotonic,
    },
};

use crate::painter::PainterState;
use crate::uapi;
use kernel::error::code::{ENODEV, ERESTARTSYS, ETIMEDOUT};

/// DDC/CI slave address on the virtual I2C bus (as used by monitor-control tools).
pub(crate) const DDCCI_ADDRESS: u16 = 0x37;
/// How long a DDC/CI transfer waits for the userspace client's reply.
const DDCCI_TIMEOUT_MS: u32 = 50;
/// Maximum DDC/CI payload carried in one event.
const DDCCI_BUFFER_SIZE: usize = 64;

/// `DRM_FORMAT_XRGB8888` (`fourcc_code('X','R','2','4')`) — EVDI scans out 32bpp.
pub(crate) const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258;
static PRIMARY_FORMATS: [u32; 1] = [DRM_FORMAT_XRGB8888];

/// Fallback mode advertised before the DLM client delivers an EDID via CONNECT.
/// Multiplier applied to the DLM client's raw pixel-rate limit before `mode_valid` enforces it --
/// see [`EvdiDrmData::set_mode_limits`] for why the raw figure is an under-estimate. `1` restores
/// the previous behaviour exactly (and caps this dock at 1440p@120); raise it to offer higher-rate
/// modes to the compositor.
const BANDWIDTH_HEADROOM: u32 = 2;

const FALLBACK_W: u32 = 1024;
const FALLBACK_H: u32 = 768;

// libevdi (`is_evdi_compatible`) requires `major == 1 && minor >= 9`; match the C evdi's version so
// DisplayLinkManager accepts the card. DLM 6.8.1.0 checks the reported version against upstream evdi
// 1.15.0 (commit 2713cd41) and rejects anything older, so keep this in lockstep with libevdi.
const INFO: drm::DriverInfo = drm::DriverInfo {
    major: 1,
    minor: 15,
    patchlevel: 0,
    name: c"evdi",
    desc: c"Extensible Virtual Display Interface",
};

/// The EVDI DRM driver marker type.
pub(crate) struct EvdiDrmDriver;

/// Convenience alias for our concrete `drm::Device`.
pub(crate) type EvdiDrmDevice = drm::Device<EvdiDrmDriver>;

/// DRM device-private data: the painter (connection + event delivery + pixel grab
/// bookkeeping) and a stashed pointer to our single connector so CONNECT can push a
/// fresh EDID into its mode list.
#[pin_data]
pub(crate) struct EvdiDrmData {
    /// Event channel to the connected DLM client (`drm_event` delivery).
    pub(crate) events: Arc<EventChannel<EvdiDrmDriver, EvdiDrmFile>>,
    /// Painter state (connection status, cached EDID, cursor-events flag, dirty rects).
    #[pin]
    pub(crate) painter: Mutex<PainterState>,
    /// The framebuffer currently flipped in on the primary plane, refcounted so GRABPIX can map
    /// and copy it after the atomic commit that set it has returned. `None` until the first flip.
    #[pin]
    pub(crate) scanout: Mutex<Option<ARef<Framebuffer<EvdiDrmDriver>>>>,
    /// Slot for a DDC/CI reply from the client, guarding the request/response handshake between
    /// the I2C `master_xfer` (which waits) and the DDCCI_RESPONSE ioctl (which fills + notifies).
    #[pin]
    pub(crate) ddcci_resp: Mutex<Option<KVec<u8>>>,
    #[pin]
    pub(crate) ddcci_cv: CondVar,
    /// Pointer to our connector's driver data, stashed during `KmsDriver::probe` so
    /// CONNECT can reach its cached-EDID slot without walking DRM's mode-object list.
    /// Written once during single-threaded probe; read-only thereafter.
    pub(crate) connector: core::sync::atomic::AtomicPtr<EvdiConnector>,
    /// Set (once, never cleared) when the card is being torn down, so blocking paths bail out
    /// with `ENODEV` instead of stalling the teardown — e.g. a DDC/CI transfer waiting out its
    /// full timeout would otherwise hold up `i2c_del_adapter` on unbind.
    pub(crate) dying: AtomicBool,
}

impl EvdiDrmData {
    pub(crate) fn new() -> impl PinInit<Self, Error> {
        try_pin_init!(Self {
            events: EventChannel::new()?,
            painter <- new_mutex!(PainterState::new()),
            scanout <- new_mutex!(None),
            ddcci_resp <- new_mutex!(None),
            ddcci_cv <- new_condvar!(),
            connector: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
            dying: AtomicBool::new(false),
        })
    }

    /// Begin card shutdown: mark the device dying and wake every in-kernel waiter so teardown
    /// (platform unbind → I2C adapter + DRM unregistration) cannot stall behind them. Called
    /// before the rest of the bound data drops; safe to call more than once.
    pub(crate) fn shutdown(&self) {
        self.dying.store(true, Ordering::Relaxed);
        self.ddcci_cv.notify_all();
    }

    /// Send one DDC/CI message to the connected client and wait for its reply. A write buffer is
    /// copied into the event; a read buffer is filled from the reply, up to its length.
    pub(crate) fn ddcci_transfer(
        &self,
        addr: u16,
        flags: u16,
        buffer: kernel::i2c::MsgBuffer<'_>,
    ) -> Result {
        if self.dying.load(Ordering::Relaxed) {
            return Err(ENODEV);
        }
        let (is_read, len) = match &buffer {
            kernel::i2c::MsgBuffer::Write(buf) => (false, buf.len()),
            kernel::i2c::MsgBuffer::Read(buf) => (true, buf.len()),
        };
        let mut ev = uapi::DrmEvdiEventDdcciData {
            base: uapi::DrmEvent {
                type_: 0,
                length: 0,
            },
            buffer: [0u8; DDCCI_BUFFER_SIZE],
            buffer_length: len as u32,
            flags,
            address: addr,
        };
        if let kernel::i2c::MsgBuffer::Write(buf) = &buffer {
            let n = core::cmp::min(buf.len(), DDCCI_BUFFER_SIZE);
            ev.buffer[..n].copy_from_slice(&buf[..n]);
        }

        // Hold the slot lock across send+wait so the reply cannot be signalled before we wait
        // (no lost wakeup): `ddcci_respond` can only run once `wait_*` has released the lock.
        let mut slot = self.ddcci_resp.lock();
        *slot = None;
        self.events.send(ev)?;
        // Condvar waits can wake spuriously, so loop until the reply actually arrives,
        // carrying the remaining timeout. A wake landing exactly at expiry is reported as
        // `Timeout`, so re-check the slot once before giving up.
        let mut jiffies = kernel::time::msecs_to_jiffies(DDCCI_TIMEOUT_MS);
        while slot.is_none() {
            if self.dying.load(Ordering::Relaxed) {
                return Err(ENODEV);
            }
            match self.ddcci_cv.wait_interruptible_timeout(&mut slot, jiffies) {
                CondVarTimeoutResult::Woken { jiffies: remaining } => jiffies = remaining,
                CondVarTimeoutResult::Timeout => {
                    if slot.is_some() {
                        break;
                    }
                    return Err(ETIMEDOUT);
                }
                CondVarTimeoutResult::Signal { .. } => return Err(ERESTARTSYS),
            }
        }
        if is_read {
            if let kernel::i2c::MsgBuffer::Read(buf) = buffer {
                if let Some(resp) = slot.take() {
                    let m = core::cmp::min(resp.len(), buf.len());
                    buf[..m].copy_from_slice(&resp[..m]);
                }
            }
        }
        Ok(())
    }

    /// Store a DDC/CI reply from the client (DDCCI_RESPONSE ioctl) and wake the waiting transfer.
    pub(crate) fn ddcci_respond(&self, resp: KVec<u8>) {
        *self.ddcci_resp.lock() = Some(resp);
        self.ddcci_cv.notify_one();
    }

    /// Record the framebuffer currently on the primary plane (bumping its refcount) so a later
    /// GRABPIX can map it. Called from the plane's atomic commit.
    pub(crate) fn set_scanout(&self, fb: Option<&Framebuffer<EvdiDrmDriver>>) {
        *self.scanout.lock() = fb.map(ARef::from);
    }

    /// Take a refcounted handle to the current scanout framebuffer, if any, for GRABPIX to map.
    pub(crate) fn scanout_fb(&self) -> Option<ARef<Framebuffer<EvdiDrmDriver>>> {
        self.scanout.lock().clone()
    }

    /// Install a new EDID blob (from CONNECT) into the connector and fire a hotplug so
    /// the compositor re-probes the connector's mode list.
    pub(crate) fn set_edid(&self, dev: &EvdiDrmDevice, blob: KVec<u8>) {
        let ptr = self.connector.load(core::sync::atomic::Ordering::Acquire);
        // SAFETY: `connector` was stashed during probe and points to a `EvdiConnector`
        // that lives as long as the device; null until probe completes.
        let Some(connector) = (unsafe { ptr.as_ref() }) else {
            return;
        };
        *connector.cached_edid.lock() = Some(blob);
        dev.hotplug_event();
    }

    /// Drop the connector's cached EDID (on CONNECT disconnect) and fire a hotplug so the connector
    /// reports disconnected again -- see [`EvdiConnector::detect`].
    pub(crate) fn clear_edid(&self, dev: &EvdiDrmDevice) {
        let ptr = self.connector.load(core::sync::atomic::Ordering::Acquire);
        // SAFETY: as in `set_edid` -- `connector` was stashed during probe and outlives the device.
        let Some(connector) = (unsafe { ptr.as_ref() }) else {
            return;
        };
        *connector.cached_edid.lock() = None;
        dev.hotplug_event();
    }

    /// Store the dock bandwidth limits the client supplied via CONNECT, for
    /// [`EvdiConnector`]'s `mode_valid` to enforce. Must be called before the EDID is
    /// published (`set_edid`'s hotplug re-probes the mode list against these limits).
    ///
    /// ★ 2026-07-26: the raw figure is scaled by [`BANDWIDTH_HEADROOM`] before being stored.
    /// DLM supplies `pixels_per_second` as `sku_area_limit * 60` -- for this panel exactly
    /// **442,368,000**, i.e. one 2560x1440@120 -- and it is a RAW pixel-rate proxy that takes no
    /// account of DisplayLink's on-wire compression. Stored unscaled it made `mode_valid` admit
    /// @120 (which compares equal) and reject **every** higher mode, so a 1440p165/180 monitor was
    /// advertised to the compositor as a 60/85/120 monitor with no indication why. That is not the
    /// EDID being trimmed -- the full 384-byte blob including its DisplayID block reaches
    /// `add_edid_modes` intact -- it is purely this ceiling.
    ///
    /// The scaling matches the standalone `revdi/module/kms.rs`, which has carried it since the
    /// dual-head work: macOS drives dual 1440p@120 on this same dock, so the real compressed USB-3
    /// bandwidth demonstrably has headroom the raw proxy does not credit.
    ///
    /// ⚠ This governs only what is **offered**. Whether the dock will actually light a given mode
    /// is a separate question -- vino sent 1440p165 and 1440p180 with correct clocks and the panel
    /// stayed dark (`docs/HIGH-REFRESH.md`). Raising this makes the modes selectable so that
    /// question can be asked; it does not answer it.
    pub(crate) fn set_mode_limits(&self, pixel_area: u32, pixels_per_second: u32) {
        let ptr = self.connector.load(core::sync::atomic::Ordering::Acquire);
        // SAFETY: as in `set_edid` -- `connector` was stashed during probe and outlives the device.
        let Some(connector) = (unsafe { ptr.as_ref() }) else {
            return;
        };
        let pixels_per_second = pixels_per_second.saturating_mul(BANDWIDTH_HEADROOM);
        pr_info!(
            "evdi: mode limits -- area {pixel_area}, pixel rate {pixels_per_second}/s \
             (client supplied {}/s x{BANDWIDTH_HEADROOM} compression headroom)\n",
            pixels_per_second / BANDWIDTH_HEADROOM
        );
        connector
            .pixel_area_limit
            .store(pixel_area, Ordering::Relaxed);
        connector
            .pixel_per_second_limit
            .store(pixels_per_second, Ordering::Relaxed);
    }
}

/// GEM object inner data. Empty: the shmem-backed object wires
/// `drm_gem_shmem_dumb_create`, so `DRM_IOCTL_MODE_CREATE_DUMB` works and the GRABPIX
/// ioctl can `vmap` the resulting framebuffer to copy pixels to userspace.
#[pin_data]
pub(crate) struct EvdiObject {}

impl drm::gem::DriverObject for EvdiObject {
    type Driver = EvdiDrmDriver;
    type Args = ();

    fn new(
        _dev: &drm::Device<EvdiDrmDriver>,
        _size: usize,
        _args: (),
    ) -> impl PinInit<Self, Error> {
        try_pin_init!(EvdiObject {})
    }
}

/// Per-open DRM client state. Empty of data, but its lifetime pins the module for the
/// duration of an open DRM file (the DLM daemon holds `/dev/dri/cardN` open across the
/// whole session; the Rust DRM `file_operations` are built with `owner = NULL`, so
/// without this an `rmmod` while DLM is running would free the fops under the open fd).
#[pin_data(PinnedDrop)]
pub(crate) struct EvdiDrmFile {
    /// The owning device, so file close can reach the event channel to disconnect.
    dev: ARef<EvdiDrmDevice>,
    /// The connection token is owned by the file, so closing or disconnecting drops it and
    /// automatically detaches this receiver from the channel.
    #[pin]
    pub(crate) connection: Mutex<Option<EventConnection<EvdiDrmDriver, EvdiDrmFile>>>,
}

impl drm::file::DriverFile for EvdiDrmFile {
    type Driver = EvdiDrmDriver;

    fn open(dev: &drm::Device<Self::Driver>) -> Result<Pin<KBox<Self>>> {
        let file = KBox::try_pin_init(
            try_pin_init!(Self {
                dev: dev.into(),
                connection <- new_mutex!(None),
            }),
            GFP_KERNEL,
        )?;
        // SAFETY: executing inside this module's own DRM `open` callback, so the module
        // is live; the reference is released 1:1 in `PinnedDrop`.
        unsafe { kernel::bindings::__module_get(crate::THIS_MODULE.as_ptr()) };
        Ok(file)
    }
}

#[pinned_drop]
impl PinnedDrop for EvdiDrmFile {
    fn drop(self: Pin<&mut Self>) {
        // Release the module reference taken in `open`.
        // SAFETY: balances the `__module_get` in `open`.
        unsafe { kernel::bindings::module_put(crate::THIS_MODULE.as_ptr()) };
    }
}

#[vtable]
impl drm::Driver for EvdiDrmDriver {
    type Data = EvdiDrmData;
    type File = EvdiDrmFile;
    type Object = drm::gem::shmem::Object<EvdiObject>;
    type ParentDevice<Ctx: kernel::device::DeviceContext> = kernel::platform::Device<Ctx>;
    type RegistrationData<'a> = ();
    type Kms = Self;

    const INFO: drm::DriverInfo = INFO;

    kernel::declare_drm_ioctls_ext! {
        (EVDI_CONNECT, crate::uapi::DrmEvdiConnect,
            crate::uapi::DRM_IOCTL_EVDI_CONNECT, 0, crate::ioctl::connect),
        (EVDI_REQUEST_UPDATE, crate::uapi::DrmEvdiRequestUpdate,
            crate::uapi::DRM_IOCTL_EVDI_REQUEST_UPDATE, 0, crate::ioctl::request_update),
        (EVDI_GRABPIX, crate::uapi::DrmEvdiGrabpix,
            crate::uapi::DRM_IOCTL_EVDI_GRABPIX, 0, crate::ioctl::grabpix),
        (EVDI_DDCCI_RESPONSE, crate::uapi::DrmEvdiDdcciResponse,
            crate::uapi::DRM_IOCTL_EVDI_DDCCI_RESPONSE, 0, crate::ioctl::ddcci_response),
        (EVDI_ENABLE_CURSOR_EVENTS, crate::uapi::DrmEvdiEnableCursorEvents,
            crate::uapi::DRM_IOCTL_EVDI_ENABLE_CURSOR_EVENTS, 0, crate::ioctl::enable_cursor_events),
    }
}

#[vtable]
impl KmsDriver for EvdiDrmDriver {
    type Connector = EvdiConnector;
    type Plane = EvdiPlane;
    type Crtc = EvdiCrtc;
    type Encoder = EvdiEncoder;

    fn mode_config_info(
        _dev: &kernel::device::Device,
        _drm_data: &Self::Data,
    ) -> Result<ModeConfigInfo> {
        Ok(ModeConfigInfo {
            min_resolution: (0, 0),
            max_resolution: (8192, 8192),
            max_cursor: (64, 64),
            preferred_depth: 32,
            preferred_fourcc: Some(DRM_FORMAT_XRGB8888),
        })
    }

    fn create_objects(dev: &UnregisteredKmsDevice<'_, Self>) -> Result {
        let primary = plane::UnregisteredPlane::<EvdiPlane>::new(
            dev,
            1,
            &PRIMARY_FORMATS,
            None,
            plane::Type::Primary,
            None,
            (),
        )?;
        // Advertise FB_DAMAGE_CLIPS so the compositor reports which region it changed each frame.
        // Without it `damage_merged` always returns the whole plane, so GRABPIX reports a full-frame
        // rect and DLM re-transmits the entire 2560x1440 buffer every update (jerky).
        // SAFETY: `primary` is a valid, not-yet-registered plane; the property is added before
        // registration (in the `Probing` phase).
        unsafe {
            kernel::bindings::drm_plane_enable_fb_damage_clips(plane::AsRawPlane::as_raw(primary))
        };
        // Keep cursor composition in the primary framebuffer for now. Creating a GEM handle for
        // a separate control client's cursor event may sleep, while a drm_file has no independent
        // refcount that a safe asynchronous channel can retain. Advertising no cursor plane avoids
        // reintroducing the erased raw-file lifetime protocol removed from EventChannel.
        let crtc_obj = crtc::UnregisteredCrtc::<EvdiCrtc>::new(
            dev,
            primary,
            None::<&plane::UnregisteredPlane<EvdiPlane>>,
            None,
            (),
        )?;
        let enc = encoder::UnregisteredEncoder::<EvdiEncoder>::new(
            dev,
            encoder::Type::Virtual,
            crtc_obj.mask(),
            0,
            None,
            (),
        )?;
        // Use DVI-I (matching the C evdi) rather than Virtual: `__drm_connector_init` skips
        // `drm_connector_attach_edid_property()` for VIRTUAL/WRITEBACK connectors, and without that
        // property `drm_edid_connector_update()` can't populate `edid_blob_ptr`, so
        // `drm_edid_connector_add_modes()` would return 0 modes for a perfectly valid EDID.
        let conn =
            connector::UnregisteredConnector::<EvdiConnector>::new(dev, connector::Type::DviI, ())?;
        conn.attach_encoder(&*enc)?;
        let data: &EvdiDrmData = dev;
        data.connector.store(
            &**conn as *const EvdiConnector as *mut EvdiConnector,
            core::sync::atomic::Ordering::Release,
        );
        Ok(())
    }
}

// ---- CRTC -------------------------------------------------------------------

/// A software vblank source: an hrtimer that fires once per frame and drives
/// `drm_crtc_handle_vblank()`, so the atomic helpers pace page-flips against a real vblank
/// (via `drm_crtc_arm_vblank_event()` in [`EvdiCrtc::atomic_flush`]) instead of completing them
/// immediately with a fake vblank -- which is what makes updates smooth rather than bursty.
///
/// The timer stops itself when vblank is disabled (mirroring the C core's
/// `drm_vblank_timer_function`, which returns `HRTIMER_NORESTART` on a zeroed interval): the
/// callback sees `enabled == false` and returns [`HrTimerRestart::NoRestart`], so an idle or
/// DPMS-off output costs no wakeups. `enable_vblank` re-arms it with a raw
/// [`HasHrTimer::start`], which re-queues the timer whether it is dead or still pending —
/// no new handle is minted, so the single [`ArcHrTimerHandle`] taken at first start remains
/// the sole owner and its drop (at CRTC teardown, before the `drm_crtc` is freed) is the only
/// full `hrtimer_cancel`. Neither enable nor disable ever blocks on the callback, which is
/// what makes them deadlock-free against `drm_crtc_handle_vblank` (see the deadlock note in
/// `drm_crtc_vblank_cancel_timer`).
#[pin_data]
pub(crate) struct VblankTimer {
    #[pin]
    timer: HrTimer<Self>,
    /// The `drm_crtc` to deliver vblanks to (set when vblank is first enabled).
    crtc: AtomicPtr<bindings::drm_crtc>,
    /// One scanout frame in nanoseconds (from the mode's `framedur_ns`).
    interval_ns: AtomicI64,
    /// Whether vblanks should currently be delivered (toggled by enable/disable_vblank).
    enabled: AtomicBool,
}

impl VblankTimer {
    fn new() -> impl PinInit<Self> {
        pin_init!(VblankTimer {
            timer <- HrTimer::new(),
            crtc: AtomicPtr::new(core::ptr::null_mut()),
            interval_ns: AtomicI64::new(16_666_666), // ~60 Hz until a mode sets it
            enabled: AtomicBool::new(false),
        })
    }
}

impl HrTimerCallback for VblankTimer {
    type Pointer<'a> = Arc<Self>;

    fn run(this: ArcBorrow<'_, Self>, mut ctx: HrTimerCallbackContext<'_, Self>) -> HrTimerRestart {
        // Vblank is off: let the timer die instead of ticking uselessly; `enable_vblank`
        // re-arms it. A concurrent re-arm racing this return is safe — hrtimer keeps a timer
        // that was re-queued during its callback enqueued even on NORESTART.
        if !this.enabled.load(Ordering::Relaxed) {
            return HrTimerRestart::NoRestart;
        }
        let crtc = this.crtc.load(Ordering::Relaxed);
        if !crtc.is_null() {
            // SAFETY: `crtc` is the `drm_crtc` stored in `enable_vblank` while the device is live;
            // the timer is cancelled (its handle dropped) before the crtc is freed at teardown.
            unsafe { bindings::drm_crtc_handle_vblank(crtc) };
        }
        let interval = this.interval_ns.load(Ordering::Relaxed).max(1_000_000);
        ctx.forward_now(Delta::from_nanos(interval));
        HrTimerRestart::Restart
    }
}

impl_has_hr_timer! {
    impl HasHrTimer<Self> for VblankTimer {
        mode: RelativeMode<Monotonic>, field: self.timer
    }
}

#[pin_data]
pub(crate) struct EvdiCrtc {
    /// The software vblank source for this CRTC.
    vblank: Arc<VblankTimer>,
    /// Sole owner of the started timer; dropping it (at CRTC teardown) is the only full
    /// `hrtimer_cancel`. `None` until vblank is first enabled.
    #[pin]
    vblank_handle: SpinLock<Option<ArcHrTimerHandle<VblankTimer>>>,
}

#[derive(Clone, Default)]
pub(crate) struct EvdiCrtcState;

impl crtc::DriverCrtcState for EvdiCrtcState {
    type Crtc = EvdiCrtc;
}

#[vtable]
impl crtc::DriverCrtc for EvdiCrtc {
    type Args = ();
    type Driver = EvdiDrmDriver;
    type State = EvdiCrtcState;
    type VblankImpl = Self;

    fn new(_device: &drm::Device<Self::Driver>, _args: &()) -> impl PinInit<Self, Error> {
        try_pin_init!(EvdiCrtc {
            vblank: Arc::pin_init(VblankTimer::new(), GFP_KERNEL)?,
            vblank_handle <- new_spinlock!(None),
        })
    }

    /// Display turning on: enable vblank delivery, then tell the DLM client DPMS-on + the mode.
    fn atomic_enable(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        crtc.vblank_on();
        let dev = crtc.drm_dev();
        let data: &EvdiDrmData = dev;
        let new = commit.take_new_state();
        let mode = new.mode();
        crate::painter::notify_dpms(data, dev, crate::painter::DPMS_ON);
        crate::painter::notify_mode_changed(
            data,
            dev,
            mode.hdisplay() as i32,
            mode.vdisplay() as i32,
            mode.vrefresh(),
            32,
            DRM_FORMAT_XRGB8888,
        );
    }

    /// Display turning off: stop vblank delivery and tell the DLM client DPMS-off.
    fn atomic_disable(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        crtc.vblank_off();
        let dev = crtc.drm_dev();
        let data: &EvdiDrmData = dev;
        crate::painter::notify_dpms(data, dev, crate::painter::DPMS_OFF);
    }

    /// Arm the page-flip completion event to be sent by the next vblank tick, so userspace is paced
    /// to the refresh rate rather than signalled immediately.
    fn atomic_flush(commit: CrtcAtomicCommit<'_, Self>) {
        let crtc = commit.crtc();
        let mut new = commit.take_new_state();
        if let Some(pending) = new.get_pending_vblank_event() {
            match crtc.vblank_get() {
                Ok(vbl_ref) => pending.arm(vbl_ref),
                // Vblank couldn't be enabled (e.g. mid-teardown): fall back to sending now.
                Err(_) => pending.send(),
            }
        }
    }
}

impl VblankSupport for EvdiCrtc {
    type Crtc = EvdiCrtc;

    fn enable_vblank(
        crtc: &crtc::Crtc<Self::Crtc>,
        vblank_guard: &VblankGuard<'_, Self::Crtc>,
        _irq: &LocalInterruptDisabled,
    ) -> Result {
        let data: &EvdiCrtc = crtc;
        // Track the mode's real frame duration so the tick matches the negotiated refresh rate.
        let fd = vblank_guard.frame_duration();
        if fd > 0 {
            data.vblank.interval_ns.store(fd as i64, Ordering::Relaxed);
        }
        data.vblank.crtc.store(crtc.as_raw(), Ordering::Relaxed);
        data.vblank.enabled.store(true, Ordering::Relaxed);
        let interval = data.vblank.interval_ns.load(Ordering::Relaxed);
        let mut handle = data.vblank_handle.lock();
        if handle.is_none() {
            // First enable: start the timer, keeping the handle as the sole owner whose drop
            // (at CRTC teardown) is the one full cancel.
            *handle = Some(data.vblank.clone().start(Delta::from_nanos(interval)));
        } else {
            // Re-enable after `disable_vblank` let the timer die (NoRestart): re-queue it in
            // place. `hrtimer_start` removes and re-inserts a still-pending timer, so this is
            // correct whether the final disabled tick has already fired or not, and it never
            // blocks on the callback (safe under the vblank locks we are called with).
            // SAFETY: the pointee is kept alive by `data.vblank` (an `Arc` held for the CRTC's
            // lifetime) and the timer is cancelled at teardown by the handle drop, satisfying
            // "lives until the timer fires or is canceled".
            unsafe {
                <VblankTimer as HasHrTimer<VblankTimer>>::start(
                    &raw const *data.vblank,
                    Delta::from_nanos(interval),
                )
            };
        }
        Ok(())
    }

    fn disable_vblank(
        crtc: &crtc::Crtc<Self::Crtc>,
        _vblank_guard: &VblankGuard<'_, Self::Crtc>,
        _irq: &LocalInterruptDisabled,
    ) {
        let data: &EvdiCrtc = crtc;
        data.vblank.enabled.store(false, Ordering::Relaxed);
    }

    fn get_vblank_timestamp(
        _crtc: &crtc::Crtc<Self::Crtc>,
        _in_vblank_irq: bool,
    ) -> Option<VblankTimestamp> {
        // Let DRM estimate the timestamp from the mode timings.
        None
    }
}

// ---- Planes (primary + cursor) ----------------------------------------------
//
// The safe KMS layer allows a single `DriverPlane` type per driver, so one `EvdiPlane` type serves
// both the primary and cursor planes, told apart by `is_cursor` (set from the plane's `Args`).

#[pin_data]
pub(crate) struct EvdiPlane;

#[derive(Clone, Default)]
pub(crate) struct EvdiPlaneState;

impl plane::DriverPlaneState for EvdiPlaneState {
    type Plane = EvdiPlane;
}

#[vtable]
impl plane::DriverPlane for EvdiPlane {
    type Args = ();
    type Driver = EvdiDrmDriver;
    type State = EvdiPlaneState;

    fn new(_device: &drm::Device<Self::Driver>, _args: ()) -> impl PinInit<Self, Error> {
        try_pin_init!(EvdiPlane {})
    }

    /// A new framebuffer was flipped in.
    ///
    /// EVDI records the new scanout buffer and signals the DLM client to grab it (UPDATE_READY).
    fn atomic_update(commit: PlaneAtomicCommit<'_, Self>) {
        let plane = commit.plane();
        let dev = plane.drm_dev();
        let data: &EvdiDrmData = dev;

        // Record the framebuffer plus each region the compositor changed, accumulate them for
        // GRABPIX, and signal the client. UPDATE_READY fires on every real flip; REQUEST_UPDATE
        // deliberately does not self-signal, avoiding a request/event/grab busy loop.
        let (old, new) = commit.take_old_new_state();
        let fb = new.framebuffer::<EvdiDrmDriver>();
        data.set_scanout(fb);
        if fb.is_some() {
            {
                let mut p = data.painter.lock();
                new.for_each_damage_clip(old, |r| p.damage.push((r.x1, r.y1, r.x2, r.y2)));
                p.frame_dirty = true;
            }
            crate::painter::notify_update_ready(data, dev);
        }
    }
}

// ---- Encoder ----------------------------------------------------------------

#[pin_data]
pub(crate) struct EvdiEncoder;

#[vtable]
impl encoder::DriverEncoder for EvdiEncoder {
    type Driver = EvdiDrmDriver;
    type Args = ();

    fn new(_device: &drm::Device<Self::Driver>, _args: ()) -> impl PinInit<Self, Error> {
        try_pin_init!(EvdiEncoder {})
    }
}

// ---- Connector --------------------------------------------------------------

#[pin_data]
pub(crate) struct EvdiConnector {
    /// EDID delivered by the DLM client through CONNECT (`None` until connected).
    #[pin]
    pub(crate) cached_edid: Mutex<Option<KVec<u8>>>,
    /// Dock bandwidth limits from CONNECT (0 = no limit): the largest mode area in pixels,
    /// and the largest pixel rate (area x vrefresh) the device can move. Enforced by
    /// `mode_valid`; without them the connector advertises every EDID mode (e.g.
    /// 2560x1440@180) and the compositor drives far more pixels than the dock can
    /// transfer, wedging it after a traffic-dependent interval.
    pub(crate) pixel_area_limit: AtomicU32,
    pub(crate) pixel_per_second_limit: AtomicU32,
}

#[derive(Clone, Default)]
pub(crate) struct EvdiConnectorState;

impl connector::DriverConnectorState for EvdiConnectorState {
    type Connector = EvdiConnector;
}

#[vtable]
impl connector::DriverConnector for EvdiConnector {
    type Args = ();
    type Driver = EvdiDrmDriver;
    type State = EvdiConnectorState;

    fn new(_device: &drm::Device<Self::Driver>, _args: ()) -> impl PinInit<Self, Error> {
        try_pin_init!(EvdiConnector {
            cached_edid <- new_mutex!(Option::<KVec<u8>>::None),
            pixel_area_limit: AtomicU32::new(0),
            pixel_per_second_limit: AtomicU32::new(0),
        })
    }

    /// Install the DLM-provided EDID when present, else advertise a fallback mode list so
    /// the connector stays usable before CONNECT.
    fn get_modes<'a>(
        connector: ConnectorGuard<'a, Self>,
        guard: &ModeConfigGuard<'a, Self::Driver>,
    ) -> i32 {
        if let Some(blob) = connector.cached_edid.lock().as_ref() {
            match connector.add_edid_modes(blob) {
                Ok(n) if n > 0 => return n,
                _ => {}
            }
        }
        let _ = guard;
        let n = connector.add_modes_noedid((FALLBACK_W, FALLBACK_H));
        connector.set_preferred_mode((FALLBACK_W, FALLBACK_H));
        n
    }

    /// Report the display as connected only once the DLM client has delivered an EDID (via CONNECT).
    /// Advertising the fallback mode as "connected" before the real EDID arrives makes the
    /// compositor configure the output at the wrong resolution and then reconfigure/disable it when
    /// the EDID (and native modes) show up -- which manifested as the dock output freezing on its
    /// first frame. This mirrors the C evdi's disconnected-until-EDID hotplug sequencing.
    fn detect(connector: &connector::Connector<Self>, _force: bool) -> connector::Status {
        if connector.cached_edid.lock().is_some() {
            connector::Status::Connected
        } else {
            connector::Status::Disconnected
        }
    }

    /// Reject modes the dock cannot move, using the `pixel_area_limit` /
    /// `pixel_per_second_limit` the DLM client supplied through CONNECT -- a port of the C
    /// evdi's `evdi_mode_valid`. Like C, the lowest-refresh mode of each resolution is kept
    /// even when it exceeds the pixel-rate budget (the device then runs it at a limited frame
    /// rate rather than losing the resolution entirely).
    fn mode_valid(connector: &connector::Connector<Self>, mode: &DisplayMode) -> ModeStatus {
        let pps = connector.pixel_per_second_limit.load(Ordering::Relaxed);
        if pps == 0 {
            return ModeStatus::Ok;
        }
        let area = u32::from(mode.hdisplay()) * u32::from(mode.vdisplay());
        let vrefresh = mode.vrefresh().max(0) as u32;
        if area > connector.pixel_area_limit.load(Ordering::Relaxed) {
            pr_warn!(
                "evdi: mode {}x{}@{} rejected: mode area too big\n",
                mode.hdisplay(),
                mode.vdisplay(),
                vrefresh
            );
            return ModeStatus::Bad;
        }
        if area.saturating_mul(vrefresh) <= pps {
            return ModeStatus::Ok;
        }
        if is_lowest_frequency_mode_of_resolution(connector, mode) {
            pr_warn!(
                "evdi: mode {}x{}@{} exceeds the device pixel rate, output frame rate may be limited\n",
                mode.hdisplay(),
                mode.vdisplay(),
                vrefresh
            );
            return ModeStatus::Ok;
        }
        pr_warn!(
            "evdi: mode {}x{}@{} rejected: pixel rate too high\n",
            mode.hdisplay(),
            mode.vdisplay(),
            vrefresh
        );
        ModeStatus::Bad
    }
}

/// C evdi's `is_lowest_frequency_mode_of_given_resolution`: true if no probed mode of the
/// same resolution has a lower vrefresh than `mode`.
fn is_lowest_frequency_mode_of_resolution(
    connector: &connector::Connector<EvdiConnector>,
    mode: &DisplayMode,
) -> bool {
    let raw = connector.as_raw();
    let (hdisplay, vdisplay) = (mode.hdisplay(), mode.vdisplay());
    let vrefresh = mode.vrefresh();
    // SAFETY: `mode_valid` runs during connector probing with `mode_config.mutex` held,
    // which protects `connector->modes`; every entry is a live `drm_display_mode`. This is
    // the same read-only walk the C `evdi_mode_valid` performs from the same context.
    // TODO: replace with a safe probed-modes iterator once the KMS bindings grow one.
    unsafe {
        let head: *mut bindings::list_head = &raw mut (*raw).modes;
        let mut node = (*head).next;
        while node != head {
            let m = kernel::container_of!(node, bindings::drm_display_mode, head);
            if (*m).hdisplay == hdisplay
                && (*m).vdisplay == vdisplay
                && bindings::drm_mode_vrefresh(m) < vrefresh
            {
                return false;
            }
            node = (*node).next;
        }
    }
    true
}
