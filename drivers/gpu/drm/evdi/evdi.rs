// SPDX-License-Identifier: GPL-2.0

//! EVDI — Extensible Virtual Display Interface, Rust implementation.
//!
//! A from-scratch Rust rewrite of the DisplayLink `evdi` kernel module. It presents virtual DRM
//! displays whose scanout is pulled by a userspace daemon (DisplayLinkManager) over a stable
//! ioctl + `drm_event` ABI. The module name and that ABI are kept identical to the C driver so
//! existing userspace (libevdi/DLM) keeps working unchanged.
//!
//! Structure:
//! - [`uapi`]: the DLM-facing ABI ([`evdi_drm.h`] mirror).
//! - [`kms`]: the DRM/KMS device (one virtual head, GEM-shmem dumb buffers).
//! - [`painter`]: connection state + `drm_event` delivery.
//! - [`ioctl`]: the five driver-private ioctls.
//! - This file: the platform driver (one DRM card per `evdi` platform device) and the module.

use kernel::{
    device::{self, Core},
    drm, platform,
    prelude::*,
    sync::{aref::ARef, new_mutex, Mutex},
    types::Opaque,
    ThisModule,
};

mod ddcci;
mod ioctl;
mod kms;
mod painter;
mod uapi;

use kms::{EvdiDrmData, EvdiDrmDevice, EvdiDrmDriver};

/// Flags the card as dying and wakes its in-kernel waiters. Declared as the FIRST field of
/// [`BoundData`] so it drops first on unbind: any DDC/CI transfer parked on the reply condvar
/// bails out with `ENODEV` immediately instead of making `i2c_del_adapter` (and thus the whole
/// unbind) wait out its timeout.
struct ShutdownGuard(ARef<EvdiDrmDevice>);

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        let data: &EvdiDrmData = &self.0;
        data.shutdown();
    }
}

/// Per-bound-device data held by the platform driver: the registered DRM card, whose lifetime is
/// otherwise tied to the bound platform device via devres. Dropped when the platform device
/// unbinds; fields drop in declaration order (shutdown flag → I2C adapter → DRM device ref).
struct BoundData<'bound> {
    _shutdown: ShutdownGuard,
    /// The DDC/CI virtual I2C adapter (dropped → `i2c_del_adapter` on unbind).
    _i2c: Option<Pin<KBox<kernel::i2c::BusAdapter<ddcci::EvdiI2c>>>>,
    _registration: drm::Registration<'bound, EvdiDrmDriver>,
}

/// The `evdi` platform driver. It binds to the `evdi` platform devices created by [`EvdiModule`]
/// (and, later, by the sysfs `add` attribute) and registers a DRM/KMS card parented to each.
struct EvdiPlatformDriver;

impl platform::Driver for EvdiPlatformDriver {
    type IdInfo = ();
    type Data<'bound> = BoundData<'bound>;

    fn probe<'bound>(
        pdev: &'bound platform::Device<Core<'_>>,
        _info: Option<&'bound Self::IdInfo>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
        let cdev: &device::Device<Core<'_>> = pdev.as_ref();

        // Allocate an unregistered DRM device parented to this platform device, wire up its KMS
        // pipeline (`KmsDriver::probe` runs inside `UnregisteredDevice::new`), then register it and
        // tie its lifetime to the bound platform device via devres. Failure fails the probe: a
        // bound platform device with no DRM card behind it would look loaded but do nothing.
        let unregistered =
            drm::UnregisteredDevice::<EvdiDrmDriver>::new(pdev, EvdiDrmData::new(), &THIS_MODULE)?;
        // SAFETY: The registration is stored in `BoundData` and is therefore dropped during the
        // platform driver's ordered unbind; it is never leaked or forgotten.
        let registration = unsafe { drm::Registration::new(pdev.as_ref(), unregistered, (), 0)? };
        let ddev: ARef<EvdiDrmDevice> = registration.device().into();
        dev_info!(cdev, "evdi: DRM/KMS card registered\n");

        // Register the DDC/CI virtual I2C adapter, parented to this platform device, so
        // monitor-control tools can reach the display. Non-fatal.
        let i2c = match kernel::i2c::BusAdapter::<ddcci::EvdiI2c>::new(
            c"DisplayLink I2C Adapter",
            cdev,
            ddev.clone(),
        ) {
            Ok(a) => Some(a),
            Err(e) => {
                dev_err!(
                    cdev,
                    "evdi: DDC/CI I2C adapter registration failed ({e:?})\n"
                );
                None
            }
        };

        Ok(BoundData {
            _shutdown: ShutdownGuard(ddev.clone()),
            _i2c: i2c,
            _registration: registration,
        })
    }
}

/// One created card: its registered `evdi` platform device plus the bus/port chain of the USB
/// device it was attached to (`usb_len == 0` for a generic, non-USB card) — used to match the
/// dock when it is unplugged (`USB_DEVICE_REMOVE`). The chain is stored instead of the
/// `usb_device` pointer itself because no reference is held on the device: its address could
/// be reused by an unrelated allocation, and matching on it could then tear down the wrong card.
struct CardEntry {
    _dev: platform::RegisteredDevice,
    usb_addr: [u32; MAX_USB_ADDR],
    usb_len: usize,
}

/// The device registry behind the sysfs control interface
/// (`/sys/devices/evdi/{count,add,remove_all}`): the set of `evdi` platform devices created on
/// demand by the DisplayLink daemon or udev. Stored as the sysfs root device's driver data.
#[pin_data(PinnedDrop)]
struct EvdiRegistry {
    #[pin]
    devices: Mutex<KVec<CardEntry>>,
    /// USB-removal notifier. When the dock a card is attached to is unplugged, its DRM device would
    /// otherwise linger — keeping the compositor's fds open and the module refcount pinned — so this
    /// tears the card down. `container_of` recovers the `EvdiRegistry` in the callback; registered in
    /// [`EvdiRegistry::new`], unregistered in the [`PinnedDrop`].
    #[pin]
    notifier: Opaque<kernel::bindings::notifier_block>,
}

// SAFETY: `notifier` is a `notifier_block` owned by the kernel's internally-synchronized USB notifier
// chain once registered; we only touch it single-threaded (register in `new`, unregister in
// `PinnedDrop`) and never mutate it otherwise. `devices` is mutex-guarded. So the registry is safe to
// share and move across threads.
unsafe impl Send for EvdiRegistry {}
// SAFETY: See `Send`.
unsafe impl Sync for EvdiRegistry {}

/// USB notifier callback: on `USB_DEVICE_REMOVE`, tear down any card attached to the removed device.
unsafe extern "C" fn usb_remove_notify(
    nb: *mut kernel::bindings::notifier_block,
    action: usize,
    data: *mut core::ffi::c_void,
) -> core::ffi::c_int {
    if action as u32 == kernel::bindings::USB_DEVICE_REMOVE {
        // SAFETY: `nb` is the `notifier` field of a live `EvdiRegistry` — registered in `new` and
        // unregistered in `PinnedDrop` before the struct is freed, so the container is valid here.
        let registry = unsafe {
            &*kernel::container_of!(
                nb.cast::<Opaque<kernel::bindings::notifier_block>>(),
                EvdiRegistry,
                notifier
            )
        };
        registry.remove_usb(data as *mut kernel::bindings::usb_device);
    }
    kernel::bindings::NOTIFY_DONE as core::ffi::c_int
}

/// Maximum bus/port depth of a parsed `usb:` device address.
const MAX_USB_ADDR: usize = 8;

/// Maximum number of evdi cards, matching C evdi's `EVDI_DEVICE_COUNT_MAX`.
const MAX_CARDS: usize = 16;

/// Scratch used to locate a USB device by its bus/port address while iterating `usb_for_each_dev`.
struct UsbPath {
    addr: [u32; MAX_USB_ADDR],
    len: usize,
    found: *mut kernel::bindings::usb_device,
}

/// `usb_for_each_dev` callback: match `data` (a [`UsbPath`]) against `usb`'s bus/port chain,
/// walking parents leaf-first exactly like the C evdi. Stores the device and returns 1 on a match.
unsafe extern "C" fn match_usb_path(
    usb: *mut kernel::bindings::usb_device,
    data: *mut core::ffi::c_void,
) -> core::ffi::c_int {
    // SAFETY: `usb_for_each_dev` is called below with a `&mut UsbPath` as `data` and passes a live
    // `usb_device` for `usb`.
    let m = unsafe { &mut *(data as *mut UsbPath) };
    let mut pdev = usb;
    let mut i = m.len as isize - 1;
    while !pdev.is_null() && i >= 0 && (i as usize) < MAX_USB_ADDR {
        // SAFETY: `pdev` walks the valid parent chain of a live `usb_device`.
        let mut port = unsafe { (*pdev).portnum } as u32;
        if port == 0 {
            // SAFETY: a live `usb_device` always has a valid `bus`.
            port = unsafe { (*(*pdev).bus).busnum } as u32;
        }
        if port != m.addr[i as usize] {
            return 0;
        }
        // SAFETY: as above.
        let parent = unsafe { (*pdev).parent };
        if parent.is_null() && i == 0 {
            // Pin the device with a refcount taken under the USB-tree lock that
            // `usb_for_each_dev` holds across this callback; `add_usb_device`
            // drops it once it is done dereferencing the pointer. Without this,
            // a disconnect racing the return from `usb_for_each_dev` could free
            // the `usb_device` before we read `dev.kobj` (use-after-free).
            // SAFETY: `usb` is a live device for the duration of this callback.
            m.found = unsafe { kernel::bindings::usb_get_dev(usb) };
            return 1;
        }
        pdev = parent;
        i -= 1;
    }
    0
}

impl EvdiRegistry {
    fn new() -> impl PinInit<Self, Error> {
        try_pin_init!(Self {
            devices <- new_mutex!(KVec::new()),
            notifier <- Opaque::ffi_init(|slot: *mut kernel::bindings::notifier_block| {
                // SAFETY: `slot` is this field's stable final address. Fill in the callback and link
                // it onto the USB notifier chain; `usb_register_notify` keeps no other reference.
                unsafe {
                    (*slot).notifier_call = Some(usb_remove_notify);
                    (*slot).next = core::ptr::null_mut();
                    (*slot).priority = 0;
                    kernel::bindings::usb_register_notify(slot);
                }
            }),
        })
    }

    /// Tear down every card attached to `usb` (the just-removed USB device). Runs in the USB
    /// notifier's sleepable process context; the platform devices are unregistered after the
    /// `devices` lock is released so the (sleeping) teardown does not run under the lock.
    fn remove_usb(&self, usb: *mut kernel::bindings::usb_device) {
        // Rebuild the removed device's bus/port chain (root-first, the same encoding
        // `add_usb_device` parsed and `match_usb_path` walked: portnum per hop, busnum at the
        // root hub) and match on that. `usb` is live for the duration of the notifier call.
        let mut addr = [0u32; MAX_USB_ADDR];
        let mut len = 0usize;
        let mut pdev = usb;
        while !pdev.is_null() && len < MAX_USB_ADDR {
            // SAFETY: `pdev` walks the valid parent chain of the still-live `usb_device` the
            // USB core passed to the removal notifier.
            let mut port = unsafe { (*pdev).portnum } as u32;
            if port == 0 {
                // SAFETY: a live `usb_device` always has a valid `bus`.
                port = unsafe { (*(*pdev).bus).busnum } as u32;
            }
            addr[len] = port;
            len += 1;
            // SAFETY: as above.
            pdev = unsafe { (*pdev).parent };
        }
        if !pdev.is_null() {
            // Deeper than MAX_USB_ADDR: `add_usb_device` cannot have stored it, so nothing to do
            // (and the truncated chain must not be allowed to match some other entry).
            return;
        }
        addr[..len].reverse();

        let mut to_drop: KVec<CardEntry> = KVec::new();
        {
            let mut devices = self.devices.lock();
            let mut i = 0;
            while i < devices.len() {
                if devices[i].usb_len == len && devices[i].usb_addr[..len] == addr[..len] {
                    if let Ok(entry) = devices.remove(i) {
                        let _ = to_drop.push(entry, GFP_KERNEL);
                    }
                } else {
                    i += 1;
                }
            }
        }
        drop(to_drop);
    }

    /// Create a card for the USB device named `token` (e.g. `"2-1.1"`), giving its platform device a
    /// `device` sysfs symlink to that USB device — the pairing libevdi/DisplayLinkManager look up.
    fn add_usb_device(&self, token: &str) -> Result {
        // Parse "<bus>-<port>[.<port>...]" into a bus/port address (bus first, leaf last).
        let mut m = UsbPath {
            addr: [0; MAX_USB_ADDR],
            len: 0,
            found: core::ptr::null_mut(),
        };
        let mut dash = token.split('-');
        let bus = dash.next().ok_or(EINVAL)?;
        m.addr[m.len] = bus.parse::<u32>().map_err(|_| EINVAL)?;
        m.len += 1;
        for p in dash.next().ok_or(EINVAL)?.split('.') {
            if m.len >= MAX_USB_ADDR {
                return Err(EINVAL);
            }
            m.addr[m.len] = p.parse::<u32>().map_err(|_| EINVAL)?;
            m.len += 1;
        }

        // SAFETY: `usb_for_each_dev` calls `match_usb_path` for each live USB device with our
        // `&mut m`; it retains no reference, so `m.found` is only used synchronously below.
        unsafe {
            kernel::bindings::usb_for_each_dev(
                &raw mut m as *mut core::ffi::c_void,
                Some(match_usb_path),
            )
        };
        let usb = m.found;
        if usb.is_null() {
            return Err(EINVAL);
        }

        let regdev = platform::RegisteredDevice::new(c"evdi", platform::DEVID_AUTO, None, 0)?;
        let pdev =
            regdev.device() as *const platform::Device as *mut kernel::bindings::platform_device;
        // SAFETY: `pdev` is the platform device just registered above; `usb` is the matched, still
        // live USB device. `sysfs_create_link` copies `name` and takes no lasting reference; the
        // link is removed automatically when the platform device's kobject is destroyed on drop.
        let ret = unsafe {
            kernel::bindings::sysfs_create_link(
                &raw mut (*pdev).dev.kobj,
                &raw mut (*usb).dev.kobj,
                c"device".as_char_ptr(),
            )
        };
        // Done dereferencing `usb`; drop the reference `match_usb_path` took. `CardEntry`
        // keeps only the parsed bus/port chain, never the pointer.
        // SAFETY: paired with the `usb_get_dev` in `match_usb_path`.
        unsafe { kernel::bindings::usb_put_dev(usb) };
        kernel::error::to_result(ret)?;
        let mut devices = self.devices.lock();
        if devices.len() >= MAX_CARDS {
            return Err(EINVAL);
        }
        devices.push(
            CardEntry {
                _dev: regdev,
                usb_addr: m.addr,
                usb_len: m.len,
            },
            GFP_KERNEL,
        )?;
        Ok(())
    }
}

#[pinned_drop]
impl PinnedDrop for EvdiRegistry {
    fn drop(self: Pin<&mut Self>) {
        // Stop USB-removal callbacks before the card list is torn down. `usb_unregister_notify`
        // waits for any in-flight callback, so no `remove_usb` can race the field drops that follow.
        // SAFETY: `notifier` was registered exactly once in `new`.
        unsafe { kernel::bindings::usb_unregister_notify(self.notifier.get()) };
    }
}

impl kernel::sysfs::DeviceAttributes for EvdiRegistry {
    const ATTRS: &'static [kernel::sysfs::Attr] = &[
        kernel::sysfs::Attr::ro(c"count"),
        kernel::sysfs::Attr::wo(c"add"),
        kernel::sysfs::Attr::wo(c"remove_all"),
    ];

    fn show(&self, name: &CStr, out: &mut kernel::sysfs::Writer<'_>) -> Result {
        if name == c"count" {
            write_uint(out, self.devices.lock().len())
        } else {
            Err(EINVAL)
        }
    }

    fn store(&self, name: &CStr, buf: &[u8]) -> Result {
        if name == c"add" {
            let text = core::str::from_utf8(buf).map_err(|_| EINVAL)?.trim();
            // DisplayLinkManager (via libevdi) creates a card for a specific dock by writing
            // "usb:<bus>-<port>[.<port>...]:<intf>"; a bare integer creates that many generic cards.
            if let Some(rest) = text.strip_prefix("usb:") {
                let token = rest.split(':').next().unwrap_or("").trim();
                self.add_usb_device(token)
            } else {
                let count: u32 = text.parse().map_err(|_| EINVAL)?;
                let mut devices = self.devices.lock();
                // Cap the total card count (mirrors C evdi's EVDI_DEVICE_COUNT_MAX) so a stray
                // large write cannot loop creating platform devices until memory runs out.
                if count as usize > MAX_CARDS || devices.len() + count as usize > MAX_CARDS {
                    return Err(EINVAL);
                }
                for _ in 0..count {
                    let dev =
                        platform::RegisteredDevice::new(c"evdi", platform::DEVID_AUTO, None, 0)?;
                    devices.push(
                        CardEntry {
                            _dev: dev,
                            usb_addr: [0; MAX_USB_ADDR],
                            usb_len: 0,
                        },
                        GFP_KERNEL,
                    )?;
                }
                Ok(())
            }
        } else if name == c"remove_all" {
            // The emergency teardown: take the entries out under the lock, then drop them after
            // releasing it — dropping a `CardEntry` unregisters the platform device (sleeping
            // DRM + I2C teardown), which must not run under the registry mutex (mirrors
            // `remove_usb`). Clients holding the DRM node see `ENODEV` on every subsequent call
            // (the cards are unplugged, not freed under them), so this is safe to use with
            // DisplayLinkManager still attached; once it closes its fds the module can be
            // unloaded normally.
            let mut to_drop: KVec<CardEntry> = KVec::new();
            core::mem::swap(&mut *self.devices.lock(), &mut to_drop);
            drop(to_drop);
            Ok(())
        } else {
            Err(EINVAL)
        }
    }
}

/// Format `v` as decimal followed by a newline into initialized sysfs output.
fn write_uint(out: &mut kernel::sysfs::Writer<'_>, mut v: usize) -> Result {
    let mut tmp = [0u8; 24];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    let digits = &tmp[i..];
    out.write(digits)?;
    out.write(b"\n")
}

/// The module state.
///
/// Field/drop order matters: `_sysfs` (which owns the device registry, and thus every card) is
/// declared first so it is dropped -- unregistering every card -- *before* the driver registration.
#[pin_data]
struct EvdiModule {
    _sysfs: Option<KBox<kernel::sysfs::AttributeGroup<EvdiRegistry>>>,
    #[pin]
    _driver: kernel::driver::Registration<platform::Adapter<EvdiPlatformDriver>>,
}

impl kernel::InPlaceModule for EvdiModule {
    fn init(module: &'static ThisModule) -> impl PinInit<Self, Error> {
        pr_info!("evdi: Rust EVDI loading\n");
        try_pin_init!(Self {
            _sysfs: kernel::sysfs::AttributeGroup::register_root(c"evdi", module, EvdiRegistry::new())
                .inspect_err(|e| pr_err!("evdi: sysfs root registration failed ({e:?})\n"))
                .ok(),
            _driver <- kernel::driver::Registration::new(c"evdi", module),
        })
    }
}

module! {
    type: EvdiModule,
    name: "evdi",
    authors: ["Mike Lothian"],
    description: "Extensible Virtual Display Interface (Rust)",
    license: "GPL",
}
