// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (C) 2025 Collabora Ltd.

//! Abstractions for the USB bus.
//!
//! C header: [`include/linux/usb.h`](srctree/include/linux/usb.h)

use crate::{
    alloc::Flags,
    bindings,
    device,
    device_id::{
        RawDeviceId,
        RawDeviceIdIndex, //
    },
    driver,
    error::{
        from_result,
        to_result, //
    },
    prelude::*,
    sync::{
        aref::{
            ARef,
            AlwaysRefCounted, //
        },
        new_condvar,
        new_mutex,
        Arc,
        ArcBorrow,
        Completion,
        CondVar,
        Mutex, //
    },
    time::Delta,
    types::Opaque,
    usb::ch9::{
        CtrlRequest,
        Direction,
        EndpointDescriptor,
        InterfaceClass,
        InterfaceDescriptor, //
    },
    ThisModule, //
};
use core::{
    marker::PhantomData,
    mem::{
        offset_of,
        MaybeUninit, //
    },
    ops::Deref,
    ptr::{
        self,
        NonNull, //
    },
    slice, //
};

pub mod ch9;

/// An adapter for the registration of USB drivers.
pub struct Adapter<T: Driver>(T);

#[pin_data]
#[doc(hidden)]
pub struct BoundData<'bound, T: Driver> {
    #[pin]
    driver_data: T::Data<'bound>,
    io: Arc<IoWindow>,
}

impl<'bound, T: Driver> BoundData<'bound, T> {
    fn driver_data<'a>(self: Pin<&'a Self>) -> Pin<&'a T::Data<'bound>> {
        // SAFETY: `driver_data` is structurally pinned with `Self`.
        unsafe { self.map_unchecked(|this| &this.driver_data) }
    }
}

// SAFETY:
// - `bindings::usb_driver` is a C type declared as `repr(C)`.
// - `BoundData<T>` is the type of the driver's device private data.
// - `struct usb_driver` embeds a `struct device_driver`.
// - `DEVICE_DRIVER_OFFSET` is the correct byte offset to the embedded `struct device_driver`.
unsafe impl<T: Driver> driver::DriverLayout for Adapter<T> {
    type DriverType = bindings::usb_driver;
    type DriverData<'bound> = BoundData<'bound, T>;
    const DEVICE_DRIVER_OFFSET: usize = core::mem::offset_of!(Self::DriverType, driver);
}

// SAFETY: A call to `unregister` for a given instance of `DriverType` is guaranteed to be valid if
// a preceding call to `register` has been successful.
unsafe impl<T: Driver> driver::RegistrationOps for Adapter<T> {
    unsafe fn register(
        udrv: &Opaque<Self::DriverType>,
        name: &'static CStr,
        module: &'static ThisModule,
    ) -> Result {
        // SAFETY: It's safe to set the fields of `struct usb_driver` on initialization.
        unsafe {
            (*udrv.get()).name = name.as_char_ptr();
            (*udrv.get()).probe = Some(Self::probe_callback);
            (*udrv.get()).disconnect = Some(Self::disconnect_callback);
            (*udrv.get()).suspend = Some(Self::suspend_callback);
            (*udrv.get()).resume = Some(Self::resume_callback);
            (*udrv.get()).reset_resume = Some(Self::reset_resume_callback);
            (*udrv.get()).pre_reset = Some(Self::pre_reset_callback);
            (*udrv.get()).post_reset = Some(Self::post_reset_callback);
            (*udrv.get()).id_table = T::ID_TABLE.as_ptr();
            (*udrv.get()).set_soft_unbind(T::SOFT_UNBIND as core::ffi::c_uint);
        }

        // SAFETY: `udrv` is guaranteed to be a valid `DriverType`.
        to_result(unsafe {
            bindings::usb_register_driver(udrv.get(), module.0, name.as_char_ptr())
        })
    }

    unsafe fn unregister(udrv: &Opaque<Self::DriverType>) {
        // SAFETY: `udrv` is guaranteed to be a valid `DriverType`.
        unsafe { bindings::usb_deregister(udrv.get()) };
    }
}

impl<T: Driver> Adapter<T> {
    extern "C" fn probe_callback(
        intf: *mut bindings::usb_interface,
        id: *const bindings::usb_device_id,
    ) -> kernel::ffi::c_int {
        // SAFETY: The USB core only ever calls the probe callback with a valid pointer to a
        // `struct usb_interface` and `struct usb_device_id`.
        //
        // INVARIANT: `intf` is valid for the duration of `probe_callback()`.
        let intf = unsafe { &*intf.cast::<Interface<device::CoreInternal<'_>>>() };

        from_result(|| {
            // SAFETY: `DeviceId` is a `#[repr(transparent)]` wrapper of `struct usb_device_id` and
            // does not add additional invariants, so it's safe to transmute.
            let id = unsafe { &*id.cast::<DeviceId>() };

            let info = T::ID_TABLE.info(id.index());
            let interface: ARef<Interface> = intf.into();
            let io = Arc::pin_init(IoWindow::new(interface), GFP_KERNEL)?;
            let data = try_pin_init!(BoundData::<T> {
                driver_data <- T::probe(intf, id, info, io.clone()),
                io,
            });

            let dev: &device::Device<device::CoreInternal<'_>> = intf.as_ref();
            dev.set_drvdata(data)?;
            Ok(0)
        })
    }

    extern "C" fn disconnect_callback(intf: *mut bindings::usb_interface) {
        // SAFETY: The USB core only ever calls the disconnect callback with a valid pointer to a
        // `struct usb_interface`.
        //
        // INVARIANT: `intf` is valid for the duration of `disconnect_callback()`.
        let intf = unsafe { &*intf.cast::<Interface<device::CoreInternal<'_>>>() };

        let dev: &device::Device<device::CoreInternal<'_>> = intf.as_ref();

        // Take ownership of the driver data here rather than leaving it to the driver core's
        // generic post-unbind teardown: `usb_unbind_interface()` calls `usb_set_intfdata(intf,
        // NULL)` as soon as this callback returns, which is *before* `device_unbind_cleanup()`
        // runs `post_unbind_rust`. The generic `drvdata_obtain()` therefore always finds NULL for
        // USB and the driver data -- with everything it owns, such as a `drm::Registration` -- is
        // leaked on every unbind.
        //
        // SAFETY: `disconnect_callback` is only ever called after a successful call to
        // `probe_callback`, hence it's guaranteed that `Device::set_drvdata()` has been called
        // and stored a `Pin<KBox<BoundData<'_, T>>>`.
        let data = unsafe { dev.drvdata_obtain::<BoundData<'_, T>>() };

        if let Some(data) = data {
            T::quiesce(intf, data.as_ref().driver_data());
            data.io.close();
            T::disconnect(intf, data.as_ref().driver_data());

            // Dropped only after `T::disconnect()` has returned, so a driver can rely on its
            // owned resources still being alive for the whole of its disconnect handling.
            drop(data);
        }
    }

    /// Recovers the typed interface and driver data shared by every power-management and reset
    /// callback, then dispatches to `f`.
    ///
    /// `f` is spelled as an explicitly higher-ranked `fn` pointer because the interface and the
    /// driver data share the `'bound` lifetime; an `impl FnOnce` bound loses that relationship and
    /// the trait methods no longer satisfy it.
    fn pm_dispatch(
        intf: *mut bindings::usb_interface,
        f: for<'bound, 'a, 'b> fn(
            &'bound Interface<device::Core<'a>>,
            Pin<&'b T::Data<'bound>>,
        ) -> Result,
        resume: bool,
    ) -> kernel::ffi::c_int {
        // SAFETY: The USB core only ever calls these with a valid `struct usb_interface`.
        //
        // INVARIANT: `intf` is valid for the duration of the callback.
        let intf = unsafe { &*intf.cast::<Interface<device::CoreInternal<'_>>>() };

        let dev: &device::Device<device::CoreInternal<'_>> = intf.as_ref();

        // SAFETY: These callbacks only ever run between a successful `probe_callback()` and
        // `disconnect_callback()`, so the driver data is present.
        let data = unsafe { dev.drvdata_borrow::<BoundData<'_, T>>() };

        from_result(|| {
            if resume {
                data.io.reopen();
            }

            if let Err(e) = f(intf, data.driver_data()) {
                if resume {
                    data.io.close();
                }
                return Err(e);
            }

            if !resume {
                data.io.close();
            }
            Ok(0)
        })
    }

    extern "C" fn suspend_callback(
        intf: *mut bindings::usb_interface,
        _message: bindings::pm_message_t,
    ) -> kernel::ffi::c_int {
        Self::pm_dispatch(intf, T::suspend, false)
    }

    extern "C" fn resume_callback(intf: *mut bindings::usb_interface) -> kernel::ffi::c_int {
        Self::pm_dispatch(intf, T::resume, true)
    }

    extern "C" fn reset_resume_callback(intf: *mut bindings::usb_interface) -> kernel::ffi::c_int {
        Self::pm_dispatch(intf, T::reset_resume, true)
    }

    extern "C" fn pre_reset_callback(intf: *mut bindings::usb_interface) -> kernel::ffi::c_int {
        Self::pm_dispatch(intf, T::pre_reset, false)
    }

    extern "C" fn post_reset_callback(intf: *mut bindings::usb_interface) -> kernel::ffi::c_int {
        Self::pm_dispatch(intf, T::post_reset, true)
    }
}

/// Abstraction for the USB device ID structure, i.e. [`struct usb_device_id`].
///
/// [`struct usb_device_id`]: https://docs.kernel.org/driver-api/basics.html#c.usb_device_id
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DeviceId(bindings::usb_device_id);

impl DeviceId {
    /// Equivalent to C's `USB_DEVICE` macro.
    pub const fn from_id(vendor: u16, product: u16) -> Self {
        Self(bindings::usb_device_id {
            match_flags: bindings::USB_DEVICE_ID_MATCH_DEVICE as u16,
            idVendor: vendor,
            idProduct: product,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_DEVICE_VER` macro.
    pub const fn from_device_ver(vendor: u16, product: u16, bcd_lo: u16, bcd_hi: u16) -> Self {
        Self(bindings::usb_device_id {
            match_flags: bindings::USB_DEVICE_ID_MATCH_DEVICE_AND_VERSION as u16,
            idVendor: vendor,
            idProduct: product,
            bcdDevice_lo: bcd_lo,
            bcdDevice_hi: bcd_hi,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_DEVICE_INFO` macro.
    pub const fn from_device_info(class: u8, subclass: u8, protocol: u8) -> Self {
        Self(bindings::usb_device_id {
            match_flags: bindings::USB_DEVICE_ID_MATCH_DEV_INFO as u16,
            bDeviceClass: class,
            bDeviceSubClass: subclass,
            bDeviceProtocol: protocol,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_INTERFACE_INFO` macro.
    pub const fn from_interface_info(class: u8, subclass: u8, protocol: u8) -> Self {
        Self(bindings::usb_device_id {
            match_flags: bindings::USB_DEVICE_ID_MATCH_INT_INFO as u16,
            bInterfaceClass: class,
            bInterfaceSubClass: subclass,
            bInterfaceProtocol: protocol,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_VENDOR_AND_INTERFACE_INFO` macro.
    ///
    /// Matches every device from one vendor that exposes an interface of the given class,
    /// subclass and protocol, whatever its product ID. This is how a driver binds to a *function*
    /// rather than to a list of the products someone happened to test.
    pub const fn from_vendor_and_interface_info(
        vendor: u16,
        class: u8,
        subclass: u8,
        protocol: u8,
    ) -> Self {
        Self(bindings::usb_device_id {
            match_flags: (bindings::USB_DEVICE_ID_MATCH_VENDOR
                | bindings::USB_DEVICE_ID_MATCH_INT_INFO) as u16,
            idVendor: vendor,
            bInterfaceClass: class,
            bInterfaceSubClass: subclass,
            bInterfaceProtocol: protocol,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_DEVICE_INTERFACE_CLASS` macro.
    pub const fn from_device_interface_class(vendor: u16, product: u16, class: u8) -> Self {
        Self(bindings::usb_device_id {
            match_flags: (bindings::USB_DEVICE_ID_MATCH_DEVICE
                | bindings::USB_DEVICE_ID_MATCH_INT_CLASS) as u16,
            idVendor: vendor,
            idProduct: product,
            bInterfaceClass: class,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_DEVICE_INTERFACE_PROTOCOL` macro.
    pub const fn from_device_interface_protocol(vendor: u16, product: u16, protocol: u8) -> Self {
        Self(bindings::usb_device_id {
            match_flags: (bindings::USB_DEVICE_ID_MATCH_DEVICE
                | bindings::USB_DEVICE_ID_MATCH_INT_PROTOCOL) as u16,
            idVendor: vendor,
            idProduct: product,
            bInterfaceProtocol: protocol,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_DEVICE_INTERFACE_NUMBER` macro.
    pub const fn from_device_interface_number(vendor: u16, product: u16, number: u8) -> Self {
        Self(bindings::usb_device_id {
            match_flags: (bindings::USB_DEVICE_ID_MATCH_DEVICE
                | bindings::USB_DEVICE_ID_MATCH_INT_NUMBER) as u16,
            idVendor: vendor,
            idProduct: product,
            bInterfaceNumber: number,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }

    /// Equivalent to C's `USB_DEVICE_AND_INTERFACE_INFO` macro.
    pub const fn from_device_and_interface_info(
        vendor: u16,
        product: u16,
        class: u8,
        subclass: u8,
        protocol: u8,
    ) -> Self {
        Self(bindings::usb_device_id {
            match_flags: (bindings::USB_DEVICE_ID_MATCH_INT_INFO
                | bindings::USB_DEVICE_ID_MATCH_DEVICE) as u16,
            idVendor: vendor,
            idProduct: product,
            bInterfaceClass: class,
            bInterfaceSubClass: subclass,
            bInterfaceProtocol: protocol,
            // SAFETY: It is safe to use all zeroes for the other fields of `usb_device_id`.
            ..unsafe { MaybeUninit::zeroed().assume_init() }
        })
    }
}

// SAFETY: `DeviceId` is a `#[repr(transparent)]` wrapper of `usb_device_id` and does not add
// additional invariants, so it's safe to transmute to `RawType`.
unsafe impl RawDeviceId for DeviceId {
    type RawType = bindings::usb_device_id;
}

// SAFETY: `DRIVER_DATA_OFFSET` is the offset to the `driver_info` field.
unsafe impl RawDeviceIdIndex for DeviceId {
    const DRIVER_DATA_OFFSET: usize = core::mem::offset_of!(bindings::usb_device_id, driver_info);

    fn index(&self) -> usize {
        self.0.driver_info
    }
}

/// [`IdTable`](kernel::device_id::IdTable) type for USB.
pub type IdTable<T> = &'static dyn kernel::device_id::IdTable<DeviceId, T>;

/// Create a USB `IdTable` with its alias for modpost.
#[macro_export]
macro_rules! usb_device_table {
    ($table_name:ident, $module_table_name:ident, $id_info_type: ty, $table_data: expr) => {
        const $table_name: $crate::device_id::IdArray<
            $crate::usb::DeviceId,
            $id_info_type,
            { $table_data.len() },
        > = $crate::device_id::IdArray::new($table_data);

        $crate::module_device_table!("usb", $module_table_name, $table_name);
    };
}

/// The USB driver trait.
///
/// # Examples
///
///```
/// # use kernel::{bindings, device::Core, sync::Arc, usb};
/// use kernel::prelude::*;
///
/// struct MyDriver;
///
/// kernel::usb_device_table!(
///     USB_TABLE,
///     MODULE_USB_TABLE,
///     <MyDriver as usb::Driver>::IdInfo,
///     [
///         (usb::DeviceId::from_id(0x1234, 0x5678), ()),
///         (usb::DeviceId::from_id(0xabcd, 0xef01), ()),
///     ]
/// );
///
/// impl usb::Driver for MyDriver {
///     type IdInfo = ();
///     type Data<'bound> = Self;
///     const ID_TABLE: usb::IdTable<Self::IdInfo> = &USB_TABLE;
///
///     fn probe<'bound>(
///         _interface: &'bound usb::Interface<Core<'_>>,
///         _id: &usb::DeviceId,
///         _info: &'bound Self::IdInfo,
///         _io: Arc<usb::IoWindow>,
///     ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
///         Err(ENODEV)
///     }
///
///     fn disconnect<'bound>(
///         _interface: &'bound usb::Interface<Core<'_>>,
///         _data: Pin<&Self::Data<'bound>>,
///     ) {
///     }
/// }
///```
pub trait Driver {
    /// The type holding information about each one of the device ids supported by the driver.
    type IdInfo: 'static;

    /// The type of the driver's bus device private data.
    type Data<'bound>: Send + 'bound;

    /// The table of device ids supported by the driver.
    const ID_TABLE: IdTable<Self::IdInfo>;

    /// Whether the USB core must leave this interface usable until the driver has let go of it.
    ///
    /// By default the core kills every outstanding URB and disables the interface's endpoints
    /// *before* it calls any driver callback, so a driver that has something to say to the device
    /// on the way out cannot say it: the transfer is refused with [`ENOENT`] because the endpoint
    /// it names no longer exists. Setting this defers that teardown until after the callbacks
    /// return, which makes cancelling outstanding transfers the driver's own responsibility.
    ///
    /// Only useful to a driver that leaves the device in a state a user can see -- a display that
    /// otherwise goes on scanning out its last frame, an interface that must be told to power
    /// down.
    const SOFT_UNBIND: bool = false;

    /// USB driver probe.
    ///
    /// Called when a new USB interface is bound to this driver.
    /// Implementers should attempt to initialize the interface here.
    fn probe<'bound>(
        interface: &'bound Interface<device::Core<'_>>,
        id: &DeviceId,
        id_info: &'bound Self::IdInfo,
        io: Arc<IoWindow>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound;

    /// Quiesces a USB driver before disconnect.
    ///
    /// The implementation must stop work which could start new transfers. Once this returns, the
    /// adapter closes the interface's [`IoWindow`].
    fn quiesce<'bound>(
        _interface: &'bound Interface<device::Core<'_>>,
        _data: Pin<&Self::Data<'bound>>,
    ) {
    }

    /// USB driver disconnect.
    ///
    /// Called after the interface's [`IoWindow`] has been closed and all I/O has completed. The
    /// bound data is dropped after this returns.
    fn disconnect<'bound>(
        interface: &'bound Interface<device::Core<'_>>,
        data: Pin<&Self::Data<'bound>>,
    );

    /// The interface is being suspended.
    ///
    /// The implementation must stop work which could start new transfers. If it returns success,
    /// the adapter closes the interface's [`IoWindow`] before returning to the USB core. Returning
    /// an error aborts the suspend and leaves the window open.
    fn suspend<'bound>(
        _interface: &'bound Interface<device::Core<'_>>,
        _data: Pin<&Self::Data<'bound>>,
    ) -> Result {
        Ok(())
    }

    /// The interface has been resumed. The adapter reopens its [`IoWindow`] before this is called.
    fn resume<'bound>(
        _interface: &'bound Interface<device::Core<'_>>,
        _data: Pin<&Self::Data<'bound>>,
    ) -> Result {
        Ok(())
    }

    /// The interface has been resumed after its device was reset while suspended.
    ///
    /// I/O is permitted again, but the device has lost the state configured before the suspend.
    /// Defaults to [`resume`](Driver::resume).
    fn reset_resume<'bound>(
        interface: &'bound Interface<device::Core<'_>>,
        data: Pin<&Self::Data<'bound>>,
    ) -> Result {
        Self::resume(interface, data)
    }

    /// The device is about to be reset.
    ///
    /// As for [`suspend`](Driver::suspend), the driver must stop work which could start new
    /// transfers. The adapter closes the [`IoWindow`] after a successful return.
    fn pre_reset<'bound>(
        _interface: &'bound Interface<device::Core<'_>>,
        _data: Pin<&Self::Data<'bound>>,
    ) -> Result {
        Ok(())
    }

    /// The device has been reset. The adapter reopens its [`IoWindow`] before this is called.
    fn post_reset<'bound>(
        _interface: &'bound Interface<device::Core<'_>>,
        _data: Pin<&Self::Data<'bound>>,
    ) -> Result {
        Ok(())
    }
}

/// A USB interface.
///
/// This structure represents the Rust abstraction for a C [`struct usb_interface`].
/// The implementation abstracts the usage of a C [`struct usb_interface`] passed
/// in from the C side.
///
/// # Invariants
///
/// An [`Interface`] instance represents a valid [`struct usb_interface`] created
/// by the C portion of the kernel.
///
/// [`struct usb_interface`]: https://www.kernel.org/doc/html/latest/driver-api/usb/usb.html#c.usb_interface
#[repr(transparent)]
pub struct Interface<Ctx: device::DeviceContext = device::Normal>(
    Opaque<bindings::usb_interface>,
    PhantomData<Ctx>,
);

impl<Ctx: device::DeviceContext> Interface<Ctx> {
    fn as_raw(&self) -> *mut bindings::usb_interface {
        self.0.get()
    }

    fn inner(&self) -> &bindings::usb_interface {
        // SAFETY: The type invariants guarantee that `self.0` wraps a valid
        // `struct usb_interface`.
        unsafe { &*self.as_raw() }
    }

    /// Returns the current alternate setting for this interface.
    pub fn cur_altsetting(&self) -> &HostInterface {
        // SAFETY: `cur_altsetting` is a valid `struct usb_host_interface`
        // pointer provided by the USB core. `HostInterface` is
        // `#[repr(transparent)]` over it.
        unsafe { &*(self.inner().cur_altsetting as *const HostInterface) }
    }

    /// Returns all alternate settings for this interface.
    pub fn altsettings(&self) -> &[HostInterface] {
        // SAFETY: `altsetting` is a valid array of `num_altsetting`
        // entries provided by the USB core. `HostInterface` is
        // `#[repr(transparent)]` over `usb_host_interface`.
        unsafe {
            slice::from_raw_parts(
                self.inner().altsetting as *const HostInterface,
                self.inner().num_altsetting as usize,
            )
        }
    }
}

impl Interface<device::Bound> {
    /// Select an alternate setting for this interface.
    ///
    /// On success the device switches to the given alternate setting,
    /// which may change the set of active endpoints. This is a convenience
    /// wrapper around [`Device<Bound>::set_interface`].
    pub fn set_interface(&self, altsetting: u8) -> Result {
        let dev: &Device<device::Bound> = self.as_ref();
        dev.set_interface(self.cur_altsetting().number(), altsetting)
    }
}

/// Abstraction for the USB Host Interface structure, i.e. `struct usb_host_interface`.
#[repr(transparent)]
pub struct HostInterface(Opaque<bindings::usb_host_interface>);

impl HostInterface {
    fn inner(&self) -> &bindings::usb_host_interface {
        // SAFETY: The type invariants guarantee that `self.0` wraps a valid
        // `struct usb_host_interface`.
        unsafe { &*self.0.get() }
    }

    /// Returns the interface descriptor.
    fn desc(&self) -> &InterfaceDescriptor {
        // SAFETY: `desc` is a valid `struct usb_interface_descriptor`
        // embedded in `usb_host_interface`. `InterfaceDescriptor` is
        // `#[repr(transparent)]` over it.
        unsafe { &*((core::ptr::from_ref(&self.inner().desc)).cast()) }
    }

    /// Returns the list of endpoints in this alternate setting.
    pub fn endpoints(&self) -> &[HostEndpoint] {
        // SAFETY: `endpoint` is a valid array of `bNumEndpoints` entries.
        // `HostEndpoint` is `#[repr(transparent)]` over
        // `usb_host_endpoint`.
        unsafe {
            core::ptr::slice_from_raw_parts(
                self.inner().endpoint as *const HostEndpoint,
                self.desc().bNumEndpoints() as usize,
            )
            .as_ref()
            .unwrap_or(&[])
        }
    }

    /// Returns the interface number (`bInterfaceNumber`).
    pub fn number(&self) -> u8 {
        self.desc().bInterfaceNumber()
    }

    /// Returns the alternate setting number (`bAlternateSetting`).
    pub fn alternate_setting(&self) -> u8 {
        self.desc().bAlternateSetting()
    }

    /// Returns the interface class (`bInterfaceClass`).
    pub fn class(&self) -> InterfaceClass {
        self.desc().bInterfaceClass()
    }
}

/// USB endpoint transfer type.
///
/// Maps to the `bmAttributes` field of the endpoint descriptor
/// (`USB_ENDPOINT_XFER_*` constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EndpointType {
    /// Control endpoint.
    Control = bindings::USB_ENDPOINT_XFER_CONTROL as u8,
    /// Isochronous endpoint.
    Isoc = bindings::USB_ENDPOINT_XFER_ISOC as u8,
    /// Bulk endpoint.
    Bulk = bindings::USB_ENDPOINT_XFER_BULK as u8,
    /// Interrupt endpoint.
    Int = bindings::USB_ENDPOINT_XFER_INT as u8,
}

/// Abstraction for the USB Host Endpoint structure, i.e. [`struct usb_host_endpoint`].
///
/// [`struct usb_host_endpoint`]: https://docs.kernel.org/driver-api/usb/usb.html#c.usb_host_endpoint
#[repr(transparent)]
pub struct HostEndpoint(Opaque<bindings::usb_host_endpoint>);

impl HostEndpoint {
    fn inner(&self) -> &bindings::usb_host_endpoint {
        // SAFETY: The type invariants guarantee that `self.0` wraps a valid
        // `struct usb_host_endpoint`.
        unsafe { &*self.0.get() }
    }

    /// Returns the endpoint descriptor.
    fn desc(&self) -> &EndpointDescriptor {
        // SAFETY: `desc` is a valid `struct usb_endpoint_descriptor`
        // embedded in `usb_host_endpoint`. `EndpointDescriptor` is
        // `#[repr(transparent)]` over it.
        unsafe { &*(core::ptr::from_ref(&self.inner().desc).cast()) }
    }

    /// Returns the direction of this endpoint (IN or OUT).
    pub fn endpoint_dir(&self) -> Direction {
        if self.desc().bEndpointAddress() & Direction::In as u8 == 0 {
            Direction::Out
        } else {
            Direction::In
        }
    }

    /// Returns the endpoint number (0-15).
    pub fn endpoint_number(&self) -> u8 {
        self.desc().bEndpointAddress() & bindings::USB_ENDPOINT_NUMBER_MASK as u8
    }

    /// Returns the transfer type of this endpoint.
    pub fn endpoint_type(&self) -> EndpointType {
        let val = self.desc().bmAttributes() & bindings::USB_ENDPOINT_XFERTYPE_MASK as u8;
        // SAFETY: `bmAttributes` masked with `USB_ENDPOINT_XFERTYPE_MASK`
        // is guaranteed to be 0-3, which maps exactly to the four
        // `EndpointType` variants.
        unsafe { core::mem::transmute::<u8, EndpointType>(val) }
    }

    /// Returns the interval for interrupt and isochronous endpoints.
    pub fn interval(&self) -> u8 {
        self.desc().bInterval()
    }

    /// Returns the maximum packet size for this endpoint.
    pub fn maxp(&self) -> u16 {
        u16::from_le(self.desc().wMaxPacketSize()) & bindings::USB_ENDPOINT_MAXP_MASK as u16
    }

    /// Returns the high-speed multiplier for isochronous endpoints.
    pub fn maxp_mult(&self) -> u16 {
        (u16::from_le(self.desc().wMaxPacketSize()) & bindings::USB_EP_MAXP_MULT_MASK as u16)
            >> bindings::USB_EP_MAXP_MULT_SHIFT
    }
}

impl<Ctx: device::DeviceContext> Interface<Ctx> {
    /// Returns this interface's `bInterfaceNumber`, or `None` if it currently has no active
    /// alternate setting.
    pub fn number(&self) -> Option<u8> {
        // SAFETY: `self.as_raw()` is a valid `struct usb_interface` by the type invariant.
        let alt = unsafe { (*self.as_raw()).cur_altsetting };
        if alt.is_null() {
            return None;
        }
        // SAFETY: `alt` is a valid `struct usb_host_interface` (checked non-null above).
        Some(unsafe { (*alt).desc.bInterfaceNumber })
    }

    /// Asks the driver core to unbind whatever driver is currently bound to this interface.
    ///
    /// This is the narrow, reviewed replacement for handing out a raw `struct device` pointer: it
    /// performs exactly one operation (`device_release_driver()`) on this interface's own device,
    /// and cannot be used to reach the device-wide state of a composite peer.
    ///
    /// It is intended for a driver-provided "release my devices" control (e.g. a sysfs attribute),
    /// and must not be called from the driver's own `probe()` or `disconnect()` callback: the
    /// driver core already holds the device lock across those.
    pub fn release_driver(&self) {
        // SAFETY: `self.as_raw()` is a valid `struct usb_interface` by the type invariant, so the
        // address of its embedded `dev` is a valid `struct device`. `device_release_driver()`
        // takes the device lock itself and tolerates a device with no driver bound.
        unsafe { bindings::device_release_driver(&raw mut (*self.as_raw()).dev) };
    }

    /// Asks the USB core to reset the device this interface belongs to, from a work item it owns.
    ///
    /// Unlike `usb_reset_device()`, this may be called from any context, including one holding the
    /// device lock or one running inside a completion handler: the core defers the reset to its
    /// own workqueue. The reset re-enumerates the device, so every driver bound to it is unbound
    /// and re-probed; a caller must therefore treat its own state as gone from this point.
    ///
    /// A driver whose device has stopped responding uses this to recover in place, instead of
    /// requiring the user to unplug it.
    pub fn queue_reset_device(&self) {
        // SAFETY: `self.as_raw()` is a valid `struct usb_interface` by the type invariant, which
        // is all `usb_queue_reset_device()` requires; it performs no I/O itself and tolerates
        // being called when a reset is already pending.
        unsafe { bindings::usb_queue_reset_device(self.as_raw()) };
    }
}

/// The transfer type and direction of a USB endpoint, used to tag an [`Endpoint`] so that a
/// transfer method cannot be pointed at an endpoint of the wrong kind.
///
/// This trait is sealed: the set of endpoint kinds is fixed by this module and matches the USB
/// endpoint types the abstraction supports.
pub trait EndpointKind: private::Sealed {
    /// The `bmAttributes` transfer type (`USB_ENDPOINT_XFER_*`) an endpoint must have.
    const XFER_TYPE: u8;

    /// Whether the endpoint must be an IN (device-to-host) endpoint.
    const DIR_IN: bool;

    /// Build the USB pipe corresponding to a validated endpoint.
    fn pipe<Ctx: device::DeviceContext>(dev: &Device<Ctx>, endpoint: &HostEndpoint) -> Pipe;
}

mod private {
    /// Seals [`EndpointKind`](super::EndpointKind) against external implementations.
    pub trait Sealed {}
}

/// Marker for a bulk IN (device-to-host) endpoint.
pub enum BulkIn {}
/// Marker for a bulk OUT (host-to-device) endpoint.
pub enum BulkOut {}
/// Marker for an interrupt IN (device-to-host) endpoint.
pub enum InterruptIn {}

impl private::Sealed for BulkIn {}
impl private::Sealed for BulkOut {}
impl private::Sealed for InterruptIn {}

impl EndpointKind for BulkIn {
    const XFER_TYPE: u8 = bindings::USB_ENDPOINT_XFER_BULK as u8;
    const DIR_IN: bool = true;

    fn pipe<Ctx: device::DeviceContext>(dev: &Device<Ctx>, endpoint: &HostEndpoint) -> Pipe {
        Pipe::new_receive_bulk_pipe(dev, endpoint)
    }
}

impl EndpointKind for BulkOut {
    const XFER_TYPE: u8 = bindings::USB_ENDPOINT_XFER_BULK as u8;
    const DIR_IN: bool = false;

    fn pipe<Ctx: device::DeviceContext>(dev: &Device<Ctx>, endpoint: &HostEndpoint) -> Pipe {
        Pipe::new_send_bulk_pipe(dev, endpoint)
    }
}

impl EndpointKind for InterruptIn {
    const XFER_TYPE: u8 = bindings::USB_ENDPOINT_XFER_INT as u8;
    const DIR_IN: bool = true;

    fn pipe<Ctx: device::DeviceContext>(dev: &Device<Ctx>, endpoint: &HostEndpoint) -> Pipe {
        Pipe::new_receive_int_pipe(dev, endpoint)
    }
}

/// An endpoint of a USB interface, looked up in the interface's active alternate setting and
/// checked to have the transfer type and direction named by `K`.
///
/// Because an [`Endpoint`] can only be produced by [`Interface::endpoint`], which validates it
/// against the descriptor, a `&Endpoint<BulkOut>` is proof that the address really names a bulk
/// OUT endpoint of that interface. Transfer methods take the correspondingly-typed endpoint, so
/// the direction/type confusion possible with a bare `u8` address cannot occur.
///
/// # Invariants
///
/// `addr` is the `bEndpointAddress` of an endpoint that was present in the interface's active
/// alternate setting, and whose direction and transfer type match `K`.
pub struct Endpoint<K: EndpointKind> {
    addr: u8,
    max_packet: u16,
    pipe: Pipe,
    _kind: PhantomData<K>,
}

impl<K: EndpointKind> Endpoint<K> {
    /// The endpoint's `bEndpointAddress`, including the direction bit.
    pub fn address(&self) -> u8 {
        self.addr
    }

    /// The endpoint's `wMaxPacketSize`.
    pub fn max_packet_size(&self) -> u16 {
        self.max_packet
    }

    fn pipe(&self) -> Pipe {
        self.pipe
    }
}

impl<K: EndpointKind> Clone for Endpoint<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: EndpointKind> Copy for Endpoint<K> {}

impl<Ctx: device::DeviceContext> Interface<Ctx> {
    /// Looks `addr` up in this interface's active alternate setting and returns it as a typed
    /// [`Endpoint`], provided its direction and transfer type match `K`.
    ///
    /// Returns [`ENODEV`] if the interface has no active alternate setting, [`ENOENT`] if no
    /// endpoint with that address is present, and [`EINVAL`] if the endpoint exists but is of the
    /// wrong direction or transfer type.
    pub fn endpoint<K: EndpointKind>(&self, addr: u8) -> Result<Endpoint<K>> {
        if self.number().is_none() {
            return Err(ENODEV);
        }
        for endpoint in self.cur_altsetting().endpoints() {
            let endpoint_addr = endpoint.endpoint_number()
                | if endpoint.endpoint_dir() == Direction::In {
                    bindings::USB_DIR_IN as u8
                } else {
                    0
                };
            if endpoint_addr != addr {
                continue;
            }

            let is_in = endpoint.endpoint_dir() == Direction::In;
            if is_in != K::DIR_IN || endpoint.endpoint_type() as u8 != K::XFER_TYPE {
                return Err(EINVAL);
            }

            let dev: &Device<Ctx> = self.as_ref();
            return Ok(Endpoint {
                addr,
                max_packet: endpoint.maxp(),
                pipe: K::pipe(dev, endpoint),
                _kind: PhantomData,
            });
        }

        Err(ENOENT)
    }
}

/// Converts a [`Delta`] into the whole-millisecond timeout the synchronous USB message helpers
/// expect.
///
/// Those helpers treat `0` as "wait forever", so a caller who asks for a short-but-non-zero
/// timeout must not have it silently truncated into an unbounded wait: any non-zero `timeout`
/// below a millisecond is rounded *up* to 1 ms. Only an explicitly zero [`Delta`] means "wait
/// indefinitely".
fn timeout_millis(timeout: Delta) -> Result<kernel::ffi::c_int> {
    let ms = timeout.as_millis();
    if ms == 0 && !timeout.is_zero() {
        return Ok(1);
    }
    Ok(ms.try_into()?)
}

/// A revocable window during which USB I/O is permitted on an interface.
///
/// A driver-`Bound` interface is *not* on its own proof that a transfer may be issued: the USB
/// core forbids I/O outside the window that opens after a successful `probe()`/resume/reset-resume
/// and must be closed again before `disconnect()`, `suspend()` or `pre_reset()` returns. This type
/// represents exactly that narrower state.
///
/// The USB adapter owns one `IoWindow` for every successfully bound interface and passes a
/// reference-counted handle to [`Driver::probe`]. Drivers take an [`Io`] token from it around every
/// transfer. The adapter revokes the window and blocks until every outstanding token has been
/// dropped and every queue URB registered against it has been killed before suspend, reset or
/// disconnect completes.
///
/// Because [`Io`] borrows the window, and the transfer methods and queues live on [`Io`], a
/// transfer cannot outlive the window that permitted it.
///
#[pin_data]
pub struct IoWindow {
    /// The interface I/O is permitted on. Holding a reference keeps the `struct usb_interface`
    /// allocated; that it is still *bound* is what the open/closed state tracks.
    interface: ARef<Interface>,
    #[pin]
    state: Mutex<IoState>,
    #[pin]
    idle: CondVar,
}

/// The mutable half of an [`IoWindow`].
struct IoState {
    /// Whether new [`Io`] tokens may still be handed out.
    open: bool,
    /// How many [`Io`] tokens are currently alive.
    active: usize,
    /// URBs belonging to queues opened against this window, so [`IoWindow::close`] can cancel
    /// them even though the queues themselves are owned by the driver.
    urbs: KVec<RegisteredUrb>,
    /// Source of the per-queue tokens used to deregister a queue's URBs on drop.
    next_token: u64,
}

/// One queue-owned URB registered with an [`IoWindow`], tagged with its queue's token.
struct RegisteredUrb {
    token: u64,
    canceller: UrbCanceller,
}

impl IoWindow {
    /// Creates the open I/O window owned by the USB adapter.
    fn new(interface: ARef<Interface>) -> impl PinInit<Self> {
        pin_init!(Self {
            interface,
            state <- new_mutex!(IoState {
                open: true,
                active: 0,
                urbs: KVec::new(),
                next_token: 0,
            }),
            idle <- new_condvar!(),
        })
    }

    /// Takes an [`Io`] token, proving that I/O is permitted for as long as the token is held.
    ///
    /// Returns [`ENODEV`] once the window has been closed.
    pub fn enter(&self) -> Result<Io<'_>> {
        let mut state = self.state.lock();
        if !state.open {
            return Err(ENODEV);
        }
        state.active = state.active.checked_add(1).ok_or(EOVERFLOW)?;
        drop(state);

        Ok(Io { window: self })
    }

    /// The interface this window permits I/O on.
    pub fn interface(&self) -> &Interface {
        &self.interface
    }

    /// Closes the window and waits until no I/O is in flight.
    ///
    /// New [`Io`] tokens are refused immediately. Every URB belonging to a queue opened against
    /// this window is killed, which releases anything blocked waiting on one, and the call then
    /// blocks until the last outstanding token has been dropped.
    ///
    /// This is idempotent and sleeps, so the adapter only calls it from process context.
    fn close(&self) {
        let mut state = self.state.lock();
        state.open = false;

        // First wake token holders blocked in `recv()` or `flush()`. A token acquired before
        // `open` was cleared may still resubmit while unwinding, so this is only the wakeup pass.
        for reg in state.urbs.iter() {
            reg.canceller.cancel();
        }

        while state.active != 0 {
            self.idle.wait(&mut state);
        }

        // No token can submit after this point. Cancel once more to catch a final resubmission
        // made while an existing token was unwinding. Cancellation waits for the completion
        // callback, which takes no window lock. Holding the lock also keeps queue deregistration
        // from releasing a cancellation capability while it is used here.
        for reg in state.urbs.iter() {
            reg.canceller.cancel();
        }
    }

    /// Reopens a window that was closed by a suspend or pre-reset.
    ///
    /// The adapter only calls this after the USB core has re-permitted I/O.
    fn reopen(&self) {
        self.state.lock().open = true;
    }

    /// Registers `urb` as belonging to the queue identified by `token`, so [`close`] can cancel
    /// it.
    ///
    /// [`close`]: IoWindow::close
    fn register_urb(&self, token: u64, canceller: UrbCanceller) -> Result {
        let mut state = self.state.lock();
        if !state.open {
            return Err(ENODEV);
        }
        Ok(state
            .urbs
            .push(RegisteredUrb { token, canceller }, GFP_KERNEL)?)
    }

    /// Allocates a fresh queue token.
    fn new_token(&self) -> Result<u64> {
        let mut state = self.state.lock();
        if !state.open {
            return Err(ENODEV);
        }
        let token = state.next_token;
        state.next_token = state.next_token.checked_add(1).ok_or(EOVERFLOW)?;
        Ok(token)
    }

    /// Drops every URB registration made under `token`.
    fn deregister(&self, token: u64) {
        self.state.lock().urbs.retain(|reg| reg.token != token);
    }
}

/// Proof that USB I/O is currently permitted on an interface, and the handle through which every
/// transfer is issued.
///
/// Obtained from [`IoWindow::enter`] and released when dropped; [`IoWindow::close`] blocks until
/// every outstanding token is gone. Because the token borrows both the window and the interface,
/// no transfer can outlive either.
pub struct Io<'a> {
    window: &'a IoWindow,
}

impl Drop for Io<'_> {
    fn drop(&mut self) {
        let mut state = self.window.state.lock();
        state.active -= 1;
        if state.active == 0 {
            self.window.idle.notify_all();
        }
    }
}

impl<'a> Io<'a> {
    /// The interface this token permits I/O on.
    pub fn interface(&self) -> &Interface {
        self.window.interface()
    }

    /// Returns the interface in the bound context proven by this token.
    fn bound_interface(&self) -> &Interface<device::Bound> {
        // SAFETY: The adapter only opens an `IoWindow` after successful
        // probe/resume/reset completion and closes it before the interface
        // leaves the bound I/O state. Holding `Io` proves that window is
        // still open for this borrow.
        unsafe { &*(core::ptr::from_ref(self.window.interface()).cast()) }
    }

    /// The `struct usb_device` that interface belongs to.
    fn device(&self) -> *mut bindings::usb_device {
        // SAFETY: the window holds a reference to a valid `struct usb_interface`, and
        // `interface_to_usbdev()` returns its valid `struct usb_device`.
        unsafe { bindings::interface_to_usbdev(self.window.interface.as_raw()) }
    }

    /// Clears a halt/stall on `endpoint`, resetting both the device-side stall and the host-side
    /// data toggle. Sleeps.
    pub fn clear_halt<K: EndpointKind>(&self, endpoint: &Endpoint<K>) -> Result {
        let dev = self.device();

        // SAFETY: `dev` is valid; `usb_clear_halt()` only issues a control request and updates
        // host-side endpoint state.
        to_result(unsafe { bindings::usb_clear_halt(dev, endpoint.pipe().0 as kernel::ffi::c_int) })
    }

    /// Issues a synchronous bulk OUT transfer of `data`, returning the number of bytes
    /// transferred.
    ///
    /// `data` is copied into a kmalloc'd bounce buffer internally, so it need not be DMA-capable.
    /// `gfp` selects that buffer's allocation flags: pass `GFP_KERNEL` normally, or `GFP_NOIO` on
    /// a reset/resume or error-handling path. Sleeps.
    pub fn bulk_send(
        &self,
        endpoint: &Endpoint<BulkOut>,
        data: &[u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result<usize> {
        let mut actual: kernel::ffi::c_int = 0;
        let millis = timeout_millis(timeout)?;

        // `usb_bulk_msg()` DMAs straight from the buffer, and `data` may live on the stack or in
        // `.rodata`, so bounce it through a kmalloc'd allocation.
        let mut buf = KVec::with_capacity(data.len(), gfp)?;
        buf.extend_from_slice(data, gfp)?;
        let len = buf.len().try_into()?;

        let dev = self.device();
        // SAFETY: `dev` is valid; `buf` is a kmalloc'd buffer valid for reads of `len` bytes for
        // the duration of the call; `actual` is a valid out-pointer.
        to_result(unsafe {
            bindings::usb_bulk_msg(
                dev,
                endpoint.pipe().0,
                buf.as_mut_ptr().cast::<kernel::ffi::c_void>(),
                len,
                &mut actual,
                millis,
            )
        })?;

        Ok(actual as usize)
    }

    /// Issues a synchronous bulk IN transfer into `data`, returning the number of bytes received.
    ///
    /// The data is received into a kmalloc'd bounce buffer and copied out, so `data` need not be
    /// DMA-capable. Sleeps.
    pub fn bulk_recv(
        &self,
        endpoint: &Endpoint<BulkIn>,
        data: &mut [u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result<usize> {
        let mut actual: kernel::ffi::c_int = 0;
        let millis = timeout_millis(timeout)?;

        let mut buf = KVec::from_elem(0u8, data.len(), gfp)?;
        let len = buf.len().try_into()?;

        let dev = self.device();
        // SAFETY: `dev` is valid; `buf` is a kmalloc'd buffer valid for writes of `len` bytes for
        // the duration of the call; `actual` is a valid out-pointer.
        to_result(unsafe {
            bindings::usb_bulk_msg(
                dev,
                endpoint.pipe().0,
                buf.as_mut_ptr().cast::<kernel::ffi::c_void>(),
                len,
                &mut actual,
                millis,
            )
        })?;

        // `usb_bulk_msg()` never reports more than the requested length.
        let n = (actual as usize).min(data.len());
        data[..n].copy_from_slice(&buf[..n]);
        Ok(n)
    }

    /// Issues a synchronous interrupt IN transfer into `data`, returning the number of bytes
    /// received.
    ///
    /// As for [`bulk_recv`](Self::bulk_recv), the transfer is bounced through a kmalloc'd buffer.
    /// Sleeps.
    pub fn interrupt_recv(
        &self,
        endpoint: &Endpoint<InterruptIn>,
        data: &mut [u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result<usize> {
        let mut actual: kernel::ffi::c_int = 0;
        let millis = timeout_millis(timeout)?;

        let mut buf = KVec::from_elem(0u8, data.len(), gfp)?;
        let len = buf.len().try_into()?;

        let dev = self.device();
        // SAFETY: `dev` is valid; `buf` is a kmalloc'd buffer valid for writes of `len` bytes for
        // the duration of the call; `actual` is a valid out-pointer.
        to_result(unsafe {
            bindings::usb_interrupt_msg(
                dev,
                endpoint.pipe().0,
                buf.as_mut_ptr().cast::<kernel::ffi::c_void>(),
                len,
                &mut actual,
                millis,
            )
        })?;

        let n = (actual as usize).min(data.len());
        data[..n].copy_from_slice(&buf[..n]);
        Ok(n)
    }

    /// Issues a synchronous control OUT transfer on the default control endpoint.
    ///
    /// `request`, `request_type`, `value` and `index` are the `bRequest`, `bmRequestType`,
    /// `wValue` and `wIndex` setup fields. The buffer is copied internally, so `data` need not be
    /// DMA-capable. Sleeps.
    pub fn control_send(
        &self,
        request: u8,
        request_type: u8,
        value: u16,
        index: u16,
        data: &[u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result {
        let millis = timeout_millis(timeout)?;
        let len = data.len().try_into()?;

        // SAFETY: `self.device()` is valid; `data` is valid for reads of `len` bytes and
        // `usb_control_msg_send()` copies out of it before returning.
        to_result(unsafe {
            bindings::usb_control_msg_send(
                self.device(),
                0,
                request,
                request_type,
                value,
                index,
                data.as_ptr().cast::<kernel::ffi::c_void>(),
                len,
                millis,
                gfp.as_raw(),
            )
        })
    }

    /// Issues a synchronous control IN transfer on the default control endpoint, filling `data`
    /// with exactly `data.len()` bytes.
    ///
    /// The transfer fails if the device returns fewer bytes than requested. Sleeps.
    pub fn control_recv(
        &self,
        request: u8,
        request_type: u8,
        value: u16,
        index: u16,
        data: &mut [u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result {
        let millis = timeout_millis(timeout)?;
        let len = data.len().try_into()?;

        // SAFETY: `self.device()` is valid; `data` is valid for writes of `len` bytes and
        // `usb_control_msg_recv()` copies into it before returning.
        to_result(unsafe {
            bindings::usb_control_msg_recv(
                self.device(),
                0,
                request,
                request_type,
                value,
                index,
                data.as_mut_ptr().cast::<kernel::ffi::c_void>(),
                len,
                millis,
                gfp.as_raw(),
            )
        })
    }

    /// Selects alternate setting `alternate` of *this* interface (`SET_INTERFACE`).
    ///
    /// Unlike a device-wide `set_interface()`, this can only ever retarget the interface the
    /// driver is bound to: the interface number comes from the bound interface itself, not from
    /// the caller, so a driver cannot disturb a sibling interface of a composite device. Sleeps.
    pub fn set_alternate_setting(&self, alternate: u8) -> Result {
        let number = self.window.interface.number().ok_or(ENODEV)?;

        // SAFETY: `self.device()` is a valid `struct usb_device`, and `number` is the number of
        // one of its interfaces -- the one this driver is bound to.
        to_result(unsafe {
            bindings::usb_set_interface(self.device(), number.into(), alternate.into())
        })
    }
}

/// The I/O-window registration shared by both queue types.
///
/// Owning this as a separate field means a queue under construction already has a working `Drop`
/// before any per-slot allocation is attempted, so a mid-construction failure
/// cannot leave URBs registered with the window.
///
/// The window is held by [`Arc`] rather than borrowed because a queue normally lives in the
/// driver's device data, which has no lifetime to borrow from. That does not weaken the guarantee
/// that matters: every queue operation still requires an [`Io`] token, which [`IoWindow::close`]
/// stops issuing, and `close()` cancels the queue's URBs through its registration.
struct QueueRegistration {
    window: Arc<IoWindow>,
    token: u64,
}

impl QueueRegistration {
    fn new(window: &Arc<IoWindow>) -> Result<Self> {
        Ok(Self {
            token: window.new_token()?,
            window: window.clone(),
        })
    }

    fn register(&self, canceller: UrbCanceller) -> Result {
        self.window.register_urb(self.token, canceller)
    }

    /// Checks that `io` was taken from the same window this queue was opened against, so a queue
    /// cannot be driven using a token that proves nothing about *its* device's I/O state.
    fn check(&self, io: &Io<'_>) -> Result {
        if !core::ptr::eq(&*self.window, io.window) {
            return Err(EINVAL);
        }
        Ok(())
    }
}

impl Drop for QueueRegistration {
    fn drop(&mut self) {
        // Drop the window's records of this queue's URBs before the queue frees them, so a
        // concurrent `IoWindow::close()` can never see a freed URB.
        self.window.deregister(self.token);
    }
}

enum QueueUrb {
    Idle(Pin<UrbHandle<Completion, Idle>>),
    Active(UrbHandle<Completion, Active>),
}

/// One persistent queue slot built from the common typed URB abstraction.
struct UrbSlot {
    urb: Option<QueueUrb>,
    done: Arc<Completion>,
    capacity: usize,
}

impl UrbSlot {
    fn new(io: &Io<'_>, pipe: Pipe, buf_len: usize) -> Result<Self> {
        let buffer: Pin<KBox<[u8]>> = KBox::pin_slice(
            |_| {
                // SAFETY: The initializer writes one valid `u8` and cannot
                // fail after partially initializing the element.
                unsafe {
                    pin_init::pin_init_from_closure(|slot: *mut u8| {
                        slot.write(0);
                        Ok::<(), Error>(())
                    })
                }
            },
            buf_len,
            GFP_KERNEL,
        )?;
        let buffer = Pin::into_inner(buffer);
        let done = Arc::pin_init(Completion::new(), GFP_KERNEL)?;
        let urb = Urb::<Completion>::new_bulk(
            GFP_KERNEL,
            io.bound_interface(),
            pipe,
            buffer,
            Some(done.clone()),
            urb_signal_complete,
            TransferFlags::default(),
        )?;

        Ok(Self {
            urb: Some(QueueUrb::Idle(urb)),
            done,
            capacity: buf_len,
        })
    }

    fn canceller(&self) -> Result<UrbCanceller> {
        match self.urb.as_ref() {
            Some(QueueUrb::Idle(urb)) => Ok(urb.canceller()),
            Some(QueueUrb::Active(urb)) => Ok(urb.canceller()),
            None => Err(EIO),
        }
    }

    fn is_active(&self) -> bool {
        matches!(self.urb, Some(QueueUrb::Active(_)))
    }

    #[inline]
    fn wait(&self, timeout: Delta) -> bool {
        let millis = timeout.as_millis();
        let millis = if millis <= 0 {
            0
        } else {
            millis.try_into().unwrap_or(u32::MAX)
        };
        self.done
            .wait_for_completion_timeout(crate::time::msecs_to_jiffies(millis))
    }

    #[inline]
    fn finish(&mut self) -> Result<(i32, usize)> {
        let state = self.urb.take().ok_or(EIO)?;
        let active = match state {
            QueueUrb::Active(urb) => urb,
            idle @ QueueUrb::Idle(_) => {
                self.urb = Some(idle);
                return Err(EINVAL);
            }
        };
        let idle = active.into_idle();
        let status = idle.status();
        let actual = idle.inner().actual_length as usize;
        self.urb = Some(QueueUrb::Idle(idle));
        Ok((status, actual))
    }

    fn copy_from_buffer(&mut self, out: &mut [u8], len: usize) -> Result<usize> {
        let state = self.urb.take().ok_or(EIO)?;
        let idle = match state {
            QueueUrb::Idle(urb) => unsafe { Pin::into_inner_unchecked(urb) },
            active @ QueueUrb::Active(_) => {
                self.urb = Some(active);
                return Err(EBUSY);
            }
        };
        let n = len.min(out.len()).min(self.capacity);
        out[..n].copy_from_slice(&idle.transfer_buffer()[..n]);
        // SAFETY: The C URB allocation is stable independently of this handle.
        self.urb = Some(QueueUrb::Idle(unsafe { Pin::new_unchecked(idle) }));
        Ok(n)
    }

    fn prepare_transfer(&mut self, data: &[u8]) -> Result {
        let state = self.urb.take().ok_or(EIO)?;
        let mut idle = match state {
            QueueUrb::Idle(urb) => unsafe { Pin::into_inner_unchecked(urb) },
            active @ QueueUrb::Active(_) => {
                self.urb = Some(active);
                return Err(EBUSY);
            }
        };
        let result = if data.len() > self.capacity {
            Err(EMSGSIZE)
        } else {
            idle.transfer_buffer_mut()[..data.len()].copy_from_slice(data);
            idle.set_transfer_buffer_length(data.len())
        };
        // SAFETY: The C URB allocation is stable independently of this handle.
        self.urb = Some(QueueUrb::Idle(unsafe { Pin::new_unchecked(idle) }));
        result
    }

    fn submit(&mut self) -> Result {
        let state = self.urb.take().ok_or(EIO)?;
        let idle = match state {
            QueueUrb::Idle(urb) => urb,
            active @ QueueUrb::Active(_) => {
                self.urb = Some(active);
                return Err(EBUSY);
            }
        };

        match idle.submit_recoverable(GFP_KERNEL) {
            Ok(active) => {
                self.urb = Some(QueueUrb::Active(active));
                Ok(())
            }
            Err((error, idle)) => {
                self.urb = Some(QueueUrb::Idle(idle));
                Err(error)
            }
        }
    }
}

/// A persistently-queued asynchronous bulk IN reader. See [`Io::bulk_in_queue`].
///
/// [`recv`](Self::recv) waits for the next queued transfer, copies its data out and immediately
/// re-submits its URB, so the endpoint stays posted.
pub struct BulkInQueue {
    inner: QueueRegistration,
    slots: KVec<UrbSlot>,
    cursor: usize,
}

// SAFETY: The queue exclusively owns its device reference, URBs, buffers and completions. None is
// tied to the creating thread, every operation that mutates it takes `&mut self`, and `Drop` kills
// each URB before releasing the resources it refers to.
unsafe impl Send for BulkInQueue {}

impl BulkInQueue {
    /// Opens a persistently-queued asynchronous bulk IN reader on `endpoint`.
    ///
    /// Allocates `depth` URBs, each with its own `buf_len`-byte DMA buffer, and submits them all
    /// up front, so the controller keeps `depth` IN transfers posted to the device continuously.
    /// This differs from [`Io::bulk_recv`], which posts a single URB only for the duration of the
    /// call and so leaves the endpoint un-posted in between -- a window in which a device that
    /// pushes a large reply while the host is blocked on an OUT can deadlock the bus.
    ///
    /// `io` must have been taken from `window`; it proves I/O is permitted right now. The queue's
    /// URBs are registered with `window`, so [`IoWindow::close`] cancels them.
    ///
    /// Sleeps; must be called from process context.
    pub fn new(
        window: &Arc<IoWindow>,
        io: &Io<'_>,
        endpoint: &Endpoint<BulkIn>,
        depth: usize,
        buf_len: usize,
    ) -> Result<Self> {
        // A zero-depth queue has no slots, but `recv()` indexes slot zero and takes the cursor
        // modulo the slot count; reject it rather than divide by zero later.
        if depth == 0 || buf_len == 0 {
            return Err(EINVAL);
        }
        if !core::ptr::eq(&**window, io.window) {
            return Err(EINVAL);
        }

        let pipe = endpoint.pipe();

        // Build the queue -- which owns the device reference and whose `Drop` releases everything
        // allocated so far -- before any fallible per-slot work, so no early return can leak.
        let mut queue = Self {
            inner: QueueRegistration::new(window)?,
            slots: KVec::with_capacity(depth, GFP_KERNEL)?,
            cursor: 0,
        };

        for _ in 0..depth {
            let slot = UrbSlot::new(io, pipe, buf_len)?;
            queue.inner.register(slot.canceller()?)?;
            queue.slots.push(slot, GFP_KERNEL)?;
        }

        // Post every URB. On failure the queue's `Drop` kills and frees the rest.
        for slot in queue.slots.iter_mut() {
            slot.submit()?;
        }

        Ok(queue)
    }

    /// Waits up to `timeout` for the next queued IN transfer and copies up to `out.len()` bytes of
    /// it into `out`.
    ///
    /// Returns `Ok(Some(n))` when a transfer completed -- its URB is re-submitted before
    /// returning, so the endpoint stays posted -- `Ok(None)` on timeout with the URB still
    /// outstanding, or `Err` if the transfer or the re-submission failed.
    ///
    /// `io` must come from the same [`IoWindow`] this queue was opened against; it proves I/O is
    /// still permitted. Sleeps.
    pub fn recv(&mut self, io: &Io<'_>, out: &mut [u8], timeout: Delta) -> Result<Option<usize>> {
        self.inner.check(io)?;

        let i = self.cursor;

        // A previous re-submission may have failed, leaving this slot un-posted. Waiting on it
        // would block on a completion that can never fire, so re-post it first.
        if !self.slots[i].is_active() {
            self.slots[i].submit()?;
        }

        if !self.slots[i].wait(timeout) {
            // Still outstanding; leave it posted so a later call keeps waiting on the same slot.
            return Ok(None);
        }

        let (status, len) = self.slots[i].finish()?;
        let n = self.slots[i].copy_from_buffer(out, len)?;

        self.cursor = (i + 1) % self.slots.len();

        // Keep the endpoint posted, then report the completed transfer's status.
        let resubmit = self.slots[i].submit();
        if status != 0 {
            return Err(Error::from_errno(status));
        }
        resubmit?;

        Ok(Some(n))
    }
}

/// An asynchronous, pipelined bulk OUT writer. See [`Io::bulk_out_queue`].
///
/// [`send`](Self::send) round-robins over the slots, waiting only for the transfer that previously
/// used the slot it is about to reuse, so up to `depth - 1` transfers stay in flight while the
/// next is prepared. [`flush`](Self::flush) drains them all.
pub struct BulkOutQueue {
    inner: QueueRegistration,
    slots: KVec<UrbSlot>,
    cursor: usize,
}

// SAFETY: As for `BulkInQueue`, the queue exclusively owns everything it refers to and cancels
// every URB before releasing it.
unsafe impl Send for BulkOutQueue {}

impl BulkOutQueue {
    /// Opens an asynchronous, pipelined bulk OUT writer on `endpoint`.
    ///
    /// Pre-allocates `depth` URBs with `buf_len`-byte DMA buffers but submits none up front: an
    /// OUT URB carries caller data, so it is filled and submitted per [`send`](Self::send). This
    /// lets up to `depth` transfers be in flight at once, instead of [`Io::bulk_send`]'s
    /// block-per-transfer round trip.
    ///
    /// `io` must have been taken from `window`. Sleeps; must be called from process context.
    pub fn new(
        window: &Arc<IoWindow>,
        io: &Io<'_>,
        endpoint: &Endpoint<BulkOut>,
        depth: usize,
        buf_len: usize,
    ) -> Result<Self> {
        if depth == 0 || buf_len == 0 {
            return Err(EINVAL);
        }
        if !core::ptr::eq(&**window, io.window) {
            return Err(EINVAL);
        }

        let pipe = endpoint.pipe();

        let mut queue = Self {
            inner: QueueRegistration::new(window)?,
            slots: KVec::with_capacity(depth, GFP_KERNEL)?,
            cursor: 0,
        };

        for _ in 0..depth {
            let slot = UrbSlot::new(io, pipe, buf_len)?;
            queue.inner.register(slot.canceller()?)?;
            queue.slots.push(slot, GFP_KERNEL)?;
        }

        Ok(queue)
    }

    /// Reaps slot `i` if it has an outstanding transfer, returning that transfer's status.
    ///
    /// `Ok(false)` means nothing was outstanding.
    fn reap(&mut self, i: usize, timeout: Delta) -> Result<bool> {
        if !self.slots[i].is_active() {
            return Ok(false);
        }

        if !self.slots[i].wait(timeout) {
            // Leave it posted so a later call keeps waiting on it.
            return Err(ETIMEDOUT);
        }
        let (status, _) = self.slots[i].finish()?;
        if status != 0 {
            return Err(Error::from_errno(status));
        }

        Ok(true)
    }

    /// Reports whether the next `count` queue slots can be submitted without waiting.
    ///
    /// Completed slots are reaped and any transfer error is returned. This is useful when a
    /// higher-level protocol must not block halfway through a multi-URB record while waiting for
    /// endpoint progress; callers can defer the whole record and service its control plane first.
    #[inline]
    pub fn can_send_n(&mut self, io: &Io<'_>, count: usize) -> Result<bool> {
        self.inner.check(io)?;
        if count > self.slots.len() {
            return Ok(false);
        }
        for off in 0..count {
            let i = (self.cursor + off) % self.slots.len();
            if self.slots[i].is_active() {
                if !self.slots[i].wait(Delta::ZERO) {
                    return Ok(false);
                }
                // `wait_for_completion_timeout()` consumes the completion signal. Reap the URB
                // now rather than leaving `send()` to wait for the signal a second time.
                let (status, _) = self.slots[i].finish()?;
                if status != 0 {
                    return Err(Error::from_errno(status));
                }
            }
        }
        Ok(true)
    }

    /// Submits `data` as a bulk OUT transfer without waiting for it to complete.
    ///
    /// If the slot about to be reused still has a transfer outstanding, this blocks up to
    /// `timeout` reaping it and surfaces its error. `data` must be no longer than the queue's
    /// `buf_len`, else [`EMSGSIZE`].
    ///
    /// `io` must come from the same [`IoWindow`] this queue was opened against. Sleeps.
    pub fn send(&mut self, io: &Io<'_>, data: &[u8], timeout: Delta) -> Result {
        self.inner.check(io)?;

        let i = self.cursor;
        if data.len() > self.slots[i].capacity {
            return Err(EMSGSIZE);
        }

        // Free the slot if its previous transfer is still outstanding.
        self.reap(i, timeout)?;

        self.slots[i].prepare_transfer(data)?;

        self.slots[i].submit()?;
        self.cursor = (i + 1) % self.slots.len();

        Ok(())
    }

    /// Waits up to `timeout` for every outstanding transfer to complete, returning the first error
    /// encountered. Every slot is reaped regardless.
    ///
    /// `io` must come from the same [`IoWindow`] this queue was opened against. Sleeps.
    pub fn flush(&mut self, io: &Io<'_>, timeout: Delta) -> Result {
        self.inner.check(io)?;

        let mut first_err = Ok(());
        for i in 0..self.slots.len() {
            if let Err(e) = self.reap(i, timeout) {
                if first_err.is_ok() {
                    first_err = Err(e);
                }
            }
        }
        first_err
    }
}

/// Wake the process-context owner of a completed queue URB.
fn urb_signal_complete(result: UrbResult<'_, Completion>) {
    if let Some(done) = result.context() {
        done.complete();
    }
}

// SAFETY: `usb::Interface` is a transparent wrapper of `struct usb_interface`.
// The offset is guaranteed to point to a valid device field inside `usb::Interface`.
unsafe impl<Ctx: device::DeviceContext> device::AsBusDevice<Ctx> for Interface<Ctx> {
    const OFFSET: usize = offset_of!(bindings::usb_interface, dev);
}

// SAFETY: `Interface` is a transparent wrapper of a type that doesn't depend on
// `Interface`'s generic argument.
kernel::impl_device_context_deref!(unsafe { Interface });
kernel::impl_device_context_into_aref!(Interface);

impl<Ctx: device::DeviceContext> AsRef<device::Device<Ctx>> for Interface<Ctx> {
    fn as_ref(&self) -> &device::Device<Ctx> {
        // SAFETY: By the type invariant of `Self`, `self.as_raw()` is a pointer to a valid
        // `struct usb_interface`.
        let dev = unsafe { &raw mut ((*self.as_raw()).dev) };

        // SAFETY: `dev` points to a valid `struct device`.
        unsafe { device::Device::from_raw(dev) }
    }
}

impl<Ctx: device::DeviceContext> AsRef<Device<Ctx>> for Interface<Ctx> {
    fn as_ref(&self) -> &Device<Ctx> {
        // SAFETY: `self.as_raw()` is valid by the type invariants.
        let usb_dev = unsafe { bindings::interface_to_usbdev(self.as_raw()) };

        // SAFETY: For a valid `struct usb_interface` pointer, the above call to
        // `interface_to_usbdev()` guarantees to return a valid pointer to a `struct usb_device`.
        unsafe { &*(usb_dev.cast()) }
    }
}

// SAFETY: Instances of `Interface` are always reference-counted.
unsafe impl AlwaysRefCounted for Interface {
    fn inc_ref(&self) {
        // SAFETY: The invariants of `Interface` guarantee that `self.as_raw()`
        // returns a valid `struct usb_interface` pointer, for which we will
        // acquire a new refcount.
        unsafe { bindings::usb_get_intf(self.as_raw()) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The safety requirements guarantee that the refcount is non-zero.
        unsafe { bindings::usb_put_intf(obj.cast().as_ptr()) }
    }
}

// SAFETY: A `Interface` is always reference-counted and can be released from any thread.
unsafe impl Send for Interface {}

// SAFETY: It is safe to send a &Interface to another thread because we do not
// allow any mutation through a shared reference.
unsafe impl Sync for Interface {}

crate::impl_flags!(
    /// URB transfer flags.
    ///
    /// These correspond to the `URB_*` constants in `include/linux/usb.h`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct TransferFlags(u32);

    /// Represents a single URB transfer flag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TransferFlag {
        /// Short packet flag: return an error if the packet is shorter than
        /// expected.
        ShortNotOk = bindings::URB_SHORT_NOT_OK,
        /// Isochronous ASAP flag: schedule the isochronous transfer as soon as
        /// possible.
        IsoAsap = bindings::URB_ISO_ASAP,
        /// Do not perform a DMA mapping for the transfer buffer.
        NoTransferDmaMap = bindings::URB_NO_TRANSFER_DMA_MAP,
        /// Send a zero-length packet at the end of the transfer.
        ZeroPacket = bindings::URB_ZERO_PACKET,
        /// Do not interrupt the CPU when the URB completes.
        NoInterrupt = bindings::URB_NO_INTERRUPT,
    }
);

/// A USB pipe encoding endpoint type, direction, device address, and
/// endpoint number into a single `u32`.
///
/// Pipe encoding follows the kernel's `PIPE_*` macros used by the USB
/// core for control, bulk, isochronous, and interrupt transfers.
#[derive(Clone, Copy)]
pub struct Pipe(u32);

impl Pipe {
    /// Create a host-to-device (OUT) control pipe (endpoint 0).
    pub fn new_send_control_pipe<Ctx: device::DeviceContext>(dev: &Device<Ctx>) -> Self {
        Self(bindings::PIPE_CONTROL << 30 | dev.devnum() << 8)
    }

    /// Create a device-to-host (IN) control pipe (endpoint 0).
    pub fn new_receive_control_pipe<Ctx: device::DeviceContext>(dev: &Device<Ctx>) -> Self {
        Self(bindings::PIPE_CONTROL << 30 | dev.devnum() << 8 | bindings::USB_DIR_IN)
    }

    /// Create a device-to-host (IN) isochronous pipe.
    pub fn new_receive_isoc_pipe<Ctx: device::DeviceContext>(
        dev: &Device<Ctx>,
        endpoint: &HostEndpoint,
    ) -> Self {
        Self(
            bindings::PIPE_ISOCHRONOUS << 30
                | dev.devnum() << 8
                | u32::from(endpoint.endpoint_number()) << 15
                | bindings::USB_DIR_IN,
        )
    }

    /// Create a host-to-device (OUT) isochronous pipe.
    pub fn new_send_isoc_pipe<Ctx: device::DeviceContext>(
        dev: &Device<Ctx>,
        endpoint: &HostEndpoint,
    ) -> Self {
        Self(
            bindings::PIPE_ISOCHRONOUS << 30
                | dev.devnum() << 8
                | u32::from(endpoint.endpoint_number()) << 15,
        )
    }

    /// Create a host-to-device (OUT) bulk pipe.
    pub fn new_send_bulk_pipe<Ctx: device::DeviceContext>(
        dev: &Device<Ctx>,
        endpoint: &HostEndpoint,
    ) -> Self {
        Self(
            bindings::PIPE_BULK << 30
                | dev.devnum() << 8
                | u32::from(endpoint.endpoint_number()) << 15,
        )
    }

    /// Create a device-to-host (IN) bulk pipe.
    pub fn new_receive_bulk_pipe<Ctx: device::DeviceContext>(
        dev: &Device<Ctx>,
        endpoint: &HostEndpoint,
    ) -> Self {
        Self(
            bindings::PIPE_BULK << 30
                | dev.devnum() << 8
                | u32::from(endpoint.endpoint_number()) << 15
                | bindings::USB_DIR_IN,
        )
    }

    /// Create a host-to-device (OUT) interrupt pipe.
    pub fn new_send_int_pipe<Ctx: device::DeviceContext>(
        dev: &Device<Ctx>,
        endpoint: &HostEndpoint,
    ) -> Self {
        Self(
            bindings::PIPE_INTERRUPT << 30
                | dev.devnum() << 8
                | u32::from(endpoint.endpoint_number()) << 15,
        )
    }

    /// Create a device-to-host (IN) interrupt pipe.
    pub fn new_receive_int_pipe<Ctx: device::DeviceContext>(
        dev: &Device<Ctx>,
        endpoint: &HostEndpoint,
    ) -> Self {
        Self(
            bindings::PIPE_INTERRUPT << 30
                | dev.devnum() << 8
                | u32::from(endpoint.endpoint_number()) << 15
                | bindings::USB_DIR_IN,
        )
    }
}

/// A single isochronous packet descriptor within an URB.
///
/// Wraps `struct usb_iso_packet_descriptor` from the C USB core.
#[repr(transparent)]
pub struct IsoPacketDescriptor(bindings::usb_iso_packet_descriptor);

impl IsoPacketDescriptor {
    /// Returns the offset of the packet's data within the transfer buffer.
    pub fn offset(&self) -> u32 {
        self.0.offset
    }

    /// Returns the length of the packet in bytes.
    pub fn length(&self) -> u32 {
        self.0.length
    }

    /// Returns the actual number of bytes transferred in this packet.
    ///
    /// Valid only after the URB completes.
    pub fn actual_length(&self) -> u32 {
        self.0.actual_length
    }

    /// Returns the per-packet completion status.
    ///
    /// Valid only after the URB completes.
    pub fn status(&self) -> i32 {
        self.0.status
    }
}

/// Trait implemented by all URB state marker types.
///
/// Each state specifies pre-cleanup behaviour that runs before the
/// underlying allocation is freed.
pub trait UrbState {
    /// Called before the URB allocation is freed.
    fn pre_drop(urb: &mut bindings::urb);
}

/// Marker type for an idle (unsubmitted) URB.
pub struct Idle;
/// Marker type for an active (submitted, in-flight) URB.
pub struct Active;

impl UrbState for Idle {
    fn pre_drop(_urb: &mut bindings::urb) {}
}
impl UrbState for Active {
    fn pre_drop(urb: &mut bindings::urb) {
        // SAFETY: `urb` is a valid pointer to an initialized `struct urb`.
        unsafe { bindings::usb_kill_urb(urb) }
    }
}

/// A USB Request Block (URB).
///
/// This structure wraps the C [`struct urb`] and provides a safe
/// abstraction for USB transfers.
///
/// [`struct urb`]: https://www.kernel.org/doc/html/latest/driver-api/usb/usb.html#c.urb
#[repr(transparent)]
pub struct Urb<T>(Opaque<bindings::urb>, PhantomData<T>);

impl<T> Urb<T> {
    fn as_raw(&self) -> *mut bindings::urb {
        self.0.get()
    }

    fn inner(&self) -> &bindings::urb {
        // SAFETY: The type invariants guarantee that `self.0` wraps a valid
        // `struct urb`.
        unsafe { &*self.as_raw() }
    }

    fn status(&self) -> i32 {
        self.inner().status
    }

    fn canceller(&self) -> UrbCanceller {
        // SAFETY: `self` is a live URB. Take an additional USB-core
        // reference for the cancellation capability.
        let urb = unsafe { bindings::usb_get_urb(self.as_raw()) };
        // `usb_get_urb()` returns its non-null argument.
        UrbCanceller(unsafe { NonNull::new_unchecked(urb) })
    }

    /// Returns a borrow of the driver-private context data, if any.
    pub fn context(&self) -> Option<ArcBorrow<'_, T>> {
        let context = self.inner().context;
        if context.is_null() {
            None
        } else {
            // SAFETY: `context` was initialized by `Arc::into_raw` in `init_common`.
            Some(unsafe { ArcBorrow::from_raw(context.cast()) })
        }
    }
}

/// A handle to a [`struct urb`] allocated via `usb_alloc_urb`.
///
/// Created by [`Urb::new_bulk`], [`Urb::new_isoc`], etc. The URB is
/// owned by this handle — dropping the handle frees the allocation.
///
/// Use [`Urb::submit`] to transition to [`UrbHandle<T, Active>`].
pub struct UrbHandle<T, S: UrbState = Idle> {
    /// Pointer to the underlying C `struct urb`.
    urb: NonNull<bindings::urb>,
    /// Size of the allocation backing `transfer_buffer`.
    transfer_buffer_capacity: usize,
    /// State marker.
    _state: PhantomData<S>,
    /// Type of driver-private context data.
    _ty: PhantomData<T>,
}

// SAFETY: The underlying URB is reference-counted and may be released from
// any thread. The context follows the same `Send + Sync` requirements as
// `Arc<T>`.
unsafe impl<T: Send + Sync, S: UrbState> Send for UrbHandle<T, S> {}

/// A reference-counted capability which can only cancel an URB.
///
/// Queue registries keep this narrow handle so they can stop transfers during
/// disconnect without gaining access to the URB or its transfer buffer.
struct UrbCanceller(NonNull<bindings::urb>);

// SAFETY: USB core reference-counts URBs and permits `usb_kill_urb()` from any
// process context.
unsafe impl Send for UrbCanceller {}
// SAFETY: `cancel()` does not mutate Rust-owned state and USB core serializes
// cancellation of an URB.
unsafe impl Sync for UrbCanceller {}

impl UrbCanceller {
    fn cancel(&self) {
        // SAFETY: This capability owns a reference to a live URB.
        unsafe { bindings::usb_kill_urb(self.0.as_ptr()) };
    }
}

impl Drop for UrbCanceller {
    fn drop(&mut self) {
        // SAFETY: Release the reference acquired by `Urb::canceller()`.
        unsafe { bindings::usb_free_urb(self.0.as_ptr()) };
    }
}

impl<T, S: UrbState> Deref for UrbHandle<T, S> {
    type Target = Urb<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `Urb<T>` is a `#[repr(transparent)]` wrapper of `struct urb`,
        unsafe { &*(self.urb.as_ptr() as *const Urb<T>) }
    }
}

impl<T> UrbHandle<T, Idle> {
    /// Returns the entire transfer-buffer allocation for an idle URB.
    ///
    /// The idle state proves that USB core cannot access the buffer while the
    /// shared slice exists.
    pub fn transfer_buffer(&self) -> &[u8] {
        if self.transfer_buffer_capacity == 0 {
            return &[];
        }
        // SAFETY: The URB is idle, its transfer buffer was allocated for
        // `transfer_buffer_capacity` bytes in `init_common()`.
        unsafe {
            slice::from_raw_parts(
                (*self.urb.as_ptr()).transfer_buffer.cast(),
                self.transfer_buffer_capacity,
            )
        }
    }

    /// Returns the entire transfer-buffer allocation for an idle URB.
    ///
    /// The idle state proves that USB core cannot access the buffer while the
    /// mutable slice exists.
    pub fn transfer_buffer_mut(&mut self) -> &mut [u8] {
        if self.transfer_buffer_capacity == 0 {
            return &mut [];
        }
        // SAFETY: The URB is idle, its transfer buffer was allocated for
        // `transfer_buffer_capacity` bytes in `init_common()`, and `&mut self`
        // grants exclusive access for the returned borrow.
        unsafe {
            slice::from_raw_parts_mut(
                (*self.urb.as_ptr()).transfer_buffer.cast(),
                self.transfer_buffer_capacity,
            )
        }
    }

    /// Sets the number of transfer-buffer bytes used by the next submission.
    pub fn set_transfer_buffer_length(&mut self, len: usize) -> Result {
        if len > self.transfer_buffer_capacity {
            return Err(EMSGSIZE);
        }
        let len = len.try_into()?;
        // SAFETY: The URB is idle and `len` is within its backing allocation.
        unsafe { (*self.urb.as_ptr()).transfer_buffer_length = len };
        Ok(())
    }
}

impl<T> UrbHandle<T, Active> {
    /// Cancel any outstanding transfer and recover an idle, reusable handle.
    pub fn into_idle(self) -> Pin<UrbHandle<T, Idle>> {
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: The active handle owns a live URB. `usb_kill_urb()` waits
        // until its completion callback has returned.
        unsafe { bindings::usb_kill_urb(this.urb.as_ptr()) };

        let handle = UrbHandle {
            urb: this.urb,
            transfer_buffer_capacity: this.transfer_buffer_capacity,
            _state: PhantomData,
            _ty: PhantomData,
        };
        // SAFETY: The C URB allocation is stable independently of the Rust
        // handle's address.
        unsafe { Pin::new_unchecked(handle) }
    }
}

impl<T, S: UrbState> Drop for UrbHandle<T, S> {
    fn drop(&mut self) {
        // SAFETY: `self.as_raw()` points to a valid, initialized C `struct urb`.
        let urb: &mut bindings::urb = unsafe { &mut *self.as_raw() };
        S::pre_drop(urb);

        if !urb.context.is_null() {
            // SAFETY: After `pre_drop` the URB is idle, so it is safe to
            // reclaim the context data.
            unsafe {
                drop(Arc::from_raw(urb.context.cast::<T>()));
            }
        }

        if !urb.setup_packet.is_null() {
            // SAFETY: The setup packet was allocated via `KBox::into_raw` in
            // `init_common` and `urb.setup_packet` is still valid.
            unsafe {
                drop(KBox::from_raw(urb.setup_packet.cast::<CtrlRequest>()));
            }
        }

        if !urb.transfer_buffer.is_null() {
            // SAFETY: The transfer buffer was allocated via `KBox::into_raw` in
            // `init_common` and `urb.transfer_buffer` is still valid.
            unsafe {
                drop(KBox::from_raw(ptr::slice_from_raw_parts_mut(
                    urb.transfer_buffer.cast::<u8>(),
                    self.transfer_buffer_capacity,
                )));
            }
        }

        // SAFETY: `urb` points to a valid, initialized `struct urb`
        // and is not in-flight.
        unsafe { bindings::usb_free_urb(ptr::from_mut(urb)) };
    }
}

/// A completed URB whose status must be checked before accessing data.
///
/// The driver receives this in its completion handler. Call
/// [`check`](UrbResult::check) to verify the transfer succeeded.
pub struct UrbResult<'a, T> {
    /// The pinned URB reference delivered by the trampoline.
    urb: Pin<&'a mut Urb<T>>,
}

impl<'a, T> Deref for UrbResult<'a, T> {
    type Target = Urb<T>;

    fn deref(&self) -> &Self::Target {
        &self.urb
    }
}

impl<'a, T> UrbResult<'a, T> {
    /// Re-submit the URB from a completion handler.
    ///
    /// Consumes this handle, transferring ownership to the kernel.
    /// This is intentionally private, since a driver should always
    /// check the result in the completion handler.
    fn resubmit(self, mem_flags: kernel::alloc::Flags) -> Result {
        // SAFETY: `self.urb.as_raw()` points to a valid, initialized C `struct urb`.
        to_result(unsafe { bindings::usb_submit_urb(self.as_raw(), mem_flags.as_raw()) })
    }

    /// Check the completion status and grant access to the URB data.
    pub fn check(&mut self) -> Result<UrbData<'_, T>> {
        if self.status() != 0 {
            Err(Error::from_errno(self.status()))
        } else {
            Ok(UrbData {
                urb: self.urb.as_mut(),
            })
        }
    }

    /// Check the completion status, granting data access on success or
    /// resubmitting the URB on failure.
    pub fn check_or_resubmit(
        self,
        mem_flags: kernel::alloc::Flags,
    ) -> Result<UrbData<'a, T>, Result> {
        if self.status() != 0 {
            Err(self.resubmit(mem_flags))
        } else {
            Ok(UrbData { urb: self.urb })
        }
    }
}

/// A successfully completed URB whose data is safe to read.
pub struct UrbData<'a, T> {
    /// The pinned URB reference.
    urb: Pin<&'a mut Urb<T>>,
}

impl<'a, T> Deref for UrbData<'a, T> {
    type Target = Urb<T>;

    fn deref(&self) -> &Self::Target {
        &self.urb
    }
}

impl<'a, T> UrbData<'a, T> {
    /// Returns the number of bytes actually transferred.
    ///
    /// For isochronous URBs this is the sum of all packet
    /// `actual_length` values.
    pub fn actual_length(&self) -> u32 {
        self.inner().actual_length
    }

    /// Returns the transfer buffer as a byte slice.
    pub fn transfer_buffer(&self) -> &[u8] {
        let urb = self.inner();
        if urb.transfer_buffer.is_null() {
            &[]
        } else {
            // SAFETY: The transfer buffer was set in `init_common`.
            // The pointer and length are valid for the lifetime of the `Urb`.
            unsafe {
                slice::from_raw_parts(
                    urb.transfer_buffer as *const u8,
                    urb.transfer_buffer_length as usize,
                )
            }
        }
    }

    /// Returns the ISO frame descriptors for this URB.
    pub fn iso_frame_descs(&self) -> &[IsoPacketDescriptor] {
        let urb = self.inner();

        if urb.number_of_packets == 0 {
            &[]
        } else {
            let data = urb.iso_frame_desc.as_ptr().cast::<IsoPacketDescriptor>();

            // SAFETY: The `iso_frame_desc` flexible array was allocated as
            // part of the `usb_alloc_urb` allocation. `number_of_packets`
            // is the corresponding length.
            unsafe { slice::from_raw_parts(data, urb.number_of_packets as usize) }
        }
    }

    /// Extracts the payload data for a given ISO packet descriptor.
    ///
    /// Returns `Err` if the packet status is non-zero.
    pub fn data_from_iso_packet_desc(
        &self,
        iso_packet_desc: &IsoPacketDescriptor,
    ) -> Result<&[u8]> {
        if iso_packet_desc.status() != 0 {
            return Err(Error::from_errno(iso_packet_desc.status()));
        }
        let urb = self.inner();
        // SAFETY: `iso_packet_desc.offset()` was computed in
        // `init_common` and lies within the transfer buffer.
        let data =
            unsafe { (urb.transfer_buffer.cast::<u8>()).add(iso_packet_desc.offset() as usize) };

        // SAFETY: After URB completion `actual_length()` reflects the
        // valid bytes in the packet. The slice is within the transfer
        // buffer allocation. The packet status was verified above.
        unsafe {
            Ok(slice::from_raw_parts(
                data,
                iso_packet_desc.actual_length() as usize,
            ))
        }
    }

    /// Re-submit the URB from a completion handler.
    ///
    /// Consumes this handle, transferring ownership to the kernel.
    pub fn resubmit(self, mem_flags: kernel::alloc::Flags) -> Result {
        // SAFETY: `self.as_raw()` points to a valid, initialized C `struct urb`.
        to_result(unsafe { bindings::usb_submit_urb(self.as_raw(), mem_flags.as_raw()) })
    }
}

/// Trampoline function to call safe completion handlers.
///
/// # Safety
///
/// `urb_ptr` must point to a valid, initialized `struct urb` whose
/// `context` and `rust_complete` fields were set by [`Urb::init_common`].
unsafe extern "C" fn urb_complete_trampoline<T>(urb_ptr: *mut bindings::urb) {
    // SAFETY: `urb_ptr` is a valid pointer provided by the USB core.
    // `rust_complete` was set to a `fn(UrbResult<'_, T>)` when initialized.
    let complete: fn(UrbResult<'_, T>) = unsafe { core::mem::transmute((*urb_ptr).rust_complete) };
    // SAFETY: `urb_ptr` points to a valid `struct urb`.
    let urb = unsafe { &mut *urb_ptr.cast() };
    // SAFETY: The data `urb` references is never moved.
    let urb = unsafe { Pin::new_unchecked(urb) };
    complete(UrbResult { urb });
}

impl<T> Urb<T> {
    #[allow(clippy::too_many_arguments)]
    fn init_common(
        mem_flags: kernel::alloc::Flags,
        intf: &Interface<device::Bound>,
        pipe: Pipe,
        setup_packet: Option<KBox<CtrlRequest>>,
        transfer_buffer: Option<KBox<[u8]>>,
        context_data: Option<Arc<T>>,
        complete: fn(UrbResult<'_, T>),
        number_of_packets: u32,
        iso_packet_len: u16,
        transfer_flags: TransferFlags,
        interval: i32,
    ) -> Result<Pin<UrbHandle<T, Idle>>> {
        let transfer_buffer_capacity = transfer_buffer.as_ref().map_or(0, |buffer| buffer.len());

        // SAFETY: `usb_alloc_urb` allocates a `struct urb` + ISO frame.
        let urb_ptr =
            unsafe { bindings::usb_alloc_urb(number_of_packets as c_int, mem_flags.as_raw()) };
        if urb_ptr.is_null() {
            return Err(ENOMEM);
        }

        // SAFETY: `urb_ptr` points to allocated and zero-initialized memory
        // of the correct layout for `struct urb` + ISO tail.
        let urb = unsafe { &mut *urb_ptr };

        let dev: &Device<device::Bound> = intf.as_ref();

        urb.complete = Some(urb_complete_trampoline::<T>);
        urb.dev = dev.as_raw();
        urb.pipe = pipe.0;
        urb.number_of_packets = number_of_packets as c_int;
        urb.transfer_flags = u32::from(transfer_flags);
        urb.interval = interval;

        // Set up ISO frame descriptors.
        if number_of_packets > 0 {
            // SAFETY: `urb_ptr` was allocated with `number_of_packets` ISO
            // descriptors via `usb_alloc_urb`. `as_mut_slice` yields a valid
            // mutable slice of that length.
            let descs = unsafe { urb.iso_frame_desc.as_mut_slice(number_of_packets as usize) };
            for (i, desc) in descs.iter_mut().enumerate() {
                let pkt_len = u32::from(iso_packet_len);
                desc.offset = (i as u32) * pkt_len;
                desc.length = pkt_len;
            }
        }

        if let Some(sp) = setup_packet {
            urb.setup_packet = KBox::into_raw(sp).cast::<u8>();
        }

        if let Some(tb) = transfer_buffer {
            let len = tb.len();
            urb.transfer_buffer_length = len as u32;
            urb.transfer_buffer = KBox::into_raw(tb).cast::<core::ffi::c_void>();
        }

        if let Some(data) = context_data {
            urb.context = Arc::into_raw(data).cast_mut().cast();
        }

        urb.rust_complete = complete as *mut core::ffi::c_void;

        let urb_handle = UrbHandle {
            // SAFETY: `urb_ptr` is guaranteed non-null by the null check above.
            urb: unsafe { NonNull::new_unchecked(urb_ptr) },
            transfer_buffer_capacity,
            _state: PhantomData,
            _ty: PhantomData,
        };

        // SAFETY: `urb_handle.urb` is never moved.
        Ok(unsafe { Pin::new_unchecked(urb_handle) })
    }

    /// Submit the URB for execution.
    ///
    /// On success the caller receives an [`UrbHandle<T, Active>`] which
    /// holds the resources for the in-flight URB. Dropping it cancels the
    /// URB and frees the allocation.
    pub fn submit(
        self: Pin<UrbHandle<T, Idle>>,
        mem_flags: kernel::alloc::Flags,
    ) -> Result<UrbHandle<T, Active>> {
        self.submit_recoverable(mem_flags)
            .map_err(|(error, _handle)| error)
    }

    /// Submit the URB while returning the idle handle when submission fails.
    ///
    /// Queue implementations use this variant so a transient submission
    /// error does not discard a preallocated URB and its transfer buffer.
    pub fn submit_recoverable(
        self: Pin<UrbHandle<T, Idle>>,
        mem_flags: kernel::alloc::Flags,
    ) -> core::result::Result<UrbHandle<T, Active>, (Error, Pin<UrbHandle<T, Idle>>)> {
        // SAFETY: The urb pointed to is not moved.
        let handle = unsafe { Pin::into_inner_unchecked(self) };
        // SAFETY: `handle.as_raw()` points to a valid, initialized `struct urb`.
        let result = unsafe { bindings::usb_submit_urb(handle.as_raw(), mem_flags.as_raw()) };

        if result == 0 {
            let urb = handle.urb;
            let transfer_buffer_capacity = handle.transfer_buffer_capacity;
            core::mem::forget(handle);
            Ok(UrbHandle {
                urb,
                transfer_buffer_capacity,
                _state: PhantomData,
                _ty: PhantomData,
            })
        } else {
            // SAFETY: Submission failed, so USB core did not take ownership
            // and the handle remains idle and reusable.
            let handle = unsafe { Pin::new_unchecked(handle) };
            Err((Error::from_errno(result), handle))
        }
    }

    /// Creates a new bulk URB.
    pub fn new_bulk(
        mem_flags: kernel::alloc::Flags,
        intf: &Interface<device::Bound>,
        pipe: Pipe,
        transfer_buffer: KBox<[u8]>,
        context_data: Option<Arc<T>>,
        complete: fn(UrbResult<'_, T>),
        transfer_flags: TransferFlags,
    ) -> Result<Pin<UrbHandle<T, Idle>>> {
        Self::init_common(
            mem_flags,
            intf,
            pipe,
            None,
            Some(transfer_buffer),
            context_data,
            complete,
            0,
            0,
            transfer_flags,
            0,
        )
    }

    /// Creates a new interrupt URB.
    #[allow(clippy::too_many_arguments)]
    pub fn new_int(
        mem_flags: kernel::alloc::Flags,
        intf: &Interface<device::Bound>,
        pipe: Pipe,
        transfer_buffer: KBox<[u8]>,
        context_data: Option<Arc<T>>,
        complete: fn(UrbResult<'_, T>),
        transfer_flags: TransferFlags,
        interval: i32,
    ) -> Result<Pin<UrbHandle<T, Idle>>> {
        Self::init_common(
            mem_flags,
            intf,
            pipe,
            None,
            Some(transfer_buffer),
            context_data,
            complete,
            0,
            0,
            transfer_flags,
            interval,
        )
    }

    /// Creates a new control URB.
    #[allow(clippy::too_many_arguments)]
    pub fn new_ctrl(
        mem_flags: kernel::alloc::Flags,
        intf: &Interface<device::Bound>,
        pipe: Pipe,
        setup_packet: KBox<CtrlRequest>,
        transfer_buffer: Option<KBox<[u8]>>,
        context_data: Option<Arc<T>>,
        complete: fn(UrbResult<'_, T>),
        transfer_flags: TransferFlags,
    ) -> Result<Pin<UrbHandle<T, Idle>>> {
        Self::init_common(
            mem_flags,
            intf,
            pipe,
            Some(setup_packet),
            transfer_buffer,
            context_data,
            complete,
            0,
            0,
            transfer_flags,
            0,
        )
    }

    /// Creates a new isochronous URB.
    #[allow(clippy::too_many_arguments)]
    pub fn new_isoc(
        mem_flags: kernel::alloc::Flags,
        intf: &Interface<device::Bound>,
        pipe: Pipe,
        transfer_buffer: KBox<[u8]>,
        context_data: Option<Arc<T>>,
        complete: fn(UrbResult<'_, T>),
        number_of_packets: u32,
        iso_packet_len: u16,
        transfer_flags: TransferFlags,
        interval: i32,
    ) -> Result<Pin<UrbHandle<T, Idle>>> {
        // Reject URBs whose buffer is too small to hold all packets.
        let needed = (number_of_packets as usize).saturating_mul(iso_packet_len as usize);
        if transfer_buffer.len() < needed {
            return Err(EINVAL);
        }

        Self::init_common(
            mem_flags,
            intf,
            pipe,
            None,
            Some(transfer_buffer),
            context_data,
            complete,
            number_of_packets,
            iso_packet_len,
            transfer_flags,
            interval,
        )
    }
}

/// A USB device.
///
/// This structure represents the Rust abstraction for a C [`struct usb_device`].
/// The implementation abstracts the usage of a C [`struct usb_device`] passed in
/// from the C side.
///
/// # Invariants
///
/// A [`Device`] instance represents a valid [`struct usb_device`] created by the C portion of the
/// kernel.
///
/// [`struct usb_device`]: https://www.kernel.org/doc/html/latest/driver-api/usb/usb.html#c.usb_device
#[repr(transparent)]
pub struct Device<Ctx: device::DeviceContext = device::Normal>(
    Opaque<bindings::usb_device>,
    PhantomData<Ctx>,
);

impl<Ctx: device::DeviceContext> Device<Ctx> {
    fn as_raw(&self) -> *mut bindings::usb_device {
        self.0.get()
    }

    fn inner(&self) -> &bindings::usb_device {
        // SAFETY: The type invariants guarantee that `self.0` wraps a valid
        // `struct usb_device`.
        unsafe { &*self.as_raw() }
    }

    /// Returns the USB device number assigned by the bus.
    fn devnum(&self) -> u32 {
        self.inner().devnum as u32
    }

    /// Returns the `idVendor` of the device descriptor.
    pub fn vendor_id(&self) -> u16 {
        self.inner().descriptor.idVendor
    }

    /// Returns the `idProduct` of the device descriptor.
    pub fn product_id(&self) -> u16 {
        self.inner().descriptor.idProduct
    }

    /// Returns the `bcdDevice` of the device descriptor.
    ///
    /// Vendors conventionally use this as the device revision, and it is the only version a driver
    /// can read without speaking the device's own protocol.
    pub fn bcd_device(&self) -> u16 {
        self.inner().descriptor.bcdDevice
    }

    /// Returns the `bcdUSB` of the device descriptor.
    pub fn bcd_usb(&self) -> u16 {
        self.inner().descriptor.bcdUSB
    }

    /// Returns the enumerated bus speed as a human-readable string.
    pub fn speed_str(&self) -> &'static str {
        match self.inner().speed {
            bindings::usb_device_speed_USB_SPEED_LOW => "low (1.5 Mbps)",
            bindings::usb_device_speed_USB_SPEED_FULL => "full (12 Mbps)",
            bindings::usb_device_speed_USB_SPEED_HIGH => "high (480 Mbps)",
            bindings::usb_device_speed_USB_SPEED_WIRELESS => "wireless",
            bindings::usb_device_speed_USB_SPEED_SUPER => "super (5 Gbps)",
            bindings::usb_device_speed_USB_SPEED_SUPER_PLUS => "super-plus (10+ Gbps)",
            _ => "unknown",
        }
    }

    /// Returns the device's `iManufacturer` string, if the core cached one.
    pub fn manufacturer(&self) -> Option<&CStr> {
        // SAFETY: `manufacturer` is either null or a NUL-terminated string owned by the USB core
        // for as long as the device exists, which outlives the borrow of `self`.
        unsafe { Self::opt_cstr(self.inner().manufacturer) }
    }

    /// Returns the device's `iProduct` string, if the core cached one.
    pub fn product(&self) -> Option<&CStr> {
        // SAFETY: As for `manufacturer`.
        unsafe { Self::opt_cstr(self.inner().product) }
    }

    /// Returns the device's `iSerialNumber` string, if the core cached one.
    pub fn serial(&self) -> Option<&CStr> {
        // SAFETY: As for `manufacturer`.
        unsafe { Self::opt_cstr(self.inner().serial) }
    }

    /// # Safety
    ///
    /// `p` must be null or point to a NUL-terminated string that outlives `'a`.
    unsafe fn opt_cstr<'a>(p: *mut crate::ffi::c_char) -> Option<&'a CStr> {
        if p.is_null() {
            return None;
        }
        // SAFETY: The caller guarantees `p` is a NUL-terminated string valid for `'a`.
        Some(unsafe { CStr::from_char_ptr(p) })
    }
}

impl Device<device::Bound> {
    /// Select an alternate setting for the given interface.
    ///
    /// On success the device switches the given interface to the given alternate setting,
    /// which may change the set of active endpoints.
    pub fn set_interface(&self, interface: u8, altsetting: u8) -> Result {
        // SAFETY: `self.as_raw()` is a valid `struct usb_device` pointer by the type
        // invariants. `usb_set_interface` is safe to call on a bound device.
        to_result(unsafe {
            bindings::usb_set_interface(self.as_raw(), i32::from(interface), i32::from(altsetting))
        })
    }

    /// Send a USB control message synchronously.
    ///
    /// Wraps `usb_control_msg`. The pipe direction is inferred from the setup
    /// packet's [`Direction`]. The optional `data` buffer is
    /// written to for IN transfers or read from for OUT transfers.
    ///
    /// Returns the number of bytes transferred on success.
    pub fn control_msg(
        &self,
        setup: &CtrlRequest,
        data: Option<&mut [u8]>,
        timeout: Delta,
    ) -> Result<i32> {
        let pipe = match setup.direction() {
            Direction::In => Pipe::new_receive_control_pipe(self),
            Direction::Out => Pipe::new_send_control_pipe(self),
        };
        let (buf, len) = match data {
            Some(d) => (d.as_mut_ptr().cast::<core::ffi::c_void>(), d.len() as u16),
            None => (ptr::null_mut(), 0),
        };
        let timeout_ms = timeout.as_millis() as i32;

        // SAFETY: `self.as_raw()` returns a valid `struct usb_device` pointer.
        let ret = unsafe {
            bindings::usb_control_msg(
                self.as_raw(),
                pipe.0,
                setup.request(),
                setup.requesttype(),
                setup.value(),
                setup.index(),
                buf,
                len,
                timeout_ms,
            )
        };

        if ret >= 0 {
            Ok(ret)
        } else {
            Err(Error::from_errno(ret))
        }
    }
}

impl<Ctx: device::DeviceContext> Device<Ctx> {
    /// Return the root-first USB bus/port path for this device.
    ///
    /// The first element is the bus number and each following element is a downstream port. Returns
    /// `None` when the topology is deeper than `N`.
    pub fn topology_path<const N: usize>(&self) -> Option<([u32; N], usize)> {
        let mut path = [0u32; N];
        let mut len = 0usize;
        let mut current = self.as_raw();

        while !current.is_null() {
            if len == N {
                return None;
            }
            // SAFETY: `current` starts as this live device and then follows its parent chain.
            let mut port = unsafe { (*current).portnum } as u32;
            if port == 0 {
                // SAFETY: Every live USB device has a live bus.
                port = unsafe { (*(*current).bus).busnum } as u32;
            }
            path[len] = port;
            len += 1;
            // SAFETY: The parent pointer belongs to the same live USB topology.
            current = unsafe { (*current).parent };
        }

        path[..len].reverse();
        Some((path, len))
    }
}

struct DeviceSearch<F> {
    predicate: F,
    found: Option<ARef<Device>>,
}

unsafe extern "C" fn find_device_callback<F>(
    usb: *mut bindings::usb_device,
    data: *mut core::ffi::c_void,
) -> core::ffi::c_int
where
    F: FnMut(&Device) -> bool,
{
    // SAFETY: `find_device` passes a live `DeviceSearch<F>` and the USB core passes a live device
    // while holding the topology iteration lock.
    let search = unsafe { &mut *data.cast::<DeviceSearch<F>>() };
    // SAFETY: `Device` is transparent over `usb_device`, which is live for this callback.
    let device = unsafe { &*usb.cast::<Device>() };
    if (search.predicate)(device) {
        search.found = Some(ARef::from(device));
        1
    } else {
        0
    }
}

/// Find the first USB device matching `predicate` and return an owned reference to it.
pub fn find_device<F>(predicate: F) -> Option<ARef<Device>>
where
    F: FnMut(&Device) -> bool,
{
    let mut search = DeviceSearch {
        predicate,
        found: None,
    };
    // SAFETY: The callback and context types match, and the call is synchronous.
    unsafe {
        bindings::usb_for_each_dev(
            core::ptr::from_mut(&mut search).cast(),
            Some(find_device_callback::<F>),
        )
    };
    search.found
}

/// Callback for USB-device removal notifications.
pub trait DeviceRemovalHandler: Send + Sync + 'static {
    /// A USB device is being removed from the topology.
    fn device_removed(&self, device: &Device);
}

/// Registration on the USB device-removal notifier chain.
#[pin_data(PinnedDrop)]
pub struct DeviceRemovalNotifier<T: DeviceRemovalHandler> {
    handler: Arc<T>,
    #[pin]
    notifier: Opaque<bindings::notifier_block>,
}

// SAFETY: The notifier chain serializes access to `notifier`; the handler is required to be
// thread-safe.
unsafe impl<T: DeviceRemovalHandler> Send for DeviceRemovalNotifier<T> {}
// SAFETY: See `Send`.
unsafe impl<T: DeviceRemovalHandler> Sync for DeviceRemovalNotifier<T> {}

impl<T: DeviceRemovalHandler> DeviceRemovalNotifier<T> {
    /// Register a removal notifier backed by `handler`.
    pub fn new(handler: Arc<T>) -> impl PinInit<Self> {
        pin_init!(Self {
            handler,
            notifier <- Opaque::ffi_init(|slot: *mut bindings::notifier_block| {
                // SAFETY: `slot` is the pinned notifier field and remains live until `PinnedDrop`
                // unregisters it.
                unsafe {
                    (*slot).notifier_call = Some(Self::notify);
                    (*slot).next = core::ptr::null_mut();
                    (*slot).priority = 0;
                    bindings::usb_register_notify(slot);
                }
            }),
        })
    }

    unsafe extern "C" fn notify(
        notifier: *mut bindings::notifier_block,
        action: usize,
        data: *mut core::ffi::c_void,
    ) -> core::ffi::c_int {
        if action as u32 == bindings::USB_DEVICE_REMOVE {
            // SAFETY: The notifier is registered from this pinned object's embedded field and is
            // unregistered before the object is dropped.
            let this = unsafe {
                &*crate::container_of!(
                    notifier.cast::<Opaque<bindings::notifier_block>>(),
                    Self,
                    notifier
                )
            };
            // SAFETY: The USB notifier chain supplies the live `usb_device` being removed.
            let device = unsafe { &*data.cast::<Device>() };
            this.handler.device_removed(device);
        }
        bindings::NOTIFY_DONE as core::ffi::c_int
    }
}

#[pinned_drop]
impl<T: DeviceRemovalHandler> PinnedDrop for DeviceRemovalNotifier<T> {
    fn drop(self: Pin<&mut Self>) {
        // SAFETY: The notifier was registered exactly once during pinned initialization.
        unsafe { bindings::usb_unregister_notify(self.notifier.get()) };
    }
}

// SAFETY: `Device` is a transparent wrapper of a type that doesn't depend on `Device`'s generic
// argument.
kernel::impl_device_context_deref!(unsafe { Device });
kernel::impl_device_context_into_aref!(Device);

// SAFETY: Instances of `Device` are always reference-counted.
unsafe impl AlwaysRefCounted for Device {
    fn inc_ref(&self) {
        // SAFETY: The invariants of `Device` guarantee that `self.as_raw()`
        // returns a valid `struct usb_device` pointer, for which we will
        // acquire a new refcount.
        unsafe { bindings::usb_get_dev(self.as_raw()) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The safety requirements guarantee that the refcount is non-zero.
        unsafe { bindings::usb_put_dev(obj.cast().as_ptr()) }
    }
}

impl<Ctx: device::DeviceContext> AsRef<device::Device<Ctx>> for Device<Ctx> {
    fn as_ref(&self) -> &device::Device<Ctx> {
        // SAFETY: By the type invariant of `Self`, `self.as_raw()` is a pointer to a valid
        // `struct usb_device`.
        let dev = unsafe { &raw mut ((*self.as_raw()).dev) };

        // SAFETY: `dev` points to a valid `struct device`.
        unsafe { device::Device::from_raw(dev) }
    }
}

// SAFETY: A `Device` is always reference-counted and can be released from any thread.
unsafe impl Send for Device {}

// SAFETY: It is safe to send a &Device to another thread because we do not
// allow any mutation through a shared reference.
unsafe impl Sync for Device {}

// SAFETY: Same as `Device<Normal>` -- the underlying `struct usb_device` is the same;
// `Bound` is a zero-sized type-state marker that does not affect thread safety.
unsafe impl Sync for Device<device::Bound> {}

/// Declares a kernel module that exposes a single USB driver.
///
/// # Examples
///
/// ```ignore
/// module_usb_driver! {
///     type: MyDriver,
///     name: "Module name",
///     author: ["Author name"],
///     description: "Description",
///     license: "GPL v2",
/// }
/// ```
#[macro_export]
macro_rules! module_usb_driver {
    ($($f:tt)*) => {
        $crate::module_driver!(<T>, $crate::usb::Adapter<T>, { $($f)* });
    }
}
