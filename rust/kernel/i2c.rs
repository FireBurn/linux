// SPDX-License-Identifier: GPL-2.0

//! I2C Driver subsystem

// I2C Driver abstractions.
use crate::{
    acpi,
    container_of,
    device,
    device_id::{
        RawDeviceId,
        RawDeviceIdIndex, //
    },
    devres::Devres,
    driver,
    error::*,
    of,
    prelude::*,
    sync::aref::{
        ARef,
        AlwaysRefCounted, //
    },
    types::Opaque, //
};

use core::{
    marker::PhantomData,
    mem::offset_of,
    ptr::{
        from_ref,
        NonNull, //
    }, //
};

/// An I2C device id table.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DeviceId(bindings::i2c_device_id);

impl DeviceId {
    const I2C_NAME_SIZE: usize = 20;

    /// Create a new device id from an I2C 'id' string.
    #[inline(always)]
    pub const fn new(id: &'static CStr) -> Self {
        let src = id.to_bytes_with_nul();
        build_assert!(src.len() <= Self::I2C_NAME_SIZE, "ID exceeds 20 bytes");
        let mut i2c: bindings::i2c_device_id = pin_init::zeroed();
        let mut i = 0;
        while i < src.len() {
            i2c.name[i] = src[i];
            i += 1;
        }

        Self(i2c)
    }
}

// SAFETY: `DeviceId` is a `#[repr(transparent)]` wrapper of `i2c_device_id` and does not add
// additional invariants, so it's safe to transmute to `RawType`.
unsafe impl RawDeviceId for DeviceId {
    type RawType = bindings::i2c_device_id;
}

// SAFETY: `DRIVER_DATA_OFFSET` is the offset to the `driver_data` field.
unsafe impl RawDeviceIdIndex for DeviceId {
    const DRIVER_DATA_OFFSET: usize = core::mem::offset_of!(bindings::i2c_device_id, driver_data);

    fn index(&self) -> usize {
        self.0.driver_data
    }
}

/// IdTable type for I2C
pub type IdTable<T> = &'static dyn kernel::device_id::IdTable<DeviceId, T>;

/// Create a I2C `IdTable` with its alias for modpost.
#[macro_export]
macro_rules! i2c_device_table {
    ($table_name:ident, $module_table_name:ident, $id_info_type: ty, $table_data: expr) => {
        const $table_name: $crate::device_id::IdArray<
            $crate::i2c::DeviceId,
            $id_info_type,
            { $table_data.len() },
        > = $crate::device_id::IdArray::new($table_data);

        $crate::module_device_table!("i2c", $module_table_name, $table_name);
    };
}

/// An adapter for the registration of I2C drivers.
pub struct Adapter<T: Driver>(T);

// SAFETY:
// - `bindings::i2c_driver` is a C type declared as `repr(C)`.
// - `T::Data` is the type of the driver's device private data.
// - `struct i2c_driver` embeds a `struct device_driver`.
// - `DEVICE_DRIVER_OFFSET` is the correct byte offset to the embedded `struct device_driver`.
unsafe impl<T: Driver> driver::DriverLayout for Adapter<T> {
    type DriverType = bindings::i2c_driver;
    type DriverData<'bound> = T::Data<'bound>;
    const DEVICE_DRIVER_OFFSET: usize = core::mem::offset_of!(Self::DriverType, driver);
}

// SAFETY: A call to `unregister` for a given instance of `DriverType` is guaranteed to be valid if
// a preceding call to `register` has been successful.
unsafe impl<T: Driver> driver::RegistrationOps for Adapter<T> {
    unsafe fn register(
        idrv: &Opaque<Self::DriverType>,
        name: &'static CStr,
        module: &'static ThisModule,
    ) -> Result {
        build_assert!(
            T::ACPI_ID_TABLE.is_some() || T::OF_ID_TABLE.is_some() || T::I2C_ID_TABLE.is_some(),
            "At least one of ACPI/OF/Legacy tables must be present when registering an i2c driver"
        );

        let i2c_table = match T::I2C_ID_TABLE {
            Some(table) => table.as_ptr(),
            None => core::ptr::null(),
        };

        let of_table = match T::OF_ID_TABLE {
            Some(table) => table.as_ptr(),
            None => core::ptr::null(),
        };

        let acpi_table = match T::ACPI_ID_TABLE {
            Some(table) => table.as_ptr(),
            None => core::ptr::null(),
        };

        // SAFETY: It's safe to set the fields of `struct i2c_client` on initialization.
        unsafe {
            (*idrv.get()).driver.name = name.as_char_ptr();
            (*idrv.get()).probe = Some(Self::probe_callback);
            (*idrv.get()).remove = Some(Self::remove_callback);
            (*idrv.get()).shutdown = Some(Self::shutdown_callback);
            (*idrv.get()).id_table = i2c_table;
            (*idrv.get()).driver.of_match_table = of_table;
            (*idrv.get()).driver.acpi_match_table = acpi_table;
        }

        // SAFETY: `idrv` is guaranteed to be a valid `DriverType`.
        to_result(unsafe { bindings::i2c_register_driver(module.0, idrv.get()) })
    }

    unsafe fn unregister(idrv: &Opaque<Self::DriverType>) {
        // SAFETY: `idrv` is guaranteed to be a valid `DriverType`.
        unsafe { bindings::i2c_del_driver(idrv.get()) }
    }
}

impl<T: Driver> Adapter<T> {
    extern "C" fn probe_callback(idev: *mut bindings::i2c_client) -> kernel::ffi::c_int {
        // SAFETY: The I2C bus only ever calls the probe callback with a valid pointer to a
        // `struct i2c_client`.
        //
        // INVARIANT: `idev` is valid for the duration of `probe_callback()`.
        let idev = unsafe { &*idev.cast::<I2cClient<device::CoreInternal<'_>>>() };

        let info =
            Self::i2c_id_info(idev).or_else(|| <Self as driver::Adapter>::id_info(idev.as_ref()));

        from_result(|| {
            let data = T::probe(idev, info);

            idev.as_ref().set_drvdata(data)?;
            Ok(0)
        })
    }

    extern "C" fn remove_callback(idev: *mut bindings::i2c_client) {
        // SAFETY: `idev` is a valid pointer to a `struct i2c_client`.
        let idev = unsafe { &*idev.cast::<I2cClient<device::CoreInternal<'_>>>() };

        // SAFETY: `remove_callback` is only ever called after a successful call to
        // `probe_callback`, hence it's guaranteed that `I2cClient::set_drvdata()` has been called
        // and stored a `Pin<KBox<T::Data<'_>>>`.
        let data = unsafe { idev.as_ref().drvdata_borrow::<T::Data<'_>>() };

        T::unbind(idev, data);
    }

    extern "C" fn shutdown_callback(idev: *mut bindings::i2c_client) {
        // SAFETY: `shutdown_callback` is only ever called for a valid `idev`
        let idev = unsafe { &*idev.cast::<I2cClient<device::CoreInternal<'_>>>() };

        // SAFETY: `shutdown_callback` is only ever called after a successful call to
        // `probe_callback`, hence it's guaranteed that `Device::set_drvdata()` has been called
        // and stored a `Pin<KBox<T::Data<'_>>>`.
        let data = unsafe { idev.as_ref().drvdata_borrow::<T::Data<'_>>() };

        T::shutdown(idev, data);
    }

    /// The [`i2c::IdTable`] of the corresponding driver.
    fn i2c_id_table() -> Option<IdTable<<Self as driver::Adapter>::IdInfo>> {
        T::I2C_ID_TABLE
    }

    /// Returns the driver's private data from the matching entry in the [`i2c::IdTable`], if any.
    ///
    /// If this returns `None`, it means there is no match with an entry in the [`i2c::IdTable`].
    fn i2c_id_info(dev: &I2cClient) -> Option<&'static <Self as driver::Adapter>::IdInfo> {
        let table = Self::i2c_id_table()?;

        // SAFETY:
        // - `table` has static lifetime, hence it's valid for reads
        // - `dev` is guaranteed to be valid while it's alive, and so is `dev.as_raw()`.
        let raw_id = unsafe { bindings::i2c_match_id(table.as_ptr(), dev.as_raw()) };

        if raw_id.is_null() {
            return None;
        }

        // SAFETY: `DeviceId` is a `#[repr(transparent)` wrapper of `struct i2c_device_id` and
        // does not add additional invariants, so it's safe to transmute.
        let id = unsafe { &*raw_id.cast::<DeviceId>() };

        Some(table.info(<DeviceId as RawDeviceIdIndex>::index(id)))
    }
}

impl<T: Driver> driver::Adapter for Adapter<T> {
    type IdInfo = T::IdInfo;

    fn of_id_table() -> Option<of::IdTable<Self::IdInfo>> {
        T::OF_ID_TABLE
    }

    fn acpi_id_table() -> Option<acpi::IdTable<Self::IdInfo>> {
        T::ACPI_ID_TABLE
    }
}

/// Declares a kernel module that exposes a single i2c driver.
///
/// # Examples
///
/// ```ignore
/// kernel::module_i2c_driver! {
///     type: MyDriver,
///     name: "Module name",
///     authors: ["Author name"],
///     description: "Description",
///     license: "GPL v2",
/// }
/// ```
#[macro_export]
macro_rules! module_i2c_driver {
    ($($f:tt)*) => {
        $crate::module_driver!(<T>, $crate::i2c::Adapter<T>, { $($f)* });
    };
}

/// The i2c driver trait.
///
/// Drivers must implement this trait in order to get a i2c driver registered.
///
/// # Example
///
///```
/// # use kernel::{acpi, bindings, device::Core, i2c, of};
///
/// struct MyDriver;
///
/// kernel::acpi_device_table!(
///     ACPI_TABLE,
///     MODULE_ACPI_TABLE,
///     <MyDriver as i2c::Driver>::IdInfo,
///     [
///         (acpi::DeviceId::new(c"LNUXBEEF"), ())
///     ]
/// );
///
/// kernel::i2c_device_table!(
///     I2C_TABLE,
///     MODULE_I2C_TABLE,
///     <MyDriver as i2c::Driver>::IdInfo,
///     [
///          (i2c::DeviceId::new(c"rust_driver_i2c"), ())
///     ]
/// );
///
/// kernel::of_device_table!(
///     OF_TABLE,
///     MODULE_OF_TABLE,
///     <MyDriver as i2c::Driver>::IdInfo,
///     [
///         (of::DeviceId::new(c"test,device"), ())
///     ]
/// );
///
/// impl i2c::Driver for MyDriver {
///     type IdInfo = ();
///     type Data<'bound> = Self;
///     const I2C_ID_TABLE: Option<i2c::IdTable<Self::IdInfo>> = Some(&I2C_TABLE);
///     const OF_ID_TABLE: Option<of::IdTable<Self::IdInfo>> = Some(&OF_TABLE);
///     const ACPI_ID_TABLE: Option<acpi::IdTable<Self::IdInfo>> = Some(&ACPI_TABLE);
///
///     fn probe<'bound>(
///         _idev: &'bound i2c::I2cClient<Core<'_>>,
///         _id_info: Option<&'bound Self::IdInfo>,
///     ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
///         Err(ENODEV)
///     }
///
///     fn shutdown<'bound>(
///         _idev: &'bound i2c::I2cClient<Core<'_>>,
///         this: Pin<&Self::Data<'bound>>,
///     ) {
///     }
/// }
///```
pub trait Driver {
    /// The type holding information about each device id supported by the driver.
    // TODO: Use `associated_type_defaults` once stabilized:
    //
    // ```
    // type IdInfo: 'static = ();
    // ```
    type IdInfo: 'static;

    /// The type of the driver's bus device private data.
    type Data<'bound>: Send + 'bound;

    /// The table of device ids supported by the driver.
    const I2C_ID_TABLE: Option<IdTable<Self::IdInfo>> = None;

    /// The table of OF device ids supported by the driver.
    const OF_ID_TABLE: Option<of::IdTable<Self::IdInfo>> = None;

    /// The table of ACPI device ids supported by the driver.
    const ACPI_ID_TABLE: Option<acpi::IdTable<Self::IdInfo>> = None;

    /// I2C driver probe.
    ///
    /// Called when a new i2c client is added or discovered.
    /// Implementers should attempt to initialize the client here.
    fn probe<'bound>(
        dev: &'bound I2cClient<device::Core<'_>>,
        id_info: Option<&'bound Self::IdInfo>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound;

    /// I2C driver shutdown.
    ///
    /// Called by the kernel during system reboot or power-off to allow the [`Driver`] to bring the
    /// [`I2cClient`] into a safe state. Implementing this callback is optional.
    ///
    /// Typical actions include stopping transfers, disabling interrupts, or resetting the hardware
    /// to prevent undesired behavior during shutdown.
    ///
    /// This callback is distinct from final resource cleanup, as the driver instance remains valid
    /// after it returns. Any deallocation or teardown of driver-owned resources should instead be
    /// handled in `Drop`.
    fn shutdown<'bound>(dev: &'bound I2cClient<device::Core<'_>>, this: Pin<&Self::Data<'bound>>) {
        let _ = (dev, this);
    }

    /// I2C driver unbind.
    ///
    /// Called when the [`I2cClient`] is unbound from its bound [`Driver`]. Implementing this
    /// callback is optional.
    ///
    /// This callback serves as a place for drivers to perform teardown operations that require a
    /// `&Device<Core>` or `&Device<Bound>` reference. For instance, drivers may try to perform I/O
    /// operations to gracefully tear down the device.
    ///
    /// Otherwise, release operations for driver resources should be performed in `Drop`.
    fn unbind<'bound>(dev: &'bound I2cClient<device::Core<'_>>, this: Pin<&Self::Data<'bound>>) {
        let _ = (dev, this);
    }
}

/// The i2c adapter representation.
///
/// This structure represents the Rust abstraction for a C `struct i2c_adapter`. The
/// implementation abstracts the usage of an existing C `struct i2c_adapter` that
/// gets passed from the C side
///
/// # Invariants
///
/// A [`I2cAdapter`] instance represents a valid `struct i2c_adapter` created by the C portion of
/// the kernel.
#[repr(transparent)]
pub struct I2cAdapter<Ctx: device::DeviceContext = device::Normal>(
    Opaque<bindings::i2c_adapter>,
    PhantomData<Ctx>,
);

impl<Ctx: device::DeviceContext> I2cAdapter<Ctx> {
    fn as_raw(&self) -> *mut bindings::i2c_adapter {
        self.0.get()
    }
}

impl I2cAdapter {
    /// Returns the I2C Adapter index.
    #[inline]
    pub fn index(&self) -> i32 {
        // SAFETY: `self.as_raw` is a valid pointer to a `struct i2c_adapter`.
        unsafe { (*self.as_raw()).nr }
    }

    /// Gets pointer to an `i2c_adapter` by index.
    pub fn get(index: i32) -> Result<ARef<Self>> {
        // SAFETY: `index` must refer to a valid I2C adapter; the kernel
        // guarantees that `i2c_get_adapter(index)` returns either a valid
        // pointer or NULL. `NonNull::new` guarantees the correct check.
        let adapter = NonNull::new(unsafe { bindings::i2c_get_adapter(index) }).ok_or(ENODEV)?;

        // SAFETY: `adapter` is non-null and points to a live `i2c_adapter`.
        // `I2cAdapter` is #[repr(transparent)], so this cast is valid.
        // `i2c_get_adapter` returned the adapter with an incremented refcount, which we pass to
        // the `ARef`.
        Ok(unsafe { ARef::from_raw(adapter.cast::<I2cAdapter<device::Normal>>()) })
    }
}

// SAFETY: `I2cAdapter` is a transparent wrapper of a type that doesn't depend on
// `I2cAdapter`'s generic argument.
kernel::impl_device_context_deref!(unsafe { I2cAdapter });
kernel::impl_device_context_into_aref!(I2cAdapter);

// SAFETY: Instances of `I2cAdapter` are always reference-counted.
unsafe impl AlwaysRefCounted for I2cAdapter {
    fn inc_ref(&self) {
        // SAFETY: The existence of a shared reference guarantees that the refcount is non-zero.
        unsafe { bindings::i2c_get_adapter(self.index()) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The safety requirements guarantee that the refcount is non-zero.
        unsafe { bindings::i2c_put_adapter(obj.as_ref().as_raw()) }
    }
}

/// The i2c board info representation
///
/// This structure represents the Rust abstraction for a C `struct i2c_board_info` structure,
/// which is used for manual I2C client creation.
#[repr(transparent)]
pub struct I2cBoardInfo(bindings::i2c_board_info);

impl I2cBoardInfo {
    const I2C_TYPE_SIZE: usize = 20;
    /// Create a new [`I2cBoardInfo`] for a kernel driver.
    #[inline(always)]
    pub const fn new(type_: &'static CStr, addr: u16) -> Self {
        let src = type_.to_bytes_with_nul();
        build_assert!(src.len() <= Self::I2C_TYPE_SIZE, "Type exceeds 20 bytes");
        let mut i2c_board_info: bindings::i2c_board_info = pin_init::zeroed();
        let mut i: usize = 0;
        while i < src.len() {
            i2c_board_info.type_[i] = src[i];
            i += 1;
        }

        i2c_board_info.addr = addr;
        Self(i2c_board_info)
    }

    fn as_raw(&self) -> *const bindings::i2c_board_info {
        from_ref(&self.0)
    }
}

/// The i2c client representation.
///
/// This structure represents the Rust abstraction for a C `struct i2c_client`. The
/// implementation abstracts the usage of an existing C `struct i2c_client` that
/// gets passed from the C side
///
/// # Invariants
///
/// A [`I2cClient`] instance represents a valid `struct i2c_client` created by the C portion of
/// the kernel.
#[repr(transparent)]
pub struct I2cClient<Ctx: device::DeviceContext = device::Normal>(
    Opaque<bindings::i2c_client>,
    PhantomData<Ctx>,
);

impl<Ctx: device::DeviceContext> I2cClient<Ctx> {
    fn as_raw(&self) -> *mut bindings::i2c_client {
        self.0.get()
    }
}

// SAFETY: `I2cClient` is a transparent wrapper of `struct i2c_client`.
// The offset is guaranteed to point to a valid device field inside `I2cClient`.
unsafe impl<Ctx: device::DeviceContext> device::AsBusDevice<Ctx> for I2cClient<Ctx> {
    const OFFSET: usize = offset_of!(bindings::i2c_client, dev);
}

// SAFETY: `I2cClient` is a transparent wrapper of a type that doesn't depend on
// `I2cClient`'s generic argument.
kernel::impl_device_context_deref!(unsafe { I2cClient });
kernel::impl_device_context_into_aref!(I2cClient);

// SAFETY: Instances of `I2cClient` are always reference-counted.
unsafe impl AlwaysRefCounted for I2cClient {
    fn inc_ref(&self) {
        // SAFETY: The existence of a shared reference guarantees that the refcount is non-zero.
        unsafe { bindings::get_device(self.as_ref().as_raw()) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The safety requirements guarantee that the refcount is non-zero.
        unsafe { bindings::put_device(&raw mut (*obj.as_ref().as_raw()).dev) }
    }
}

impl<Ctx: device::DeviceContext> AsRef<device::Device<Ctx>> for I2cClient<Ctx> {
    fn as_ref(&self) -> &device::Device<Ctx> {
        let raw = self.as_raw();
        // SAFETY: By the type invariant of `Self`, `self.as_raw()` is a pointer to a valid
        // `struct i2c_client`.
        let dev = unsafe { &raw mut (*raw).dev };

        // SAFETY: `dev` points to a valid `struct device`.
        unsafe { device::Device::from_raw(dev) }
    }
}

impl<Ctx: device::DeviceContext> TryFrom<&device::Device<Ctx>> for &I2cClient<Ctx> {
    type Error = kernel::error::Error;

    fn try_from(dev: &device::Device<Ctx>) -> Result<Self, Self::Error> {
        // SAFETY: By the type invariant of `Device`, `dev.as_raw()` is a valid pointer to a
        // `struct device`.
        if unsafe { bindings::i2c_verify_client(dev.as_raw()).is_null() } {
            return Err(EINVAL);
        }

        // SAFETY: We've just verified that the type of `dev` equals to
        // `bindings::i2c_client_type`, hence `dev` must be embedded in a valid
        // `struct i2c_client` as guaranteed by the corresponding C code.
        let idev = unsafe { container_of!(dev.as_raw(), bindings::i2c_client, dev) };

        // SAFETY: `idev` is a valid pointer to a `struct i2c_client`.
        Ok(unsafe { &*idev.cast() })
    }
}

// SAFETY: A `I2cClient` is always reference-counted and can be released from any thread.
unsafe impl Send for I2cClient {}

// SAFETY: `I2cClient` can be shared among threads because all methods of `I2cClient`
// (i.e. `I2cClient<Normal>) are thread safe.
unsafe impl Sync for I2cClient {}

/// The registration of an i2c client device.
///
/// This type represents the registration of a [`struct i2c_client`]. When an instance of this
/// type is dropped, its respective i2c client device will be unregistered from the system.
///
/// # Invariants
///
/// `self.0` always holds a valid pointer to an initialized and registered
/// [`struct i2c_client`].
#[repr(transparent)]
pub struct Registration(NonNull<bindings::i2c_client>);

impl Registration {
    /// The C `i2c_new_client_device` function wrapper for manual I2C client creation.
    pub fn new<'a>(
        i2c_adapter: &I2cAdapter,
        i2c_board_info: &I2cBoardInfo,
        parent_dev: &'a device::Device<device::Bound>,
    ) -> impl PinInit<Devres<Self>, Error> + 'a {
        Devres::new(parent_dev, Self::try_new(i2c_adapter, i2c_board_info))
    }

    fn try_new(i2c_adapter: &I2cAdapter, i2c_board_info: &I2cBoardInfo) -> Result<Self> {
        // SAFETY: the kernel guarantees that `i2c_new_client_device()` returns either a valid
        // pointer or NULL. `from_err_ptr` separates errors. Following `NonNull::new`
        // checks for NULL.
        let raw_dev = from_err_ptr(unsafe {
            bindings::i2c_new_client_device(i2c_adapter.as_raw(), i2c_board_info.as_raw())
        })?;

        let dev_ptr = NonNull::new(raw_dev).ok_or(ENODEV)?;

        Ok(Self(dev_ptr))
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        // SAFETY: `Drop` is only called for a valid `Registration`, which by invariant
        // always contains a non-null pointer to an `i2c_client`.
        unsafe { bindings::i2c_unregister_device(self.0.as_ptr()) }
    }
}

// SAFETY: A `Registration` of a `struct i2c_client` can be released from any thread.
unsafe impl Send for Registration {}

// SAFETY: `Registration` offers no interior mutability (no mutation through &self
// and no mutable access is exposed)
unsafe impl Sync for Registration {}

// ===========================================================================
// I2C adapter (bus controller / provider) side
// ===========================================================================

/// A single I2C message presented to a [`BusController`]'s transfer routine.
///
/// Wraps a `struct i2c_msg`; `#[repr(transparent)]` so a C `i2c_msg` array can be viewed directly
/// as a `[Msg]`.
#[repr(transparent)]
pub struct Msg(Opaque<bindings::i2c_msg>);

/// Direction-typed access to an I2C message buffer.
pub enum MsgBuffer<'a> {
    /// Bytes which the controller must write to the addressed device.
    Write(&'a [u8]),
    /// Storage which the controller must fill with bytes read from the addressed device.
    Read(&'a mut [u8]),
}

impl Msg {
    /// The 7-bit (or 10-bit) target address.
    #[inline]
    pub fn addr(&self) -> u16 {
        // SAFETY: `self.0` is a valid `i2c_msg` for the callback's duration.
        unsafe { (*self.0.get()).addr }
    }

    /// The message flags (`I2C_M_*`).
    #[inline]
    pub fn flags(&self) -> u16 {
        // SAFETY: as above.
        unsafe { (*self.0.get()).flags }
    }

    /// Whether this is a read (`I2C_M_RD`) rather than a write.
    #[inline]
    pub fn is_read(&self) -> bool {
        self.flags() & (bindings::I2C_M_RD as u16) != 0
    }

    /// The message length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        // SAFETY: as above.
        unsafe { (*self.0.get()).len as usize }
    }

    /// Whether the message is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return direction-appropriate access to the message buffer.
    ///
    /// A zero-length C message is represented by an empty Rust slice without inspecting its
    /// potentially null `buf` pointer.
    #[inline]
    pub fn buffer(&mut self) -> MsgBuffer<'_> {
        let len = self.len();
        let is_read = self.is_read();
        if len == 0 {
            return if is_read {
                MsgBuffer::Read(&mut [])
            } else {
                MsgBuffer::Write(&[])
            };
        }

        // SAFETY: For a non-empty message, I2C core guarantees `buf` points to `len` bytes valid
        // for the callback. It is writable exactly when `I2C_M_RD` asks the controller to fill it.
        let buf = unsafe { (*self.0.get()).buf };
        if is_read {
            // SAFETY: The read-message contract above grants mutable access to all `len` bytes.
            MsgBuffer::Read(unsafe { core::slice::from_raw_parts_mut(buf, len) })
        } else {
            // SAFETY: The write-message contract above grants shared access to all `len` bytes.
            MsgBuffer::Write(unsafe { core::slice::from_raw_parts(buf, len) })
        }
    }
}

/// `I2C_FUNC_I2C`: the adapter supports plain I2C (not just SMBus) transfers.
pub const FUNC_I2C: u32 = bindings::I2C_FUNC_I2C;

/// A Rust-implemented I2C bus adapter (the controller/provider side).
///
/// This is the counterpart to the [`Driver`] trait above: rather than binding to an I2C client on
/// an existing bus, an implementer *is* a bus and services transfers. Virtual buses -- e.g. the
/// DDC/CI channel DisplayLink's evdi exposes so userspace monitor-control tools can reach the
/// display -- use this.
pub trait BusController: 'static {
    /// Per-adapter driver context, made available to the transfer callbacks.
    type Context: Send + Sync;

    /// Transfer `msgs` on the bus, returning the number of messages successfully transferred
    /// (as the C `master_xfer` does), or an error.
    fn master_xfer(ctx: &Self::Context, msgs: &mut [Msg]) -> Result<usize>;

    /// Report the bus functionality bitmask (`I2C_FUNC_*`).
    fn functionality(ctx: &Self::Context) -> u32;
}

/// An owned, registered I2C adapter driven by a [`BusController`].
///
/// [`BusAdapter::new`] fills a `struct i2c_adapter` (pointing its algorithm at trampolines that
/// dispatch to `T`) and calls `i2c_add_adapter()`; dropping it calls `i2c_del_adapter()`.
#[pin_data(PinnedDrop)]
pub struct BusAdapter<T: BusController> {
    #[pin]
    adapter: Opaque<bindings::i2c_adapter>,
    #[pin]
    algo: Opaque<bindings::i2c_algorithm>,
    ctx: T::Context,
    registered: core::sync::atomic::AtomicBool,
}

// SAFETY: the adapter is internally synchronized by the I2C core; `ctx` is `Send + Sync`.
unsafe impl<T: BusController> Send for BusAdapter<T> {}
// SAFETY: see `Send`.
unsafe impl<T: BusController> Sync for BusAdapter<T> {}

impl<T: BusController> BusAdapter<T> {
    /// # Safety
    /// `adap` is a valid adapter whose `algo_data` points to a live `T::Context`.
    unsafe extern "C" fn xfer_trampoline(
        adap: *mut bindings::i2c_adapter,
        msgs: *mut bindings::i2c_msg,
        num: crate::ffi::c_int,
    ) -> crate::ffi::c_int {
        if num < 0 {
            return EINVAL.to_errno();
        }

        // SAFETY: by the invariant established in `new`, `algo_data` points to the `T::Context`.
        let ctx = unsafe { &*((*adap).algo_data as *const T::Context) };
        let count = num as usize;
        let slice: &mut [Msg] = if count == 0 {
            &mut []
        } else {
            // SAFETY: the I2C core passes `num` valid `i2c_msg`s; `Msg` is
            // `#[repr(transparent)]`. The zero-count case above avoids constructing a slice from a
            // potentially null C pointer.
            unsafe { core::slice::from_raw_parts_mut(msgs.cast::<Msg>(), count) }
        };
        match T::master_xfer(ctx, slice) {
            Ok(n) if n <= count => match crate::ffi::c_int::try_from(n) {
                Ok(n) => n,
                Err(_) => EINVAL.to_errno(),
            },
            Ok(_) => EINVAL.to_errno(),
            Err(e) => e.to_errno(),
        }
    }

    /// # Safety
    /// `adap` is a valid adapter whose `algo_data` points to a live `T::Context`.
    unsafe extern "C" fn func_trampoline(adap: *mut bindings::i2c_adapter) -> u32 {
        // SAFETY: by the invariant established in `new`, `algo_data` points to the `T::Context`.
        let ctx = unsafe { &*((*adap).algo_data as *const T::Context) };
        T::functionality(ctx)
    }

    /// Create and register a new I2C adapter named `name`, parented to `parent`, with driver
    /// context `ctx`.
    pub fn new(name: &CStr, parent: &device::Device, ctx: T::Context) -> Result<Pin<KBox<Self>>> {
        let this = KBox::try_pin_init(
            try_pin_init!(Self {
                adapter <- Opaque::ffi_init(|slot: *mut bindings::i2c_adapter| {
                    // SAFETY: `slot` is valid, uninitialized storage; a zeroed adapter is a valid
                    // starting point (self-referential fields are set below, after pinning).
                    unsafe { *slot = bindings::i2c_adapter::default() };
                }),
                algo <- Opaque::ffi_init(|slot: *mut bindings::i2c_algorithm| {
                    // SAFETY: `slot` is valid, uninitialized storage.
                    unsafe {
                        *slot = bindings::i2c_algorithm::default();
                        (*slot).__bindgen_anon_1.master_xfer = Some(Self::xfer_trampoline);
                        (*slot).functionality = Some(Self::func_trampoline);
                    }
                }),
                ctx: ctx,
                registered: core::sync::atomic::AtomicBool::new(false),
            }),
            GFP_KERNEL,
        )?;

        // `this` now has a stable address; wire the self-referential adapter fields and register.
        let adap = this.adapter.get();
        // Copy the name without its NUL into the fixed 48-byte array (capped at 47 so the
        // already-zeroed array stays NUL-terminated even if the name is over-long).
        let name_bytes = name.to_bytes();
        let n = core::cmp::min(name_bytes.len(), 47);
        // SAFETY: `adap`/`algo`/`ctx` live for `this`'s (pinned) lifetime; `name_bytes` is valid
        // for `n <= 47` bytes; `parent` is a valid device.
        unsafe {
            (*adap).algo = this.algo.get();
            (*adap).algo_data = (&this.ctx as *const T::Context).cast_mut().cast();
            core::ptr::copy_nonoverlapping(
                name_bytes.as_ptr(),
                (*adap).name.as_mut_ptr().cast::<u8>(),
                n,
            );
            (*adap).dev.parent = parent.as_raw();
        }

        // SAFETY: `adap` is a fully-initialized adapter.
        to_result(unsafe { bindings::i2c_add_adapter(adap) })?;
        this.registered
            .store(true, core::sync::atomic::Ordering::Release);
        Ok(this)
    }
}

#[pinned_drop]
impl<T: BusController> PinnedDrop for BusAdapter<T> {
    fn drop(self: Pin<&mut Self>) {
        if self.registered.load(core::sync::atomic::Ordering::Acquire) {
            // SAFETY: the adapter was successfully registered with `i2c_add_adapter` and has not
            // been deleted yet.
            unsafe { bindings::i2c_del_adapter(self.adapter.get()) };
        }
    }
}

#[kunit_tests(rust_i2c_bus_adapter)]
mod bus_adapter_tests {
    use super::*;

    struct ZeroController;

    impl BusController for ZeroController {
        type Context = ();

        fn master_xfer(_ctx: &Self::Context, msgs: &mut [Msg]) -> Result<usize> {
            Ok(msgs.len())
        }

        fn functionality(_ctx: &Self::Context) -> u32 {
            FUNC_I2C
        }
    }

    struct InvalidCountController;

    impl BusController for InvalidCountController {
        type Context = ();

        fn master_xfer(_ctx: &Self::Context, msgs: &mut [Msg]) -> Result<usize> {
            Ok(msgs.len() + 1)
        }

        fn functionality(_ctx: &Self::Context) -> u32 {
            FUNC_I2C
        }
    }

    #[test]
    fn zero_length_null_buffers_are_empty_slices() {
        let mut raw = bindings::i2c_msg::default();
        raw.buf = core::ptr::null_mut();
        raw.len = 0;

        // SAFETY: `Msg` is transparent over the initialized `i2c_msg` above.
        let msg = unsafe { &mut *(&raw mut raw).cast::<Msg>() };
        match msg.buffer() {
            MsgBuffer::Write(buf) => assert!(buf.is_empty()),
            MsgBuffer::Read(_) => panic!("write message exposed as read"),
        }

        raw.flags = bindings::I2C_M_RD as u16;
        // SAFETY: `Msg` is transparent over the initialized `i2c_msg` above.
        let msg = unsafe { &mut *(&raw mut raw).cast::<Msg>() };
        match msg.buffer() {
            MsgBuffer::Read(buf) => assert!(buf.is_empty()),
            MsgBuffer::Write(_) => panic!("read message exposed as write"),
        }
    }

    #[test]
    fn transfer_result_cannot_exceed_input_count() {
        let ctx = ();
        let mut adapter = bindings::i2c_adapter::default();
        adapter.algo_data = (&raw const ctx).cast_mut().cast();

        // SAFETY: `adapter.algo_data` points to the live context above; a zero-message callback may
        // pass a null message pointer.
        let ret = unsafe {
            BusAdapter::<InvalidCountController>::xfer_trampoline(
                &raw mut adapter,
                core::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ret, EINVAL.to_errno());

        // SAFETY: Same setup; the valid controller returns exactly the empty input count.
        let ret = unsafe {
            BusAdapter::<ZeroController>::xfer_trampoline(
                &raw mut adapter,
                core::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ret, 0);
    }
}
