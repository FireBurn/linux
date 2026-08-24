// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM display connectors.
//!
//! C header: [`include/drm/drm_connector.h`](srctree/include/drm/drm_connector.h)

use super::{
    atomic::*, encoder::*, modes::DisplayMode, KmsDriver, ModeConfigGuard, ModeObject,
    ModeObjectVtable, Sealed,
};
use crate::{
    alloc::KBox,
    bindings,
    drm::{device::Device, kms::UnregisteredKmsDevice},
    error::to_result,
    prelude::*,
    types::{NotThreadSafe, Opaque},
};
use core::{
    cell::Cell,
    marker::*,
    mem::{self, ManuallyDrop},
    ops::*,
    ptr::{null_mut, NonNull},
    stringify,
};
use macros::paste;

/// A macro for generating our type ID enumerator.
macro_rules! declare_conn_types {
    ($( $oldname:ident as $newname:ident ),+) => {
        /// An enumerator for all possible [`Connector`] type IDs.
        #[repr(i32)]
        #[non_exhaustive]
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub enum Type {
            // Note: bindgen defaults the macro values to u32 and not i32, but DRM takes them as an
            // i32 - so just do the conversion here
            $(
                #[doc = concat!("The connector type ID for a ", stringify!($newname), " connector.")]
                $newname = paste!(crate::bindings::[<DRM_MODE_CONNECTOR_ $oldname>]) as i32
            ),+,

            // 9PinDIN is special because of the 9, making it an invalid ident. Just define it here
            // manually since it's the only one

            /// The connector type ID for a 9PinDIN connector.
            _9PinDin = crate::bindings::DRM_MODE_CONNECTOR_9PinDIN as i32
        }
    };
}

declare_conn_types! {
    Unknown     as Unknown,
    Composite   as Composite,
    Component   as Component,
    DisplayPort as DisplayPort,
    VGA         as Vga,
    DVII        as DviI,
    DVID        as DviD,
    DVIA        as DviA,
    SVIDEO      as SVideo,
    LVDS        as Lvds,
    HDMIA       as HdmiA,
    HDMIB       as HdmiB,
    TV          as Tv,
    eDP         as Edp,
    VIRTUAL     as Virtual,
    DSI         as Dsi,
    DPI         as Dpi,
    WRITEBACK   as Writeback,
    SPI         as Spi,
    USB         as Usb
}

/// The connection status of a [`Connector`], as returned by [`DriverConnector::detect`].
///
/// This is identical to [`enum drm_connector_status`].
///
/// [`enum drm_connector_status`]: srctree/include/drm/drm_connector.h
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Status {
    /// The connector is connected to a display and a mode list can be retrieved.
    Connected = bindings::drm_connector_status_connector_status_connected,
    /// The connector has no display attached.
    Disconnected = bindings::drm_connector_status_connector_status_disconnected,
    /// The connection state could not be determined (treated as connected for probing).
    Unknown = bindings::drm_connector_status_connector_status_unknown,
}

/// The result of validating a display mode against a [`Connector`], as returned by
/// [`DriverConnector::mode_valid`].
///
/// This mirrors a small, commonly-used subset of [`enum drm_mode_status`]; use [`ModeStatus::Bad`]
/// for a generic rejection, or the clock-specific variants when a mode is out of the driver's
/// pixel-clock range.
///
/// [`enum drm_mode_status`]: srctree/include/drm/drm_modes.h
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ModeStatus {
    /// The mode is usable.
    Ok = bindings::drm_mode_status_MODE_OK,
    /// The mode is rejected for an unspecified reason.
    Bad = bindings::drm_mode_status_MODE_BAD,
    /// The mode's pixel clock is above what the driver can drive.
    ClockHigh = bindings::drm_mode_status_MODE_CLOCK_HIGH,
    /// The mode's pixel clock is below what the driver can drive.
    ClockLow = bindings::drm_mode_status_MODE_CLOCK_LOW,
}

/// The main trait for implementing the [`struct drm_connector`] API for [`Connector`].
///
/// Any KMS driver should have at least one implementation of this type, which allows them to create
/// [`Connector`] objects. Additionally, a driver may store driver-private data within the type that
/// implements [`DriverConnector`] - and it will be made available when using a fully typed
/// [`Connector`] object.
///
/// # Invariants
///
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_connector`] pointers are contained within a [`Connector<Self>`].
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_connector_state`] pointers are contained within a
///   [`ConnectorState<Self::State>`].
///
/// [`struct drm_connector`]: srctree/include/drm/drm_connector.h
/// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
#[vtable]
pub trait DriverConnector: Send + Sync + Sized {
    /// The generated C vtable for this [`DriverConnector`] implementation
    const OPS: &'static DriverConnectorOps = &DriverConnectorOps {
        funcs: bindings::drm_connector_funcs {
            atomic_create_state: None,
            dpms: None,
            atomic_get_property: None,
            atomic_set_property: None,
            early_unregister: None,
            late_register: None,
            set_property: None,
            reset: Some(connector_reset_callback::<Self::State>),
            atomic_print_state: None,
            atomic_destroy_state: Some(atomic_destroy_state_callback::<Self::State>),
            destroy: Some(connector_destroy_callback::<Self>),
            force: None,
            detect: if Self::HAS_DETECT {
                Some(detect_callback::<Self>)
            } else {
                None
            },
            fill_modes: Some(bindings::drm_helper_probe_single_connector_modes),
            debugfs_init: None,
            oob_hotplug_event: None,
            atomic_duplicate_state: Some(atomic_duplicate_state_callback::<Self::State>),
            color_format: None,
        },
        helper_funcs: bindings::drm_connector_helper_funcs {
            mode_valid: if Self::HAS_MODE_VALID {
                Some(mode_valid_callback::<Self>)
            } else {
                None
            },
            atomic_check: None,
            get_modes: Some(get_modes_callback::<Self>),
            detect_ctx: None,
            enable_hpd: None,
            disable_hpd: None,
            best_encoder: None,
            atomic_commit: None,
            mode_valid_ctx: None,
            atomic_best_encoder: None,
            prepare_writeback_job: None,
            cleanup_writeback_job: None,
        },
    };

    /// The type to pass to the `args` field of [`UnregisteredConnector::new`].
    ///
    /// This type will be made available in in the `args` argument of [`Self::new`]. Drivers which
    /// don't need this can simply pass [`()`] here.
    type Args;

    /// The parent [`KmsDriver`] implementation.
    type Driver: KmsDriver;

    /// The [`DriverConnectorState`] implementation for this [`DriverConnector`].
    ///
    /// See [`DriverConnectorState`] for more info.
    type State: DriverConnectorState;

    /// The constructor for creating a [`Connector`] using this [`DriverConnector`] implementation.
    ///
    /// Drivers may use this to instantiate their [`DriverConnector`] object.
    fn new(device: &Device<Self::Driver>, args: Self::Args) -> impl PinInit<Self, Error>;

    /// Retrieve a list of available display modes for this [`Connector`].
    fn get_modes<'a>(
        connector: ConnectorGuard<'a, Self>,
        guard: &ModeConfigGuard<'a, Self::Driver>,
    ) -> i32;

    /// The optional [`drm_connector_funcs.detect`] hook for this connector.
    ///
    /// Drivers may implement this to report whether a display is currently attached. If not
    /// implemented, the connector is always considered connected (DRM's default with no `detect`
    /// hook). `force` is set when userspace explicitly requested a forced probe.
    ///
    /// [`drm_connector_funcs.detect`]: srctree/include/drm/drm_connector.h
    fn detect(_connector: &Connector<Self>, _force: bool) -> Status {
        build_error::build_error("This should not be reachable")
    }

    /// The optional [`drm_connector_helper_funcs.mode_valid`] hook for this connector.
    ///
    /// Drivers may implement this to reject modes they cannot drive (for example, a mode whose
    /// pixel clock exceeds the hardware's budget). Returning anything other than [`ModeStatus::Ok`]
    /// prunes the mode from the probed list. If not implemented, every mode is accepted.
    ///
    /// [`drm_connector_helper_funcs.mode_valid`]: srctree/include/drm/drm_modeset_helper_vtables.h
    fn mode_valid(
        _connector: ConnectorModeValidation<'_, Self>,
        _mode: &DisplayMode,
    ) -> ModeStatus {
        build_error::build_error("This should not be reachable")
    }
}

/// The generated C vtable for a [`DriverConnector`].
///
/// This type is created internally by DRM.
pub struct DriverConnectorOps {
    funcs: bindings::drm_connector_funcs,
    helper_funcs: bindings::drm_connector_helper_funcs,
}

/// The main interface for a [`struct drm_connector`].
///
/// This type is the main interface for dealing with DRM connectors. In addition, it also allows
/// immutable access to whatever private data is contained within an implementor's
/// [`DriverConnector`] type.
///
/// # Invariants
///
/// - The DRM C API and our interface guarantees that only the user has mutable access to `state`,
///   up until [`drm_atomic_helper_commit_hw_done`] is called. Therefore, `connector` follows rust's
///   data aliasing rules and does not need to be behind an [`Opaque`] type.
/// - `connector` and `inner` are initialized for as long as this object is made available to users.
/// - The data layout of this structure begins with [`struct drm_connector`].
/// - The atomic state for this type can always be assumed to be of type
///   [`ConnectorState<T::State>`].
///
/// [`struct drm_connector`]: srctree/include/drm/drm_connector.h
/// [`drm_atomic_helper_commit_hw_done`]: srctree/include/drm/drm_atomic_helper.h
#[repr(C)]
#[pin_data]
pub struct Connector<T: DriverConnector> {
    connector: Opaque<bindings::drm_connector>,
    #[pin]
    inner: T,
    #[pin]
    _p: PhantomPinned,
}

impl<T: DriverConnector> Sealed for Connector<T> {}

impl<T: DriverConnector> Deref for Connector<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: DriverConnector> Connector<T> {
    super::impl_from_opaque_mode_obj! {
        fn <'a, D>(&'a OpaqueConnector<D>) -> &'a Self;
        use
            T as DriverConnector,
            D as KmsDriver<Connector = ...>
    }

    /// Acquire a [`ConnectorGuard`] for this connector from a [`ModeConfigGuard`].
    ///
    /// This verifies using the provided reference that the given guard is actually for the same
    /// device as this connector's parent.
    ///
    /// # Panics
    ///
    /// Panics if `guard` is not a [`ModeConfigGuard`] for this connector's parent [`Device`].
    pub fn guard<'a>(&'a self, guard: &ModeConfigGuard<'a, T::Driver>) -> ConnectorGuard<'a, T> {
        guard.assert_owner(self.drm_dev());
        ConnectorGuard(self)
    }
}

/// A trait implemented by any type that acts as a [`struct drm_connector`] interface.
///
/// This is implemented internally by DRM.
///
/// # Safety
///
/// [`as_raw()`] must always return a pointer to a valid initialized [`struct drm_connector`].
///
/// [`as_raw()`]: AsRawConnector::as_raw()
/// [`struct drm_connector`]: srctree/include/drm/drm_connector.h
pub unsafe trait AsRawConnector {
    /// Return the raw [`struct drm_connector`] for this DRM connector.
    ///
    /// Drivers should never use this directly
    ///
    /// [`struct drm_Connector`]: srctree/include/drm/drm_connector.h
    fn as_raw(&self) -> *mut bindings::drm_connector;

    /// Convert a raw `bindings::drm_connector` pointer into an object of this type.
    ///
    /// # Safety
    ///
    /// Callers promise that `ptr` points to a valid instance of this type.
    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_connector) -> &'a Self;
}

/// A supertrait of [`AsRawConnector`] for [`struct drm_connector`] interfaces that can perform
/// modesets.
///
/// This is implemented internally by DRM.
///
/// # Safety
///
/// Any object implementing this trait must only be made directly available to the user after
/// [`create_objects`] has completed.
///
/// [`struct drm_connector`]: srctree/include/drm/drm_connector.h
/// [`create_objects`]: KmsDriver::create_objects
pub unsafe trait ModesettableConnector: AsRawConnector {
    /// The type that should be returned for a plane state acquired using this plane interface
    type State: FromRawConnectorState;
}

// SAFETY: Our connector interfaces are guaranteed to be thread-safe
unsafe impl<T: DriverConnector> Send for Connector<T> {}

// SAFETY: Our connector interfaces are guaranteed to be thread-safe
unsafe impl<T: DriverConnector> Sync for Connector<T> {}

// SAFETY: We don't expose Connector<T> to users before `base` is initialized in ::new(), so
// `raw_mode_obj` always returns a valid pointer to a bindings::drm_mode_object.
unsafe impl<T: DriverConnector> ModeObject for Connector<T> {
    type Driver = T::Driver;

    fn drm_dev(&self) -> &Device<Self::Driver> {
        // SAFETY: The parent device for a DRM connector will never outlive the connector, and this
        // pointer is invariant through the lifetime of the connector
        unsafe { Device::from_raw((*self.as_raw()).dev) }
    }

    fn raw_mode_obj(&self) -> *mut bindings::drm_mode_object {
        // SAFETY: We don't expose DRM connectors to users before `base` is initialized
        unsafe { &raw mut (*self.as_raw()).base }
    }
}

// Connectors are refcounted objects.
super::impl_aref_for_mode_object! {
    impl<T: DriverConnector> for Connector<T>
}

// SAFETY: `funcs` is initialized by DRM when the connector is allocated
unsafe impl<T: DriverConnector> ModeObjectVtable for Connector<T> {
    type Vtable = bindings::drm_connector_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        // SAFETY: `funcs` is initialized by DRM when the connector is allocated
        unsafe { *self.as_raw() }.funcs
    }
}

// SAFETY:
// * Via our type variants our data layout starts with `drm_connector`
// * Since we don't expose `Connector` to users before it has been initialized, this and our data
//   layout ensure that `as_raw()` always returns a valid pointer to a `drm_connector`.
unsafe impl<T: DriverConnector> AsRawConnector for Connector<T> {
    fn as_raw(&self) -> *mut bindings::drm_connector {
        self.connector.get()
    }

    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_connector) -> &'a Self {
        // SAFETY: Our data layout starts with `bindings::drm_connector`
        unsafe { &*ptr.cast() }
    }
}

// SAFETY: We only expose this object to users directly after KmsDriver::create_objects has been
// called.
unsafe impl<T: DriverConnector> ModesettableConnector for Connector<T> {
    type State = ConnectorState<T::State>;
}

/// A [`Connector`] that has not yet been registered with userspace.
///
/// KMS registration is single-threaded, so this object is not thread-safe.
///
/// # Invariants
///
/// - This object can only exist before its respective KMS device has been registered.
/// - Otherwise, it inherits all invariants of [`Connector`] and has an identical data layout.
pub struct UnregisteredConnector<T: DriverConnector>(Connector<T>, NotThreadSafe);

// SAFETY: We share the invariants of `Connector`
unsafe impl<T: DriverConnector> AsRawConnector for UnregisteredConnector<T> {
    fn as_raw(&self) -> *mut bindings::drm_connector {
        self.0.as_raw()
    }

    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_connector) -> &'a Self {
        // SAFETY: This is another from_raw() call, so this function shares the same safety contract
        let connector = unsafe { Connector::<T>::from_raw(ptr) };

        // SAFETY: Our data layout is identical via our type invariants.
        unsafe { mem::transmute(connector) }
    }
}

impl<T: DriverConnector> Deref for UnregisteredConnector<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}

impl<T: DriverConnector> UnregisteredConnector<T> {
    /// Construct a new [`UnregisteredConnector`].
    ///
    /// A driver may use this to create new [`UnregisteredConnector`] objects.
    ///
    /// [`KmsDriver::create_objects`]: kernel::drm::kms::KmsDriver::create_objects
    pub fn new<'a>(
        dev: &'a UnregisteredKmsDevice<'a, T::Driver>,
        type_: Type,
        args: T::Args,
    ) -> Result<&'a Self> {
        let new: Pin<KBox<Connector<T>>> = KBox::try_pin_init(
            try_pin_init!(Connector {
                connector: Opaque::new(bindings::drm_connector {
                    helper_private: &T::OPS.helper_funcs,
                    ..Default::default()
                }),
                inner <- T::new(dev, args),
                _p: PhantomPinned,
            }),
            GFP_KERNEL,
        )?;

        // SAFETY:
        // - `dev` will hold a reference to the new connector, and thus outlives us.
        // - We just allocated `new` above
        // - `new` starts with `drm_connector` via its type invariants.
        to_result(unsafe {
            bindings::drm_connector_init(dev.as_raw(), new.as_raw(), &T::OPS.funcs, type_ as i32)
        })?;

        // SAFETY: We don't move anything
        let this = unsafe { Pin::into_inner_unchecked(new) };

        // We'll re-assemble the box in connector_destroy_callback()
        let this = KBox::into_raw(this);

        // UnregisteredConnector has an equivalent data layout
        let this: *mut Self = this.cast();

        // SAFETY: We just allocated the connector above, so this pointer must be valid
        Ok(unsafe { &*this })
    }

    /// Attach an encoder to this [`Connector`].
    #[must_use]
    pub fn attach_encoder<E>(&self, encoder: &UnregisteredEncoder<E>) -> Result
    where
        E: DriverEncoder<Driver = T::Driver>,
    {
        // SAFETY: Both unregistered objects have been initialized, so their parent device
        // pointers are valid and invariant for their lifetimes.
        let same_device = unsafe { (*self.as_raw()).dev == (*encoder.as_raw()).dev };
        if !same_device {
            return Err(EINVAL);
        }

        // SAFETY:
        // - Both `as_raw()` calls return valid pointers.
        // - The generic bound and check above prove that both objects belong to the same driver
        //   and device.
        // - `self` is unregistered, as required by the C API.
        to_result(unsafe {
            bindings::drm_connector_attach_encoder(self.as_raw(), encoder.as_raw())
        })
    }

    /// Attach the HDR output metadata property to this [`Connector`].
    ///
    /// This property carries a blob supplied by userspace. Drivers must still validate and apply
    /// the metadata in their atomic commit path before claiming that HDR output is supported.
    pub fn attach_hdr_output_metadata_property(&self) {
        // SAFETY: `self` is an initialized connector owned by this DRM device. The helper only
        // attaches the mode-config-owned standard property to its mode object.
        unsafe {
            bindings::drm_connector_attach_hdr_output_metadata_property(self.as_raw());
        }
    }

    /// Create and attach the standard DP colorspace property to this [`Connector`].
    ///
    /// A zero mask asks DRM to expose every colorspace defined for DisplayPort. A driver must
    /// still reject values its sink or transport cannot actually carry in its atomic check.
    pub fn attach_colorspace_property(&self) -> Result {
        to_result(unsafe { bindings::drm_mode_create_dp_colorspace_property(self.as_raw(), 0) })?;
        // SAFETY: the successful create call above initialized `colorspace_property` for this
        // connector; the C helper only attaches that property to this connector's mode object.
        to_result(unsafe { bindings::drm_connector_attach_colorspace_property(self.as_raw()) })
    }

    /// Attach the standard `max bpc` range property to this [`Connector`].
    ///
    /// `min_bpc` and `max_bpc` are validated before conversion so callers cannot wrap an invalid
    /// range through the C `int` API. DRM requires the connector to have an atomic state before
    /// this helper is called; newly-created Rust connectors acquire that state here.
    pub fn attach_max_bpc_property(&self, min_bpc: u32, max_bpc: u32) -> Result {
        if min_bpc == 0 || min_bpc > max_bpc || max_bpc > i32::MAX as u32 {
            return Err(EINVAL);
        }

        // `drm_connector_attach_max_bpc_property()` writes the initial bpc values into the
        // connector state. `KmsDriver::create_objects()` runs before the mode-config-wide reset,
        // so initialize our state through the driver's Rust reset callback when necessary.
        let state = unsafe { (*self.as_raw()).state };
        if state.is_null() {
            // SAFETY: `self` is a newly initialized `Connector<T>` and this unregistered typestate
            // prevents concurrent access. The callback creates the matching `ConnectorState<T>`.
            unsafe { connector_reset_callback::<T::State>(self.as_raw()) };
        }

        // SAFETY: `self` is initialized and now owns a connector state. The validated bounds fit
        // the C API's signed integer parameters, and the helper only installs a DRM core property.
        to_result(unsafe {
            bindings::drm_connector_attach_max_bpc_property(
                self.as_raw(),
                min_bpc as i32,
                max_bpc as i32,
            )
        })
    }
}

/// Common methods available on any type which implements [`AsRawConnector`].
///
/// This is implemented internally by DRM, and provides many of the basic methods for working with
/// connectors.
pub trait RawConnector: AsRawConnector {
    /// Return the index of this DRM connector
    #[inline]
    fn index(&self) -> u32 {
        // SAFETY: The index is initialized by the time we expose DRM connector objects to users,
        // and is invariant throughout the lifetime of the connector
        unsafe { (*self.as_raw()).index }
    }

    /// Return the bitmask derived from this DRM connector's index
    #[inline]
    fn mask(&self) -> u32 {
        1 << self.index()
    }
}
impl<T: AsRawConnector> RawConnector for T {}

unsafe extern "C" fn connector_destroy_callback<T: DriverConnector>(
    connector: *mut bindings::drm_connector,
) {
    // SAFETY: DRM guarantees that `connector` points to a valid initialized `drm_connector`.
    unsafe {
        bindings::drm_connector_unregister(connector);
        bindings::drm_connector_cleanup(connector);
    };

    // SAFETY:
    // - We originally created the connector in a `Box`
    // - We are guaranteed to hold the last remaining reference to this connector
    // - This cast is safe via `DriverConnector`s type invariants.
    drop(unsafe { KBox::from_raw(connector as *mut Connector<T>) });
}

unsafe extern "C" fn get_modes_callback<T: DriverConnector>(
    connector: *mut bindings::drm_connector,
) -> core::ffi::c_int {
    // SAFETY: This is safe via `DriverConnector`s type invariants.
    let connector = unsafe { Connector::<T>::from_raw(connector) };

    // SAFETY: This FFI callback is only called while `mode_config.lock` is held
    // We use ManuallyDrop here to prevent the lock from being released after the callback
    // completes, as that should be handled by DRM.
    let guard = ManuallyDrop::new(unsafe { ModeConfigGuard::new(connector.drm_dev()) });

    T::get_modes(connector.guard(&guard), &guard)
}

unsafe extern "C" fn detect_callback<T: DriverConnector>(
    connector: *mut bindings::drm_connector,
    force: bool,
) -> bindings::drm_connector_status {
    // SAFETY: This is safe via `DriverConnector`s type invariants.
    let connector = unsafe { Connector::<T>::from_raw(connector) };

    T::detect(connector, force) as bindings::drm_connector_status
}

unsafe extern "C" fn mode_valid_callback<T: DriverConnector>(
    connector: *mut bindings::drm_connector,
    mode: *const bindings::drm_display_mode,
) -> bindings::drm_mode_status {
    // SAFETY: This is safe via `DriverConnector`s type invariants.
    let connector = unsafe { Connector::<T>::from_raw(connector) };

    // SAFETY: DRM guarantees `mode` points to a valid `drm_display_mode` for the duration of this
    // callback, and only passes us shared access to it.
    let mode = unsafe { DisplayMode::as_ref(mode) };

    // DRM invokes the connector helper while the mode list is stable. Keep that guarantee in a
    // capability type so drivers can safely compare this mode with the other probed modes.
    T::mode_valid(ConnectorModeValidation(connector), mode) as bindings::drm_mode_status
}

/// A [`struct drm_connector`] without a known [`DriverConnector`] implementation.
///
/// This is mainly for situations where our bindings can't infer the [`DriverConnector`]
/// implementation for a [`struct drm_connector`] automatically. It is identical to [`Connector`],
/// except that it does not provide access to the driver's private data.
///
/// # Invariants
///
/// - `connector` is initialized for as long as this object is exposed to users.
/// - The data layout of this type is equivalent to [`struct drm_connector`].
///
/// [`struct drm_connector`]: srctree/include/drm/drm_connector.h
#[repr(transparent)]
pub struct OpaqueConnector<T: KmsDriver> {
    connector: Opaque<bindings::drm_connector>,
    _p: PhantomData<T>,
}

impl<T: KmsDriver> Sealed for OpaqueConnector<T> {}

// SAFETY:
// - Via our type variants our data layout starts is identical to `drm_connector`
// - Since we don't expose `OpaqueConnector` to users before it has been initialized, this and our
//   data layout ensure that `as_raw()` always returns a valid pointer to a `drm_connector`.
unsafe impl<T: KmsDriver> AsRawConnector for OpaqueConnector<T> {
    fn as_raw(&self) -> *mut bindings::drm_connector {
        self.connector.get()
    }

    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_connector) -> &'a Self {
        // SAFETY: Our data layout is identical to `bindings::drm_connector`
        unsafe { &*ptr.cast() }
    }
}

// SAFETY: We only expose this object to users directly after KmsDriver::create_objects has been
// called.
unsafe impl<T: KmsDriver> ModesettableConnector for OpaqueConnector<T> {
    type State = OpaqueConnectorState<T>;
}

// SAFETY: We don't expose OpaqueConnector<T> to users before `base` is initialized in
// Connector::new(), so `raw_mode_obj` always returns a valid pointer to a bindings::drm_mode_object.
unsafe impl<T: KmsDriver> ModeObject for OpaqueConnector<T> {
    type Driver = T;

    fn drm_dev(&self) -> &Device<Self::Driver> {
        // SAFETY: The parent device for a DRM connector will never outlive the connector, and this
        // pointer is invariant through the lifetime of the connector
        unsafe { Device::from_raw((*self.as_raw()).dev) }
    }

    fn raw_mode_obj(&self) -> *mut bindings::drm_mode_object {
        // SAFETY: We don't expose DRM connectors to users before `base` is initialized
        unsafe { &mut (*self.as_raw()).base }
    }
}

super::impl_aref_for_mode_object! {
    impl<T: KmsDriver> for OpaqueConnector<T>
}

// SAFETY: `funcs` is initialized by DRM when the connector is allocated
unsafe impl<T: KmsDriver> ModeObjectVtable for OpaqueConnector<T> {
    type Vtable = bindings::drm_connector_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        // SAFETY: `funcs` is initialized by DRM when the connector is allocated
        unsafe { *self.as_raw() }.funcs
    }
}

// SAFETY: Our connector interfaces are guaranteed to be thread-safe
unsafe impl<T: KmsDriver> Send for OpaqueConnector<T> {}
unsafe impl<T: KmsDriver> Sync for OpaqueConnector<T> {}

/// A privileged [`Connector`] obtained while holding a [`ModeConfigGuard`].
///
/// This provides access to various methods for [`Connector`] that must happen under lock, such as
/// setting resolution preferences and adding display modes.
///
/// # Invariants
///
/// Shares the invariants of [`ModeConfigGuard`].
#[derive(Copy, Clone)]
pub struct ConnectorGuard<'a, T: DriverConnector>(&'a Connector<T>);

impl<T: DriverConnector> Deref for ConnectorGuard<'_, T> {
    type Target = Connector<T>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// A connector being validated while its mode list is stable.
///
/// This is only constructed by the DRM connector-helper callback. It permits read-only iteration
/// over the connector's modes without exposing list pointers or extending a mode reference beyond
/// the callback.
#[derive(Copy, Clone)]
pub struct ConnectorModeValidation<'a, T: DriverConnector>(&'a Connector<T>);

impl<T: DriverConnector> Deref for ConnectorModeValidation<'_, T> {
    type Target = Connector<T>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<T: DriverConnector> ConnectorModeValidation<'_, T> {
    /// Return whether any mode currently on this connector satisfies `predicate`.
    pub fn any_mode(&self, mut predicate: impl FnMut(&DisplayMode) -> bool) -> bool {
        let raw = self.as_raw();
        // SAFETY: DRM only constructs this capability while `connector->modes` is stable. Each
        // list entry is an initialized `drm_display_mode`, and the shared reference is confined to
        // this callback invocation.
        unsafe {
            let head: *mut bindings::list_head = &raw mut (*raw).modes;
            let mut node = (*head).next;
            while node != head {
                let mode = crate::container_of!(node, bindings::drm_display_mode, head);
                if predicate(DisplayMode::as_ref(mode)) {
                    return true;
                }
                node = (*node).next;
            }
        }
        false
    }
}

impl<'a, T: DriverConnector> ConnectorGuard<'a, T> {
    /// Add modes for a [`ConnectorGuard`] without an EDID.
    ///
    /// Add the specified modes to the connector's mode list up to the given maximum resultion.
    /// Returns how many modes were added.
    pub fn add_modes_noedid(&self, (max_h, max_v): (u32, u32)) -> i32 {
        // SAFETY: We hold the locks required to call this via our type invariants.
        unsafe { bindings::drm_add_modes_noedid(self.as_raw(), max_h, max_v) }
    }

    /// Set the preferred display mode for the underlying [`Connector`].
    pub fn set_preferred_mode(&self, (h_pref, w_pref): (u32, u32)) {
        // SAFETY: We hold the locks required to call this via our type invariants.
        unsafe { bindings::drm_set_preferred_mode(self.as_raw(), h_pref, w_pref) }
    }

    /// Add a driver-synthesised CVT mode to this connector's probed mode list.
    ///
    /// For a display whose EDID declares continuous frequencies, a driver may legitimately offer a
    /// timing the EDID does not itself enumerate. `reduced` selects CVT reduced blanking (CVT-RB),
    /// which matters when the *pixel clock* rather than the pixel rate is the constrained
    /// resource -- RB cuts the clock for the same active pixels.
    ///
    /// Returns `EINVAL` if the core could not build the timing.
    pub fn add_cvt_mode(
        &self,
        hdisplay: i32,
        vdisplay: i32,
        vrefresh: i32,
        reduced: bool,
    ) -> Result {
        let dev = self.drm_dev().as_raw();
        // SAFETY: `dev` is this connector's live `drm_device`; `drm_cvt_mode` only computes a
        // timing and allocates it, and we hold the mode-config lock via our type invariants.
        let mode = unsafe {
            bindings::drm_cvt_mode(dev, hdisplay, vdisplay, vrefresh, reduced, false, false)
        };
        if mode.is_null() {
            return Err(EINVAL);
        }
        // SAFETY: `mode` was just allocated by `drm_cvt_mode` and ownership passes to the
        // connector here; we hold the locks required to modify its mode list.
        unsafe { bindings::drm_mode_probed_add(self.as_raw(), mode) };
        Ok(())
    }

    /// Parse an EDID, update the connector information, and add its advertised modes.
    ///
    /// Returns the number of modes added.
    pub fn add_edid_modes(&self, edid: &[u8]) -> Result<i32> {
        const EDID_BASE_BLOCK_LEN: usize = 128;

        if edid.len() < EDID_BASE_BLOCK_LEN {
            return Err(EINVAL);
        }

        // SAFETY: `edid` points to `edid.len()` initialized bytes, which the helper copies.
        let drm_edid = unsafe { bindings::drm_edid_alloc(edid.as_ptr().cast(), edid.len()) };
        if drm_edid.is_null() {
            return Err(ENOMEM);
        }

        // SAFETY: The connector is live and the guard holds the mode-config lock. `drm_edid`
        // points to an allocation returned by `drm_edid_alloc` above.
        let ret = unsafe { bindings::drm_edid_connector_update(self.as_raw(), drm_edid) };
        if let Err(err) = to_result(ret) {
            // SAFETY: `drm_edid` was allocated above and has not been freed.
            unsafe { bindings::drm_edid_free(drm_edid) };
            return Err(err);
        }

        // SAFETY: The connector information was successfully updated from this EDID above.
        let count = unsafe { bindings::drm_edid_connector_add_modes(self.as_raw()) };
        // SAFETY: `drm_edid` was allocated above and is no longer needed.
        unsafe { bindings::drm_edid_free(drm_edid) };

        Ok(count)
    }
}

/// A trait implemented by any type which can produce a reference to a
/// [`struct drm_connector_state`].
///
/// This is implemented internally by DRM.
///
/// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
pub trait AsRawConnectorState: private::AsRawConnectorState {
    /// The type that represents this connector state's DRM connector.
    type Connector: AsRawConnector;
}

pub(super) mod private {
    use super::*;

    /// Trait for retrieving references to the base connector state contained within any connector
    /// state compatible type
    #[allow(unreachable_pub)]
    pub trait AsRawConnectorState {
        /// Return an immutable reference to the raw connector state.
        fn as_raw(&self) -> &bindings::drm_connector_state;

        /// Get a mutable reference to the raw [`struct drm_connector_state`] contained within this
        /// type.
        ///
        ///
        /// # Safety
        ///
        /// The caller promises this mutable reference will not be used to modify any contents of
        /// [`struct drm_connector_state`] which DRM would consider to be static - like the
        /// backpointer to the DRM connector that owns this state. This also means the mutable
        /// reference should never be exposed outside of this crate.
        ///
        /// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
        unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_connector_state;
    }
}

pub(super) use private::AsRawConnectorState as AsRawConnectorStatePrivate;

/// A trait implemented for any type which can be constructed directly from a
/// [`struct drm_connector_state`] pointer.
///
/// This is implemented internally by DRM.
///
/// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
pub trait FromRawConnectorState: AsRawConnectorState {
    /// Get an immutable reference to this type from the given raw [`struct drm_connector_state`]
    /// pointer.
    ///
    /// # Safety
    ///
    /// - The caller guarantees `ptr` is contained within a valid instance of `Self`.
    /// - The caller guarantees that `ptr` cannot not be modified for the lifetime of `'a`.
    ///
    /// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
    unsafe fn from_raw<'a>(ptr: *const bindings::drm_connector_state) -> &'a Self;

    /// Get a mutable reference to this type from the given raw [`struct drm_connector_state`]
    /// pointer.
    ///
    /// # Safety
    ///
    /// - The caller guarantees that `ptr` is contained within a valid instance of `Self`.
    /// - The caller guarantees that `ptr` cannot have any other references taken out for the
    ///   lifetime of `'a`.
    ///
    /// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
    unsafe fn from_raw_mut<'a>(ptr: *mut bindings::drm_connector_state) -> &'a mut Self;
}

/// Common methods available on any type which implements [`AsRawConnectorState`].
///
/// This is implemented internally by DRM, and provides many of the basic methods for working with
/// the atomic state of [`Connector`]s.
pub trait RawConnectorState: AsRawConnectorState {
    /// Return the connector that this atomic state belongs to.
    fn connector(&self) -> &Self::Connector {
        // SAFETY: This is guaranteed safe by type invariance, and we're guaranteed by DRM that
        // `self.state.connector` points to a valid instance of a `Connector<T>`
        unsafe { Self::Connector::from_raw((*self.as_raw()).connector) }
    }

    /// The colorimetry userspace has requested through the `Colorspace` property, as a
    /// [`enum drm_colorspace`] value.
    ///
    /// Meaningful only on a connector that
    /// [`UnregisteredConnector::attach_colorspace_property`] was called for; everything else
    /// leaves it at `DRM_MODE_COLORIMETRY_DEFAULT`.
    ///
    /// [`enum drm_colorspace`]: srctree/include/drm/drm_connector.h
    fn colorspace(&self) -> u32 {
        self.as_raw().colorspace
    }

    /// The bits per colour channel userspace has asked the link to carry, through the `max bpc`
    /// property.
    ///
    /// This is a property of the *link*, not of the framebuffer: userspace routinely scans out an
    /// eight-bit surface over a ten-bit link, and a driver that derives its output depth from the
    /// framebuffer format alone silently ignores what was asked for.
    ///
    /// Meaningful only on a connector that
    /// [`UnregisteredConnector::attach_max_bpc_property`] was called for; everything else leaves
    /// it at zero.
    fn max_requested_bpc(&self) -> u32 {
        // `max_requested_bpc` is an `unsigned int` clamped by DRM to the range the driver gave
        // `drm_connector_attach_max_bpc_property()`, so it needs no validation here.
        self.as_raw().max_requested_bpc as u32
    }

    /// The electro-optical transfer function from the `HDR_OUTPUT_METADATA` blob, or [`None`] if
    /// userspace has not set one.
    ///
    /// This is deliberately just the curve: the rest of the infoframe is mastering-display
    /// metadata for the sink, and a driver that only needs to know *which curve the pixels are
    /// encoded in* should not have to reason about the union's other members or their versioning.
    ///
    /// [`struct hdr_output_metadata`]: srctree/include/uapi/drm/drm_mode.h
    fn hdr_output_eotf(&self) -> Option<Eotf> {
        let blob = self.as_raw().hdr_output_metadata;
        if blob.is_null() {
            return None;
        }
        // SAFETY: a non-null `hdr_output_metadata` blob is valid for the state's lifetime.
        let (data, length) = unsafe { ((*blob).data, (*blob).length) };
        // DRM validates the blob length when the property is set, but this is the boundary where
        // a short blob would become an out-of-bounds read.
        if data.is_null() || length < core::mem::size_of::<bindings::hdr_output_metadata>() {
            return None;
        }
        // SAFETY: the blob is at least a whole `hdr_output_metadata` and lives as long as the
        // state. `eotf` is the first byte of the only union member DRM defines.
        let eotf = unsafe {
            (*data.cast::<bindings::hdr_output_metadata>())
                .__bindgen_anon_1
                .hdmi_metadata_type1
                .eotf
        };
        Some(Eotf::from_raw(eotf))
    }
}
/// An electro-optical transfer function named by a `HDR_OUTPUT_METADATA` blob.
///
/// Mirrors the `HDMI_EOTF_*` values in [`enum hdmi_eotf`]. A driver matches on this rather than
/// comparing against the raw constants, so the one place that has to agree with the C enum is
/// [`Eotf::from_raw`].
///
/// [`enum hdmi_eotf`]: srctree/include/linux/hdmi.h
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Eotf {
    /// Ordinary SDR gamma.
    TraditionalGammaSdr,
    /// The traditional HDR gamma curve.
    TraditionalGammaHdr,
    /// SMPTE ST 2084, i.e. PQ. What a compositor sets to drive an output in HDR10.
    SmpteSt2084,
    /// BT.2100 hybrid log-gamma.
    Bt2100Hlg,
    /// A value this kernel does not name, carried through rather than discarded.
    Other(u8),
}

impl Eotf {
    /// Classify the raw `eotf` byte from an infoframe.
    fn from_raw(eotf: u8) -> Self {
        match u32::from(eotf) {
            bindings::hdmi_eotf_HDMI_EOTF_TRADITIONAL_GAMMA_SDR => Self::TraditionalGammaSdr,
            bindings::hdmi_eotf_HDMI_EOTF_TRADITIONAL_GAMMA_HDR => Self::TraditionalGammaHdr,
            bindings::hdmi_eotf_HDMI_EOTF_SMPTE_ST2084 => Self::SmpteSt2084,
            bindings::hdmi_eotf_HDMI_EOTF_BT_2100_HLG => Self::Bt2100Hlg,
            _ => Self::Other(eotf),
        }
    }
}

impl<T: AsRawConnectorState> RawConnectorState for T {}

/// The main interface for a [`struct drm_connector_state`].
///
/// This type is the main interface for dealing with the atomic state of DRM connectors. In
/// addition, it allows access to whatever private data is contained within an implementor's
/// [`DriverConnectorState`] type.
///
/// # Invariants
///
/// - The DRM C API and our interface guarantees that only the user has mutable access to `state`,
///   up until [`drm_atomic_helper_commit_hw_done`] is called. Therefore, `connector` follows rust's
///   data aliasing rules and does not need to be behind an [`Opaque`] type.
/// - `state` and `inner` initialized for as long as this object is exposed to users.
/// - The data layout of this structure begins with [`struct drm_connector_state`].
/// - The connector for this atomic state can always be assumed to be of type
///   [`Connector<T::Connector>`].
///
/// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
/// [`drm_atomic_helper_commit_hw_done`]: srctree/include/drm/drm_atomic_helper.h
#[derive(Default)]
#[repr(C)]
pub struct ConnectorState<T: DriverConnectorState> {
    state: bindings::drm_connector_state,
    inner: T,
}

/// The main trait for implementing the [`struct drm_connector_state`] API for a [`Connector`].
///
/// A driver may store driver-private data within the implementor's type, which will be available
/// when using a full typed [`ConnectorState`] object.
///
/// # Invariants
///
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_connector`] pointers are contained within a [`Connector<Self::Connector>`].
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_connector_state`] pointers are contained within a [`ConnectorState<Self>`].
///
/// [`struct drm_connector`]: srctree/include/drm_connector.h
/// [`struct drm_connector_state`]: srctree/include/drm_connector.h
pub trait DriverConnectorState: Clone + Default + Sized {
    /// The parent [`DriverConnector`].
    type Connector: DriverConnector;
}

impl<T: DriverConnectorState> Sealed for ConnectorState<T> {}

impl<T: DriverConnectorState> AsRawConnectorState for ConnectorState<T> {
    type Connector = Connector<T::Connector>;
}

impl<T: DriverConnectorState> private::AsRawConnectorState for ConnectorState<T> {
    fn as_raw(&self) -> &bindings::drm_connector_state {
        &self.state
    }

    unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_connector_state {
        &mut self.state
    }
}

impl<T: DriverConnectorState> FromRawConnectorState for ConnectorState<T> {
    unsafe fn from_raw<'a>(ptr: *const bindings::drm_connector_state) -> &'a Self {
        // Our data layout starts with `bindings::drm_connector_state`.
        let ptr: *const Self = ptr.cast();

        // SAFETY:
        // - Our safety contract requires that `ptr` be contained within `Self`.
        // - Our safety contract requires the caller ensure that it is safe for us to take an
        //   immutable reference.
        unsafe { &*ptr }
    }

    unsafe fn from_raw_mut<'a>(ptr: *mut bindings::drm_connector_state) -> &'a mut Self {
        // Our data layout starts with `bindings::drm_connector_state`.
        let ptr: *mut Self = ptr.cast();

        // SAFETY:
        // - Our safety contract requires that `ptr` be contained within `Self`.
        // - Our safety contract requires the caller ensure it is safe for us to take a mutable
        //   reference.
        unsafe { &mut *ptr }
    }
}

// SAFETY: `funcs` is initialized by DRM when the connector is allocated
unsafe impl<T: DriverConnectorState> ModeObjectVtable for ConnectorState<T> {
    type Vtable = bindings::drm_connector_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        self.connector().vtable()
    }
}

impl<T: DriverConnectorState> ConnectorState<T> {
    super::impl_from_opaque_mode_obj! {
        fn <'a, D, C>(&'a OpaqueConnectorState<D>) -> &'a Self
        where
            T: DriverConnectorState<Connector = C>;
        use
            C as DriverConnector,
            D as KmsDriver<Connector = ...>
    }
}

/// A [`struct drm_connector_state`] without a known [`DriverConnectorState`] implementation.
///
/// This is mainly for situations where our bindings can't infer the [`DriverConnectorState`]
/// implementation for a [`struct drm_connector_state`] automatically. It is identical to
/// [`Connector`], except that it does not provide access to the driver's private data.
///
/// # Invariants
///
/// - `state` is initialized for as long as this object is exposed to users.
/// - The data layout of this type is identical to [`struct drm_connector_state`].
/// - The DRM C API and our interface guarantees that only the user has mutable access to `state`,
///   up until [`drm_atomic_helper_commit_hw_done`] is called. Therefore, `connector` follows rust's
///   data aliasing rules and does not need to be behind an [`Opaque`] type.
///
/// [`struct drm_connector_state`]: srctree/include/drm/drm_connector.h
/// [`drm_atomic_helper_commit_hw_done`]: srctree/include/drm/drm_atomic_helper.h
#[repr(transparent)]
pub struct OpaqueConnectorState<T: KmsDriver> {
    state: bindings::drm_connector_state,
    _p: PhantomData<T>,
}

impl<T: KmsDriver> AsRawConnectorState for OpaqueConnectorState<T> {
    type Connector = OpaqueConnector<T>;
}

impl<T: KmsDriver> private::AsRawConnectorState for OpaqueConnectorState<T> {
    fn as_raw(&self) -> &bindings::drm_connector_state {
        &self.state
    }

    unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_connector_state {
        &mut self.state
    }
}

impl<T: KmsDriver> FromRawConnectorState for OpaqueConnectorState<T> {
    unsafe fn from_raw<'a>(ptr: *const bindings::drm_connector_state) -> &'a Self {
        // SAFETY: Our data layout is identical to `bindings::drm_connector_state`
        unsafe { &*ptr.cast() }
    }

    unsafe fn from_raw_mut<'a>(ptr: *mut bindings::drm_connector_state) -> &'a mut Self {
        // SAFETY: Our data layout is identical to `bindings::drm_connector_state`
        unsafe { &mut *ptr.cast() }
    }
}

// SAFETY: See OpaqueConnector's ModeObjectVtable implementation
unsafe impl<T: KmsDriver> ModeObjectVtable for OpaqueConnectorState<T> {
    type Vtable = bindings::drm_connector_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        self.connector().vtable()
    }
}

/// An interface for mutating a [`Connector`]s atomic state.
///
/// This type is typically returned by an [`AtomicStateMutator`] within contexts where it is
/// possible to safely mutate a connector's state. In order to uphold rust's data-aliasing rules,
/// only [`ConnectorStateMutator`] may exist at a time.
pub struct ConnectorStateMutator<'a, T: FromRawConnectorState> {
    state: &'a mut T,
    mask: &'a Cell<u32>,
}

impl<'a, T: FromRawConnectorState> ConnectorStateMutator<'a, T> {
    pub(super) fn new<D: KmsDriver>(
        mutator: &'a AtomicStateMutator<D>,
        state: NonNull<bindings::drm_connector_state>,
    ) -> Option<Self> {
        // SAFETY:
        // - `connector` is invariant throughout the lifetime of the atomic state.
        // - `state` is initialized by the time it is passed to this function.
        // - We're guaranteed that `state` is compatible with `drm_connector` by type invariants.
        let connector = unsafe { T::Connector::from_raw((*state.as_ptr()).connector) };
        let conn_mask = connector.mask();
        let borrowed_mask = mutator.borrowed_connectors.get();

        if borrowed_mask & conn_mask == 0 {
            mutator.borrowed_connectors.set(borrowed_mask | conn_mask);
            Some(Self {
                mask: &mutator.borrowed_connectors,
                // SAFETY: We're guaranteed `state` is of `T` by type invariance, and we just
                // confirmed by checking `borrowed_connectors` that no other mutable borrows have
                // been taken out for `state`
                state: unsafe { T::from_raw_mut(state.as_ptr()) },
            })
        } else {
            None
        }
    }
}

impl<'a, T: DriverConnectorState> Deref for ConnectorStateMutator<'a, ConnectorState<T>> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.state.inner
    }
}

impl<'a, T: DriverConnectorState> DerefMut for ConnectorStateMutator<'a, ConnectorState<T>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state.inner
    }
}

impl<'a, T: FromRawConnectorState> Drop for ConnectorStateMutator<'a, T> {
    fn drop(&mut self) {
        let mask = self.state.connector().mask();
        self.mask.set(self.mask.get() & !mask);
    }
}

impl<'a, T: FromRawConnectorState> AsRawConnectorState for ConnectorStateMutator<'a, T> {
    type Connector = T::Connector;
}

impl<'a, T: FromRawConnectorState> private::AsRawConnectorState for ConnectorStateMutator<'a, T> {
    fn as_raw(&self) -> &bindings::drm_connector_state {
        self.state.as_raw()
    }

    unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_connector_state {
        // SAFETY: We're bound by the same safety contract as this function
        unsafe { self.state.as_raw_mut() }
    }
}

// SAFETY: we inherit the safety guarantees of `T`
unsafe impl<'a, T> ModeObjectVtable for ConnectorStateMutator<'a, T>
where
    T: FromRawConnectorState + ModeObjectVtable,
{
    type Vtable = T::Vtable;

    fn vtable(&self) -> *const Self::Vtable {
        self.state.vtable()
    }
}

impl<'a, T: DriverConnectorState> ConnectorStateMutator<'a, ConnectorState<T>> {
    super::impl_from_opaque_mode_obj! {
        fn <D, C>(ConnectorStateMutator<'a, OpaqueConnectorState<D>>) -> Self
        where
            T: DriverConnectorState<Connector = C>;
        use
            C as DriverConnector,
            D as KmsDriver<Connector = ...>
    }
}

unsafe extern "C" fn atomic_duplicate_state_callback<T: DriverConnectorState>(
    connector: *mut bindings::drm_connector,
) -> *mut bindings::drm_connector_state {
    // SAFETY: DRM guarantees that `connector` points to a valid initialized `drm_connector`.
    let state = unsafe { (*connector).state };
    if state.is_null() {
        return null_mut();
    }

    // SAFETY:
    // - We just verified that `state` is non-null
    // - This cast is guaranteed to be safe via our type invariants.
    let state = unsafe { ConnectorState::<T>::from_raw(state) };

    let new: Result<KBox<_>> = KBox::init(
        init!(ConnectorState::<T> {
            inner: state.inner.clone(),
            state: bindings::drm_connector_state {
                ..Default::default()
            },
        }),
        GFP_KERNEL,
    );

    if let Ok(mut new) = new {
        // SAFETY:
        // - `new` provides a valid pointer to a newly allocated `drm_plane_state` via type
        //   invariants
        // - This initializes `new` via memcpy()
        unsafe {
            bindings::__drm_atomic_helper_connector_duplicate_state(connector, new.as_raw_mut())
        };

        KBox::into_raw(new).cast()
    } else {
        null_mut()
    }
}

unsafe extern "C" fn atomic_destroy_state_callback<T: DriverConnectorState>(
    _connector: *mut bindings::drm_connector,
    connector_state: *mut bindings::drm_connector_state,
) {
    // SAFETY: DRM guarantees that `state` points to a valid instance of `drm_connector_state`
    unsafe { bindings::__drm_atomic_helper_connector_destroy_state(connector_state) };

    // SAFETY:
    // - DRM guarantees we are the only one with access to this `drm_connector_state`
    // - This cast is safe via our type invariants.
    drop(unsafe { KBox::from_raw(connector_state.cast::<ConnectorState<T>>()) });
}

unsafe extern "C" fn connector_reset_callback<T: DriverConnectorState>(
    connector: *mut bindings::drm_connector,
) {
    // SAFETY: DRM guarantees that `state` points to a valid instance of `drm_connector_state`
    let state = unsafe { (*connector).state };
    if !state.is_null() {
        // SAFETY:
        // - We're guaranteed `connector` is `Connector<T>` via type invariants
        // - We're guaranteed `state` is `ConnectorState<T>` via type invariants.
        unsafe { atomic_destroy_state_callback::<T>(connector, state) }

        // SAFETY: No special requirements here, DRM expects this to be NULL
        unsafe { (*connector).state = null_mut() };
    }

    // Unfortunately, this is the best we can do at the moment as this FFI callback was mistakenly
    // presumed to be infallible :(
    let new = KBox::new(ConnectorState::<T>::default(), GFP_KERNEL).expect("Blame the API, sorry!");

    // DRM takes ownership of the state from here, resets it, and then assigns it to the connector
    // SAFETY:
    // - DRM guarantees that `connector` points to a valid instance of `drm_connector`.
    // - The cast to `drm_connector_state` is safe via `ConnectorState`s type invariants.
    unsafe { bindings::__drm_atomic_helper_connector_reset(connector, Box::into_raw(new).cast()) };
}
