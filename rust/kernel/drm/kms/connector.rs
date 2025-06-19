// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM display connectors.
//!
//! C header: [`include/drm/drm_connector.h`](srctree/include/drm/drm_connector.h)

use super::{encoder::*, KmsDriver, ModeObject, Sealed};
use crate::{
    alloc::KBox,
    bindings,
    drm::{device::Device, kms::UnregisteredKmsDevice},
    error::to_result,
    prelude::*,
    types::{NotThreadSafe, Opaque},
};
use core::{
    marker::*,
    mem,
    ops::*,
    ptr::null_mut,
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
            detect: None,
            fill_modes: None,
            debugfs_init: None,
            oob_hotplug_event: None,
            atomic_duplicate_state: Some(atomic_duplicate_state_callback::<Self::State>),
        },
        helper_funcs: bindings::drm_connector_helper_funcs {
            mode_valid: None,
            atomic_check: None,
            get_modes: None,
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
    pub fn attach_encoder(&self, encoder: &impl AsRawEncoder) -> Result {
        // SAFETY:
        // - Both as_raw() calls are guaranteed to return a valid pointer
        // - We're guaranteed this connector is not registered via our type invariants, thus this
        //   function is safe to call
        to_result(unsafe {
            bindings::drm_connector_attach_encoder(self.as_raw(), encoder.as_raw())
        })
    }
}

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
