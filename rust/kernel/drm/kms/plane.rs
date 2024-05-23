// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM display planes.
//!
//! C header: [`include/drm/drm_plane.h`](srctree/include/drm/drm_plane.h)

use super::{
    atomic::*, KmsDriver, ModeObject, ModeObjectVtable, StaticModeObject, UnregisteredKmsDevice,
    Sealed
};
use crate::{
    alloc::KBox,
    bindings,
    drm::{device::Device, fourcc::*},
    error::{to_result, Error},
    prelude::*,
    types::{NotThreadSafe, Opaque},
};
use core::{
    cell::Cell,
    marker::*,
    mem,
    ops::*,
    pin::Pin,
    ptr::{null, null_mut, NonNull},
};

/// The main trait for implementing the [`struct drm_plane`] API for [`Plane`].
///
/// Any KMS driver should have at least one implementation of this type, which allows them to create
/// [`Plane`] objects. Additionally, a driver may store driver-private data within the type that
/// implements [`DriverPlane`] - and it will be made available when using a fully typed [`Plane`]
/// object.
///
/// # Invariants
///
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_plane`] pointers are contained within a [`Plane<Self>`].
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_plane_state`] pointers are contained within a [`PlaneState<Self::State>`].
///
/// [`struct drm_plane`]: srctree/include/drm/drm_plane.h
/// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
#[vtable]
pub trait DriverPlane: Send + Sync + Sized {
    /// The generated C vtable for this [`DriverPlane`] implementation.
    const OPS: &'static DriverPlaneOps = &DriverPlaneOps {
        funcs: bindings::drm_plane_funcs {
            update_plane: Some(bindings::drm_atomic_helper_update_plane),
            disable_plane: Some(bindings::drm_atomic_helper_disable_plane),
            destroy: Some(plane_destroy_callback::<Self>),
            reset: Some(plane_reset_callback::<Self>),
            set_property: None,
            atomic_duplicate_state: Some(atomic_duplicate_state_callback::<Self::State>),
            atomic_destroy_state: Some(atomic_destroy_state_callback::<Self::State>),
            atomic_set_property: None,
            atomic_get_property: None,
            late_register: None,
            early_unregister: None,
            atomic_print_state: None,
            format_mod_supported: None,
            format_mod_supported_async: None,
        },

        helper_funcs: bindings::drm_plane_helper_funcs {
            prepare_fb: None,
            cleanup_fb: None,
            begin_fb_access: None,
            end_fb_access: None,
            atomic_check: None,
            atomic_update: None,
            atomic_enable: None,
            atomic_disable: None,
            atomic_async_check: None,
            atomic_async_update: None,
            panic_flush: None,
            get_scanout_buffer: None,
        },
    };

    /// The type to pass to the `args` field of [`UnregisteredPlane::new`].
    ///
    /// This type will be made available in in the `args` argument of [`Self::new`]. Drivers which
    /// don't need this can simply pass [`()`] here.
    type Args;

    /// The parent [`KmsDriver`] implementation.
    type Driver: KmsDriver;

    /// The [`DriverPlaneState`] implementation for this [`DriverPlane`].
    ///
    /// See [`DriverPlaneState`] for more info.
    type State: DriverPlaneState;

    /// The constructor for creating a [`Plane`] using this [`DriverPlane`] implementation.
    ///
    /// Drivers may use this to instantiate their [`DriverPlane`] object.
    fn new(device: &Device<Self::Driver>, args: Self::Args) -> impl PinInit<Self, Error>;
}

/// The generated C vtable for a [`DriverPlane`].
///
/// This type is created internally by DRM.
pub struct DriverPlaneOps {
    funcs: bindings::drm_plane_funcs,
    helper_funcs: bindings::drm_plane_helper_funcs,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
/// An enumerator describing a type of [`Plane`].
///
/// This is mainly just relevant for DRM legacy drivers.
///
/// # Invariants
///
/// This type is identical to [`enum drm_plane_type`].
///
/// [`enum drm_plane_type`]: srctree/include/drm/drm_plane.h
pub enum Type {
    /// Overlay planes represent all non-primary, non-cursor planes. Some drivers refer to these
    /// types of planes as "sprites" internally.
    Overlay = bindings::drm_plane_type_DRM_PLANE_TYPE_OVERLAY,

    /// A primary plane attached to a CRTC that is the most likely to be able to light up the CRTC
    /// when no scaling/cropping is used, and the plane covers the whole CRTC.
    Primary = bindings::drm_plane_type_DRM_PLANE_TYPE_PRIMARY,

    /// A cursor plane attached to a CRTC that is more likely to be enabled when no scaling/cropping
    /// is used, and the framebuffer has the size indicated by [`ModeConfigInfo::max_cursor`].
    ///
    /// [`ModeConfigInfo::max_cursor`]: crate::drm::kms::ModeConfigInfo
    Cursor = bindings::drm_plane_type_DRM_PLANE_TYPE_CURSOR,
}

/// The main interface for a [`struct drm_plane`].
///
/// This type is the main interface for dealing with DRM planes. In addition, it also allows
/// immutable access to whatever private data is contained within an implementor's [`DriverPlane`]
/// type.
///
/// # Invariants
///
/// - `plane` and `inner` are initialized for as long as this object is made available to users.
/// - The data layout of this structure begins with [`struct drm_plane`].
/// - The atomic state for this type can always be assumed to be of type [`PlaneState<T::State>`].
///
/// [`struct drm_plane`]: srctree/include/drm/drm_plane.h
#[repr(C)]
#[pin_data]
pub struct Plane<T: DriverPlane> {
    /// The FFI drm_plane object
    plane: Opaque<bindings::drm_plane>,
    /// The driver's private inner data
    #[pin]
    inner: T,
    #[pin]
    _p: PhantomPinned,
}

impl<T: DriverPlane> Sealed for Plane<T> {}

impl<T: DriverPlane> Deref for Plane<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// SAFETY: `funcs` is initialized when the plane is allocated
unsafe impl<T: DriverPlane> ModeObjectVtable for Plane<T> {
    type Vtable = bindings::drm_plane_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        // SAFETY: `as_raw()` always returns a valid plane pointer
        unsafe { *self.as_raw() }.funcs
    }
}

impl<T: DriverPlane> Plane<T> {
    super::impl_from_opaque_mode_obj! {
        fn <'a, D>(&'a OpaquePlane<D>) -> &'a Self;
        use
            T as DriverPlane,
            D as KmsDriver<Plane = ...>
    }
}

/// A [`Plane`] that has not yet been registered with userspace.
///
/// KMS registration is single-threaded, so this object is not thread-safe.
///
/// # Invariants
///
/// - This object can only exist before its respective KMS device has been registered.
/// - Otherwise, it inherits all invariants of [`Plane`] and has an identical data layout.
pub struct UnregisteredPlane<T: DriverPlane>(Plane<T>, NotThreadSafe);

// SAFETY: We share the invariants of `Plane`
unsafe impl<T: DriverPlane> AsRawPlane for UnregisteredPlane<T> {
    fn as_raw(&self) -> *mut bindings::drm_plane {
        self.0.as_raw()
    }

    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_plane) -> &'a Self {
        // SAFETY: This is another from_raw() call, so this function shares the same safety contract
        let plane = unsafe { Plane::<T>::from_raw(ptr) };

        // SAFETY: Our data layout is identical via our type invariants.
        unsafe { mem::transmute(plane) }
    }
}

impl<T: DriverPlane> Deref for UnregisteredPlane<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}

impl<T: DriverPlane> UnregisteredPlane<T> {
    /// Construct a new [`UnregisteredPlane`].
    ///
    /// A driver may use this from their [`KmsDriver::create_objects`] callback in order to
    /// construct new [`UnregisteredPlane`] objects.
    ///
    /// [`KmsDriver::create_objects`]: kernel::drm::kms::KmsDriver::create_objects
    pub fn new<'a, 'b: 'a>(
        dev: &'a UnregisteredKmsDevice<'a, T::Driver>,
        possible_crtcs: u32,
        formats: &[u32],
        format_modifiers: Option<&[u64]>,
        type_: Type,
        name: Option<&CStr>,
        args: T::Args,
    ) -> Result<&'b Self> {
        let this: Pin<KBox<Plane<T>>> = KBox::try_pin_init(
            try_pin_init!(Plane {
                plane: Opaque::new(bindings::drm_plane {
                    helper_private: &T::OPS.helper_funcs,
                    ..Default::default()
                }),
                inner <- T::new(dev, args),
                _p: PhantomPinned
            }),
            GFP_KERNEL,
        )?;

        // TODO: Move this over to using collect() someday
        // Create a modifiers array with the sentinel for passing to DRM
        let format_modifiers_raw;
        if let Some(modifiers) = format_modifiers {
            let mut raw = KVec::with_capacity(modifiers.len() + 1, GFP_KERNEL)?;
            for modifier in modifiers {
                raw.push(*modifier, GFP_KERNEL)?;
            }
            raw.push(FORMAT_MOD_INVALID, GFP_KERNEL)?;

            format_modifiers_raw = Some(raw);
        } else {
            format_modifiers_raw = None;
        }

        // SAFETY:
        // - `dev` handles destroying the plane, and thus will outlive us and always be valid.
        // - We just allocated `this`, and we won't move it since it's pinned
        // - We just allocated the `format_modifiers_raw` vec and added the sentinel DRM expects
        //   above
        // - `drm_universal_plane_init` will memcpy() the following parameters into its own storage,
        //   so it's safe for them to become inaccessible after this call returns:
        //   - `formats`
        //   - `format_modifiers_raw`
        //   - `name`
        // - `type_` is equivalent to `drm_plane_type` via its type invariants.
        to_result(unsafe {
            bindings::drm_universal_plane_init(
                dev.as_raw(),
                this.as_raw(),
                possible_crtcs,
                &T::OPS.funcs,
                formats.as_ptr(),
                formats.len() as _,
                format_modifiers_raw.map_or(null(), |f| f.as_ptr()),
                type_ as _,
                name.map_or(null(), |n| n.as_char_ptr()),
            )
        })?;

        // SAFETY: We don't move anything
        let this = unsafe { Pin::into_inner_unchecked(this) };

        // We'll re-assemble the box in plane_destroy_callback()
        let this = KBox::into_raw(this);

        // UnregisteredPlane has an equivalent data layout
        let this: *mut Self = this.cast();

        // SAFETY: We just allocated the plane above, so this pointer must be valid
        Ok(unsafe { &*this })
    }
}

/// A trait implemented by any type that acts as a [`struct drm_plane`] interface.
///
/// This is implemented internally by DRM.
///
/// # Safety
///
/// [`as_raw()`] must always return a valid pointer to an initialized [`struct drm_plane`].
///
/// [`struct drm_plane`]: srctree/include/drm/drm_plane.h
/// [`as_raw()`]: AsRawPlane::as_raw()
pub unsafe trait AsRawPlane {
    /// Return the raw `bindings::drm_plane` for this DRM plane.
    ///
    /// Drivers should never use this directly.
    fn as_raw(&self) -> *mut bindings::drm_plane;

    /// Convert a raw `bindings::drm_plane` pointer into an object of this type.
    ///
    /// # Safety
    ///
    /// Callers promise that `ptr` points to a valid instance of this type
    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_plane) -> &'a Self;
}

// SAFETY:
// - Via our type variants our data layout starts with `drm_plane`
// - Since we don't expose `plane` to users before it has been initialized, this and our data
//   layout ensure that `as_raw()` always returns a valid pointer to a `drm_plane`.
unsafe impl<T: DriverPlane> AsRawPlane for Plane<T> {
    fn as_raw(&self) -> *mut bindings::drm_plane {
        self.plane.get()
    }

    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_plane) -> &'a Self {
        // Our data layout start with `bindings::drm_plane`.
        let ptr: *mut Self = ptr.cast();

        // SAFETY: Our safety contract requires that `ptr` point to a valid intance of `Self`.
        unsafe { &*ptr }
    }
}

// SAFETY: We only expose this object to users directly after KmsDriver::create_objects has been
// called.
unsafe impl<T: DriverPlane> ModesettablePlane for Plane<T> {
    type State = PlaneState<T::State>;
}

// SAFETY: We don't expose Plane<T> to users before `base` is initialized in ::new(), so
// `raw_mode_obj` always returns a valid pointer to a bindings::drm_mode_object.
unsafe impl<T: DriverPlane> ModeObject for Plane<T> {
    type Driver = T::Driver;

    fn drm_dev(&self) -> &Device<Self::Driver> {
        // SAFETY: DRM planes exist for as long as the device does, so this pointer is always valid
        unsafe { Device::from_raw((*self.as_raw()).dev) }
    }

    fn raw_mode_obj(&self) -> *mut bindings::drm_mode_object {
        // SAFETY: We don't expose DRM planes to users before `base` is initialized
        unsafe { &raw mut (*self.as_raw()).base }
    }
}

// SAFETY: Planes do not have a refcount
unsafe impl<T: DriverPlane> StaticModeObject for Plane<T> {}

// SAFETY: Our interface is thread-safe.
unsafe impl<T: DriverPlane> Send for Plane<T> {}

// SAFETY: Our interface is thread-safe.
unsafe impl<T: DriverPlane> Sync for Plane<T> {}

/// A supertrait of [`AsRawPlane`] for [`struct drm_plane`] interfaces that can perform modesets.
///
/// This is implemented internally by DRM.
///
/// # Safety
///
/// Any object implementing this trait must only be made directly available to the user after
/// [`create_objects`] has completed.
///
/// [`struct drm_plane`]: srctree/include/drm/drm_plane.h
/// [`create_objects`]: KmsDriver::create_objects
pub unsafe trait ModesettablePlane: AsRawPlane {
    /// The type that should be returned for a plane state acquired using this plane interface
    type State: FromRawPlaneState;
}

/// Common methods available on any type which implements [`AsRawPlane`].
///
/// This is implemented internally by DRM, and provides many of the basic methods for working with
/// planes.
pub trait RawPlane: AsRawPlane {
    /// Return the index of this DRM plane
    #[inline]
    fn index(&self) -> u32 {
        // SAFETY:
        // - The index is initialized by the time we expose planes to users, and does not change
        //   throughout its lifetime
        // - `.as_raw()` always returns a valid poiinter.
        unsafe { *self.as_raw() }.index
    }

    /// Return the index of this DRM plane in the form of a bitmask
    #[inline]
    fn mask(&self) -> u32 {
        1 << self.index()
    }
}
impl<T: AsRawPlane> RawPlane for T {}

/// A [`struct drm_plane`] without a known [`DriverPlane`] implementation.
///
/// This is mainly for situations where our bindings can't infer the [`DriverPlane`] implementation
/// for a [`struct drm_plane`] automatically. It is identical to [`Plane`], except that it does not
/// provide access to the driver's private data.
///
/// It may be upcasted to a full [`Plane`] using [`Plane::from_opaque`] or
/// [`Plane::try_from_opaque`].
///
/// # Invariants
///
/// - `plane` is initialized for as long as this object is made available to users.
/// - The data layout of this structure is equivalent to [`struct drm_plane`].
///
/// [`struct drm_plane`]: srctree/include/drm/drm_plane.h
#[repr(transparent)]
pub struct OpaquePlane<T: KmsDriver> {
    plane: Opaque<bindings::drm_plane>,
    _p: PhantomData<T>,
}

impl<T: KmsDriver> Sealed for OpaquePlane<T> {}

// SAFETY:
// * Via our type variants our data layout is identical to `drm_plane`
// * Since we don't expose `plane` to users before it has been initialized, this and our data
//   layout ensure that `as_raw()` always returns a valid pointer to a `drm_plane`.
unsafe impl<T: KmsDriver> AsRawPlane for OpaquePlane<T> {
    fn as_raw(&self) -> *mut bindings::drm_plane {
        self.plane.get()
    }

    unsafe fn from_raw<'a>(ptr: *mut bindings::drm_plane) -> &'a Self {
        // SAFETY: Our data layout is identical to `bindings::drm_plane`
        unsafe { &*ptr.cast() }
    }
}

// SAFETY: We only expose this object to users directly after KmsDriver::create_objects has been
// called.
unsafe impl<T: KmsDriver> ModesettablePlane for OpaquePlane<T> {
    type State = OpaquePlaneState<T>;
}

// SAFETY: We don't expose OpaquePlane<T> to users before `base` is initialized in
// Plane::<T>::new(), so `raw_mode_obj` always returns a valid pointer to a
// bindings::drm_mode_object.
unsafe impl<T: KmsDriver> ModeObject for OpaquePlane<T> {
    type Driver = T;

    fn drm_dev(&self) -> &Device<Self::Driver> {
        // SAFETY: DRM planes exist for as long as the device does, so this pointer is always valid
        unsafe { Device::from_raw((*self.as_raw()).dev) }
    }

    fn raw_mode_obj(&self) -> *mut bindings::drm_mode_object {
        // SAFETY: We don't expose DRM planes to users before `base` is initialized
        unsafe { &raw mut (*self.as_raw()).base }
    }
}

// SAFETY: Planes do not have a refcount
unsafe impl<T: KmsDriver> StaticModeObject for OpaquePlane<T> {}

// SAFETY: `funcs` is initialized when the plane is allocated
unsafe impl<T: KmsDriver> ModeObjectVtable for OpaquePlane<T> {
    type Vtable = bindings::drm_plane_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        // SAFETY: `as_raw()` always returns a valid pointer to a plane
        unsafe { *self.as_raw() }.funcs
    }
}

// SAFETY: Our plane interfaces are guaranteed to be thread-safe
unsafe impl<T: KmsDriver> Send for OpaquePlane<T> {}
unsafe impl<T: KmsDriver> Sync for OpaquePlane<T> {}

/// A trait implemented by any type which can produce a reference to a [`struct drm_plane_state`].
///
/// This is implemented internally by DRM.
///
/// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
pub trait AsRawPlaneState: private::AsRawPlaneState {
    /// The type that this plane state interface returns to represent the parent DRM plane
    type Plane: ModesettablePlane;
}

pub(crate) mod private {
    /// Trait for retrieving references to the base plane state contained within any plane state
    /// compatible type
    #[allow(unreachable_pub)]
    pub trait AsRawPlaneState {
        /// Return an immutable reference to the raw plane state
        fn as_raw(&self) -> &bindings::drm_plane_state;

        /// Get a mutable reference to the raw `bindings::drm_plane_state` contained within this
        /// type.
        ///
        /// # Safety
        ///
        /// The caller promises this mutable reference will not be used to modify any contents of
        /// `bindings::drm_plane_state` which DRM would consider to be static - like the backpointer
        /// to the DRM plane that owns this state. This also means the mutable reference should
        /// never be exposed outside of this crate.
        unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_plane_state;
    }
}

pub(crate) use private::AsRawPlaneState as AsRawPlaneStatePrivate;

/// A trait implemented for any type which can be constructed directly from a
/// [`struct drm_plane_state`] pointer.
///
/// This is implemented internally by DRM.
///
/// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
pub trait FromRawPlaneState: AsRawPlaneState {
    /// Get an immutable reference to this type from the given raw [`struct drm_plane_state`]
    /// pointer.
    ///
    /// # Safety
    ///
    /// - The caller guarantees `ptr` is contained within a valid instance of `Self`
    /// - The caller guarantees that `ptr` cannot not be modified for the lifetime of `'a`.
    ///
    /// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
    unsafe fn from_raw<'a>(ptr: *const bindings::drm_plane_state) -> &'a Self;

    /// Get a mutable reference to this type from the given raw [`struct drm_plane_state`] pointer.
    ///
    /// # Safety
    ///
    /// - The caller guarantees that `ptr` is contained within a valid instance of `Self`
    /// - The caller guarantees that `ptr` cannot have any other references taken out for the
    ///   lifetime of `'a`.
    ///
    /// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
    unsafe fn from_raw_mut<'a>(ptr: *mut bindings::drm_plane_state) -> &'a mut Self;
}

/// Common methods available on any type which implements [`AsRawPlane`].
///
/// This is implemented internally by DRM, and provides many of the basic methods for working with
/// the atomic state of [`Plane`]s.
pub trait RawPlaneState: AsRawPlaneState {
    /// Return the plane that this plane state belongs to.
    fn plane(&self) -> &Self::Plane {
        // SAFETY: The index is initialized by the time we expose Plane objects to users, and is
        // invariant throughout the lifetime of the Plane
        unsafe { Self::Plane::from_raw(self.as_raw().plane) }
    }
}
impl<T: AsRawPlaneState + ?Sized> RawPlaneState for T {}

/// The main interface for a [`struct drm_plane_state`].
///
/// This type is the main interface for dealing with the atomic state of DRM planes. In addition, it
/// allows access to whatever private data is contained within an implementor's [`DriverPlaneState`]
/// type.
///
/// # Invariants
///
/// - The DRM C API and our interface guarantees that only the user has mutable access to `state`,
///   up until [`drm_atomic_helper_commit_hw_done`] is called. Therefore, `plane` follows rust's
///   data aliasing rules and does not need to be behind an [`Opaque`] type.
/// - `state` and `inner` initialized for as long as this object is exposed to users.
/// - The data layout of this structure begins with [`struct drm_plane_state`].
/// - The plane for this atomic state can always be assumed to be of type [`Plane<T::Plane>`].
///
/// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
/// [`drm_atomic_helper_commit_hw_done`]: srctree/include/drm/drm_atomic_helper.h
#[derive(Default)]
#[repr(C)]
pub struct PlaneState<T: DriverPlaneState> {
    state: bindings::drm_plane_state,
    inner: T,
}

/// The main trait for implementing the [`struct drm_plane_state`] API for a [`Plane`].
///
/// A driver may store driver-private data within the implementor's type, which will be available
/// when using a full typed [`PlaneState`] object.
///
/// # Invariants
///
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_plane`] pointers are contained within a [`Plane<Self::Plane>`].
/// - Any C FFI callbacks generated using this trait are guaranteed that passed-in
///   [`struct drm_plane_state`] pointers are contained within a [`PlaneState<Self>`].
///
/// [`struct drm_plane`]: srctree/include/drm_plane.h
/// [`struct drm_plane_state`]: srctree/include/drm_plane.h
pub trait DriverPlaneState: Clone + Default + Sized {
    /// The type for this driver's drm_plane implementation
    type Plane: DriverPlane;
}

impl<T: DriverPlaneState> Sealed for PlaneState<T> {}

impl<T: DriverPlaneState> AsRawPlaneState for PlaneState<T> {
    type Plane = Plane<T::Plane>;
}

impl<T: DriverPlaneState> private::AsRawPlaneState for PlaneState<T> {
    fn as_raw(&self) -> &bindings::drm_plane_state {
        &self.state
    }

    unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_plane_state {
        &mut self.state
    }
}

impl<T: DriverPlaneState> FromRawPlaneState for PlaneState<T> {
    unsafe fn from_raw<'a>(ptr: *const bindings::drm_plane_state) -> &'a Self {
        // Our data layout starts with `bindings::drm_plane_state`.
        let ptr: *const Self = ptr.cast();

        // SAFETY:
        // - Our safety contract requires that `ptr` be contained within `Self`.
        // - Our safety contract requires the caller ensure that it is safe for us to take an
        //   immutable reference.
        unsafe { &*ptr }
    }

    unsafe fn from_raw_mut<'a>(ptr: *mut bindings::drm_plane_state) -> &'a mut Self {
        // Our data layout starts with `bindings::drm_plane_state`.
        let ptr: *mut Self = ptr.cast();

        // SAFETY:
        // - Our safety contract requires that `ptr` be contained within `Self`.
        // - Our safety contract requires the caller ensure it is safe for us to take a mutable
        //   reference.
        unsafe { &mut *ptr }
    }
}

impl<T: DriverPlaneState> Deref for PlaneState<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: DriverPlaneState> DerefMut for PlaneState<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// SAFETY: Shares the safety guarantee of Plane<T>'s vtable impl
unsafe impl<T: DriverPlaneState> ModeObjectVtable for PlaneState<T> {
    type Vtable = bindings::drm_plane_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        self.plane().vtable()
    }
}

impl<T: DriverPlaneState> PlaneState<T> {
    super::impl_from_opaque_mode_obj! {
        fn <'a, D, P>(&'a OpaquePlaneState<D>) -> &'a Self
        where
            T: DriverPlaneState<Plane = P>;
        use
            P as DriverPlane,
            D as KmsDriver<Plane = ...>
    }
}

/// A [`struct drm_plane_state`] without a known [`DriverPlaneState`] implementation.
///
/// This is mainly for situations where our bindings can't infer the [`DriverPlaneState`]
/// implementation for a [`struct drm_plane_state`] automatically. It is identical to [`Plane`],
/// except that it does not provide access to the driver's private data.
///
/// # Invariants
///
/// - The DRM C API and our interface guarantees that only the user has mutable access to `state`,
///   up until [`drm_atomic_helper_commit_hw_done`] is called. Therefore, `plane` follows rust's
///   data aliasing rules and does not need to be behind an [`Opaque`] type.
/// - `state` is initialized for as long as this object is exposed to users.
/// - The data layout of this structure is identical to [`struct drm_plane_state`].
///
/// [`struct drm_plane_state`]: srctree/include/drm/drm_plane.h
/// [`drm_atomic_helper_commit_hw_done`]: srctree/include/drm/drm_atomic_helper.h
#[repr(transparent)]
pub struct OpaquePlaneState<T: KmsDriver> {
    state: bindings::drm_plane_state,
    _p: PhantomData<T>,
}

impl<T: KmsDriver> AsRawPlaneState for OpaquePlaneState<T> {
    type Plane = OpaquePlane<T>;
}

impl<T: KmsDriver> private::AsRawPlaneState for OpaquePlaneState<T> {
    fn as_raw(&self) -> &bindings::drm_plane_state {
        &self.state
    }

    unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_plane_state {
        &mut self.state
    }
}

impl<T: KmsDriver> FromRawPlaneState for OpaquePlaneState<T> {
    unsafe fn from_raw<'a>(ptr: *const bindings::drm_plane_state) -> &'a Self {
        // SAFETY: Our data layout is identical to `ptr`
        unsafe { &*ptr.cast() }
    }

    unsafe fn from_raw_mut<'a>(ptr: *mut bindings::drm_plane_state) -> &'a mut Self {
        // SAFETY: Our data layout is identical to `ptr`
        unsafe { &mut *ptr.cast() }
    }
}

// SAFETY: Shares the safety guarantee of OpaquePlane<T>'s vtable impl
unsafe impl<T: KmsDriver> ModeObjectVtable for OpaquePlaneState<T> {
    type Vtable = bindings::drm_plane_funcs;

    fn vtable(&self) -> *const Self::Vtable {
        self.plane().vtable()
    }
}

/// An interface for mutating a [`Plane`]s atomic state.
///
/// This type is typically returned by an [`AtomicStateMutator`] within contexts where it is
/// possible to safely mutate a plane's state. In order to uphold rust's data-aliasing rules, only
/// [`PlaneStateMutator`] may exist at a time.
pub struct PlaneStateMutator<'a, T: FromRawPlaneState> {
    state: &'a mut T,
    mask: &'a Cell<u32>,
}

impl<'a, T: FromRawPlaneState> PlaneStateMutator<'a, T> {
    pub(super) fn new<D: KmsDriver>(
        mutator: &'a AtomicStateMutator<D>,
        state: NonNull<bindings::drm_plane_state>,
    ) -> Option<Self> {
        // SAFETY: `plane` is invariant throughout the lifetime of the atomic state, is
        // initialized by this point, and we're guaranteed it is of type `AsRawPlane` by type
        // invariance
        let plane = unsafe { T::Plane::from_raw((*state.as_ptr()).plane) };
        let plane_mask = plane.mask();
        let borrowed_mask = mutator.borrowed_planes.get();

        if borrowed_mask & plane_mask == 0 {
            mutator.borrowed_planes.set(borrowed_mask | plane_mask);
            Some(Self {
                mask: &mutator.borrowed_planes,
                // SAFETY: We're guaranteed `state` is of `FromRawPlaneState` by type invariance,
                // and we just confirmed by checking `borrowed_planes` that no other mutable borrows
                // have been taken out for `state`
                state: unsafe { T::from_raw_mut(state.as_ptr()) },
            })
        } else {
            None
        }
    }
}

impl<'a, T: FromRawPlaneState> Drop for PlaneStateMutator<'a, T> {
    fn drop(&mut self) {
        let mask = self.state.plane().mask();
        self.mask.set(self.mask.get() & !mask);
    }
}

impl<'a, T: FromRawPlaneState> AsRawPlaneState for PlaneStateMutator<'a, T> {
    type Plane = T::Plane;
}

impl<'a, T: FromRawPlaneState> private::AsRawPlaneState for PlaneStateMutator<'a, T> {
    fn as_raw(&self) -> &bindings::drm_plane_state {
        self.state.as_raw()
    }

    unsafe fn as_raw_mut(&mut self) -> &mut bindings::drm_plane_state {
        // SAFETY: This function is bound by the same safety contract as `self.inner.as_raw_mut()`
        unsafe { self.state.as_raw_mut() }
    }
}

impl<'a, T: FromRawPlaneState> Sealed for PlaneStateMutator<'a, T> {}

impl<'a, T: DriverPlaneState> Deref for PlaneStateMutator<'a, PlaneState<T>> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.state.inner
    }
}

impl<'a, T: DriverPlaneState> DerefMut for PlaneStateMutator<'a, PlaneState<T>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state.inner
    }
}

// SAFETY: Shares the safety guarantees of `T`'s ModeObjectVtable impl
unsafe impl<'a, T: FromRawPlaneState> ModeObjectVtable for PlaneStateMutator<'a, T>
where
    T: FromRawPlaneState + ModeObjectVtable,
{
    type Vtable = T::Vtable;

    fn vtable(&self) -> *const Self::Vtable {
        self.state.vtable()
    }
}

impl<'a, T: DriverPlaneState> PlaneStateMutator<'a, PlaneState<T>> {
    super::impl_from_opaque_mode_obj! {
        fn <D, P>(PlaneStateMutator<'a, OpaquePlaneState<D>>) -> Self
        where
            T: DriverPlaneState<Plane = P>;
        use
            P as DriverPlane,
            D as KmsDriver<Plane = ...>
    }
}

unsafe extern "C" fn plane_destroy_callback<T: DriverPlane>(plane: *mut bindings::drm_plane) {
    // SAFETY: DRM guarantees that `plane` points to a valid initialized `drm_plane`.
    unsafe { bindings::drm_plane_cleanup(plane) };

    // SAFETY:
    // - DRM guarantees we are now the only one with access to this [`drm_plane`].
    // - This cast is safe via `DriverPlane`s type invariants.
    drop(unsafe { KBox::from_raw(plane as *mut Plane<T>) });
}

unsafe extern "C" fn atomic_duplicate_state_callback<T: DriverPlaneState>(
    plane: *mut bindings::drm_plane,
) -> *mut bindings::drm_plane_state {
    // SAFETY: DRM guarantees that `plane` points to a valid initialized `drm_plane`.
    let state = unsafe { (*plane).state };
    if state.is_null() {
        return null_mut();
    }

    // SAFETY: This cast is safe via `DriverPlaneState`s type invariants.
    let state = unsafe { PlaneState::<T>::from_raw(state) };

    let new: Result<KBox<_>> = KBox::try_init(
        try_init!(PlaneState {
            inner: state.inner.clone(),
            state: bindings::drm_plane_state {
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
        unsafe { bindings::__drm_atomic_helper_plane_duplicate_state(plane, new.as_raw_mut()) };

        KBox::into_raw(new).cast()
    } else {
        null_mut()
    }
}

unsafe extern "C" fn atomic_destroy_state_callback<T: DriverPlaneState>(
    _plane: *mut bindings::drm_plane,
    state: *mut bindings::drm_plane_state,
) {
    // SAFETY: DRM guarantees that `state` points to a valid instance of `drm_plane_state`
    unsafe { bindings::__drm_atomic_helper_plane_destroy_state(state) };

    // SAFETY:
    // * DRM guarantees we are the only one with access to this `drm_plane_state`
    // * This cast is safe via our type invariants.
    drop(unsafe { KBox::from_raw(state.cast::<PlaneState<T>>()) });
}

unsafe extern "C" fn plane_reset_callback<T: DriverPlane>(plane: *mut bindings::drm_plane) {
    // SAFETY: DRM guarantees that `state` points to a valid instance of `drm_plane_state`
    let state = unsafe { (*plane).state };
    if !state.is_null() {
        // SAFETY:
        // - We're guaranteed `plane` is `Plane<T>` via type invariants
        // - We're guaranteed `state` is `PlaneState<T>` via type invariants.
        unsafe { atomic_destroy_state_callback::<T::State>(plane, state) }

        // SAFETY: No special requirements here, DRM expects this to be NULL
        unsafe {
            (*plane).state = null_mut();
        }
    }

    // Unfortunately, this is the best we can do at the moment as this FFI callback was mistakenly
    // presumed to be infallible :(
    let new =
        KBox::new(PlaneState::<T::State>::default(), GFP_KERNEL).expect("Blame the API, sorry!");

    // DRM takes ownership of the state from here, resets it, and then assigns it to the plane
    // SAFETY:
    // - DRM guarantees that `plane` points to a valid instance of `drm_plane`.
    // - The cast to `drm_plane_state` is safe via `PlaneState`s type invariants.
    unsafe { bindings::__drm_atomic_helper_plane_reset(plane, KBox::into_raw(new).cast()) };
}
