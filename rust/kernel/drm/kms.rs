// SPDX-License-Identifier: GPL-2.0 OR MIT

//! KMS driver abstractions for rust.

pub mod connector;
pub mod crtc;
pub mod encoder;
pub mod plane;

use crate::{
    device,
    drm::{device::Device, driver::Driver, private::Sealed},
    error::to_result,
    prelude::*,
    types::*,
};
use bindings;
use core::{marker::PhantomData, ops::Deref, ptr::NonNull};

/// The C vtable for a [`Device`].
///
/// This is created internally by DRM.
pub struct ModeConfigOps {
    pub(crate) kms_vtable: bindings::drm_mode_config_funcs,
    pub(crate) kms_helper_vtable: bindings::drm_mode_config_helper_funcs,
}

/// A trait representing a type that can be used for setting up KMS, or a stub.
///
/// For drivers which don't have KMS support, the methods provided by this trait may be stubs. It is
/// implemented internally by DRM.
pub trait KmsImpl: private::KmsImpl {}

pub(crate) mod private {
    use super::*;

    /// Private callback implemented internally by DRM for setting up KMS on a device, or stubbing
    /// the KMS setup for devices which don't have KMS support.
    #[allow(unreachable_pub)]
    pub trait KmsImpl {
        /// The parent driver for this KMS implementation
        type Driver: Driver;

        /// The optional KMS callback operations for this driver.
        const MODE_CONFIG_OPS: Option<ModeConfigOps>;

        /// The callback for setting up KMS on a device
        ///
        /// # Safety
        ///
        /// `drm` must be unregistered.
        unsafe fn setup_kms(_drm: &Device<Self::Driver>) -> Result<ModeConfigInfo> {
            build_error::build_error("This should never be reachable")
        }
    }
}

/// A [`Device`] with KMS initialized that has not been registered with userspace.
///
/// This type is identical to [`Device`], except that it is able to create new static KMS resources.
/// It represents a KMS device that is not yet visible to userspace, and also contains miscellaneous
/// state required during the initialization process of a [`Device`].
pub struct UnregisteredKmsDevice<'a, T: Driver> {
    drm: &'a Device<T>,
}

impl<'a, T: Driver> Deref for UnregisteredKmsDevice<'a, T> {
    type Target = Device<T>;

    fn deref(&self) -> &Self::Target {
        self.drm
    }
}

impl<'a, T: Driver> UnregisteredKmsDevice<'a, T> {
    /// Construct a new [`UnregisteredKmsDevice`].
    ///
    /// # Safety
    ///
    /// The caller promises that `drm` is an unregistered [`Device`].
    pub(crate) unsafe fn new(drm: &'a Device<T>) -> Self {
        Self { drm }
    }
}

/// A trait which must be implemented by drivers that wish to support KMS
///
/// It should be implemented for the same type that implements [`Driver`]. Drivers which don't
/// support KMS should use [`PhantomData<Self>`].
///
/// [`PhantomData<Self>`]: PhantomData
#[vtable]
pub trait KmsDriver: Driver {
    /// The driver's [`DriverConnector`] implementation.
    ///
    /// TODO: This will be unneeded in the future once we support multiple [`DriverConnector`]
    /// implementations.
    ///
    /// [`DriverConnector`]: connector::DriverConnector
    type Connector: connector::DriverConnector;

    /// The driver's [`DriverPlane`] implementation.
    ///
    /// TODO: This will be unneeded in the future once we support multiple [`DriverPlane`]
    /// implementations.
    ///
    type Plane: plane::DriverPlane;

    /// The driver's [`DriverCrtc`] implementation.
    ///
    /// TODO: This will be unneeded in the future once we support multiple [`DriverCrtc`]
    /// implementations.
    ///
    /// [`DriverCrtc`]: crtc::DriverCrtc
    type Crtc: crtc::DriverCrtc;

    /// The driver's [`DriverEncoder`] implementation.
    ///
    /// TODO: This will be unneeded in the future once we support multiple [`DriverEncoder`]
    /// implementations.
    ///
    /// [`DriverEncoder`]: encoder::DriverEncoder
    type Encoder: encoder::DriverEncoder;

    /// Return a [`ModeConfigInfo`] structure for this [`device::Device`].
    fn mode_config_info(
        dev: &device::Device,
        drm_data: &Self::Data,
    ) -> Result<ModeConfigInfo>;

    /// Create mode objects like [`crtc::Crtc`], [`plane::Plane`], etc. for this device
    fn create_objects(drm: &UnregisteredKmsDevice<'_, Self>) -> Result
    where
        Self: Sized;
}

impl<T: KmsDriver> private::KmsImpl for T {
    type Driver = Self;

    const MODE_CONFIG_OPS: Option<ModeConfigOps> = Some(ModeConfigOps {
        kms_vtable: bindings::drm_mode_config_funcs {
            atomic_check: Some(bindings::drm_atomic_helper_check),
            fb_create: Some(bindings::drm_gem_fb_create),
            mode_valid: None,
            atomic_commit: Some(bindings::drm_atomic_helper_commit),
            get_format_info: None,
            atomic_state_free: None,
            atomic_state_alloc: None,
            atomic_state_clear: None,
        },

        kms_helper_vtable: bindings::drm_mode_config_helper_funcs {
            atomic_commit_setup: None,
            atomic_commit_tail: None,
        },
    });

    unsafe fn setup_kms(drm: &Device<Self::Driver>) -> Result<ModeConfigInfo> {
        let mode_config_info = T::mode_config_info(drm.as_ref().as_ref(), drm)?;

        // SAFETY: `MODE_CONFIG_OPS` is always Some() in this implementation
        let ops = unsafe { T::MODE_CONFIG_OPS.as_ref().unwrap_unchecked() };

        // SAFETY:
        // - This function can only be called before registration via our safety contract.
        // - Before registration, we are the only ones with access to this device.
        unsafe {
            (*drm.as_raw()).mode_config = bindings::drm_mode_config {
                funcs: &ops.kms_vtable,
                helper_private: &ops.kms_helper_vtable,
                min_width: mode_config_info.min_resolution.0,
                min_height: mode_config_info.min_resolution.1,
                max_width: mode_config_info.max_resolution.0,
                max_height: mode_config_info.max_resolution.1,
                cursor_width: mode_config_info.max_cursor.0,
                cursor_height: mode_config_info.max_cursor.1,
                preferred_depth: mode_config_info.preferred_depth,
                ..Default::default()
            };
        }

        // SAFETY: We just setup all of the required info this function needs in `drm_device`
        to_result(unsafe { bindings::drmm_mode_config_init(drm.as_raw()) })?;

        // SAFETY: `drm` is guaranteed to be unregistered via our safety contract.
        let drm = unsafe { UnregisteredKmsDevice::new(drm) };

        T::create_objects(&drm)?;

        // TODO: Eventually add a hook to customize how state readback happens, for now just reset
        // SAFETY: Since all static modesetting objects were created in `T::create_objects()`, and
        // that is the only place they can be created, this fulfills the C API requirements.
        unsafe { bindings::drm_mode_config_reset(drm.as_raw()) };

        Ok(mode_config_info)
    }
}

impl<T: KmsDriver> KmsImpl for T {}

impl<T: Driver> private::KmsImpl for PhantomData<T> {
    type Driver = T;

    const MODE_CONFIG_OPS: Option<ModeConfigOps> = None;
}

impl<T: Driver> KmsImpl for PhantomData<T> {}

/// Various device-wide information for a [`Device`] that is provided during initialization.
#[derive(Copy, Clone)]
pub struct ModeConfigInfo {
    /// The minimum (w, h) resolution this driver can support
    pub min_resolution: (u32, u32),
    /// The maximum (w, h) resolution this driver can support
    pub max_resolution: (u32, u32),
    /// The maximum (w, h) cursor size this driver can support
    pub max_cursor: (u32, u32),
    /// The preferred depth for dumb ioctls
    pub preferred_depth: u32,
    /// An optional default fourcc format code to be preferred for clients.
    pub preferred_fourcc: Option<u32>,
}

/// A modesetting object in DRM.
///
/// This is any type of object where the underlying C object contains a [`struct drm_mode_object`].
/// This type requires [`Send`] + [`Sync`] as all modesetting objects in DRM are able to be sent
/// between threads.
///
/// This type is only implemented by the DRM crate itself.
///
/// # Safety
///
/// [`raw_mode_obj()`] must always return a valid pointer to an initialized
/// [`struct drm_mode_object`].
///
/// [`struct drm_mode_object`]: srctree/include/drm/drm_mode_object.h
/// [`raw_mode_obj()`]: ModeObject::raw_mode_obj()
pub unsafe trait ModeObject: Sealed + Send + Sync {
    /// The parent driver for this [`ModeObject`].
    type Driver: KmsDriver;

    /// Return the [`Device`] for this [`ModeObject`].
    fn drm_dev(&self) -> &Device<Self::Driver>;

    /// Return a pointer to the [`struct drm_mode_object`] for this [`ModeObject`].
    ///
    /// [`struct drm_mode_object`]: (srctree/include/drm/drm_mode_object.h)
    fn raw_mode_obj(&self) -> *mut bindings::drm_mode_object;
}

/// A trait for modesetting objects which don't come with their own reference-counting.
///
/// Some [`ModeObject`] types in DRM do not have a reference count. These types are considered
/// "static" and share the lifetime of their parent [`Device`]. To retrieve an owned reference to
/// such types, see [`KmsRef`].
///
/// # Safety
///
/// This trait must only be implemented for modesetting objects which do not have a refcount within
/// their [`struct drm_mode_object`], otherwise [`KmsRef`] can't guarantee the object will stay
/// alive.
///
/// [`struct drm_mode_object`]: (srctree/include/drm/drm_mode_object.h)
pub unsafe trait StaticModeObject: ModeObject {}

/// An owned reference to a [`StaticModeObject`].
///
/// Note that since [`StaticModeObject`] types share the lifetime of their parent [`Device`], the
/// parent [`Device`] will stay alive as long as this type exists. Thus, users should be aware that
/// storing a [`KmsRef`] within a [`ModeObject`] is a circular reference.
///
/// # Invariants
///
/// `self.0` points to a valid instance of `T` throughout the lifetime of this type.
pub struct KmsRef<T: StaticModeObject>(NonNull<T>);

// SAFETY: Owned references to DRM device are thread-safe.
unsafe impl<T: StaticModeObject> Send for KmsRef<T> {}
// SAFETY: Owned references to DRM device are thread-safe.
unsafe impl<T: StaticModeObject> Sync for KmsRef<T> {}

impl<T: StaticModeObject> From<&T> for KmsRef<T> {
    fn from(value: &T) -> Self {
        // INVARIANT: Because the lifetime of the StaticModeObject is the same as the lifetime of
        // its parent device, we can ensure that `value` remains alive by incrementing the device's
        // reference count. The device will only disappear once we drop this reference in `Drop`.
        value.drm_dev().inc_ref();

        Self(value.into())
    }
}

impl<T: StaticModeObject> Drop for KmsRef<T> {
    fn drop(&mut self) {
        // SAFETY: We're reclaiming the reference we leaked in From<&T>
        drop(unsafe { ARef::from_raw(self.drm_dev().into()) })
    }
}

impl<T: StaticModeObject> Deref for KmsRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: We're guaranteed object will point to a valid object as long as we hold dev
        unsafe { self.0.as_ref() }
    }
}

impl<T: StaticModeObject> Clone for KmsRef<T> {
    fn clone(&self) -> Self {
        // INVARIANT: Because the lifetime of the StaticModeObject is the same as the lifetime of
        // its parent device, we can ensure that `value` remains alive by incrementing the device's
        // reference count. The device will only disappear once we drop this reference in `Drop`.
        self.drm_dev().inc_ref();

        Self(self.0)
    }
}

macro_rules! impl_aref_for_mode_object {
    (impl $( < $( $param:ident: $bound:ident ),+ > )? for $type:ty) => {
        // SAFETY: drm_mode_object_get()/put() ensure the type is ref-counted according to the
        // safety contract
        unsafe impl $( < $( $param: $bound ),+ > )? kernel::types::AlwaysRefCounted for $type {
            #[inline]
            fn inc_ref(&self) {
                // SAFETY: We're guaranteed by the safety contract of `ModeObject` that
                // `raw_mode_obj()` always returns a pointer to an initialized `drm_mode_object`.
                unsafe { kernel::bindings::drm_mode_object_get(self.raw_mode_obj()) }
            }

            #[inline]
            unsafe fn dec_ref(obj: core::ptr::NonNull<Self>) {
                // SAFETY: We're guaranteed by the safety contract of `ModeObject` that
                // `raw_mode_obj()` always returns a pointer to an initialized `drm_mode_object`.
                unsafe { kernel::bindings::drm_mode_object_put(obj.as_ref().raw_mode_obj()) }
            }
        }
    };
}

pub(super) use impl_aref_for_mode_object;
