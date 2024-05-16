// SPDX-License-Identifier: GPL-2.0 OR MIT

//! KMS driver abstractions for rust.

use crate::{
    device,
    drm::{device::Device, driver::Driver},
    error::to_result,
    prelude::*,
};
use bindings;
use core::{marker::PhantomData, ops::Deref};

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
