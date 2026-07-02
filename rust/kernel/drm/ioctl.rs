// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM IOCTL definitions.
//!
//! C header: [`include/drm/drm_ioctl.h`](srctree/include/drm/drm_ioctl.h)

use crate::{drm, error::Result, ioctl, uaccess::UserPtr};

const BASE: u32 = uapi::DRM_IOCTL_BASE as u32;

/// Construct a DRM ioctl number with no argument.
#[allow(non_snake_case)]
#[inline(always)]
pub const fn IO(nr: u32) -> u32 {
    ioctl::_IO(BASE, nr)
}

/// Construct a DRM ioctl number with a read-only argument.
#[allow(non_snake_case)]
#[inline(always)]
pub const fn IOR<T>(nr: u32) -> u32 {
    ioctl::_IOR::<T>(BASE, nr)
}

/// Construct a DRM ioctl number with a write-only argument.
#[allow(non_snake_case)]
#[inline(always)]
pub const fn IOW<T>(nr: u32) -> u32 {
    ioctl::_IOW::<T>(BASE, nr)
}

/// Construct a DRM ioctl number with a read-write argument.
#[allow(non_snake_case)]
#[inline(always)]
pub const fn IOWR<T>(nr: u32) -> u32 {
    ioctl::_IOWR::<T>(BASE, nr)
}

/// Descriptor type for DRM ioctls. Use the `declare_drm_ioctls!{}` macro to construct them.
pub type DrmIoctlDescriptor = bindings::drm_ioctl_desc;

/// Descriptor for a 32-bit translation of a driver-private DRM ioctl.
pub struct DrmCompatIoctlDescriptor {
    cmd: u32,
    func: unsafe extern "C" fn(*mut bindings::file, core::ffi::c_uint, usize) -> isize,
}

impl DrmCompatIoctlDescriptor {
    #[doc(hidden)]
    pub const fn new(
        cmd: u32,
        func: unsafe extern "C" fn(*mut bindings::file, core::ffi::c_uint, usize) -> isize,
    ) -> Self {
        Self { cmd, func }
    }
}

/// Conversion between an established 32-bit ioctl argument and its native UAPI type.
///
/// Implementations use [`UserPtr`] and the safe uaccess APIs to copy the compat structure. The
/// DRM layer invokes the native ioctl handler only after conversion and performs normal DRM
/// permission and unplug checks through `drm_ioctl_kernel()`.
pub trait CompatIoctl: Sized {
    /// Native argument type consumed by the regular DRM ioctl handler.
    type Native;

    /// Read one 32-bit argument from userspace.
    fn read_from_user(ptr: UserPtr) -> Result<Self>;

    /// Convert the 32-bit argument to its native representation.
    fn into_native(&self) -> Self::Native;

    /// Copy output fields back to the 32-bit userspace argument.
    fn write_back(&mut self, _native: &Self::Native, _ptr: UserPtr) -> Result {
        Ok(())
    }
}

/// This is for ioctl which are used for rendering, and require that the file descriptor is either
/// for a render node, or if it’s a legacy/primary node, then it must be authenticated.
pub const AUTH: u32 = bindings::drm_ioctl_flags_DRM_AUTH;

/// This must be set for any ioctl which can change the modeset or display state. Userspace must
/// call the ioctl through a primary node, while it is the active master.
///
/// Note that read-only modeset ioctl can also be called by unauthenticated clients, or when a
/// master is not the currently active one.
pub const MASTER: u32 = bindings::drm_ioctl_flags_DRM_MASTER;

/// Anything that could potentially wreak a master file descriptor needs to have this flag set.
///
/// Current that’s only for the SETMASTER and DROPMASTER ioctl, which e.g. logind can call to
/// force a non-behaving master (display compositor) into compliance.
///
/// This is equivalent to callers with the SYSADMIN capability.
pub const ROOT_ONLY: u32 = bindings::drm_ioctl_flags_DRM_ROOT_ONLY;

/// This is used for all ioctl needed for rendering only, for drivers which support render nodes.
/// This should be all new render drivers, and hence it should be always set for any ioctl with
/// `AUTH` set. Note though that read-only query ioctl might have this set, but have not set
/// DRM_AUTH because they do not require authentication.
pub const RENDER_ALLOW: u32 = bindings::drm_ioctl_flags_DRM_RENDER_ALLOW;

/// Internal structures used by the `declare_drm_ioctls!{}` macro. Do not use directly.
#[doc(hidden)]
pub mod internal {
    pub use bindings::drm_device;
    pub use bindings::drm_file;
    pub use bindings::drm_ioctl_desc;
    pub use bindings::file;

    /// Cast an [`Ioctl`] DRM device pointer to [`Registered`], preserving the driver type
    /// parameter `T`.
    ///
    /// Used by [`declare_drm_ioctls!`] to anchor type inference.
    #[doc(hidden)]
    #[inline]
    pub const fn __dev_ctx_cast<T: crate::drm::Driver>(
        ptr: *const crate::drm::Device<T, crate::drm::Ioctl>,
    ) -> *const crate::drm::Device<T, crate::drm::Registered> {
        ptr.cast()
    }
}

/// Dispatch a compat ioctl declared by a Rust DRM driver.
///
/// Driver-private translations are matched by their complete compat command number. All other
/// commands retain DRM core's standard compat handling.
#[doc(hidden)]
#[cfg(CONFIG_COMPAT)]
pub unsafe extern "C" fn compat_ioctl<T: drm::Driver>(
    file: *mut bindings::file,
    cmd: core::ffi::c_uint,
    arg: usize,
) -> isize {
    for desc in T::COMPAT_IOCTLS {
        if desc.cmd == cmd {
            // SAFETY: This is the callback stored by the driver's compat declaration for this
            // exact command. The VFS guarantees that `file` is live for the ioctl call.
            return unsafe { (desc.func)(file, cmd, arg) };
        }
    }

    // SAFETY: Forwarding the unchanged VFS callback arguments to DRM core.
    unsafe { bindings::drm_compat_ioctl(file, cmd, arg) }
}

/// Declare the DRM ioctls for a driver.
///
/// Each entry in the list should have the form:
///
/// `(ioctl_number, argument_type, flags, user_callback),`
///
/// `argument_type` is the type name within the `bindings` crate.
/// `user_callback` should have the following prototype:
///
/// ```ignore
/// fn foo(device: &kernel::drm::Device<Self, kernel::drm::Registered>,
///        reg_data: &Self::RegistrationData<'_>,
///        data: &mut uapi::argument_type,
///        file: &kernel::drm::File<Self::File>,
/// ) -> Result<u32>
/// ```
/// where `Self` is the drm::drv::Driver implementation these ioctls are being declared within.
///
/// # Examples
///
/// ```ignore
/// kernel::declare_drm_ioctls! {
///     (FOO_GET_PARAM, drm_foo_get_param, ioctl::RENDER_ALLOW, my_get_param_handler),
/// }
/// ```
///
#[macro_export]
macro_rules! declare_drm_ioctls {
    ( $(($cmd:ident, $struct:ident, $flags:expr, $func:expr)),* $(,)? ) => {
        const IOCTLS: &'static [$crate::drm::ioctl::DrmIoctlDescriptor] = {
            use $crate::uapi::*;
            const _:() = {
                let i: u32 = $crate::uapi::DRM_COMMAND_BASE;
                // Assert that all the IOCTLs are in the right order and there are no gaps,
                // and that the size of the specified type is correct.
                $(
                    let cmd: u32 = $crate::macros::concat_idents!(DRM_IOCTL_, $cmd);
                    ::core::assert!(i == $crate::ioctl::_IOC_NR(cmd));
                    ::core::assert!(core::mem::size_of::<$crate::uapi::$struct>() ==
                                    $crate::ioctl::_IOC_SIZE(cmd));
                    let i: u32 = i + 1;
                )*
            };

            let ioctls = &[$(
                $crate::drm::ioctl::internal::drm_ioctl_desc {
                    cmd: $crate::macros::concat_idents!(DRM_IOCTL_, $cmd) as u32,
                    func: {
                        #[allow(non_snake_case)]
                        unsafe extern "C" fn $cmd(
                                raw_dev: *mut $crate::drm::ioctl::internal::drm_device,
                                raw_data: *mut ::core::ffi::c_void,
                                raw_file: *mut $crate::drm::ioctl::internal::drm_file,
                        ) -> core::ffi::c_int {
                            // SAFETY:
                            // - The DRM core ensures the device lives while callbacks are being
                            //   called.
                            // - The DRM device must have been registered when we're called through
                            //   an IOCTL.
                            //
                            // INVARIANT: The `Ioctl` context requires that the device has been
                            // registered via `drm_dev_register()` at some point; the DRM core
                            // guarantees this for ioctl dispatch callbacks.
                            //
                            // FIXME: Currently there is nothing enforcing that the types of the
                            // dev/file match the current driver these ioctls are being declared
                            // for, and it's not clear how to enforce this within the type system.
                            let dev: &$crate::drm::device::Device<_, $crate::drm::Ioctl> =
                                $crate::drm::device::Device::from_raw(raw_dev);

                            // Type-inference anchor: the closure is never called but ties `dev`'s
                            // type to `$func`'s first parameter, which the compiler cannot infer
                            // through method resolution and associated-type projections alone.
                            #[allow(unreachable_code)]
                            let _ = || {
                                let __ptr = $crate::drm::ioctl::internal::__dev_ctx_cast(
                                    ::core::ptr::from_ref(dev),
                                );

                                $func(
                                    // SAFETY: This closure is never executed; the dereference
                                    // exists purely to unify the type parameter with `$func`.
                                    // The pointer is valid regardless.
                                    unsafe { &*__ptr },
                                    unreachable!(),
                                    unreachable!(),
                                    unreachable!(),
                                )
                            };

                            // Enforce that the handler accepts higher-ranked
                            // lifetimes, preventing it from requiring 'static
                            // references that could escape this scope.
                            let _: for<'a> fn(&'a _, &'a _, &'a mut _, &'a _) -> _ = $func;

                            let Some(guard) = dev.registration_guard() else {
                                return $crate::error::code::ENODEV.to_errno();
                            };

                            // SAFETY: The ioctl argument has size `_IOC_SIZE(cmd)`, which we
                            // asserted above matches the size of this type, and all bit patterns of
                            // UAPI structs must be valid.
                            // The `ioctl` argument is exclusively owned by the handler
                            // and guaranteed by the C implementation (`drm_ioctl()`) to remain
                            // valid for the entire lifetime of the reference taken here.
                            // There is no concurrent access or aliasing; no other references
                            // to this object exist during this call.
                            let data = unsafe { &mut *(raw_data.cast::<$crate::uapi::$struct>()) };
                            // SAFETY: This is just the DRM file structure
                            let file = unsafe { $crate::drm::File::from_raw(raw_file) };

                            match guard.registration_data_with(|reg_data| {
                                $func(&*guard, reg_data, data, file)
                            }) {
                                Err(e) => e.to_errno(),
                                Ok(i) => i.try_into()
                                            .unwrap_or($crate::error::code::ERANGE.to_errno()),
                            }
                        }
                        Some($cmd)
                    },
                    flags: $flags,
                    name: $crate::str::as_char_ptr_in_const_context(
                        $crate::c_str!(::core::stringify!($cmd)),
                    ),
                }
            ),*];
            ioctls
        };
    };
}

/// Declare translations for driver-private ioctls whose 32-bit layout differs from native.
///
/// Each entry has the form:
///
/// `(ioctl_name, native_argument_type, compat_argument_type, direction, native_handler),`
///
/// `compat_argument_type` must implement [`CompatIoctl`] with `Native` equal to the generated UAPI
/// `native_argument_type`. `direction` is one of `IOR`, `IOW`, or `IOWR`.
#[macro_export]
macro_rules! declare_drm_compat_ioctls {
    (
        $driver:ty;
        $(($cmd:ident, $native:ident, $compat:ty, $dir:ident, $func:expr)),* $(,)?
    ) => {
        const COMPAT_IOCTLS: &'static [$crate::drm::ioctl::DrmCompatIoctlDescriptor] = &[$(
            {
                use $crate::uapi::*;

                const COMPAT_CMD: u32 = $crate::drm::ioctl::$dir::<$compat>(
                    $crate::uapi::DRM_COMMAND_BASE
                        + ($crate::ioctl::_IOC_NR(
                            $crate::macros::concat_idents!(DRM_IOCTL_, $cmd)
                        ) - $crate::uapi::DRM_COMMAND_BASE)
                );

                const _: () = {
                    ::core::assert!(
                        $crate::ioctl::_IOC_NR(COMPAT_CMD)
                            == $crate::ioctl::_IOC_NR(
                                $crate::macros::concat_idents!(DRM_IOCTL_, $cmd)
                            )
                    );
                    ::core::assert!(
                        ::core::mem::size_of::<$compat>()
                            == $crate::ioctl::_IOC_SIZE(COMPAT_CMD)
                    );
                };

                #[allow(non_snake_case)]
                unsafe extern "C" fn $cmd(
                    raw_file: *mut $crate::drm::ioctl::internal::file,
                    _raw_cmd: core::ffi::c_uint,
                    raw_arg: usize,
                ) -> isize {
                    let user = $crate::uaccess::UserPtr::from_addr(raw_arg);
                    let mut compat =
                        match <$compat as $crate::drm::ioctl::CompatIoctl>::read_from_user(user) {
                            Ok(value) => value,
                            Err(err) => return err.to_errno() as isize,
                        };
                    let mut native:
                        <$compat as $crate::drm::ioctl::CompatIoctl>::Native =
                        <$compat as $crate::drm::ioctl::CompatIoctl>::into_native(&compat);

                    #[allow(non_snake_case)]
                    unsafe extern "C" fn native_handler(
                        raw_dev: *mut $crate::drm::ioctl::internal::drm_device,
                        raw_data: *mut ::core::ffi::c_void,
                        raw_drm_file: *mut $crate::drm::ioctl::internal::drm_file,
                    ) -> core::ffi::c_int {
                        // SAFETY: DRM core calls this with the registered device selected from
                        // `raw_file` and the native argument allocated below.
                        let dev: &$crate::drm::device::Device<
                            $driver,
                            $crate::drm::Ioctl,
                        > = unsafe { $crate::drm::device::Device::from_raw(raw_dev) };
                        let Some(guard) = dev.registration_guard() else {
                            return $crate::error::code::ENODEV.to_errno();
                        };
                        // SAFETY: The compat trampoline passes a live, exclusively borrowed
                        // `$native` value to drm_ioctl_kernel().
                        let data =
                            unsafe { &mut *(raw_data.cast::<$crate::uapi::$native>()) };
                        // SAFETY: DRM core supplies the live drm_file associated with `raw_file`.
                        let drm_file: &$crate::drm::File<
                            <$driver as $crate::drm::Driver>::File
                        > = unsafe { $crate::drm::File::from_raw(raw_drm_file) };

                        match guard.registration_data_with(|reg_data| {
                            $func(&*guard, reg_data, data, drm_file)
                        }) {
                            Err(err) => err.to_errno(),
                            Ok(value) => value.try_into().unwrap_or(
                                $crate::error::code::ERANGE.to_errno()
                            ),
                        }
                    }

                    // Ensure the conversion's native type is the UAPI type expected by the
                    // generated handler before passing it through a void pointer.
                    let native_ref: &mut $crate::uapi::$native = &mut native;
                    let index = ($crate::ioctl::_IOC_NR(
                        $crate::macros::concat_idents!(DRM_IOCTL_, $cmd)
                    ) - $crate::uapi::DRM_COMMAND_BASE) as usize;
                    const _: () = {
                        let index = ($crate::ioctl::_IOC_NR(
                            $crate::macros::concat_idents!(DRM_IOCTL_, $cmd)
                        ) - $crate::uapi::DRM_COMMAND_BASE) as usize;
                        ::core::assert!(
                            index < <$driver as $crate::drm::Driver>::IOCTLS.len()
                        );
                        ::core::assert!(
                            <$driver as $crate::drm::Driver>::IOCTLS[index].cmd
                                == $crate::macros::concat_idents!(DRM_IOCTL_, $cmd)
                        );
                    };
                    let flags = <$driver as $crate::drm::Driver>::IOCTLS[index].flags;

                    // SAFETY: `raw_file` is live for the VFS callback; `native_ref` has the exact
                    // type consumed by `native_handler`; DRM core applies the regular permission
                    // and unplug checks before invoking it.
                    let ret = unsafe {
                        $crate::bindings::drm_ioctl_kernel(
                            raw_file,
                            Some(native_handler),
                            ::core::ptr::from_mut(native_ref).cast(),
                            flags,
                        )
                    };
                    if ret >= 0 {
                        if let Err(err) =
                            <$compat as $crate::drm::ioctl::CompatIoctl>::write_back(
                                &mut compat,
                                native_ref,
                                user,
                            )
                        {
                            return err.to_errno() as isize;
                        }
                    }
                    ret
                }

                $crate::drm::ioctl::DrmCompatIoctlDescriptor::new(COMPAT_CMD, $cmd)
            }
        ),*];
    };
}
