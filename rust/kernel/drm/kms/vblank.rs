// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM KMS vblank support.
//!
//! C header: [`include/drm/drm_vblank.h`](srcfree/include/drm/drm_vblank.h)

use super::{crtc::*, modes::*, ModeObject};
use bindings;
use core::{
    marker::*,
    mem::{self, ManuallyDrop},
    ops::{Deref, Drop},
    ptr::null_mut,
};
use kernel::{
    drm::device::Device,
    error::{from_result, to_result},
    interrupt::LocalInterruptDisabled,
    prelude::*,
    time::Delta,
    types::Opaque,
};

/// The main trait for a driver to implement hardware vblank support for a [`Crtc`].
///
/// # Invariants
///
/// C FFI callbacks generated using this trait can safely assume that input pointers to
/// [`struct drm_crtc`] are always contained within a [`Crtc<Self::Crtc>`].
///
/// [`struct drm_crtc`]: srctree/include/drm/drm_crtc.h
pub trait VblankSupport: Sized {
    /// The parent [`DriverCrtc`].
    type Crtc: VblankDriverCrtc<VblankImpl = Self>;

    /// Enable vblank interrupts for this [`DriverCrtc`].
    fn enable_vblank(
        crtc: &Crtc<Self::Crtc>,
        vblank_guard: &VblankGuard<'_, Self::Crtc>,
        irq: &LocalInterruptDisabled,
    ) -> Result;

    /// Disable vblank interrupts for this [`DriverCrtc`].
    fn disable_vblank(
        crtc: &Crtc<Self::Crtc>,
        vblank_guard: &VblankGuard<'_, Self::Crtc>,
        irq: &LocalInterruptDisabled,
    );

    /// Retrieve the current vblank timestamp for this [`Crtc`]
    ///
    /// If this function is being called from the driver's vblank interrupt handler,
    /// `handling_vblank_irq` will be `true`.
    fn get_vblank_timestamp(
        crtc: &Crtc<Self::Crtc>,
        in_vblank_irq: bool,
    ) -> Option<VblankTimestamp>;
}

/// Trait used for CRTC vblank (or lack there-of) implementations. Implemented internally.
///
/// Drivers interested in implementing vblank support should refer to [`VblankSupport`], drivers
/// that don't have vblank support can use [`PhantomData`].
pub trait VblankImpl {
    /// The parent [`DriverCrtc`].
    type Crtc: DriverCrtc<VblankImpl = Self>;

    /// The generated [`VblankOps`].
    const VBLANK_OPS: VblankOps;
}

/// C FFI callbacks for vblank management.
///
/// Created internally by DRM.
#[derive(Default)]
pub struct VblankOps {
    pub(crate) enable_vblank: Option<unsafe extern "C" fn(crtc: *mut bindings::drm_crtc) -> i32>,
    pub(crate) disable_vblank: Option<unsafe extern "C" fn(crtc: *mut bindings::drm_crtc)>,
    pub(crate) get_vblank_timestamp: Option<
        unsafe extern "C" fn(
            crtc: *mut bindings::drm_crtc,
            max_error: *mut i32,
            vblank_time: *mut bindings::ktime_t,
            in_vblank_irq: bool,
        ) -> bool,
    >,
}

impl<T: VblankSupport> VblankImpl for T {
    type Crtc = T::Crtc;

    const VBLANK_OPS: VblankOps = VblankOps {
        enable_vblank: Some(enable_vblank_callback::<T>),
        disable_vblank: Some(disable_vblank_callback::<T>),
        get_vblank_timestamp: Some(get_vblank_timestamp_callback::<T>),
    };
}

impl<T> VblankImpl for PhantomData<T>
where
    T: DriverCrtc<VblankImpl = PhantomData<T>>,
{
    type Crtc = T;

    const VBLANK_OPS: VblankOps = VblankOps {
        enable_vblank: None,
        disable_vblank: None,
        get_vblank_timestamp: None,
    };
}

unsafe extern "C" fn enable_vblank_callback<T: VblankSupport>(
    crtc: *mut bindings::drm_crtc,
) -> i32 {
    // SAFETY: We're guaranteed that `crtc` is of type `Crtc<T::Crtc>` by type invariants.
    let crtc = unsafe { Crtc::<T::Crtc>::from_raw(crtc) };

    // SAFETY: This callback happens with IRQs disabled
    let irq = unsafe { LocalInterruptDisabled::assume_disabled() };

    // SAFETY: This callback happens with `vbl_lock` already held
    // We don't want to drop `vbl_lock` when this callback completes since DRM will do this for us,
    // so wrap the `VblankGuard` in a `ManuallyDrop`
    let vblank_guard = ManuallyDrop::new(unsafe { VblankGuard::new(crtc, irq) });

    from_result(|| T::enable_vblank(crtc, &vblank_guard, irq).map(|_| 0))
}

unsafe extern "C" fn disable_vblank_callback<T: VblankSupport>(crtc: *mut bindings::drm_crtc) {
    // SAFETY: We're guaranteed that `crtc` is of type `Crtc<T::Crtc>` by type invariants.
    let crtc = unsafe { Crtc::<T::Crtc>::from_raw(crtc) };

    // SAFETY: This callback happens with IRQs disabled
    let irq = unsafe { LocalInterruptDisabled::assume_disabled() };

    // SAFETY: This call happens with `vbl_lock` already held
    // We don't want to drop `vbl_lock` when this callback completes since DRM will do this for us,
    // so wrap the `VblankGuard` in a `ManuallyDrop`
    let vblank_guard = ManuallyDrop::new(unsafe { VblankGuard::new(crtc, irq) });

    T::disable_vblank(crtc, &vblank_guard, irq);
}

unsafe extern "C" fn get_vblank_timestamp_callback<T: VblankSupport>(
    crtc: *mut bindings::drm_crtc,
    max_error: *mut i32,
    vblank_time: *mut bindings::ktime_t,
    in_vblank_irq: bool,
) -> bool {
    // SAFETY: We're guaranteed `crtc` is of type `Crtc<T::Crtc>` by type invariance
    let crtc = unsafe { Crtc::<T::Crtc>::from_raw(crtc) };

    if let Some(timestamp) = T::get_vblank_timestamp(crtc, in_vblank_irq) {
        // SAFETY: Both of these pointers are guaranteed by the C API to be valid
        unsafe {
            (*max_error) = timestamp.max_error;
            (*vblank_time) = timestamp.time.as_nanos();
        };

        true
    } else {
        false
    }
}

/// A vblank timestamp.
///
/// This type is used by [`VblankSupport::get_vblank_timestamp`] for the implementor to return the
/// current vblank timestamp for the hardware.
#[derive(Copy, Clone)]
pub struct VblankTimestamp {
    /// The actual vblank timestamp in nanoseconds, accuracy to within [`Self::max_error`]
    /// nanoseconds.
    pub time: Delta,

    /// Maximum allowable timestamp error in nanoseconds
    pub max_error: i32,
}

/// A trait for [`DriverCrtc`] implementations with hardware vblank support.
///
/// This trait is implemented internally by DRM for any [`DriverCrtc`] implementation that
/// implements [`VblankSupport`]. It is used to expose hardware-vblank driver exclusive methods and
/// data to users.
pub trait VblankDriverCrtc: DriverCrtc {}

impl<T, V> VblankDriverCrtc for T
where
    T: DriverCrtc<VblankImpl = V>,
    V: VblankSupport<Crtc = T>,
{
}

impl<T: VblankDriverCrtc> Crtc<T> {
    /// Retrieve a reference to the [`VblankCrtc`] for this [`Crtc`].
    pub(crate) fn vblank_crtc(&self) -> &VblankCrtc<T> {
        // SAFETY:
        // - The data layouts of these types are equivalent via `VblankCrtc`s type invariants
        // - We don't expose any way of calling `vblank_crtc()` before `drm_vblank_init()` has been
        //   called.
        unsafe { VblankCrtc::from_raw(self.get_vblank_ptr()) }
    }

    /// Access vblank related infrastructure for a [`Crtc`].
    ///
    /// This function explicitly locks the device's vblank lock, and allows access to controlling
    /// the vblank configuration for this CRTC. The lock is dropped once [`VblankGuard`] is
    /// dropped.
    pub fn vblank_lock<'a>(&'a self, irq: &'a LocalInterruptDisabled) -> VblankGuard<'a, T> {
        // SAFETY: `vbl_lock` is initialized for as long as `Crtc` is available to users
        // INVARIANT: We just acquired `vbl_lock`, fulfilling the invariants of `VblankGuard`
        unsafe { bindings::spin_lock(&raw mut (*self.drm_dev().as_raw()).vbl_lock) };

        // SAFETY: We just acquired vbl_lock above
        unsafe { VblankGuard::new(self, irq) }
    }

    /// Trigger a vblank event on this [`Crtc`].
    ///
    /// Drivers should use this in their vblank interrupt handlers to update the vblank counter and
    /// send any signals that may be pending.
    ///
    /// Returns whether or not the vblank event was handled.
    #[inline]
    pub fn handle_vblank(&self) -> bool {
        // SAFETY: `as_raw()` always returns a valid pointer to an initialized drm_crtc.
        unsafe { bindings::drm_crtc_handle_vblank(self.as_raw()) }
    }

    /// Forbid vblank events for a [`Crtc`].
    ///
    /// This function disables vblank events for a [`Crtc`], even if [`VblankRef`] objects exist.
    #[inline]
    pub fn vblank_off(&self) {
        // SAFETY: `as_raw()` always returns a valid pointer to an initialized drm_crtc.
        unsafe { bindings::drm_crtc_vblank_off(self.as_raw()) }
    }

    /// Allow vblank events for a [`Crtc`].
    ///
    /// This function allows users to enable vblank events and acquire [`VblankRef`] objects again.
    #[inline]
    pub fn vblank_on(&self) {
        // SAFETY: `as_raw()` always returns a valid pointer to an initialized drm_crtc.
        unsafe { bindings::drm_crtc_vblank_on(self.as_raw()) }
    }

    /// Enable vblank events for a [`Crtc`].
    ///
    /// Returns a [`VblankRef`] which will allow vblank events to be sent until it is dropped. Note
    /// that vblank events may still be disabled by [`Self::vblank_off`].
    #[must_use = "Vblanks are only enabled until the result from this function is dropped"]
    pub fn vblank_get(&self) -> Result<VblankRef<'_, T>> {
        VblankRef::new(self)
    }
}

/// Common methods available on any [`CrtcState`] whose [`Crtc`] implements [`VblankSupport`].
///
/// This trait is implemented automatically by DRM for any [`DriverCrtc`] implementation that
/// implements [`VblankSupport`].
pub trait RawVblankCrtcState: AsRawCrtcState {
    /// Return the [`PendingVblankEvent`] for this CRTC state, if there is one.
    fn get_pending_vblank_event(&mut self) -> Option<PendingVblankEvent<'_, Self>>
    where
        Self: Sized,
    {
        // SAFETY: The driver is the only one that will ever modify this data, and since our
        // interface follows rust's data aliasing rules that means this is safe to read
        let event_ptr = unsafe { *self.as_raw() }.event;

        (!event_ptr.is_null()).then_some(PendingVblankEvent(self))
    }
}

impl<T, C> RawVblankCrtcState for T
where
    T: AsRawCrtcState<Crtc = Crtc<C>>,
    C: VblankDriverCrtc,
{
}

/// A pending vblank event from an atomic state
pub struct PendingVblankEvent<'a, T: RawVblankCrtcState>(&'a mut T);

impl<'a, T: RawVblankCrtcState> PendingVblankEvent<'a, T> {
    /// Send this [`PendingVblankEvent`].
    ///
    /// A [`PendingVblankEvent`] can only be sent once, so this function consumes the
    /// [`PendingVblankEvent`].
    pub fn send<C>(self)
    where
        T: RawVblankCrtcState<Crtc = Crtc<C>>,
        C: VblankDriverCrtc,
    {
        let crtc: &Crtc<C> = self.0.crtc();
        let event_lock = crtc.drm_dev().event_lock();
        let _guard = event_lock.lock();

        // SAFETY:
        // - We now hold the appropriate lock to call this function
        // - Vblanks are enabled as proved by `vbl_ref`, as per the C api requirements
        // - Our interface is proof that `event` is non-null
        unsafe { bindings::drm_crtc_send_vblank_event(crtc.as_raw(), (*self.0.as_raw()).event) };

        // SAFETY: The mutable reference in `self.state` is proof that it is safe to mutate this,
        // and DRM expects us to set this to NULL once we've sent the vblank event.
        unsafe { (*self.0.as_raw()).event = null_mut() };
    }

    /// Arm this [`PendingVblankEvent`] to be sent later by the CRTC's vblank interrupt handler.
    ///
    /// A [`PendingVblankEvent`] can only be armed once, so this function consumes the
    /// [`PendingVblankEvent`]. As well, it requires a [`VblankRef`] so that vblank interrupts
    /// remain enabled until the [`PendingVblankEvent`] has been sent out by the driver's vblank
    /// interrupt handler.
    pub fn arm<C>(self, vbl_ref: VblankRef<'_, C>)
    where
        T: RawVblankCrtcState<Crtc = Crtc<C>>,
        C: VblankDriverCrtc,
    {
        let crtc: &Crtc<C> = self.0.crtc();
        let event_lock = crtc.drm_dev().event_lock();
        let _guard = event_lock.lock();

        // SAFETY:
        // - We now hold the appropriate lock to call this function
        // - Vblanks are enabled as proved by `vbl_ref`, as per the C api requirements
        // - Our interface is proof that `event` is non-null
        unsafe { bindings::drm_crtc_arm_vblank_event(crtc.as_raw(), (*self.0.as_raw()).event) };

        // SAFETY: The mutable reference in `self.state` is proof that it is safe to mutate this,
        // and DRM expects us to set this to NULL once we've armed the vblank event.
        unsafe { (*self.0.as_raw()).event = null_mut() };

        // DRM took ownership of `vbl_ref` after we called `drm_crtc_arm_vblank_event`
        mem::forget(vbl_ref);
    }
}

/// A borrowed vblank reference.
///
/// This object keeps the vblank reference count for a [`Crtc`] incremented for as long as it
/// exists, enabling vblank interrupts for said [`Crtc`] until all references are dropped, or
/// [`Crtc::vblank_off`] is called - whichever comes first.
pub struct VblankRef<'a, T: VblankDriverCrtc>(&'a Crtc<T>);

impl<T: VblankDriverCrtc> Drop for VblankRef<'_, T> {
    fn drop(&mut self) {
        // SAFETY: as_raw() returns a valid pointer to an initialized drm_crtc
        unsafe { bindings::drm_crtc_vblank_put(self.0.as_raw()) };
    }
}

impl<'a, T: VblankDriverCrtc> VblankRef<'a, T> {
    fn new(crtc: &'a Crtc<T>) -> Result<Self> {
        // SAFETY: as_raw() returns a valid pointer to an initialized drm_crtc
        to_result(unsafe { bindings::drm_crtc_vblank_get(crtc.as_raw()) })?;

        Ok(Self(crtc))
    }
}

/// The base wrapper for [`drm_vblank_crtc`].
///
/// Users will rarely interact with this object directly, it is a simple wrapper around
/// [`drm_vblank_crtc`] which provides access to methods and data that is not protected by a lock.
///
/// # Invariants
///
/// This type has an identical data layout to [`drm_vblank_crtc`].
///
/// [`drm_vblank_crtc`]: srctree/include/drm/drm_vblank.h
#[repr(transparent)]
pub struct VblankCrtc<T>(Opaque<bindings::drm_vblank_crtc>, PhantomData<T>);

impl<T: VblankDriverCrtc> VblankCrtc<T> {
    pub(crate) fn as_raw(&self) -> *mut bindings::drm_vblank_crtc {
        self.0.get()
    }

    // SAFETY: The caller promises that `ptr` points to a valid instance of
    // `bindings::drm_vblank_crtc`, and that access to this structure has been properly serialized
    pub(crate) unsafe fn from_raw<'a>(ptr: *mut bindings::drm_vblank_crtc) -> &'a Self {
        // SAFETY: Our data layouts are identical via #[repr(transparent)]
        unsafe { &*ptr.cast() }
    }

    /// Returns the [`Device`] for this [`VblankGuard`]
    pub fn drm_dev(&self) -> &Device<T::Driver> {
        // SAFETY: `drm` is initialized, invariant and valid throughout our lifetime
        unsafe { Device::from_raw((*self.as_raw()).dev) }
    }
}

// NOTE: This type does not use a `Guard` because the mutex is not contained within the same
// structure as the relevant CRTC
/// An interface for accessing and controlling vblank related state for a [`Crtc`].
///
/// This type may be returned from some [`VblankSupport`] callbacks, or manually via
/// [`Crtc::vblank_lock`]. It provides access to methods and data which require
/// [`drm_device.vbl_lock`] be held.
///
/// # Invariants
///
/// - [`drm_device.vbl_lock`] is acquired whenever an instance of this type exists.
/// - Shares the invariants of [`VblankCrtc`].
///
/// [`drm_device.vbl_lock`]: srctree/include/drm/drm_device.h
#[repr(transparent)]
pub struct VblankGuard<'a, T: VblankDriverCrtc>(&'a VblankCrtc<T>);

impl<'a, T: VblankDriverCrtc> VblankGuard<'a, T> {
    /// Construct a new [`VblankGuard`]
    ///
    /// # Safety
    ///
    /// The caller must have already acquired [`drm_device.vbl_lock`].
    ///
    /// [`drm_device.vbl_lock`]: srctree/include/drm/drm_device.h
    pub(crate) unsafe fn new(crtc: &'a Crtc<T>, _irq: &'a LocalInterruptDisabled) -> Self {
        // INVARIANT: The caller promises that we've acquired `vbl_lock`
        Self(crtc.vblank_crtc())
    }

    /// Returns the duration of a single scanout frame in ns.
    pub fn frame_duration(&self) -> i32 {
        // SAFETY: We hold the appropriate lock for this read via our type invariants.
        unsafe { *self.as_raw() }.framedur_ns
    }

    /// Return the vblank core's cached copy of the currently set display mode.
    ///
    /// If the display is disabled, this will return `None`.
    pub fn hwmode(&self) -> Option<&DisplayMode> {
        // SAFETY: We hold the appropriate lock for this read via our type invariants.
        let ptr = unsafe { &raw const (*self.as_raw()).hwmode };

        // SAFETY: We check here if the cached DisplayMode is Null, which means the only other
        // possibility is that the pointer points to a valid initialized drm_display_mode.
        (!ptr.is_null()).then(|| unsafe { DisplayMode::as_ref(ptr) })
    }
}

impl<T: VblankDriverCrtc> Deref for VblankGuard<'_, T> {
    type Target = VblankCrtc<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: VblankDriverCrtc> Drop for VblankGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY:
        // - We acquired this spinlock when creating this object
        // - This lock is guaranteed to be initialized for as long as our DRM device is exposed to
        //   users.
        unsafe { bindings::spin_unlock(&raw mut (*self.drm_dev().as_raw()).vbl_lock) }
    }
}
