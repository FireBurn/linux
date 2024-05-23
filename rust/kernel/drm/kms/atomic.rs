// SPDX-License-Identifier: GPL-2.0 OR MIT

//! [`struct drm_atomic_state`] related bindings for rust.
//!
//! [`struct drm_atomic_state`]: srctree/include/drm/drm_atomic.h
use super::{connector::*, crtc::*, plane::*, KmsDriver, ModeObject};
use crate::{
    bindings,
    drm::device::Device,
    error::{from_err_ptr, to_result},
    prelude::*,
    types::*,
};
use core::{cell::Cell, marker::*, mem::ManuallyDrop, ops::*, ptr::NonNull};

/// The main wrapper around [`struct drm_atomic_state`].
///
/// This type is usually embedded within another interface such as an [`AtomicStateMutator`].
///
/// # Invariants
///
/// - The data layout of this type is identical to [`struct drm_atomic_state`].
/// - `state` is initialized for as long as this type is exposed to users.
///
/// [`struct drm_atomic_state`]: srctree/include/drm/drm_atomic.h
#[repr(transparent)]
pub struct AtomicState<T: KmsDriver> {
    pub(super) state: Opaque<bindings::drm_atomic_state>,
    _p: PhantomData<T>,
}

impl<T: KmsDriver> AtomicState<T> {
    /// Reconstruct an immutable reference to an atomic state from the given pointer
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid initialized instance of [`struct drm_atomic_state`].
    ///
    /// [`struct drm_atomic_state`]: srctree/include/drm/drm_atomic.h
    #[allow(dead_code)]
    pub(super) unsafe fn from_raw<'a>(ptr: *const bindings::drm_atomic_state) -> &'a Self {
        // SAFETY: Our data layout is identical
        // INVARIANT: Our safety contract upholds the guarantee that `state` is initialized for as
        // long as this type is exposed to users.
        unsafe { &*ptr.cast() }
    }

    pub(crate) fn as_raw(&self) -> *mut bindings::drm_atomic_state {
        self.state.get()
    }

    /// Return the [`Device`] associated with this [`AtomicState`].
    pub fn drm_dev(&self) -> &Device<T> {
        // SAFETY:
        // - `state` is initialized via our type invariants.
        // - `dev` is invariant throughout the lifetime of `AtomicState`
        unsafe { Device::from_raw((*self.state.get()).dev) }
    }

    /// Return the old atomic state for `crtc`, if it is present within this [`AtomicState`].
    pub fn get_old_crtc_state<C>(&self, crtc: &C) -> Option<&C::State>
    where
        C: ModesettableCrtc + ModeObject<Driver = T>,
    {
        // SAFETY: This function either returns NULL or a valid pointer to a `drm_crtc_state`
        unsafe {
            bindings::drm_atomic_get_old_crtc_state(self.as_raw(), crtc.as_raw())
                .as_ref()
                .map(|p| C::State::from_raw(p))
        }
    }

    /// Return the old atomic state for `plane`, if it is present within this [`AtomicState`].
    pub fn get_old_plane_state<P>(&self, plane: &P) -> Option<&P::State>
    where
        P: ModesettablePlane + ModeObject<Driver = T>,
    {
        // SAFETY: This function either returns NULL or a valid pointer to a `drm_plane_state`
        unsafe {
            bindings::drm_atomic_get_old_plane_state(self.as_raw(), plane.as_raw())
                .as_ref()
                .map(|p| P::State::from_raw(p))
        }
    }

    /// Return the old atomic state for `connector` if it is present within this [`AtomicState`].
    pub fn get_old_connector_state<C>(&self, connector: &C) -> Option<&C::State>
    where
        C: ModesettableConnector + ModeObject<Driver = T>,
    {
        // SAFETY: This function either returns NULL or a valid pointer to a `drm_connector_state`.
        unsafe {
            bindings::drm_atomic_get_old_connector_state(self.as_raw(), connector.as_raw())
                .as_ref()
                .map(|p| C::State::from_raw(p))
        }
    }
}

// SAFETY: DRM atomic state objects are always reference counted and the get/put functions satisfy
// the requirements.
unsafe impl<T: KmsDriver> AlwaysRefCounted for AtomicState<T> {
    fn inc_ref(&self) {
        // SAFETY: `state` is initialized for as long as this type is exposed to users
        unsafe { bindings::drm_atomic_state_get(self.state.get()) }
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: `obj` contains a valid non-null pointer to an initialized `Self`.
        unsafe { bindings::drm_atomic_state_put(obj.as_ptr().cast()) }
    }
}

/// A smart-pointer for modifying the contents of an atomic state.
///
/// As it's not unreasonable for a modesetting driver to want to have references to the state of
/// multiple modesetting objects at once, along with mutating multiple states for unique modesetting
/// objects at once, this type provides a mechanism for safely doing both of these things.
///
/// To honor Rust's aliasing rules regarding mutable references, this structure ensures only one
/// mutable reference to a mode object's atomic state may exist at a time - and refuses to provide
/// another if one has already been taken out using runtime checks.
pub struct AtomicStateMutator<T: KmsDriver> {
    /// The state being mutated. Note that the use of `ManuallyDrop` here is because mutators are
    /// only constructed in FFI callbacks and thus borrow their references to the atomic state from
    /// DRM. Composers, which make use of mutators internally, can potentially be owned by rust code
    /// if a driver is performing an atomic commit internally - and thus will call the drop
    /// implementation here.
    state: ManuallyDrop<ARef<AtomicState<T>>>,

    /// Bitmask of borrowed CRTC state objects
    pub(super) borrowed_crtcs: Cell<u32>,
    /// Bitmask of borrowed plane state objects
    pub(super) borrowed_planes: Cell<u32>,
    /// Bitmask of borrowed connector state objects
    pub(super) borrowed_connectors: Cell<u32>,
}

impl<T: KmsDriver> AtomicStateMutator<T> {
    /// Construct a new [`AtomicStateMutator`]
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid `drm_atomic_state`
    #[allow(dead_code)]
    pub(super) unsafe fn new(ptr: NonNull<bindings::drm_atomic_state>) -> Self {
        Self {
            // SAFETY: The data layout of `AtomicState<T>` is identical to drm_atomic_state
            // We use `ManuallyDrop` because `AtomicStateMutator` is only ever provided to users in
            // the context of KMS callbacks. As such, skipping ref inc/dec for the atomic state is
            // convienent for our bindings.
            state: ManuallyDrop::new(unsafe { ARef::from_raw(ptr.cast()) }),
            borrowed_planes: Cell::default(),
            borrowed_crtcs: Cell::default(),
            borrowed_connectors: Cell::default(),
        }
    }

    pub(crate) fn as_raw(&self) -> *mut bindings::drm_atomic_state {
        self.state.as_raw()
    }

    /// Return the [`Device`] for this [`AtomicStateMutator`].
    pub fn drm_dev(&self) -> &Device<T> {
        self.state.drm_dev()
    }

    /// Retrieve the last committed atomic state for `crtc` if `crtc` has already been added to the
    /// atomic state being composed.
    ///
    /// Returns `None` otherwise.
    pub fn get_old_crtc_state<C>(&self, crtc: &C) -> Option<&C::State>
    where
        C: ModesettableCrtc + ModeObject<Driver = T>,
    {
        self.state.get_old_crtc_state(crtc)
    }

    /// Retrieve the last committed atomic state for `connector` if `connector` has already been
    /// added to the atomic state being composed.
    ///
    /// Returns `None` otherwise.
    pub fn get_old_connector_state<C>(&self, connector: &C) -> Option<&C::State>
    where
        C: ModesettableConnector + ModeObject<Driver = T>,
    {
        self.state.get_old_connector_state(connector)
    }

    /// Retrieve the last committed atomic state for `plane` if `plane` has already been added to
    /// the atomic state being composed.
    ///
    /// Returns `None` otherwise.
    pub fn get_old_plane_state<P>(&self, plane: &P) -> Option<&P::State>
    where
        P: ModesettablePlane + ModeObject<Driver = T>,
    {
        self.state.get_old_plane_state(plane)
    }

    /// Return a composer for `plane`s new atomic state if it was previously added to the atomic
    /// state being composed.
    ///
    /// Returns `None` otherwise, or if another mutator still exists for this state.
    pub fn get_new_crtc_state<C>(&self, crtc: &C) -> Option<CrtcStateMutator<'_, C::State>>
    where
        C: ModesettableCrtc + ModeObject<Driver = T>,
    {
        // SAFETY: DRM either returns NULL or a valid pointer to a `drm_crtc_state`
        let state =
            unsafe { bindings::drm_atomic_get_new_crtc_state(self.as_raw(), crtc.as_raw()) };

        CrtcStateMutator::<C::State>::new(self, NonNull::new(state)?)
    }

    /// Return a composer for `plane`s new atomic state if it was previously added to the atomic
    /// state being composed.
    ///
    /// Returns `None` otherwise, or if another mutator still exists for this state.
    pub fn get_new_plane_state<P>(&self, plane: &P) -> Option<PlaneStateMutator<'_, P::State>>
    where
        P: ModesettablePlane + ModeObject<Driver = T>,
    {
        // SAFETY: DRM either returns NULL or a valid pointer to a `drm_plane_state`.
        let state =
            unsafe { bindings::drm_atomic_get_new_plane_state(self.as_raw(), plane.as_raw()) };

        PlaneStateMutator::<P::State>::new(self, NonNull::new(state)?)
    }

    /// Return a composer for `crtc`s new atomic state if it was previously added to the atomic
    /// state being composed.
    ///
    /// Returns `None` otherwise, or if another mutator still exists for this state.
    pub fn get_new_connector_state<C>(
        &self,
        connector: &C,
    ) -> Option<ConnectorStateMutator<'_, C::State>>
    where
        C: ModesettableConnector + ModeObject<Driver = T>,
    {
        // SAFETY: DRM either returns NULL or a valid pointer to a `drm_connector_state`
        let state = unsafe {
            bindings::drm_atomic_get_new_connector_state(self.as_raw(), connector.as_raw())
        };

        ConnectorStateMutator::<C::State>::new(self, NonNull::new(state)?)
    }
}

/// An [`AtomicStateMutator`] wrapper which is not yet part of any commit operation.
///
/// Since it's not yet part of a commit operation, new mode objects may be added to the state. It
/// also holds a reference to the underlying [`AtomicState`] that will be released when this object
/// is dropped.
pub struct AtomicStateComposer<T: KmsDriver>(AtomicStateMutator<T>);

impl<T: KmsDriver> Deref for AtomicStateComposer<T> {
    type Target = AtomicStateMutator<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: KmsDriver> Drop for AtomicStateComposer<T> {
    fn drop(&mut self) {
        // SAFETY: We're in drop, so this is guaranteed to be the last possible reference
        unsafe { ManuallyDrop::drop(&mut self.0.state) }
    }
}

impl<T: KmsDriver> AtomicStateComposer<T> {
    /// # Safety
    ///
    /// The caller guarantees that `ptr` points to a valid instance of `drm_atomic_state`.
    #[allow(dead_code)]
    pub(crate) unsafe fn new(ptr: NonNull<bindings::drm_atomic_state>) -> Self {
        // SAFETY: see `AtomicStateMutator::from_raw()`
        Self(unsafe { AtomicStateMutator::new(ptr) })
    }

    /// Attempt to add the state for `crtc` to the atomic state for this composer if it hasn't
    /// already been added, and create a mutator for it.
    ///
    /// If a composer already exists for this `crtc`, this function returns `Error(EBUSY)`. If
    /// attempting to add the state fails, another error code will be returned.
    pub fn add_crtc_state<C>(&self, crtc: &C) -> Result<CrtcStateMutator<'_, C::State>>
    where
        C: ModesettableCrtc + ModeObject<Driver = T>,
    {
        // SAFETY: DRM will only return a valid pointer to a `drm_crtc_state` - or an error.
        let state = unsafe {
            from_err_ptr(bindings::drm_atomic_get_crtc_state(
                self.as_raw(),
                crtc.as_raw(),
            ))
            .map(|c| NonNull::new_unchecked(c))
        }?;

        CrtcStateMutator::<C::State>::new(self, state).ok_or(EBUSY)
    }

    /// Attempt to add the state for `plane` to the atomic state for this composer if it hasn't
    /// already been added, and create a mutator for it.
    ///
    /// If a composer already exists for this `plane`, this function returns `Error(EBUSY)`. If
    /// attempting to add the state fails, another error code will be returned.
    pub fn add_plane_state<P>(&self, plane: &P) -> Result<PlaneStateMutator<'_, P::State>>
    where
        P: ModesettablePlane + ModeObject<Driver = T>,
    {
        // SAFETY: DRM will only return a valid pointer to a `drm_plane_state` - or an error.
        let state = unsafe {
            from_err_ptr(bindings::drm_atomic_get_plane_state(
                self.as_raw(),
                plane.as_raw(),
            ))
            .map(|p| NonNull::new_unchecked(p))
        }?;

        PlaneStateMutator::<P::State>::new(self, state).ok_or(EBUSY)
    }

    /// Attempt to add the state for `connector` to the atomic state for this composer if it hasn't
    /// already been added, and create a mutator for it.
    ///
    /// If a composer already exists for this `connector`, this function returns `Error(EBUSY)`. If
    /// attempting to add the state fails, another error code will be returned.
    pub fn add_connector_state<C>(
        &self,
        connector: &C,
    ) -> Result<ConnectorStateMutator<'_, C::State>>
    where
        C: ModesettableConnector + ModeObject<Driver = T>,
    {
        // SAFETY: DRM will only return a valid pointer to a `drm_plane_state` - or an error.
        let state = unsafe {
            from_err_ptr(bindings::drm_atomic_get_connector_state(
                self.as_raw(),
                connector.as_raw(),
            ))
            .map(|c| NonNull::new_unchecked(c))
        }?;

        ConnectorStateMutator::<C::State>::new(self, state).ok_or(EBUSY)
    }

    /// Attempt to add any planes affected by changes on `crtc` to this [`AtomicStateComposer`].
    ///
    /// Will return an [`Error`] if this fails.
    pub fn add_affected_planes<C>(&self, crtc: &C) -> Result
    where
        C: ModesettableCrtc + ModeObject<Driver = T>,
    {
        // SAFETY: Both .as_raw() values are guaranteed to return a valid pointer
        to_result(unsafe { bindings::drm_atomic_add_affected_planes(self.as_raw(), crtc.as_raw()) })
    }
}
