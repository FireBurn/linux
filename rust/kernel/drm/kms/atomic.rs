// SPDX-License-Identifier: GPL-2.0 OR MIT

//! [`struct drm_atomic_commit`] related bindings for rust.
//!
//! [`struct drm_atomic_commit`]: srctree/include/drm/drm_atomic.h
use super::{connector::*, crtc::*, plane::*, KmsDriver, ModeObject};
use crate::{
    bindings,
    drm::device::Device,
    error::{from_err_ptr, to_result},
    prelude::*,
    sync::aref::{ARef, AlwaysRefCounted},
    types::*,
};
use core::{cell::Cell, marker::*, mem::ManuallyDrop, ops::*, ptr::NonNull};

/// The main wrapper around [`struct drm_atomic_commit`].
///
/// This type is usually embedded within another interface such as an [`AtomicStateMutator`].
///
/// # Invariants
///
/// - The data layout of this type is identical to [`struct drm_atomic_commit`].
/// - `state` is initialized for as long as this type is exposed to users.
///
/// [`struct drm_atomic_commit`]: srctree/include/drm/drm_atomic.h
#[repr(transparent)]
pub struct AtomicState<T: KmsDriver> {
    pub(super) state: Opaque<bindings::drm_atomic_commit>,
    _p: PhantomData<T>,
}

impl<T: KmsDriver> AtomicState<T> {
    /// Reconstruct an immutable reference to an atomic state from the given pointer
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid initialized instance of [`struct drm_atomic_commit`].
    ///
    /// [`struct drm_atomic_commit`]: srctree/include/drm/drm_atomic.h
    #[allow(dead_code)]
    pub(super) unsafe fn from_raw<'a>(ptr: *const bindings::drm_atomic_commit) -> &'a Self {
        // SAFETY: Our data layout is identical
        // INVARIANT: Our safety contract upholds the guarantee that `state` is initialized for as
        // long as this type is exposed to users.
        unsafe { &*ptr.cast() }
    }

    pub(crate) fn as_raw(&self) -> *mut bindings::drm_atomic_commit {
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
        unsafe { bindings::drm_atomic_commit_get(self.state.get()) }
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: `obj` contains a valid non-null pointer to an initialized `Self`.
        unsafe { bindings::drm_atomic_commit_put(obj.as_ptr().cast()) }
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
    /// `ptr` must point to a valid `drm_atomic_commit`
    #[allow(dead_code)]
    pub(super) unsafe fn new(ptr: NonNull<bindings::drm_atomic_commit>) -> Self {
        Self {
            // SAFETY: The data layout of `AtomicState<T>` is identical to drm_atomic_commit
            // We use `ManuallyDrop` because `AtomicStateMutator` is only ever provided to users in
            // the context of KMS callbacks. As such, skipping ref inc/dec for the atomic state is
            // convienent for our bindings.
            state: ManuallyDrop::new(unsafe { ARef::from_raw(ptr.cast()) }),
            borrowed_planes: Cell::default(),
            borrowed_crtcs: Cell::default(),
            borrowed_connectors: Cell::default(),
        }
    }

    pub(crate) fn as_raw(&self) -> *mut bindings::drm_atomic_commit {
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
    /// The caller guarantees that `ptr` points to a valid instance of `drm_atomic_commit`.
    pub(crate) unsafe fn new(ptr: NonNull<bindings::drm_atomic_commit>) -> Self {
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

/// A macro for declaring the repetitive take_all(), take_state(), etc. methods for atomic state
/// token types.
///
/// It is assumed that $token_name refers to a struct that contains two members:
///
/// - `state`: This should be the atomic state type to use
/// - The object in question. The name of this member is generated by converting $obj to lowercase.
///
/// The struct should have one lifetime ($lifetime_a) declared, and one meta-variable ($meta) which
/// should be bound to the Driver* trait for the given mode object.
macro_rules! impl_atomic_state_token_ops {
    (
        $token_name:ident,
        $state:ident,
        $obj:ident,
        use <$lifetime_a:lifetime, $meta:ident>
    ) => {
        kernel::macros::paste! {
            /// Create a new token.
            ///
            /// # Safety
            ///
            /// To use this function it must be known in the current context that:
            ///
            /// - The object has had its atomic states added to `state`.
            /// - No state mutator can possibly be taken out for the objects new state.
            pub(crate) unsafe fn new(
                [<$obj:lower>]: &$lifetime_a $obj<$meta>,
                state: &$lifetime_a $state<$meta::Driver>,
            ) -> Self {
                Self { [<$obj:lower>], state }
            }

            #[doc = concat!("Get the [`", stringify!($obj), "`] associated with this",
                            " [`", stringify!($token_name), "`].")]
            pub fn [<$obj:lower>](&self) -> &$lifetime_a $obj<$meta> {
                self.[<$obj:lower>]
            }

            /// Exchange this token for a (atomic_state, old_state, new_state) tuple.
            pub fn take_all(self) -> (
                &$lifetime_a $state<$meta::Driver>,
                &$lifetime_a [<$obj State>]<$meta::State>,
                [<$obj StateMutator>]<$lifetime_a, [<$obj State>]<$meta::State>>,
            ) {
                let (old_state, new_state) = (
                    self.state.[<get_old_ $obj:lower _state>](self.[<$obj:lower>]),
                    self.state.[<get_new_ $obj:lower _state>](self.[<$obj:lower>]),
                );

                // SAFETY:
                // - Both the old and new object state are present in `state` via our type
                //   invariants.
                // - The new state is guaranteed to have no mutators taken out via our type
                //   invariants.
                let (old_state, new_state) = unsafe {
                    (old_state.unwrap_unchecked(), new_state.unwrap_unchecked())
                };

                (self.state, old_state, new_state)
            }

            #[doc = concat!("Exchange this token for the old [`", stringify!($obj), "State`].")]
            pub fn take_old_state(self) -> &$lifetime_a [<$obj State>]<$meta::State> {
                let old = self.state.[<get_old_ $obj:lower _state>](self.[<$obj:lower>]);

                // SAFETY: The old state is guaranteed to be present in `state` via our type
                // invariants.
                unsafe { old.unwrap_unchecked() }
            }

            #[doc = concat!("Exchange this token for the new [`", stringify!($obj), "State`].")]
            pub fn take_new_state(
                self
            ) -> [<$obj StateMutator>]<$lifetime_a, [<$obj State>]<$meta::State>> {
                let new = self.state.[<get_new_ $obj:lower _state>](self.[<$obj:lower>]);

                // SAFETY:
                // - The new state is guaranteed to be present in our `state` via our type
                //   invariants.
                // - The new state is guaranteed not to have any mutators taken out for it via our
                //   type invariants.
                unsafe { new.unwrap_unchecked() }
            }

            #[doc = concat!("Exchange this token for both the old and new [`",
                            stringify!($obj), "State`].")]
            pub fn take_old_new_state(self) -> (
                &$lifetime_a [<$obj State>]<$meta::State>,
                [<$obj StateMutator>]<$lifetime_a, [<$obj State>]<$meta::State>>,
            ) {
                let (old_state, new_state) = (
                    self.state.[<get_old_ $obj:lower _state>](self.[<$obj:lower>]),
                    self.state.[<get_new_ $obj:lower _state>](self.[<$obj:lower>]),
                );

                // SAFETY:
                // - Both the old and new object state are present in `state` via our type
                //   invariants.
                // - The new state is guaranteed to have no mutators taken out via our type
                //   invariants.
                let (old_state, new_state) = unsafe {
                    (old_state.unwrap_unchecked(), new_state.unwrap_unchecked())
                };

                (old_state, new_state)
            }

            #[doc = concat!("Exchange this token for both the [`", stringify!($state),
                            "`] and the old [`", stringify!($obj), "State`].")]
            pub fn take_state_old_state(self) -> (
                &$lifetime_a $state<$meta::Driver>,
                &$lifetime_a [<$obj State>]<$meta::State>,
            ) {
                let old = self.state.[<get_old_ $obj:lower _state>](self.[<$obj:lower>]);

                // SAFETY: The old state is guaranteed to be present in `state` via our type
                // invariants.
                (self.state, unsafe { old.unwrap_unchecked() })
            }

            #[doc = concat!("Exchange this token for both the [`", stringify!($state),
                            "`] and the new [`", stringify!($obj), "State`].")]
            pub fn take_state_new_state(self) -> (
                &$lifetime_a $state<$meta::Driver>,
                [<$obj StateMutator>]<$lifetime_a, [<$obj State>]<$meta::State>>,
            ) {
                let new = self.state.[<get_new_ $obj:lower _state>](self.[<$obj:lower>]);

                // SAFETY:
                // - The new state is guaranteed to be present in `state` via our type
                //   invariants.
                // - The new state is guaranteed to have no mutators taken out via our type
                //   invariants.
                (self.state, unsafe { new.unwrap_unchecked() })
            }
        }

        #[doc = concat!("Exchange this token for the [`", stringify!($state), "`].")]
        pub fn take_state(self) -> &$lifetime_a $state<$meta::Driver> {
            self.state
        }
    };
}

pub(crate) use impl_atomic_state_token_ops;

/// A token proving that no modesets for a commit have completed.
///
/// This token is proof that no commits have yet completed, and is provided as an argument to
/// [`KmsDriver::atomic_commit_tail`]. This may be used with
/// [`AtomicCommitTail::commit_modeset_disables`].
pub struct ModesetsReadyToken<'a>(PhantomData<&'a ()>);

/// A token proving that modeset disables for a commit have completed.
///
/// This token is proof that an implementor's [`KmsDriver::atomic_commit_tail`] phase has finished
/// committing any operations which disable mode objects. It is returned by
/// [`AtomicCommitTail::commit_modeset_disables`], and can be used with
/// [`AtomicCommitTail::commit_modeset_enables`] to acquire a [`EnablesCommittedToken`].
pub struct DisablesCommittedToken<'a>(PhantomData<&'a ()>);

/// A token proving that modeset enables for a commit have completed.
///
/// This token is proof that an implementor's [`KmsDriver::atomic_commit_tail`] phase has finished
/// committing any operations which enable mode objects. It is returned by
/// [`AtomicCommitTail::commit_modeset_enables`].
pub struct EnablesCommittedToken<'a>(PhantomData<&'a ()>);

/// A token proving that no plane updates for a commit have completed.
///
/// This token is proof that no plane updates have yet been completed within an implementor's
/// [`KmsDriver::atomic_commit_tail`] implementation, and that we are ready to begin updating planes. It
/// is provided as an argument to [`KmsDriver::atomic_commit_tail`].
pub struct PlaneUpdatesReadyToken<'a>(PhantomData<&'a ()>);

/// A token proving that all plane updates for a commit have completed.
///
/// This token is proof that all plane updates within an implementor's [`KmsDriver::atomic_commit_tail`]
/// implementation have completed. It is returned by [`AtomicCommitTail::commit_planes`].
pub struct PlaneUpdatesCommittedToken<'a>(PhantomData<&'a ()>);

/// An [`AtomicState`] interface that allows a driver to control the [`atomic_commit_tail`]
/// callback.
///
/// This object is provided as an argument to [`KmsDriver::atomic_commit_tail`], and represents an atomic
/// state within the commit tail phase which is still in the process of being committed to hardware.
/// It may be used to control the order in which the commit process happens.
///
/// # Invariants
///
/// Same as [`AtomicState`].
///
/// [`atomic_commit_tail`]: srctree/include/drm/drm_modeset_helper_vtables.h
pub struct AtomicCommitTail<'a, T: KmsDriver>(&'a AtomicState<T>);

impl<'a, T: KmsDriver> AtomicCommitTail<'a, T> {
    /// Commit modesets which would disable outputs.
    ///
    /// This function commits any modesets which would shut down outputs, along with preparing them
    /// for a new mode (if needed).
    ///
    /// Since it is physically impossible to disable an output multiple times, and since it is
    /// logically unsound to disable an output within an atomic commit after the output was enabled
    /// in the same commit - this function requires a [`ModesetsReadyToken`] to consume and returns
    /// a [`DisablesCommittedToken`].
    ///
    /// If compatibility with legacy CRTC helpers is desired, this
    /// should be called before [`commit_planes`] which is what the default commit function does.
    /// But drivers with different needs can group the modeset commits tgether and do the plane
    /// commits at the end. This is useful for drivers doing runtime PM since then plane updates
    /// only happen when the CRTC is actually enabled.
    ///
    /// [`commit_planes`]: AtomicCommitTail::commit_planes
    #[inline]
    #[must_use]
    pub fn commit_modeset_disables<'b>(
        &mut self,
        _token: ModesetsReadyToken<'_>,
    ) -> DisablesCommittedToken<'b> {
        // SAFETY: Both `as_raw()` calls are guaranteed to return valid pointers
        unsafe {
            bindings::drm_atomic_helper_commit_modeset_disables(
                self.0.drm_dev().as_raw(),
                self.0.as_raw(),
            )
        }

        DisablesCommittedToken(PhantomData)
    }

    /// Commit all plane updates.
    ///
    /// This function performs all plane updates for the given [`AtomicCommitTail`]. Since it is
    /// logically unsound to perform the same plane update more then once in a given atomic commit,
    /// this function requires a [`PlaneUpdatesReadyToken`] to consume and returns a
    /// [`PlaneUpdatesCommittedToken`] to prove that plane updates for the state have completed.
    #[inline]
    #[must_use]
    pub fn commit_planes<'b>(
        &mut self,
        _token: PlaneUpdatesReadyToken<'_>,
        flags: PlaneCommitFlags,
    ) -> PlaneUpdatesCommittedToken<'b> {
        // SAFETY: Both `as_raw()` calls are guaranteed to return valid pointers
        unsafe {
            bindings::drm_atomic_helper_commit_planes(
                self.0.drm_dev().as_raw(),
                self.0.as_raw(),
                flags.into(),
            )
        }

        PlaneUpdatesCommittedToken(PhantomData)
    }

    /// Commit modesets which would enable outputs.
    ///
    /// This function commits any modesets in the given [`AtomicCommitTail`] which would enable
    /// outputs, along with preparing them for their new modes (if needed).
    ///
    /// Since it is logically unsound to enable an output before any disabling modesets within the
    /// same atomic commit have been performed, and physically impossible to enable the same output
    /// multiple times - this function requires a [`DisablesCommittedToken`] to consume and returns
    /// a [`EnablesCommittedToken`] which may be used as proof that all modesets in the state have
    /// been completed.
    #[inline]
    #[must_use]
    pub fn commit_modeset_enables<'b>(
        &mut self,
        _token: DisablesCommittedToken<'_>,
    ) -> EnablesCommittedToken<'b> {
        // SAFETY: Both `as_raw()` calls are guaranteed to return valid pointers
        unsafe {
            bindings::drm_atomic_helper_commit_modeset_enables(
                self.0.drm_dev().as_raw(),
                self.0.as_raw(),
            )
        }

        EnablesCommittedToken(PhantomData)
    }

    /// Fake vblank events if needed.
    ///
    /// Note that this is still relevant to drivers which don't implement [`VblankSupport`] for any
    /// of their CRTCs.
    ///
    /// TODO: more doc
    ///
    /// [`VblankSupport`]: super::vblank::VblankSupport
    pub fn fake_vblank(&mut self) {
        // SAFETY: `as_raw()` is guaranteed to always return a valid pointer
        unsafe { bindings::drm_atomic_helper_fake_vblank(self.0.as_raw()) }
    }

    /// Signal completion of the hardware commit step.
    ///
    /// This swaps the atomic state into the relevant atomic state pointers and marks the hardware
    /// commit step as completed. Since this step can only happen after all plane updates and
    /// modesets within an [`AtomicCommitTail`] have been completed, it requires both a
    /// [`EnablesCommittedToken`] and a [`PlaneUpdatesCommittedToken`] to consume. After this
    /// function is called, the caller no longer has exclusive access to the underlying atomic
    /// state. As such, this function consumes the [`AtomicCommitTail`] object and returns a
    /// [`CommittedAtomicState`] accessor for performing post-hw commit tasks.
    pub fn commit_hw_done<'b>(
        self,
        _modeset_token: EnablesCommittedToken<'_>,
        _plane_updates_token: PlaneUpdatesCommittedToken<'_>,
    ) -> CommittedAtomicState<'b, T>
    where
        'a: 'b,
    {
        // SAFETY: we consume the `AtomicCommitTail` object, making it impossible for the user to
        // mutate the state after this function has been called - which upholds the safety
        // requirements of the C API allowing us to safely call this function
        unsafe { bindings::drm_atomic_helper_commit_hw_done(self.0.as_raw()) };

        CommittedAtomicState(self.0)
    }
}

// The actual raw C callback for custom atomic commit tail implementations
pub(crate) unsafe extern "C" fn commit_tail_callback<T: KmsDriver>(
    state: *mut bindings::drm_atomic_commit,
) {
    // SAFETY:
    // - We're guaranteed by DRM that `state` always points to a valid instance of
    //   `bindings::drm_atomic_commit`
    // - This conversion is safe via the type invariants
    let state = unsafe { AtomicState::from_raw(state.cast_const()) };

    T::atomic_commit_tail(
        AtomicCommitTail(state),
        ModesetsReadyToken(PhantomData),
        PlaneUpdatesReadyToken(PhantomData),
    );
}

/// An [`AtomicState`] which was just committed with [`AtomicCommitTail::commit_hw_done`].
///
/// This object represents an [`AtomicState`] which has been fully committed to hardware, and as
/// such may no longer be mutated as it is visible to userspace. It may be used to control what
/// happens immediately after an atomic commit finishes within the [`atomic_commit_tail`] callback.
///
/// Since acquiring this object means that all modesetting locks have been dropped, a non-blocking
/// commit could happen at the same time an [`atomic_commit_tail`] implementer has access to this
/// object. Thus, it cannot be assumed that this object represents the current hardware state - and
/// instead only represents the final result of the [`AtomicCommitTail`] that was just committed.
///
/// # Invariants
///
/// It may be assumed that [`drm_atomic_helper_commit_hw_done`] has been called as long as this type
/// exists.
///
/// [`atomic_commit_tail`]: KmsDriver::atomic_commit_tail
/// [`drm_atomic_helper_commit_hw_done`]: srctree/include/drm/drm_atomic_helper.h
pub struct CommittedAtomicState<'a, T: KmsDriver>(&'a AtomicState<T>);

impl<'a, T: KmsDriver> CommittedAtomicState<'a, T> {
    /// Wait for page flips on this state to complete
    pub fn wait_for_flip_done(&self) {
        // SAFETY: `drm_atomic_helper_commit_hw_done` has been called via our invariants
        unsafe {
            bindings::drm_atomic_helper_wait_for_flip_done(
                self.0.drm_dev().as_raw(),
                self.0.as_raw(),
            )
        }
    }
}

impl<'a, T: KmsDriver> Drop for CommittedAtomicState<'a, T> {
    fn drop(&mut self) {
        // SAFETY:
        // * This interface represents the last atomic state accessor which could be affected as a
        //   result of resources from an atomic commit being cleaned up.
        unsafe {
            bindings::drm_atomic_helper_cleanup_planes(self.0.drm_dev().as_raw(), self.0.as_raw())
        }
    }
}

/// An enumator representing a single flag in [`PlaneCommitFlags`].
///
/// This is a non-exhaustive list, as the C side could add more later.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlaneCommitFlag {
    /// Don't notify applications of plane updates for newly-disabled planes. Drivers are encouraged
    /// to set this flag by default, as otherwise they need to ignore plane updates for disabled
    /// planes by hand.
    ActiveOnly = (1 << 0),
    /// Tell the DRM core that the display hardware requires that a [`Crtc`]'s planes must be
    /// disabled when the [`Crtc`] is disabled. When not specified,
    /// [`AtomicCommitTail::commit_planes`] will skip the atomic disable callbacks for a plane if
    /// the [`Crtc`] in the old [`PlaneState`] needs a modesetting operation. It is still up to the
    /// driver to disable said planes in their [`DriverCrtc::atomic_disable`] callback.
    NoDisableAfterModeset = (1 << 1),
}

impl BitOr for PlaneCommitFlag {
    type Output = PlaneCommitFlags;

    fn bitor(self, rhs: Self) -> Self::Output {
        PlaneCommitFlags(self as u32 | rhs as u32)
    }
}

impl BitOr<PlaneCommitFlags> for PlaneCommitFlag {
    type Output = PlaneCommitFlags;

    fn bitor(self, rhs: PlaneCommitFlags) -> Self::Output {
        PlaneCommitFlags(self as u32 | rhs.0)
    }
}

/// A bitmask for controlling the behavior of [`AtomicCommitTail::commit_planes`].
///
/// This corresponds to the `DRM_PLANE_COMMIT_*` flags on the C side. Note that this bitmask does
/// not discard unknown values in order to ensure that adding new flags on the C side of things does
/// not break anything in the future.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct PlaneCommitFlags(u32);

impl From<PlaneCommitFlag> for PlaneCommitFlags {
    fn from(value: PlaneCommitFlag) -> Self {
        Self(value as u32)
    }
}

impl From<PlaneCommitFlags> for u32 {
    fn from(value: PlaneCommitFlags) -> Self {
        value.0
    }
}

impl BitOr for PlaneCommitFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PlaneCommitFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs
    }
}

impl BitAnd for PlaneCommitFlags {
    type Output = PlaneCommitFlags;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PlaneCommitFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs
    }
}

impl BitOr<PlaneCommitFlag> for PlaneCommitFlags {
    type Output = Self;

    fn bitor(self, rhs: PlaneCommitFlag) -> Self::Output {
        self | Self::from(rhs)
    }
}

impl BitOrAssign<PlaneCommitFlag> for PlaneCommitFlags {
    fn bitor_assign(&mut self, rhs: PlaneCommitFlag) {
        *self = *self | rhs
    }
}

impl BitAnd<PlaneCommitFlag> for PlaneCommitFlags {
    type Output = PlaneCommitFlags;

    fn bitand(self, rhs: PlaneCommitFlag) -> Self::Output {
        self & Self::from(rhs)
    }
}

impl BitAndAssign<PlaneCommitFlag> for PlaneCommitFlags {
    fn bitand_assign(&mut self, rhs: PlaneCommitFlag) {
        *self = *self & rhs
    }
}

impl PlaneCommitFlags {
    /// Create a new bitmask.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the bitmask has the given commit flag set.
    pub fn has(&self, flag: PlaneCommitFlag) -> bool {
        *self & flag == flag.into()
    }
}
