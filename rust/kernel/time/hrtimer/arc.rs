// SPDX-License-Identifier: GPL-2.0

use super::HasHrTimer;
use super::HrTimer;
use super::HrTimerCallback;
use super::HrTimerCallbackContext;
use super::HrTimerHandle;
use super::HrTimerMode;
use super::HrTimerPointer;
use super::RawHrTimerCallback;
use crate::sync::Arc;
use crate::sync::ArcBorrow;

/// A handle for an `Arc<HasHrTimer<T>>` returned by a call to
/// [`HrTimerPointer::start`].
pub struct ArcHrTimerHandle<T>
where
    T: HasHrTimer<T>,
{
    pub(crate) inner: Arc<T>,
}

// SAFETY: We implement drop below, and we cancel the timer in the drop
// implementation.
unsafe impl<T> HrTimerHandle for ArcHrTimerHandle<T>
where
    T: HasHrTimer<T>,
{
    fn cancel(&mut self) -> bool {
        let self_ptr = Arc::as_ptr(&self.inner);

        // SAFETY: As we obtained `self_ptr` from a valid reference above, it
        // must point to a valid `T`.
        let timer_ptr = unsafe { <T as HasHrTimer<T>>::raw_get_timer(self_ptr) };

        // SAFETY: As `timer_ptr` points into `T` and `T` is valid, `timer_ptr`
        // must point to a valid `HrTimer` instance.
        unsafe { HrTimer::<T>::raw_cancel(timer_ptr) }
    }
}

impl<T> ArcHrTimerHandle<T>
where
    T: HasHrTimer<T>,
{
    /// Restart the timer with a new expiry time, without cancelling it first.
    ///
    /// If the timer is queued it is removed and re-inserted at the new expiry; if it has already
    /// expired it is queued again. Unlike dropping the handle and calling
    /// [`HrTimerPointer::start`] again, this never blocks waiting for a running callback, so it
    /// can be used from contexts that cannot sleep -- re-arming a timer from a driver callback
    /// invoked with interrupts disabled, for instance.
    ///
    /// This handle keeps its timer alive and still cancels it on drop, so the timer cannot outlive
    /// the restart.
    pub fn restart(&self, expires: <<T as HasHrTimer<T>>::TimerMode as HrTimerMode>::Expires) {
        // SAFETY:
        // - `self.inner` is a live `Arc<T>` held by this handle, so the pointer is valid.
        // - The caller cannot leak past the timer's death: this handle owns the `Arc` and cancels
        //   the timer when dropped, which is the requirement `HasHrTimer::start` places on us.
        unsafe { T::start(Arc::as_ptr(&self.inner), expires) };
    }
}

impl<T> Drop for ArcHrTimerHandle<T>
where
    T: HasHrTimer<T>,
{
    fn drop(&mut self) {
        self.cancel();
    }
}

impl<T> HrTimerPointer for Arc<T>
where
    T: 'static,
    T: Send + Sync,
    T: HasHrTimer<T>,
    T: for<'a> HrTimerCallback<Pointer<'a> = Self>,
{
    type TimerMode = <T as HasHrTimer<T>>::TimerMode;
    type TimerHandle = ArcHrTimerHandle<T>;

    fn start(
        self,
        expires: <<T as HasHrTimer<T>>::TimerMode as HrTimerMode>::Expires,
    ) -> ArcHrTimerHandle<T> {
        // SAFETY:
        //  - We keep `self` alive by wrapping it in a handle below.
        //  - Since we generate the pointer passed to `start` from a valid
        //    reference, it is a valid pointer.
        unsafe { T::start(Arc::as_ptr(&self), expires) };
        ArcHrTimerHandle { inner: self }
    }
}

impl<T> RawHrTimerCallback for Arc<T>
where
    T: 'static,
    T: HasHrTimer<T>,
    T: for<'a> HrTimerCallback<Pointer<'a> = Self>,
{
    type CallbackTarget<'a> = ArcBorrow<'a, T>;

    unsafe extern "C" fn run(ptr: *mut bindings::hrtimer) -> bindings::hrtimer_restart {
        // `HrTimer` is `repr(C)`
        let timer_ptr = ptr.cast::<super::HrTimer<T>>();

        // SAFETY: By C API contract `ptr` is the pointer we passed when
        // queuing the timer, so it is a `HrTimer<T>` embedded in a `T`.
        let data_ptr = unsafe { T::timer_container_of(timer_ptr) };

        // SAFETY:
        //  - `data_ptr` is derived form the pointer to the `T` that was used to
        //    queue the timer.
        //  - As per the safety requirements of the trait `HrTimerHandle`, the
        //    `ArcHrTimerHandle` associated with this timer is guaranteed to
        //    be alive until this method returns. That handle borrows the `T`
        //    behind `data_ptr` thus guaranteeing the validity of
        //    the `ArcBorrow` created below.
        //  - We own one refcount in the `ArcTimerHandle` associated with this
        //    timer, so it is not possible to get a `UniqueArc` to this
        //    allocation from other `Arc` clones.
        let receiver = unsafe { ArcBorrow::from_raw(data_ptr) };

        // SAFETY:
        // - By C API contract `timer_ptr` is the pointer that we passed when queuing the timer, so
        //   it is a valid pointer to a `HrTimer<T>` embedded in a `T`.
        // - We are within `RawHrTimerCallback::run`
        let context = unsafe { HrTimerCallbackContext::from_raw(timer_ptr) };

        T::run(receiver, context).into_c()
    }
}
