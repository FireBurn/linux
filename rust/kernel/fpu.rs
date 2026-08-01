// SPDX-License-Identifier: GPL-2.0

//! Kernel-mode FPU and vector register sections.
//!
//! C header: [`arch/x86/include/asm/fpu/api.h`](srctree/arch/x86/include/asm/fpu/api.h)
//!
//! The kernel does not preserve FPU or vector state across context switches, so code that wants to
//! use them has to say so. [`FpuGuard`] marks such a section: the registers become usable when it
//! is constructed and are restored when it is dropped.

use crate::bindings;

/// The 387 state will be initialized.
pub const KFPU_387: u32 = 1 << 0;
/// `MXCSR` will be initialized.
pub const KFPU_MXCSR: u32 = 1 << 1;

/// A kernel-mode FPU section.
///
/// While one exists, the FPU and vector registers may be used on this CPU. Dropping it restores
/// whatever state was displaced.
///
/// # Constraints
///
/// The section runs with preemption disabled, so it must be **short and must not sleep**: no
/// allocation that can block, no locks that can sleep, no I/O. Keep it around a bounded arithmetic
/// kernel and nothing else.
///
/// The guard is deliberately neither [`Send`] nor [`Sync`]: the section belongs to the CPU it was
/// opened on, so it cannot be moved to another task or shared.
///
/// # Examples
///
/// ```
/// # use kernel::fpu::{FpuGuard, KFPU_MXCSR};
/// fn sum_vectorized(data: &[i32]) -> i32 {
///     let _fpu = FpuGuard::new(KFPU_MXCSR);
///     // SAFETY: the guard is live, so vector registers may be used here.
///     unsafe { do_the_simd_thing(data) }
/// }
/// # unsafe fn do_the_simd_thing(_d: &[i32]) -> i32 { 0 }
/// ```
pub struct FpuGuard {
    /// Ties the guard to the current thread and CPU: no `Send`, no `Sync`.
    _not_send: crate::types::NotThreadSafe,
}

impl FpuGuard {
    /// Open a kernel-mode FPU section.
    ///
    /// `mask` selects which state the caller will initialize itself, as
    /// [`KFPU_387`] and [`KFPU_MXCSR`]. Pass `0` when the code inside touches neither, which is the
    /// case for integer vector work.
    #[inline]
    pub fn new(mask: u32) -> Self {
        // SAFETY: FFI call with no preconditions beyond being in process context, which the
        // sleeping restriction documented on this type already requires of the caller. The
        // matching `kernel_fpu_end()` is guaranteed by `Drop`.
        unsafe { bindings::kernel_fpu_begin_mask(mask) };
        Self {
            _not_send: crate::types::NotThreadSafe,
        }
    }
}

impl Drop for FpuGuard {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: a guard exists only between a successful `kernel_fpu_begin_mask()` and its
        // matching end, and `new()` is the only constructor.
        unsafe { bindings::kernel_fpu_end() };
    }
}
