// SPDX-License-Identifier: GPL-2.0
//!
//! DRM display modes.
//!
//! C header: [`include/drm/drm_modes.h`](srctree/include/drm/drm_modes.h)

use bindings;

use crate::types::Opaque;

/// DRM kernel-internal display mode structure.
///
/// This structure contains various resolution and timing information for a given display mode in
/// DRM.
///
/// # Invariants
///
/// - The data layout of this structure is guaranteed to be equivalent to that of `struct
///   drm_display_mode`.
/// - We ensure through our bindings that rust's data aliasing rules are maintained, ensuring it is
///   safe to read any fields inside of `self.inner`.
#[repr(transparent)]
pub struct DisplayMode {
    inner: Opaque<bindings::drm_display_mode>,
}

// SAFETY: Our bindings are thread-safe via our type invariants.
unsafe impl Send for DisplayMode {}
// SAFETY: Our bindings are thread-safe via our type invariants.
unsafe impl Sync for DisplayMode {}

impl DisplayMode {
    /// Convert a raw pointer to a `struct drm_display_mode` into an immutable [`DisplayMode`] ref.
    ///
    /// # SAFETY
    ///
    /// - The caller guarantees that `self_ptr` points to a valid initialized `struct
    ///   drm_display_mode`.
    /// - The caller must ensure that rust's data aliasing rules will not be broken for the lifetime
    ///   of `'a`, e.g. no mutable references may exist while immutable references exist to Self.
    #[inline]
    pub(crate) unsafe fn as_ref<'a>(self_ptr: *const bindings::drm_display_mode) -> &'a Self {
        // SAFETY: The pointer is valid via our safety contract, and the data layout of this struct
        // is equivalent to `Self` via our type invariants.
        unsafe { &*self_ptr.cast() }
    }

    /// Return a raw pointer to the `struct drm_display_mode` contained within this [`DisplayMode`].
    #[inline]
    pub(crate) fn as_raw(&self) -> *const bindings::drm_display_mode {
        self.inner.get().cast_const()
    }

    /// Retrieve the pixel clock for the adjusted display mode in kHz.
    #[inline]
    pub fn crtc_clock(&self) -> i32 {
        // SAFETY: Reading these fields is safe via our type invariants
        unsafe { (*self.as_raw()).crtc_clock }
    }

    /// Retrieve the start of the vertical sync period for the adjusted display mode.
    #[inline]
    pub fn crtc_vblank_start(&self) -> u16 {
        unsafe { (*self.as_raw()).crtc_vblank_start }
    }

    /// Retrieve the end of the vertical sync period for the adjusted display mode.
    #[inline]
    pub fn crtc_vblank_end(&self) -> u16 {
        // SAFETY: Reading these fields is safe via our type invariants
        unsafe { (*self.as_raw()).crtc_vblank_end }
    }

    /// Retrieve the number of vertical scanlines for a full scanout frame in this adjusted display
    /// mode.
    #[inline]
    pub fn crtc_vtotal(&self) -> u16 {
        // SAFETY: Reading these fields is safe via our type invariants
        unsafe { (*self.as_raw()).crtc_vtotal }
    }

    /// Return the horizontal active pixels.
    #[inline]
    pub fn hdisplay(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).hdisplay }
    }

    /// Return the start of the horizontal sync pulse.
    #[inline]
    pub fn hsync_start(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).hsync_start }
    }

    /// Return the end of the horizontal sync pulse.
    #[inline]
    pub fn hsync_end(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).hsync_end }
    }

    /// Return the total horizontal pixels including blanking.
    #[inline]
    pub fn htotal(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).htotal }
    }

    /// Return the vertical active scanlines.
    #[inline]
    pub fn vdisplay(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).vdisplay }
    }

    /// Return the start of the vertical sync pulse.
    #[inline]
    pub fn vsync_start(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).vsync_start }
    }

    /// Return the end of the vertical sync pulse.
    #[inline]
    pub fn vsync_end(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).vsync_end }
    }

    /// Return the total vertical scanlines including blanking.
    #[inline]
    pub fn vtotal(&self) -> u16 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).vtotal }
    }

    /// Return the pixel clock in kHz.
    #[inline]
    pub fn clock(&self) -> i32 {
        // SAFETY: Reading this field is safe via the type invariants.
        unsafe { (*self.as_raw()).clock }
    }

    /// Return the refresh rate in Hz as computed by DRM.
    #[inline]
    pub fn vrefresh(&self) -> i32 {
        // SAFETY: `drm_mode_vrefresh` only reads this valid display mode.
        unsafe { bindings::drm_mode_vrefresh(self.as_raw()) }
    }
}
