// SPDX-License-Identifier: GPL-2.0
//!
//! DRM display modes.
//!
//! C header: [`include/drm/drm_modes.h`](srctree/include/drm/drm_modes.h)

use bindings;

use crate::{
    error::{code::EINVAL, Result},
    types::Opaque,
};

/// Flags describing signal polarity and scan format for a display mode.
///
/// These correspond to the `DRM_MODE_FLAG_*` values accepted by the DRM mode helpers.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct ModeFlags(u32);

impl ModeFlags {
    /// Horizontal sync is active high.
    pub const PHSYNC: Self = Self(bindings::DRM_MODE_FLAG_PHSYNC);
    /// Horizontal sync is active low.
    pub const NHSYNC: Self = Self(bindings::DRM_MODE_FLAG_NHSYNC);
    /// Vertical sync is active high.
    pub const PVSYNC: Self = Self(bindings::DRM_MODE_FLAG_PVSYNC);
    /// Vertical sync is active low.
    pub const NVSYNC: Self = Self(bindings::DRM_MODE_FLAG_NVSYNC);
    /// The mode is interlaced.
    pub const INTERLACE: Self = Self(bindings::DRM_MODE_FLAG_INTERLACE);
    /// The mode uses doublescan.
    pub const DBLSCAN: Self = Self(bindings::DRM_MODE_FLAG_DBLSCAN);
    /// The mode uses composite sync.
    pub const CSYNC: Self = Self(bindings::DRM_MODE_FLAG_CSYNC);
    /// Composite sync is active high.
    pub const PCSYNC: Self = Self(bindings::DRM_MODE_FLAG_PCSYNC);
    /// Composite sync is active low.
    pub const NCSYNC: Self = Self(bindings::DRM_MODE_FLAG_NCSYNC);
    /// The mode carries a horizontal skew value.
    pub const HSKEW: Self = Self(bindings::DRM_MODE_FLAG_HSKEW);
    /// The mode is double-clocked.
    pub const DBLCLK: Self = Self(bindings::DRM_MODE_FLAG_DBLCLK);
    /// The mode uses a half-rate clock.
    pub const CLKDIV2: Self = Self(bindings::DRM_MODE_FLAG_CLKDIV2);

    /// Return whether all flags in `other` are set.
    pub fn contains(self, other: Self) -> bool {
        self & other == other
    }
}

impl core::ops::BitOr for ModeFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitAnd for ModeFlags {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// The essential timing fields of a display mode.
#[derive(Clone, Copy)]
pub struct ModeTimings {
    /// Pixel clock in kHz.
    pub clock_khz: i32,
    /// Horizontal active pixels.
    pub hdisplay: u16,
    /// Start of the horizontal sync pulse.
    pub hsync_start: u16,
    /// End of the horizontal sync pulse.
    pub hsync_end: u16,
    /// Total horizontal pixels including blanking.
    pub htotal: u16,
    /// Vertical active lines.
    pub vdisplay: u16,
    /// Start of the vertical sync pulse.
    pub vsync_start: u16,
    /// End of the vertical sync pulse.
    pub vsync_end: u16,
    /// Total vertical lines including blanking.
    pub vtotal: u16,
    /// Signal polarity and scan-format flags.
    pub flags: ModeFlags,
}

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
    /// Creates a standalone display mode from validated timings.
    ///
    /// This is useful when a driver needs an owned mode for validation or tests rather than a
    /// reference to a mode owned by the DRM core.
    pub fn from_timings(t: ModeTimings) -> Result<Self> {
        if t.clock_khz <= 0
            || t.hdisplay == 0
            || t.hdisplay > t.hsync_start
            || t.hsync_start > t.hsync_end
            || t.hsync_end > t.htotal
            || t.vdisplay == 0
            || t.vdisplay > t.vsync_start
            || t.vsync_start > t.vsync_end
            || t.vsync_end > t.vtotal
        {
            return Err(EINVAL);
        }

        let mut mode = bindings::drm_display_mode::default();
        mode.clock = t.clock_khz;
        mode.hdisplay = t.hdisplay;
        mode.hsync_start = t.hsync_start;
        mode.hsync_end = t.hsync_end;
        mode.htotal = t.htotal;
        mode.vdisplay = t.vdisplay;
        mode.vsync_start = t.vsync_start;
        mode.vsync_end = t.vsync_end;
        mode.vtotal = t.vtotal;
        mode.flags = t.flags.0;

        Ok(Self {
            inner: Opaque::new(mode),
        })
    }

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

    /// Return the mode's signal-polarity and scan-format flags.
    #[inline]
    pub fn flags(&self) -> ModeFlags {
        // SAFETY: Reading this field is safe via the type invariants.
        ModeFlags(unsafe { (*self.as_raw()).flags })
    }

    /// Return the refresh rate in Hz as computed by DRM.
    #[inline]
    pub fn vrefresh(&self) -> i32 {
        // SAFETY: `drm_mode_vrefresh` only reads this valid display mode.
        unsafe { bindings::drm_mode_vrefresh(self.as_raw()) }
    }

    /// Return the CTA-861 Video Identification Code matching this mode.
    ///
    /// A return value of zero means the mode is not one of the CTA-861 modes known to DRM.
    #[inline]
    pub fn cea_vic(&self) -> u8 {
        // SAFETY: `drm_match_cea_mode` only reads this valid display mode.
        unsafe { bindings::drm_match_cea_mode(self.as_raw()) }
    }
}
