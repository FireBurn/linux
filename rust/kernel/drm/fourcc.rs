// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM fourcc bindings.
//!
//! C header: [`include/uapi/drm/drm_fourcc.h`](srctree/include/uapi/drm/drm_fourcc.h)

/// Return a fourcc format code.
const fn fourcc_code(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | (b as u32) << 8 | (c as u32) << 16 | (d as u32) << 24
}

// TODO: We manually import this because we don't have a reasonable way of getting constants from
// function-like macros in bindgen yet.
pub(crate) const FORMAT_MOD_INVALID: u64 = 0xffffffffffffff;

// TODO: We need to automate importing all of these. For the time being, just add the single one
// that we need

/// 32 bpp RGB
pub const XRGB888: u32 = fourcc_code(b'X', b'R', b'2', b'4');
