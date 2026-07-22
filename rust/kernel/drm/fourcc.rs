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
/// Linear framebuffer layout (`DRM_FORMAT_MOD_LINEAR`).
pub(crate) const FORMAT_MOD_LINEAR: u64 = 0;

/// 32 bpp RGB with unused alpha.
pub const XRGB8888: u32 = fourcc_code(b'X', b'R', b'2', b'4');

/// 32 bpp RGB with alpha.
pub const ARGB8888: u32 = fourcc_code(b'A', b'R', b'2', b'4');

/// 32 bpp BGR with unused alpha.
pub const XBGR8888: u32 = fourcc_code(b'X', b'B', b'2', b'4');

/// 32 bpp BGR with alpha.
pub const ABGR8888: u32 = fourcc_code(b'A', b'B', b'2', b'4');

/// 30 bpp 10:10:10 RGB with unused alpha.
pub const XRGB2101010: u32 = fourcc_code(b'X', b'R', b'3', b'0');

/// 30 bpp 10:10:10 RGB with alpha.
pub const ARGB2101010: u32 = fourcc_code(b'A', b'R', b'3', b'0');

/// 30 bpp 10:10:10 BGR with unused alpha.
pub const XBGR2101010: u32 = fourcc_code(b'X', b'B', b'3', b'0');

/// 30 bpp 10:10:10 BGR with alpha.
pub const ABGR2101010: u32 = fourcc_code(b'A', b'B', b'3', b'0');
