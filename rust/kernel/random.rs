// SPDX-License-Identifier: GPL-2.0

//! Random number generation.
//!
//! C header: [`include/linux/random.h`](srctree/include/linux/random.h)

use crate::bindings;

/// Fills `buf` with cryptographically secure random bytes from the kernel's CSPRNG.
///
/// This is the in-kernel equivalent of reading from `/dev/urandom`, and is suitable for generating
/// keys, nonces and other secrets. It never blocks: once the CSPRNG has been seeded during boot it
/// stays seeded, and callers running that early should use
/// [`wait_for_random_bytes()`] instead of assuming otherwise.
///
/// [`wait_for_random_bytes()`]: srctree/include/linux/random.h
///
/// # Examples
///
/// ```
/// use kernel::random;
///
/// let mut key = [0u8; 16];
/// random::fill_bytes(&mut key);
///
/// // A zero-length request is valid and does nothing.
/// random::fill_bytes(&mut []);
/// ```
#[inline]
pub fn fill_bytes(buf: &mut [u8]) {
    // SAFETY: `buf` is a valid slice, so its pointer is valid for writes of `buf.len()` bytes, and
    // `get_random_bytes()` writes exactly that many.
    unsafe { bindings::get_random_bytes(buf.as_mut_ptr().cast(), buf.len()) };
}
