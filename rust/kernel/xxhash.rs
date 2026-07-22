// SPDX-License-Identifier: GPL-2.0

//! xxHash, a fast non-cryptographic hash function.
//!
//! C header: [`include/linux/xxhash.h`](srctree/include/linux/xxhash.h)

use crate::bindings;

/// Returns the 64-bit xxHash of `data`, starting from `seed`.
///
/// This is a non-cryptographic hash: use it for change detection, bucketing and similar, never for
/// anything security-relevant.
///
/// Chaining calls by passing the previous result as the next `seed` hashes a sequence of buffers,
/// which lets a caller fingerprint scattered regions -- the rows of an image tile, say -- without
/// first copying them into one contiguous buffer.
///
/// # Examples
///
/// ```
/// use kernel::xxhash::xxh64;
///
/// // The same input and seed always produce the same hash.
/// assert_eq!(xxh64(b"hello", 0), xxh64(b"hello", 0));
///
/// // Different seeds produce different hashes.
/// assert_ne!(xxh64(b"hello", 0), xxh64(b"hello", 1));
///
/// // Hashing the empty slice is well-defined.
/// let _ = xxh64(&[], 0);
/// ```
#[inline]
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    // SAFETY: `data` is a valid slice, so its pointer is valid for reads of `data.len()` bytes,
    // and `xxh64()` only reads from it for the duration of the call.
    unsafe { bindings::xxh64(data.as_ptr().cast(), data.len(), seed) }
}
