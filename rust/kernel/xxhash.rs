// SPDX-License-Identifier: GPL-2.0

//! xxHash, a fast non-cryptographic hash function.
//!
//! C header: [`include/linux/xxhash.h`](srctree/include/linux/xxhash.h)

use core::mem::MaybeUninit;

use crate::{bindings, error::to_result, prelude::*};

/// Incremental xxHash64 state.
///
/// Use this when the input is split across several buffers. Calling
/// [`update`](Self::update) for each buffer produces the same digest as hashing
/// their concatenation with [`xxh64`].
pub struct Xxh64(bindings::xxh64_state);

impl Xxh64 {
    /// Start a new hash with `seed`.
    pub fn new(seed: u64) -> Self {
        let mut state = MaybeUninit::uninit();
        // SAFETY: `xxh64_reset()` initializes every byte of the state before
        // returning.
        unsafe {
            bindings::xxh64_reset(state.as_mut_ptr(), seed);
            Self(state.assume_init())
        }
    }

    /// Add `data` to the hash.
    pub fn update(&mut self, data: &[u8]) -> Result {
        // SAFETY: `self.0` is initialized and exclusively borrowed; `data` is
        // valid for reads of `data.len()` bytes.
        to_result(unsafe { bindings::xxh64_update(&mut self.0, data.as_ptr().cast(), data.len()) })
    }

    /// Return the current digest without consuming the state.
    pub fn digest(&self) -> u64 {
        // SAFETY: `self.0` was initialized by `xxh64_reset()` and remains live.
        unsafe { bindings::xxh64_digest(&self.0) }
    }
}

/// Returns the 64-bit xxHash of `data`, starting from `seed`.
///
/// This is a non-cryptographic hash: use it for change detection, bucketing and similar, never for
/// anything security-relevant.
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
