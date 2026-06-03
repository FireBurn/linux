// SPDX-License-Identifier: GPL-2.0

//! Safe wrappers over the kernel's synchronous library crypto.
//!
//! Exposes the `lib/crypto` primitives -- AES-128 (an [`Aes128`] key expanded
//! once, for single blocks and for AES-CTR), the in-tree AES-CMAC
//! ([`aes_cmac`]), SHA-256 and HMAC-SHA256 -- for use from Rust. They run
//! synchronously in the calling context with no allocation; the hashes and the
//! MAC are infallible.
//!
//! C headers: [`include/crypto/aes.h`](srctree/include/crypto/aes.h),
//! [`include/crypto/aes-ctr.h`](srctree/include/crypto/aes-ctr.h),
//! [`include/crypto/aes-cbc-macs.h`](srctree/include/crypto/aes-cbc-macs.h),
//! [`include/crypto/sha2.h`](srctree/include/crypto/sha2.h).

use crate::bindings;
#[cfg(CONFIG_RUST_CRYPTO_LIB_AES)]
use crate::{error::to_result, prelude::*};

/// Size of a SHA-256 / HMAC-SHA256 digest, in bytes.
pub const SHA256_DIGEST_SIZE: usize = 32;
/// AES-128 block and key size, in bytes.
pub const AES128_BLOCK_SIZE: usize = 16;

/// Overwrites `bytes` with zeroes so that the compiler cannot elide the write.
///
/// Rust has no equivalent of `memzero_explicit()`, so this forwards to the C one.
#[inline]
pub fn zeroize(bytes: &mut [u8]) {
    // SAFETY: `bytes` is valid for `bytes.len()` writes.
    unsafe { bindings::memzero_explicit(bytes.as_mut_ptr().cast(), bytes.len()) };
}

/// Returns the SHA-256 digest of `data`.
#[cfg(CONFIG_RUST_CRYPTO_LIB_SHA256)]
#[inline]
pub fn sha256(data: &[u8]) -> [u8; SHA256_DIGEST_SIZE] {
    let mut out = [0u8; SHA256_DIGEST_SIZE];
    // SAFETY: `data` is valid for `data.len()` reads and `out` is a valid
    // `SHA256_DIGEST_SIZE`-byte output buffer, as `sha256()` requires.
    unsafe { bindings::sha256(data.as_ptr(), data.len(), out.as_mut_ptr()) };
    out
}

/// Returns `HMAC-SHA256(key, data)`.
#[cfg(CONFIG_RUST_CRYPTO_LIB_SHA256)]
#[inline]
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; SHA256_DIGEST_SIZE] {
    let mut out = [0u8; SHA256_DIGEST_SIZE];
    // SAFETY: `key` and `data` are valid for their respective lengths and `out`
    // is a valid `SHA256_DIGEST_SIZE`-byte output buffer, as required.
    unsafe {
        bindings::hmac_sha256_usingrawkey(
            key.as_ptr(),
            key.len(),
            data.as_ptr(),
            data.len(),
            out.as_mut_ptr(),
        )
    };
    out
}

/// Returns `AES-CMAC-128(key, data)` (RFC 4493), computed by the in-tree
/// AES-CMAC library ([`include/crypto/aes-cbc-macs.h`]). The 128-bit key is
/// prepared and wiped internally; the call is infallible.
///
/// [`include/crypto/aes-cbc-macs.h`]: srctree/include/crypto/aes-cbc-macs.h
#[cfg(CONFIG_RUST_CRYPTO_LIB_AES)]
#[inline]
pub fn aes_cmac(key: &[u8; AES128_BLOCK_SIZE], data: &[u8]) -> [u8; AES128_BLOCK_SIZE] {
    let mut out = [0u8; AES128_BLOCK_SIZE];
    // SAFETY: `key` is a valid 16-byte key, `data` is valid for `data.len()`
    // reads, and `out` is a valid `AES128_BLOCK_SIZE`-byte output buffer, as the
    // helper requires.
    unsafe { bindings::aes_cmac(key.as_ptr(), data.as_ptr(), data.len(), out.as_mut_ptr()) };
    out
}

/// An AES-128 key, expanded once and reused.
///
/// Expanding a key is much more expensive than a block, so a caller that uses
/// one key for many messages should keep the expanded form rather than pass raw
/// key bytes to a one-shot. [`ctr`](Aes128::ctr) and
/// [`encrypt_block`](Aes128::encrypt_block) both run on the schedule computed
/// in [`Aes128::new`].
///
/// # Examples
///
/// ```
/// use kernel::crypto::Aes128;
/// let cipher = Aes128::new(&[0u8; 16])?;
/// let _ct = cipher.encrypt_block(&[0u8; 16]);
///
/// let mut counter = [0u8; 16];
/// let mut data = *b"sixteen bytes...";
/// let original = data;
/// cipher.ctr(&mut counter, &mut data);
/// assert_ne!(data, original);
///
/// let mut counter = [0u8; 16];
/// cipher.ctr(&mut counter, &mut data);
/// assert_eq!(data, original);
/// # Ok::<(), Error>(())
/// ```
#[cfg(CONFIG_RUST_CRYPTO_LIB_AES)]
pub struct Aes128(bindings::aes_enckey);

#[cfg(CONFIG_RUST_CRYPTO_LIB_AES)]
impl Aes128 {
    /// Expands an AES-128 key from 16 raw key bytes.
    #[inline]
    pub fn new(key: &[u8; AES128_BLOCK_SIZE]) -> Result<Self> {
        // SAFETY: `aes_enckey` is a plain-old-data key schedule (integer arrays
        // in a union of integer arrays); an all-zero bit pattern is a valid,
        // inert initial value, fully overwritten by `aes_prepareenckey()` below.
        let mut enckey: bindings::aes_enckey = unsafe { core::mem::zeroed() };
        // SAFETY: `enckey` is a valid, owned `aes_enckey`; `key` is a valid
        // 16-byte buffer; `AES128_BLOCK_SIZE` (16) is a supported key length.
        let ret =
            unsafe { bindings::aes_prepareenckey(&mut enckey, key.as_ptr(), AES128_BLOCK_SIZE) };
        to_result(ret)?;
        Ok(Self(enckey))
    }

    /// Encrypts one 16-byte block with the prepared key: returns
    /// `AES-128-ECB(key, block)`.
    #[inline]
    pub fn encrypt_block(&self, block: &[u8; AES128_BLOCK_SIZE]) -> [u8; AES128_BLOCK_SIZE] {
        let mut out = [0u8; AES128_BLOCK_SIZE];
        // SAFETY: `self.0` is a prepared encryption key; `block` and `out` are
        // valid 16-byte buffers, as the helper requires.
        unsafe { bindings::aes_enckey_encrypt_block(&self.0, out.as_mut_ptr(), block.as_ptr()) };
        out
    }

    /// Encrypts or decrypts `data` in place with AES-CTR under the prepared key.
    ///
    /// `counter` is the initial counter block and is advanced past `data`, so a
    /// caller can continue one keystream across several calls. It is a full
    /// 128-bit big-endian counter: a protocol carrying a narrower counter field
    /// in the low bytes matches this only until that field would carry into the
    /// bytes above it.
    ///
    /// Encryption and decryption are the same operation.
    #[inline]
    pub fn ctr(&self, counter: &mut [u8; AES128_BLOCK_SIZE], data: &mut [u8]) {
        // SAFETY: `self.0` is a prepared encryption key, `counter` is a valid
        // 16-byte block valid for reads and writes, and `data` is valid for
        // `data.len()` reads and writes, as the helper requires.
        unsafe {
            bindings::aes_enckey_ctr(&self.0, counter.as_mut_ptr(), data.as_mut_ptr(), data.len())
        };
    }
}

#[cfg(CONFIG_RUST_CRYPTO_LIB_AES)]
impl Drop for Aes128 {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `aes_enckey` is a plain-old-data key schedule (integer arrays in
        // a union of integer arrays), so its storage is valid to overwrite as
        // bytes, and `self.0` is owned and about to be dropped.
        let key = unsafe {
            core::slice::from_raw_parts_mut(
                (&raw mut self.0).cast::<u8>(),
                core::mem::size_of::<bindings::aes_enckey>(),
            )
        };
        zeroize(key);
    }
}
