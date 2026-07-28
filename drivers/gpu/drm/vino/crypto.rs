// SPDX-License-Identifier: GPL-2.0

//! Thin adapters onto the shared [`kernel::crypto`] library-crypto bindings, so the
//! protocol code keeps its `crypto::aes128_ecb` / `crypto::hmac_sha256` call sites.

use super::*;

/// An AES-128 key prepared once for single-block encryption; re-exported so the
/// AES-CTR keystream loops in [`cp`](super::cp) can expand the key once and reuse
/// it across every block (rather than re-expanding per block).
pub(super) use kernel::crypto::Aes128;

/// `AES_ECB(key, block)` -- one 16-byte AES-128 block. Convenience one-shot for
/// callers that encrypt a single block (e.g. HDCP dKey derivation); the AES-CTR
/// paths build an [`Aes128`] once and call [`Aes128::encrypt_block`] in a loop.
pub(super) fn aes128_ecb(key: &[u8; 16], block: &[u8; 16]) -> Result<[u8; 16]> {
    Ok(Aes128::new(key)?.encrypt_block(block))
}

/// `HMAC-SHA256(key, data)`.
pub(super) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    kernel::crypto::hmac_sha256(key, data)
}

/// `AES-CMAC-128(key, data)` (RFC 4493) via the in-tree AES-CMAC library.
/// This is DisplayLink's "Dl3Cmac" core -- the CP per-message integrity tag is
/// `AES_CMAC(ks, nonce8 || BE64(counter) || content)` (see `cp::dl3cmac_tag`).
pub(super) fn aes_cmac(key: &[u8; 16], data: &[u8]) -> [u8; 16] {
    kernel::crypto::aes_cmac(key, data)
}

/// `SHA256(data)`.
pub(super) fn sha256(data: &[u8]) -> [u8; 32] {
    kernel::crypto::sha256(data)
}
