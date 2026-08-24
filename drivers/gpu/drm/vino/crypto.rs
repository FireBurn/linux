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

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
#[kunit_tests(vino_crypto)]
mod tests {
    use super::*;

    #[test]
    fn aes128_ecb_fips197_kat() -> Result {
        // FIPS-197 / NIST SP800-38A F.1.1 AES-128 ECB known-answer vector.
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let plaintext = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
            0x17, 0x2a,
        ];
        assert_eq!(
            aes128_ecb(&key, &plaintext)?,
            [
                0x3a, 0xd7, 0x7b, 0xb4, 0x0d, 0x7a, 0x36, 0x60, 0xa8, 0x9e, 0xca, 0xf3, 0x24, 0x66,
                0xef, 0x97,
            ]
        );
        Ok(())
    }

    #[test]
    fn aes_cmac_rfc4493_kat() -> Result {
        // RFC 4493 sec 4 AES-CMAC test vectors (same key as above).
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        assert_eq!(
            aes_cmac(&key, &[]),
            [
                0xbb, 0x1d, 0x69, 0x29, 0xe9, 0x59, 0x37, 0x28, 0x7f, 0xa3, 0x7d, 0x12, 0x9b, 0x75,
                0x67, 0x46,
            ]
        );
        let msg = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
            0x17, 0x2a,
        ];
        assert_eq!(
            aes_cmac(&key, &msg),
            [
                0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44, 0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a,
                0x28, 0x7c,
            ]
        );
        Ok(())
    }
}
