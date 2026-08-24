// SPDX-License-Identifier: GPL-2.0

//! Cryptographically-secure randomness for the per-session HDCP nonces/keys
//! (`rtx`, `km`, `rn`, `ks`, `riv`, the OAEP seed).

/// Fills `buf` with random bytes from the kernel CSPRNG.
pub(super) fn fill(buf: &mut [u8]) {
    kernel::random::fill_bytes(buf);
}
