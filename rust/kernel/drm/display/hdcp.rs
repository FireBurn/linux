// SPDX-License-Identifier: GPL-2.0 OR MIT

//! High-bandwidth Digital Content Protection definitions.
//!
//! C header: [`include/drm/display/drm_hdcp.h`](srctree/include/drm/display/drm_hdcp.h)

use crate::bindings;

/// Transmitter nonce length.
pub const RTX_LEN: usize = bindings::HDCP_2_2_RTX_LEN as usize;
/// Receiver nonce length.
pub const RRX_LEN: usize = bindings::HDCP_2_2_RRX_LEN as usize;
/// Receiver RSA modulus length.
pub const RSA_MODULUS_LEN: usize = bindings::HDCP_2_2_K_PUB_RX_MOD_N_LEN as usize;
/// Receiver RSA public exponent length.
pub const RSA_EXPONENT_LEN: usize = bindings::HDCP_2_2_K_PUB_RX_EXP_E_LEN as usize;
/// Encrypted master-key length.
pub const ENCRYPTED_MASTER_KEY_LEN: usize = bindings::HDCP_2_2_E_KPUB_KM_LEN as usize;
/// H-prime verifier length.
pub const H_PRIME_LEN: usize = bindings::HDCP_2_2_H_PRIME_LEN as usize;
/// Locality-check nonce length.
pub const RN_LEN: usize = bindings::HDCP_2_2_RN_LEN as usize;
/// L-prime verifier length.
pub const L_PRIME_LEN: usize = bindings::HDCP_2_2_L_PRIME_LEN as usize;
/// Encrypted session-key length.
pub const ENCRYPTED_SESSION_KEY_LEN: usize = bindings::HDCP_2_2_E_DKEY_KS_LEN as usize;
/// Session-key nonce length.
pub const RIV_LEN: usize = bindings::HDCP_2_2_RIV_LEN as usize;
/// One half of the repeater V-prime verifier.
pub const V_PRIME_HALF_LEN: usize = bindings::HDCP_2_2_V_PRIME_HALF_LEN as usize;

/// An HDCP 2.2 protocol message identifier.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MessageId(u8);

impl MessageId {
    /// No message.
    pub const NULL: Self = Self(bindings::HDCP_2_2_NULL_MSG as u8);
    /// Authentication and Key Exchange initialization.
    pub const AKE_INIT: Self = Self(bindings::HDCP_2_2_AKE_INIT as u8);
    /// Receiver certificate.
    pub const AKE_SEND_CERT: Self = Self(bindings::HDCP_2_2_AKE_SEND_CERT as u8);
    /// Encrypted master key for a receiver without stored pairing information.
    pub const AKE_NO_STORED_KM: Self = Self(bindings::HDCP_2_2_AKE_NO_STORED_KM as u8);
    /// Encrypted master key and pairing nonce for a paired receiver.
    pub const AKE_STORED_KM: Self = Self(bindings::HDCP_2_2_AKE_STORED_KM as u8);
    /// Receiver's H-prime authentication value.
    pub const AKE_SEND_H_PRIME: Self = Self(bindings::HDCP_2_2_AKE_SEND_HPRIME as u8);
    /// Receiver pairing information.
    pub const AKE_SEND_PAIRING_INFO: Self = Self(bindings::HDCP_2_2_AKE_SEND_PAIRING_INFO as u8);
    /// Locality-check initialization.
    pub const LC_INIT: Self = Self(bindings::HDCP_2_2_LC_INIT as u8);
    /// Receiver's L-prime locality-check value.
    pub const LC_SEND_L_PRIME: Self = Self(bindings::HDCP_2_2_LC_SEND_LPRIME as u8);
    /// Session-key exchange.
    pub const SKE_SEND_EKS: Self = Self(bindings::HDCP_2_2_SKE_SEND_EKS as u8);
    /// Repeater receiver-ID list.
    pub const REPEATERAUTH_SEND_RECEIVERID_LIST: Self =
        Self(bindings::HDCP_2_2_REP_SEND_RECVID_LIST as u8);
    /// Repeater receiver-ID-list acknowledgment.
    pub const REPEATERAUTH_SEND_ACK: Self = Self(bindings::HDCP_2_2_REP_SEND_ACK as u8);
    /// Repeater stream-management request.
    pub const REPEATERAUTH_STREAM_MANAGE: Self = Self(bindings::HDCP_2_2_REP_STREAM_MANAGE as u8);
    /// Repeater stream-ready response.
    pub const REPEATERAUTH_STREAM_READY: Self = Self(bindings::HDCP_2_2_REP_STREAM_READY as u8);

    /// Return the wire value.
    pub const fn as_u8(self) -> u8 {
        self.0
    }
}

impl From<MessageId> for u8 {
    fn from(value: MessageId) -> Self {
        value.as_u8()
    }
}
