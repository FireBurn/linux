// SPDX-License-Identifier: GPL-2.0

//! KUnit coverage for the parts of the protocol that can be checked without hardware: the CP seal
//! and its nonces, the HDCP key derivations, mode-set geometry against the decrypted DLM corpus,
//! and the codec's framing invariants.

use super::*;

use super::*;
use kernel::drm::kms::modes::{DisplayMode, ModeFlags, ModeTimings};
use kernel::error::code::EINVAL;

/// S31.32 sign-magnitude constants: 1.0, +0.5, and -0.5 as sign bit + magnitude.
const CTM_ONE: u64 = 1 << 32;
const CTM_HALF: u64 = 1 << 31;
const CTM_NEG_HALF: u64 = (1u64 << 63) | (1u64 << 31);

fn ctm_diag(r: u64, g: u64, b: u64) -> kernel::drm::kms::crtc::ColorCtm {
    kernel::drm::kms::crtc::ColorCtm::from_raw([r, 0, 0, 0, g, 0, 0, 0, b])
}

/// A ramp that halves every channel, at the LUT's full 16-bit precision. The `+ 1` rounds:
/// entry 255 is 65535/2 = 32767.5, and truncating it would make the fixture itself ask for
/// 127 rather than 128.
fn half_lut() -> KVec<kernel::drm::kms::crtc::ColorLut> {
    let mut v = KVec::new();
    for i in 0..color::LUT_LEN {
        let h = ((i * 257 + 1) / 2) as u16;
        let _ = v.push(kernel::drm::kms::crtc::ColorLut::new(h, h, h), GFP_KERNEL);
    }
    v
}

#[test]
fn ctm_decodes_sign_magnitude_not_twos_complement() -> Result {
    // The UAPI encodes CTM entries in sign-magnitude. Reading the u64 as an i64 would make
    // -0.5 come back as a huge positive number and saturate instead of darkening.
    let m = ctm_diag(CTM_ONE, CTM_NEG_HALF, CTM_ONE);
    assert_eq!(m.coefficient(0), Some(1i64 << 32));
    assert_eq!(m.coefficient(4), Some(-(1i64 << 31)));
    assert_eq!(m.coefficient(9), None);
    Ok(())
}

#[test]
fn identity_transform_builds_nothing() -> Result {
    // Turning a corrector off programs an identity matrix rather than removing the blob. If
    // that did not collapse to None the encoder would never regain its direct-scanout path.
    assert!(color::ColorPipeline::build(None, None).is_none());
    let ident = ctm_diag(CTM_ONE, CTM_ONE, CTM_ONE);
    assert!(color::ColorPipeline::build(None, Some(&ident)).is_none());
    Ok(())
}

#[test]
fn identity_gamma_ramp_is_a_no_op() -> Result {
    // The reason `narrow` divides by 257 and not 256. With the wrong divisor every value above
    // about 128 came back one level high, so merely *enabling* colour management shifted the
    // whole image even when the ramp asked for nothing.
    let mut lut = KVec::new();
    for i in 0..color::LUT_LEN {
        let v = (i * 257) as u16;
        let _ = lut.push(kernel::drm::kms::crtc::ColorLut::new(v, v, v), GFP_KERNEL);
    }
    let p = color::ColorPipeline::build(Some(&lut), None).ok_or(EINVAL)?;
    for v in 0..=255u8 {
        assert_eq!(p.apply(v, v, v), (v, v, v));
    }
    Ok(())
}

#[test]
fn gamma_only_applies_the_ramp() -> Result {
    let lut = half_lut();
    let p = color::ColorPipeline::build(Some(&lut), None).ok_or(EINVAL)?;
    assert_eq!(p.apply(0, 0, 0), (0, 0, 0));
    assert_eq!(p.apply(255, 255, 255), (128, 128, 128));
    Ok(())
}

#[test]
fn diagonal_ctm_matches_the_general_matrix() -> Result {
    // The diagonal fast path exists for speed; if it ever disagreed with the general path the
    // colour would silently change with the optimisation rather than with the CTM.
    let fast =
        color::ColorPipeline::build(None, Some(&ctm_diag(CTM_ONE, CTM_HALF, CTM_ONE)))
            .ok_or(EINVAL)?;
    // The same transform with a real off-diagonal zero-effect term, so it must take the
    // mixing path. A sub-Q16 term would be truncated to zero and stay on the fast path.
    let mixed = kernel::drm::kms::crtc::ColorCtm::from_raw([
        CTM_ONE,
        0,
        CTM_ONE / 65536,
        0,
        CTM_HALF,
        0,
        0,
        0,
        CTM_ONE,
    ]);
    let slow = color::ColorPipeline::build(None, Some(&mixed)).ok_or(EINVAL)?;
    for v in [0u8, 1, 63, 127, 128, 200, 254, 255] {
        assert_eq!(fast.apply(v, v, v), slow.apply(v, v, v));
    }
    assert_eq!(fast.apply(255, 255, 255), (255, 128, 255));
    Ok(())
}

#[test]
fn mixing_ctm_moves_channels() -> Result {
    // Swap red and blue: proves the matrix is row-major and applied the way the UAPI documents.
    let swap = kernel::drm::kms::crtc::ColorCtm::from_raw([
        0, 0, CTM_ONE, 0, CTM_ONE, 0, CTM_ONE, 0, 0,
    ]);
    let p = color::ColorPipeline::build(None, Some(&swap)).ok_or(EINVAL)?;
    assert_eq!(p.apply(200, 100, 50), (50, 100, 200));
    Ok(())
}

#[test]
fn negative_coefficient_clamps_to_black() -> Result {
    let p = color::ColorPipeline::build(None, Some(&ctm_diag(CTM_ONE, CTM_NEG_HALF, CTM_ONE)))
        .ok_or(EINVAL)?;
    assert_eq!(p.apply(255, 255, 255), (255, 0, 255));
    Ok(())
}

#[test]
fn out_of_gamut_saturates_instead_of_wrapping() -> Result {
    // An intermediate above 1.0 must clamp. Wrapping would put the brightest pixels at the
    // opposite corner of the colour cube -- the failure looks like inverted highlights.
    let gain4 = ctm_diag(4 * CTM_ONE, 4 * CTM_ONE, 4 * CTM_ONE);
    let p = color::ColorPipeline::build(None, Some(&gain4)).ok_or(EINVAL)?;
    assert_eq!(p.apply(200, 100, 255), (255, 255, 255));
    assert_eq!(p.apply(0, 0, 0), (0, 0, 0));
    Ok(())
}

#[test]
fn short_lut_extends_with_identity_not_black() -> Result {
    // A LUT blob shorter than the advertised size must not leave the tail at zero, which would
    // render everything above the truncation point black.
    let mut lut = KVec::new();
    for i in 0..4usize {
        let v = (i * 257) as u16;
        let _ = lut.push(kernel::drm::kms::crtc::ColorLut::new(v, v, v), GFP_KERNEL);
    }
    let p = color::ColorPipeline::build(Some(&lut), None).ok_or(EINVAL)?;
    assert_eq!(p.apply(255, 255, 255), (255, 255, 255));
    Ok(())
}

#[test]
fn transform_change_changes_the_strip_cache_tag() -> Result {
    // The encoded-strip cache keys on source pixels, so a transform change that leaves the
    // pixels alone must still invalidate it or the whole screen keeps its old colours.
    let a = color::ColorPipeline::build(None, Some(&ctm_diag(CTM_ONE, CTM_HALF, CTM_ONE)))
        .ok_or(EINVAL)?;
    let b = color::ColorPipeline::build(None, Some(&ctm_diag(CTM_HALF, CTM_ONE, CTM_ONE)))
        .ok_or(EINVAL)?;
    assert_ne!(a.tag(), b.tag());
    // `assert!` rather than `assert_ne!`: the latter needs `Debug`, and deriving it on a type
    // holding 768-entry tables is code the driver would carry purely for one test message.
    assert!(a != b);
    Ok(())
}

#[test]
fn aes128_ecb_fips197_kat() -> Result {
    // FIPS-197 / NIST SP800-38A F.1.1 AES-128 ECB known-answer vector.
    let key = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
        0x4f, 0x3c,
    ];
    let pt = [
        0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
        0x17, 0x2a,
    ];
    assert_eq!(
        crypto::aes128_ecb(&key, &pt)?,
        [
            0x3a, 0xd7, 0x7b, 0xb4, 0x0d, 0x7a, 0x36, 0x60, 0xa8, 0x9e, 0xca, 0xf3, 0x24, 0x66,
            0xef, 0x97,
        ]
    );
    Ok(())
}

#[test]
fn colour_frame_ep08_damage_selects_changed_strips() -> Result {
    // Deterministic gradient source (a plain fn item so it's Copy/reusable across calls).
    fn g(x: usize, y: usize) -> (u8, u8, u8) {
        (
            ((x * 7) & 0xff) as u8,
            ((y * 5) & 0xff) as u8,
            (((x + y) * 3) & 0xff) as u8,
        )
    }
    let total = |fs: &KVec<KVec<u8>>| fs.iter().map(|f| f.len()).sum::<usize>();
    let flat = |fs: &KVec<KVec<u8>>| -> Result<KVec<u8>> {
        let mut v = KVec::new();
        for f in fs.iter() {
            v.extend_from_slice(f, GFP_KERNEL)?;
        }
        Ok(v)
    };
    // Damage granularity is the 256x64 macro-tile (`MACRO_W`/`MACRO_H`), not the 64x16 strip:
    // every strip of a touched macro-tile is resent. Use several macro-tiles so the partial
    // update assertions below can distinguish one selected tile from the full frame.
    //
    // 512x128 = 8 strips wide (512/64) x 8 bands (128/16) = 64 strips
    //         = 2 x 2 macro-tiles, each 4 strips wide x 4 bands = 16 strips.
    let (w, h) = (512usize, 128usize);
    const STRIPS_PER_MACRO: usize = 16;
    let geom = PROFILE_D6000.geometry();
    let (full, _) = video::wht::colour_frame_ep08(geom, w, h, 0, 0, g)?;

    // A damage clip covering the WHOLE surface selects every strip in the same raster order as
    // the full-frame path, so the wire bytes are identical.
    let (dfull, _) =
        video::wht::colour_frame_ep08_damage(geom, w, h, 0, 0, &[(0, 0, w, h)], g)?;
    assert_eq!(flat(&full)?.as_slice(), flat(&dfull)?.as_slice());

    // No damage -> no strips -> empty frame list (caller must skip the USB write).
    let (empty, _) = video::wht::colour_frame_ep08_damage(geom, w, h, 0, 0, &[], g)?;
    assert!(empty.is_empty());

    // Selection is exact and macro-tile-quantised. Assert the strip COUNT directly (the shared
    // selector both encoders use) as well as the byte totals -- a count is a far sharper
    // statement than "smaller than full", and it is what actually pins the tiling behaviour.
    let coords = |clips: &[(usize, usize, usize, usize)]| -> Result<usize> {
        Ok(video::wht::damage_strip_coords(geom, w, h, clips)?.len())
    };
    assert_eq!(coords(&[])?, 0);
    assert_eq!(coords(&[(0, 0, w, h)])?, 4 * STRIPS_PER_MACRO); // all four macro-tiles

    // A 1-pixel clip lands in ONE macro-tile and selects all 16 of its strips -- not 1.
    assert_eq!(coords(&[(1, 1, 2, 2)])?, STRIPS_PER_MACRO);
    let (d1, _) =
        video::wht::colour_frame_ep08_damage(geom, w, h, 0, 0, &[(1, 1, 2, 2)], g)?;
    assert!(!d1.is_empty());
    assert!(total(&d1) < total(&full));

    // A 1-pixel-wide clip down the whole left edge spans the left macro-tile COLUMN: 2 tiles.
    assert_eq!(coords(&[(0, 0, 1, h)])?, 2 * STRIPS_PER_MACRO);
    let (d2, _) =
        video::wht::colour_frame_ep08_damage(geom, w, h, 0, 0, &[(0, 0, 1, h)], g)?;
    assert!(total(&d1) < total(&d2) && total(&d2) < total(&full));

    // Non-aligned geometry is rejected (same contract as colour_frame_ep08).
    assert!(video::wht::colour_frame_ep08_damage(geom, 100, 32, 0, 0, &[(0, 0, 1, 1)], g)
        .is_err());
    Ok(())
}

#[test]
fn black_training_frame_matches_captured_1440p_size() -> Result {
    // Captured first writes are 205,696 bytes:
    // 2,560-byte arm prefix + 203,040-byte black image + 96-byte frame trailer.
    let geom = PROFILE_D6000.geometry();
    let frame = video::wht::black_frame_ep08(geom, 2560, 1440, 0)?;
    let image_len = frame.iter().map(|part| part.len()).sum::<usize>();
    assert_eq!(image_len, 203_040);
    assert_eq!(
        2_560 + image_len + video::wht::frame_trailer(geom, 0, 0).len(),
        205_696
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
        crypto::aes_cmac(&key, &[]),
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
        crypto::aes_cmac(&key, &msg),
        [
            0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44, 0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a,
            0x28, 0x7c,
        ]
    );
    Ok(())
}

#[test]
fn seal_livemac_roundtrip() -> Result {
    // A sealed CP frame must decrypt back to its content under the IN riv, and its
    // appended tag must equal a fresh Dl3Cmac over the ciphertext (encrypt-then-MAC).
    let ks = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
        0xee, 0xff,
    ];
    let riv = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
    let content = [0xa5u8; 32];
    let mut hdr = [0u8; 16];
    hdr[12..16].copy_from_slice(&4u32.to_le_bytes()); // wire_seq = 4
    let frame = cp::seal_livemac(&ks, &riv, &hdr, &content)?;
    assert_eq!(frame.len(), 16 + 32 + 16);
    let body = &frame[16..];
    let ct = &frame[16..16 + 32];
    // `open_in` verifies the appended Dl3Cmac, then applies AES-CTR with the supplied nonce.
    assert_eq!(&cp::open_in(&ks, &riv, 4, body, true)?[..], &content[..]);
    // And pin that contract rather than leaving it implicit: the IN nonce really is different,
    // so both its MAC nonce and content keystream reject this fixture.
    assert_ne!(cp::in_riv(&riv), riv);
    assert!(cp::open_in(&ks, &cp::in_riv(&riv), 4, body, true).is_err());
    assert_eq!(&frame[16 + 32..], &cp::dl3cmac_tag(&ks, &riv, 4, ct)?[..]);

    let mut damaged = KVec::new();
    damaged.extend_from_slice(body, GFP_KERNEL)?;
    let last = damaged.len() - 1;
    damaged[last] ^= 1;
    assert!(cp::open_in(&ks, &riv, 4, &damaged, true).is_err());
    Ok(())
}

#[test]
fn reply_decoders_accept_all_supported_rivs() -> Result {
    let ks = [0x5au8; 16];
    let out_head0 = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
    let in_head0 = cp::in_riv(&out_head0);
    let mut out_head1 = out_head0;
    out_head1[0] ^= 0x80;
    let mut in_head1 = in_head0;
    in_head1[0] ^= 0x80;

    let mut header = [0u8; 16];
    header[8..10].copy_from_slice(&0x45u16.to_le_bytes());
    header[12..16].copy_from_slice(&7u32.to_le_bytes());
    let inner = [0x14, 0, 0x30, 0, 9, 0, 0, 0];

    for riv in [in_head0, in_head1, out_head0, out_head1] {
        let frame = cp::seal_livemac(&ks, &riv, &header, &inner)?;
        assert_eq!(
            cp::verify_in_ack(&ks, &out_head0, &frame, true),
            Some((0x14, 0x30, 9))
        );
        assert_eq!(
            cp::decode_in_lenient(&ks, &out_head0, &frame, true),
            Some((0x14, 0x30, 9))
        );
    }

    // Navarro's connector 2/3 HDCP pushes carry the upper half of their one-hot selector at
    // inner offsets 6--7. They are authenticated messages, not malformed zero-pad headers.
    let selector_push = [0x10, 0, 0x84, 0, 0, 0, 0, 0x80];
    let frame = cp::seal_livemac(&ks, &in_head0, &header, &selector_push)?;
    assert_eq!(
        cp::decode_in_lenient(&ks, &out_head0, &frame, true),
        Some((0x10, 0x84, 0))
    );
    assert_eq!(
        &cp::inner_plaintext(&ks, &out_head0, &frame, true).unwrap()[..],
        &selector_push
    );
    Ok(())
}

#[test]
fn stream_content_nonce_matches_golden_vectors() {
    // Ridge: each head's video stream is `0x08 | head`.
    let h0 = cp::stream_content_nonce(&[0xa1, 0x2b, 0xaa, 0xb7, 0x0e, 0x0b, 0x02, 0x74], 0x08);
    assert_eq!(h0, [0xa1, 0x2b, 0xaa, 0xb7, 0x0e, 0x0b, 0x02, 0x7c]);

    let h1 = cp::stream_content_nonce(&[0xd0, 0x2a, 0xc0, 0x83, 0xb6, 0x42, 0x72, 0x57], 0x09);
    assert_eq!(h1, [0xd0, 0x2a, 0xc0, 0x83, 0xb6, 0x42, 0x72, 0x5e]);

    // Navarro: the RIV each connector's SKE_Send_Eks delivered, and the AES-CTR nonce the
    // dock then expects for that connector's stream.
    let riv = [0x7d, 0x2c, 0xb6, 0x6b, 0x2c, 0xd1, 0x75, 0x7c];
    let link = cp::stream_content_nonce(&riv, 0x04);
    assert_eq!(link, [0x7d, 0x2c, 0xb6, 0x6b, 0x2c, 0xd1, 0x75, 0x78]);

    let c0 = cp::stream_content_nonce(&[0xc3, 0x45, 0xfe, 0x55, 0x93, 0x61, 0x39, 0x01], 0x07);
    assert_eq!(c0, [0xc3, 0x45, 0xfe, 0x55, 0x93, 0x61, 0x39, 0x06]);

    let c1 = cp::stream_content_nonce(&[0x94, 0x46, 0xc8, 0x3d, 0xa5, 0xfa, 0x39, 0xe3], 0x0f);
    assert_eq!(c1, [0x94, 0x46, 0xc8, 0x3d, 0xa5, 0xfa, 0x39, 0xec]);
}

#[test]
fn stream_ids_follow_the_dock_profile() {
    // Each dock's ids come from its own geometry value, so the two cannot interfere --
    // which is the whole reason this is no longer a pair of module-global statics.
    let ridge = PROFILE_D6000.geometry();
    assert_eq!(ridge.stream_id(0), 0x0008);
    assert_eq!(ridge.stream_id(1), 0x0009);

    let navarro = PROFILE_DL7400.geometry();
    assert_eq!(navarro.stream_id(0), 0x0007);
    assert_eq!(navarro.stream_id(1), 0x000f);
    assert_eq!(navarro.stream_id(2), 0x0017);
    assert_eq!(navarro.stream_id(3), 0x001f);

    // And the Ridge values are unchanged by having read the Navarro ones.
    assert_eq!(ridge.stream_id(0), 0x0008);
}

#[test]
fn aux_for_id_constants() {
    // The CP header `aux` field is a per-inner-id constant, not body_len/4.
    assert_eq!(cp::aux_for_id(0x14, 48), 0x0a);
    assert_eq!(cp::aux_for_id(0x15, 32), 0x09);
    assert_eq!(cp::aux_for_id(0x36, 80), 0x08);
    assert_eq!(cp::aux_for_id(0x48, 96), 0x06);
    // Cursor message IDs have fixed auxiliary fields; deriving them as `body_len / 4` would
    // produce 0x0c for all three.
    assert_eq!(cp::aux_for_id(0x1a, 48), 0x04); // cursor move
    assert_eq!(cp::aux_for_id(0x1b, 48), 0x03); // cursor create
    assert_eq!(cp::aux_for_id(0x1c, 48), 0x02); // cursor image
    assert_eq!(cp::aux_for_id(0x99, 40), 10); // unknown id falls back to body_len/4
}

#[test]
fn cp_setup_burst_table_framing() -> Result {
    // Pin the post-msg0 `(aux, body_len)` wire profile. `body_len` includes the encrypted
    // content and its 16-byte Dl3Cmac tag.
    const PER_HEAD_FINGERPRINT: [(u16, usize); 9] = [
        (0x0c, 64),
        (0x0f, 64),
        (0x04, 176),
        (0x0c, 64),
        (0x0c, 80),
        (0x04, 64),
        (0x08, 64),
        (0x0a, 48),
        (0x05, 48),
    ];
    // Finalization bodies contain 32 bytes of content and a 16-byte tag. Keep one fingerprint
    // per step so table growth cannot cause an out-of-bounds test access.
    const FINALIZE_FINGERPRINT: [(u16, usize); 3] = [(0x08, 48), (0x09, 48), (0x08, 48)];
    // Keep the fingerprint table and the step table in lockstep: growing one without the
    // other is exactly the defect above.
    build_assert!(FINALIZE_FINGERPRINT.len() == cp::CP_SETUP_FINALIZE_STEPS.len());

    let ks = [0x5au8; 16];
    let riv = [0x11u8; 8];
    for (i, &(id, _sub, content_len)) in cp::CP_SETUP_PER_HEAD.iter().enumerate() {
        let content = KVec::from_elem(0u8, content_len, GFP_KERNEL)?;
        let frame = cp::seal_interactive(&ks, &riv, id, 0, &content)?;
        let (want_aux, want_body) = PER_HEAD_FINGERPRINT[i];
        assert_eq!(frame.len(), 16 + want_body);
        assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), want_aux);
    }
    for (i, &(id, _sub)) in cp::CP_SETUP_FINALIZE_STEPS.iter().enumerate() {
        let frame = cp::seal_interactive(&ks, &riv, id, 0, &[0u8; 32])?;
        let (want_aux, want_body) = FINALIZE_FINGERPRINT[i];
        assert_eq!(frame.len(), 16 + want_body);
        assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), want_aux);
    }
    Ok(())
}

#[test]
fn stream_manage_restatement_matches_dlm() -> Result {
    // All deterministic fields must match the captured plaintext for both heads. The head
    // marker is at offset 23, the HDCP message ID at offset 27, and the final three u32 fields
    // contain `0`, `1`, and `head + 8`.
    const WANT: [[u8; 40]; 2] = [
        [
            0x26, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x08, 0x00, 0x00, 0x00,
        ],
        [
            0x26, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00,
            0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x09, 0x00, 0x00, 0x00,
        ],
    ];
    for head in 0..2u8 {
        let c = cp::stream_manage_restatement(0, head)?;
        assert_eq!(c.len(), 48);
        // Bytes 4..6 are the live counter (passed as 0 here, so already covered); the last
        // 8 bytes (offset 40..48) are host-random.
        assert_eq!(&c[..40], &WANT[head as usize][..]);
    }
    Ok(())
}

#[test]
fn video_arm_burst_table_framing() -> Result {
    // Pin every video-arm entry's type, sub-ID, auxiliary value, and body length to captured
    // traffic. Head 0 uses the table's base sub-IDs; the builders add one for head 1. The
    // compile-time length check prevents the fixture and production table from drifting.
    const FINGERPRINT_H0: [(u32, u16, u16, usize); 10] = [
        (2, 0x0008, 0x0000, 16),
        (2, 0x0018, 0x0000, 16),
        (4, 0x0008, 0x000a, 16),
        (4, 0x0018, 0x000a, 16),
        (2, 0x0000, 0x0000, 16),
        (2, 0x0010, 0x0000, 16),
        (4, 0x0000, 0x0004, 16),
        (4, 0x0010, 0x0004, 16),
        (4, 0x0008, 0x000e, 1104),
        (4, 0x0018, 0x000e, 1104),
    ];
    build_assert!(FINGERPRINT_H0.len() == cp::VIDEO_ARM_BURST.len());
    let ks = [0x5au8; 16];
    let riv = [0x11u8; 8];
    for (i, &(wire_type, sub_base, aux, body_len)) in cp::VIDEO_ARM_BURST.iter().enumerate() {
        let (want_type, want_sub, want_aux, want_body) = FINGERPRINT_H0[i];
        assert_eq!(
            (wire_type, sub_base, aux, body_len),
            (want_type, want_sub, want_aux, want_body)
        );
        if wire_type == 2 {
            let body = cp::video_arm_plaintext_body(i, 0);
            let frame = cp::video_arm_plain_frame(sub_base, &body);
            assert_eq!(frame.len(), 32);
            assert_eq!(u16::from_le_bytes([frame[8], frame[9]]), want_sub);
        } else {
            let content = KVec::from_elem(0u8, body_len, GFP_KERNEL)?;
            let frame = cp::seal_video_arm(&ks, &riv, sub_base, aux, 0, &content)?;
            assert_eq!(frame.len(), 16 + body_len + 16);
            assert_eq!(u16::from_le_bytes([frame[8], frame[9]]), want_sub);
            assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), want_aux);
        }
    }
    Ok(())
}

#[test]
fn navarro_stream_open_matches_the_wire() -> Result {
    // The connector lives solely in the wire sub, never in the content: all four connectors
    // send the same marker, followed by a two-byte opaque tail.
    let open = cp::navarro_stream_open();
    assert_eq!(open.len(), 16);
    assert_eq!(open[..14], cp::NAVARRO_STREAM_MARKER);

    // Sealing it produces the 48-byte frame the dock is sent: a 16-byte header, the 16-byte
    // ciphertext and a 16-byte Dl3Cmac, with `size` covering all but the first four bytes.
    let frame = cp::seal_video_arm(&[0u8; 16], &[0u8; 8], 0x0007, 0x0002, 0, &open)?;
    assert_eq!(frame.len(), 48);
    assert_eq!(u16::from_le_bytes([frame[2], frame[3]]), 0x002c);
    assert_eq!(u16::from_le_bytes([frame[8], frame[9]]), 0x0007);
    assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), 0x0002);
    Ok(())
}

#[test]
fn edid_reply_guards() -> Result {
    // The pre-decrypt guards reject non-EDID frames without touching the cipher.
    let ks = [0u8; 16];
    let riv = [0u8; 8];
    assert!(cp::parse_edid_from_reply(&ks, &riv, &[0u8; 10], true)?.is_none());
    let mut wrong_sub = [0u8; 20];
    wrong_sub[8] = 0x44; // wire sub != 0x45
    assert!(cp::parse_edid_from_reply(&ks, &riv, &wrong_sub, true)?.is_none());
    Ok(())
}

#[test]
fn get_edid_req_matches_dlm_wire_shape() -> Result {
    // The captured request is 32 bytes: an 8-byte header, 14 zero bytes, and a 10-byte random
    // tail at offset 22.
    let req = cp::get_edid_req(0x2c, 0)?;
    assert_eq!(req.len(), 32);
    assert_eq!(
        &req[0..8],
        &[0x15, 0x00, 0x21, 0x00, 0x2c, 0x00, 0x00, 0x00]
    );
    assert_eq!(&req[8..22], &[0u8; 14]);
    // Pin the complete wire framing as well:
    // aux=0x09 (cp::aux_for_id(0x15, ..)), body = 32 + 16 (tag) = 48 bytes.
    let frame = cp::seal_interactive(&[0x5au8; 16], &[0x11u8; 8], 0x15, 0, &req)?;
    assert_eq!(frame.len(), 16 + 32 + 16);
    assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), 0x09);
    Ok(())
}

#[test]
fn edid_engage_req_matches_dlm_wire_shape() -> Result {
    // Independent captures agree on `id=0x16 sub=0x0023` with the same 32-byte shape as
    // `get_edid_req`: an 8-byte header, 14 zero bytes, and a 10-byte random tail.
    let req = cp::edid_engage_req(0x30, 0)?;
    assert_eq!(req.len(), 32);
    assert_eq!(
        &req[0..8],
        &[0x16, 0x00, 0x23, 0x00, 0x30, 0x00, 0x00, 0x00]
    );
    assert_eq!(&req[8..22], &[0u8; 14]);
    let frame = cp::seal_interactive(&[0x5au8; 16], &[0x11u8; 8], 0x16, 0, &req)?;
    assert_eq!(frame.len(), 16 + 32 + 16);
    assert_eq!(u16::from_le_bytes([frame[10], frame[11]]), 0x08); // cp::aux_for_id(0x16, ..)
    Ok(())
}

#[test]
fn edid_poll_ready_byte_matches_golden_replies() -> Result {
    // Golden dock-to-host `id=0x0044 sub=0x0020` replies pin the readiness bit at inner offset
    // 26. The first precedes a placeholder `id=0x114` fetch and the second precedes a real
    // `id=0x194` EDID fetch.
    const KS: [u8; 16] = [
        0xd9, 0xec, 0x1f, 0xbc, 0x8b, 0x5a, 0xb3, 0xd8, 0x71, 0x0f, 0xd3, 0xbd, 0x42, 0x04,
        0x06, 0x55,
    ];
    const OUT_RIV: [u8; 8] = [0xf6, 0x21, 0xdc, 0x0d, 0x22, 0x7e, 0xf4, 0xaf];
    #[rustfmt::skip]
    const NOT_READY: [u8; 112] = [
        0x00, 0x00, 0x6c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x45, 0x00, 0x0a, 0x00, 0xcd, 0x00,
        0x00, 0x00, 0xa5, 0xea, 0x5d, 0x51, 0xf6, 0xa8, 0x6b, 0xb6, 0x89, 0x88, 0x01, 0xa2,
        0x47, 0x30, 0xbd, 0x6c, 0x84, 0xb8, 0xaf, 0x9f, 0x85, 0xf2, 0x8a, 0x20, 0xc8, 0xec,
        0x51, 0x9e, 0x8d, 0xeb, 0xef, 0x5a, 0x3a, 0x1d, 0xb5, 0xc7, 0x80, 0x02, 0xfe, 0x1e,
        0xed, 0x07, 0xdd, 0x71, 0x00, 0x7f, 0x45, 0x77, 0x6c, 0x82, 0xf6, 0xe9, 0xc3, 0x0d,
        0xdf, 0x67, 0x82, 0xac, 0xa8, 0x23, 0xd5, 0x5a, 0x1c, 0xce, 0xcb, 0x89, 0xb5, 0x98,
        0x65, 0xba, 0xbb, 0xb6, 0x2d, 0x0e, 0x9b, 0x55, 0xee, 0xfd, 0x46, 0x0c, 0x22, 0x35,
        0x6f, 0x84, 0xe5, 0x36, 0x95, 0xd0, 0xdc, 0xfc, 0x6f, 0x8a, 0x57, 0xda, 0xa2, 0xae,
    ];
    #[rustfmt::skip]
    const READY: [u8; 112] = [
        0x00, 0x00, 0x6c, 0x00, 0x04, 0x00, 0x00, 0x00, 0x45, 0x00, 0x0a, 0x00, 0x59, 0x03,
        0x00, 0x00, 0xf7, 0x8e, 0x70, 0xb2, 0xa3, 0x24, 0xe2, 0x6f, 0x9f, 0xb6, 0xe9, 0x8e,
        0x32, 0x55, 0x11, 0x21, 0x99, 0x74, 0xf6, 0xfb, 0xea, 0x97, 0xd5, 0x7f, 0xa6, 0x45,
        0x9d, 0x35, 0xf0, 0xa7, 0xbe, 0xd3, 0x9b, 0x19, 0x24, 0x8c, 0x98, 0xa6, 0x0c, 0xa2,
        0x4d, 0x8e, 0x83, 0xaa, 0x74, 0xd5, 0x8b, 0xe0, 0x6f, 0xb1, 0x9f, 0xa4, 0xb9, 0xae,
        0x39, 0xc6, 0x0a, 0x9c, 0x63, 0x70, 0xdb, 0x49, 0x74, 0xe5, 0x85, 0x42, 0x07, 0x7e,
        0xc2, 0x49, 0xfb, 0x67, 0x54, 0xd5, 0x47, 0x72, 0xb7, 0x19, 0x24, 0x8f, 0xb1, 0xb0,
        0xb2, 0x83, 0x89, 0x62, 0x4b, 0xcb, 0x59, 0x15, 0x1f, 0x8f, 0x85, 0xc3, 0xa5, 0x9d,
    ];
    assert_eq!(cp::edid_poll_ready(&KS, &OUT_RIV, &NOT_READY, true), Some(false));
    assert_eq!(cp::edid_poll_ready(&KS, &OUT_RIV, &READY, true), Some(true));
    Ok(())
}

#[test]
fn rgb565_packing() {
    assert_eq!(video::rgb565(0xff, 0x00, 0x00), 0xf800);
    assert_eq!(video::rgb565(0x00, 0xff, 0x00), 0x07e0);
    assert_eq!(video::rgb565(0x00, 0x00, 0xff), 0x001f);
    let _ = EINVAL; // silence unused import on configs without the assert paths
}

#[test]
fn cursor_messages_structure() -> Result {
    // Shared 32-byte cursor layout: marker 0x02 at 22, head ID at 23, and two little-endian u16
    // fields at 24 and 26.
    // Create (head 0): id=0x1b sub=0x42, fields = w,h.
    let c = cp::cursor_create(7, 0, 64, 64)?;
    assert_eq!(c.len(), 32);
    assert_eq!(&c[0..6], &[0x1b, 0x00, 0x42, 0x00, 0x07, 0x00]); // id, sub, counter (LE)
    assert_eq!(c[22], 0x02); // marker
    assert_eq!(c[23], 0); // head id
    assert_eq!(u16::from_le_bytes([c[24], c[25]]), 64); // width
    assert_eq!(u16::from_le_bytes([c[26], c[27]]), 64); // height

    // Move (head 1): id=0x1a sub=0x43, marker@22, head@23, X@24, Y@26 (LE).
    let m = cp::cursor_move(9, 1, 0x0140, 0x00f0, true)?;
    assert_eq!(m.len(), 32);
    assert_eq!(&m[0..4], &[0x1a, 0x00, 0x43, 0x00]); // id, sub
    assert_eq!(m[22], 0x02); // marker
    assert_eq!(m[23], 1); // head id
    assert_eq!(u16::from_le_bytes([m[24], m[25]]), 0x0140); // X
    assert_eq!(u16::from_le_bytes([m[26], m[27]]), 0x00f0); // Y

    // Image: 32-byte header (inner id 0x401c, the 0x40 bitmap flag) + w*h*4 BGRA at off32;
    // wrong-size input rejected.
    let bitmap = KVec::from_elem(0xabu8, 64 * 64 * 4, GFP_KERNEL)?;
    let img = cp::cursor_image(3, 0, 64, 64, &bitmap)?;
    assert_eq!(img.len(), 32 + 64 * 64 * 4);
    assert_eq!(&img[0..4], &[0x1c, 0x40, 0x41, 0x00]); // inner id 0x401c, sub 0x41
    assert_eq!(img[22], 0x02); // marker
    assert_eq!(img[32], 0xab); // bitmap begins at off32
    assert!(cp::cursor_image(3, 0, 64, 64, &[0u8; 16]).is_err()); // wrong bitmap length
    Ok(())
}

#[test]
fn timing_from_drm_mode_1080p60() -> Result {
    // CEA 1920x1080@60: clock 148.5 MHz, h 2008/2052/2200, v 1084/1089/1125.
    let mode = DisplayMode::from_timings(ModeTimings {
        clock_khz: 148_500,
        hdisplay: 1920,
        hsync_start: 2008,
        hsync_end: 2052,
        htotal: 2200,
        vdisplay: 1080,
        vsync_start: 1084,
        vsync_end: 1089,
        vtotal: 1125,
        flags: ModeFlags::PHSYNC | ModeFlags::PVSYNC,
    })?;
    assert_eq!(mode.cea_vic(), 16);
    let t = cp::timing_from_drm_mode(&mode, false)?;
    assert_eq!(t.hactive, 1920);
    assert_eq!(t.hblank, 280); // htotal - hdisplay
    assert_eq!(t.hsync_front, 88); // hsync_start - hdisplay
    assert_eq!(t.hsync_width, 44); // hsync_end - hsync_start
    assert_eq!(t.vactive, 1080);
    assert_eq!(t.vblank, 45); // vtotal - vdisplay
    assert_eq!(t.vsync_front, 4);
    assert_eq!(t.vsync_width, 5);
    assert_eq!(t.pixel_clock_10khz, 14_850); // clock(kHz) / 10
    assert_eq!(t.refresh_hz, 60); // via drm_mode_vrefresh
    Ok(())
}

/// A mode is bounded by link rate, not by refresh, except where the dock itself clamps.
///
/// The boundary cases matter most: each must pass by **equality**. A `<` would prune a dock's
/// working configuration and dark its panels.
#[test]
fn mode_ceilings_bound_bandwidth_not_refresh() {
    let refresh_ok =
        |p: &DockProfile, hz: i32| hz <= 0 || (hz as u32) <= p.max_refresh_hz;
    let clock_ok = |p: &DockProfile, khz: u32| khz <= p.max_head_clock_khz;

    // Ridge is clamped by refresh regardless of resolution: DLM puts 119.998 Hz on the wire
    // for a 180 Hz request and the 59.95 Hz CVT-RB timing for an 85 Hz one.
    assert_eq!(PROFILE_D6000.max_refresh_hz, 120);
    assert!(refresh_ok(&PROFILE_D6000, 120) && !refresh_ok(&PROFILE_D6000, 165));

    // The DL7400 has no such clamp -- DLM drives it at 2560x1440@164.96 -- so its modes are
    // bounded by link rate alone.
    assert!(refresh_ok(&PROFILE_DL7400, 180) && refresh_ok(&PROFILE_DL7400, 240));

    // 2560x1440: p165 is 699.50 MHz and carried; p180 is 714.81 MHz and is the mode the dock
    // accepts and then fails to deliver.
    assert!(clock_ok(&PROFILE_DL7400, 699_500));
    assert!(!clock_ok(&PROFILE_DL7400, 714_810));
    // Ridge cannot express the high half of the offset-70 u32 in any capture taken.
    assert!(clock_ok(&PROFILE_D6000, 655_350) && !clock_ok(&PROFILE_D6000, 699_500));

    // A degenerate mode reports 0 Hz and carries no rate information; a signed refresh must
    // never be read as a huge unsigned one.
    assert!(refresh_ok(&PROFILE_DL7400, 0) && refresh_ok(&PROFILE_DL7400, -1));

    // Each budget admits its own dual-head configuration and nothing beyond it.
    let rate = drm_sink::active_pixel_rate;
    assert_eq!(rate(2560, 1440, 120), 442_368_000);
    assert_eq!(PROFILE_D6000.pixel_budget, 2 * rate(2560, 1440, 120));
    assert_eq!(PROFILE_DL7400.pixel_budget, 2 * rate(2560, 1440, 165));
    assert_eq!(rate(65535, 65535, 65535), u32::MAX); // saturates, never wraps small
    assert_eq!(rate(2560, 1440, -1), 0);
}

/// Verify set-mode geometry and profile words against the decrypted DLM corpus.
///
/// The first four cases are byte-exact DLM messages (1920x1080p60/p120, 2560x1440p60/p120); the
/// 1280x720p60 and 3840x2160p60 cases predate the corpus and no capture backs them. All six are
/// reproduced by `cp::mode_profile`'s derivation apart from the 4K mode's off42 low bit, which
/// it carries as an explicit override.
#[test]
fn set_mode_matches_dlm_corpus() -> Result {
    // (hact, htotal, hsync_start, hsync_end, vact, vtotal, vsync_start, vsync_end, clock kHz,
    //  refresh, off42, off66)
    let cases: [(u16, u16, u16, u16, u16, u16, u16, u16, i32, u16, u16, u16); 6] = [
        (
            1280, 1650, 1390, 1430, 720, 750, 725, 730, 74_250, 60, 0x0400, 0x2804,
        ),
        (
            1920, 2200, 2008, 2052, 1080, 1125, 1084, 1089, 148_500, 60, 0x0400, 0x2810,
        ),
        (
            1920, 2200, 2008, 2052, 1080, 1125, 1084, 1089, 297_000, 120, 0x0400, 0x083f,
        ),
        (
            2560, 2720, 2608, 2640, 1440, 1481, 1443, 1448, 241_500, 60, 0x0600, 0x0800,
        ),
        (
            2560, 2720, 2608, 2640, 1440, 1525, 1443, 1448, 497_750, 120, 0x0600, 0x0800,
        ),
        (
            3840, 4000, 3888, 3920, 2160, 2222, 2163, 2168, 533_120, 60, 0x0604, 0x0800,
        ),
    ];
    for (hact, htotal, hss, hse, vact, vtotal, vss, vse, clock, refresh, off42, off66) in cases
    {
        let mode = DisplayMode::from_timings(ModeTimings {
            clock_khz: clock,
            hdisplay: hact,
            hsync_start: hss,
            hsync_end: hse,
            htotal,
            vdisplay: vact,
            vsync_start: vss,
            vsync_end: vse,
            vtotal,
            flags: if hact <= 1920 {
                ModeFlags::PHSYNC | ModeFlags::PVSYNC
            } else {
                ModeFlags::PHSYNC | ModeFlags::NVSYNC
            },
        })?;
        let t = cp::timing_from_drm_mode(&mode, false)?;
        let w = cp::set_mode(7, 1, &t)?;
        assert_eq!(w.len(), 80);
        let u16_at = |off: usize| u16::from_le_bytes([w[off], w[off + 1]]);
        assert_eq!(u16_at(26), hact); // hactive
        assert_eq!(u16_at(28), htotal - hact); // hblank
        assert_eq!(u16_at(30), hss - hact); // hsync front porch
        assert_eq!(u16_at(32), hse - hss); // hsync width
        assert_eq!(u16_at(34), vact); // vactive
        assert_eq!(u16_at(36), vtotal - vact); // vblank
        assert_eq!(u16_at(38), vss - vact); // vsync front porch
        assert_eq!(u16_at(40), vse - vss); // vsync width
        assert_eq!(u16_at(42), off42);
        assert_eq!(u16_at(44), refresh);
        assert_eq!(u16_at(66), off66);
        assert_eq!(u16_at(68), 0x0200);
        assert_eq!(u16_at(70), (clock as u32 / 10) as u16); // pixel clock / 10 kHz
        assert_eq!(&w[72..74], &[0, 0]);
    }
    Ok(())
}

#[test]
fn unmeasured_mode_is_accepted_with_a_derived_profile() -> Result {
    // 2560x1440@165: no decrypted message exists for it, but the profile is now derived rather
    // than refused. `drm_sink::mode_valid` is what prunes it -- its 699.5 MHz clock is past
    // `MAX_HEAD_CLOCK_KHZ`, and 165 Hz is past `DOCK_MAX_REFRESH_HZ`.
    let mode = DisplayMode::from_timings(ModeTimings {
        clock_khz: 699_500,
        hdisplay: 2560,
        hsync_start: 2608,
        hsync_end: 2640,
        htotal: 2720,
        vdisplay: 1440,
        vsync_start: 1443,
        vsync_end: 1451,
        vtotal: 1559,
        flags: ModeFlags::PHSYNC | ModeFlags::NVSYNC,
    })?;
    assert!(cp::mode_supported(&mode));
    // Still out of the dock's envelope, so userspace never sees it.
    assert!(mode.clock() as u32 > PROFILE_DL7400.max_head_clock_khz);
    // The clock field itself carries it fine: offsets 70..73 are a u32, as the DL7400's
    // 2560x1440p165 mode set proves (0x0001113d = 699.49 MHz). Admission is the refresh
    // limit's job, not a silent conversion failure.
    let t = cp::timing_from_drm_mode(&mode, false)?;
    assert_eq!(t.pixel_clock_10khz, (mode.clock() as u32) / 10);
    assert!(t.pixel_clock_10khz > u32::from(u16::MAX));
    Ok(())
}

/// A resolution with no capture at all must still produce a usable profile, so a monitor whose
/// native mode was never sampled is driven rather than refused.
#[test]
fn derived_profile_covers_an_unsampled_resolution() -> Result {
    // 1680x1050@60 CVT-RB: 119.00 MHz, no CTA VIC.
    let mode = DisplayMode::from_timings(ModeTimings {
        clock_khz: 119_000,
        hdisplay: 1680,
        hsync_start: 1728,
        hsync_end: 1760,
        htotal: 1840,
        vdisplay: 1050,
        vsync_start: 1053,
        vsync_end: 1059,
        vtotal: 1080,
        flags: ModeFlags::PHSYNC | ModeFlags::NVSYNC,
    })?;
    let t = cp::timing_from_drm_mode(&mode, false)?;
    // 1680 wide, so the bottom step of the off42 ladder.
    assert_eq!(t.field42, 0x0400);
    // No VIC, so the low byte is zero and the base is the common 0x0800.
    assert_eq!(t.off66, 0x0800);
    assert_eq!(t.pixel_clock_10khz, 11_900);
    Ok(())
}

#[test]
fn rotation_pixel_mapping() {
    use drm::kms::plane::Rotation;

    // Source 2x3 (sw=2, sh=3). 0deg is identity; 180deg mirrors both axes.
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_0, 0, 0, 2, 3), (0, 0));
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_0, 1, 2, 2, 3), (1, 2));
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_180, 0, 0, 2, 3), (1, 2));
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_180, 1, 2, 2, 3), (0, 0));
    // 90deg: output dims are (sh,sw)=(3,2); (dx,dy) -> (dy, sh-1-dx).
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_90, 0, 0, 2, 3), (0, 2));
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_90, 2, 1, 2, 3), (1, 0));
    // 270deg: (dx,dy) -> (sw-1-dy, dx).
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_270, 0, 0, 2, 3), (1, 0));
    assert_eq!(drm_sink::rot_src(Rotation::ROTATE_270, 2, 1, 2, 3), (0, 2));
    // Reflect-X composes on top of the rotation (here identity): sx -> sw-1-sx.
    assert_eq!(
        drm_sink::rot_src(Rotation::ROTATE_0 | Rotation::REFLECT_X, 0, 0, 2, 3),
        (1, 0)
    );
}

#[test]
fn parallel_encoder_matches_serial_for_every_plane_transform() -> Result {
    use drm::kms::plane::Rotation;

    let transforms = [
        Rotation::ROTATE_0,
        Rotation::ROTATE_90,
        Rotation::ROTATE_180,
        Rotation::ROTATE_270,
        Rotation::ROTATE_0 | Rotation::REFLECT_X,
        Rotation::ROTATE_90 | Rotation::REFLECT_X,
        Rotation::ROTATE_180 | Rotation::REFLECT_X,
        Rotation::ROTATE_270 | Rotation::REFLECT_X,
        Rotation::ROTATE_0 | Rotation::REFLECT_Y,
        Rotation::ROTATE_90 | Rotation::REFLECT_Y,
        Rotation::ROTATE_180 | Rotation::REFLECT_Y,
        Rotation::ROTATE_270 | Rotation::REFLECT_Y,
        Rotation::ROTATE_0 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
        Rotation::ROTATE_90 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
        Rotation::ROTATE_180 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
        Rotation::ROTATE_270 | Rotation::REFLECT_X | Rotation::REFLECT_Y,
    ];
    for transform in transforms {
        drm_sink::parallel_rotation_matches_serial(transform)?;
    }
    Ok(())
}

#[test]
fn wht_colour_and_quantize() {
    use video::wht;
    // Colour transform against captured transform-DC values: white maps to Y=16320,
    // achromatic pixels have zero chroma, and red's floored luma is 4032.
    assert_eq!(wht::colour(255, 255, 255), (16320, 0, 0));
    assert_eq!(wht::colour(128, 128, 128), (128 * 64, 0, 0));
    assert_eq!(wht::colour(255, 0, 0), (4032, 64 * 255, 0));
    // Green has two negative signed-chroma components.
    assert_eq!(wht::colour(0, 255, 0), (8128, -64 * 255, -64 * 255));
    assert_eq!(wht::colour(0, 0, 255), (4032, 0, 64 * 255));
    // White Y_DC=16320 quantizes to 1020 at DC position zero.
    assert_eq!(wht::quantize(16320, 0), 1020);
    // AC clamps to the 12-bit signed long-token range.
    assert_eq!(wht::quantize(1_000_000, 16), 2047);
    assert_eq!(wht::quantize(-1_000_000, 16), -2048);
}

#[test]
fn wht_transform_uniform() {
    use video::wht;
    // A uniform block has the per-pixel value at DC and zero AC terms.
    let block = [16320i32; wht::BLOCK];
    let c = wht::transform(&block);
    assert_eq!(c[0], 16320);
    assert!(c[1..].iter().all(|&x| x == 0));
    // White pixel -> Y plane -> WHT DC -> quantized value.
    let (y, _, _) = wht::colour(255, 255, 255);
    assert_eq!(wht::quantize(wht::transform(&[y; wht::BLOCK])[0], 0), 1020);
}

/// The quantisers divide by powers of two with arithmetic shifts, avoiding a runtime division
/// for every coefficient.
///
/// The rewrite is only valid because floor division by `2^k` IS an arithmetic right shift, for
/// negative operands as well as positive. That identity is easy to state and easy to get wrong
/// (a *truncating* `/` is not the same thing on negatives), and a coefficient off by one is a
/// wire-visible codec change. So assert it directly, over the full coefficient range the
/// transform can produce, against the equivalent division.
#[test]
fn quantiser_shifts_match_division() {
    use video::wht::{quantize, COEFFS};
    // 8-bit input in the codec's x64 fixed point, summed over an 8x8 block and floor-divided by
    // 64 by `transform`, bounds |coeff| well inside this; step past it on both signs anyway.
    const LIMIT: i32 = 200_000;
    // The luma table, restated here as DIVISORS so the test is not written against the same
    // shift constants it is checking.
    let step_bias = |i: usize| -> (i32, i32) {
        match i {
            0 | 1 | 2 => (16, 8),
            3 => (32, 16),
            4..=11 => (4, 2),
            12..=15 => (8, 4),
            16..=47 => (2, 0),
            _ => (4, 2),
        }
    };
    for i in 0..COEFFS {
        let (step, bias) = step_bias(i);
        for coeff in (-LIMIT..=LIMIT).step_by(37) {
            let want = if bias == 0 {
                let q = coeff.abs() / step;
                if coeff < 0 {
                    -q
                } else {
                    q
                }
            } else {
                (coeff + bias).div_euclid(step)
            }
            .clamp(-2048, 2047);
            assert_eq!(quantize(coeff, i), want);
        }
    }
    // Boundary cases the stride above can step over: every exact multiple and half-step of the
    // coarsest divisor, on both signs, is where floor-vs-truncate actually differs.
    for i in 0..COEFFS {
        let (step, bias) = step_bias(i);
        for m in -4i32..=4 {
            for d in [-1, 0, 1, step / 2, -step / 2] {
                let coeff = m * step + d;
                let want = if bias == 0 {
                    let q = coeff.abs() / step;
                    if coeff < 0 {
                        -q
                    } else {
                        q
                    }
                } else {
                    (coeff + bias).div_euclid(step)
                }
                .clamp(-2048, 2047);
                assert_eq!(quantize(coeff, i), want);
            }
        }
    }
}

#[test]
fn wht_transform_haar_vectors() {
    // Independent golden vectors cover the source gradient blocks. Input luma is
    // `Y = 64 * gray`.
    use video::wht::{transform, DIM, PIXELS};
    // Build an 8x8 Y block by evaluating a gray-per-(row,col) selector.
    fn build(gray: impl Fn(usize, usize) -> i32) -> [i32; PIXELS] {
        let mut b = [0i32; PIXELS];
        for r in 0..DIM {
            for c in 0..DIM {
                b[r * DIM + c] = 64 * gray(r, c);
            }
        }
        b
    }
    // vstripe2 (period-2 vertical, full contrast 0/255) -> level-2 HL band c[4..8] = -2040.
    let c = transform(&build(|_, c| if (c / 2) & 1 != 0 { 255 } else { 0 }));
    assert_eq!(&c[4..8], &[-2040, -2040, -2040, -2040]);
    assert!(c[1..4].iter().all(|&x| x == 0) && c[8..].iter().all(|&x| x == 0));
    // Period-four vertical stripe: coarse HL c[1] = -8160.
    let c = transform(&build(|_, c| if (c / 4) & 1 != 0 { 255 } else { 0 }));
    assert_eq!(c[1], -8160);
    // Period-two horizontal stripe: level-two LH band is -2040.
    let c = transform(&build(|r, _| if (r / 2) & 1 != 0 { 255 } else { 0 }));
    assert_eq!(&c[8..12], &[-2040, -2040, -2040, -2040]);
    // A per-column gradient exercises DC, coarse-HL, and the finest band.
    let c = transform(&build(|_, col| 36 * col as i32));
    assert_eq!(c[0], 8064); // DC = mean(36*0..36*7)*64/64 = 8064
    assert_eq!(&c[4..8], &[-576, -576, -576, -576]);
    // The level-1 tail contains three 4x4 Morton-scanned bands:
    // c[16..32] = HL1, c[32..48] = LH1, and c[48..64] = HH1.
    //
    // A per-column ramp has no vertical detail: HL1 is uniformly -72 and LH1/HH1 are zero.
    assert!(c[16..32].iter().all(|&x| x == -72)); // finest HL: horizontal detail only
    assert!(c[32..].iter().all(|&x| x == 0)); // LH1 + HH1: no vertical detail
}

#[test]
fn wht_vlc_codebook_byte_exact() -> Result {
    // The LSB-first entropy VLC is checked against independent golden output. Symbol 7 is the
    // AC code
    // 0b1110000 (LSB-first); four of them pack to the wire's per-block AC unit bytes, and the
    // final byte is padded with 1-bits (a truncated all-ones code), exactly as the dock emits.
    use video::wht::Vlc;
    let mut w = Vlc::new();
    for _ in 0..4 {
        w.symbol(7)?;
    }
    assert_eq!(&w.finish()?[..], &[0x87, 0xc3, 0xe1, 0xf0]);
    // The full per-block AC unit `0 0 0 7 7 7 7` (idx1-3 zero, idx4-7 AC) -- matches the live
    // wire bytes `38 1c 0e ...` captured for vstripe2.
    let mut w = Vlc::new();
    for s in [0usize, 0, 0, 7, 7, 7, 7] {
        w.symbol(s)?;
    }
    assert_eq!(&w.finish()?[..4], &[0x38, 0x1c, 0x0e, 0x87]);
    // Symbol 0 (the 1-bit `0` code) alone -> one byte padded with seven 1-bits.
    let mut w = Vlc::new();
    w.symbol(0)?;
    assert_eq!(&w.finish()?[..], &[0xfe]);
    Ok(())
}

#[test]
fn wht_coeff_magnitude_code() -> Result {
    // The AC magnitude-code emitter is checked against per-coefficient golden wire bits for
    // q-4, q-8, and q-16.
    use video::wht::Vlc;
    // Four q-4 coefficients (category 3, zero offset) == four sym7 -- the per-block AC unit.
    let mut w = Vlc::new();
    for _ in 0..4 {
        w.coeff(-4)?;
    }
    assert_eq!(&w.finish()?[..], &[0x87, 0xc3, 0xe1, 0xf0]);
    // A zero coefficient is the 1-bit symbol 0 -> one byte padded with seven 1-bits.
    let mut w = Vlc::new();
    w.coeff(0)?;
    assert_eq!(&w.finish()?[..], &[0xfe]);
    // Within-category offset (q-6 = category 3, offset 2) and sign polarity (negative vs +).
    let mut w = Vlc::new();
    w.coeff(-6)?;
    assert_eq!(&w.finish()?[..], &[0x97]);
    let mut w = Vlc::new();
    w.coeff(6)?;
    assert_eq!(&w.finish()?[..], &[0xd7]); // same magnitude, sign bit flipped
                                           // Category 5 with offset (q-16) spans two bytes.
    let mut w = Vlc::new();
    w.coeff(-16)?;
    assert_eq!(&w.finish()?[..], &[0x1f, 0xf8]);
    // The unsupported long-form escape is rejected.
    let mut w = Vlc::new();
    assert!(w.coeff(-256).is_err());
    Ok(())
}

#[test]
fn wht_magnitude_category() {
    // Magnitude category is `bit_length(abs(coeff))`.
    use video::wht::mag_category;
    assert_eq!(mag_category(0), 0);
    assert_eq!(mag_category(1), 1);
    assert_eq!(mag_category(-4), 3);
    assert_eq!(mag_category(7), 3);
    assert_eq!(mag_category(-8), 4);
    assert_eq!(mag_category(16), 5);
    assert_eq!(mag_category(-128), 8);
    assert_eq!(mag_category(255), 8);
}

#[test]
fn wht_chroma_last_is_exact() {
    use video::wht::{chroma_last, COEFFS};
    let mut q = [0i32; COEFFS];
    assert_eq!(chroma_last(&q), 0);
    for exact in [1usize, 2, 3, 4, 7, 8, 11, 15, 16, 27, 31, 32, 48, 62, 63] {
        q.fill(0);
        q[exact] = 1;
        assert_eq!(chroma_last(&q), exact);
    }
}

#[test]
fn display_capability_reply_reports_presence() -> Result {
    let key = [0x5au8; 16];
    let riv = [0x33u8; 8];
    let mut inner = [0u8; 32];
    inner[0..2].copy_from_slice(&0x78u16.to_le_bytes());
    inner[2..4].copy_from_slice(&0x20u16.to_le_bytes());
    inner[22..26].copy_from_slice(&0x1234u32.to_le_bytes());
    inner[26] = 0x80;
    let mut wire = cp::seal_interactive(&key, &riv, 0x78, 11, &inner)?;
    wire[8..10].copy_from_slice(&0x45u16.to_le_bytes());

    assert_eq!(
        cp::probe_reply_status(&key, &riv, &wire, true),
        Some((0x78, 0x1234, true))
    );
    Ok(())
}

#[test]
fn set_mode_has_head_and_exact_dlm_plaintext_length() -> Result {
    let timing = cp::Timing {
        hactive: 3840,
        hblank: 160,
        hsync_front: 48,
        hsync_width: 32,
        vactive: 2160,
        vblank: 62,
        vsync_front: 3,
        vsync_width: 5,
        refresh_hz: 60,
        pixel_clock_10khz: 0xd040,
        field42: 0x0604,
        off46: 0x4000,
        off48: 0x6000,
        off66: 0x0800,
    };
    let m = cp::set_mode(0x1234, 1, &timing)?;
    assert_eq!(m.len(), 80);
    assert_eq!(&m[0..6], &[0x48, 0x00, 0x22, 0x00, 0x34, 0x12]);
    assert!(m[6..22].iter().all(|&x| x == 0));
    assert_eq!(&m[22..26], &[1, 2, 0, 0]);
    assert_eq!(u16::from_le_bytes([m[26], m[27]]), 3840);
    assert_eq!(u16::from_le_bytes([m[34], m[35]]), 2160);
    assert_eq!(u32::from_le_bytes([m[70], m[71], m[72], m[73]]), 0xd040);
    assert_eq!(u16::from_le_bytes([m[68], m[69]]), 0x0200);
    Ok(())
}

#[test]
fn stream_marker_routes_the_selected_head() -> Result {
    let h0 = cp::stream_marker(0x1234, 0, 0x2f, 1)?;
    let h1 = cp::stream_marker(0x1235, 1, 0x2e, 3)?;
    assert_eq!(&h0[0..6], &[0x16, 0, 0x2f, 0, 0x34, 0x12]);
    assert_eq!(&h0[22..24], &[0, 1]);
    assert_eq!(&h1[0..6], &[0x16, 0, 0x2e, 0, 0x35, 0x12]);
    assert_eq!(&h1[22..24], &[1, 3]);
    Ok(())
}

#[test]
fn video_frame_trailer_matches_dlm_cycle_and_head() {
    let geom = PROFILE_D6000.geometry();
    let t0 = video::wht::frame_trailer(geom, 0, 0);
    assert_eq!(
        &t0[..32],
        &[
            0, 0, 0x1c, 0, 4, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 5, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
        ]
    );
    assert_eq!(
        &t0[32..64],
        &[
            0, 0, 0x1c, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0x0a, 0, 4, 2, 0, 0, 0, 8, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]
    );
    // Record C ORs the head selector with 0x10. `sub` is the little-endian u16 at bytes 8..10,
    // so the selector belongs in byte 8; placing it in byte 9 would encode 0x1100 instead of
    // 0x0011 and prevent head 1 from presenting the frame.
    assert_eq!(
        &t0[64..],
        &[
            0, 0, 0x1c, 0, 4, 0, 0, 0, 0x10, 0, 4, 0, 0, 0, 0, 0, 0x0a, 0, 4, 2, 0, 0, 0, 8, 0,
            0, 0, 0, 0, 0, 0, 0,
        ]
    );

    let t1 = video::wht::frame_trailer(geom, 1, 1);
    assert_eq!(u16::from_le_bytes([t1[8], t1[9]]), 0x0001);
    assert_eq!(u16::from_le_bytes([t1[32 + 8], t1[32 + 9]]), 0x0001);
    assert_eq!(u16::from_le_bytes([t1[64 + 8], t1[64 + 9]]), 0x0011);
    assert_eq!(t1[19], 2);
    assert_eq!(t1[23], 8);
    assert_eq!(t1[25], 2);
    assert_eq!(t1[32 + 19], 4);
    assert_eq!(t1[32 + 23], 16);
    assert_eq!(t1[32 + 27], 8);
}

#[test]
fn video_arm_configuration_uses_mode_and_nonce() -> Result {
    let nonce = [0x5a; 14];
    let config = video_arm::build(1920, 1080, &nonce)?;

    assert_eq!(config.len(), 1104);
    assert_eq!(&config[10..14], &[0x80, 0x07, 0x38, 0x04]);
    assert_eq!(&config[18..22], &[0x80, 0x07, 0x38, 0x04]);
    assert_eq!(&config[1090..], &nonce);
    Ok(())
}

#[test]
fn navarro_pipe_descriptor_matches_authenticated_capture() -> Result {
    // Slot ids and the three dock-side addresses of every record, for both connectors of the
    // authenticated capture.
    for (connector, slots) in [
        (0u8, [
            (0x0000u16, 0x6fccu32, 0x71fb_9000u32, 0x7216_6000u32),
            (0x0001, 0x6db0, 0x71fb_4000, 0x7215_e000),
            (0x0002, 0x6b94, 0x71fa_f000, 0x7215_6000),
            (0x0003, 0x6978, 0x71fa_a000, 0x7214_e000),
            (0x0004, 0x675c, 0x71fa_5000, 0x7214_6000),
            (0x0005, 0x6540, 0x71fa_0000, 0x7213_e000),
        ]),
        (1u8, [
            (0x0008, 0x5eec, 0x71f9_b000, 0x7213_6000),
            (0x0009, 0x5cd0, 0x71f9_6000, 0x7212_e000),
            (0x000a, 0x5ab4, 0x71f9_1000, 0x7212_6000),
            (0x000b, 0x5898, 0x71f8_c000, 0x7211_e000),
            (0x000c, 0x567c, 0x71f8_7000, 0x7211_6000),
            (0x000d, 0x5460, 0x71f8_2000, 0x7210_e000),
        ]),
    ] {
        let descriptor = cp::navarro_pipe_descriptor(connector)?;
        assert_eq!(descriptor.len(), 304);
        assert_eq!(&descriptor[..14], &cp::NAVARRO_STREAM_MARKER);
        for (index, &(slot, ring, plane0, plane1)) in slots.iter().enumerate() {
            let at = 14 + index * 46;
            assert_eq!(&descriptor[at..at + 4], &[0x2c, 0x00, 0x0e, 0x00]);
            assert_eq!(
                u16::from_le_bytes([descriptor[at + 4], descriptor[at + 5]]),
                slot
            );
            let cfg = &descriptor[at + 6..at + 46];
            let word = |o: usize| {
                u32::from_le_bytes([cfg[o], cfg[o + 1], cfg[o + 2], cfg[o + 3]])
            };
            assert_eq!(word(12), ring);
            assert_eq!(word(18), plane0);
            assert_eq!(word(26), plane1);
        }
    }

    // The decoder configuration is the same message Ridge sends, with the DL7400's layout word.
    let tail = [0x5a; 14];
    let config = video_arm::build_with_layout_word(2560, 1440, 0x2100, &tail)?;
    assert_eq!(config.len(), 1104);
    assert_eq!(
        &config[..26],
        &[
            0x18, 0x00, 0x0b, 0x03, 0x04, 0x02, 0x02, 0x00, 0x02, 0x00, 0x00, 0x0a, 0xa0,
            0x05, 0x00, 0x21, 0x02, 0x00, 0x00, 0x0a, 0xa0, 0x05, 0x00, 0x21, 0x00, 0x00,
        ]
    );
    assert_eq!(&config[1090..], &tail);
    Ok(())
}
