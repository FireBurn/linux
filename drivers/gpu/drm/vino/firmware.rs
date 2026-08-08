// SPDX-License-Identifier: GPL-2.0
//! Dock firmware: identity, package parsing, and the USB DFU update.
//!
//! A DisplayLink dock carries its firmware version in a vendor descriptor rather than in
//! `bcdDevice`, which does not change across an update. The shipped `*-release.spkg` packages carry
//! theirs in a tagged table. Comparing the two says whether an update is due; the transfer itself
//! is textbook USB DFU 1.1 on the dock's DFU interface.

use kernel::device::Device;
#[cfg(CONFIG_RUST_FW_LOADER_ABSTRACTIONS)]
use kernel::firmware::Firmware;
use kernel::prelude::*;
use kernel::time::Delta;
use kernel::usb;

/// The dock's DFU interface: `bInterfaceClass 0xfe`, `bInterfaceSubClass 1`, and the interface
/// number every DFU class request is addressed to.
pub(crate) const DFU_INTERFACE: u8 = 1;

/// Vendor descriptor carrying the platform name and running firmware version.
///
/// `bcdDevice` does **not** change when a dock is updated, so it cannot answer "is an update due".
/// This descriptor can: it is 16 bytes, `[len, 0x40, major, minor, patch, ..., name(8)]`, and the
/// name selects the package that targets this hardware.
pub(crate) const DESCRIPTOR_IDENTITY: u8 = 0x40;
const IDENTITY_LEN: usize = 16;
const IDENTITY_NAME: usize = 8;

/// A dock's platform name and the firmware version it is running.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Identity {
    pub(crate) version: Version,
    name: [u8; IDENTITY_NAME],
}

/// A three-part firmware version, ordered major-minor-patch.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Version(pub(crate) u8, pub(crate) u8, pub(crate) u8);


impl Identity {
    /// Parse the identity descriptor out of a device's raw configuration descriptors.
    pub(crate) fn parse(raw: &[u8]) -> Option<Self> {
        let mut i = 0usize;
        while i + 2 <= raw.len() {
            let len = usize::from(raw[i]);
            if len < 2 || i + len > raw.len() {
                return None;
            }
            if raw[i + 1] == DESCRIPTOR_IDENTITY && len >= IDENTITY_LEN {
                let d = &raw[i..i + IDENTITY_LEN];
                let mut name = [0u8; IDENTITY_NAME];
                name.copy_from_slice(&d[8..16]);
                return Some(Self {
                    version: Version(d[2], d[3], d[4]),
                    name,
                });
            }
            i += len;
        }
        None
    }

    /// The platform name, trimmed of its padding.
    pub(crate) fn platform(&self) -> &[u8] {
        let end = self
            .name
            .iter()
            .position(|&c| c == 0 || c == b' ')
            .unwrap_or(IDENTITY_NAME);
        &self.name[..end]
    }

    /// The firmware file that targets this platform, under `/lib/firmware`.
    ///
    /// Named after the package DisplayLink ships for the platform, so a distribution can drop the
    /// vendor's own file in unmodified.
    pub(crate) fn firmware_name(&self) -> Option<&'static CStr> {
        match self.platform() {
            b"NavaDock" => Some(c"vino/navarro-dock-release.spkg"),
            b"Ridge" => Some(c"vino/ridge-dock-release.spkg"),
            b"Ella" => Some(c"vino/ella-dock-release.spkg"),
            b"Firefly" => Some(c"vino/firefly-monitor-release.spkg"),
            _ => None,
        }
    }
}

/// Standard `GET_DESCRIPTOR` for the configuration descriptor, which the identity blob rides in.
const GET_DESCRIPTOR: u8 = 6;
const DEVICE_TO_HOST_STANDARD: u8 = 0x80;
const DESCRIPTOR_CONFIG: u16 = 0x0200;
/// Enough for any configuration this hardware reports; the dock's is well under 1 KiB.
const CONFIG_DESCRIPTOR_MAX: usize = 1024;

/// Read the dock's identity by walking its configuration descriptor.
///
/// The blob is a vendor descriptor inside the configuration, not a separately addressable one, so
/// the whole configuration is fetched and walked. `usb_device::rawdescriptors` holds the same bytes
/// already, but no binding exposes them and one standard control read costs nothing on probe.
pub(crate) fn read_identity(io: &usb::Io<'_>) -> Result<Identity> {
    let mut head = [0u8; 9];
    io.control_recv(
        GET_DESCRIPTOR,
        DEVICE_TO_HOST_STANDARD,
        DESCRIPTOR_CONFIG,
        0,
        &mut head,
        XFER_TIMEOUT,
        GFP_KERNEL,
    )?;
    let total = usize::from(u16::from_le_bytes([head[2], head[3]]));
    if total < head.len() || total > CONFIG_DESCRIPTOR_MAX {
        return Err(EINVAL);
    }
    let mut all = KVec::with_capacity(total, GFP_KERNEL)?;
    all.resize(total, 0u8, GFP_KERNEL)?;
    io.control_recv(
        GET_DESCRIPTOR,
        DEVICE_TO_HOST_STANDARD,
        DESCRIPTOR_CONFIG,
        0,
        &mut all,
        XFER_TIMEOUT,
        GFP_KERNEL,
    )?;
    Identity::parse(&all).ok_or(ENODEV)
}

/// The version a `.spkg` will install.
///
/// The package is a tagged table: a two-byte ASCII tag, a `u16` length, then the value. `VE` holds
/// the three version bytes. Its offset differs per package -- 100 in one shipped image and 24574 in
/// another -- so it is searched for rather than assumed.
pub(crate) fn package_version(image: &[u8]) -> Option<Version> {
    const TAG: [u8; 4] = [b'V', b'E', 3, 0];
    image
        .windows(TAG.len() + 3)
        .find(|w| w[..TAG.len()] == TAG)
        .map(|w| Version(w[4], w[5], w[6]))
}

/// `.spkg` container magic, checked before anything is written to a dock.
const PACKAGE_MAGIC: &[u8; 4] = b"ELLA";

/// Whether `image` is a firmware package at all.
pub(crate) fn is_package(image: &[u8]) -> bool {
    image.len() > 8 && &image[..4] == PACKAGE_MAGIC
}

// USB DFU 1.1 class requests, on the dock's DFU interface.
const DFU_OUT: u8 = 0x21; // host-to-device, class, interface
const DFU_IN: u8 = 0xa1; // device-to-host, class, interface
const DFU_DETACH: u8 = 0;
const DFU_DNLOAD: u8 = 1;
const DFU_GETSTATUS: u8 = 3;

/// Payload per `DFU_DNLOAD`.
///
/// The DFU functional descriptor advertises a 16384-byte `wTransferSize`, but the vendor's own
/// updater sends 4096 and the dock is only known to accept that.
const BLOCK: usize = 4096;

/// `wValue` of `DFU_DETACH`, in milliseconds. The vendor sends 100.
const DETACH_TIMEOUT_MS: u16 = 100;

/// How long a single control transfer may take.
const XFER_TIMEOUT: Delta = Delta::from_secs(5);

/// `DFU_GETSTATUS` reply: `bStatus, bwPollTimeout[3], bState, iString`.
const STATUS_LEN: usize = 6;
const STATUS_OK: u8 = 0;
const STATE_DNLOAD_IDLE: u8 = 5;
const STATE_DNBUSY: u8 = 4;
const STATE_MANIFEST_SYNC: u8 = 6;
const STATE_MANIFEST: u8 = 7;

/// Poll `DFU_GETSTATUS` until the dock leaves `dfuDNBUSY`, honouring its own poll timeout.
fn wait_ready(io: &usb::Io<'_>, iface: u16) -> Result<u8> {
    for _ in 0..1000 {
        let mut st = [0u8; STATUS_LEN];
        io.control_recv(
            DFU_GETSTATUS,
            DFU_IN,
            0,
            iface,
            &mut st,
            XFER_TIMEOUT,
            GFP_KERNEL,
        )?;
        if st[0] != STATUS_OK {
            pr_err!(
                "vino: firmware update rejected: DFU status {} in state {}\n",
                st[0],
                st[4]
            );
            return Err(EIO);
        }
        // bwPollTimeout is a 24-bit little-endian millisecond count the device asks us to wait.
        let poll = u32::from(st[1]) | u32::from(st[2]) << 8 | u32::from(st[3]) << 16;
        if st[4] != STATE_DNBUSY && st[4] != STATE_MANIFEST {
            return Ok(st[4]);
        }
        kernel::time::delay::fsleep(Delta::from_millis(poll.clamp(1, 1000) as i64));
    }
    Err(ETIMEDOUT)
}

/// Write `image` to the dock over USB DFU.
///
/// ⛔ **This is not reversible from the host.** The DFU functional descriptor clears
/// `bitCanUpload`, so the running firmware cannot be read back and there is nothing to restore
/// from; and it sets `bitManifestationTolerant = 0`, so the dock re-enumerates when the image is
/// manifested. An interrupted write leaves the dock with a partial image.
///
/// The sequence is the vendor updater's, recorded end to end: `DFU_DETACH`, then the whole package
/// verbatim in 4096-byte blocks with an ascending `wValue`, each followed by `DFU_GETSTATUS`, then
/// a zero-length `DFU_DNLOAD` to manifest. There is no bus reset between the detach and the first
/// block -- the dock accepts the download in its runtime interface.
pub(crate) fn flash(io: &usb::Io<'_>, iface: u16, image: &[u8]) -> Result {
    if !is_package(image) {
        pr_err!("vino: refusing to flash: not a DisplayLink firmware package\n");
        return Err(EINVAL);
    }
    let blocks = image.len().div_ceil(BLOCK);
    if blocks > usize::from(u16::MAX) {
        return Err(EFBIG);
    }
    pr_info!(
        "vino: flashing {} bytes of dock firmware in {} block(s) -- do not disconnect\n",
        image.len(),
        blocks
    );

    io.control_send(
        DFU_DETACH,
        DFU_OUT,
        DETACH_TIMEOUT_MS,
        iface,
        &[],
        XFER_TIMEOUT,
        GFP_KERNEL,
    )?;

    for (n, chunk) in image.chunks(BLOCK).enumerate() {
        io.control_send(
            DFU_DNLOAD,
            DFU_OUT,
            n as u16,
            iface,
            chunk,
            XFER_TIMEOUT,
            GFP_KERNEL,
        )?;
        let state = wait_ready(io, iface)?;
        if state != STATE_DNLOAD_IDLE {
            pr_err!("vino: firmware block {n} left the dock in DFU state {state}\n");
            return Err(EIO);
        }
    }

    // Zero-length download: the image is complete, manifest it. The dock re-enumerates from here,
    // so a failure to read status back afterwards is expected rather than an error.
    io.control_send(
        DFU_DNLOAD,
        DFU_OUT,
        0,
        iface,
        &[],
        XFER_TIMEOUT,
        GFP_KERNEL,
    )?;
    match wait_ready(io, iface) {
        Ok(state) if state == STATE_MANIFEST_SYNC || state == STATE_DNLOAD_IDLE => {}
        Ok(state) => pr_info!("vino: dock manifested in DFU state {state}\n"),
        Err(_) => pr_info!("vino: dock stopped answering after manifest, as it re-enumerates\n"),
    }
    pr_info!("vino: dock firmware written; it will re-enumerate to run it\n");
    Ok(())
}

/// Decide whether `dev` needs an update, and apply one if it does.
///
/// An update is applied only when the packaged version is **strictly newer** than the one the dock
/// reports, so the ordinary path on a current dock is one descriptor read and no writes at all. A
/// missing firmware file is not an error: a dock runs perfectly well on the firmware it shipped
/// with, and most systems will never carry the package.
#[cfg(not(CONFIG_RUST_FW_LOADER_ABSTRACTIONS))]
pub(crate) fn update_if_newer(
    _io: &usb::Io<'_>,
    dev: &Device,
    identity: &Identity,
    _iface: u16,
    _force: bool,
) -> Result {
    dev_info!(
        dev,
        "vino: dock firmware {}.{}.{}; updates need CONFIG_RUST_FW_LOADER_ABSTRACTIONS\n",
        identity.version.0,
        identity.version.1,
        identity.version.2
    );
    Ok(())
}

#[cfg(CONFIG_RUST_FW_LOADER_ABSTRACTIONS)]
pub(crate) fn update_if_newer(
    io: &usb::Io<'_>,
    dev: &Device,
    identity: &Identity,
    iface: u16,
    force: bool,
) -> Result {
    let Some(name) = identity.firmware_name() else {
        return Ok(());
    };
    let Ok(fw) = Firmware::request_nowarn(name, dev) else {
        dev_info!(
            dev,
            "vino: no {} available; leaving the dock on {}.{}.{}\n",
            name,
            identity.version.0,
            identity.version.1,
            identity.version.2
        );
        return Ok(());
    };
    let Some(packaged) = package_version(fw.data()) else {
        dev_warn!(dev, "vino: {} carries no version tag; ignoring it\n", name);
        return Ok(());
    };
    if packaged <= identity.version && !force {
        dev_info!(
            dev,
            "vino: dock firmware {}.{}.{} is current ({} offers {}.{}.{})\n",
            identity.version.0, identity.version.1, identity.version.2,
            name,
            packaged.0, packaged.1, packaged.2
        );
        return Ok(());
    }
    dev_info!(
        dev,
        "vino: updating dock firmware {}.{}.{} -> {}.{}.{}{}\n",
        identity.version.0, identity.version.1, identity.version.2,
        packaged.0, packaged.1, packaged.2,
        if force { " (forced)" } else { "" }
    );
    flash(io, iface, fw.data())
}
