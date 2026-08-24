// SPDX-License-Identifier: GPL-2.0

//! Registration: the DRM driver description, its GEM object and file types, and the KMS
//! entry points the core calls into.

use super::*;

/// GEM object inner data. Empty: the shmem-backed `drm::gem::shmem::Object` (which
/// wires `drm_gem_shmem_dumb_create`, so userspace `DRM_IOCTL_MODE_CREATE_DUMB`
/// works) is enough until the EP08 scanout path consumes the framebuffers.
#[pin_data]
pub(crate) struct VinoObject {}

impl drm::gem::DriverObject for VinoObject {
    type Driver = VinoDrmDriver;
    type Args = ();

    fn new(
        _dev: &drm::Device<VinoDrmDriver>,
        _size: usize,
        _args: (),
    ) -> impl PinInit<Self, Error> {
        try_pin_init!(VinoObject {})
    }
}

/// Per-open DRM client state. The generic DRM fops pin the owning module for the file lifetime.
#[pin_data]
pub(crate) struct VinoDrmFile {}

impl drm::file::DriverFile for VinoDrmFile {
    type Driver = VinoDrmDriver;

    fn open(_dev: &drm::Device<Self::Driver>) -> Result<Pin<KBox<Self>>> {
        KBox::try_pin_init(try_pin_init!(Self {}), GFP_KERNEL)
    }
}

pub(super) const INFO: drm::DriverInfo = drm::DriverInfo {
    major: 0,
    minor: 1,
    patchlevel: 0,
    name: c"vino",
    desc: c"DisplayLink DL3 (Dell D6000) DRM driver",
};

#[vtable]
impl drm::Driver for VinoDrmDriver {
    type Data = VinoDrmData;
    type File = VinoDrmFile;
    type Object = drm::gem::shmem::Object<VinoObject>;
    type ParentDevice<Ctx: kernel::device::DeviceContext> = crate::usb::Interface<Ctx>;
    type RegistrationData<'a> = ();
    type Kms = Self;

    const INFO: drm::DriverInfo = INFO;

    // No driver-private ioctls (GEM/dumb + KMS handled by the DRM core).
    kernel::declare_drm_ioctls! {}
}

#[vtable]
impl KmsDriver for VinoDrmDriver {
    type Connector = VinoConnector;
    type Plane = VinoPlane;
    type Crtc = VinoCrtc;
    type Encoder = VinoEncoder;

    fn mode_config_info(
        _dev: &kernel::device::Device,
        _drm_data: &Self::Data,
    ) -> Result<ModeConfigInfo> {
        Ok(ModeConfigInfo {
            min_resolution: (0, 0),
            max_resolution: (4096, 4096),
            max_cursor: (64, 64),
            preferred_depth: 32,
            preferred_fourcc: Some(drm::fourcc::XRGB8888),
        })
    }

    fn create_objects(dev: &UnregisteredKmsDevice<'_, Self>) -> Result {
        let data: &VinoDrmData = dev;
        // Build one independent connector (CRTC + primary/cursor plane + encoder + connector) per
        // wired display, each pinned to its own video endpoint via its connector index.
        //
        // Only as many as the dock has sockets. `MAX_CONNECTORS` is the largest any supported dock
        // has, so it sizes the per-connector arrays, but building that many objects on a
        // two-connector dock publishes outputs with nothing behind them: they never gain an EDID, a
        // compositor is free to enable one anyway, and the driver then encodes and transmits whole
        // frames to a socket that cannot display them -- onto the same endpoint the real connector
        // is using.
        for connector in 0..data.connector_count() {
            // `possible_crtcs` for the plane/encoder is a bitmask of CRTC *indices*, which only
            // exist once `UnregisteredCrtc::new` runs -- but planes must exist before the CRTC that
            // references them. CRTCs are created here one per connector in order, so this
            // connector's CRTC index is `connector` and its mask is `1 << connector`.
            let crtc_mask = 1u32 << connector;
            let primary = plane::UnregisteredPlane::<VinoPlane>::new(
                dev,
                crtc_mask,
                if data.hdr_capable() {
                    &PRIMARY_FORMATS_HDR[..]
                } else {
                    &PRIMARY_FORMATS[..]
                },
                None,
                plane::Type::Primary,
                None,
                PlaneArgs {
                    connector: connector as u8,
                    is_cursor: false,
                },
            )?;
            // Tell compositors that this primary plane accepts the standard FB_DAMAGE_CLIPS
            // property. The scanout path already consumes those clips and emits only intersecting
            // 64x16 Haar strips, but without attaching the property KWin cannot provide them:
            // unchanged commits arrive with an empty clip list while real updates fall back to
            // ambiguous framebuffer swaps. That left the first keyframe frozen when empty damage
            // was correctly treated as a no-op, or forced multi-megabyte full frames when it was
            // treated as a repaint. EVDI exposes the same property before plane registration.
            primary.enable_fb_damage_clips();
            // Advertise every rotation vino's re-encode can produce by remapping source pixels
            // (`rot_src`): the four 90-degree rotations plus the two reflections.
            primary.create_rotation_property(
                plane::Rotation::ROTATE_0,
                plane::Rotation::ROTATE_0
                    | plane::Rotation::ROTATE_90
                    | plane::Rotation::ROTATE_180
                    | plane::Rotation::ROTATE_270
                    | plane::Rotation::REFLECT_X
                    | plane::Rotation::REFLECT_Y,
            )?;
            // A dock that composites no cursor of its own gets no cursor plane, rather than a
            // plane whose messages are then withheld: a cursor plane whose atomic commit succeeds
            // makes the compositor hand the pointer over and stop drawing its own, so starving one
            // loses the pointer entirely instead of falling back to software. A CRTC with no
            // cursor plane is how a driver says "draw it yourself".
            let cursor = if data.hw_cursor() {
                let cursor = plane::UnregisteredPlane::<VinoPlane>::new(
                    dev,
                    crtc_mask,
                    &CURSOR_FORMATS,
                    None,
                    plane::Type::Cursor,
                    None,
                    PlaneArgs {
                        connector: connector as u8,
                        is_cursor: true,
                    },
                )?;
                // An alpha framebuffer requires a blend-mode property. The dock composites the
                // cursor from a premultiplied bitmap, so premultiplied is the only supported mode.
                cursor.create_blend_mode_property(plane::BlendModes::PREMULTIPLIED)?;
                Some(cursor)
            } else {
                None
            };
            let crtc_obj = crtc::UnregisteredCrtc::<VinoCrtc>::new(
                dev,
                primary,
                cursor,
                None,
                connector as u8,
            )?;
            // Advertise CTM and a 256-entry GAMMA_LUT; the scanout applies both (cached via the
            // CRTC hooks). The dock has no colour hardware, so software application here is the
            // only place a compositor's correction can land -- KDE's Night Colour and GNOME's
            // Night Light drive these properties rather than rewriting the framebuffer.
            crtc_obj.enable_color_mgmt(0, true, crate::color::LUT_LEN as u32);
            let enc = encoder::UnregisteredEncoder::<VinoEncoder>::new(
                dev,
                encoder::Type::Virtual,
                crtc_obj.mask(),
                0,
                None,
                (),
            )?;
            let conn = connector::UnregisteredConnector::<VinoConnector>::new(
                dev,
                // DisplayPort connectors receive DRM's standard EDID property. A virtual connector
                // would not, and therefore could not publish the downstream monitor's modes.
                connector::Type::DisplayPort,
                connector as u8,
            )?;
            conn.attach_encoder(&*enc)?;
            // HDR is a property of the dock's pipeline, not of the monitor: a sink that declares
            // ST 2084 is useless if the dock cannot be told to carry ten bits. `hdr_capable`
            // keeps a Ridge connector from advertising an output it has no set-mode encoding for.
            //
            // The whole path exists now: the transfer function is offset-42 bit 6
            // (`ST2084 colorspace used (HDR)`, read out of DLM's own `setupVideo` decode),
            // `atomic_enable` takes it from this connector's HDR_OUTPUT_METADATA EOTF, and the
            // depth (offset 69) and DMA format (offset 23, `NM30`) go with it.
            //
            // Attaching these is what makes a compositor re-encode the desktop in PQ/BT.2020. If
            // the sink does not follow, the failure is a washed-out grey desktop that is invisible
            // on the wire: a capture shows correct PQ code words in both the working and the
            // broken case.
            if data.hdr_capable() {
                conn.attach_max_bpc_property(8, 10)?;
                conn.attach_colorspace_property()?;
                conn.attach_hdr_output_metadata_property();
            }
        }
        Ok(())
    }
}
