// SPDX-License-Identifier: GPL-2.0

//! The dock's USB endpoints, and the I/O handle every transfer goes through.
//!
//! [`Endpoints`] is resolved once against interface 0's descriptor and direction/type-checked, so
//! the rest of the driver names an endpoint by what it is rather than by a bare address. A
//! [`UsbLink`] can only exist while the device's [`usb::IoWindow`] is open, so a transfer cannot
//! outlive the binding.

use super::*;

/// The dock's endpoints, resolved once against interface 0's descriptor and validated for
/// direction and transfer type by [`usb::Interface::endpoint`].
///
/// Resolving up front means the rest of the driver names an endpoint by what it *is* rather than
/// by a bare address, so a bulk-OUT transfer cannot be aimed at the interrupt-IN status endpoint.
/// The whole set is [`Copy`] (each entry is an address plus a max-packet size), so it can be
/// carried by value in a [`UsbLink`] rather than borrowed.
#[derive(Clone, Copy)]
pub(crate) struct Endpoints {
    /// EP02: host->dock control-plane bulk writes.
    pub(crate) ctrl_out: usb::Endpoint<usb::BulkOut>,
    /// EP84: dock->host control-plane bulk replies.
    pub(crate) ctrl_in: usb::Endpoint<usb::BulkIn>,
    /// Per-head video bulk-OUT endpoints, from [`DockProfile::video_eps`].
    pub(crate) video: [usb::Endpoint<usb::BulkOut>; drm_sink::HEADS],
}

impl Endpoints {
    /// Resolves every endpoint the driver uses against `intf`'s active alternate setting.
    ///
    /// The control endpoints and the video endpoints are required; the interrupt status endpoint
    /// is optional because only the bring-up probe reads it.
    pub(crate) fn resolve<Ctx: device::DeviceContext>(
        intf: &usb::Interface<Ctx>,
        profile: &DockProfile,
    ) -> Result<Self> {
        let mut video = [intf.endpoint::<usb::BulkOut>(profile.video_eps[0])?; drm_sink::HEADS];
        for (slot, addr) in video.iter_mut().zip(profile.video_eps).skip(1) {
            *slot = intf.endpoint::<usb::BulkOut>(addr)?;
        }

        Ok(Self {
            ctrl_out: intf.endpoint::<usb::BulkOut>(EP_CTRL_OUT)?,
            ctrl_in: intf.endpoint::<usb::BulkIn>(EP_CTRL_IN)?,
            video,
        })
    }
}

/// A live USB transfer handle: an [`usb::Io`] token proving I/O is currently permitted, plus the
/// resolved [`Endpoints`].
///
/// Obtaining one requires the device's [`usb::IoWindow`] to still be open, so a transfer cannot be
/// issued after `disconnect()` has closed it. Carrying the endpoints in the handle also keeps raw
/// endpoint addresses out of transfer call sites.
pub(crate) struct UsbLink<'a> {
    window: &'a Arc<usb::IoWindow>,
    io: usb::Io<'a>,
    pub(crate) eps: Endpoints,
}

impl<'a> UsbLink<'a> {
    /// Opens a link on `window`, failing with `ENODEV` once the window has been closed.
    pub(crate) fn open(window: &'a Arc<usb::IoWindow>, eps: Endpoints) -> Result<Self> {
        Ok(Self {
            io: window.enter()?,
            window,
            eps,
        })
    }

    /// Opens the persistent EP84 control-plane reader.
    pub(crate) fn ctrl_in_queue(&self, depth: usize, buf_len: usize) -> Result<usb::BulkInQueue> {
        usb::BulkInQueue::new(self.window, &self.io, &self.eps.ctrl_in, depth, buf_len)
    }

    /// Opens the pipelined EP02 control-plane writer.
    pub(crate) fn ctrl_out_queue(&self, depth: usize, buf_len: usize) -> Result<usb::BulkOutQueue> {
        usb::BulkOutQueue::new(self.window, &self.io, &self.eps.ctrl_out, depth, buf_len)
    }

    /// Returns the canonical queue slot for `head`'s physical video endpoint.
    ///
    /// A dock profile may repeat an endpoint address for multiple physical connectors (Navarro
    /// uses EP08 for connectors 0/2 and EP0a for 1/3).  Those connectors must share one
    /// persistent queue: separate queues could submit interleaved URBs to the same pipe.
    pub(crate) fn video_pipe_index(&self, head: usize) -> Result<usize> {
        let endpoint = self.eps.video.get(head).ok_or(EINVAL)?;
        self.eps
            .video
            .iter()
            .position(|candidate| candidate.address() == endpoint.address())
            .ok_or(EINVAL)
    }

    /// Opens the pipelined video writer for `head`'s physical video endpoint.
    pub(crate) fn video_queue(
        &self,
        head: usize,
        depth: usize,
        buf_len: usize,
    ) -> Result<usb::BulkOutQueue> {
        let ep = self.eps.video.get(head).ok_or(EINVAL)?;
        usb::BulkOutQueue::new(self.window, &self.io, ep, depth, buf_len)
    }

    /// Writes one video transfer synchronously for transport diagnostics.
    ///
    /// This is deliberately not described as DLM's normal transport. The authenticated Navarro
    /// capture reaches eight outstanding video URBs. Only its first prologue chunk is synchronous;
    /// after that completion and a measured producer delay, DLM pipelines the remaining chunks.
    pub(crate) fn video_send(
        &self,
        head: usize,
        data: &[u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result<usize> {
        let ep = self.eps.video.get(head).ok_or(EINVAL)?;
        self.io.bulk_send(ep, data, timeout, gfp)
    }

    /// The underlying I/O token, for the paths that open a persistent queue.
    pub(crate) fn io(&self) -> &usb::Io<'a> {
        &self.io
    }

    /// Writes a control-plane message to EP02.
    pub(crate) fn ctrl_send(&self, data: &[u8], timeout: Delta, gfp: Flags) -> Result<usize> {
        self.io.bulk_send(&self.eps.ctrl_out, data, timeout, gfp)
    }

    /// Reads a control-plane reply from EP84.
    pub(crate) fn ctrl_recv(&self, data: &mut [u8], timeout: Delta, gfp: Flags) -> Result<usize> {
        self.io.bulk_recv(&self.eps.ctrl_in, data, timeout, gfp)
    }

    /// Clears a stall on `head`'s video endpoint.
    pub(crate) fn clear_video_halt(&self, head: usize) -> Result {
        self.io.clear_halt(self.eps.video.get(head).ok_or(EINVAL)?)
    }

    /// Reads the standard two-byte status word for a video endpoint.
    ///
    /// Bit zero is `ENDPOINT_HALT`.  This is deliberately read-only: it distinguishes a dock
    /// which has stalled the pipe from one which is merely returning NRDY without perturbing the
    /// endpoint's host or device sequence state.
    pub(crate) fn video_endpoint_status(&self, head: usize) -> Result<u16> {
        let address = self.eps.video.get(head).ok_or(EINVAL)?.address();
        let mut status = [0u8; 2];
        self.control_recv(
            0x00, // USB_REQ_GET_STATUS
            0x82, // device-to-host, standard, endpoint
            0,
            u16::from(address),
            &mut status,
            timeout(),
            GFP_KERNEL,
        )?;
        Ok(u16::from_le_bytes(status))
    }

    /// Sends the standard endpoint-halt clear without running Linux's host endpoint reset helper.
    ///
    /// Navarro uses this only during cold setup, before any video URB has touched the endpoint, so
    /// the host sequence state is already its enumeration value. The working DLM wire transaction
    /// is two raw `CLEAR_FEATURE(ENDPOINT_HALT)` requests 12.647 ms apart followed by its vendor
    /// commit 143 us later. `usb_clear_halt()` adds an xHCI endpoint reset after the request; that
    /// made each Vino operation return about 18 ms late and changed this transaction's ordering.
    pub(crate) fn clear_video_halt_wire(&self, head: usize) -> Result {
        let address = self.eps.video.get(head).ok_or(EINVAL)?.address();
        self.control_send(
            0x01, // USB_REQ_CLEAR_FEATURE
            0x02, // host-to-device, standard, endpoint
            0,    // USB_ENDPOINT_HALT
            u16::from(address),
            &[],
            timeout(),
            GFP_KERNEL,
        )
    }

    /// Issues a control OUT transfer on EP0.
    pub(crate) fn control_send(
        &self,
        request: u8,
        request_type: u8,
        value: u16,
        index: u16,
        data: &[u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result {
        self.io
            .control_send(request, request_type, value, index, data, timeout, gfp)
    }

    /// Issues a control IN transfer on EP0.
    pub(crate) fn control_recv(
        &self,
        request: u8,
        request_type: u8,
        value: u16,
        index: u16,
        data: &mut [u8],
        timeout: Delta,
        gfp: Flags,
    ) -> Result {
        self.io
            .control_recv(request, request_type, value, index, data, timeout, gfp)
    }

    /// Selects an alternate setting on the driver's own interface.
    pub(crate) fn set_alternate_setting(&self, alternate: u8) -> Result {
        self.io.set_alternate_setting(alternate)
    }
}
/// EP84 (dock-to-host) transfer size.
///
/// Replies larger than one transfer are delivered as consecutive fragments.
pub(crate) const EP84_BUF: usize = 4096;
