// SPDX-License-Identifier: GPL-2.0
//
// The EVDI "painter": the per-device connection + event-delivery bookkeeping that
// bridges the KMS callbacks (and ioctls) to the DisplayLinkManager userspace client.
//
// Events are delivered through the DRM-core `drm_event` mechanism (the safe
// `kernel::drm::event::EventChannel` binding), which serializes delivery against file
// close under `event_lock` — so an event can never be sent to a client that has just
// disconnected.

use kernel::drm::event::EventPayload;

use crate::kms::{EvdiDrmData, EvdiDrmDevice};
use crate::uapi;

/// DPMS mode codes as understood by the DLM client (matching `DRM_MODE_DPMS_*`).
pub(crate) const DPMS_ON: i32 = 0;
pub(crate) const DPMS_OFF: i32 = 3;

/// Mutable per-device painter state, guarded by a mutex in [`EvdiDrmData`].
///
/// The connected client's EDID is the connector's `cached_edid` (the source of truth for the mode
/// list), so it is not duplicated here.
pub(crate) struct PainterState {
    /// Whether a DLM client has issued CONNECT.
    pub(crate) connected: bool,
    /// Whether the client asked to receive cursor events.
    pub(crate) cursor_events_enabled: bool,
    /// A frame has been flipped in but not yet grabbed (the C evdi's `num_dirts > 0`). Lets
    /// REQUEST_UPDATE answer "grab now" (ioctl returns 1) when fresh pixels are already waiting,
    /// instead of self-triggering an UPDATE_READY event (which busy-loops the client).
    pub(crate) frame_dirty: bool,
    /// The individual regions changed since the last GRABPIX, accumulated across flips. GRABPIX
    /// reports and copies exactly these rectangles (so DLM transmits just the deltas rather than the
    /// whole 2560x1440 frame), then clears them.
    pub(crate) damage: Damage,
}

/// Maximum number of distinct damage rectangles tracked between grabs (mirrors the C evdi's
/// `MAX_DIRTS`); on overflow they collapse into a single bounding box.
pub(crate) const MAX_DAMAGE_RECTS: usize = 16;

/// Accumulated frame damage: up to [`MAX_DAMAGE_RECTS`] changed rectangles `(x1, y1, x2, y2)` since
/// the last GRABPIX. `count == 0` means nothing was recorded (GRABPIX falls back to a full frame).
#[derive(Copy, Clone)]
pub(crate) struct Damage {
    pub(crate) rects: [(i32, i32, i32, i32); MAX_DAMAGE_RECTS],
    pub(crate) count: usize,
}

impl Damage {
    pub(crate) const fn new() -> Self {
        Self {
            rects: [(0, 0, 0, 0); MAX_DAMAGE_RECTS],
            count: 0,
        }
    }

    /// Record a changed rectangle. When the list is full, collapse everything (including `r`) into a
    /// single bounding box so a burst of damage never drops updates -- it just gets coarser.
    pub(crate) fn push(&mut self, r: (i32, i32, i32, i32)) {
        if self.count < MAX_DAMAGE_RECTS {
            self.rects[self.count] = r;
            self.count += 1;
            return;
        }
        let mut bb = r;
        for i in 0..self.count {
            let (x1, y1, x2, y2) = self.rects[i];
            bb = (bb.0.min(x1), bb.1.min(y1), bb.2.max(x2), bb.3.max(y2));
        }
        self.rects[0] = bb;
        self.count = 1;
    }

    pub(crate) fn clear(&mut self) {
        self.count = 0;
    }
}

impl PainterState {
    pub(crate) fn new() -> Self {
        Self {
            connected: false,
            cursor_events_enabled: false,
            frame_dirty: false,
            damage: Damage::new(),
        }
    }
}

// SAFETY: each of these event structs is `#[repr(C)]` with a `drm_event`-layout header
// (`uapi::DrmEvent`, identical to `bindings::drm_event`) as its first field, is `Copy`
// plain-old-data, and is copied verbatim to userspace. See `uapi.rs`.
unsafe impl EventPayload for uapi::DrmEvdiEventUpdateReady {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_UPDATE_READY;
}
// SAFETY: See above.
unsafe impl EventPayload for uapi::DrmEvdiEventDpms {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_DPMS;
}
// SAFETY: See above.
unsafe impl EventPayload for uapi::DrmEvdiEventModeChanged {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_MODE_CHANGED;
}
// SAFETY: See above.
unsafe impl EventPayload for uapi::DrmEvdiEventCrtcState {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_CRTC_STATE;
}
// SAFETY: See above.
unsafe impl EventPayload for uapi::DrmEvdiEventCursorSet {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_CURSOR_SET;
}
// SAFETY: See above.
unsafe impl EventPayload for uapi::DrmEvdiEventCursorMove {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_CURSOR_MOVE;
}
// SAFETY: See above.
unsafe impl EventPayload for uapi::DrmEvdiEventDdcciData {
    const TYPE: u32 = uapi::DRM_EVDI_EVENT_DDCCI_DATA;
}

/// Zeroed `drm_event` header; [`EventChannel::send`] overwrites `type`/`length`.
const fn hdr() -> uapi::DrmEvent {
    uapi::DrmEvent {
        type_: 0,
        length: 0,
    }
}

/// Tell the DLM client a fresh frame is ready to be grabbed (`UPDATE_READY`).
pub(crate) fn notify_update_ready(data: &EvdiDrmData, _dev: &EvdiDrmDevice) {
    let ev = uapi::DrmEvdiEventUpdateReady { base: hdr() };
    let _ = data.events.send(ev);
}

/// Tell the DLM client the display's DPMS power state changed.
pub(crate) fn notify_dpms(data: &EvdiDrmData, _dev: &EvdiDrmDevice, mode: i32) {
    let ev = uapi::DrmEvdiEventDpms { base: hdr(), mode };
    let _ = data.events.send(ev);
}

/// Tell the DLM client the negotiated mode changed.
pub(crate) fn notify_mode_changed(
    data: &EvdiDrmData,
    _dev: &EvdiDrmDevice,
    hdisplay: i32,
    vdisplay: i32,
    vrefresh: i32,
    bits_per_pixel: i32,
    pixel_format: u32,
) {
    let ev = uapi::DrmEvdiEventModeChanged {
        base: hdr(),
        hdisplay,
        vdisplay,
        vrefresh,
        bits_per_pixel,
        pixel_format,
    };
    let _ = data.events.send(ev);
}
