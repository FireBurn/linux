/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note
 *
 * Copyright (c) 2015 - 2020 DisplayLink (UK) Ltd.
 */

#ifndef __UAPI_EVDI_DRM_H__
#define __UAPI_EVDI_DRM_H__

#include <linux/types.h>
#include <drm/drm.h>

/* Output events sent from the driver to libevdi. */
#define DRM_EVDI_EVENT_UPDATE_READY	0x80000000
#define DRM_EVDI_EVENT_DPMS		0x80000001
#define DRM_EVDI_EVENT_MODE_CHANGED	0x80000002
#define DRM_EVDI_EVENT_CRTC_STATE	0x80000003
#define DRM_EVDI_EVENT_CURSOR_SET	0x80000004
#define DRM_EVDI_EVENT_CURSOR_MOVE	0x80000005
#define DRM_EVDI_EVENT_DDCCI_DATA	0x80000006

struct drm_evdi_event_update_ready {
	struct drm_event base;
};

struct drm_evdi_event_dpms {
	struct drm_event base;
	__s32 mode;
};

struct drm_evdi_event_mode_changed {
	struct drm_event base;
	__s32 hdisplay;
	__s32 vdisplay;
	__s32 vrefresh;
	__s32 bits_per_pixel;
	__u32 pixel_format;
};

struct drm_evdi_event_crtc_state {
	struct drm_event base;
	__s32 state;
};

struct drm_evdi_connect {
	__s32 connected;
	__s32 dev_index;
	const __u8 * __user edid;
	__u32 edid_length;
	__u32 pixel_area_limit;
	__u32 pixel_per_second_limit;
};

struct drm_evdi_request_update {
	__s32 reserved;
};

enum drm_evdi_grabpix_mode {
	EVDI_GRABPIX_MODE_RECTS = 0,
	EVDI_GRABPIX_MODE_DIRTY = 1,
};

struct drm_evdi_grabpix {
	enum drm_evdi_grabpix_mode mode;
	__s32 buf_width;
	__s32 buf_height;
	__s32 buf_byte_stride;
	__u8 __user *buffer;
	__s32 num_rects;
	struct drm_clip_rect __user *rects;
};

struct drm_evdi_event_cursor_set {
	struct drm_event base;
	__s32 hot_x;
	__s32 hot_y;
	__u32 width;
	__u32 height;
	__u8 enabled;
	__u32 buffer_handle;
	__u32 buffer_length;
	__u32 pixel_format;
	__u32 stride;
};

struct drm_evdi_event_cursor_move {
	struct drm_event base;
	__s32 x;
	__s32 y;
};

struct drm_evdi_ddcci_response {
	const __u8 * __user buffer;
	__u32 buffer_length;
	__u8 result;
};

struct drm_evdi_enable_cursor_events {
	struct drm_event base;
	__u8 enable;
};

#define DDCCI_BUFFER_SIZE 64

struct drm_evdi_event_ddcci_data {
	struct drm_event base;
	__u8 buffer[DDCCI_BUFFER_SIZE];
	__u32 buffer_length;
	__u16 flags;
	__u16 address;
};

/* Input ioctls from libevdi to the driver. */
#define DRM_EVDI_CONNECT		0x00
#define DRM_EVDI_REQUEST_UPDATE		0x01
#define DRM_EVDI_GRABPIX		0x02
#define DRM_EVDI_DDCCI_RESPONSE		0x03
#define DRM_EVDI_ENABLE_CURSOR_EVENTS	0x04

/* This is an enum so that it can be resolved by Rust bindgen. */
enum {
	DRM_IOCTL_EVDI_CONNECT = DRM_IOWR(DRM_COMMAND_BASE +
		DRM_EVDI_CONNECT, struct drm_evdi_connect),
	DRM_IOCTL_EVDI_REQUEST_UPDATE = DRM_IOWR(DRM_COMMAND_BASE +
		DRM_EVDI_REQUEST_UPDATE, struct drm_evdi_request_update),
	DRM_IOCTL_EVDI_GRABPIX = DRM_IOWR(DRM_COMMAND_BASE +
		DRM_EVDI_GRABPIX, struct drm_evdi_grabpix),
	DRM_IOCTL_EVDI_DDCCI_RESPONSE = DRM_IOWR(DRM_COMMAND_BASE +
		DRM_EVDI_DDCCI_RESPONSE, struct drm_evdi_ddcci_response),
	DRM_IOCTL_EVDI_ENABLE_CURSOR_EVENTS = DRM_IOWR(DRM_COMMAND_BASE +
		DRM_EVDI_ENABLE_CURSOR_EVENTS, struct drm_evdi_enable_cursor_events),
};

#endif /* __UAPI_EVDI_DRM_H__ */
