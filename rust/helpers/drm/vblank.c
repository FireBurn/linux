// SPDX-License-Identifier: GPL-2.0

#include <drm/drm_vblank.h>

__rust_helper struct drm_vblank_crtc *
rust_helper_drm_crtc_vblank_crtc(struct drm_crtc *crtc)
{
	return drm_crtc_vblank_crtc(crtc);
}
