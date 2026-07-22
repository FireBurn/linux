// SPDX-License-Identifier: GPL-2.0

#include <drm/drm_framebuffer.h>

__rust_helper void rust_helper_drm_framebuffer_get(struct drm_framebuffer *fb)
{
	drm_framebuffer_get(fb);
}

__rust_helper void rust_helper_drm_framebuffer_put(struct drm_framebuffer *fb)
{
	drm_framebuffer_put(fb);
}
