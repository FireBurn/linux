// SPDX-License-Identifier: GPL-2.0

#ifdef CONFIG_DRM
#ifdef CONFIG_DRM_KMS_HELPER
#include "atomic.c"
#include "framebuffer.c"
#include "vblank.c"
#endif

#include "gem.c"
#include "vma_manager.c"
#endif
