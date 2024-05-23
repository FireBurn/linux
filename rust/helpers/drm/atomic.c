// SPDX-License-Identifier: GPL-2.0

#include <drm/drm_atomic.h>

void rust_helper_drm_atomic_state_get(struct drm_atomic_state *state)
{
	drm_atomic_state_get(state);
}

void rust_helper_drm_atomic_state_put(struct drm_atomic_state *state)
{
	drm_atomic_state_put(state);
}

// Macros for generating one repetitive atomic state accessors (like drm_atomic_get_new_plane_state)
#define STATE_FUNC(type, tense)                                                                     \
	struct drm_ ## type ## _state *rust_helper_drm_atomic_get_ ## tense ## _ ## type ## _state( \
		const struct drm_atomic_state *state,                                               \
		struct drm_ ## type *type                                                           \
	) {                                                                                         \
		return drm_atomic_get_## tense ## _ ## type ## _state(state, type);                 \
	}
#define STATE_FUNCS(type) \
	STATE_FUNC(type, new); \
	STATE_FUNC(type, old);

STATE_FUNCS(plane);
STATE_FUNCS(crtc);
STATE_FUNCS(connector);

#undef STATE_FUNCS
#undef STATE_FUNC
