// SPDX-License-Identifier: GPL-2.0

//! What a dock will accept: pixel-clock ceilings, refresh ceilings and the shared bandwidth
//! budget a multi-connector commit has to fit inside.
//!
//! A dock silently refuses to light a mode past its budget rather than reporting anything, so
//! these are the checks that keep a connector from being handed one.

use super::*;

/// Per-mode pixel-clock ceiling in kHz for a dock whose profile has not been applied yet.
///
/// Ridge's DLM never programs above 497.75 MHz, so no Ridge capture fills the high half of the
/// offset-70 `u32`; this keeps the value that half can express on its own.
pub(super) const DEFAULT_MAX_HEAD_CLOCK_KHZ: u32 = 655_350;

/// Refresh ceiling for a dock whose profile has not been applied yet.
///
/// This is Ridge's limit, which is also DLM's: asked for 2560x1440@180 it puts 119.998 Hz on the
/// wire, and asked for @85 it programs the 59.95 Hz CVT-RB timing.
pub(super) const DEFAULT_MAX_REFRESH_HZ: u32 = 120;

/// Return the active pixel rate, saturating on invalidly large modes.
pub(crate) fn active_pixel_rate(hdisplay: u16, vdisplay: u16, vrefresh: i32) -> u32 {
    u32::from(hdisplay)
        .saturating_mul(u32::from(vdisplay))
        .saturating_mul(vrefresh.max(0) as u32)
}

/// Nonzero generation key for every deterministic field of a set-mode timing.
///
/// Zero means "disabled" in the atomics that carry this key.  This is a fingerprint rather than
/// a packed subset: porches, sync widths, allocation, VIC, depth and dual-pipe state all change
/// bytes the dock consumes and therefore all have to invalidate a previously active mode.
pub(crate) fn timing_key(t: &crate::cp::Timing) -> u64 {
    let mut hash = 0x7669_6e6f_6d6f_6465u64;
    for field in [
        u64::from(t.hactive),
        u64::from(t.hblank),
        u64::from(t.hsync_front),
        u64::from(t.hsync_width),
        u64::from(t.vactive),
        u64::from(t.vblank),
        u64::from(t.vsync_front),
        u64::from(t.vsync_width),
        u64::from(t.refresh_hz),
        u64::from(t.pixel_clock_10khz),
        u64::from(t.sync_flags),
        u64::from(t.stride),
        u64::from(t.total_rows),
        u64::from(t.vic_word),
        u64::from(t.ten_bit),
        u64::from(t.st2084),
        u64::from(t.dual_nivo),
    ] {
        hash = xxhash::xxh64(&field.to_le_bytes(), hash);
    }
    if hash == 0 {
        1
    } else {
        hash
    }
}

/// Whether a live connector already holds the exact effective Timing a command would send.
pub(crate) fn programmed_mode_matches(
    active_generation: u64,
    programmed: Option<crate::cp::Timing>,
    effective: crate::cp::Timing,
) -> bool {
    active_generation != 0 && programmed == Some(effective)
}

impl VinoDrmData {
    /// Exact Timing that would be put on the wire for the current requested topology.
    pub(super) fn effective_timing(
        &self,
        connector: usize,
        timing: &crate::cp::Timing,
    ) -> crate::cp::Timing {
        self.effective_timing_in_mask(connector, timing, self.requested_connector_mask())
    }

    /// Exact Timing for a stable requested-connector snapshot shared by a multi-connector
    /// transaction.
    pub(super) fn effective_timing_in_mask(
        &self,
        connector: usize,
        timing: &crate::cp::Timing,
        requested_heads: u32,
    ) -> crate::cp::Timing {
        crate::cp::Timing {
            dual_nivo: self.endpoint_is_shared_in_mask(connector, requested_heads),
            ..*timing
        }
    }

    /// Adopt an already-programmed exact mode under the caller's current request generation.
    ///
    /// A dynamic `dual_nivo` correction can make two request tokens differ while the exact timing
    /// the dock holds is unchanged. Conversely, equal tokens are not enough if endpoint topology
    /// changed. Compare the separately recorded wire state and change only `modeset_active`, whose
    /// job is to gate scanout against the current producer request. An explicit repair clears
    /// `modeset_active`, so it can never be optimized away here.
    pub(super) fn adopt_programmed_mode(
        &self,
        connector: usize,
        timing: &crate::cp::Timing,
        want: u64,
    ) -> bool {
        if connector >= MAX_CONNECTORS
            || self.modeset_requested[connector].load(Ordering::Acquire) != want
        {
            return false;
        }
        let active = self.modeset_active[connector].load(Ordering::Acquire);
        if !programmed_mode_matches(
            active,
            self.programmed_timing.lock()[connector],
            self.effective_timing(connector, timing),
        ) {
            return false;
        }
        // A disable that races this compare either clears `active` first (CAS fails) or clears it
        // after (the disable wins). A newer nonzero request may leave the old active token in
        // place, but scanout remains gated until its own command adopts or programs that request.
        self.modeset_active[connector]
            .compare_exchange(active, want, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The dock's total pixel-rate budget shared across all connectors.
    ///
    /// Zero means unknown and disables limiting.
    pub(super) fn dock_budget(&self) -> u32 {
        self.dock_pixel_budget
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Record this dock's pixel-rate budget, refresh ceiling and pixel-clock ceiling.
    pub(crate) fn set_mode_limits(
        &self,
        pixel_budget: u32,
        max_refresh_hz: u32,
        max_connector_clock_khz: u32,
    ) {
        self.dock_pixel_budget
            .store(pixel_budget, core::sync::atomic::Ordering::Relaxed);
        self.max_refresh_hz.store(
            if max_refresh_hz == 0 {
                DEFAULT_MAX_REFRESH_HZ
            } else {
                max_refresh_hz
            },
            core::sync::atomic::Ordering::Relaxed,
        );
        self.max_connector_clock_khz.store(
            if max_connector_clock_khz == 0 {
                DEFAULT_MAX_HEAD_CLOCK_KHZ
            } else {
                max_connector_clock_khz
            },
            core::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Highest per-mode pixel clock in kHz this dock is known to accept.
    pub(crate) fn max_connector_clock_khz(&self) -> u32 {
        self.max_connector_clock_khz
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Highest refresh rate this dock is known to drive.
    pub(crate) fn max_refresh_hz(&self) -> u32 {
        self.max_refresh_hz
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Whether DRM's rounded refresh rate is within this dock's limit.
    pub(super) fn refresh_within_limit(&self, vrefresh: i32) -> bool {
        vrefresh <= 0 || (vrefresh as u32) <= self.max_refresh_hz()
    }

    /// Combined pixel rate of every connector *except* `connector` that currently has a mode driven
    /// onto it.
    ///
    /// A connector the commit carries is taken from the commit; every other connector is taken from
    /// what is programmed. Both halves are needed: a commit reconfiguring several connectors at
    /// once must be weighed at the rates it is asking for, and a connector standing outside it
    /// still spends what it was last given.
    ///
    /// Only active connectors consume the shared limit. `last_timing` survives `atomic_disable`, so
    /// activity is read from `modeset_requested`, which is cleared on disable.
    ///
    /// A connector whose monitor has gone is not spending anything either, whatever mode it was
    /// last asked for: its downstream sink is already torn down. That has to be read from presence
    /// rather than from the mode state, because a removal and the arrival that replaces it reach
    /// userspace as two events -- moving a monitor from one socket to another is checked at its new
    /// socket while the disable of the old one has not been committed yet, and charging the dock
    /// for the dark connector is what refuses the new one its mode.
    pub(super) fn other_connectors_rate(
        &self,
        state: &kernel::drm::kms::atomic::AtomicStateMutator<VinoDrmDriver>,
        connector: usize,
    ) -> u32 {
        // A connector this commit describes is charged what the commit gives it. Reading its
        // programmed rate instead would price every connector in a multi-connector commit at what
        // it is leaving, so a pair that rises together would be admitted at the sum of the rates it
        // is abandoning.
        let mut proposed: [Option<u32>; MAX_CONNECTORS] = [None; MAX_CONNECTORS];
        state.for_each_new_crtc_state(|crtc, crtc_state| {
            let Some(slot) = proposed.get_mut(crtc.connector as usize) else {
                return;
            };
            *slot = Some(if crtc_state.active() {
                let m = crtc_state.mode();
                active_pixel_rate(m.hdisplay(), m.vdisplay(), m.vrefresh())
            } else {
                0
            });
        });

        let timings = *self.last_timing.lock();
        let mut total: u32 = 0;
        for (i, t) in timings.iter().enumerate() {
            if i == connector {
                continue;
            }
            if let Some(rate) = proposed[i] {
                total = total.saturating_add(rate);
                continue;
            }
            if self.modeset_requested[i].load(Ordering::Acquire) == 0 || !self.connector_present(i)
            {
                continue;
            }
            if let Some(t) = t {
                total = total.saturating_add(
                    u32::from(t.hactive)
                        .saturating_mul(u32::from(t.vactive))
                        .saturating_mul(u32::from(t.refresh_hz)),
                );
            }
        }
        total
    }
}

#[cfg(CONFIG_DRM_VINO_KUNIT_TEST)]
#[kunit_tests(vino_mode_limits)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn timing_key_covers_every_set_mode_field() {
        let base = cp::Timing {
            hactive: 1920,
            hblank: 280,
            hsync_front: 88,
            hsync_width: 44,
            vactive: 1080,
            vblank: 45,
            vsync_front: 4,
            vsync_width: 5,
            refresh_hz: 60,
            pixel_clock_10khz: 14_850,
            sync_flags: 0x0400,
            stride: 0x0800,
            total_rows: 0x2000,
            vic_word: 0x2810,
            ten_bit: false,
            st2084: false,
            dual_nivo: false,
        };
        let timings = [
            base,
            cp::Timing {
                hactive: 1921,
                ..base
            },
            cp::Timing {
                hblank: 281,
                ..base
            },
            cp::Timing {
                hsync_front: 89,
                ..base
            },
            cp::Timing {
                hsync_width: 45,
                ..base
            },
            cp::Timing {
                vactive: 1081,
                ..base
            },
            cp::Timing { vblank: 46, ..base },
            cp::Timing {
                vsync_front: 5,
                ..base
            },
            cp::Timing {
                vsync_width: 6,
                ..base
            },
            // Exercise the high byte that the former packed key discarded.
            cp::Timing {
                refresh_hz: 0x013c,
                ..base
            },
            // Exercise bits above the former 22-bit pixel-clock mask.
            cp::Timing {
                pixel_clock_10khz: 0x0140_3a02,
                ..base
            },
            cp::Timing {
                sync_flags: 0x0401,
                ..base
            },
            cp::Timing {
                stride: 0x0880,
                ..base
            },
            cp::Timing {
                total_rows: 0x2001,
                ..base
            },
            cp::Timing {
                vic_word: 0x281f,
                ..base
            },
            cp::Timing {
                ten_bit: true,
                ..base
            },
            cp::Timing {
                st2084: true,
                ..base
            },
            cp::Timing {
                dual_nivo: true,
                ..base
            },
        ];
        let mut keys = [0u64; 18];
        for (i, timing) in timings.iter().enumerate() {
            keys[i] = timing_key(timing);
            assert_ne!(keys[i], 0);
            for previous in &keys[..i] {
                assert_ne!(keys[i], *previous);
            }
        }
    }

    #[test]
    fn no_op_mode_set_compares_the_exact_programmed_timing() {
        let raw = cp::Timing {
            hactive: 2560,
            hblank: 160,
            hsync_front: 48,
            hsync_width: 32,
            vactive: 1440,
            vblank: 41,
            vsync_front: 3,
            vsync_width: 5,
            refresh_hz: 60,
            pixel_clock_10khz: 24_150,
            sync_flags: 0x0600,
            stride: 0x0a80,
            total_rows: 0x66db,
            vic_word: 0x0800,
            ten_bit: false,
            st2084: false,
            dual_nivo: false,
        };
        let effective = cp::Timing {
            dual_nivo: true,
            ..raw
        };

        // The request can have been queued before an endpoint partner appeared, so its raw token
        // and the exact Timing corrected at send time legitimately differ.
        assert_ne!(timing_key(&raw), timing_key(&effective));
        assert!(programmed_mode_matches(
            timing_key(&raw),
            Some(effective),
            effective
        ));
        assert!(!programmed_mode_matches(
            timing_key(&raw),
            Some(raw),
            effective
        ));
        // Clearing the active generation is an explicit request to touch hardware, even if an old
        // programmed-state snapshot remains available for diagnostics.
        assert!(!programmed_mode_matches(0, Some(effective), effective));
    }

    /// The boundary cases matter most: each must pass by equality. A `<` would prune a dock's
    /// working configuration and dark its panels.
    #[test]
    fn mode_ceilings_bound_bandwidth_not_refresh() {
        let refresh_ok =
            |p: &DockProfile, hz: i32| hz <= 0 || (hz as u32) <= p.capabilities.max_refresh_hz;
        let clock_ok = |p: &DockProfile, khz: u32| khz <= p.capabilities.max_connector_clock_khz;
        let rate = active_pixel_rate;

        // Ridge carries 2560x1440p144, so no refresh cap may hide it: 597.29 MHz of clock and
        // 530,841,600 pixels per second both sit inside its ceilings. The 180 Hz request DLM
        // answers with 119.998 Hz is 746.64 MHz, which the clock ceiling already refuses -- that
        // is the whole of what a refresh cap here would have bought.
        assert!(refresh_ok(&profile::PROFILE_RIDGE, 144));
        assert!(clock_ok(&profile::PROFILE_RIDGE, 597_290));
        assert!(rate(2560, 1440, 144) <= profile::PROFILE_RIDGE.capabilities.pixel_budget);
        assert!(!clock_ok(&profile::PROFILE_RIDGE, 746_640));

        // The DL7400 is bounded by link rate alone too -- DLM drives it at 2560x1440@164.96.
        assert!(
            refresh_ok(&profile::PROFILE_NAVARRO, 180)
                && refresh_ok(&profile::PROFILE_NAVARRO, 240)
        );

        // 2560x1440: p165 is 699.50 MHz and carried; p180 is 714.81 MHz and is the mode the dock
        // accepts and then fails to deliver.
        assert!(clock_ok(&profile::PROFILE_NAVARRO, 699_500));
        assert!(!clock_ok(&profile::PROFILE_NAVARRO, 714_810));
        // Ridge carries 2560x1440p144 at 597.29 MHz and blanks the sink at p165's 699.50 MHz.
        assert!(
            clock_ok(&profile::PROFILE_RIDGE, 597_290)
                && !clock_ok(&profile::PROFILE_RIDGE, 699_500)
        );

        // A degenerate mode reports 0 Hz and carries no rate information; a signed refresh must
        // never be read as a huge unsigned one.
        assert!(
            refresh_ok(&profile::PROFILE_NAVARRO, 0) && refresh_ok(&profile::PROFILE_NAVARRO, -1)
        );

        // Each budget admits its own dual-connector configuration and nothing beyond it.
        assert_eq!(rate(2560, 1440, 120), 442_368_000);
        // Ridge sustains 2560x1440p144 beside 2560x1440p120, so its budget must admit that pair.
        assert_eq!(
            profile::PROFILE_RIDGE.capabilities.pixel_budget,
            rate(2560, 1440, 144) + rate(2560, 1440, 120)
        );
        assert_eq!(
            profile::PROFILE_NAVARRO.capabilities.pixel_budget,
            2 * rate(2560, 1440, 165)
        );
        assert_eq!(rate(65535, 65535, 65535), u32::MAX); // saturates, never wraps small
        assert_eq!(rate(2560, 1440, -1), 0);
    }
}
