# SPDX-License-Identifier: GPL-2.0-only

snd-soc-avs-y := dsp.o ipc.o messages.o utils.o core.o loader.o \
		 topology.o path.o pcm.o board_selection.o control.o \
		 sysfs.o
snd-soc-avs-y += cldma.o
snd-soc-avs-y += skl.o apl.o cnl.o icl.o tgl.o mtl.o lnl.o ptl.o

snd-soc-avs-y += trace.o
# tell define_trace.h where to find the trace header
CFLAGS_trace.o := -I$(src)

ifneq ($(CONFIG_DEBUG_FS),)
snd-soc-avs-y += probes.o debugfs.o
endif

obj-$(CONFIG_SND_SOC_INTEL_AVS) += snd-soc-avs.o

# Machine support
obj-$(CONFIG_SND_SOC) += boards/
