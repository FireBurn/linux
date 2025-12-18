/* SPDX-License-Identifier: GPL-2.0 */
#ifndef AMDKCL_HYPERVISOR_H
#define AMDKCL_HYPERVISOR_H

#include <asm/hypervisor.h>

#ifdef CONFIG_X86

#ifndef HAVE_HYPERVISOR_IS_TYPE
static inline bool hypervisor_is_type(enum x86_hypervisor_type type)
{
	return false;
}
#endif

#endif /* CONFIG_X86 */
#endif /* AMDKCL_HYPERVISOR_H */
