// SPDX-License-Identifier: GPL-2.0

#include <linux/workqueue.h>

__rust_helper void rust_helper_init_work_with_key(struct work_struct *work,
						  work_func_t func,
						  bool onstack,
						  const char *name,
						  struct lock_class_key *key)
{
	__init_work(work, onstack);
	work->data = (atomic_long_t)WORK_DATA_INIT();
	lockdep_init_map(&work->lockdep_map, name, key, 0);
	INIT_LIST_HEAD(&work->entry);
	work->func = func;
}

/*
 * `alloc_workqueue()` is variadic (the name is a format string), so bindgen cannot generate a
 * callable binding for it. Wrap it with a fixed `"%s"` format so Rust can pass a plain name.
 */
__rust_helper struct workqueue_struct *
rust_helper_alloc_workqueue(const char *name, unsigned int flags, int max_active)
{
	return alloc_workqueue("%s", flags, max_active, name);
}
