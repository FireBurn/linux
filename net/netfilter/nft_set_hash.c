// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (c) 2008-2014 Patrick McHardy <kaber@trash.net>
 *
 * Development of this code funded by Astaro AG (http://www.astaro.com/)
 */

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/list.h>
#include <linux/log2.h>
#include <linux/jhash.h>
#include <linux/netlink.h>
#include <linux/workqueue.h>
#include <linux/rhashtable.h>
#include <linux/netfilter.h>
#include <linux/netfilter/nf_tables.h>
#include <net/netfilter/nf_tables_core.h>

/* We target a hash table size of 4, element hint is 75% of final size */
#define NFT_RHASH_ELEMENT_HINT 3

struct nft_rhash {
	struct rhashtable		ht;
	struct delayed_work		gc_work;
	u32				wq_gc_seq;
};

struct nft_rhash_elem {
	struct nft_elem_priv		priv;
	struct rhash_head		node;
	u32				wq_gc_seq;
	struct nft_set_ext		ext;
};

struct nft_rhash_cmp_arg {
	const struct nft_set		*set;
	const u32			*key;
	u8				genmask;
	u64				tstamp;
};

static inline u32 nft_rhash_key(const void *data, u32 len, u32 seed)
{
	const struct nft_rhash_cmp_arg *arg = data;

	return jhash(arg->key, len, seed);
}

static inline u32 nft_rhash_obj(const void *data, u32 len, u32 seed)
{
	const struct nft_rhash_elem *he = data;

	return jhash(nft_set_ext_key(&he->ext), len, seed);
}

static inline int nft_rhash_cmp(struct rhashtable_compare_arg *arg,
				const void *ptr)
{
	const struct nft_rhash_cmp_arg *x = arg->key;
	const struct nft_rhash_elem *he = ptr;

	if (memcmp(nft_set_ext_key(&he->ext), x->key, x->set->klen))
		return 1;
	if (nft_set_elem_is_dead(&he->ext))
		return 1;
	if (__nft_set_elem_expired(&he->ext, x->tstamp))
		return 1;
	if (!nft_set_elem_active(&he->ext, x->genmask))
		return 1;
	return 0;
}

static const struct rhashtable_params nft_rhash_params = {
	.head_offset		= offsetof(struct nft_rhash_elem, node),
	.hashfn			= nft_rhash_key,
	.obj_hashfn		= nft_rhash_obj,
	.obj_cmpfn		= nft_rhash_cmp,
	.automatic_shrinking	= true,
};

INDIRECT_CALLABLE_SCOPE
const struct nft_set_ext *
nft_rhash_lookup(const struct net *net, const struct nft_set *set,
		 const u32 *key)
{
	struct nft_rhash *priv = nft_set_priv(set);
	const struct nft_rhash_elem *he;
	struct nft_rhash_cmp_arg arg = {
		.genmask = nft_genmask_cur(net),
		.set	 = set,
		.key	 = key,
		.tstamp  = get_jiffies_64(),
	};

	he = rhashtable_lookup(&priv->ht, &arg, nft_rhash_params);
	if (he != NULL)
		return &he->ext;

	return NULL;
}

static struct nft_elem_priv *
nft_rhash_get(const struct net *net, const struct nft_set *set,
	      const struct nft_set_elem *elem, unsigned int flags)
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_elem *he;
	struct nft_rhash_cmp_arg arg = {
		.genmask = nft_genmask_cur(net),
		.set	 = set,
		.key	 = elem->key.val.data,
		.tstamp  = get_jiffies_64(),
	};

	he = rhashtable_lookup(&priv->ht, &arg, nft_rhash_params);
	if (he != NULL)
		return &he->priv;

	return ERR_PTR(-ENOENT);
}

static const struct nft_set_ext *
nft_rhash_update(struct nft_set *set, const u32 *key,
		 const struct nft_expr *expr, struct nft_regs *regs)
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_elem *he, *prev;
	struct nft_elem_priv *elem_priv;
	struct nft_rhash_cmp_arg arg = {
		.genmask = NFT_GENMASK_ANY,
		.set	 = set,
		.key	 = key,
		.tstamp  = get_jiffies_64(),
	};

	he = rhashtable_lookup(&priv->ht, &arg, nft_rhash_params);
	if (he != NULL)
		goto out;

	elem_priv = nft_dynset_new(set, expr, regs);
	if (!elem_priv)
		goto err1;

	he = nft_elem_priv_cast(elem_priv);
	prev = rhashtable_lookup_get_insert_key(&priv->ht, &arg, &he->node,
						nft_rhash_params);
	if (IS_ERR(prev))
		goto err2;

	/* Another cpu may race to insert the element with the same key */
	if (prev) {
		nft_set_elem_destroy(set, &he->priv, true);
		atomic_dec(&set->nelems);
		he = prev;
	}

out:
	return &he->ext;

err2:
	nft_set_elem_destroy(set, &he->priv, true);
	atomic_dec(&set->nelems);
err1:
	return NULL;
}

static int nft_rhash_insert(const struct net *net, const struct nft_set *set,
			    const struct nft_set_elem *elem,
			    struct nft_elem_priv **elem_priv)
{
	struct nft_rhash_elem *he = nft_elem_priv_cast(elem->priv);
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_cmp_arg arg = {
		.genmask = nft_genmask_next(net),
		.set	 = set,
		.key	 = elem->key.val.data,
		.tstamp	 = nft_net_tstamp(net),
	};
	struct nft_rhash_elem *prev;

	prev = rhashtable_lookup_get_insert_key(&priv->ht, &arg, &he->node,
						nft_rhash_params);
	if (IS_ERR(prev))
		return PTR_ERR(prev);
	if (prev) {
		*elem_priv = &prev->priv;
		return -EEXIST;
	}
	return 0;
}

static void nft_rhash_activate(const struct net *net, const struct nft_set *set,
			       struct nft_elem_priv *elem_priv)
{
	struct nft_rhash_elem *he = nft_elem_priv_cast(elem_priv);

	nft_clear(net, &he->ext);
}

static void nft_rhash_flush(const struct net *net,
			    const struct nft_set *set,
			    struct nft_elem_priv *elem_priv)
{
	struct nft_rhash_elem *he = nft_elem_priv_cast(elem_priv);

	nft_set_elem_change_active(net, set, &he->ext);
}

static struct nft_elem_priv *
nft_rhash_deactivate(const struct net *net, const struct nft_set *set,
		     const struct nft_set_elem *elem)
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_elem *he;
	struct nft_rhash_cmp_arg arg = {
		.genmask = nft_genmask_next(net),
		.set	 = set,
		.key	 = elem->key.val.data,
		.tstamp	 = nft_net_tstamp(net),
	};

	rcu_read_lock();
	he = rhashtable_lookup(&priv->ht, &arg, nft_rhash_params);
	if (he)
		nft_set_elem_change_active(net, set, &he->ext);

	rcu_read_unlock();

	return &he->priv;
}

static void nft_rhash_remove(const struct net *net,
			     const struct nft_set *set,
			     struct nft_elem_priv *elem_priv)
{
	struct nft_rhash_elem *he = nft_elem_priv_cast(elem_priv);
	struct nft_rhash *priv = nft_set_priv(set);

	rhashtable_remove_fast(&priv->ht, &he->node, nft_rhash_params);
}

static bool nft_rhash_delete(const struct nft_set *set,
			     const u32 *key)
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_cmp_arg arg = {
		.genmask = NFT_GENMASK_ANY,
		.set = set,
		.key = key,
	};
	struct nft_rhash_elem *he;

	he = rhashtable_lookup(&priv->ht, &arg, nft_rhash_params);
	if (he == NULL)
		return false;

	nft_set_elem_dead(&he->ext);

	return true;
}

static void nft_rhash_walk(const struct nft_ctx *ctx, struct nft_set *set,
			   struct nft_set_iter *iter)
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_elem *he;
	struct rhashtable_iter hti;

	rhashtable_walk_enter(&priv->ht, &hti);
	rhashtable_walk_start(&hti);

	while ((he = rhashtable_walk_next(&hti))) {
		if (IS_ERR(he)) {
			if (PTR_ERR(he) != -EAGAIN) {
				iter->err = PTR_ERR(he);
				break;
			}

			continue;
		}

		if (iter->count < iter->skip)
			goto cont;

		iter->err = iter->fn(ctx, set, iter, &he->priv);
		if (iter->err < 0)
			break;

cont:
		iter->count++;
	}
	rhashtable_walk_stop(&hti);
	rhashtable_walk_exit(&hti);
}

static bool nft_rhash_expr_needs_gc_run(const struct nft_set *set,
					struct nft_set_ext *ext)
{
	struct nft_set_elem_expr *elem_expr = nft_set_ext_expr(ext);
	struct nft_expr *expr;
	u32 size;

	nft_setelem_expr_foreach(expr, elem_expr, size) {
		if (expr->ops->gc &&
		    expr->ops->gc(read_pnet(&set->net), expr) &&
		    set->flags & NFT_SET_EVAL)
			return true;
	}

	return false;
}

static void nft_rhash_gc(struct work_struct *work)
{
	struct nftables_pernet *nft_net;
	struct nft_set *set;
	struct nft_rhash_elem *he;
	struct nft_rhash *priv;
	struct rhashtable_iter hti;
	struct nft_trans_gc *gc;
	struct net *net;
	u32 gc_seq;

	priv = container_of(work, struct nft_rhash, gc_work.work);
	set  = nft_set_container_of(priv);
	net  = read_pnet(&set->net);
	nft_net = nft_pernet(net);
	gc_seq = READ_ONCE(nft_net->gc_seq);

	if (nft_set_gc_is_pending(set))
		goto done;

	gc = nft_trans_gc_alloc(set, gc_seq, GFP_KERNEL);
	if (!gc)
		goto done;

	/* Elements never collected use a zero gc worker sequence number. */
	if (unlikely(++priv->wq_gc_seq == 0))
		priv->wq_gc_seq++;

	rhashtable_walk_enter(&priv->ht, &hti);
	rhashtable_walk_start(&hti);

	while ((he = rhashtable_walk_next(&hti))) {
		if (IS_ERR(he)) {
			nft_trans_gc_destroy(gc);
			gc = NULL;
			goto try_later;
		}

		/* Ruleset has been updated, try later. */
		if (READ_ONCE(nft_net->gc_seq) != gc_seq) {
			nft_trans_gc_destroy(gc);
			gc = NULL;
			goto try_later;
		}

		/* rhashtable walk is unstable, already seen in this gc run?
		 * Then, skip this element. In case of (unlikely) sequence
		 * wraparound and stale element wq_gc_seq, next gc run will
		 * just find this expired element.
		 */
		if (he->wq_gc_seq == priv->wq_gc_seq)
			continue;

		if (nft_set_elem_is_dead(&he->ext))
			goto dead_elem;

		if (nft_set_ext_exists(&he->ext, NFT_SET_EXT_EXPRESSIONS) &&
		    nft_rhash_expr_needs_gc_run(set, &he->ext))
			goto needs_gc_run;

		if (!nft_set_elem_expired(&he->ext))
			continue;
needs_gc_run:
		nft_set_elem_dead(&he->ext);
dead_elem:
		gc = nft_trans_gc_queue_async(gc, gc_seq, GFP_ATOMIC);
		if (!gc)
			goto try_later;

		/* annotate gc sequence for this attempt. */
		he->wq_gc_seq = priv->wq_gc_seq;
		nft_trans_gc_elem_add(gc, he);
	}

	gc = nft_trans_gc_catchall_async(gc, gc_seq);

try_later:
	/* catchall list iteration requires rcu read side lock. */
	rhashtable_walk_stop(&hti);
	rhashtable_walk_exit(&hti);

	if (gc)
		nft_trans_gc_queue_async_done(gc);

done:
	queue_delayed_work(system_power_efficient_wq, &priv->gc_work,
			   nft_set_gc_interval(set));
}

static u64 nft_rhash_privsize(const struct nlattr * const nla[],
			      const struct nft_set_desc *desc)
{
	return sizeof(struct nft_rhash);
}

static void nft_rhash_gc_init(const struct nft_set *set)
{
	struct nft_rhash *priv = nft_set_priv(set);

	queue_delayed_work(system_power_efficient_wq, &priv->gc_work,
			   nft_set_gc_interval(set));
}

static int nft_rhash_init(const struct nft_set *set,
			  const struct nft_set_desc *desc,
			  const struct nlattr * const tb[])
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct rhashtable_params params = nft_rhash_params;
	int err;

	BUILD_BUG_ON(offsetof(struct nft_rhash_elem, priv) != 0);

	params.nelem_hint = desc->size ?: NFT_RHASH_ELEMENT_HINT;
	params.key_len	  = set->klen;

	err = rhashtable_init(&priv->ht, &params);
	if (err < 0)
		return err;

	INIT_DEFERRABLE_WORK(&priv->gc_work, nft_rhash_gc);
	if (set->flags & (NFT_SET_TIMEOUT | NFT_SET_EVAL))
		nft_rhash_gc_init(set);

	return 0;
}

struct nft_rhash_ctx {
	const struct nft_ctx	ctx;
	const struct nft_set	*set;
};

static void nft_rhash_elem_destroy(void *ptr, void *arg)
{
	struct nft_rhash_ctx *rhash_ctx = arg;
	struct nft_rhash_elem *he = ptr;

	nf_tables_set_elem_destroy(&rhash_ctx->ctx, rhash_ctx->set, &he->priv);
}

static void nft_rhash_destroy(const struct nft_ctx *ctx,
			      const struct nft_set *set)
{
	struct nft_rhash *priv = nft_set_priv(set);
	struct nft_rhash_ctx rhash_ctx = {
		.ctx	= *ctx,
		.set	= set,
	};

	cancel_delayed_work_sync(&priv->gc_work);
	rhashtable_free_and_destroy(&priv->ht, nft_rhash_elem_destroy,
				    (void *)&rhash_ctx);
}

/* Number of buckets is stored in u32, so cap our result to 1U<<31 */
#define NFT_MAX_BUCKETS (1U << 31)

static u32 nft_hash_buckets(u32 size)
{
	u64 val = div_u64((u64)size * 4, 3);

	if (val >= NFT_MAX_BUCKETS)
		return NFT_MAX_BUCKETS;

	return roundup_pow_of_two(val);
}

static bool nft_rhash_estimate(const struct nft_set_desc *desc, u32 features,
			       struct nft_set_estimate *est)
{
	est->size   = ~0;
	est->lookup = NFT_SET_CLASS_O_1;
	est->space  = NFT_SET_CLASS_O_N;

	return true;
}

struct nft_hash {
	u32				seed;
	u32				buckets;
	struct hlist_head		table[];
};

struct nft_hash_elem {
	struct nft_elem_priv		priv;
	struct hlist_node		node;
	struct nft_set_ext		ext;
};

INDIRECT_CALLABLE_SCOPE
const struct nft_set_ext *
nft_hash_lookup(const struct net *net, const struct nft_set *set,
		const u32 *key)
{
	struct nft_hash *priv = nft_set_priv(set);
	u8 genmask = nft_genmask_cur(net);
	const struct nft_hash_elem *he;
	u32 hash;

	hash = jhash(key, set->klen, priv->seed);
	hash = reciprocal_scale(hash, priv->buckets);
	hlist_for_each_entry_rcu(he, &priv->table[hash], node) {
		if (!memcmp(nft_set_ext_key(&he->ext), key, set->klen) &&
		    nft_set_elem_active(&he->ext, genmask))
			return &he->ext;
	}
	return NULL;
}

static struct nft_elem_priv *
nft_hash_get(const struct net *net, const struct nft_set *set,
	     const struct nft_set_elem *elem, unsigned int flags)
{
	struct nft_hash *priv = nft_set_priv(set);
	u8 genmask = nft_genmask_cur(net);
	struct nft_hash_elem *he;
	u32 hash;

	hash = jhash(elem->key.val.data, set->klen, priv->seed);
	hash = reciprocal_scale(hash, priv->buckets);
	hlist_for_each_entry_rcu(he, &priv->table[hash], node) {
		if (!memcmp(nft_set_ext_key(&he->ext), elem->key.val.data, set->klen) &&
		    nft_set_elem_active(&he->ext, genmask))
			return &he->priv;
	}
	return ERR_PTR(-ENOENT);
}

INDIRECT_CALLABLE_SCOPE
const struct nft_set_ext *
nft_hash_lookup_fast(const struct net *net, const struct nft_set *set,
		     const u32 *key)
{
	struct nft_hash *priv = nft_set_priv(set);
	u8 genmask = nft_genmask_cur(net);
	const struct nft_hash_elem *he;
	u32 hash, k1, k2;

	k1 = *key;
	hash = jhash_1word(k1, priv->seed);
	hash = reciprocal_scale(hash, priv->buckets);
	hlist_for_each_entry_rcu(he, &priv->table[hash], node) {
		k2 = *(u32 *)nft_set_ext_key(&he->ext)->data;
		if (k1 == k2 &&
		    nft_set_elem_active(&he->ext, genmask))
			return &he->ext;
	}
	return NULL;
}

static u32 nft_jhash(const struct nft_set *set, const struct nft_hash *priv,
		     const struct nft_set_ext *ext)
{
	const struct nft_data *key = nft_set_ext_key(ext);
	u32 hash, k1;

	if (set->klen == 4) {
		k1 = *(u32 *)key;
		hash = jhash_1word(k1, priv->seed);
	} else {
		hash = jhash(key, set->klen, priv->seed);
	}
	hash = reciprocal_scale(hash, priv->buckets);

	return hash;
}

static int nft_hash_insert(const struct net *net, const struct nft_set *set,
			   const struct nft_set_elem *elem,
			   struct nft_elem_priv **elem_priv)
{
	struct nft_hash_elem *this = nft_elem_priv_cast(elem->priv), *he;
	struct nft_hash *priv = nft_set_priv(set);
	u8 genmask = nft_genmask_next(net);
	u32 hash;

	hash = nft_jhash(set, priv, &this->ext);
	hlist_for_each_entry(he, &priv->table[hash], node) {
		if (!memcmp(nft_set_ext_key(&this->ext),
			    nft_set_ext_key(&he->ext), set->klen) &&
		    nft_set_elem_active(&he->ext, genmask)) {
			*elem_priv = &he->priv;
			return -EEXIST;
		}
	}
	hlist_add_head_rcu(&this->node, &priv->table[hash]);
	return 0;
}

static void nft_hash_activate(const struct net *net, const struct nft_set *set,
			      struct nft_elem_priv *elem_priv)
{
	struct nft_hash_elem *he = nft_elem_priv_cast(elem_priv);

	nft_clear(net, &he->ext);
}

static void nft_hash_flush(const struct net *net,
			   const struct nft_set *set,
			   struct nft_elem_priv *elem_priv)
{
	struct nft_hash_elem *he = nft_elem_priv_cast(elem_priv);

	nft_set_elem_change_active(net, set, &he->ext);
}

static struct nft_elem_priv *
nft_hash_deactivate(const struct net *net, const struct nft_set *set,
		    const struct nft_set_elem *elem)
{
	struct nft_hash_elem *this = nft_elem_priv_cast(elem->priv), *he;
	struct nft_hash *priv = nft_set_priv(set);
	u8 genmask = nft_genmask_next(net);
	u32 hash;

	hash = nft_jhash(set, priv, &this->ext);
	hlist_for_each_entry(he, &priv->table[hash], node) {
		if (!memcmp(nft_set_ext_key(&he->ext), &elem->key.val,
			    set->klen) &&
		    nft_set_elem_active(&he->ext, genmask)) {
			nft_set_elem_change_active(net, set, &he->ext);
			return &he->priv;
		}
	}
	return NULL;
}

static void nft_hash_remove(const struct net *net,
			    const struct nft_set *set,
			    struct nft_elem_priv *elem_priv)
{
	struct nft_hash_elem *he = nft_elem_priv_cast(elem_priv);

	hlist_del_rcu(&he->node);
}

static void nft_hash_walk(const struct nft_ctx *ctx, struct nft_set *set,
			  struct nft_set_iter *iter)
{
	struct nft_hash *priv = nft_set_priv(set);
	struct nft_hash_elem *he;
	int i;

	for (i = 0; i < priv->buckets; i++) {
		hlist_for_each_entry_rcu(he, &priv->table[i], node,
					 lockdep_is_held(&nft_pernet(ctx->net)->commit_mutex)) {
			if (iter->count < iter->skip)
				goto cont;

			iter->err = iter->fn(ctx, set, iter, &he->priv);
			if (iter->err < 0)
				return;
cont:
			iter->count++;
		}
	}
}

static u64 nft_hash_privsize(const struct nlattr * const nla[],
			     const struct nft_set_desc *desc)
{
	return sizeof(struct nft_hash) +
	       (u64)nft_hash_buckets(desc->size) * sizeof(struct hlist_head);
}

static int nft_hash_init(const struct nft_set *set,
			 const struct nft_set_desc *desc,
			 const struct nlattr * const tb[])
{
	struct nft_hash *priv = nft_set_priv(set);

	priv->buckets = nft_hash_buckets(desc->size);
	get_random_bytes(&priv->seed, sizeof(priv->seed));

	return 0;
}

static void nft_hash_destroy(const struct nft_ctx *ctx,
			     const struct nft_set *set)
{
	struct nft_hash *priv = nft_set_priv(set);
	struct nft_hash_elem *he;
	struct hlist_node *next;
	int i;

	for (i = 0; i < priv->buckets; i++) {
		hlist_for_each_entry_safe(he, next, &priv->table[i], node) {
			hlist_del_rcu(&he->node);
			nf_tables_set_elem_destroy(ctx, set, &he->priv);
		}
	}
}

static bool nft_hash_estimate(const struct nft_set_desc *desc, u32 features,
			      struct nft_set_estimate *est)
{
	if (!desc->size)
		return false;

	if (desc->klen == 4)
		return false;

	est->size   = sizeof(struct nft_hash) +
		      (u64)nft_hash_buckets(desc->size) * sizeof(struct hlist_head) +
		      (u64)desc->size * sizeof(struct nft_hash_elem);
	est->lookup = NFT_SET_CLASS_O_1;
	est->space  = NFT_SET_CLASS_O_N;

	return true;
}

static bool nft_hash_fast_estimate(const struct nft_set_desc *desc, u32 features,
				   struct nft_set_estimate *est)
{
	if (!desc->size)
		return false;

	if (desc->klen != 4)
		return false;

	est->size   = sizeof(struct nft_hash) +
		      (u64)nft_hash_buckets(desc->size) * sizeof(struct hlist_head) +
		      (u64)desc->size * sizeof(struct nft_hash_elem);
	est->lookup = NFT_SET_CLASS_O_1;
	est->space  = NFT_SET_CLASS_O_N;

	return true;
}

const struct nft_set_type nft_set_rhash_type = {
	.features	= NFT_SET_MAP | NFT_SET_OBJECT |
			  NFT_SET_TIMEOUT | NFT_SET_EVAL,
	.ops		= {
		.privsize       = nft_rhash_privsize,
		.elemsize	= offsetof(struct nft_rhash_elem, ext),
		.estimate	= nft_rhash_estimate,
		.init		= nft_rhash_init,
		.gc_init	= nft_rhash_gc_init,
		.destroy	= nft_rhash_destroy,
		.insert		= nft_rhash_insert,
		.activate	= nft_rhash_activate,
		.deactivate	= nft_rhash_deactivate,
		.flush		= nft_rhash_flush,
		.remove		= nft_rhash_remove,
		.lookup		= nft_rhash_lookup,
		.update		= nft_rhash_update,
		.delete		= nft_rhash_delete,
		.walk		= nft_rhash_walk,
		.get		= nft_rhash_get,
	},
};

const struct nft_set_type nft_set_hash_type = {
	.features	= NFT_SET_MAP | NFT_SET_OBJECT,
	.ops		= {
		.privsize       = nft_hash_privsize,
		.elemsize	= offsetof(struct nft_hash_elem, ext),
		.estimate	= nft_hash_estimate,
		.init		= nft_hash_init,
		.destroy	= nft_hash_destroy,
		.insert		= nft_hash_insert,
		.activate	= nft_hash_activate,
		.deactivate	= nft_hash_deactivate,
		.flush		= nft_hash_flush,
		.remove		= nft_hash_remove,
		.lookup		= nft_hash_lookup,
		.walk		= nft_hash_walk,
		.get		= nft_hash_get,
	},
};

const struct nft_set_type nft_set_hash_fast_type = {
	.features	= NFT_SET_MAP | NFT_SET_OBJECT,
	.ops		= {
		.privsize       = nft_hash_privsize,
		.elemsize	= offsetof(struct nft_hash_elem, ext),
		.estimate	= nft_hash_fast_estimate,
		.init		= nft_hash_init,
		.destroy	= nft_hash_destroy,
		.insert		= nft_hash_insert,
		.activate	= nft_hash_activate,
		.deactivate	= nft_hash_deactivate,
		.flush		= nft_hash_flush,
		.remove		= nft_hash_remove,
		.lookup		= nft_hash_lookup_fast,
		.walk		= nft_hash_walk,
		.get		= nft_hash_get,
	},
};
