/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * RSA public-key primitive (RSAEP), library interface.
 *
 * Copyright (c) 2026 Mike Lothian <mike@fireburn.co.uk>
 */
#ifndef _CRYPTO_RSA_H
#define _CRYPTO_RSA_H

#include <linux/types.h>

int rsa_pubkey_encrypt(const u8 *n, size_t n_len, const u8 *e, size_t e_len,
		       const u8 *in, size_t in_len, u8 *out, size_t out_len);

#endif /* _CRYPTO_RSA_H */
