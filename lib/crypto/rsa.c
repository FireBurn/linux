// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * RSA public-key primitive (RSAEP) [RFC8017 sec 5.1.1].
 *
 * A minimal synchronous library entry point for the RSA public-key operation,
 * built on the MPI big-integer library (the same arithmetic the crypto_akcipher
 * "rsa" transform uses), but without the akcipher tfm allocation, DER key
 * encoding, scatterlists or asynchronous completion machinery. It performs only
 * the bare RSAEP primitive; callers apply any encoding/padding (RSAES-OAEP,
 * RSAES-PKCS1-v1_5, ...) themselves.
 *
 * Copyright (c) 2026 Mike Lothian <mike@fireburn.co.uk>
 */

#include <crypto/rsa.h>
#include <linux/errno.h>
#include <linux/export.h>
#include <linux/mpi.h>
#include <linux/module.h>
#include <linux/string.h>

/**
 * rsa_pubkey_encrypt() - RSA public-key operation, out = (in ^ e) mod n
 * @n: modulus, unsigned big-endian
 * @n_len: length of @n, in bytes
 * @e: public exponent, unsigned big-endian
 * @e_len: length of @e, in bytes
 * @in: input, unsigned big-endian; already padded by the caller
 * @in_len: length of @in, in bytes
 * @out: output buffer; receives the result big-endian, left zero-padded to
 *	 exactly @out_len bytes
 * @out_len: size of @out; pass the modulus length (e.g. 128 for RSA-1024)
 *
 * Computes the bare RSAEP primitive c = m^e mod n, where m is @in interpreted
 * as a big-endian integer. This is the raw public-key operation [RFC8017 sec
 * 5.1.1]; the caller is responsible for any message encoding/padding.
 *
 * @in interpreted as an integer must be numerically less than @n, as RSA
 * requires: a larger value would be silently reduced mod n by the modular
 * exponentiation and yield a ciphertext the peer cannot decrypt, so it is
 * rejected. On any error path @out is zeroed, so it never retains data from a
 * partial computation.
 *
 * Return: 0 on success; -EINVAL on a malformed key or out-of-range input;
 * -EOVERFLOW if the result does not fit in @out_len; -ENOMEM on allocation
 * failure.
 */
int rsa_pubkey_encrypt(const u8 *n, size_t n_len, const u8 *e, size_t e_len,
		       const u8 *in, size_t in_len, u8 *out, size_t out_len)
{
	MPI mn, me, mbase, mres;
	unsigned int nbytes;
	int ret = -ENOMEM;

	if (!n_len || !e_len || !in_len || !out_len)
		return -EINVAL;

	mn = mpi_read_raw_data(n, n_len);
	me = mpi_read_raw_data(e, e_len);
	mbase = mpi_read_raw_data(in, in_len);
	mres = mpi_alloc(0);
	if (!mn || !me || !mbase || !mres)
		goto out;

	/*
	 * RSA requires 0 <= m < n; reject m >= n rather than let mpi_powm()
	 * silently reduce it and produce an undecryptable ciphertext.
	 */
	if (mpi_cmp(mbase, mn) >= 0) {
		ret = -EINVAL;
		goto out;
	}

	ret = mpi_powm(mres, mbase, me, mn);
	if (ret)
		goto out;

	/*
	 * mpi_read_buffer() writes the minimal big-endian form left-aligned
	 * (and returns -EOVERFLOW without writing if @out_len is too small);
	 * right-align it into the fixed-width output and zero-pad the front.
	 */
	ret = mpi_read_buffer(mres, out, out_len, &nbytes, NULL);
	if (ret)
		goto out;
	if (nbytes < out_len) {
		memmove(out + (out_len - nbytes), out, nbytes);
		memset(out, 0, out_len - nbytes);
	}
out:
	if (ret)
		memzero_explicit(out, out_len);
	mpi_free(mn);
	mpi_free(me);
	mpi_free(mbase);
	mpi_free(mres);
	return ret;
}
EXPORT_SYMBOL_GPL(rsa_pubkey_encrypt);

MODULE_DESCRIPTION("RSA public-key primitive (RSAEP)");
MODULE_LICENSE("GPL");
