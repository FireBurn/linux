// SPDX-License-Identifier: GPL-2.0-only
/*
 * Support code for Analog Devices Sigma-Delta ADCs
 *
 * Copyright 2012 Analog Devices Inc.
 *  Author: Lars-Peter Clausen <lars@metafoo.de>
 */

#include <linux/align.h>
#include <linux/bitmap.h>
#include <linux/bitops.h>
#include <linux/cleanup.h>
#include <linux/completion.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/export.h>
#include <linux/find.h>
#include <linux/gpio/consumer.h>
#include <linux/interrupt.h>
#include <linux/module.h>
#include <linux/property.h>
#include <linux/slab.h>
#include <linux/spi/offload/consumer.h>
#include <linux/spi/spi.h>
#include <linux/spinlock.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/unaligned.h>

#include <linux/iio/adc/ad_sigma_delta.h>
#include <linux/iio/buffer-dmaengine.h>
#include <linux/iio/buffer.h>
#include <linux/iio/iio.h>
#include <linux/iio/trigger_consumer.h>
#include <linux/iio/trigger.h>
#include <linux/iio/triggered_buffer.h>

#define AD_SD_COMM_CHAN_MASK	0x3

#define AD_SD_REG_COMM		0x00
#define AD_SD_REG_STATUS	0x00
#define AD_SD_REG_DATA		0x03

#define AD_SD_REG_STATUS_RDY	0x80

/**
 * ad_sd_set_comm() - Set communications register
 *
 * @sigma_delta: The sigma delta device
 * @comm: New value for the communications register
 */
void ad_sd_set_comm(struct ad_sigma_delta *sigma_delta, u8 comm)
{
	/* Some variants use the lower two bits of the communications register
	 * to select the channel */
	sigma_delta->comm = comm & AD_SD_COMM_CHAN_MASK;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_set_comm, "IIO_AD_SIGMA_DELTA");

/**
 * ad_sd_write_reg() - Write a register
 *
 * @sigma_delta: The sigma delta device
 * @reg: Address of the register
 * @size: Size of the register (0-3)
 * @val: Value to write to the register
 *
 * Returns 0 on success, an error code otherwise.
 **/
int ad_sd_write_reg(struct ad_sigma_delta *sigma_delta, unsigned int reg,
	unsigned int size, unsigned int val)
{
	u8 *data = sigma_delta->tx_buf;
	struct spi_transfer t = {
		.tx_buf		= data,
		.len		= size + 1,
		.cs_change	= sigma_delta->keep_cs_asserted,
	};
	struct spi_message m;
	int ret;

	data[0] = (reg << sigma_delta->info->addr_shift) | sigma_delta->comm;

	switch (size) {
	case 3:
		put_unaligned_be24(val, &data[1]);
		break;
	case 2:
		put_unaligned_be16(val, &data[1]);
		break;
	case 1:
		data[1] = val;
		break;
	case 0:
		break;
	default:
		return -EINVAL;
	}

	spi_message_init(&m);
	spi_message_add_tail(&t, &m);

	if (sigma_delta->bus_locked)
		ret = spi_sync_locked(sigma_delta->spi, &m);
	else
		ret = spi_sync(sigma_delta->spi, &m);

	return ret;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_write_reg, "IIO_AD_SIGMA_DELTA");

static void ad_sd_set_read_reg_addr(struct ad_sigma_delta *sigma_delta, u8 reg,
				    u8 *data)
{
	data[0] = reg << sigma_delta->info->addr_shift;
	data[0] |= sigma_delta->info->read_mask;
	data[0] |= sigma_delta->comm;
}

static int ad_sd_read_reg_raw(struct ad_sigma_delta *sigma_delta,
			      unsigned int reg, unsigned int size, u8 *val)
{
	u8 *data = sigma_delta->tx_buf;
	int ret;
	struct spi_transfer t[] = {
		{
			.tx_buf = data,
			.len = 1,
		}, {
			.rx_buf = val,
			.len = size,
			.cs_change = sigma_delta->keep_cs_asserted,
		},
	};
	struct spi_message m;

	spi_message_init(&m);

	if (sigma_delta->info->has_registers) {
		ad_sd_set_read_reg_addr(sigma_delta, reg, data);
		spi_message_add_tail(&t[0], &m);
	}
	spi_message_add_tail(&t[1], &m);

	if (sigma_delta->bus_locked)
		ret = spi_sync_locked(sigma_delta->spi, &m);
	else
		ret = spi_sync(sigma_delta->spi, &m);

	return ret;
}

/**
 * ad_sd_read_reg() - Read a register
 *
 * @sigma_delta: The sigma delta device
 * @reg: Address of the register
 * @size: Size of the register (1-4)
 * @val: Read value
 *
 * Returns 0 on success, an error code otherwise.
 **/
int ad_sd_read_reg(struct ad_sigma_delta *sigma_delta,
	unsigned int reg, unsigned int size, unsigned int *val)
{
	int ret;

	ret = ad_sd_read_reg_raw(sigma_delta, reg, size, sigma_delta->rx_buf);
	if (ret < 0)
		goto out;

	switch (size) {
	case 4:
		*val = get_unaligned_be32(sigma_delta->rx_buf);
		break;
	case 3:
		*val = get_unaligned_be24(sigma_delta->rx_buf);
		break;
	case 2:
		*val = get_unaligned_be16(sigma_delta->rx_buf);
		break;
	case 1:
		*val = sigma_delta->rx_buf[0];
		break;
	default:
		ret = -EINVAL;
		break;
	}

out:
	return ret;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_read_reg, "IIO_AD_SIGMA_DELTA");

/**
 * ad_sd_reset() - Reset the serial interface
 *
 * @sigma_delta: The sigma delta device
 *
 * Returns 0 on success, an error code otherwise.
 **/
int ad_sd_reset(struct ad_sigma_delta *sigma_delta)
{
	unsigned int reset_length = sigma_delta->info->num_resetclks;
	unsigned int size;
	u8 *buf;
	int ret;

	size = BITS_TO_BYTES(reset_length);
	buf = kcalloc(size, sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	memset(buf, 0xff, size);
	ret = spi_write(sigma_delta->spi, buf, size);
	kfree(buf);

	return ret;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_reset, "IIO_AD_SIGMA_DELTA");

static bool ad_sd_disable_irq(struct ad_sigma_delta *sigma_delta)
{
	guard(spinlock_irqsave)(&sigma_delta->irq_lock);

	/* It's already off, return false to indicate nothing was changed */
	if (sigma_delta->irq_dis)
		return false;

	sigma_delta->irq_dis = true;
	disable_irq_nosync(sigma_delta->irq_line);
	return true;
}

static void ad_sd_enable_irq(struct ad_sigma_delta *sigma_delta)
{
	guard(spinlock_irqsave)(&sigma_delta->irq_lock);

	sigma_delta->irq_dis = false;
	enable_irq(sigma_delta->irq_line);
}

#define AD_SD_CLEAR_DATA_BUFLEN 9

/* Called with `sigma_delta->bus_locked == true` only. */
static int ad_sigma_delta_clear_pending_event(struct ad_sigma_delta *sigma_delta)
{
	bool pending_event;
	unsigned int data_read_len = BITS_TO_BYTES(sigma_delta->info->num_resetclks);
	u8 *data;
	struct spi_transfer t[] = {
		{
			.len = 1,
		}, {
			.len = data_read_len,
		}
	};
	struct spi_message m;
	int ret;

	/*
	 * Read R̅D̅Y̅ pin (if possible) or status register to check if there is an
	 * old event.
	 */
	if (sigma_delta->rdy_gpiod) {
		pending_event = gpiod_get_value(sigma_delta->rdy_gpiod);
	} else {
		unsigned int status_reg;

		ret = ad_sd_read_reg(sigma_delta, AD_SD_REG_STATUS, 1, &status_reg);
		if (ret)
			return ret;

		pending_event = !(status_reg & AD_SD_REG_STATUS_RDY);
	}

	if (!pending_event)
		return 0;

	/*
	 * In general the size of the data register is unknown. It varies from
	 * device to device, might be one byte longer if CONTROL.DATA_STATUS is
	 * set and even varies on some devices depending on which input is
	 * selected. So send one byte to start reading the data register and
	 * then just clock for some bytes with DIN (aka MOSI) high to not
	 * confuse the register access state machine after the data register was
	 * completely read. Note however that the sequence length must be
	 * shorter than the reset procedure.
	 */

	data = kzalloc(data_read_len + 1, GFP_KERNEL);
	if (!data)
		return -ENOMEM;

	spi_message_init(&m);
	if (sigma_delta->info->has_registers) {
		unsigned int data_reg = sigma_delta->info->data_reg ?: AD_SD_REG_DATA;

		ad_sd_set_read_reg_addr(sigma_delta, data_reg, data);
		t[0].tx_buf = data;
		spi_message_add_tail(&t[0], &m);
	}

	/*
	 * The first transferred byte is part of the real data register,
	 * so this doesn't need to be 0xff. In the remaining
	 * `data_read_len - 1` bytes are less than $num_resetclks ones.
	 */
	t[1].tx_buf = data + 1;
	data[1] = 0x00;
	memset(data + 2, 0xff, data_read_len - 1);
	spi_message_add_tail(&t[1], &m);

	ret = spi_sync_locked(sigma_delta->spi, &m);

	kfree(data);

	return ret;
}

int ad_sd_calibrate(struct ad_sigma_delta *sigma_delta,
	unsigned int mode, unsigned int channel)
{
	int ret;
	unsigned long time_left;

	ret = ad_sigma_delta_set_channel(sigma_delta, channel);
	if (ret)
		return ret;

	spi_bus_lock(sigma_delta->spi->controller);
	sigma_delta->bus_locked = true;
	sigma_delta->keep_cs_asserted = true;
	reinit_completion(&sigma_delta->completion);

	ret = ad_sigma_delta_clear_pending_event(sigma_delta);
	if (ret)
		goto out;

	ret = ad_sigma_delta_set_mode(sigma_delta, mode);
	if (ret < 0)
		goto out;

	ad_sd_enable_irq(sigma_delta);
	time_left = wait_for_completion_timeout(&sigma_delta->completion, 2 * HZ);
	if (time_left == 0) {
		ad_sd_disable_irq(sigma_delta);
		ret = -EIO;
	} else {
		ret = 0;
	}
out:
	sigma_delta->keep_cs_asserted = false;
	ad_sigma_delta_set_mode(sigma_delta, AD_SD_MODE_IDLE);
	ad_sigma_delta_disable_one(sigma_delta, channel);
	sigma_delta->bus_locked = false;
	spi_bus_unlock(sigma_delta->spi->controller);

	return ret;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_calibrate, "IIO_AD_SIGMA_DELTA");

/**
 * ad_sd_calibrate_all() - Performs channel calibration
 * @sigma_delta: The sigma delta device
 * @cb: Array of channels and calibration type to perform
 * @n: Number of items in cb
 *
 * Returns 0 on success, an error code otherwise.
 **/
int ad_sd_calibrate_all(struct ad_sigma_delta *sigma_delta,
	const struct ad_sd_calib_data *cb, unsigned int n)
{
	unsigned int i;
	int ret;

	for (i = 0; i < n; i++) {
		ret = ad_sd_calibrate(sigma_delta, cb[i].mode, cb[i].channel);
		if (ret)
			return ret;
	}

	return 0;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_calibrate_all, "IIO_AD_SIGMA_DELTA");

/**
 * ad_sigma_delta_single_conversion() - Performs a single data conversion
 * @indio_dev: The IIO device
 * @chan: The conversion is done for this channel
 * @val: Pointer to the location where to store the read value
 *
 * Returns: 0 on success, an error value otherwise.
 */
int ad_sigma_delta_single_conversion(struct iio_dev *indio_dev,
	const struct iio_chan_spec *chan, int *val)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);
	unsigned int sample, raw_sample;
	unsigned int data_reg;
	int ret = 0;

	if (!iio_device_claim_direct(indio_dev))
		return -EBUSY;

	ret = ad_sigma_delta_set_channel(sigma_delta, chan->address);
	if (ret)
		goto out_release;

	spi_bus_lock(sigma_delta->spi->controller);
	sigma_delta->bus_locked = true;
	sigma_delta->keep_cs_asserted = true;
	reinit_completion(&sigma_delta->completion);

	ret = ad_sigma_delta_clear_pending_event(sigma_delta);
	if (ret)
		goto out_unlock;

	ad_sigma_delta_set_mode(sigma_delta, AD_SD_MODE_SINGLE);

	ad_sd_enable_irq(sigma_delta);
	ret = wait_for_completion_interruptible_timeout(
			&sigma_delta->completion, HZ);

	if (ret == 0)
		ret = -EIO;
	if (ret < 0)
		goto out;

	if (sigma_delta->info->data_reg != 0)
		data_reg = sigma_delta->info->data_reg;
	else
		data_reg = AD_SD_REG_DATA;

	ret = ad_sd_read_reg(sigma_delta, data_reg,
		BITS_TO_BYTES(chan->scan_type.realbits + chan->scan_type.shift),
		&raw_sample);

out:
	ad_sd_disable_irq(sigma_delta);

	ad_sigma_delta_set_mode(sigma_delta, AD_SD_MODE_IDLE);
	ad_sigma_delta_disable_one(sigma_delta, chan->address);

out_unlock:
	sigma_delta->keep_cs_asserted = false;
	sigma_delta->bus_locked = false;
	spi_bus_unlock(sigma_delta->spi->controller);
out_release:
	iio_device_release_direct(indio_dev);

	if (ret)
		return ret;

	sample = raw_sample >> chan->scan_type.shift;
	sample &= (1 << chan->scan_type.realbits) - 1;
	*val = sample;

	ret = ad_sigma_delta_postprocess_sample(sigma_delta, raw_sample);
	if (ret)
		return ret;

	return IIO_VAL_INT;
}
EXPORT_SYMBOL_NS_GPL(ad_sigma_delta_single_conversion, "IIO_AD_SIGMA_DELTA");

static int ad_sd_buffer_postenable(struct iio_dev *indio_dev)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);
	const struct iio_scan_type *scan_type = &indio_dev->channels[0].scan_type;
	struct spi_transfer *xfer = sigma_delta->sample_xfer;
	unsigned int i, slot, channel;
	u8 *samples_buf;
	int ret;

	if (sigma_delta->num_slots == 1) {
		channel = find_first_bit(indio_dev->active_scan_mask,
					 iio_get_masklength(indio_dev));
		ret = ad_sigma_delta_set_channel(sigma_delta,
						 indio_dev->channels[channel].address);
		if (ret)
			return ret;
		slot = 1;
	} else {
		/*
		 * At this point update_scan_mode already enabled the required channels.
		 * For sigma-delta sequencer drivers with multiple slots, an update_scan_mode
		 * implementation is mandatory.
		 */
		slot = 0;
		iio_for_each_active_channel(indio_dev, i) {
			sigma_delta->slots[slot] = indio_dev->channels[i].address;
			slot++;
		}
	}

	sigma_delta->active_slots = slot;
	sigma_delta->current_slot = 0;

	if (ad_sigma_delta_has_spi_offload(sigma_delta)) {
		xfer[1].offload_flags = SPI_OFFLOAD_XFER_RX_STREAM;
		xfer[1].bits_per_word = scan_type->realbits;
		xfer[1].len = spi_bpw_to_bytes(scan_type->realbits);
	} else {
		unsigned int samples_buf_size, scan_size;

		if (sigma_delta->active_slots > 1) {
			ret = ad_sigma_delta_append_status(sigma_delta, true);
			if (ret)
				return ret;
		}

		samples_buf_size =
			ALIGN(slot * BITS_TO_BYTES(scan_type->storagebits),
			      sizeof(s64));
		samples_buf_size += sizeof(s64);
		samples_buf = devm_krealloc(&sigma_delta->spi->dev,
					    sigma_delta->samples_buf,
					    samples_buf_size, GFP_KERNEL);
		if (!samples_buf)
			return -ENOMEM;

		sigma_delta->samples_buf = samples_buf;
		scan_size = BITS_TO_BYTES(scan_type->realbits + scan_type->shift);
		/* For 24-bit data, there is an extra byte of padding. */
		xfer[1].rx_buf = &sigma_delta->rx_buf[scan_size == 3 ? 1 : 0];
		xfer[1].len = scan_size + (sigma_delta->status_appended ? 1 : 0);
	}
	xfer[1].cs_change = 1;

	if (sigma_delta->info->has_registers) {
		xfer[0].tx_buf = &sigma_delta->sample_addr;
		xfer[0].len = 1;

		ad_sd_set_read_reg_addr(sigma_delta,
					sigma_delta->info->data_reg ?: AD_SD_REG_DATA,
					&sigma_delta->sample_addr);
		spi_message_init_with_transfers(&sigma_delta->sample_msg, xfer, 2);
	} else {
		spi_message_init_with_transfers(&sigma_delta->sample_msg,
						&xfer[1], 1);
	}

	sigma_delta->sample_msg.offload = sigma_delta->offload;

	ret = spi_optimize_message(sigma_delta->spi, &sigma_delta->sample_msg);
	if (ret)
		return ret;

	spi_bus_lock(sigma_delta->spi->controller);
	sigma_delta->bus_locked = true;
	sigma_delta->keep_cs_asserted = true;

	ret = ad_sigma_delta_clear_pending_event(sigma_delta);
	if (ret)
		goto err_unlock;

	ret = ad_sigma_delta_set_mode(sigma_delta, AD_SD_MODE_CONTINUOUS);
	if (ret)
		goto err_unlock;

	if (ad_sigma_delta_has_spi_offload(sigma_delta)) {
		struct spi_offload_trigger_config config = {
			.type = SPI_OFFLOAD_TRIGGER_DATA_READY,
		};

		ret = spi_offload_trigger_enable(sigma_delta->offload,
						 sigma_delta->offload_trigger,
						 &config);
		if (ret)
			goto err_unlock;
	} else {
		ad_sd_enable_irq(sigma_delta);
	}

	return 0;

err_unlock:
	spi_bus_unlock(sigma_delta->spi->controller);
	spi_unoptimize_message(&sigma_delta->sample_msg);

	return ret;
}

static int ad_sd_buffer_predisable(struct iio_dev *indio_dev)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);

	if (ad_sigma_delta_has_spi_offload(sigma_delta)) {
		spi_offload_trigger_disable(sigma_delta->offload,
					    sigma_delta->offload_trigger);
	} else {
		reinit_completion(&sigma_delta->completion);
		wait_for_completion_timeout(&sigma_delta->completion, HZ);

		ad_sd_disable_irq(sigma_delta);
	}

	sigma_delta->keep_cs_asserted = false;
	ad_sigma_delta_set_mode(sigma_delta, AD_SD_MODE_IDLE);

	if (sigma_delta->status_appended)
		ad_sigma_delta_append_status(sigma_delta, false);

	ad_sigma_delta_disable_all(sigma_delta);
	sigma_delta->bus_locked = false;
	spi_bus_unlock(sigma_delta->spi->controller);
	spi_unoptimize_message(&sigma_delta->sample_msg);

	return 0;
}

static irqreturn_t ad_sd_trigger_handler(int irq, void *p)
{
	struct iio_poll_func *pf = p;
	struct iio_dev *indio_dev = pf->indio_dev;
	const struct iio_scan_type *scan_type = &indio_dev->channels[0].scan_type;
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);
	u8 *data = sigma_delta->rx_buf;
	unsigned int sample_size;
	unsigned int sample_pos;
	unsigned int status_pos;
	unsigned int reg_size;
	int ret;

	reg_size = BITS_TO_BYTES(scan_type->realbits + scan_type->shift);
	/* For 24-bit data, there is an extra byte of padding. */
	status_pos = reg_size + (reg_size == 3 ? 1 : 0);

	ret = spi_sync_locked(sigma_delta->spi, &sigma_delta->sample_msg);
	if (ret)
		goto irq_handled;

	/*
	 * For devices sampling only one channel at
	 * once, there is no need for sample number tracking.
	 */
	if (sigma_delta->active_slots == 1) {
		iio_push_to_buffers_with_timestamp(indio_dev, data, pf->timestamp);
		goto irq_handled;
	}

	if (sigma_delta->status_appended) {
		u8 converted_channel;

		converted_channel = data[status_pos] & sigma_delta->info->status_ch_mask;
		if (converted_channel != sigma_delta->slots[sigma_delta->current_slot]) {
			/*
			 * Desync occurred during continuous sampling of multiple channels.
			 * Drop this incomplete sample and start from first channel again.
			 */

			sigma_delta->current_slot = 0;
			goto irq_handled;
		}
	}

	sample_size = BITS_TO_BYTES(scan_type->storagebits);
	sample_pos = sample_size * sigma_delta->current_slot;
	memcpy(&sigma_delta->samples_buf[sample_pos], data, sample_size);
	sigma_delta->current_slot++;

	if (sigma_delta->current_slot == sigma_delta->active_slots) {
		sigma_delta->current_slot = 0;
		iio_push_to_buffers_with_timestamp(indio_dev, sigma_delta->samples_buf,
						   pf->timestamp);
	}

irq_handled:
	iio_trigger_notify_done(indio_dev->trig);
	ad_sd_enable_irq(sigma_delta);

	return IRQ_HANDLED;
}

static bool ad_sd_validate_scan_mask(struct iio_dev *indio_dev, const unsigned long *mask)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);

	return bitmap_weight(mask, iio_get_masklength(indio_dev)) <= sigma_delta->num_slots;
}

static const struct iio_buffer_setup_ops ad_sd_buffer_setup_ops = {
	.postenable = &ad_sd_buffer_postenable,
	.predisable = &ad_sd_buffer_predisable,
	.validate_scan_mask = &ad_sd_validate_scan_mask,
};

static irqreturn_t ad_sd_data_rdy_trig_poll(int irq, void *private)
{
	struct ad_sigma_delta *sigma_delta = private;

	/*
	 * AD7124 and a few others use the same physical line for interrupt
	 * reporting (R̅D̅Y̅) and MISO.
	 * As MISO toggles when reading a register, this likely results in a
	 * pending interrupt. This has two consequences: a) The irq might
	 * trigger immediately after it's enabled even though the conversion
	 * isn't done yet; and b) checking the STATUS register's R̅D̅Y̅ flag is
	 * off-limits as reading that would trigger another irq event.
	 *
	 * So read the MOSI line as GPIO (if available) and only trigger the irq
	 * if the line is active. Without such a GPIO assume this is a valid
	 * interrupt.
	 *
	 * Also as disable_irq_nosync() is used to disable the irq, only act if
	 * the irq wasn't disabled before.
	 */
	if ((!sigma_delta->rdy_gpiod || gpiod_get_value(sigma_delta->rdy_gpiod)) &&
	    ad_sd_disable_irq(sigma_delta)) {
		complete(&sigma_delta->completion);
		if (sigma_delta->trig)
			iio_trigger_poll(sigma_delta->trig);

		return IRQ_HANDLED;
	}

	return IRQ_NONE;
}

/**
 * ad_sd_validate_trigger() - validate_trigger callback for ad_sigma_delta devices
 * @indio_dev: The IIO device
 * @trig: The new trigger
 *
 * Returns: 0 if the 'trig' matches the trigger registered by the ad_sigma_delta
 * device, -EINVAL otherwise.
 */
int ad_sd_validate_trigger(struct iio_dev *indio_dev, struct iio_trigger *trig)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);

	if (sigma_delta->trig != trig)
		return -EINVAL;

	return 0;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_validate_trigger, "IIO_AD_SIGMA_DELTA");

static int devm_ad_sd_probe_trigger(struct device *dev, struct iio_dev *indio_dev)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);
	unsigned long irq_flags = irq_get_trigger_type(sigma_delta->irq_line);
	int ret;

	init_completion(&sigma_delta->completion);

	sigma_delta->irq_dis = true;

	/* the IRQ core clears IRQ_DISABLE_UNLAZY flag when freeing an IRQ */
	irq_set_status_flags(sigma_delta->irq_line, IRQ_DISABLE_UNLAZY);

	/* Allow overwriting the flags from firmware */
	if (!irq_flags)
		irq_flags = sigma_delta->info->irq_flags;

	ret = devm_request_irq(dev, sigma_delta->irq_line,
			       ad_sd_data_rdy_trig_poll,
			       irq_flags | IRQF_NO_AUTOEN,
			       indio_dev->name,
			       sigma_delta);
	if (ret)
		return ret;

	if (ad_sigma_delta_has_spi_offload(sigma_delta)) {
		sigma_delta->offload_trigger =
			devm_spi_offload_trigger_get(dev, sigma_delta->offload,
						     SPI_OFFLOAD_TRIGGER_DATA_READY);
		if (IS_ERR(sigma_delta->offload_trigger))
			return dev_err_probe(dev, PTR_ERR(sigma_delta->offload_trigger),
					     "Failed to get SPI offload trigger\n");
	} else {
		if (dev != &sigma_delta->spi->dev)
			return dev_err_probe(dev, -EFAULT,
				"Trigger parent should be '%s', got '%s'\n",
				dev_name(dev), dev_name(&sigma_delta->spi->dev));

		sigma_delta->trig = devm_iio_trigger_alloc(dev, "%s-dev%d",
			indio_dev->name, iio_device_id(indio_dev));
		if (!sigma_delta->trig)
			return -ENOMEM;

		iio_trigger_set_drvdata(sigma_delta->trig, sigma_delta);

		ret = devm_iio_trigger_register(dev, sigma_delta->trig);
		if (ret)
			return ret;

		/* select default trigger */
		indio_dev->trig = iio_trigger_get(sigma_delta->trig);
	}

	return 0;
}

/**
 * devm_ad_sd_setup_buffer_and_trigger() - Device-managed buffer & trigger setup
 * @dev: Device object to which to bind the life-time of the resources attached
 * @indio_dev: The IIO device
 */
int devm_ad_sd_setup_buffer_and_trigger(struct device *dev, struct iio_dev *indio_dev)
{
	struct ad_sigma_delta *sigma_delta = iio_device_get_drvdata(indio_dev);
	int ret;

	sigma_delta->slots = devm_kcalloc(dev, sigma_delta->num_slots,
					  sizeof(*sigma_delta->slots), GFP_KERNEL);
	if (!sigma_delta->slots)
		return -ENOMEM;

	if (ad_sigma_delta_has_spi_offload(sigma_delta)) {
		struct dma_chan *rx_dma;

		rx_dma = devm_spi_offload_rx_stream_request_dma_chan(dev,
			sigma_delta->offload);
		if (IS_ERR(rx_dma))
			return dev_err_probe(dev, PTR_ERR(rx_dma),
					     "Failed to get RX DMA channel\n");

		ret = devm_iio_dmaengine_buffer_setup_with_handle(dev, indio_dev,
			rx_dma, IIO_BUFFER_DIRECTION_IN);
		if (ret)
			return dev_err_probe(dev, ret, "Cannot setup DMA buffer\n");

		indio_dev->setup_ops = &ad_sd_buffer_setup_ops;
	} else {
		ret = devm_iio_triggered_buffer_setup(dev, indio_dev,
						      &iio_pollfunc_store_time,
						      &ad_sd_trigger_handler,
						      &ad_sd_buffer_setup_ops);
		if (ret)
			return ret;
	}

	return devm_ad_sd_probe_trigger(dev, indio_dev);
}
EXPORT_SYMBOL_NS_GPL(devm_ad_sd_setup_buffer_and_trigger, "IIO_AD_SIGMA_DELTA");

/**
 * ad_sd_init() - Initializes a ad_sigma_delta struct
 * @sigma_delta: The ad_sigma_delta device
 * @indio_dev: The IIO device which the Sigma Delta device is used for
 * @spi: The SPI device for the ad_sigma_delta device
 * @info: Device specific callbacks and options
 *
 * This function needs to be called before any other operations are performed on
 * the ad_sigma_delta struct.
 */
int ad_sd_init(struct ad_sigma_delta *sigma_delta, struct iio_dev *indio_dev,
	struct spi_device *spi, const struct ad_sigma_delta_info *info)
{
	sigma_delta->spi = spi;
	sigma_delta->info = info;

	/* If the field is unset in ad_sigma_delta_info, assume there can only be 1 slot. */
	if (!info->num_slots)
		sigma_delta->num_slots = 1;
	else
		sigma_delta->num_slots = info->num_slots;

	if (sigma_delta->num_slots > 1) {
		if (!indio_dev->info->update_scan_mode) {
			dev_err(&spi->dev, "iio_dev lacks update_scan_mode().\n");
			return -EINVAL;
		}

		if (!info->disable_all) {
			dev_err(&spi->dev, "ad_sigma_delta_info lacks disable_all().\n");
			return -EINVAL;
		}
	}

	spin_lock_init(&sigma_delta->irq_lock);

	if (info->has_named_irqs) {
		sigma_delta->irq_line = fwnode_irq_get_byname(dev_fwnode(&spi->dev),
							      "rdy");
		if (sigma_delta->irq_line < 0)
			return dev_err_probe(&spi->dev, sigma_delta->irq_line,
					     "Interrupt 'rdy' is required\n");
	} else {
		sigma_delta->irq_line = spi->irq;
	}

	sigma_delta->rdy_gpiod = devm_gpiod_get_optional(&spi->dev, "rdy", GPIOD_IN);
	if (IS_ERR(sigma_delta->rdy_gpiod))
		return dev_err_probe(&spi->dev, PTR_ERR(sigma_delta->rdy_gpiod),
				     "Failed to find rdy gpio\n");

	if (sigma_delta->rdy_gpiod && !sigma_delta->irq_line) {
		sigma_delta->irq_line = gpiod_to_irq(sigma_delta->rdy_gpiod);
		if (sigma_delta->irq_line < 0)
			return sigma_delta->irq_line;
	}

	if (info->supports_spi_offload) {
		struct spi_offload_config offload_config = {
			.capability_flags = SPI_OFFLOAD_CAP_TRIGGER |
					    SPI_OFFLOAD_CAP_RX_STREAM_DMA,
		};
		int ret;

		sigma_delta->offload = devm_spi_offload_get(&spi->dev, spi,
							    &offload_config);
		ret = PTR_ERR_OR_ZERO(sigma_delta->offload);
		if (ret && ret != -ENODEV)
			return dev_err_probe(&spi->dev, ret, "Failed to get SPI offload\n");
	}

	iio_device_set_drvdata(indio_dev, sigma_delta);

	return 0;
}
EXPORT_SYMBOL_NS_GPL(ad_sd_init, "IIO_AD_SIGMA_DELTA");

MODULE_AUTHOR("Lars-Peter Clausen <lars@metafoo.de>");
MODULE_DESCRIPTION("Analog Devices Sigma-Delta ADCs");
MODULE_LICENSE("GPL v2");
MODULE_IMPORT_NS("IIO_DMAENGINE_BUFFER");
