// SPDX-License-Identifier: GPL-2.0
/* Copyright(c) 2013 - 2018 Intel Corporation. */

#include <linux/bitfield.h>
#include <linux/net/intel/libie/rx.h>
#include <linux/prefetch.h>

#include "iavf.h"
#include "iavf_trace.h"
#include "iavf_prototype.h"
#include "iavf_ptp.h"

/**
 * iavf_is_descriptor_done - tests DD bit in Rx descriptor
 * @qw1: quad word 1 from descriptor to get Descriptor Done field from
 * @flex: is the descriptor flex or legacy
 *
 * This function tests the descriptor done bit in specified descriptor. Because
 * there are two types of descriptors (legacy and flex) the parameter rx_ring
 * is used to distinguish.
 *
 * Return: true or false based on the state of DD bit in Rx descriptor.
 */
static bool iavf_is_descriptor_done(u64 qw1, bool flex)
{
	if (flex)
		return FIELD_GET(IAVF_RXD_FLEX_DD_M, qw1);
	else
		return FIELD_GET(IAVF_RXD_LEGACY_DD_M, qw1);
}

static __le64 build_ctob(u32 td_cmd, u32 td_offset, unsigned int size,
			 u32 td_tag)
{
	return cpu_to_le64(IAVF_TX_DESC_DTYPE_DATA |
			   ((u64)td_cmd  << IAVF_TXD_QW1_CMD_SHIFT) |
			   ((u64)td_offset << IAVF_TXD_QW1_OFFSET_SHIFT) |
			   ((u64)size  << IAVF_TXD_QW1_TX_BUF_SZ_SHIFT) |
			   ((u64)td_tag  << IAVF_TXD_QW1_L2TAG1_SHIFT));
}

#define IAVF_TXD_CMD (IAVF_TX_DESC_CMD_EOP | IAVF_TX_DESC_CMD_RS)

/**
 * iavf_unmap_and_free_tx_resource - Release a Tx buffer
 * @ring:      the ring that owns the buffer
 * @tx_buffer: the buffer to free
 **/
static void iavf_unmap_and_free_tx_resource(struct iavf_ring *ring,
					    struct iavf_tx_buffer *tx_buffer)
{
	if (tx_buffer->skb) {
		if (tx_buffer->tx_flags & IAVF_TX_FLAGS_FD_SB)
			kfree(tx_buffer->raw_buf);
		else
			dev_kfree_skb_any(tx_buffer->skb);
		if (dma_unmap_len(tx_buffer, len))
			dma_unmap_single(ring->dev,
					 dma_unmap_addr(tx_buffer, dma),
					 dma_unmap_len(tx_buffer, len),
					 DMA_TO_DEVICE);
	} else if (dma_unmap_len(tx_buffer, len)) {
		dma_unmap_page(ring->dev,
			       dma_unmap_addr(tx_buffer, dma),
			       dma_unmap_len(tx_buffer, len),
			       DMA_TO_DEVICE);
	}

	tx_buffer->next_to_watch = NULL;
	tx_buffer->skb = NULL;
	dma_unmap_len_set(tx_buffer, len, 0);
	/* tx_buffer must be completely set up in the transmit path */
}

/**
 * iavf_clean_tx_ring - Free any empty Tx buffers
 * @tx_ring: ring to be cleaned
 **/
static void iavf_clean_tx_ring(struct iavf_ring *tx_ring)
{
	unsigned long bi_size;
	u16 i;

	/* ring already cleared, nothing to do */
	if (!tx_ring->tx_bi)
		return;

	/* Free all the Tx ring sk_buffs */
	for (i = 0; i < tx_ring->count; i++)
		iavf_unmap_and_free_tx_resource(tx_ring, &tx_ring->tx_bi[i]);

	bi_size = sizeof(struct iavf_tx_buffer) * tx_ring->count;
	memset(tx_ring->tx_bi, 0, bi_size);

	/* Zero out the descriptor ring */
	memset(tx_ring->desc, 0, tx_ring->size);

	tx_ring->next_to_use = 0;
	tx_ring->next_to_clean = 0;

	if (!tx_ring->netdev)
		return;

	/* cleanup Tx queue statistics */
	netdev_tx_reset_queue(txring_txq(tx_ring));
}

/**
 * iavf_free_tx_resources - Free Tx resources per queue
 * @tx_ring: Tx descriptor ring for a specific queue
 *
 * Free all transmit software resources
 **/
void iavf_free_tx_resources(struct iavf_ring *tx_ring)
{
	iavf_clean_tx_ring(tx_ring);
	kfree(tx_ring->tx_bi);
	tx_ring->tx_bi = NULL;

	if (tx_ring->desc) {
		dma_free_coherent(tx_ring->dev, tx_ring->size,
				  tx_ring->desc, tx_ring->dma);
		tx_ring->desc = NULL;
	}
}

/**
 * iavf_get_tx_pending - how many Tx descriptors not processed
 * @ring: the ring of descriptors
 * @in_sw: is tx_pending being checked in SW or HW
 *
 * Since there is no access to the ring head register
 * in XL710, we need to use our local copies
 **/
static u32 iavf_get_tx_pending(struct iavf_ring *ring, bool in_sw)
{
	u32 head, tail;

	/* underlying hardware might not allow access and/or always return
	 * 0 for the head/tail registers so just use the cached values
	 */
	head = ring->next_to_clean;
	tail = ring->next_to_use;

	if (head != tail)
		return (head < tail) ?
			tail - head : (tail + ring->count - head);

	return 0;
}

/**
 * iavf_force_wb - Issue SW Interrupt so HW does a wb
 * @vsi: the VSI we care about
 * @q_vector: the vector on which to force writeback
 **/
static void iavf_force_wb(struct iavf_vsi *vsi, struct iavf_q_vector *q_vector)
{
	u32 val = IAVF_VFINT_DYN_CTLN1_INTENA_MASK |
		  IAVF_VFINT_DYN_CTLN1_ITR_INDX_MASK | /* set noitr */
		  IAVF_VFINT_DYN_CTLN1_SWINT_TRIG_MASK |
		  IAVF_VFINT_DYN_CTLN1_SW_ITR_INDX_ENA_MASK
		  /* allow 00 to be written to the index */;

	wr32(&vsi->back->hw,
	     IAVF_VFINT_DYN_CTLN1(q_vector->reg_idx),
	     val);
}

/**
 * iavf_detect_recover_hung - Function to detect and recover hung_queues
 * @vsi:  pointer to vsi struct with tx queues
 *
 * VSI has netdev and netdev has TX queues. This function is to check each of
 * those TX queues if they are hung, trigger recovery by issuing SW interrupt.
 **/
void iavf_detect_recover_hung(struct iavf_vsi *vsi)
{
	struct iavf_ring *tx_ring = NULL;
	struct net_device *netdev;
	unsigned int i;
	int packets;

	if (!vsi)
		return;

	if (test_bit(__IAVF_VSI_DOWN, vsi->state))
		return;

	netdev = vsi->netdev;
	if (!netdev)
		return;

	if (!netif_carrier_ok(netdev))
		return;

	for (i = 0; i < vsi->back->num_active_queues; i++) {
		tx_ring = &vsi->back->tx_rings[i];
		if (tx_ring && tx_ring->desc) {
			/* If packet counter has not changed the queue is
			 * likely stalled, so force an interrupt for this
			 * queue.
			 *
			 * prev_pkt_ctr would be negative if there was no
			 * pending work.
			 */
			packets = tx_ring->stats.packets & INT_MAX;
			if (tx_ring->prev_pkt_ctr == packets) {
				iavf_force_wb(vsi, tx_ring->q_vector);
				continue;
			}

			/* Memory barrier between read of packet count and call
			 * to iavf_get_tx_pending()
			 */
			smp_rmb();
			tx_ring->prev_pkt_ctr =
			  iavf_get_tx_pending(tx_ring, true) ? packets : -1;
		}
	}
}

#define WB_STRIDE 4

/**
 * iavf_clean_tx_irq - Reclaim resources after transmit completes
 * @vsi: the VSI we care about
 * @tx_ring: Tx ring to clean
 * @napi_budget: Used to determine if we are in netpoll
 *
 * Returns true if there's any budget left (e.g. the clean is finished)
 **/
static bool iavf_clean_tx_irq(struct iavf_vsi *vsi,
			      struct iavf_ring *tx_ring, int napi_budget)
{
	int i = tx_ring->next_to_clean;
	struct iavf_tx_buffer *tx_buf;
	struct iavf_tx_desc *tx_desc;
	unsigned int total_bytes = 0, total_packets = 0;
	unsigned int budget = IAVF_DEFAULT_IRQ_WORK;

	tx_buf = &tx_ring->tx_bi[i];
	tx_desc = IAVF_TX_DESC(tx_ring, i);
	i -= tx_ring->count;

	do {
		struct iavf_tx_desc *eop_desc = tx_buf->next_to_watch;

		/* if next_to_watch is not set then there is no work pending */
		if (!eop_desc)
			break;

		/* prevent any other reads prior to eop_desc */
		smp_rmb();

		iavf_trace(clean_tx_irq, tx_ring, tx_desc, tx_buf);
		/* if the descriptor isn't done, no work yet to do */
		if (!(eop_desc->cmd_type_offset_bsz &
		      cpu_to_le64(IAVF_TX_DESC_DTYPE_DESC_DONE)))
			break;

		/* clear next_to_watch to prevent false hangs */
		tx_buf->next_to_watch = NULL;

		/* update the statistics for this packet */
		total_bytes += tx_buf->bytecount;
		total_packets += tx_buf->gso_segs;

		/* free the skb */
		napi_consume_skb(tx_buf->skb, napi_budget);

		/* unmap skb header data */
		dma_unmap_single(tx_ring->dev,
				 dma_unmap_addr(tx_buf, dma),
				 dma_unmap_len(tx_buf, len),
				 DMA_TO_DEVICE);

		/* clear tx_buffer data */
		tx_buf->skb = NULL;
		dma_unmap_len_set(tx_buf, len, 0);

		/* unmap remaining buffers */
		while (tx_desc != eop_desc) {
			iavf_trace(clean_tx_irq_unmap,
				   tx_ring, tx_desc, tx_buf);

			tx_buf++;
			tx_desc++;
			i++;
			if (unlikely(!i)) {
				i -= tx_ring->count;
				tx_buf = tx_ring->tx_bi;
				tx_desc = IAVF_TX_DESC(tx_ring, 0);
			}

			/* unmap any remaining paged data */
			if (dma_unmap_len(tx_buf, len)) {
				dma_unmap_page(tx_ring->dev,
					       dma_unmap_addr(tx_buf, dma),
					       dma_unmap_len(tx_buf, len),
					       DMA_TO_DEVICE);
				dma_unmap_len_set(tx_buf, len, 0);
			}
		}

		/* move us one more past the eop_desc for start of next pkt */
		tx_buf++;
		tx_desc++;
		i++;
		if (unlikely(!i)) {
			i -= tx_ring->count;
			tx_buf = tx_ring->tx_bi;
			tx_desc = IAVF_TX_DESC(tx_ring, 0);
		}

		prefetch(tx_desc);

		/* update budget accounting */
		budget--;
	} while (likely(budget));

	i += tx_ring->count;
	tx_ring->next_to_clean = i;
	u64_stats_update_begin(&tx_ring->syncp);
	tx_ring->stats.bytes += total_bytes;
	tx_ring->stats.packets += total_packets;
	u64_stats_update_end(&tx_ring->syncp);
	tx_ring->q_vector->tx.total_bytes += total_bytes;
	tx_ring->q_vector->tx.total_packets += total_packets;

	if (tx_ring->flags & IAVF_TXR_FLAGS_WB_ON_ITR) {
		/* check to see if there are < 4 descriptors
		 * waiting to be written back, then kick the hardware to force
		 * them to be written back in case we stay in NAPI.
		 * In this mode on X722 we do not enable Interrupt.
		 */
		unsigned int j = iavf_get_tx_pending(tx_ring, false);

		if (budget &&
		    ((j / WB_STRIDE) == 0) && (j > 0) &&
		    !test_bit(__IAVF_VSI_DOWN, vsi->state) &&
		    (IAVF_DESC_UNUSED(tx_ring) != tx_ring->count))
			tx_ring->flags |= IAVF_TXR_FLAGS_ARM_WB;
	}

	/* notify netdev of completed buffers */
	netdev_tx_completed_queue(txring_txq(tx_ring),
				  total_packets, total_bytes);

#define TX_WAKE_THRESHOLD ((s16)(DESC_NEEDED * 2))
	if (unlikely(total_packets && netif_carrier_ok(tx_ring->netdev) &&
		     (IAVF_DESC_UNUSED(tx_ring) >= TX_WAKE_THRESHOLD))) {
		/* Make sure that anybody stopping the queue after this
		 * sees the new next_to_clean.
		 */
		smp_mb();
		if (__netif_subqueue_stopped(tx_ring->netdev,
					     tx_ring->queue_index) &&
		   !test_bit(__IAVF_VSI_DOWN, vsi->state)) {
			netif_wake_subqueue(tx_ring->netdev,
					    tx_ring->queue_index);
			++tx_ring->tx_stats.restart_queue;
		}
	}

	return !!budget;
}

/**
 * iavf_enable_wb_on_itr - Arm hardware to do a wb, interrupts are not enabled
 * @vsi: the VSI we care about
 * @q_vector: the vector on which to enable writeback
 *
 **/
static void iavf_enable_wb_on_itr(struct iavf_vsi *vsi,
				  struct iavf_q_vector *q_vector)
{
	u16 flags = q_vector->tx.ring[0].flags;
	u32 val;

	if (!(flags & IAVF_TXR_FLAGS_WB_ON_ITR))
		return;

	if (q_vector->arm_wb_state)
		return;

	val = IAVF_VFINT_DYN_CTLN1_WB_ON_ITR_MASK |
	      IAVF_VFINT_DYN_CTLN1_ITR_INDX_MASK; /* set noitr */

	wr32(&vsi->back->hw,
	     IAVF_VFINT_DYN_CTLN1(q_vector->reg_idx), val);
	q_vector->arm_wb_state = true;
}

static bool iavf_container_is_rx(struct iavf_q_vector *q_vector,
				 struct iavf_ring_container *rc)
{
	return &q_vector->rx == rc;
}

#define IAVF_AIM_MULTIPLIER_100G	2560
#define IAVF_AIM_MULTIPLIER_50G		1280
#define IAVF_AIM_MULTIPLIER_40G		1024
#define IAVF_AIM_MULTIPLIER_20G		512
#define IAVF_AIM_MULTIPLIER_10G		256
#define IAVF_AIM_MULTIPLIER_1G		32

static unsigned int iavf_mbps_itr_multiplier(u32 speed_mbps)
{
	switch (speed_mbps) {
	case SPEED_100000:
		return IAVF_AIM_MULTIPLIER_100G;
	case SPEED_50000:
		return IAVF_AIM_MULTIPLIER_50G;
	case SPEED_40000:
		return IAVF_AIM_MULTIPLIER_40G;
	case SPEED_25000:
	case SPEED_20000:
		return IAVF_AIM_MULTIPLIER_20G;
	case SPEED_10000:
	default:
		return IAVF_AIM_MULTIPLIER_10G;
	case SPEED_1000:
	case SPEED_100:
		return IAVF_AIM_MULTIPLIER_1G;
	}
}

static unsigned int
iavf_virtchnl_itr_multiplier(enum virtchnl_link_speed speed_virtchnl)
{
	switch (speed_virtchnl) {
	case VIRTCHNL_LINK_SPEED_40GB:
		return IAVF_AIM_MULTIPLIER_40G;
	case VIRTCHNL_LINK_SPEED_25GB:
	case VIRTCHNL_LINK_SPEED_20GB:
		return IAVF_AIM_MULTIPLIER_20G;
	case VIRTCHNL_LINK_SPEED_10GB:
	default:
		return IAVF_AIM_MULTIPLIER_10G;
	case VIRTCHNL_LINK_SPEED_1GB:
	case VIRTCHNL_LINK_SPEED_100MB:
		return IAVF_AIM_MULTIPLIER_1G;
	}
}

static unsigned int iavf_itr_divisor(struct iavf_adapter *adapter)
{
	if (ADV_LINK_SUPPORT(adapter))
		return IAVF_ITR_ADAPTIVE_MIN_INC *
			iavf_mbps_itr_multiplier(adapter->link_speed_mbps);
	else
		return IAVF_ITR_ADAPTIVE_MIN_INC *
			iavf_virtchnl_itr_multiplier(adapter->link_speed);
}

/**
 * iavf_update_itr - update the dynamic ITR value based on statistics
 * @q_vector: structure containing interrupt and ring information
 * @rc: structure containing ring performance data
 *
 * Stores a new ITR value based on packets and byte
 * counts during the last interrupt.  The advantage of per interrupt
 * computation is faster updates and more accurate ITR for the current
 * traffic pattern.  Constants in this function were computed
 * based on theoretical maximum wire speed and thresholds were set based
 * on testing data as well as attempting to minimize response time
 * while increasing bulk throughput.
 **/
static void iavf_update_itr(struct iavf_q_vector *q_vector,
			    struct iavf_ring_container *rc)
{
	unsigned int avg_wire_size, packets, bytes, itr;
	unsigned long next_update = jiffies;

	/* If we don't have any rings just leave ourselves set for maximum
	 * possible latency so we take ourselves out of the equation.
	 */
	if (!rc->ring || !ITR_IS_DYNAMIC(rc->ring->itr_setting))
		return;

	/* For Rx we want to push the delay up and default to low latency.
	 * for Tx we want to pull the delay down and default to high latency.
	 */
	itr = iavf_container_is_rx(q_vector, rc) ?
	      IAVF_ITR_ADAPTIVE_MIN_USECS | IAVF_ITR_ADAPTIVE_LATENCY :
	      IAVF_ITR_ADAPTIVE_MAX_USECS | IAVF_ITR_ADAPTIVE_LATENCY;

	/* If we didn't update within up to 1 - 2 jiffies we can assume
	 * that either packets are coming in so slow there hasn't been
	 * any work, or that there is so much work that NAPI is dealing
	 * with interrupt moderation and we don't need to do anything.
	 */
	if (time_after(next_update, rc->next_update))
		goto clear_counts;

	/* If itr_countdown is set it means we programmed an ITR within
	 * the last 4 interrupt cycles. This has a side effect of us
	 * potentially firing an early interrupt. In order to work around
	 * this we need to throw out any data received for a few
	 * interrupts following the update.
	 */
	if (q_vector->itr_countdown) {
		itr = rc->target_itr;
		goto clear_counts;
	}

	packets = rc->total_packets;
	bytes = rc->total_bytes;

	if (iavf_container_is_rx(q_vector, rc)) {
		/* If Rx there are 1 to 4 packets and bytes are less than
		 * 9000 assume insufficient data to use bulk rate limiting
		 * approach unless Tx is already in bulk rate limiting. We
		 * are likely latency driven.
		 */
		if (packets && packets < 4 && bytes < 9000 &&
		    (q_vector->tx.target_itr & IAVF_ITR_ADAPTIVE_LATENCY)) {
			itr = IAVF_ITR_ADAPTIVE_LATENCY;
			goto adjust_by_size;
		}
	} else if (packets < 4) {
		/* If we have Tx and Rx ITR maxed and Tx ITR is running in
		 * bulk mode and we are receiving 4 or fewer packets just
		 * reset the ITR_ADAPTIVE_LATENCY bit for latency mode so
		 * that the Rx can relax.
		 */
		if (rc->target_itr == IAVF_ITR_ADAPTIVE_MAX_USECS &&
		    (q_vector->rx.target_itr & IAVF_ITR_MASK) ==
		     IAVF_ITR_ADAPTIVE_MAX_USECS)
			goto clear_counts;
	} else if (packets > 32) {
		/* If we have processed over 32 packets in a single interrupt
		 * for Tx assume we need to switch over to "bulk" mode.
		 */
		rc->target_itr &= ~IAVF_ITR_ADAPTIVE_LATENCY;
	}

	/* We have no packets to actually measure against. This means
	 * either one of the other queues on this vector is active or
	 * we are a Tx queue doing TSO with too high of an interrupt rate.
	 *
	 * Between 4 and 56 we can assume that our current interrupt delay
	 * is only slightly too low. As such we should increase it by a small
	 * fixed amount.
	 */
	if (packets < 56) {
		itr = rc->target_itr + IAVF_ITR_ADAPTIVE_MIN_INC;
		if ((itr & IAVF_ITR_MASK) > IAVF_ITR_ADAPTIVE_MAX_USECS) {
			itr &= IAVF_ITR_ADAPTIVE_LATENCY;
			itr += IAVF_ITR_ADAPTIVE_MAX_USECS;
		}
		goto clear_counts;
	}

	if (packets <= 256) {
		itr = min(q_vector->tx.current_itr, q_vector->rx.current_itr);
		itr &= IAVF_ITR_MASK;

		/* Between 56 and 112 is our "goldilocks" zone where we are
		 * working out "just right". Just report that our current
		 * ITR is good for us.
		 */
		if (packets <= 112)
			goto clear_counts;

		/* If packet count is 128 or greater we are likely looking
		 * at a slight overrun of the delay we want. Try halving
		 * our delay to see if that will cut the number of packets
		 * in half per interrupt.
		 */
		itr /= 2;
		itr &= IAVF_ITR_MASK;
		if (itr < IAVF_ITR_ADAPTIVE_MIN_USECS)
			itr = IAVF_ITR_ADAPTIVE_MIN_USECS;

		goto clear_counts;
	}

	/* The paths below assume we are dealing with a bulk ITR since
	 * number of packets is greater than 256. We are just going to have
	 * to compute a value and try to bring the count under control,
	 * though for smaller packet sizes there isn't much we can do as
	 * NAPI polling will likely be kicking in sooner rather than later.
	 */
	itr = IAVF_ITR_ADAPTIVE_BULK;

adjust_by_size:
	/* If packet counts are 256 or greater we can assume we have a gross
	 * overestimation of what the rate should be. Instead of trying to fine
	 * tune it just use the formula below to try and dial in an exact value
	 * give the current packet size of the frame.
	 */
	avg_wire_size = bytes / packets;

	/* The following is a crude approximation of:
	 *  wmem_default / (size + overhead) = desired_pkts_per_int
	 *  rate / bits_per_byte / (size + ethernet overhead) = pkt_rate
	 *  (desired_pkt_rate / pkt_rate) * usecs_per_sec = ITR value
	 *
	 * Assuming wmem_default is 212992 and overhead is 640 bytes per
	 * packet, (256 skb, 64 headroom, 320 shared info), we can reduce the
	 * formula down to
	 *
	 *  (170 * (size + 24)) / (size + 640) = ITR
	 *
	 * We first do some math on the packet size and then finally bitshift
	 * by 8 after rounding up. We also have to account for PCIe link speed
	 * difference as ITR scales based on this.
	 */
	if (avg_wire_size <= 60) {
		/* Start at 250k ints/sec */
		avg_wire_size = 4096;
	} else if (avg_wire_size <= 380) {
		/* 250K ints/sec to 60K ints/sec */
		avg_wire_size *= 40;
		avg_wire_size += 1696;
	} else if (avg_wire_size <= 1084) {
		/* 60K ints/sec to 36K ints/sec */
		avg_wire_size *= 15;
		avg_wire_size += 11452;
	} else if (avg_wire_size <= 1980) {
		/* 36K ints/sec to 30K ints/sec */
		avg_wire_size *= 5;
		avg_wire_size += 22420;
	} else {
		/* plateau at a limit of 30K ints/sec */
		avg_wire_size = 32256;
	}

	/* If we are in low latency mode halve our delay which doubles the
	 * rate to somewhere between 100K to 16K ints/sec
	 */
	if (itr & IAVF_ITR_ADAPTIVE_LATENCY)
		avg_wire_size /= 2;

	/* Resultant value is 256 times larger than it needs to be. This
	 * gives us room to adjust the value as needed to either increase
	 * or decrease the value based on link speeds of 10G, 2.5G, 1G, etc.
	 *
	 * Use addition as we have already recorded the new latency flag
	 * for the ITR value.
	 */
	itr += DIV_ROUND_UP(avg_wire_size,
			    iavf_itr_divisor(q_vector->adapter)) *
		IAVF_ITR_ADAPTIVE_MIN_INC;

	if ((itr & IAVF_ITR_MASK) > IAVF_ITR_ADAPTIVE_MAX_USECS) {
		itr &= IAVF_ITR_ADAPTIVE_LATENCY;
		itr += IAVF_ITR_ADAPTIVE_MAX_USECS;
	}

clear_counts:
	/* write back value */
	rc->target_itr = itr;

	/* next update should occur within next jiffy */
	rc->next_update = next_update + 1;

	rc->total_bytes = 0;
	rc->total_packets = 0;
}

/**
 * iavf_setup_tx_descriptors - Allocate the Tx descriptors
 * @tx_ring: the tx ring to set up
 *
 * Return 0 on success, negative on error
 **/
int iavf_setup_tx_descriptors(struct iavf_ring *tx_ring)
{
	struct device *dev = tx_ring->dev;
	int bi_size;

	if (!dev)
		return -ENOMEM;

	/* warn if we are about to overwrite the pointer */
	WARN_ON(tx_ring->tx_bi);
	bi_size = sizeof(struct iavf_tx_buffer) * tx_ring->count;
	tx_ring->tx_bi = kzalloc(bi_size, GFP_KERNEL);
	if (!tx_ring->tx_bi)
		goto err;

	/* round up to nearest 4K */
	tx_ring->size = tx_ring->count * sizeof(struct iavf_tx_desc);
	tx_ring->size = ALIGN(tx_ring->size, 4096);
	tx_ring->desc = dma_alloc_coherent(dev, tx_ring->size,
					   &tx_ring->dma, GFP_KERNEL);
	if (!tx_ring->desc) {
		dev_info(dev, "Unable to allocate memory for the Tx descriptor ring, size=%d\n",
			 tx_ring->size);
		goto err;
	}

	tx_ring->next_to_use = 0;
	tx_ring->next_to_clean = 0;
	tx_ring->prev_pkt_ctr = -1;
	return 0;

err:
	kfree(tx_ring->tx_bi);
	tx_ring->tx_bi = NULL;
	return -ENOMEM;
}

/**
 * iavf_clean_rx_ring - Free Rx buffers
 * @rx_ring: ring to be cleaned
 **/
static void iavf_clean_rx_ring(struct iavf_ring *rx_ring)
{
	/* ring already cleared, nothing to do */
	if (!rx_ring->rx_fqes)
		return;

	if (rx_ring->skb) {
		dev_kfree_skb(rx_ring->skb);
		rx_ring->skb = NULL;
	}

	/* Free all the Rx ring buffers */
	for (u32 i = rx_ring->next_to_clean; i != rx_ring->next_to_use; ) {
		const struct libeth_fqe *rx_fqes = &rx_ring->rx_fqes[i];

		libeth_rx_recycle_slow(rx_fqes->netmem);

		if (unlikely(++i == rx_ring->count))
			i = 0;
	}

	rx_ring->next_to_clean = 0;
	rx_ring->next_to_use = 0;
}

/**
 * iavf_free_rx_resources - Free Rx resources
 * @rx_ring: ring to clean the resources from
 *
 * Free all receive software resources
 **/
void iavf_free_rx_resources(struct iavf_ring *rx_ring)
{
	struct libeth_fq fq = {
		.fqes	= rx_ring->rx_fqes,
		.pp	= rx_ring->pp,
	};

	iavf_clean_rx_ring(rx_ring);

	if (rx_ring->desc) {
		dma_free_coherent(rx_ring->pp->p.dev, rx_ring->size,
				  rx_ring->desc, rx_ring->dma);
		rx_ring->desc = NULL;
	}

	libeth_rx_fq_destroy(&fq);
	rx_ring->rx_fqes = NULL;
	rx_ring->pp = NULL;
}

/**
 * iavf_setup_rx_descriptors - Allocate Rx descriptors
 * @rx_ring: Rx descriptor ring (for a specific queue) to setup
 *
 * Returns 0 on success, negative on failure
 **/
int iavf_setup_rx_descriptors(struct iavf_ring *rx_ring)
{
	struct libeth_fq fq = {
		.count		= rx_ring->count,
		.buf_len	= LIBIE_MAX_RX_BUF_LEN,
		.nid		= NUMA_NO_NODE,
	};
	int ret;

	ret = libeth_rx_fq_create(&fq, &rx_ring->q_vector->napi);
	if (ret)
		return ret;

	rx_ring->pp = fq.pp;
	rx_ring->rx_fqes = fq.fqes;
	rx_ring->truesize = fq.truesize;
	rx_ring->rx_buf_len = fq.buf_len;

	u64_stats_init(&rx_ring->syncp);

	/* Round up to nearest 4K */
	rx_ring->size = rx_ring->count * sizeof(struct iavf_rx_desc);
	rx_ring->size = ALIGN(rx_ring->size, 4096);
	rx_ring->desc = dma_alloc_coherent(fq.pp->p.dev, rx_ring->size,
					   &rx_ring->dma, GFP_KERNEL);

	if (!rx_ring->desc) {
		dev_info(fq.pp->p.dev, "Unable to allocate memory for the Rx descriptor ring, size=%d\n",
			 rx_ring->size);
		goto err;
	}

	rx_ring->next_to_clean = 0;
	rx_ring->next_to_use = 0;

	return 0;

err:
	libeth_rx_fq_destroy(&fq);
	rx_ring->rx_fqes = NULL;
	rx_ring->pp = NULL;

	return -ENOMEM;
}

/**
 * iavf_release_rx_desc - Store the new tail and head values
 * @rx_ring: ring to bump
 * @val: new head index
 **/
static void iavf_release_rx_desc(struct iavf_ring *rx_ring, u32 val)
{
	rx_ring->next_to_use = val;

	/* Force memory writes to complete before letting h/w
	 * know there are new descriptors to fetch.  (Only
	 * applicable for weak-ordered memory model archs,
	 * such as IA-64).
	 */
	wmb();
	writel(val, rx_ring->tail);
}

/**
 * iavf_receive_skb - Send a completed packet up the stack
 * @rx_ring:  rx ring in play
 * @skb: packet to send up
 * @vlan_tag: vlan tag for packet
 **/
static void iavf_receive_skb(struct iavf_ring *rx_ring,
			     struct sk_buff *skb, u16 vlan_tag)
{
	struct iavf_q_vector *q_vector = rx_ring->q_vector;

	if ((rx_ring->netdev->features & NETIF_F_HW_VLAN_CTAG_RX) &&
	    (vlan_tag & VLAN_VID_MASK))
		__vlan_hwaccel_put_tag(skb, htons(ETH_P_8021Q), vlan_tag);
	else if ((rx_ring->netdev->features & NETIF_F_HW_VLAN_STAG_RX) &&
		 vlan_tag & VLAN_VID_MASK)
		__vlan_hwaccel_put_tag(skb, htons(ETH_P_8021AD), vlan_tag);

	napi_gro_receive(&q_vector->napi, skb);
}

/**
 * iavf_alloc_rx_buffers - Replace used receive buffers
 * @rx_ring: ring to place buffers on
 * @cleaned_count: number of buffers to replace
 *
 * Returns false if all allocations were successful, true if any fail
 **/
bool iavf_alloc_rx_buffers(struct iavf_ring *rx_ring, u16 cleaned_count)
{
	const struct libeth_fq_fp fq = {
		.pp		= rx_ring->pp,
		.fqes		= rx_ring->rx_fqes,
		.truesize	= rx_ring->truesize,
		.count		= rx_ring->count,
	};
	u16 ntu = rx_ring->next_to_use;
	struct iavf_rx_desc *rx_desc;

	/* do nothing if no valid netdev defined */
	if (!rx_ring->netdev || !cleaned_count)
		return false;

	rx_desc = IAVF_RX_DESC(rx_ring, ntu);

	do {
		dma_addr_t addr;

		addr = libeth_rx_alloc(&fq, ntu);
		if (addr == DMA_MAPPING_ERROR)
			goto no_buffers;

		/* Refresh the desc even if buffer_addrs didn't change
		 * because each write-back erases this info.
		 */
		rx_desc->qw0 = cpu_to_le64(addr);

		rx_desc++;
		ntu++;
		if (unlikely(ntu == rx_ring->count)) {
			rx_desc = IAVF_RX_DESC(rx_ring, 0);
			ntu = 0;
		}

		/* clear the status bits for the next_to_use descriptor */
		rx_desc->qw1 = 0;

		cleaned_count--;
	} while (cleaned_count);

	if (rx_ring->next_to_use != ntu)
		iavf_release_rx_desc(rx_ring, ntu);

	return false;

no_buffers:
	if (rx_ring->next_to_use != ntu)
		iavf_release_rx_desc(rx_ring, ntu);

	rx_ring->rx_stats.alloc_page_failed++;

	/* make sure to come back via polling to try again after
	 * allocation failure
	 */
	return true;
}

/**
 * iavf_rx_csum - Indicate in skb if hw indicated a good checksum
 * @vsi: the VSI we care about
 * @skb: skb currently being received and modified
 * @decoded_pt: decoded ptype information
 * @csum_bits: decoded Rx descriptor information
 **/
static void iavf_rx_csum(const struct iavf_vsi *vsi, struct sk_buff *skb,
			 struct libeth_rx_pt decoded_pt,
			 struct libeth_rx_csum csum_bits)
{
	bool ipv4, ipv6;

	skb->ip_summed = CHECKSUM_NONE;

	/* did the hardware decode the packet and checksum? */
	if (unlikely(!csum_bits.l3l4p))
		return;

	ipv4 = libeth_rx_pt_get_ip_ver(decoded_pt) == LIBETH_RX_PT_OUTER_IPV4;
	ipv6 = libeth_rx_pt_get_ip_ver(decoded_pt) == LIBETH_RX_PT_OUTER_IPV6;

	if (unlikely(ipv4 && (csum_bits.ipe || csum_bits.eipe)))
		goto checksum_fail;

	/* likely incorrect csum if alternate IP extension headers found */
	if (unlikely(ipv6 && csum_bits.ipv6exadd))
		return;

	/* there was some L4 error, count error and punt packet to the stack */
	if (unlikely(csum_bits.l4e))
		goto checksum_fail;

	/* handle packets that were not able to be checksummed due
	 * to arrival speed, in this case the stack can compute
	 * the csum.
	 */
	if (unlikely(csum_bits.pprs))
		return;

	skb->ip_summed = CHECKSUM_UNNECESSARY;
	return;

checksum_fail:
	vsi->back->hw_csum_rx_error++;
}

/**
 * iavf_legacy_rx_csum - Indicate in skb if hw indicated a good checksum
 * @vsi: the VSI we care about
 * @qw1: quad word 1
 * @decoded_pt: decoded packet type
 *
 * This function only operates on the VIRTCHNL_RXDID_1_32B_BASE legacy 32byte
 * descriptor writeback format.
 *
 * Return: decoded checksum bits.
 **/
static struct libeth_rx_csum
iavf_legacy_rx_csum(const struct iavf_vsi *vsi, u64 qw1,
		    const struct libeth_rx_pt decoded_pt)
{
	struct libeth_rx_csum csum_bits = {};

	if (!libeth_rx_pt_has_checksum(vsi->netdev, decoded_pt))
		return csum_bits;

	csum_bits.ipe = FIELD_GET(IAVF_RXD_LEGACY_IPE_M, qw1);
	csum_bits.eipe = FIELD_GET(IAVF_RXD_LEGACY_EIPE_M, qw1);
	csum_bits.l4e = FIELD_GET(IAVF_RXD_LEGACY_L4E_M, qw1);
	csum_bits.pprs = FIELD_GET(IAVF_RXD_LEGACY_PPRS_M, qw1);
	csum_bits.l3l4p = FIELD_GET(IAVF_RXD_LEGACY_L3L4P_M, qw1);
	csum_bits.ipv6exadd = FIELD_GET(IAVF_RXD_LEGACY_IPV6EXADD_M, qw1);

	return csum_bits;
}

/**
 * iavf_flex_rx_csum - Indicate in skb if hw indicated a good checksum
 * @vsi: the VSI we care about
 * @qw1: quad word 1
 * @decoded_pt: decoded packet type
 *
 * This function only operates on the VIRTCHNL_RXDID_2_FLEX_SQ_NIC flexible
 * descriptor writeback format.
 *
 * Return: decoded checksum bits.
 **/
static struct libeth_rx_csum
iavf_flex_rx_csum(const struct iavf_vsi *vsi, u64 qw1,
		  const struct libeth_rx_pt decoded_pt)
{
	struct libeth_rx_csum csum_bits = {};

	if (!libeth_rx_pt_has_checksum(vsi->netdev, decoded_pt))
		return csum_bits;

	csum_bits.ipe = FIELD_GET(IAVF_RXD_FLEX_XSUM_IPE_M, qw1);
	csum_bits.eipe = FIELD_GET(IAVF_RXD_FLEX_XSUM_EIPE_M, qw1);
	csum_bits.l4e = FIELD_GET(IAVF_RXD_FLEX_XSUM_L4E_M, qw1);
	csum_bits.eudpe = FIELD_GET(IAVF_RXD_FLEX_XSUM_EUDPE_M, qw1);
	csum_bits.l3l4p = FIELD_GET(IAVF_RXD_FLEX_L3L4P_M, qw1);
	csum_bits.ipv6exadd = FIELD_GET(IAVF_RXD_FLEX_IPV6EXADD_M, qw1);
	csum_bits.nat = FIELD_GET(IAVF_RXD_FLEX_NAT_M, qw1);

	return csum_bits;
}

/**
 * iavf_legacy_rx_hash - set the hash value in the skb
 * @ring: descriptor ring
 * @qw0: quad word 0
 * @qw1: quad word 1
 * @skb: skb currently being received and modified
 * @decoded_pt: decoded packet type
 *
 * This function only operates on the VIRTCHNL_RXDID_1_32B_BASE legacy 32byte
 * descriptor writeback format.
 **/
static void iavf_legacy_rx_hash(const struct iavf_ring *ring, __le64 qw0,
				__le64 qw1, struct sk_buff *skb,
				const struct libeth_rx_pt decoded_pt)
{
	const __le64 rss_mask = cpu_to_le64(IAVF_RXD_LEGACY_FLTSTAT_M);
	u32 hash;

	if (!libeth_rx_pt_has_hash(ring->netdev, decoded_pt))
		return;

	if ((qw1 & rss_mask) == rss_mask) {
		hash = le64_get_bits(qw0, IAVF_RXD_LEGACY_RSS_M);
		libeth_rx_pt_set_hash(skb, hash, decoded_pt);
	}
}

/**
 * iavf_flex_rx_hash - set the hash value in the skb
 * @ring: descriptor ring
 * @qw1: quad word 1
 * @skb: skb currently being received and modified
 * @decoded_pt: decoded packet type
 *
 * This function only operates on the VIRTCHNL_RXDID_2_FLEX_SQ_NIC flexible
 * descriptor writeback format.
 **/
static void iavf_flex_rx_hash(const struct iavf_ring *ring, __le64 qw1,
			      struct sk_buff *skb,
			      const struct libeth_rx_pt decoded_pt)
{
	bool rss_valid;
	u32 hash;

	if (!libeth_rx_pt_has_hash(ring->netdev, decoded_pt))
		return;

	rss_valid = le64_get_bits(qw1, IAVF_RXD_FLEX_RSS_VALID_M);
	if (rss_valid) {
		hash = le64_get_bits(qw1, IAVF_RXD_FLEX_RSS_HASH_M);
		libeth_rx_pt_set_hash(skb, hash, decoded_pt);
	}
}

/**
 * iavf_flex_rx_tstamp - Capture Rx timestamp from the descriptor
 * @rx_ring: descriptor ring
 * @qw2: quad word 2 of descriptor
 * @qw3: quad word 3 of descriptor
 * @skb: skb currently being received
 *
 * Read the Rx timestamp value from the descriptor and pass it to the stack.
 *
 * This function only operates on the VIRTCHNL_RXDID_2_FLEX_SQ_NIC flexible
 * descriptor writeback format.
 */
static void iavf_flex_rx_tstamp(const struct iavf_ring *rx_ring, __le64 qw2,
				__le64 qw3, struct sk_buff *skb)
{
	u32 tstamp;
	u64 ns;

	/* Skip processing if timestamps aren't enabled */
	if (!(rx_ring->flags & IAVF_TXRX_FLAGS_HW_TSTAMP))
		return;

	/* Check if this Rx descriptor has a valid timestamp */
	if (!le64_get_bits(qw2, IAVF_PTP_40B_TSTAMP_VALID))
		return;

	/* the ts_low field only contains the valid bit and sub-nanosecond
	 * precision, so we don't need to extract it.
	 */
	tstamp = le64_get_bits(qw3, IAVF_RXD_FLEX_QW3_TSTAMP_HIGH_M);

	ns = iavf_ptp_extend_32b_timestamp(rx_ring->ptp->cached_phc_time,
					   tstamp);

	*skb_hwtstamps(skb) = (struct skb_shared_hwtstamps) {
		.hwtstamp = ns_to_ktime(ns),
	};
}

/**
 * iavf_process_skb_fields - Populate skb header fields from Rx descriptor
 * @rx_ring: rx descriptor ring packet is being transacted on
 * @rx_desc: pointer to the EOP Rx descriptor
 * @skb: pointer to current skb being populated
 * @ptype: the packet type decoded by hardware
 * @flex: is the descriptor flex or legacy
 *
 * This function checks the ring, descriptor, and packet information in
 * order to populate the hash, checksum, VLAN, protocol, and
 * other fields within the skb.
 **/
static void iavf_process_skb_fields(const struct iavf_ring *rx_ring,
				    const struct iavf_rx_desc *rx_desc,
				    struct sk_buff *skb, u32 ptype,
				    bool flex)
{
	struct libeth_rx_csum csum_bits;
	struct libeth_rx_pt decoded_pt;
	__le64 qw0 = rx_desc->qw0;
	__le64 qw1 = rx_desc->qw1;
	__le64 qw2 = rx_desc->qw2;
	__le64 qw3 = rx_desc->qw3;

	decoded_pt = libie_rx_pt_parse(ptype);

	if (flex) {
		iavf_flex_rx_hash(rx_ring, qw1, skb, decoded_pt);
		iavf_flex_rx_tstamp(rx_ring, qw2, qw3, skb);
		csum_bits = iavf_flex_rx_csum(rx_ring->vsi, le64_to_cpu(qw1),
					      decoded_pt);
	} else {
		iavf_legacy_rx_hash(rx_ring, qw0, qw1, skb, decoded_pt);
		csum_bits = iavf_legacy_rx_csum(rx_ring->vsi, le64_to_cpu(qw1),
						decoded_pt);
	}
	iavf_rx_csum(rx_ring->vsi, skb, decoded_pt, csum_bits);

	skb_record_rx_queue(skb, rx_ring->queue_index);

	/* modifies the skb - consumes the enet header */
	skb->protocol = eth_type_trans(skb, rx_ring->netdev);
}

/**
 * iavf_cleanup_headers - Correct empty headers
 * @rx_ring: rx descriptor ring packet is being transacted on
 * @skb: pointer to current skb being fixed
 *
 * Also address the case where we are pulling data in on pages only
 * and as such no data is present in the skb header.
 *
 * In addition if skb is not at least 60 bytes we need to pad it so that
 * it is large enough to qualify as a valid Ethernet frame.
 *
 * Returns true if an error was encountered and skb was freed.
 **/
static bool iavf_cleanup_headers(struct iavf_ring *rx_ring, struct sk_buff *skb)
{
	/* if eth_skb_pad returns an error the skb was freed */
	if (eth_skb_pad(skb))
		return true;

	return false;
}

/**
 * iavf_add_rx_frag - Add contents of Rx buffer to sk_buff
 * @skb: sk_buff to place the data into
 * @rx_buffer: buffer containing page to add
 * @size: packet length from rx_desc
 *
 * This function will add the data contained in rx_buffer->page to the skb.
 * It will just attach the page as a frag to the skb.
 *
 * The function will then update the page offset.
 **/
static void iavf_add_rx_frag(struct sk_buff *skb,
			     const struct libeth_fqe *rx_buffer,
			     unsigned int size)
{
	u32 hr = netmem_get_pp(rx_buffer->netmem)->p.offset;

	skb_add_rx_frag_netmem(skb, skb_shinfo(skb)->nr_frags,
			       rx_buffer->netmem, rx_buffer->offset + hr,
			       size, rx_buffer->truesize);
}

/**
 * iavf_build_skb - Build skb around an existing buffer
 * @rx_buffer: Rx buffer to pull data from
 * @size: size of buffer to add to skb
 *
 * This function builds an skb around an existing Rx buffer, taking care
 * to set up the skb correctly and avoid any memcpy overhead.
 */
static struct sk_buff *iavf_build_skb(const struct libeth_fqe *rx_buffer,
				      unsigned int size)
{
	struct page *buf_page = __netmem_to_page(rx_buffer->netmem);
	u32 hr = pp_page_to_nmdesc(buf_page)->pp->p.offset;
	struct sk_buff *skb;
	void *va;

	/* prefetch first cache line of first page */
	va = page_address(buf_page) + rx_buffer->offset;
	net_prefetch(va + hr);

	/* build an skb around the page buffer */
	skb = napi_build_skb(va, rx_buffer->truesize);
	if (unlikely(!skb))
		return NULL;

	skb_mark_for_recycle(skb);

	/* update pointers within the skb to store the data */
	skb_reserve(skb, hr);
	__skb_put(skb, size);

	return skb;
}

/**
 * iavf_is_non_eop - process handling of non-EOP buffers
 * @rx_ring: Rx ring being processed
 * @fields: Rx descriptor extracted fields
 *
 * This function updates next to clean.  If the buffer is an EOP buffer
 * this function exits returning false, otherwise it will place the
 * sk_buff in the next buffer to be chained and return true indicating
 * that this is in fact a non-EOP buffer.
 **/
static bool iavf_is_non_eop(struct iavf_ring *rx_ring,
			    struct libeth_rqe_info fields)
{
	u32 ntc = rx_ring->next_to_clean + 1;

	/* fetch, update, and store next to clean */
	ntc = (ntc < rx_ring->count) ? ntc : 0;
	rx_ring->next_to_clean = ntc;

	prefetch(IAVF_RX_DESC(rx_ring, ntc));

	/* if we are the last buffer then there is nothing else to do */
	if (likely(fields.eop))
		return false;

	rx_ring->rx_stats.non_eop_descs++;

	return true;
}

/**
 * iavf_extract_legacy_rx_fields - Extract fields from the Rx descriptor
 * @rx_ring: rx descriptor ring
 * @rx_desc: the descriptor to process
 *
 * Decode the Rx descriptor and extract relevant information including the
 * size, VLAN tag, Rx packet type, end of packet field and RXE field value.
 *
 * This function only operates on the VIRTCHNL_RXDID_1_32B_BASE legacy 32byte
 * descriptor writeback format.
 *
 * Return: fields extracted from the Rx descriptor.
 */
static struct libeth_rqe_info
iavf_extract_legacy_rx_fields(const struct iavf_ring *rx_ring,
			      const struct iavf_rx_desc *rx_desc)
{
	u64 qw0 = le64_to_cpu(rx_desc->qw0);
	u64 qw1 = le64_to_cpu(rx_desc->qw1);
	u64 qw2 = le64_to_cpu(rx_desc->qw2);
	struct libeth_rqe_info fields;
	bool l2tag1p, l2tag2p;

	fields.eop = FIELD_GET(IAVF_RXD_LEGACY_EOP_M, qw1);
	fields.len = FIELD_GET(IAVF_RXD_LEGACY_LENGTH_M, qw1);

	if (!fields.eop)
		return fields;

	fields.rxe = FIELD_GET(IAVF_RXD_LEGACY_RXE_M, qw1);
	fields.ptype = FIELD_GET(IAVF_RXD_LEGACY_PTYPE_M, qw1);
	fields.vlan = 0;

	if (rx_ring->flags & IAVF_TXRX_FLAGS_VLAN_TAG_LOC_L2TAG1) {
		l2tag1p = FIELD_GET(IAVF_RXD_LEGACY_L2TAG1P_M, qw1);
		if (l2tag1p)
			fields.vlan = FIELD_GET(IAVF_RXD_LEGACY_L2TAG1_M, qw0);
	} else if (rx_ring->flags & IAVF_RXR_FLAGS_VLAN_TAG_LOC_L2TAG2_2) {
		l2tag2p = FIELD_GET(IAVF_RXD_LEGACY_L2TAG2P_M, qw2);
		if (l2tag2p)
			fields.vlan = FIELD_GET(IAVF_RXD_LEGACY_L2TAG2_M, qw2);
	}

	return fields;
}

/**
 * iavf_extract_flex_rx_fields - Extract fields from the Rx descriptor
 * @rx_ring: rx descriptor ring
 * @rx_desc: the descriptor to process
 *
 * Decode the Rx descriptor and extract relevant information including the
 * size, VLAN tag, Rx packet type, end of packet field and RXE field value.
 *
 * This function only operates on the VIRTCHNL_RXDID_2_FLEX_SQ_NIC flexible
 * descriptor writeback format.
 *
 * Return: fields extracted from the Rx descriptor.
 */
static struct libeth_rqe_info
iavf_extract_flex_rx_fields(const struct iavf_ring *rx_ring,
			    const struct iavf_rx_desc *rx_desc)
{
	struct libeth_rqe_info fields = {};
	u64 qw0 = le64_to_cpu(rx_desc->qw0);
	u64 qw1 = le64_to_cpu(rx_desc->qw1);
	u64 qw2 = le64_to_cpu(rx_desc->qw2);
	bool l2tag1p, l2tag2p;

	fields.eop = FIELD_GET(IAVF_RXD_FLEX_EOP_M, qw1);
	fields.len = FIELD_GET(IAVF_RXD_FLEX_PKT_LEN_M, qw0);

	if (!fields.eop)
		return fields;

	fields.rxe = FIELD_GET(IAVF_RXD_FLEX_RXE_M, qw1);
	fields.ptype = FIELD_GET(IAVF_RXD_FLEX_PTYPE_M, qw0);
	fields.vlan = 0;

	if (rx_ring->flags & IAVF_TXRX_FLAGS_VLAN_TAG_LOC_L2TAG1) {
		l2tag1p = FIELD_GET(IAVF_RXD_FLEX_L2TAG1P_M, qw1);
		if (l2tag1p)
			fields.vlan = FIELD_GET(IAVF_RXD_FLEX_L2TAG1_M, qw1);
	} else if (rx_ring->flags & IAVF_RXR_FLAGS_VLAN_TAG_LOC_L2TAG2_2) {
		l2tag2p = FIELD_GET(IAVF_RXD_FLEX_L2TAG2P_M, qw2);
		if (l2tag2p)
			fields.vlan = FIELD_GET(IAVF_RXD_FLEX_L2TAG2_2_M, qw2);
	}

	return fields;
}

static struct libeth_rqe_info
iavf_extract_rx_fields(const struct iavf_ring *rx_ring,
		       const struct iavf_rx_desc *rx_desc,
		       bool flex)
{
	if (flex)
		return iavf_extract_flex_rx_fields(rx_ring, rx_desc);
	else
		return iavf_extract_legacy_rx_fields(rx_ring, rx_desc);
}

/**
 * iavf_clean_rx_irq - Clean completed descriptors from Rx ring - bounce buf
 * @rx_ring: rx descriptor ring to transact packets on
 * @budget: Total limit on number of packets to process
 *
 * This function provides a "bounce buffer" approach to Rx interrupt
 * processing.  The advantage to this is that on systems that have
 * expensive overhead for IOMMU access this provides a means of avoiding
 * it by maintaining the mapping of the page to the system.
 *
 * Returns amount of work completed
 **/
static int iavf_clean_rx_irq(struct iavf_ring *rx_ring, int budget)
{
	bool flex = rx_ring->rxdid == VIRTCHNL_RXDID_2_FLEX_SQ_NIC;
	unsigned int total_rx_bytes = 0, total_rx_packets = 0;
	struct sk_buff *skb = rx_ring->skb;
	u16 cleaned_count = IAVF_DESC_UNUSED(rx_ring);
	bool failure = false;

	while (likely(total_rx_packets < (unsigned int)budget)) {
		struct libeth_rqe_info fields;
		struct libeth_fqe *rx_buffer;
		struct iavf_rx_desc *rx_desc;
		u64 qw1;

		/* return some buffers to hardware, one at a time is too slow */
		if (cleaned_count >= IAVF_RX_BUFFER_WRITE) {
			failure = failure ||
				  iavf_alloc_rx_buffers(rx_ring, cleaned_count);
			cleaned_count = 0;
		}

		rx_desc = IAVF_RX_DESC(rx_ring, rx_ring->next_to_clean);

		/* This memory barrier is needed to keep us from reading
		 * any other fields out of the rx_desc until we have
		 * verified the descriptor has been written back.
		 */
		dma_rmb();

		qw1 = le64_to_cpu(rx_desc->qw1);
		/* If DD field (descriptor done) is unset then other fields are
		 * not valid
		 */
		if (!iavf_is_descriptor_done(qw1, flex))
			break;

		fields = iavf_extract_rx_fields(rx_ring, rx_desc, flex);

		iavf_trace(clean_rx_irq, rx_ring, rx_desc, skb);

		rx_buffer = &rx_ring->rx_fqes[rx_ring->next_to_clean];
		if (!libeth_rx_sync_for_cpu(rx_buffer, fields.len))
			goto skip_data;

		/* retrieve a buffer from the ring */
		if (skb)
			iavf_add_rx_frag(skb, rx_buffer, fields.len);
		else
			skb = iavf_build_skb(rx_buffer, fields.len);

		/* exit if we failed to retrieve a buffer */
		if (!skb) {
			rx_ring->rx_stats.alloc_buff_failed++;
			break;
		}

skip_data:
		cleaned_count++;

		if (iavf_is_non_eop(rx_ring, fields) || unlikely(!skb))
			continue;

		/* RXE field in descriptor is an indication of the MAC errors
		 * (like CRC, alignment, oversize etc). If it is set then iavf
		 * should finish.
		 */
		if (unlikely(fields.rxe)) {
			dev_kfree_skb_any(skb);
			skb = NULL;
			continue;
		}

		if (iavf_cleanup_headers(rx_ring, skb)) {
			skb = NULL;
			continue;
		}

		/* probably a little skewed due to removing CRC */
		total_rx_bytes += skb->len;

		/* populate checksum, VLAN, and protocol */
		iavf_process_skb_fields(rx_ring, rx_desc, skb, fields.ptype, flex);

		iavf_trace(clean_rx_irq_rx, rx_ring, rx_desc, skb);
		iavf_receive_skb(rx_ring, skb, fields.vlan);
		skb = NULL;

		/* update budget accounting */
		total_rx_packets++;
	}

	rx_ring->skb = skb;

	u64_stats_update_begin(&rx_ring->syncp);
	rx_ring->stats.packets += total_rx_packets;
	rx_ring->stats.bytes += total_rx_bytes;
	u64_stats_update_end(&rx_ring->syncp);
	rx_ring->q_vector->rx.total_packets += total_rx_packets;
	rx_ring->q_vector->rx.total_bytes += total_rx_bytes;

	/* guarantee a trip back through this routine if there was a failure */
	return failure ? budget : (int)total_rx_packets;
}

static inline u32 iavf_buildreg_itr(const int type, u16 itr)
{
	u32 val;

	/* We don't bother with setting the CLEARPBA bit as the data sheet
	 * points out doing so is "meaningless since it was already
	 * auto-cleared". The auto-clearing happens when the interrupt is
	 * asserted.
	 *
	 * Hardware errata 28 for also indicates that writing to a
	 * xxINT_DYN_CTLx CSR with INTENA_MSK (bit 31) set to 0 will clear
	 * an event in the PBA anyway so we need to rely on the automask
	 * to hold pending events for us until the interrupt is re-enabled
	 *
	 * The itr value is reported in microseconds, and the register
	 * value is recorded in 2 microsecond units. For this reason we
	 * only need to shift by the interval shift - 1 instead of the
	 * full value.
	 */
	itr &= IAVF_ITR_MASK;

	val = IAVF_VFINT_DYN_CTLN1_INTENA_MASK |
	      (type << IAVF_VFINT_DYN_CTLN1_ITR_INDX_SHIFT) |
	      (itr << (IAVF_VFINT_DYN_CTLN1_INTERVAL_SHIFT - 1));

	return val;
}

/* a small macro to shorten up some long lines */
#define INTREG IAVF_VFINT_DYN_CTLN1

/* The act of updating the ITR will cause it to immediately trigger. In order
 * to prevent this from throwing off adaptive update statistics we defer the
 * update so that it can only happen so often. So after either Tx or Rx are
 * updated we make the adaptive scheme wait until either the ITR completely
 * expires via the next_update expiration or we have been through at least
 * 3 interrupts.
 */
#define ITR_COUNTDOWN_START 3

/**
 * iavf_update_enable_itr - Update itr and re-enable MSIX interrupt
 * @vsi: the VSI we care about
 * @q_vector: q_vector for which itr is being updated and interrupt enabled
 *
 **/
static void iavf_update_enable_itr(struct iavf_vsi *vsi,
				   struct iavf_q_vector *q_vector)
{
	struct iavf_hw *hw = &vsi->back->hw;
	u32 intval;

	/* These will do nothing if dynamic updates are not enabled */
	iavf_update_itr(q_vector, &q_vector->tx);
	iavf_update_itr(q_vector, &q_vector->rx);

	/* This block of logic allows us to get away with only updating
	 * one ITR value with each interrupt. The idea is to perform a
	 * pseudo-lazy update with the following criteria.
	 *
	 * 1. Rx is given higher priority than Tx if both are in same state
	 * 2. If we must reduce an ITR that is given highest priority.
	 * 3. We then give priority to increasing ITR based on amount.
	 */
	if (q_vector->rx.target_itr < q_vector->rx.current_itr) {
		/* Rx ITR needs to be reduced, this is highest priority */
		intval = iavf_buildreg_itr(IAVF_RX_ITR,
					   q_vector->rx.target_itr);
		q_vector->rx.current_itr = q_vector->rx.target_itr;
		q_vector->itr_countdown = ITR_COUNTDOWN_START;
	} else if ((q_vector->tx.target_itr < q_vector->tx.current_itr) ||
		   ((q_vector->rx.target_itr - q_vector->rx.current_itr) <
		    (q_vector->tx.target_itr - q_vector->tx.current_itr))) {
		/* Tx ITR needs to be reduced, this is second priority
		 * Tx ITR needs to be increased more than Rx, fourth priority
		 */
		intval = iavf_buildreg_itr(IAVF_TX_ITR,
					   q_vector->tx.target_itr);
		q_vector->tx.current_itr = q_vector->tx.target_itr;
		q_vector->itr_countdown = ITR_COUNTDOWN_START;
	} else if (q_vector->rx.current_itr != q_vector->rx.target_itr) {
		/* Rx ITR needs to be increased, third priority */
		intval = iavf_buildreg_itr(IAVF_RX_ITR,
					   q_vector->rx.target_itr);
		q_vector->rx.current_itr = q_vector->rx.target_itr;
		q_vector->itr_countdown = ITR_COUNTDOWN_START;
	} else {
		/* No ITR update, lowest priority */
		intval = iavf_buildreg_itr(IAVF_ITR_NONE, 0);
		if (q_vector->itr_countdown)
			q_vector->itr_countdown--;
	}

	if (!test_bit(__IAVF_VSI_DOWN, vsi->state))
		wr32(hw, INTREG(q_vector->reg_idx), intval);
}

/**
 * iavf_napi_poll - NAPI polling Rx/Tx cleanup routine
 * @napi: napi struct with our devices info in it
 * @budget: amount of work driver is allowed to do this pass, in packets
 *
 * This function will clean all queues associated with a q_vector.
 *
 * Returns the amount of work done
 **/
int iavf_napi_poll(struct napi_struct *napi, int budget)
{
	struct iavf_q_vector *q_vector =
			       container_of(napi, struct iavf_q_vector, napi);
	struct iavf_vsi *vsi = q_vector->vsi;
	struct iavf_ring *ring;
	bool clean_complete = true;
	bool arm_wb = false;
	int budget_per_ring;
	int work_done = 0;

	if (test_bit(__IAVF_VSI_DOWN, vsi->state)) {
		napi_complete(napi);
		return 0;
	}

	/* Since the actual Tx work is minimal, we can give the Tx a larger
	 * budget and be more aggressive about cleaning up the Tx descriptors.
	 */
	iavf_for_each_ring(ring, q_vector->tx) {
		if (!iavf_clean_tx_irq(vsi, ring, budget)) {
			clean_complete = false;
			continue;
		}
		arm_wb |= !!(ring->flags & IAVF_TXR_FLAGS_ARM_WB);
		ring->flags &= ~IAVF_TXR_FLAGS_ARM_WB;
	}

	/* Handle case where we are called by netpoll with a budget of 0 */
	if (budget <= 0)
		goto tx_only;

	/* We attempt to distribute budget to each Rx queue fairly, but don't
	 * allow the budget to go below 1 because that would exit polling early.
	 */
	budget_per_ring = max(budget/q_vector->num_ringpairs, 1);

	iavf_for_each_ring(ring, q_vector->rx) {
		int cleaned = iavf_clean_rx_irq(ring, budget_per_ring);

		work_done += cleaned;
		/* if we clean as many as budgeted, we must not be done */
		if (cleaned >= budget_per_ring)
			clean_complete = false;
	}

	/* If work not completed, return budget and polling will return */
	if (!clean_complete) {
		int cpu_id = smp_processor_id();

		/* It is possible that the interrupt affinity has changed but,
		 * if the cpu is pegged at 100%, polling will never exit while
		 * traffic continues and the interrupt will be stuck on this
		 * cpu.  We check to make sure affinity is correct before we
		 * continue to poll, otherwise we must stop polling so the
		 * interrupt can move to the correct cpu.
		 */
		if (!cpumask_test_cpu(cpu_id,
				      &q_vector->napi.config->affinity_mask)) {
			/* Tell napi that we are done polling */
			napi_complete_done(napi, work_done);

			/* Force an interrupt */
			iavf_force_wb(vsi, q_vector);

			/* Return budget-1 so that polling stops */
			return budget - 1;
		}
tx_only:
		if (arm_wb) {
			q_vector->tx.ring[0].tx_stats.tx_force_wb++;
			iavf_enable_wb_on_itr(vsi, q_vector);
		}
		return budget;
	}

	if (vsi->back->flags & IAVF_TXR_FLAGS_WB_ON_ITR)
		q_vector->arm_wb_state = false;

	/* Exit the polling mode, but don't re-enable interrupts if stack might
	 * poll us due to busy-polling
	 */
	if (likely(napi_complete_done(napi, work_done)))
		iavf_update_enable_itr(vsi, q_vector);

	return min_t(int, work_done, budget - 1);
}

/**
 * iavf_tx_prepare_vlan_flags - prepare generic TX VLAN tagging flags for HW
 * @skb:     send buffer
 * @tx_ring: ring to send buffer on
 * @flags:   the tx flags to be set
 *
 * Checks the skb and set up correspondingly several generic transmit flags
 * related to VLAN tagging for the HW, such as VLAN, DCB, etc.
 *
 * Returns error code indicate the frame should be dropped upon error and the
 * otherwise  returns 0 to indicate the flags has been set properly.
 **/
static void iavf_tx_prepare_vlan_flags(struct sk_buff *skb,
				       struct iavf_ring *tx_ring, u32 *flags)
{
	u32  tx_flags = 0;


	/* stack will only request hardware VLAN insertion offload for protocols
	 * that the driver supports and has enabled
	 */
	if (!skb_vlan_tag_present(skb))
		return;

	tx_flags |= skb_vlan_tag_get(skb) << IAVF_TX_FLAGS_VLAN_SHIFT;
	if (tx_ring->flags & IAVF_TXR_FLAGS_VLAN_TAG_LOC_L2TAG2) {
		tx_flags |= IAVF_TX_FLAGS_HW_OUTER_SINGLE_VLAN;
	} else if (tx_ring->flags & IAVF_TXRX_FLAGS_VLAN_TAG_LOC_L2TAG1) {
		tx_flags |= IAVF_TX_FLAGS_HW_VLAN;
	} else {
		dev_dbg(tx_ring->dev, "Unsupported Tx VLAN tag location requested\n");
		return;
	}

	*flags = tx_flags;
}

/**
 * iavf_tso - set up the tso context descriptor
 * @first:    pointer to first Tx buffer for xmit
 * @hdr_len:  ptr to the size of the packet header
 * @cd_type_cmd_tso_mss: Quad Word 1
 *
 * Returns 0 if no TSO can happen, 1 if tso is going, or error
 **/
static int iavf_tso(struct iavf_tx_buffer *first, u8 *hdr_len,
		    u64 *cd_type_cmd_tso_mss)
{
	struct sk_buff *skb = first->skb;
	u64 cd_cmd, cd_tso_len, cd_mss;
	union {
		struct iphdr *v4;
		struct ipv6hdr *v6;
		unsigned char *hdr;
	} ip;
	union {
		struct tcphdr *tcp;
		struct udphdr *udp;
		unsigned char *hdr;
	} l4;
	u32 paylen, l4_offset;
	u16 gso_segs, gso_size;
	int err;

	if (skb->ip_summed != CHECKSUM_PARTIAL)
		return 0;

	if (!skb_is_gso(skb))
		return 0;

	err = skb_cow_head(skb, 0);
	if (err < 0)
		return err;

	ip.hdr = skb_network_header(skb);
	l4.hdr = skb_transport_header(skb);

	/* initialize outer IP header fields */
	if (ip.v4->version == 4) {
		ip.v4->tot_len = 0;
		ip.v4->check = 0;
	} else {
		ip.v6->payload_len = 0;
	}

	if (skb_shinfo(skb)->gso_type & (SKB_GSO_GRE |
					 SKB_GSO_GRE_CSUM |
					 SKB_GSO_IPXIP4 |
					 SKB_GSO_IPXIP6 |
					 SKB_GSO_UDP_TUNNEL |
					 SKB_GSO_UDP_TUNNEL_CSUM)) {
		if (!(skb_shinfo(skb)->gso_type & SKB_GSO_PARTIAL) &&
		    (skb_shinfo(skb)->gso_type & SKB_GSO_UDP_TUNNEL_CSUM)) {
			l4.udp->len = 0;

			/* determine offset of outer transport header */
			l4_offset = l4.hdr - skb->data;

			/* remove payload length from outer checksum */
			paylen = skb->len - l4_offset;
			csum_replace_by_diff(&l4.udp->check,
					     (__force __wsum)htonl(paylen));
		}

		/* reset pointers to inner headers */
		ip.hdr = skb_inner_network_header(skb);
		l4.hdr = skb_inner_transport_header(skb);

		/* initialize inner IP header fields */
		if (ip.v4->version == 4) {
			ip.v4->tot_len = 0;
			ip.v4->check = 0;
		} else {
			ip.v6->payload_len = 0;
		}
	}

	/* determine offset of inner transport header */
	l4_offset = l4.hdr - skb->data;
	/* remove payload length from inner checksum */
	paylen = skb->len - l4_offset;

	if (skb_shinfo(skb)->gso_type & SKB_GSO_UDP_L4) {
		csum_replace_by_diff(&l4.udp->check,
				     (__force __wsum)htonl(paylen));
		/* compute length of UDP segmentation header */
		*hdr_len = (u8)sizeof(l4.udp) + l4_offset;
	} else {
		csum_replace_by_diff(&l4.tcp->check,
				     (__force __wsum)htonl(paylen));
		/* compute length of TCP segmentation header */
		*hdr_len = (u8)((l4.tcp->doff * 4) + l4_offset);
	}

	/* pull values out of skb_shinfo */
	gso_size = skb_shinfo(skb)->gso_size;
	gso_segs = skb_shinfo(skb)->gso_segs;

	/* update GSO size and bytecount with header size */
	first->gso_segs = gso_segs;
	first->bytecount += (first->gso_segs - 1) * *hdr_len;

	/* find the field values */
	cd_cmd = IAVF_TX_CTX_DESC_TSO;
	cd_tso_len = skb->len - *hdr_len;
	cd_mss = gso_size;
	*cd_type_cmd_tso_mss |= (cd_cmd << IAVF_TXD_CTX_QW1_CMD_SHIFT) |
				(cd_tso_len << IAVF_TXD_CTX_QW1_TSO_LEN_SHIFT) |
				(cd_mss << IAVF_TXD_CTX_QW1_MSS_SHIFT);
	return 1;
}

/**
 * iavf_tx_enable_csum - Enable Tx checksum offloads
 * @skb: send buffer
 * @tx_flags: pointer to Tx flags currently set
 * @td_cmd: Tx descriptor command bits to set
 * @td_offset: Tx descriptor header offsets to set
 * @tx_ring: Tx descriptor ring
 * @cd_tunneling: ptr to context desc bits
 **/
static int iavf_tx_enable_csum(struct sk_buff *skb, u32 *tx_flags,
			       u32 *td_cmd, u32 *td_offset,
			       struct iavf_ring *tx_ring,
			       u32 *cd_tunneling)
{
	union {
		struct iphdr *v4;
		struct ipv6hdr *v6;
		unsigned char *hdr;
	} ip;
	union {
		struct tcphdr *tcp;
		struct udphdr *udp;
		unsigned char *hdr;
	} l4;
	unsigned char *exthdr;
	u32 offset, cmd = 0;
	__be16 frag_off;
	u8 l4_proto = 0;

	if (skb->ip_summed != CHECKSUM_PARTIAL)
		return 0;

	ip.hdr = skb_network_header(skb);
	l4.hdr = skb_transport_header(skb);

	/* compute outer L2 header size */
	offset = ((ip.hdr - skb->data) / 2) << IAVF_TX_DESC_LENGTH_MACLEN_SHIFT;

	if (skb->encapsulation) {
		u32 tunnel = 0;
		/* define outer network header type */
		if (*tx_flags & IAVF_TX_FLAGS_IPV4) {
			tunnel |= (*tx_flags & IAVF_TX_FLAGS_TSO) ?
				  IAVF_TX_CTX_EXT_IP_IPV4 :
				  IAVF_TX_CTX_EXT_IP_IPV4_NO_CSUM;

			l4_proto = ip.v4->protocol;
		} else if (*tx_flags & IAVF_TX_FLAGS_IPV6) {
			tunnel |= IAVF_TX_CTX_EXT_IP_IPV6;

			exthdr = ip.hdr + sizeof(*ip.v6);
			l4_proto = ip.v6->nexthdr;
			if (l4.hdr != exthdr)
				ipv6_skip_exthdr(skb, exthdr - skb->data,
						 &l4_proto, &frag_off);
		}

		/* define outer transport */
		switch (l4_proto) {
		case IPPROTO_UDP:
			tunnel |= IAVF_TXD_CTX_UDP_TUNNELING;
			*tx_flags |= IAVF_TX_FLAGS_VXLAN_TUNNEL;
			break;
		case IPPROTO_GRE:
			tunnel |= IAVF_TXD_CTX_GRE_TUNNELING;
			*tx_flags |= IAVF_TX_FLAGS_VXLAN_TUNNEL;
			break;
		case IPPROTO_IPIP:
		case IPPROTO_IPV6:
			*tx_flags |= IAVF_TX_FLAGS_VXLAN_TUNNEL;
			l4.hdr = skb_inner_network_header(skb);
			break;
		default:
			if (*tx_flags & IAVF_TX_FLAGS_TSO)
				return -1;

			skb_checksum_help(skb);
			return 0;
		}

		/* compute outer L3 header size */
		tunnel |= ((l4.hdr - ip.hdr) / 4) <<
			  IAVF_TXD_CTX_QW0_EXT_IPLEN_SHIFT;

		/* switch IP header pointer from outer to inner header */
		ip.hdr = skb_inner_network_header(skb);

		/* compute tunnel header size */
		tunnel |= ((ip.hdr - l4.hdr) / 2) <<
			  IAVF_TXD_CTX_QW0_NATLEN_SHIFT;

		/* indicate if we need to offload outer UDP header */
		if ((*tx_flags & IAVF_TX_FLAGS_TSO) &&
		    !(skb_shinfo(skb)->gso_type & SKB_GSO_PARTIAL) &&
		    (skb_shinfo(skb)->gso_type & SKB_GSO_UDP_TUNNEL_CSUM))
			tunnel |= IAVF_TXD_CTX_QW0_L4T_CS_MASK;

		/* record tunnel offload values */
		*cd_tunneling |= tunnel;

		/* switch L4 header pointer from outer to inner */
		l4.hdr = skb_inner_transport_header(skb);
		l4_proto = 0;

		/* reset type as we transition from outer to inner headers */
		*tx_flags &= ~(IAVF_TX_FLAGS_IPV4 | IAVF_TX_FLAGS_IPV6);
		if (ip.v4->version == 4)
			*tx_flags |= IAVF_TX_FLAGS_IPV4;
		if (ip.v6->version == 6)
			*tx_flags |= IAVF_TX_FLAGS_IPV6;
	}

	/* Enable IP checksum offloads */
	if (*tx_flags & IAVF_TX_FLAGS_IPV4) {
		l4_proto = ip.v4->protocol;
		/* the stack computes the IP header already, the only time we
		 * need the hardware to recompute it is in the case of TSO.
		 */
		cmd |= (*tx_flags & IAVF_TX_FLAGS_TSO) ?
		       IAVF_TX_DESC_CMD_IIPT_IPV4_CSUM :
		       IAVF_TX_DESC_CMD_IIPT_IPV4;
	} else if (*tx_flags & IAVF_TX_FLAGS_IPV6) {
		cmd |= IAVF_TX_DESC_CMD_IIPT_IPV6;

		exthdr = ip.hdr + sizeof(*ip.v6);
		l4_proto = ip.v6->nexthdr;
		if (l4.hdr != exthdr)
			ipv6_skip_exthdr(skb, exthdr - skb->data,
					 &l4_proto, &frag_off);
	}

	/* compute inner L3 header size */
	offset |= ((l4.hdr - ip.hdr) / 4) << IAVF_TX_DESC_LENGTH_IPLEN_SHIFT;

	/* Enable L4 checksum offloads */
	switch (l4_proto) {
	case IPPROTO_TCP:
		/* enable checksum offloads */
		cmd |= IAVF_TX_DESC_CMD_L4T_EOFT_TCP;
		offset |= l4.tcp->doff << IAVF_TX_DESC_LENGTH_L4_FC_LEN_SHIFT;
		break;
	case IPPROTO_SCTP:
		/* enable SCTP checksum offload */
		cmd |= IAVF_TX_DESC_CMD_L4T_EOFT_SCTP;
		offset |= (sizeof(struct sctphdr) >> 2) <<
			  IAVF_TX_DESC_LENGTH_L4_FC_LEN_SHIFT;
		break;
	case IPPROTO_UDP:
		/* enable UDP checksum offload */
		cmd |= IAVF_TX_DESC_CMD_L4T_EOFT_UDP;
		offset |= (sizeof(struct udphdr) >> 2) <<
			  IAVF_TX_DESC_LENGTH_L4_FC_LEN_SHIFT;
		break;
	default:
		if (*tx_flags & IAVF_TX_FLAGS_TSO)
			return -1;
		skb_checksum_help(skb);
		return 0;
	}

	*td_cmd |= cmd;
	*td_offset |= offset;

	return 1;
}

/**
 * iavf_create_tx_ctx - Build the Tx context descriptor
 * @tx_ring:  ring to create the descriptor on
 * @cd_type_cmd_tso_mss: Quad Word 1
 * @cd_tunneling: Quad Word 0 - bits 0-31
 * @cd_l2tag2: Quad Word 0 - bits 32-63
 **/
static void iavf_create_tx_ctx(struct iavf_ring *tx_ring,
			       const u64 cd_type_cmd_tso_mss,
			       const u32 cd_tunneling, const u32 cd_l2tag2)
{
	struct iavf_tx_context_desc *context_desc;
	int i = tx_ring->next_to_use;

	if ((cd_type_cmd_tso_mss == IAVF_TX_DESC_DTYPE_CONTEXT) &&
	    !cd_tunneling && !cd_l2tag2)
		return;

	/* grab the next descriptor */
	context_desc = IAVF_TX_CTXTDESC(tx_ring, i);

	i++;
	tx_ring->next_to_use = (i < tx_ring->count) ? i : 0;

	/* cpu_to_le32 and assign to struct fields */
	context_desc->tunneling_params = cpu_to_le32(cd_tunneling);
	context_desc->l2tag2 = cpu_to_le16(cd_l2tag2);
	context_desc->rsvd = cpu_to_le16(0);
	context_desc->type_cmd_tso_mss = cpu_to_le64(cd_type_cmd_tso_mss);
}

/**
 * __iavf_chk_linearize - Check if there are more than 8 buffers per packet
 * @skb:      send buffer
 *
 * Note: Our HW can't DMA more than 8 buffers to build a packet on the wire
 * and so we need to figure out the cases where we need to linearize the skb.
 *
 * For TSO we need to count the TSO header and segment payload separately.
 * As such we need to check cases where we have 7 fragments or more as we
 * can potentially require 9 DMA transactions, 1 for the TSO header, 1 for
 * the segment payload in the first descriptor, and another 7 for the
 * fragments.
 **/
bool __iavf_chk_linearize(struct sk_buff *skb)
{
	const skb_frag_t *frag, *stale;
	int nr_frags, sum;

	/* no need to check if number of frags is less than 7 */
	nr_frags = skb_shinfo(skb)->nr_frags;
	if (nr_frags < (IAVF_MAX_BUFFER_TXD - 1))
		return false;

	/* We need to walk through the list and validate that each group
	 * of 6 fragments totals at least gso_size.
	 */
	nr_frags -= IAVF_MAX_BUFFER_TXD - 2;
	frag = &skb_shinfo(skb)->frags[0];

	/* Initialize size to the negative value of gso_size minus 1.  We
	 * use this as the worst case scenerio in which the frag ahead
	 * of us only provides one byte which is why we are limited to 6
	 * descriptors for a single transmit as the header and previous
	 * fragment are already consuming 2 descriptors.
	 */
	sum = 1 - skb_shinfo(skb)->gso_size;

	/* Add size of frags 0 through 4 to create our initial sum */
	sum += skb_frag_size(frag++);
	sum += skb_frag_size(frag++);
	sum += skb_frag_size(frag++);
	sum += skb_frag_size(frag++);
	sum += skb_frag_size(frag++);

	/* Walk through fragments adding latest fragment, testing it, and
	 * then removing stale fragments from the sum.
	 */
	for (stale = &skb_shinfo(skb)->frags[0];; stale++) {
		int stale_size = skb_frag_size(stale);

		sum += skb_frag_size(frag++);

		/* The stale fragment may present us with a smaller
		 * descriptor than the actual fragment size. To account
		 * for that we need to remove all the data on the front and
		 * figure out what the remainder would be in the last
		 * descriptor associated with the fragment.
		 */
		if (stale_size > IAVF_MAX_DATA_PER_TXD) {
			int align_pad = -(skb_frag_off(stale)) &
					(IAVF_MAX_READ_REQ_SIZE - 1);

			sum -= align_pad;
			stale_size -= align_pad;

			do {
				sum -= IAVF_MAX_DATA_PER_TXD_ALIGNED;
				stale_size -= IAVF_MAX_DATA_PER_TXD_ALIGNED;
			} while (stale_size > IAVF_MAX_DATA_PER_TXD);
		}

		/* if sum is negative we failed to make sufficient progress */
		if (sum < 0)
			return true;

		if (!nr_frags--)
			break;

		sum -= stale_size;
	}

	return false;
}

/**
 * __iavf_maybe_stop_tx - 2nd level check for tx stop conditions
 * @tx_ring: the ring to be checked
 * @size:    the size buffer we want to assure is available
 *
 * Returns -EBUSY if a stop is needed, else 0
 **/
int __iavf_maybe_stop_tx(struct iavf_ring *tx_ring, int size)
{
	netif_stop_subqueue(tx_ring->netdev, tx_ring->queue_index);
	/* Memory barrier before checking head and tail */
	smp_mb();

	/* Check again in a case another CPU has just made room available. */
	if (likely(IAVF_DESC_UNUSED(tx_ring) < size))
		return -EBUSY;

	/* A reprieve! - use start_queue because it doesn't call schedule */
	netif_start_subqueue(tx_ring->netdev, tx_ring->queue_index);
	++tx_ring->tx_stats.restart_queue;
	return 0;
}

/**
 * iavf_tx_map - Build the Tx descriptor
 * @tx_ring:  ring to send buffer on
 * @skb:      send buffer
 * @first:    first buffer info buffer to use
 * @tx_flags: collected send information
 * @hdr_len:  size of the packet header
 * @td_cmd:   the command field in the descriptor
 * @td_offset: offset for checksum or crc
 **/
static void iavf_tx_map(struct iavf_ring *tx_ring, struct sk_buff *skb,
			struct iavf_tx_buffer *first, u32 tx_flags,
			const u8 hdr_len, u32 td_cmd, u32 td_offset)
{
	unsigned int data_len = skb->data_len;
	unsigned int size = skb_headlen(skb);
	skb_frag_t *frag;
	struct iavf_tx_buffer *tx_bi;
	struct iavf_tx_desc *tx_desc;
	u16 i = tx_ring->next_to_use;
	u32 td_tag = 0;
	dma_addr_t dma;

	if (tx_flags & IAVF_TX_FLAGS_HW_VLAN) {
		td_cmd |= IAVF_TX_DESC_CMD_IL2TAG1;
		td_tag = FIELD_GET(IAVF_TX_FLAGS_VLAN_MASK, tx_flags);
	}

	first->tx_flags = tx_flags;

	dma = dma_map_single(tx_ring->dev, skb->data, size, DMA_TO_DEVICE);

	tx_desc = IAVF_TX_DESC(tx_ring, i);
	tx_bi = first;

	for (frag = &skb_shinfo(skb)->frags[0];; frag++) {
		unsigned int max_data = IAVF_MAX_DATA_PER_TXD_ALIGNED;

		if (dma_mapping_error(tx_ring->dev, dma))
			goto dma_error;

		/* record length, and DMA address */
		dma_unmap_len_set(tx_bi, len, size);
		dma_unmap_addr_set(tx_bi, dma, dma);

		/* align size to end of page */
		max_data += -dma & (IAVF_MAX_READ_REQ_SIZE - 1);
		tx_desc->buffer_addr = cpu_to_le64(dma);

		while (unlikely(size > IAVF_MAX_DATA_PER_TXD)) {
			tx_desc->cmd_type_offset_bsz =
				build_ctob(td_cmd, td_offset,
					   max_data, td_tag);

			tx_desc++;
			i++;

			if (i == tx_ring->count) {
				tx_desc = IAVF_TX_DESC(tx_ring, 0);
				i = 0;
			}

			dma += max_data;
			size -= max_data;

			max_data = IAVF_MAX_DATA_PER_TXD_ALIGNED;
			tx_desc->buffer_addr = cpu_to_le64(dma);
		}

		if (likely(!data_len))
			break;

		tx_desc->cmd_type_offset_bsz = build_ctob(td_cmd, td_offset,
							  size, td_tag);

		tx_desc++;
		i++;

		if (i == tx_ring->count) {
			tx_desc = IAVF_TX_DESC(tx_ring, 0);
			i = 0;
		}

		size = skb_frag_size(frag);
		data_len -= size;

		dma = skb_frag_dma_map(tx_ring->dev, frag, 0, size,
				       DMA_TO_DEVICE);

		tx_bi = &tx_ring->tx_bi[i];
	}

	netdev_tx_sent_queue(txring_txq(tx_ring), first->bytecount);

	i++;
	if (i == tx_ring->count)
		i = 0;

	tx_ring->next_to_use = i;

	iavf_maybe_stop_tx(tx_ring, DESC_NEEDED);

	/* write last descriptor with RS and EOP bits */
	td_cmd |= IAVF_TXD_CMD;
	tx_desc->cmd_type_offset_bsz =
			build_ctob(td_cmd, td_offset, size, td_tag);

	skb_tx_timestamp(skb);

	/* Force memory writes to complete before letting h/w know there
	 * are new descriptors to fetch.
	 *
	 * We also use this memory barrier to make certain all of the
	 * status bits have been updated before next_to_watch is written.
	 */
	wmb();

	/* set next_to_watch value indicating a packet is present */
	first->next_to_watch = tx_desc;

	/* notify HW of packet */
	if (netif_xmit_stopped(txring_txq(tx_ring)) || !netdev_xmit_more()) {
		writel(i, tx_ring->tail);
	}

	return;

dma_error:
	dev_info(tx_ring->dev, "TX DMA map failed\n");

	/* clear dma mappings for failed tx_bi map */
	for (;;) {
		tx_bi = &tx_ring->tx_bi[i];
		iavf_unmap_and_free_tx_resource(tx_ring, tx_bi);
		if (tx_bi == first)
			break;
		if (i == 0)
			i = tx_ring->count;
		i--;
	}

	tx_ring->next_to_use = i;
}

/**
 * iavf_xmit_frame_ring - Sends buffer on Tx ring
 * @skb:     send buffer
 * @tx_ring: ring to send buffer on
 *
 * Returns NETDEV_TX_OK if sent, else an error code
 **/
static netdev_tx_t iavf_xmit_frame_ring(struct sk_buff *skb,
					struct iavf_ring *tx_ring)
{
	u64 cd_type_cmd_tso_mss = IAVF_TX_DESC_DTYPE_CONTEXT;
	u32 cd_tunneling = 0, cd_l2tag2 = 0;
	struct iavf_tx_buffer *first;
	u32 td_offset = 0;
	u32 tx_flags = 0;
	__be16 protocol;
	u32 td_cmd = 0;
	u8 hdr_len = 0;
	int tso, count;

	/* prefetch the data, we'll need it later */
	prefetch(skb->data);

	iavf_trace(xmit_frame_ring, skb, tx_ring);

	count = iavf_xmit_descriptor_count(skb);
	if (iavf_chk_linearize(skb, count)) {
		if (__skb_linearize(skb)) {
			dev_kfree_skb_any(skb);
			return NETDEV_TX_OK;
		}
		count = iavf_txd_use_count(skb->len);
		tx_ring->tx_stats.tx_linearize++;
	}

	/* need: 1 descriptor per page * PAGE_SIZE/IAVF_MAX_DATA_PER_TXD,
	 *       + 1 desc for skb_head_len/IAVF_MAX_DATA_PER_TXD,
	 *       + 4 desc gap to avoid the cache line where head is,
	 *       + 1 desc for context descriptor,
	 * otherwise try next time
	 */
	if (iavf_maybe_stop_tx(tx_ring, count + 4 + 1)) {
		tx_ring->tx_stats.tx_busy++;
		return NETDEV_TX_BUSY;
	}

	/* record the location of the first descriptor for this packet */
	first = &tx_ring->tx_bi[tx_ring->next_to_use];
	first->skb = skb;
	first->bytecount = skb->len;
	first->gso_segs = 1;

	/* prepare the xmit flags */
	iavf_tx_prepare_vlan_flags(skb, tx_ring, &tx_flags);
	if (tx_flags & IAVF_TX_FLAGS_HW_OUTER_SINGLE_VLAN) {
		cd_type_cmd_tso_mss |= IAVF_TX_CTX_DESC_IL2TAG2 <<
			IAVF_TXD_CTX_QW1_CMD_SHIFT;
		cd_l2tag2 = FIELD_GET(IAVF_TX_FLAGS_VLAN_MASK, tx_flags);
	}

	/* obtain protocol of skb */
	protocol = vlan_get_protocol(skb);

	/* setup IPv4/IPv6 offloads */
	if (protocol == htons(ETH_P_IP))
		tx_flags |= IAVF_TX_FLAGS_IPV4;
	else if (protocol == htons(ETH_P_IPV6))
		tx_flags |= IAVF_TX_FLAGS_IPV6;

	tso = iavf_tso(first, &hdr_len, &cd_type_cmd_tso_mss);

	if (tso < 0)
		goto out_drop;
	else if (tso)
		tx_flags |= IAVF_TX_FLAGS_TSO;

	/* Always offload the checksum, since it's in the data descriptor */
	tso = iavf_tx_enable_csum(skb, &tx_flags, &td_cmd, &td_offset,
				  tx_ring, &cd_tunneling);
	if (tso < 0)
		goto out_drop;

	/* always enable CRC insertion offload */
	td_cmd |= IAVF_TX_DESC_CMD_ICRC;

	iavf_create_tx_ctx(tx_ring, cd_type_cmd_tso_mss,
			   cd_tunneling, cd_l2tag2);

	iavf_tx_map(tx_ring, skb, first, tx_flags, hdr_len,
		    td_cmd, td_offset);

	return NETDEV_TX_OK;

out_drop:
	iavf_trace(xmit_frame_ring_drop, first->skb, tx_ring);
	dev_kfree_skb_any(first->skb);
	first->skb = NULL;
	return NETDEV_TX_OK;
}

/**
 * iavf_xmit_frame - Selects the correct VSI and Tx queue to send buffer
 * @skb:    send buffer
 * @netdev: network interface device structure
 *
 * Returns NETDEV_TX_OK if sent, else an error code
 **/
netdev_tx_t iavf_xmit_frame(struct sk_buff *skb, struct net_device *netdev)
{
	struct iavf_adapter *adapter = netdev_priv(netdev);
	struct iavf_ring *tx_ring = &adapter->tx_rings[skb->queue_mapping];

	/* hardware can't handle really short frames, hardware padding works
	 * beyond this point
	 */
	if (unlikely(skb->len < IAVF_MIN_TX_LEN)) {
		if (skb_pad(skb, IAVF_MIN_TX_LEN - skb->len))
			return NETDEV_TX_OK;
		skb->len = IAVF_MIN_TX_LEN;
		skb_set_tail_pointer(skb, IAVF_MIN_TX_LEN);
	}

	return iavf_xmit_frame_ring(skb, tx_ring);
}
