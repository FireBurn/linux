// SPDX-License-Identifier: GPL-2.0
/* Copyright(c) 1999 - 2024 Intel Corporation. */

#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/iopoll.h>
#include <linux/sched.h>

#include "ixgbe.h"
#include "ixgbe_phy.h"

static void ixgbe_i2c_start(struct ixgbe_hw *hw);
static void ixgbe_i2c_stop(struct ixgbe_hw *hw);
static int ixgbe_clock_in_i2c_byte(struct ixgbe_hw *hw, u8 *data);
static int ixgbe_clock_out_i2c_byte(struct ixgbe_hw *hw, u8 data);
static int ixgbe_get_i2c_ack(struct ixgbe_hw *hw);
static int ixgbe_clock_in_i2c_bit(struct ixgbe_hw *hw, bool *data);
static int ixgbe_clock_out_i2c_bit(struct ixgbe_hw *hw, bool data);
static void ixgbe_raise_i2c_clk(struct ixgbe_hw *hw, u32 *i2cctl);
static void ixgbe_lower_i2c_clk(struct ixgbe_hw *hw, u32 *i2cctl);
static int ixgbe_set_i2c_data(struct ixgbe_hw *hw, u32 *i2cctl, bool data);
static bool ixgbe_get_i2c_data(struct ixgbe_hw *hw, u32 *i2cctl);
static void ixgbe_i2c_bus_clear(struct ixgbe_hw *hw);
static enum ixgbe_phy_type ixgbe_get_phy_type_from_id(u32 phy_id);
static int ixgbe_get_phy_id(struct ixgbe_hw *hw);
static int ixgbe_identify_qsfp_module_generic(struct ixgbe_hw *hw);

/**
 *  ixgbe_out_i2c_byte_ack - Send I2C byte with ack
 *  @hw: pointer to the hardware structure
 *  @byte: byte to send
 *
 *  Returns an error code on error.
 **/
static int ixgbe_out_i2c_byte_ack(struct ixgbe_hw *hw, u8 byte)
{
	int status;

	status = ixgbe_clock_out_i2c_byte(hw, byte);
	if (status)
		return status;
	return ixgbe_get_i2c_ack(hw);
}

/**
 *  ixgbe_in_i2c_byte_ack - Receive an I2C byte and send ack
 *  @hw: pointer to the hardware structure
 *  @byte: pointer to a u8 to receive the byte
 *
 *  Returns an error code on error.
 **/
static int ixgbe_in_i2c_byte_ack(struct ixgbe_hw *hw, u8 *byte)
{
	int status;

	status = ixgbe_clock_in_i2c_byte(hw, byte);
	if (status)
		return status;
	/* ACK */
	return ixgbe_clock_out_i2c_bit(hw, false);
}

/**
 *  ixgbe_ones_comp_byte_add - Perform one's complement addition
 *  @add1: addend 1
 *  @add2: addend 2
 *
 *  Returns one's complement 8-bit sum.
 **/
static u8 ixgbe_ones_comp_byte_add(u8 add1, u8 add2)
{
	u16 sum = add1 + add2;

	sum = (sum & 0xFF) + (sum >> 8);
	return sum & 0xFF;
}

/**
 *  ixgbe_read_i2c_combined_generic_int - Perform I2C read combined operation
 *  @hw: pointer to the hardware structure
 *  @addr: I2C bus address to read from
 *  @reg: I2C device register to read from
 *  @val: pointer to location to receive read value
 *  @lock: true if to take and release semaphore
 *
 *  Returns an error code on error.
 */
int ixgbe_read_i2c_combined_generic_int(struct ixgbe_hw *hw, u8 addr,
					u16 reg, u16 *val, bool lock)
{
	u32 swfw_mask = hw->phy.phy_semaphore_mask;
	int max_retry = 3;
	int retry = 0;
	u8 csum_byte;
	u8 high_bits;
	u8 low_bits;
	u8 reg_high;
	u8 csum;

	reg_high = ((reg >> 7) & 0xFE) | 1;     /* Indicate read combined */
	csum = ixgbe_ones_comp_byte_add(reg_high, reg & 0xFF);
	csum = ~csum;
	do {
		if (lock && hw->mac.ops.acquire_swfw_sync(hw, swfw_mask))
			return -EBUSY;
		ixgbe_i2c_start(hw);
		/* Device Address and write indication */
		if (ixgbe_out_i2c_byte_ack(hw, addr))
			goto fail;
		/* Write bits 14:8 */
		if (ixgbe_out_i2c_byte_ack(hw, reg_high))
			goto fail;
		/* Write bits 7:0 */
		if (ixgbe_out_i2c_byte_ack(hw, reg & 0xFF))
			goto fail;
		/* Write csum */
		if (ixgbe_out_i2c_byte_ack(hw, csum))
			goto fail;
		/* Re-start condition */
		ixgbe_i2c_start(hw);
		/* Device Address and read indication */
		if (ixgbe_out_i2c_byte_ack(hw, addr | 1))
			goto fail;
		/* Get upper bits */
		if (ixgbe_in_i2c_byte_ack(hw, &high_bits))
			goto fail;
		/* Get low bits */
		if (ixgbe_in_i2c_byte_ack(hw, &low_bits))
			goto fail;
		/* Get csum */
		if (ixgbe_clock_in_i2c_byte(hw, &csum_byte))
			goto fail;
		/* NACK */
		if (ixgbe_clock_out_i2c_bit(hw, false))
			goto fail;
		ixgbe_i2c_stop(hw);
		if (lock)
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
		*val = (high_bits << 8) | low_bits;
		return 0;

fail:
		ixgbe_i2c_bus_clear(hw);
		if (lock)
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
		retry++;
		if (retry < max_retry)
			hw_dbg(hw, "I2C byte read combined error - Retry.\n");
		else
			hw_dbg(hw, "I2C byte read combined error.\n");
	} while (retry < max_retry);

	return -EIO;
}

/**
 *  ixgbe_write_i2c_combined_generic_int - Perform I2C write combined operation
 *  @hw: pointer to the hardware structure
 *  @addr: I2C bus address to write to
 *  @reg: I2C device register to write to
 *  @val: value to write
 *  @lock: true if to take and release semaphore
 *
 *  Returns an error code on error.
 */
int ixgbe_write_i2c_combined_generic_int(struct ixgbe_hw *hw, u8 addr,
					 u16 reg, u16 val, bool lock)
{
	u32 swfw_mask = hw->phy.phy_semaphore_mask;
	int max_retry = 3;
	int retry = 0;
	u8 reg_high;
	u8 csum;

	reg_high = (reg >> 7) & 0xFE;   /* Indicate write combined */
	csum = ixgbe_ones_comp_byte_add(reg_high, reg & 0xFF);
	csum = ixgbe_ones_comp_byte_add(csum, val >> 8);
	csum = ixgbe_ones_comp_byte_add(csum, val & 0xFF);
	csum = ~csum;
	do {
		if (lock && hw->mac.ops.acquire_swfw_sync(hw, swfw_mask))
			return -EBUSY;
		ixgbe_i2c_start(hw);
		/* Device Address and write indication */
		if (ixgbe_out_i2c_byte_ack(hw, addr))
			goto fail;
		/* Write bits 14:8 */
		if (ixgbe_out_i2c_byte_ack(hw, reg_high))
			goto fail;
		/* Write bits 7:0 */
		if (ixgbe_out_i2c_byte_ack(hw, reg & 0xFF))
			goto fail;
		/* Write data 15:8 */
		if (ixgbe_out_i2c_byte_ack(hw, val >> 8))
			goto fail;
		/* Write data 7:0 */
		if (ixgbe_out_i2c_byte_ack(hw, val & 0xFF))
			goto fail;
		/* Write csum */
		if (ixgbe_out_i2c_byte_ack(hw, csum))
			goto fail;
		ixgbe_i2c_stop(hw);
		if (lock)
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
		return 0;

fail:
		ixgbe_i2c_bus_clear(hw);
		if (lock)
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
		retry++;
		if (retry < max_retry)
			hw_dbg(hw, "I2C byte write combined error - Retry.\n");
		else
			hw_dbg(hw, "I2C byte write combined error.\n");
	} while (retry < max_retry);

	return -EIO;
}

/**
 *  ixgbe_probe_phy - Probe a single address for a PHY
 *  @hw: pointer to hardware structure
 *  @phy_addr: PHY address to probe
 *
 *  Returns true if PHY found
 **/
static bool ixgbe_probe_phy(struct ixgbe_hw *hw, u16 phy_addr)
{
	u16 ext_ability = 0;

	hw->phy.mdio.prtad = phy_addr;
	if (mdio45_probe(&hw->phy.mdio, phy_addr) != 0)
		return false;

	if (ixgbe_get_phy_id(hw))
		return false;

	hw->phy.type = ixgbe_get_phy_type_from_id(hw->phy.id);

	if (hw->phy.type == ixgbe_phy_unknown) {
		hw->phy.ops.read_reg(hw,
				     MDIO_PMA_EXTABLE,
				     MDIO_MMD_PMAPMD,
				     &ext_ability);
		if (ext_ability &
		    (MDIO_PMA_EXTABLE_10GBT |
		     MDIO_PMA_EXTABLE_1000BT))
			hw->phy.type = ixgbe_phy_cu_unknown;
		else
			hw->phy.type = ixgbe_phy_generic;
	}

	return true;
}

/**
 *  ixgbe_identify_phy_generic - Get physical layer module
 *  @hw: pointer to hardware structure
 *
 *  Determines the physical layer module found on the current adapter.
 **/
int ixgbe_identify_phy_generic(struct ixgbe_hw *hw)
{
	u32 status = -EFAULT;
	u32 phy_addr;

	if (!hw->phy.phy_semaphore_mask) {
		if (hw->bus.lan_id)
			hw->phy.phy_semaphore_mask = IXGBE_GSSR_PHY1_SM;
		else
			hw->phy.phy_semaphore_mask = IXGBE_GSSR_PHY0_SM;
	}

	if (hw->phy.type != ixgbe_phy_unknown)
		return 0;

	if (hw->phy.nw_mng_if_sel) {
		phy_addr = FIELD_GET(IXGBE_NW_MNG_IF_SEL_MDIO_PHY_ADD,
				     hw->phy.nw_mng_if_sel);
		if (ixgbe_probe_phy(hw, phy_addr))
			return 0;
		else
			return -EFAULT;
	}

	for (phy_addr = 0; phy_addr < IXGBE_MAX_PHY_ADDR; phy_addr++) {
		if (ixgbe_probe_phy(hw, phy_addr)) {
			status = 0;
			break;
		}
	}

	/* Certain media types do not have a phy so an address will not
	 * be found and the code will take this path.  Caller has to
	 * decide if it is an error or not.
	 */
	if (status)
		hw->phy.mdio.prtad = MDIO_PRTAD_NONE;

	return status;
}

/**
 * ixgbe_check_reset_blocked - check status of MNG FW veto bit
 * @hw: pointer to the hardware structure
 *
 * This function checks the MMNGC.MNG_VETO bit to see if there are
 * any constraints on link from manageability.  For MAC's that don't
 * have this bit just return false since the link can not be blocked
 * via this method.
 **/
bool ixgbe_check_reset_blocked(struct ixgbe_hw *hw)
{
	u32 mmngc;

	/* If we don't have this bit, it can't be blocking */
	if (hw->mac.type == ixgbe_mac_82598EB)
		return false;

	mmngc = IXGBE_READ_REG(hw, IXGBE_MMNGC);
	if (mmngc & IXGBE_MMNGC_MNG_VETO) {
		hw_dbg(hw, "MNG_VETO bit detected.\n");
		return true;
	}

	return false;
}

/**
 *  ixgbe_get_phy_id - Get the phy type
 *  @hw: pointer to hardware structure
 *
 **/
static int ixgbe_get_phy_id(struct ixgbe_hw *hw)
{
	u16 phy_id_high = 0;
	u16 phy_id_low = 0;
	int status;

	status = hw->phy.ops.read_reg(hw, MDIO_DEVID1, MDIO_MMD_PMAPMD,
				      &phy_id_high);

	if (!status) {
		hw->phy.id = (u32)(phy_id_high << 16);
		status = hw->phy.ops.read_reg(hw, MDIO_DEVID2, MDIO_MMD_PMAPMD,
					      &phy_id_low);
		hw->phy.id |= (u32)(phy_id_low & IXGBE_PHY_REVISION_MASK);
		hw->phy.revision = (u32)(phy_id_low & ~IXGBE_PHY_REVISION_MASK);
	}
	return status;
}

/**
 *  ixgbe_get_phy_type_from_id - Get the phy type
 *  @phy_id: hardware phy id
 *
 **/
static enum ixgbe_phy_type ixgbe_get_phy_type_from_id(u32 phy_id)
{
	enum ixgbe_phy_type phy_type;

	switch (phy_id) {
	case TN1010_PHY_ID:
		phy_type = ixgbe_phy_tn;
		break;
	case X550_PHY_ID2:
	case X550_PHY_ID3:
	case X540_PHY_ID:
		phy_type = ixgbe_phy_aq;
		break;
	case QT2022_PHY_ID:
		phy_type = ixgbe_phy_qt;
		break;
	case ATH_PHY_ID:
		phy_type = ixgbe_phy_nl;
		break;
	case X557_PHY_ID:
	case X557_PHY_ID2:
		phy_type = ixgbe_phy_x550em_ext_t;
		break;
	case BCM54616S_E_PHY_ID:
		phy_type = ixgbe_phy_ext_1g_t;
		break;
	default:
		phy_type = ixgbe_phy_unknown;
		break;
	}

	return phy_type;
}

/**
 *  ixgbe_reset_phy_generic - Performs a PHY reset
 *  @hw: pointer to hardware structure
 **/
int ixgbe_reset_phy_generic(struct ixgbe_hw *hw)
{
	u32 i;
	u16 ctrl = 0;
	int status = 0;

	if (hw->phy.type == ixgbe_phy_unknown)
		status = ixgbe_identify_phy_generic(hw);

	if (status != 0 || hw->phy.type == ixgbe_phy_none)
		return status;

	/* Don't reset PHY if it's shut down due to overtemp. */
	if (!hw->phy.reset_if_overtemp && hw->phy.ops.check_overtemp(hw))
		return 0;

	/* Blocked by MNG FW so bail */
	if (ixgbe_check_reset_blocked(hw))
		return 0;

	/*
	 * Perform soft PHY reset to the PHY_XS.
	 * This will cause a soft reset to the PHY
	 */
	hw->phy.ops.write_reg(hw, MDIO_CTRL1,
			      MDIO_MMD_PHYXS,
			      MDIO_CTRL1_RESET);

	/*
	 * Poll for reset bit to self-clear indicating reset is complete.
	 * Some PHYs could take up to 3 seconds to complete and need about
	 * 1.7 usec delay after the reset is complete.
	 */
	for (i = 0; i < 30; i++) {
		msleep(100);
		if (hw->phy.type == ixgbe_phy_x550em_ext_t) {
			status = hw->phy.ops.read_reg(hw,
						  IXGBE_MDIO_TX_VENDOR_ALARMS_3,
						  MDIO_MMD_PMAPMD, &ctrl);
			if (status)
				return status;

			if (ctrl & IXGBE_MDIO_TX_VENDOR_ALARMS_3_RST_MASK) {
				udelay(2);
				break;
			}
		} else {
			status = hw->phy.ops.read_reg(hw, MDIO_CTRL1,
						      MDIO_MMD_PHYXS, &ctrl);
			if (status)
				return status;

			if (!(ctrl & MDIO_CTRL1_RESET)) {
				udelay(2);
				break;
			}
		}
	}

	if (ctrl & MDIO_CTRL1_RESET) {
		hw_dbg(hw, "PHY reset polling failed to complete.\n");
		return -EIO;
	}

	return 0;
}

/**
 *  ixgbe_read_phy_reg_mdi - read PHY register
 *  @hw: pointer to hardware structure
 *  @reg_addr: 32 bit address of PHY register to read
 *  @device_type: 5 bit device type
 *  @phy_data: Pointer to read data from PHY register
 *
 *  Reads a value from a specified PHY register without the SWFW lock
 **/
int ixgbe_read_phy_reg_mdi(struct ixgbe_hw *hw, u32 reg_addr, u32 device_type,
			   u16 *phy_data)
{
	u32 i, data, command;

	/* Setup and write the address cycle command */
	command = ((reg_addr << IXGBE_MSCA_NP_ADDR_SHIFT)  |
		   (device_type << IXGBE_MSCA_DEV_TYPE_SHIFT) |
		   (hw->phy.mdio.prtad << IXGBE_MSCA_PHY_ADDR_SHIFT) |
		   (IXGBE_MSCA_ADDR_CYCLE | IXGBE_MSCA_MDI_COMMAND));

	IXGBE_WRITE_REG(hw, IXGBE_MSCA, command);

	/* Check every 10 usec to see if the address cycle completed.
	 * The MDI Command bit will clear when the operation is
	 * complete
	 */
	for (i = 0; i < IXGBE_MDIO_COMMAND_TIMEOUT; i++) {
		udelay(10);

		command = IXGBE_READ_REG(hw, IXGBE_MSCA);
		if ((command & IXGBE_MSCA_MDI_COMMAND) == 0)
				break;
	}


	if ((command & IXGBE_MSCA_MDI_COMMAND) != 0) {
		hw_dbg(hw, "PHY address command did not complete.\n");
		return -EIO;
	}

	/* Address cycle complete, setup and write the read
	 * command
	 */
	command = ((reg_addr << IXGBE_MSCA_NP_ADDR_SHIFT)  |
		   (device_type << IXGBE_MSCA_DEV_TYPE_SHIFT) |
		   (hw->phy.mdio.prtad << IXGBE_MSCA_PHY_ADDR_SHIFT) |
		   (IXGBE_MSCA_READ | IXGBE_MSCA_MDI_COMMAND));

	IXGBE_WRITE_REG(hw, IXGBE_MSCA, command);

	/* Check every 10 usec to see if the address cycle
	 * completed. The MDI Command bit will clear when the
	 * operation is complete
	 */
	for (i = 0; i < IXGBE_MDIO_COMMAND_TIMEOUT; i++) {
		udelay(10);

		command = IXGBE_READ_REG(hw, IXGBE_MSCA);
		if ((command & IXGBE_MSCA_MDI_COMMAND) == 0)
			break;
	}

	if ((command & IXGBE_MSCA_MDI_COMMAND) != 0) {
		hw_dbg(hw, "PHY read command didn't complete\n");
		return -EIO;
	}

	/* Read operation is complete.  Get the data
	 * from MSRWD
	 */
	data = IXGBE_READ_REG(hw, IXGBE_MSRWD);
	data >>= IXGBE_MSRWD_READ_DATA_SHIFT;
	*phy_data = (u16)(data);

	return 0;
}

/**
 *  ixgbe_read_phy_reg_generic - Reads a value from a specified PHY register
 *  using the SWFW lock - this function is needed in most cases
 *  @hw: pointer to hardware structure
 *  @reg_addr: 32 bit address of PHY register to read
 *  @device_type: 5 bit device type
 *  @phy_data: Pointer to read data from PHY register
 **/
int ixgbe_read_phy_reg_generic(struct ixgbe_hw *hw, u32 reg_addr,
			       u32 device_type, u16 *phy_data)
{
	u32 gssr = hw->phy.phy_semaphore_mask;
	int status;

	if (hw->mac.ops.acquire_swfw_sync(hw, gssr) == 0) {
		status = ixgbe_read_phy_reg_mdi(hw, reg_addr, device_type,
						phy_data);
		hw->mac.ops.release_swfw_sync(hw, gssr);
	} else {
		return -EBUSY;
	}

	return status;
}

/**
 *  ixgbe_write_phy_reg_mdi - Writes a value to specified PHY register
 *  without SWFW lock
 *  @hw: pointer to hardware structure
 *  @reg_addr: 32 bit PHY register to write
 *  @device_type: 5 bit device type
 *  @phy_data: Data to write to the PHY register
 **/
int ixgbe_write_phy_reg_mdi(struct ixgbe_hw *hw, u32 reg_addr, u32 device_type,
			    u16 phy_data)
{
	u32 i, command;

	/* Put the data in the MDI single read and write data register*/
	IXGBE_WRITE_REG(hw, IXGBE_MSRWD, (u32)phy_data);

	/* Setup and write the address cycle command */
	command = ((reg_addr << IXGBE_MSCA_NP_ADDR_SHIFT)  |
		   (device_type << IXGBE_MSCA_DEV_TYPE_SHIFT) |
		   (hw->phy.mdio.prtad << IXGBE_MSCA_PHY_ADDR_SHIFT) |
		   (IXGBE_MSCA_ADDR_CYCLE | IXGBE_MSCA_MDI_COMMAND));

	IXGBE_WRITE_REG(hw, IXGBE_MSCA, command);

	/*
	 * Check every 10 usec to see if the address cycle completed.
	 * The MDI Command bit will clear when the operation is
	 * complete
	 */
	for (i = 0; i < IXGBE_MDIO_COMMAND_TIMEOUT; i++) {
		udelay(10);

		command = IXGBE_READ_REG(hw, IXGBE_MSCA);
		if ((command & IXGBE_MSCA_MDI_COMMAND) == 0)
			break;
	}

	if ((command & IXGBE_MSCA_MDI_COMMAND) != 0) {
		hw_dbg(hw, "PHY address cmd didn't complete\n");
		return -EIO;
	}

	/*
	 * Address cycle complete, setup and write the write
	 * command
	 */
	command = ((reg_addr << IXGBE_MSCA_NP_ADDR_SHIFT)  |
		   (device_type << IXGBE_MSCA_DEV_TYPE_SHIFT) |
		   (hw->phy.mdio.prtad << IXGBE_MSCA_PHY_ADDR_SHIFT) |
		   (IXGBE_MSCA_WRITE | IXGBE_MSCA_MDI_COMMAND));

	IXGBE_WRITE_REG(hw, IXGBE_MSCA, command);

	/* Check every 10 usec to see if the address cycle
	 * completed. The MDI Command bit will clear when the
	 * operation is complete
	 */
	for (i = 0; i < IXGBE_MDIO_COMMAND_TIMEOUT; i++) {
		udelay(10);

		command = IXGBE_READ_REG(hw, IXGBE_MSCA);
		if ((command & IXGBE_MSCA_MDI_COMMAND) == 0)
			break;
	}

	if ((command & IXGBE_MSCA_MDI_COMMAND) != 0) {
		hw_dbg(hw, "PHY write cmd didn't complete\n");
		return -EIO;
	}

	return 0;
}

/**
 *  ixgbe_write_phy_reg_generic - Writes a value to specified PHY register
 *  using SWFW lock- this function is needed in most cases
 *  @hw: pointer to hardware structure
 *  @reg_addr: 32 bit PHY register to write
 *  @device_type: 5 bit device type
 *  @phy_data: Data to write to the PHY register
 **/
int ixgbe_write_phy_reg_generic(struct ixgbe_hw *hw, u32 reg_addr,
				u32 device_type, u16 phy_data)
{
	u32 gssr = hw->phy.phy_semaphore_mask;
	int status;

	if (hw->mac.ops.acquire_swfw_sync(hw, gssr) == 0) {
		status = ixgbe_write_phy_reg_mdi(hw, reg_addr, device_type,
						 phy_data);
		hw->mac.ops.release_swfw_sync(hw, gssr);
	} else {
		return -EBUSY;
	}

	return status;
}

#define IXGBE_HW_READ_REG(addr) IXGBE_READ_REG(hw, addr)

/**
 *  ixgbe_msca_cmd - Write the command register and poll for completion/timeout
 *  @hw: pointer to hardware structure
 *  @cmd: command register value to write
 **/
static int ixgbe_msca_cmd(struct ixgbe_hw *hw, u32 cmd)
{
	IXGBE_WRITE_REG(hw, IXGBE_MSCA, cmd);

	return readx_poll_timeout(IXGBE_HW_READ_REG, IXGBE_MSCA, cmd,
				  !(cmd & IXGBE_MSCA_MDI_COMMAND), 10,
				  10 * IXGBE_MDIO_COMMAND_TIMEOUT);
}

/**
 *  ixgbe_mii_bus_read_generic_c22 - Read a clause 22 register with gssr flags
 *  @hw: pointer to hardware structure
 *  @addr: address
 *  @regnum: register number
 *  @gssr: semaphore flags to acquire
 **/
static int ixgbe_mii_bus_read_generic_c22(struct ixgbe_hw *hw, int addr,
					  int regnum, u32 gssr)
{
	u32 hwaddr, cmd;
	int data;

	if (hw->mac.ops.acquire_swfw_sync(hw, gssr))
		return -EBUSY;

	hwaddr = addr << IXGBE_MSCA_PHY_ADDR_SHIFT;
	hwaddr |= (regnum & GENMASK(5, 0)) << IXGBE_MSCA_DEV_TYPE_SHIFT;
	cmd = hwaddr | IXGBE_MSCA_OLD_PROTOCOL |
		IXGBE_MSCA_READ_AUTOINC | IXGBE_MSCA_MDI_COMMAND;

	data = ixgbe_msca_cmd(hw, cmd);
	if (data < 0)
		goto mii_bus_read_done;

	data = IXGBE_READ_REG(hw, IXGBE_MSRWD);
	data = (data >> IXGBE_MSRWD_READ_DATA_SHIFT) & GENMASK(16, 0);

mii_bus_read_done:
	hw->mac.ops.release_swfw_sync(hw, gssr);
	return data;
}

/**
 *  ixgbe_mii_bus_read_generic_c45 - Read a clause 45 register with gssr flags
 *  @hw: pointer to hardware structure
 *  @addr: address
 *  @devad: device address to read
 *  @regnum: register number
 *  @gssr: semaphore flags to acquire
 **/
static int ixgbe_mii_bus_read_generic_c45(struct ixgbe_hw *hw, int addr,
					  int devad, int regnum, u32 gssr)
{
	u32 hwaddr, cmd;
	int data;

	if (hw->mac.ops.acquire_swfw_sync(hw, gssr))
		return -EBUSY;

	hwaddr = addr << IXGBE_MSCA_PHY_ADDR_SHIFT;
	hwaddr |= devad << 16 | regnum;
	cmd = hwaddr | IXGBE_MSCA_ADDR_CYCLE | IXGBE_MSCA_MDI_COMMAND;

	data = ixgbe_msca_cmd(hw, cmd);
	if (data < 0)
		goto mii_bus_read_done;

	cmd = hwaddr | IXGBE_MSCA_READ | IXGBE_MSCA_MDI_COMMAND;
	data = ixgbe_msca_cmd(hw, cmd);
	if (data < 0)
		goto mii_bus_read_done;

	data = IXGBE_READ_REG(hw, IXGBE_MSRWD);
	data = (data >> IXGBE_MSRWD_READ_DATA_SHIFT) & GENMASK(16, 0);

mii_bus_read_done:
	hw->mac.ops.release_swfw_sync(hw, gssr);
	return data;
}

/**
 *  ixgbe_mii_bus_write_generic_c22 - Write a clause 22 register with gssr flags
 *  @hw: pointer to hardware structure
 *  @addr: address
 *  @regnum: register number
 *  @val: value to write
 *  @gssr: semaphore flags to acquire
 **/
static int ixgbe_mii_bus_write_generic_c22(struct ixgbe_hw *hw, int addr,
					   int regnum, u16 val, u32 gssr)
{
	u32 hwaddr, cmd;
	int err;

	if (hw->mac.ops.acquire_swfw_sync(hw, gssr))
		return -EBUSY;

	IXGBE_WRITE_REG(hw, IXGBE_MSRWD, (u32)val);

	hwaddr = addr << IXGBE_MSCA_PHY_ADDR_SHIFT;
	hwaddr |= (regnum & GENMASK(5, 0)) << IXGBE_MSCA_DEV_TYPE_SHIFT;
	cmd = hwaddr | IXGBE_MSCA_OLD_PROTOCOL | IXGBE_MSCA_WRITE |
		IXGBE_MSCA_MDI_COMMAND;

	err = ixgbe_msca_cmd(hw, cmd);

	hw->mac.ops.release_swfw_sync(hw, gssr);
	return err;
}

/**
 *  ixgbe_mii_bus_write_generic_c45 - Write a clause 45 register with gssr flags
 *  @hw: pointer to hardware structure
 *  @addr: address
 *  @devad: device address to read
 *  @regnum: register number
 *  @val: value to write
 *  @gssr: semaphore flags to acquire
 **/
static int ixgbe_mii_bus_write_generic_c45(struct ixgbe_hw *hw, int addr,
					   int devad, int regnum, u16 val,
					   u32 gssr)
{
	u32 hwaddr, cmd;
	int err;

	if (hw->mac.ops.acquire_swfw_sync(hw, gssr))
		return -EBUSY;

	IXGBE_WRITE_REG(hw, IXGBE_MSRWD, (u32)val);

	hwaddr = addr << IXGBE_MSCA_PHY_ADDR_SHIFT;
	hwaddr |= devad << 16 | regnum;
	cmd = hwaddr | IXGBE_MSCA_ADDR_CYCLE | IXGBE_MSCA_MDI_COMMAND;

	err = ixgbe_msca_cmd(hw, cmd);
	if (err < 0)
		goto mii_bus_write_done;

	cmd = hwaddr | IXGBE_MSCA_WRITE | IXGBE_MSCA_MDI_COMMAND;
	err = ixgbe_msca_cmd(hw, cmd);

mii_bus_write_done:
	hw->mac.ops.release_swfw_sync(hw, gssr);
	return err;
}

/**
 *  ixgbe_mii_bus_read_c22 - Read a clause 22 register
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @regnum: register number
 **/
static int ixgbe_mii_bus_read_c22(struct mii_bus *bus, int addr, int regnum)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	return ixgbe_mii_bus_read_generic_c22(hw, addr, regnum, gssr);
}

/**
 *  ixgbe_mii_bus_read_c45 - Read a clause 45 register
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @devad: device address to read
 *  @addr: address
 *  @regnum: register number
 **/
static int ixgbe_mii_bus_read_c45(struct mii_bus *bus, int devad, int addr,
				  int regnum)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	return ixgbe_mii_bus_read_generic_c45(hw, addr, devad, regnum, gssr);
}

/**
 *  ixgbe_mii_bus_write_c22 - Write a clause 22 register
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @regnum: register number
 *  @val: value to write
 **/
static int ixgbe_mii_bus_write_c22(struct mii_bus *bus, int addr, int regnum,
				   u16 val)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	return ixgbe_mii_bus_write_generic_c22(hw, addr, regnum, val, gssr);
}

/**
 *  ixgbe_mii_bus_write_c45 - Write a clause 45 register
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @devad: device address to read
 *  @regnum: register number
 *  @val: value to write
 **/
static int ixgbe_mii_bus_write_c45(struct mii_bus *bus, int addr, int devad,
				   int regnum, u16 val)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	return ixgbe_mii_bus_write_generic_c45(hw, addr, devad, regnum, val,
					       gssr);
}

/**
 *  ixgbe_x550em_a_mii_bus_read_c22 - Read a clause 22 register on x550em_a
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @regnum: register number
 **/
static int ixgbe_x550em_a_mii_bus_read_c22(struct mii_bus *bus, int addr,
					   int regnum)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	gssr |= IXGBE_GSSR_TOKEN_SM | IXGBE_GSSR_PHY0_SM;
	return ixgbe_mii_bus_read_generic_c22(hw, addr, regnum, gssr);
}

/**
 *  ixgbe_x550em_a_mii_bus_read_c45 - Read a clause 45 register on x550em_a
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @devad: device address to read
 *  @regnum: register number
 **/
static int ixgbe_x550em_a_mii_bus_read_c45(struct mii_bus *bus, int addr,
					   int devad, int regnum)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	gssr |= IXGBE_GSSR_TOKEN_SM | IXGBE_GSSR_PHY0_SM;
	return ixgbe_mii_bus_read_generic_c45(hw, addr, devad, regnum, gssr);
}

/**
 *  ixgbe_x550em_a_mii_bus_write_c22 - Write a clause 22 register on x550em_a
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @regnum: register number
 *  @val: value to write
 **/
static int ixgbe_x550em_a_mii_bus_write_c22(struct mii_bus *bus, int addr,
					    int regnum, u16 val)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	gssr |= IXGBE_GSSR_TOKEN_SM | IXGBE_GSSR_PHY0_SM;
	return ixgbe_mii_bus_write_generic_c22(hw, addr, regnum, val, gssr);
}

/**
 *  ixgbe_x550em_a_mii_bus_write_c45 - Write a clause 45 register on x550em_a
 *  @bus: pointer to mii_bus structure which points to our driver private
 *  @addr: address
 *  @devad: device address to read
 *  @regnum: register number
 *  @val: value to write
 **/
static int ixgbe_x550em_a_mii_bus_write_c45(struct mii_bus *bus, int addr,
					    int devad, int regnum, u16 val)
{
	struct ixgbe_adapter *adapter = bus->priv;
	struct ixgbe_hw *hw = &adapter->hw;
	u32 gssr = hw->phy.phy_semaphore_mask;

	gssr |= IXGBE_GSSR_TOKEN_SM | IXGBE_GSSR_PHY0_SM;
	return ixgbe_mii_bus_write_generic_c45(hw, addr, devad, regnum, val,
					       gssr);
}

/**
 * ixgbe_get_first_secondary_devfn - get first device downstream of root port
 * @devfn: PCI_DEVFN of root port on domain 0, bus 0
 *
 * Returns pci_dev pointer to PCI_DEVFN(0, 0) on subordinate side of root
 * on domain 0, bus 0, devfn = 'devfn'
 **/
static struct pci_dev *ixgbe_get_first_secondary_devfn(unsigned int devfn)
{
	struct pci_dev *rp_pdev;
	int bus;

	rp_pdev = pci_get_domain_bus_and_slot(0, 0, devfn);
	if (rp_pdev && rp_pdev->subordinate) {
		bus = rp_pdev->subordinate->number;
		pci_dev_put(rp_pdev);
		return pci_get_domain_bus_and_slot(0, bus, 0);
	}

	pci_dev_put(rp_pdev);
	return NULL;
}

/**
 * ixgbe_x550em_a_has_mii - is this the first ixgbe x550em_a PCI function?
 * @hw: pointer to hardware structure
 *
 * Returns true if hw points to lowest numbered PCI B:D.F x550_em_a device in
 * the SoC.  There are up to 4 MACs sharing a single MDIO bus on the x550em_a,
 * but we only want to register one MDIO bus.
 **/
static bool ixgbe_x550em_a_has_mii(struct ixgbe_hw *hw)
{
	struct ixgbe_adapter *adapter = hw->back;
	struct pci_dev *pdev = adapter->pdev;
	struct pci_dev *func0_pdev;
	bool has_mii = false;

	/* For the C3000 family of SoCs (x550em_a) the internal ixgbe devices
	 * are always downstream of root ports @ 0000:00:16.0 & 0000:00:17.0
	 * It's not valid for function 0 to be disabled and function 1 is up,
	 * so the lowest numbered ixgbe dev will be device 0 function 0 on one
	 * of those two root ports
	 */
	func0_pdev = ixgbe_get_first_secondary_devfn(PCI_DEVFN(0x16, 0));
	if (func0_pdev) {
		if (func0_pdev == pdev)
			has_mii = true;
		goto out;
	}
	func0_pdev = ixgbe_get_first_secondary_devfn(PCI_DEVFN(0x17, 0));
	if (func0_pdev == pdev)
		has_mii = true;

out:
	pci_dev_put(func0_pdev);
	return has_mii;
}

/**
 * ixgbe_mii_bus_init - mii_bus structure setup
 * @hw: pointer to hardware structure
 *
 * Returns 0 on success, negative on failure
 *
 * ixgbe_mii_bus_init initializes a mii_bus structure in adapter
 **/
int ixgbe_mii_bus_init(struct ixgbe_hw *hw)
{
	int (*write_c22)(struct mii_bus *bus, int addr, int regnum, u16 val);
	int (*read_c22)(struct mii_bus *bus, int addr, int regnum);
	int (*write_c45)(struct mii_bus *bus, int addr, int devad, int regnum,
			 u16 val);
	int (*read_c45)(struct mii_bus *bus, int addr, int devad, int regnum);
	struct ixgbe_adapter *adapter = hw->back;
	struct pci_dev *pdev = adapter->pdev;
	struct device *dev = &adapter->netdev->dev;
	struct mii_bus *bus;

	switch (hw->device_id) {
	/* C3000 SoCs */
	case IXGBE_DEV_ID_X550EM_A_KR:
	case IXGBE_DEV_ID_X550EM_A_KR_L:
	case IXGBE_DEV_ID_X550EM_A_SFP_N:
	case IXGBE_DEV_ID_X550EM_A_SGMII:
	case IXGBE_DEV_ID_X550EM_A_SGMII_L:
	case IXGBE_DEV_ID_X550EM_A_10G_T:
	case IXGBE_DEV_ID_X550EM_A_SFP:
	case IXGBE_DEV_ID_X550EM_A_1G_T:
	case IXGBE_DEV_ID_X550EM_A_1G_T_L:
		if (!ixgbe_x550em_a_has_mii(hw))
			return 0;
		read_c22 = ixgbe_x550em_a_mii_bus_read_c22;
		write_c22 = ixgbe_x550em_a_mii_bus_write_c22;
		read_c45 = ixgbe_x550em_a_mii_bus_read_c45;
		write_c45 = ixgbe_x550em_a_mii_bus_write_c45;
		break;
	default:
		read_c22 = ixgbe_mii_bus_read_c22;
		write_c22 = ixgbe_mii_bus_write_c22;
		read_c45 = ixgbe_mii_bus_read_c45;
		write_c45 = ixgbe_mii_bus_write_c45;
		break;
	}

	bus = devm_mdiobus_alloc(dev);
	if (!bus)
		return -ENOMEM;

	bus->read = read_c22;
	bus->write = write_c22;
	bus->read_c45 = read_c45;
	bus->write_c45 = write_c45;

	/* Use the position of the device in the PCI hierarchy as the id */
	snprintf(bus->id, MII_BUS_ID_SIZE, "%s-mdio-%s", ixgbe_driver_name,
		 pci_name(pdev));

	bus->name = "ixgbe-mdio";
	bus->priv = adapter;
	bus->parent = dev;
	bus->phy_mask = GENMASK(31, 0);

	/* Support clause 22/45 natively.  ixgbe_probe() sets MDIO_EMULATE_C22
	 * unfortunately that causes some clause 22 frames to be sent with
	 * clause 45 addressing.  We don't want that.
	 */
	hw->phy.mdio.mode_support = MDIO_SUPPORTS_C45 | MDIO_SUPPORTS_C22;

	adapter->mii_bus = bus;
	return mdiobus_register(bus);
}

/**
 *  ixgbe_setup_phy_link_generic - Set and restart autoneg
 *  @hw: pointer to hardware structure
 *
 *  Restart autonegotiation and PHY and waits for completion.
 **/
int ixgbe_setup_phy_link_generic(struct ixgbe_hw *hw)
{
	u16 autoneg_reg = IXGBE_MII_AUTONEG_REG;
	ixgbe_link_speed speed;
	bool autoneg = false;
	int status = 0;

	ixgbe_get_copper_link_capabilities_generic(hw, &speed, &autoneg);

	/* Set or unset auto-negotiation 10G advertisement */
	hw->phy.ops.read_reg(hw, MDIO_AN_10GBT_CTRL, MDIO_MMD_AN, &autoneg_reg);

	autoneg_reg &= ~MDIO_AN_10GBT_CTRL_ADV10G;
	if ((hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_10GB_FULL) &&
	    (speed & IXGBE_LINK_SPEED_10GB_FULL))
		autoneg_reg |= MDIO_AN_10GBT_CTRL_ADV10G;

	hw->phy.ops.write_reg(hw, MDIO_AN_10GBT_CTRL, MDIO_MMD_AN, autoneg_reg);

	hw->phy.ops.read_reg(hw, IXGBE_MII_AUTONEG_VENDOR_PROVISION_1_REG,
			     MDIO_MMD_AN, &autoneg_reg);

	if (hw->mac.type == ixgbe_mac_X550 || hw->mac.type == ixgbe_mac_e610) {
		/* Set or unset auto-negotiation 5G advertisement */
		autoneg_reg &= ~IXGBE_MII_5GBASE_T_ADVERTISE;
		if ((hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_5GB_FULL) &&
		    (speed & IXGBE_LINK_SPEED_5GB_FULL))
			autoneg_reg |= IXGBE_MII_5GBASE_T_ADVERTISE;

		/* Set or unset auto-negotiation 2.5G advertisement */
		autoneg_reg &= ~IXGBE_MII_2_5GBASE_T_ADVERTISE;
		if ((hw->phy.autoneg_advertised &
		     IXGBE_LINK_SPEED_2_5GB_FULL) &&
		    (speed & IXGBE_LINK_SPEED_2_5GB_FULL))
			autoneg_reg |= IXGBE_MII_2_5GBASE_T_ADVERTISE;
	}

	/* Set or unset auto-negotiation 1G advertisement */
	autoneg_reg &= ~IXGBE_MII_1GBASE_T_ADVERTISE;
	if ((hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_1GB_FULL) &&
	    (speed & IXGBE_LINK_SPEED_1GB_FULL))
		autoneg_reg |= IXGBE_MII_1GBASE_T_ADVERTISE;

	hw->phy.ops.write_reg(hw, IXGBE_MII_AUTONEG_VENDOR_PROVISION_1_REG,
			      MDIO_MMD_AN, autoneg_reg);

	/* Set or unset auto-negotiation 100M advertisement */
	hw->phy.ops.read_reg(hw, MDIO_AN_ADVERTISE, MDIO_MMD_AN, &autoneg_reg);

	autoneg_reg &= ~(ADVERTISE_100FULL | ADVERTISE_100HALF);
	if ((hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_100_FULL) &&
	    (speed & IXGBE_LINK_SPEED_100_FULL))
		autoneg_reg |= ADVERTISE_100FULL;

	hw->phy.ops.write_reg(hw, MDIO_AN_ADVERTISE, MDIO_MMD_AN, autoneg_reg);

	/* Blocked by MNG FW so don't reset PHY */
	if (ixgbe_check_reset_blocked(hw))
		return 0;

	/* Restart PHY autonegotiation and wait for completion */
	hw->phy.ops.read_reg(hw, MDIO_CTRL1,
			     MDIO_MMD_AN, &autoneg_reg);

	autoneg_reg |= MDIO_AN_CTRL1_RESTART;

	hw->phy.ops.write_reg(hw, MDIO_CTRL1,
			      MDIO_MMD_AN, autoneg_reg);

	return status;
}

/**
 *  ixgbe_setup_phy_link_speed_generic - Sets the auto advertised capabilities
 *  @hw: pointer to hardware structure
 *  @speed: new link speed
 *  @autoneg_wait_to_complete: unused
 **/
int ixgbe_setup_phy_link_speed_generic(struct ixgbe_hw *hw,
				       ixgbe_link_speed speed,
				       bool autoneg_wait_to_complete)
{
	/* Clear autoneg_advertised and set new values based on input link
	 * speed.
	 */
	hw->phy.autoneg_advertised = 0;

	if (speed & IXGBE_LINK_SPEED_10GB_FULL)
		hw->phy.autoneg_advertised |= IXGBE_LINK_SPEED_10GB_FULL;

	if (speed & IXGBE_LINK_SPEED_5GB_FULL)
		hw->phy.autoneg_advertised |= IXGBE_LINK_SPEED_5GB_FULL;

	if (speed & IXGBE_LINK_SPEED_2_5GB_FULL)
		hw->phy.autoneg_advertised |= IXGBE_LINK_SPEED_2_5GB_FULL;

	if (speed & IXGBE_LINK_SPEED_1GB_FULL)
		hw->phy.autoneg_advertised |= IXGBE_LINK_SPEED_1GB_FULL;

	if (speed & IXGBE_LINK_SPEED_100_FULL)
		hw->phy.autoneg_advertised |= IXGBE_LINK_SPEED_100_FULL;

	if (speed & IXGBE_LINK_SPEED_10_FULL)
		hw->phy.autoneg_advertised |= IXGBE_LINK_SPEED_10_FULL;

	/* Setup link based on the new speed settings */
	if (hw->phy.ops.setup_link)
		hw->phy.ops.setup_link(hw);

	return 0;
}

/**
 * ixgbe_get_copper_speeds_supported - Get copper link speed from phy
 * @hw: pointer to hardware structure
 *
 * Determines the supported link capabilities by reading the PHY auto
 * negotiation register.
 */
static int ixgbe_get_copper_speeds_supported(struct ixgbe_hw *hw)
{
	u16 speed_ability;
	int status;

	status = hw->phy.ops.read_reg(hw, MDIO_SPEED, MDIO_MMD_PMAPMD,
				      &speed_ability);
	if (status)
		return status;

	if (speed_ability & MDIO_SPEED_10G)
		hw->phy.speeds_supported |= IXGBE_LINK_SPEED_10GB_FULL;
	if (speed_ability & MDIO_PMA_SPEED_1000)
		hw->phy.speeds_supported |= IXGBE_LINK_SPEED_1GB_FULL;
	if (speed_ability & MDIO_PMA_SPEED_100)
		hw->phy.speeds_supported |= IXGBE_LINK_SPEED_100_FULL;

	switch (hw->mac.type) {
	case ixgbe_mac_X550:
	case ixgbe_mac_e610:
		hw->phy.speeds_supported |= IXGBE_LINK_SPEED_2_5GB_FULL;
		hw->phy.speeds_supported |= IXGBE_LINK_SPEED_5GB_FULL;
		break;
	case ixgbe_mac_X550EM_x:
	case ixgbe_mac_x550em_a:
		hw->phy.speeds_supported &= ~IXGBE_LINK_SPEED_100_FULL;
		break;
	default:
		break;
	}

	return 0;
}

/**
 * ixgbe_get_copper_link_capabilities_generic - Determines link capabilities
 * @hw: pointer to hardware structure
 * @speed: pointer to link speed
 * @autoneg: boolean auto-negotiation value
 */
int ixgbe_get_copper_link_capabilities_generic(struct ixgbe_hw *hw,
					       ixgbe_link_speed *speed,
					       bool *autoneg)
{
	int status = 0;

	*autoneg = true;
	if (!hw->phy.speeds_supported)
		status = ixgbe_get_copper_speeds_supported(hw);

	*speed = hw->phy.speeds_supported;
	return status;
}

/**
 *  ixgbe_check_phy_link_tnx - Determine link and speed status
 *  @hw: pointer to hardware structure
 *  @speed: link speed
 *  @link_up: status of link
 *
 *  Reads the VS1 register to determine if link is up and the current speed for
 *  the PHY.
 **/
int ixgbe_check_phy_link_tnx(struct ixgbe_hw *hw, ixgbe_link_speed *speed,
			     bool *link_up)
{
	u32 max_time_out = 10;
	u16 phy_speed = 0;
	u16 phy_link = 0;
	u16 phy_data = 0;
	u32 time_out;
	int status;

	/* Initialize speed and link to default case */
	*link_up = false;
	*speed = IXGBE_LINK_SPEED_10GB_FULL;

	/*
	 * Check current speed and link status of the PHY register.
	 * This is a vendor specific register and may have to
	 * be changed for other copper PHYs.
	 */
	for (time_out = 0; time_out < max_time_out; time_out++) {
		udelay(10);
		status = hw->phy.ops.read_reg(hw,
					      MDIO_STAT1,
					      MDIO_MMD_VEND1,
					      &phy_data);
		phy_link = phy_data &
			    IXGBE_MDIO_VENDOR_SPECIFIC_1_LINK_STATUS;
		phy_speed = phy_data &
			    IXGBE_MDIO_VENDOR_SPECIFIC_1_SPEED_STATUS;
		if (phy_link == IXGBE_MDIO_VENDOR_SPECIFIC_1_LINK_STATUS) {
			*link_up = true;
			if (phy_speed ==
			    IXGBE_MDIO_VENDOR_SPECIFIC_1_SPEED_STATUS)
				*speed = IXGBE_LINK_SPEED_1GB_FULL;
			break;
		}
	}

	return status;
}

/**
 *	ixgbe_setup_phy_link_tnx - Set and restart autoneg
 *	@hw: pointer to hardware structure
 *
 *	Restart autonegotiation and PHY and waits for completion.
 *	This function always returns success, this is necessary since
 *	it is called via a function pointer that could call other
 *	functions that could return an error.
 **/
int ixgbe_setup_phy_link_tnx(struct ixgbe_hw *hw)
{
	u16 autoneg_reg = IXGBE_MII_AUTONEG_REG;
	bool autoneg = false;
	ixgbe_link_speed speed;

	ixgbe_get_copper_link_capabilities_generic(hw, &speed, &autoneg);

	if (speed & IXGBE_LINK_SPEED_10GB_FULL) {
		/* Set or unset auto-negotiation 10G advertisement */
		hw->phy.ops.read_reg(hw, MDIO_AN_10GBT_CTRL,
				     MDIO_MMD_AN,
				     &autoneg_reg);

		autoneg_reg &= ~MDIO_AN_10GBT_CTRL_ADV10G;
		if (hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_10GB_FULL)
			autoneg_reg |= MDIO_AN_10GBT_CTRL_ADV10G;

		hw->phy.ops.write_reg(hw, MDIO_AN_10GBT_CTRL,
				      MDIO_MMD_AN,
				      autoneg_reg);
	}

	if (speed & IXGBE_LINK_SPEED_1GB_FULL) {
		/* Set or unset auto-negotiation 1G advertisement */
		hw->phy.ops.read_reg(hw, IXGBE_MII_AUTONEG_XNP_TX_REG,
				     MDIO_MMD_AN,
				     &autoneg_reg);

		autoneg_reg &= ~IXGBE_MII_1GBASE_T_ADVERTISE_XNP_TX;
		if (hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_1GB_FULL)
			autoneg_reg |= IXGBE_MII_1GBASE_T_ADVERTISE_XNP_TX;

		hw->phy.ops.write_reg(hw, IXGBE_MII_AUTONEG_XNP_TX_REG,
				      MDIO_MMD_AN,
				      autoneg_reg);
	}

	if (speed & IXGBE_LINK_SPEED_100_FULL) {
		/* Set or unset auto-negotiation 100M advertisement */
		hw->phy.ops.read_reg(hw, MDIO_AN_ADVERTISE,
				     MDIO_MMD_AN,
				     &autoneg_reg);

		autoneg_reg &= ~(ADVERTISE_100FULL |
				 ADVERTISE_100HALF);
		if (hw->phy.autoneg_advertised & IXGBE_LINK_SPEED_100_FULL)
			autoneg_reg |= ADVERTISE_100FULL;

		hw->phy.ops.write_reg(hw, MDIO_AN_ADVERTISE,
				      MDIO_MMD_AN,
				      autoneg_reg);
	}

	/* Blocked by MNG FW so don't reset PHY */
	if (ixgbe_check_reset_blocked(hw))
		return 0;

	/* Restart PHY autonegotiation and wait for completion */
	hw->phy.ops.read_reg(hw, MDIO_CTRL1,
			     MDIO_MMD_AN, &autoneg_reg);

	autoneg_reg |= MDIO_AN_CTRL1_RESTART;

	hw->phy.ops.write_reg(hw, MDIO_CTRL1,
			      MDIO_MMD_AN, autoneg_reg);
	return 0;
}

/**
 *  ixgbe_reset_phy_nl - Performs a PHY reset
 *  @hw: pointer to hardware structure
 **/
int ixgbe_reset_phy_nl(struct ixgbe_hw *hw)
{
	u16 phy_offset, control, eword, edata, block_crc;
	u16 list_offset, data_offset;
	bool end_data = false;
	u16 phy_data = 0;
	int ret_val;
	u32 i;

	/* Blocked by MNG FW so bail */
	if (ixgbe_check_reset_blocked(hw))
		return 0;

	hw->phy.ops.read_reg(hw, MDIO_CTRL1, MDIO_MMD_PHYXS, &phy_data);

	/* reset the PHY and poll for completion */
	hw->phy.ops.write_reg(hw, MDIO_CTRL1, MDIO_MMD_PHYXS,
			      (phy_data | MDIO_CTRL1_RESET));

	for (i = 0; i < 100; i++) {
		hw->phy.ops.read_reg(hw, MDIO_CTRL1, MDIO_MMD_PHYXS,
				     &phy_data);
		if ((phy_data & MDIO_CTRL1_RESET) == 0)
			break;
		usleep_range(10000, 20000);
	}

	if ((phy_data & MDIO_CTRL1_RESET) != 0) {
		hw_dbg(hw, "PHY reset did not complete.\n");
		return -EIO;
	}

	/* Get init offsets */
	ret_val = ixgbe_get_sfp_init_sequence_offsets(hw, &list_offset,
						      &data_offset);
	if (ret_val)
		return ret_val;

	ret_val = hw->eeprom.ops.read(hw, data_offset, &block_crc);
	data_offset++;
	while (!end_data) {
		/*
		 * Read control word from PHY init contents offset
		 */
		ret_val = hw->eeprom.ops.read(hw, data_offset, &eword);
		if (ret_val)
			goto err_eeprom;
		control = FIELD_GET(IXGBE_CONTROL_MASK_NL, eword);
		edata = eword & IXGBE_DATA_MASK_NL;
		switch (control) {
		case IXGBE_DELAY_NL:
			data_offset++;
			hw_dbg(hw, "DELAY: %d MS\n", edata);
			usleep_range(edata * 1000, edata * 2000);
			break;
		case IXGBE_DATA_NL:
			hw_dbg(hw, "DATA:\n");
			data_offset++;
			ret_val = hw->eeprom.ops.read(hw, data_offset++,
						      &phy_offset);
			if (ret_val)
				goto err_eeprom;
			for (i = 0; i < edata; i++) {
				ret_val = hw->eeprom.ops.read(hw, data_offset,
							      &eword);
				if (ret_val)
					goto err_eeprom;
				hw->phy.ops.write_reg(hw, phy_offset,
						      MDIO_MMD_PMAPMD, eword);
				hw_dbg(hw, "Wrote %4.4x to %4.4x\n", eword,
				       phy_offset);
				data_offset++;
				phy_offset++;
			}
			break;
		case IXGBE_CONTROL_NL:
			data_offset++;
			hw_dbg(hw, "CONTROL:\n");
			if (edata == IXGBE_CONTROL_EOL_NL) {
				hw_dbg(hw, "EOL\n");
				end_data = true;
			} else if (edata == IXGBE_CONTROL_SOL_NL) {
				hw_dbg(hw, "SOL\n");
			} else {
				hw_dbg(hw, "Bad control value\n");
				return -EIO;
			}
			break;
		default:
			hw_dbg(hw, "Bad control type\n");
			return -EIO;
		}
	}

	return ret_val;

err_eeprom:
	hw_err(hw, "eeprom read at offset %d failed\n", data_offset);
	return -EIO;
}

/**
 *  ixgbe_identify_module_generic - Identifies module type
 *  @hw: pointer to hardware structure
 *
 *  Determines HW type and calls appropriate function.
 **/
int ixgbe_identify_module_generic(struct ixgbe_hw *hw)
{
	switch (hw->mac.ops.get_media_type(hw)) {
	case ixgbe_media_type_fiber:
		return ixgbe_identify_sfp_module_generic(hw);
	case ixgbe_media_type_fiber_qsfp:
		return ixgbe_identify_qsfp_module_generic(hw);
	default:
		hw->phy.sfp_type = ixgbe_sfp_type_not_present;
		return -ENOENT;
	}

	return -ENOENT;
}

/**
 *  ixgbe_identify_sfp_module_generic - Identifies SFP modules
 *  @hw: pointer to hardware structure
 *
 *  Searches for and identifies the SFP module and assigns appropriate PHY type.
 **/
int ixgbe_identify_sfp_module_generic(struct ixgbe_hw *hw)
{
	enum ixgbe_sfp_type stored_sfp_type = hw->phy.sfp_type;
	struct ixgbe_adapter *adapter = hw->back;
	u8 oui_bytes[3] = {0, 0, 0};
	u8 bitrate_nominal = 0;
	u8 comp_codes_10g = 0;
	u8 comp_codes_1g = 0;
	u16 enforce_sfp = 0;
	u32 vendor_oui = 0;
	u8 identifier = 0;
	u8 cable_tech = 0;
	u8 cable_spec = 0;
	int status;

	if (hw->mac.ops.get_media_type(hw) != ixgbe_media_type_fiber) {
		hw->phy.sfp_type = ixgbe_sfp_type_not_present;
		return -ENOENT;
	}

	/* LAN ID is needed for sfp_type determination */
	hw->mac.ops.set_lan_id(hw);

	status = hw->phy.ops.read_i2c_eeprom(hw,
					     IXGBE_SFF_IDENTIFIER,
					     &identifier);

	if (status)
		goto err_read_i2c_eeprom;

	if (identifier != IXGBE_SFF_IDENTIFIER_SFP) {
		hw->phy.type = ixgbe_phy_sfp_unsupported;
		return -EOPNOTSUPP;
	}
	status = hw->phy.ops.read_i2c_eeprom(hw,
					     IXGBE_SFF_1GBE_COMP_CODES,
					     &comp_codes_1g);

	if (status)
		goto err_read_i2c_eeprom;

	status = hw->phy.ops.read_i2c_eeprom(hw,
					     IXGBE_SFF_10GBE_COMP_CODES,
					     &comp_codes_10g);

	if (status)
		goto err_read_i2c_eeprom;
	status = hw->phy.ops.read_i2c_eeprom(hw,
					     IXGBE_SFF_CABLE_TECHNOLOGY,
					     &cable_tech);
	if (status)
		goto err_read_i2c_eeprom;

	status = hw->phy.ops.read_i2c_eeprom(hw,
					     IXGBE_SFF_BITRATE_NOMINAL,
					     &bitrate_nominal);
	if (status)
		goto err_read_i2c_eeprom;

	 /* ID Module
	  * =========
	  * 0   SFP_DA_CU
	  * 1   SFP_SR
	  * 2   SFP_LR
	  * 3   SFP_DA_CORE0 - 82599-specific
	  * 4   SFP_DA_CORE1 - 82599-specific
	  * 5   SFP_SR/LR_CORE0 - 82599-specific
	  * 6   SFP_SR/LR_CORE1 - 82599-specific
	  * 7   SFP_act_lmt_DA_CORE0 - 82599-specific
	  * 8   SFP_act_lmt_DA_CORE1 - 82599-specific
	  * 9   SFP_1g_cu_CORE0 - 82599-specific
	  * 10  SFP_1g_cu_CORE1 - 82599-specific
	  * 11  SFP_1g_sx_CORE0 - 82599-specific
	  * 12  SFP_1g_sx_CORE1 - 82599-specific
	  */
	if (hw->mac.type == ixgbe_mac_82598EB) {
		if (cable_tech & IXGBE_SFF_DA_PASSIVE_CABLE)
			hw->phy.sfp_type = ixgbe_sfp_type_da_cu;
		else if (comp_codes_10g & IXGBE_SFF_10GBASESR_CAPABLE)
			hw->phy.sfp_type = ixgbe_sfp_type_sr;
		else if (comp_codes_10g & IXGBE_SFF_10GBASELR_CAPABLE)
			hw->phy.sfp_type = ixgbe_sfp_type_lr;
		else
			hw->phy.sfp_type = ixgbe_sfp_type_unknown;
	} else {
		if (cable_tech & IXGBE_SFF_DA_PASSIVE_CABLE) {
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
					     ixgbe_sfp_type_da_cu_core0;
			else
				hw->phy.sfp_type =
					     ixgbe_sfp_type_da_cu_core1;
		} else if (cable_tech & IXGBE_SFF_DA_ACTIVE_CABLE) {
			hw->phy.ops.read_i2c_eeprom(
					hw, IXGBE_SFF_CABLE_SPEC_COMP,
					&cable_spec);
			if (cable_spec &
			    IXGBE_SFF_DA_SPEC_ACTIVE_LIMITING) {
				if (hw->bus.lan_id == 0)
					hw->phy.sfp_type =
					ixgbe_sfp_type_da_act_lmt_core0;
				else
					hw->phy.sfp_type =
					ixgbe_sfp_type_da_act_lmt_core1;
			} else {
				hw->phy.sfp_type =
						ixgbe_sfp_type_unknown;
			}
		} else if (comp_codes_10g &
			   (IXGBE_SFF_10GBASESR_CAPABLE |
			    IXGBE_SFF_10GBASELR_CAPABLE)) {
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
					      ixgbe_sfp_type_srlr_core0;
			else
				hw->phy.sfp_type =
					      ixgbe_sfp_type_srlr_core1;
		} else if (comp_codes_1g & IXGBE_SFF_1GBASET_CAPABLE) {
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_cu_core0;
			else
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_cu_core1;
		} else if (comp_codes_1g & IXGBE_SFF_1GBASESX_CAPABLE) {
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_sx_core0;
			else
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_sx_core1;
		} else if (comp_codes_1g & IXGBE_SFF_1GBASELX_CAPABLE) {
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_lx_core0;
			else
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_lx_core1;
		/* Support only Ethernet 1000BASE-BX10, checking the Bit Rate
		 * Nominal Value as per SFF-8472 by convention 1.25 Gb/s should
		 * be rounded up to 0Dh (13 in units of 100 MBd) for 1000BASE-BX
		 */
		} else if ((comp_codes_1g & IXGBE_SFF_BASEBX10_CAPABLE) &&
			   (bitrate_nominal == 0xD)) {
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_bx_core0;
			else
				hw->phy.sfp_type =
					ixgbe_sfp_type_1g_bx_core1;
		} else {
			hw->phy.sfp_type = ixgbe_sfp_type_unknown;
		}
	}

	if (hw->phy.sfp_type != stored_sfp_type)
		hw->phy.sfp_setup_needed = true;

	/* Determine if the SFP+ PHY is dual speed or not. */
	hw->phy.multispeed_fiber = false;
	if (((comp_codes_1g & IXGBE_SFF_1GBASESX_CAPABLE) &&
	     (comp_codes_10g & IXGBE_SFF_10GBASESR_CAPABLE)) ||
	    ((comp_codes_1g & IXGBE_SFF_1GBASELX_CAPABLE) &&
	     (comp_codes_10g & IXGBE_SFF_10GBASELR_CAPABLE)))
		hw->phy.multispeed_fiber = true;

	/* Determine PHY vendor */
	if (hw->phy.type != ixgbe_phy_nl) {
		hw->phy.id = identifier;
		status = hw->phy.ops.read_i2c_eeprom(hw,
					    IXGBE_SFF_VENDOR_OUI_BYTE0,
					    &oui_bytes[0]);

		if (status != 0)
			goto err_read_i2c_eeprom;

		status = hw->phy.ops.read_i2c_eeprom(hw,
					    IXGBE_SFF_VENDOR_OUI_BYTE1,
					    &oui_bytes[1]);

		if (status != 0)
			goto err_read_i2c_eeprom;

		status = hw->phy.ops.read_i2c_eeprom(hw,
					    IXGBE_SFF_VENDOR_OUI_BYTE2,
					    &oui_bytes[2]);

		if (status != 0)
			goto err_read_i2c_eeprom;

		vendor_oui =
		  ((oui_bytes[0] << IXGBE_SFF_VENDOR_OUI_BYTE0_SHIFT) |
		   (oui_bytes[1] << IXGBE_SFF_VENDOR_OUI_BYTE1_SHIFT) |
		   (oui_bytes[2] << IXGBE_SFF_VENDOR_OUI_BYTE2_SHIFT));

		switch (vendor_oui) {
		case IXGBE_SFF_VENDOR_OUI_TYCO:
			if (cable_tech & IXGBE_SFF_DA_PASSIVE_CABLE)
				hw->phy.type =
					    ixgbe_phy_sfp_passive_tyco;
			break;
		case IXGBE_SFF_VENDOR_OUI_FTL:
			if (cable_tech & IXGBE_SFF_DA_ACTIVE_CABLE)
				hw->phy.type = ixgbe_phy_sfp_ftl_active;
			else
				hw->phy.type = ixgbe_phy_sfp_ftl;
			break;
		case IXGBE_SFF_VENDOR_OUI_AVAGO:
			hw->phy.type = ixgbe_phy_sfp_avago;
			break;
		case IXGBE_SFF_VENDOR_OUI_INTEL:
			hw->phy.type = ixgbe_phy_sfp_intel;
			break;
		default:
			if (cable_tech & IXGBE_SFF_DA_PASSIVE_CABLE)
				hw->phy.type =
					 ixgbe_phy_sfp_passive_unknown;
			else if (cable_tech & IXGBE_SFF_DA_ACTIVE_CABLE)
				hw->phy.type =
					ixgbe_phy_sfp_active_unknown;
			else
				hw->phy.type = ixgbe_phy_sfp_unknown;
			break;
		}
	}

	/* Allow any DA cable vendor */
	if (cable_tech & (IXGBE_SFF_DA_PASSIVE_CABLE |
	    IXGBE_SFF_DA_ACTIVE_CABLE))
		return 0;

	/* Verify supported 1G SFP modules */
	if (comp_codes_10g == 0 &&
	    !(hw->phy.sfp_type == ixgbe_sfp_type_1g_cu_core1 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_cu_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_lx_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_lx_core1 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_sx_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_sx_core1 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_bx_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_bx_core1)) {
		hw->phy.type = ixgbe_phy_sfp_unsupported;
		return -EOPNOTSUPP;
	}

	/* Anything else 82598-based is supported */
	if (hw->mac.type == ixgbe_mac_82598EB)
		return 0;

	hw->mac.ops.get_device_caps(hw, &enforce_sfp);
	if (!(enforce_sfp & IXGBE_DEVICE_CAPS_ALLOW_ANY_SFP) &&
	    !(hw->phy.sfp_type == ixgbe_sfp_type_1g_cu_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_cu_core1 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_lx_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_lx_core1 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_sx_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_sx_core1 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_bx_core0 ||
	      hw->phy.sfp_type == ixgbe_sfp_type_1g_bx_core1)) {
		/* Make sure we're a supported PHY type */
		if (hw->phy.type == ixgbe_phy_sfp_intel)
			return 0;
		if (hw->allow_unsupported_sfp) {
			e_warn(drv, "WARNING: Intel (R) Network Connections are quality tested using Intel (R) Ethernet Optics.  Using untested modules is not supported and may cause unstable operation or damage to the module or the adapter.  Intel Corporation is not responsible for any harm caused by using untested modules.\n");
			return 0;
		}
		hw_dbg(hw, "SFP+ module not supported\n");
		hw->phy.type = ixgbe_phy_sfp_unsupported;
		return -EOPNOTSUPP;
	}
	return 0;

err_read_i2c_eeprom:
	hw->phy.sfp_type = ixgbe_sfp_type_not_present;
	if (hw->phy.type != ixgbe_phy_nl) {
		hw->phy.id = 0;
		hw->phy.type = ixgbe_phy_unknown;
	}
	return -ENOENT;
}

/**
 * ixgbe_identify_qsfp_module_generic - Identifies QSFP modules
 * @hw: pointer to hardware structure
 *
 * Searches for and identifies the QSFP module and assigns appropriate PHY type
 **/
static int ixgbe_identify_qsfp_module_generic(struct ixgbe_hw *hw)
{
	struct ixgbe_adapter *adapter = hw->back;
	int status;
	u32 vendor_oui = 0;
	enum ixgbe_sfp_type stored_sfp_type = hw->phy.sfp_type;
	u8 identifier = 0;
	u8 comp_codes_1g = 0;
	u8 comp_codes_10g = 0;
	u8 oui_bytes[3] = {0, 0, 0};
	u16 enforce_sfp = 0;
	u8 connector = 0;
	u8 cable_length = 0;
	u8 device_tech = 0;
	bool active_cable = false;

	if (hw->mac.ops.get_media_type(hw) != ixgbe_media_type_fiber_qsfp) {
		hw->phy.sfp_type = ixgbe_sfp_type_not_present;
		return -ENOENT;
	}

	/* LAN ID is needed for sfp_type determination */
	hw->mac.ops.set_lan_id(hw);

	status = hw->phy.ops.read_i2c_eeprom(hw, IXGBE_SFF_IDENTIFIER,
					     &identifier);

	if (status != 0)
		goto err_read_i2c_eeprom;

	if (identifier != IXGBE_SFF_IDENTIFIER_QSFP_PLUS) {
		hw->phy.type = ixgbe_phy_sfp_unsupported;
		return -EOPNOTSUPP;
	}

	hw->phy.id = identifier;

	status = hw->phy.ops.read_i2c_eeprom(hw, IXGBE_SFF_QSFP_10GBE_COMP,
					     &comp_codes_10g);

	if (status != 0)
		goto err_read_i2c_eeprom;

	status = hw->phy.ops.read_i2c_eeprom(hw, IXGBE_SFF_QSFP_1GBE_COMP,
					     &comp_codes_1g);

	if (status != 0)
		goto err_read_i2c_eeprom;

	if (comp_codes_10g & IXGBE_SFF_QSFP_DA_PASSIVE_CABLE) {
		hw->phy.type = ixgbe_phy_qsfp_passive_unknown;
		if (hw->bus.lan_id == 0)
			hw->phy.sfp_type = ixgbe_sfp_type_da_cu_core0;
		else
			hw->phy.sfp_type = ixgbe_sfp_type_da_cu_core1;
	} else if (comp_codes_10g & (IXGBE_SFF_10GBASESR_CAPABLE |
				     IXGBE_SFF_10GBASELR_CAPABLE)) {
		if (hw->bus.lan_id == 0)
			hw->phy.sfp_type = ixgbe_sfp_type_srlr_core0;
		else
			hw->phy.sfp_type = ixgbe_sfp_type_srlr_core1;
	} else {
		if (comp_codes_10g & IXGBE_SFF_QSFP_DA_ACTIVE_CABLE)
			active_cable = true;

		if (!active_cable) {
			/* check for active DA cables that pre-date
			 * SFF-8436 v3.6
			 */
			hw->phy.ops.read_i2c_eeprom(hw,
					IXGBE_SFF_QSFP_CONNECTOR,
					&connector);

			hw->phy.ops.read_i2c_eeprom(hw,
					IXGBE_SFF_QSFP_CABLE_LENGTH,
					&cable_length);

			hw->phy.ops.read_i2c_eeprom(hw,
					IXGBE_SFF_QSFP_DEVICE_TECH,
					&device_tech);

			if ((connector ==
				     IXGBE_SFF_QSFP_CONNECTOR_NOT_SEPARABLE) &&
			    (cable_length > 0) &&
			    ((device_tech >> 4) ==
				     IXGBE_SFF_QSFP_TRANSMITER_850NM_VCSEL))
				active_cable = true;
		}

		if (active_cable) {
			hw->phy.type = ixgbe_phy_qsfp_active_unknown;
			if (hw->bus.lan_id == 0)
				hw->phy.sfp_type =
						ixgbe_sfp_type_da_act_lmt_core0;
			else
				hw->phy.sfp_type =
						ixgbe_sfp_type_da_act_lmt_core1;
		} else {
			/* unsupported module type */
			hw->phy.type = ixgbe_phy_sfp_unsupported;
			return -EOPNOTSUPP;
		}
	}

	if (hw->phy.sfp_type != stored_sfp_type)
		hw->phy.sfp_setup_needed = true;

	/* Determine if the QSFP+ PHY is dual speed or not. */
	hw->phy.multispeed_fiber = false;
	if (((comp_codes_1g & IXGBE_SFF_1GBASESX_CAPABLE) &&
	     (comp_codes_10g & IXGBE_SFF_10GBASESR_CAPABLE)) ||
	    ((comp_codes_1g & IXGBE_SFF_1GBASELX_CAPABLE) &&
	     (comp_codes_10g & IXGBE_SFF_10GBASELR_CAPABLE)))
		hw->phy.multispeed_fiber = true;

	/* Determine PHY vendor for optical modules */
	if (comp_codes_10g & (IXGBE_SFF_10GBASESR_CAPABLE |
			      IXGBE_SFF_10GBASELR_CAPABLE)) {
		status = hw->phy.ops.read_i2c_eeprom(hw,
					IXGBE_SFF_QSFP_VENDOR_OUI_BYTE0,
					&oui_bytes[0]);

		if (status != 0)
			goto err_read_i2c_eeprom;

		status = hw->phy.ops.read_i2c_eeprom(hw,
					IXGBE_SFF_QSFP_VENDOR_OUI_BYTE1,
					&oui_bytes[1]);

		if (status != 0)
			goto err_read_i2c_eeprom;

		status = hw->phy.ops.read_i2c_eeprom(hw,
					IXGBE_SFF_QSFP_VENDOR_OUI_BYTE2,
					&oui_bytes[2]);

		if (status != 0)
			goto err_read_i2c_eeprom;

		vendor_oui =
			((oui_bytes[0] << IXGBE_SFF_VENDOR_OUI_BYTE0_SHIFT) |
			 (oui_bytes[1] << IXGBE_SFF_VENDOR_OUI_BYTE1_SHIFT) |
			 (oui_bytes[2] << IXGBE_SFF_VENDOR_OUI_BYTE2_SHIFT));

		if (vendor_oui == IXGBE_SFF_VENDOR_OUI_INTEL)
			hw->phy.type = ixgbe_phy_qsfp_intel;
		else
			hw->phy.type = ixgbe_phy_qsfp_unknown;

		hw->mac.ops.get_device_caps(hw, &enforce_sfp);
		if (!(enforce_sfp & IXGBE_DEVICE_CAPS_ALLOW_ANY_SFP)) {
			/* Make sure we're a supported PHY type */
			if (hw->phy.type == ixgbe_phy_qsfp_intel)
				return 0;
			if (hw->allow_unsupported_sfp) {
				e_warn(drv, "WARNING: Intel (R) Network Connections are quality tested using Intel (R) Ethernet Optics. Using untested modules is not supported and may cause unstable operation or damage to the module or the adapter. Intel Corporation is not responsible for any harm caused by using untested modules.\n");
				return 0;
			}
			hw_dbg(hw, "QSFP module not supported\n");
			hw->phy.type = ixgbe_phy_sfp_unsupported;
			return -EOPNOTSUPP;
		}
		return 0;
	}
	return 0;

err_read_i2c_eeprom:
	hw->phy.sfp_type = ixgbe_sfp_type_not_present;
	hw->phy.id = 0;
	hw->phy.type = ixgbe_phy_unknown;

	return -ENOENT;
}

/**
 *  ixgbe_get_sfp_init_sequence_offsets - Provides offset of PHY init sequence
 *  @hw: pointer to hardware structure
 *  @list_offset: offset to the SFP ID list
 *  @data_offset: offset to the SFP data block
 *
 *  Checks the MAC's EEPROM to see if it supports a given SFP+ module type, if
 *  so it returns the offsets to the phy init sequence block.
 **/
int ixgbe_get_sfp_init_sequence_offsets(struct ixgbe_hw *hw,
					u16 *list_offset,
					u16 *data_offset)
{
	u16 sfp_id;
	u16 sfp_type = hw->phy.sfp_type;

	if (hw->phy.sfp_type == ixgbe_sfp_type_unknown)
		return -EOPNOTSUPP;

	if (hw->phy.sfp_type == ixgbe_sfp_type_not_present)
		return -ENOENT;

	if ((hw->device_id == IXGBE_DEV_ID_82598_SR_DUAL_PORT_EM) &&
	    (hw->phy.sfp_type == ixgbe_sfp_type_da_cu))
		return -EOPNOTSUPP;

	/*
	 * Limiting active cables and 1G Phys must be initialized as
	 * SR modules
	 */
	if (sfp_type == ixgbe_sfp_type_da_act_lmt_core0 ||
	    sfp_type == ixgbe_sfp_type_1g_lx_core0 ||
	    sfp_type == ixgbe_sfp_type_1g_cu_core0 ||
	    sfp_type == ixgbe_sfp_type_1g_sx_core0 ||
	    sfp_type == ixgbe_sfp_type_1g_bx_core0)
		sfp_type = ixgbe_sfp_type_srlr_core0;
	else if (sfp_type == ixgbe_sfp_type_da_act_lmt_core1 ||
		 sfp_type == ixgbe_sfp_type_1g_lx_core1 ||
		 sfp_type == ixgbe_sfp_type_1g_cu_core1 ||
		 sfp_type == ixgbe_sfp_type_1g_sx_core1 ||
		 sfp_type == ixgbe_sfp_type_1g_bx_core1)
		sfp_type = ixgbe_sfp_type_srlr_core1;

	/* Read offset to PHY init contents */
	if (hw->eeprom.ops.read(hw, IXGBE_PHY_INIT_OFFSET_NL, list_offset)) {
		hw_err(hw, "eeprom read at %d failed\n",
		       IXGBE_PHY_INIT_OFFSET_NL);
		return -EIO;
	}

	if ((!*list_offset) || (*list_offset == 0xFFFF))
		return -EIO;

	/* Shift offset to first ID word */
	(*list_offset)++;

	/*
	 * Find the matching SFP ID in the EEPROM
	 * and program the init sequence
	 */
	if (hw->eeprom.ops.read(hw, *list_offset, &sfp_id))
		goto err_phy;

	while (sfp_id != IXGBE_PHY_INIT_END_NL) {
		if (sfp_id == sfp_type) {
			(*list_offset)++;
			if (hw->eeprom.ops.read(hw, *list_offset, data_offset))
				goto err_phy;
			if ((!*data_offset) || (*data_offset == 0xFFFF)) {
				hw_dbg(hw, "SFP+ module not supported\n");
				return -EOPNOTSUPP;
			} else {
				break;
			}
		} else {
			(*list_offset) += 2;
			if (hw->eeprom.ops.read(hw, *list_offset, &sfp_id))
				goto err_phy;
		}
	}

	if (sfp_id == IXGBE_PHY_INIT_END_NL) {
		hw_dbg(hw, "No matching SFP+ module found\n");
		return -EOPNOTSUPP;
	}

	return 0;

err_phy:
	hw_err(hw, "eeprom read at offset %d failed\n", *list_offset);
	return -EIO;
}

/**
 *  ixgbe_read_i2c_eeprom_generic - Reads 8 bit EEPROM word over I2C interface
 *  @hw: pointer to hardware structure
 *  @byte_offset: EEPROM byte offset to read
 *  @eeprom_data: value read
 *
 *  Performs byte read operation to SFP module's EEPROM over I2C interface.
 **/
int ixgbe_read_i2c_eeprom_generic(struct ixgbe_hw *hw, u8 byte_offset,
				  u8 *eeprom_data)
{
	return hw->phy.ops.read_i2c_byte(hw, byte_offset,
					 IXGBE_I2C_EEPROM_DEV_ADDR,
					 eeprom_data);
}

/**
 *  ixgbe_read_i2c_sff8472_generic - Reads 8 bit word over I2C interface
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset at address 0xA2
 *  @sff8472_data: value read
 *
 *  Performs byte read operation to SFP module's SFF-8472 data over I2C
 **/
int ixgbe_read_i2c_sff8472_generic(struct ixgbe_hw *hw, u8 byte_offset,
				   u8 *sff8472_data)
{
	return hw->phy.ops.read_i2c_byte(hw, byte_offset,
					 IXGBE_I2C_EEPROM_DEV_ADDR2,
					 sff8472_data);
}

/**
 *  ixgbe_write_i2c_eeprom_generic - Writes 8 bit EEPROM word over I2C interface
 *  @hw: pointer to hardware structure
 *  @byte_offset: EEPROM byte offset to write
 *  @eeprom_data: value to write
 *
 *  Performs byte write operation to SFP module's EEPROM over I2C interface.
 **/
int ixgbe_write_i2c_eeprom_generic(struct ixgbe_hw *hw, u8 byte_offset,
				   u8 eeprom_data)
{
	return hw->phy.ops.write_i2c_byte(hw, byte_offset,
					  IXGBE_I2C_EEPROM_DEV_ADDR,
					  eeprom_data);
}

/**
 * ixgbe_is_sfp_probe - Returns true if SFP is being detected
 * @hw: pointer to hardware structure
 * @offset: eeprom offset to be read
 * @addr: I2C address to be read
 */
static bool ixgbe_is_sfp_probe(struct ixgbe_hw *hw, u8 offset, u8 addr)
{
	if (addr == IXGBE_I2C_EEPROM_DEV_ADDR &&
	    offset == IXGBE_SFF_IDENTIFIER &&
	    hw->phy.sfp_type == ixgbe_sfp_type_not_present)
		return true;
	return false;
}

/**
 *  ixgbe_read_i2c_byte_generic_int - Reads 8 bit word over I2C
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset to read
 *  @dev_addr: device address
 *  @data: value read
 *  @lock: true if to take and release semaphore
 *
 *  Performs byte read operation to SFP module's EEPROM over I2C interface at
 *  a specified device address.
 */
static int ixgbe_read_i2c_byte_generic_int(struct ixgbe_hw *hw, u8 byte_offset,
					   u8 dev_addr, u8 *data, bool lock)
{
	u32 swfw_mask = hw->phy.phy_semaphore_mask;
	u32 max_retry = 10;
	bool nack = true;
	u32 retry = 0;
	int status;

	if (hw->mac.type >= ixgbe_mac_X550)
		max_retry = 3;
	if (ixgbe_is_sfp_probe(hw, byte_offset, dev_addr))
		max_retry = IXGBE_SFP_DETECT_RETRIES;

	*data = 0;

	do {
		if (lock && hw->mac.ops.acquire_swfw_sync(hw, swfw_mask))
			return -EBUSY;

		ixgbe_i2c_start(hw);

		/* Device Address and write indication */
		status = ixgbe_clock_out_i2c_byte(hw, dev_addr);
		if (status != 0)
			goto fail;

		status = ixgbe_get_i2c_ack(hw);
		if (status != 0)
			goto fail;

		status = ixgbe_clock_out_i2c_byte(hw, byte_offset);
		if (status != 0)
			goto fail;

		status = ixgbe_get_i2c_ack(hw);
		if (status != 0)
			goto fail;

		ixgbe_i2c_start(hw);

		/* Device Address and read indication */
		status = ixgbe_clock_out_i2c_byte(hw, (dev_addr | 0x1));
		if (status != 0)
			goto fail;

		status = ixgbe_get_i2c_ack(hw);
		if (status != 0)
			goto fail;

		status = ixgbe_clock_in_i2c_byte(hw, data);
		if (status != 0)
			goto fail;

		status = ixgbe_clock_out_i2c_bit(hw, nack);
		if (status != 0)
			goto fail;

		ixgbe_i2c_stop(hw);
		if (lock)
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
		return 0;

fail:
		ixgbe_i2c_bus_clear(hw);
		if (lock) {
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
			msleep(100);
		}
		retry++;
		if (retry < max_retry)
			hw_dbg(hw, "I2C byte read error - Retrying.\n");
		else
			hw_dbg(hw, "I2C byte read error.\n");

	} while (retry < max_retry);

	return status;
}

/**
 *  ixgbe_read_i2c_byte_generic - Reads 8 bit word over I2C
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset to read
 *  @dev_addr: device address
 *  @data: value read
 *
 *  Performs byte read operation to SFP module's EEPROM over I2C interface at
 *  a specified device address.
 */
int ixgbe_read_i2c_byte_generic(struct ixgbe_hw *hw, u8 byte_offset,
				u8 dev_addr, u8 *data)
{
	return ixgbe_read_i2c_byte_generic_int(hw, byte_offset, dev_addr,
					       data, true);
}

/**
 *  ixgbe_read_i2c_byte_generic_unlocked - Reads 8 bit word over I2C
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset to read
 *  @dev_addr: device address
 *  @data: value read
 *
 *  Performs byte read operation to SFP module's EEPROM over I2C interface at
 *  a specified device address.
 */
int ixgbe_read_i2c_byte_generic_unlocked(struct ixgbe_hw *hw, u8 byte_offset,
					 u8 dev_addr, u8 *data)
{
	return ixgbe_read_i2c_byte_generic_int(hw, byte_offset, dev_addr,
					       data, false);
}

/**
 *  ixgbe_write_i2c_byte_generic_int - Writes 8 bit word over I2C
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset to write
 *  @dev_addr: device address
 *  @data: value to write
 *  @lock: true if to take and release semaphore
 *
 *  Performs byte write operation to SFP module's EEPROM over I2C interface at
 *  a specified device address.
 */
static int ixgbe_write_i2c_byte_generic_int(struct ixgbe_hw *hw, u8 byte_offset,
					    u8 dev_addr, u8 data, bool lock)
{
	u32 swfw_mask = hw->phy.phy_semaphore_mask;
	u32 max_retry = 3;
	u32 retry = 0;
	int status;

	if (lock && hw->mac.ops.acquire_swfw_sync(hw, swfw_mask))
		return -EBUSY;

	do {
		ixgbe_i2c_start(hw);

		status = ixgbe_clock_out_i2c_byte(hw, dev_addr);
		if (status != 0)
			goto fail;

		status = ixgbe_get_i2c_ack(hw);
		if (status != 0)
			goto fail;

		status = ixgbe_clock_out_i2c_byte(hw, byte_offset);
		if (status != 0)
			goto fail;

		status = ixgbe_get_i2c_ack(hw);
		if (status != 0)
			goto fail;

		status = ixgbe_clock_out_i2c_byte(hw, data);
		if (status != 0)
			goto fail;

		status = ixgbe_get_i2c_ack(hw);
		if (status != 0)
			goto fail;

		ixgbe_i2c_stop(hw);
		if (lock)
			hw->mac.ops.release_swfw_sync(hw, swfw_mask);
		return 0;

fail:
		ixgbe_i2c_bus_clear(hw);
		retry++;
		if (retry < max_retry)
			hw_dbg(hw, "I2C byte write error - Retrying.\n");
		else
			hw_dbg(hw, "I2C byte write error.\n");
	} while (retry < max_retry);

	if (lock)
		hw->mac.ops.release_swfw_sync(hw, swfw_mask);

	return status;
}

/**
 *  ixgbe_write_i2c_byte_generic - Writes 8 bit word over I2C
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset to write
 *  @dev_addr: device address
 *  @data: value to write
 *
 *  Performs byte write operation to SFP module's EEPROM over I2C interface at
 *  a specified device address.
 */
int ixgbe_write_i2c_byte_generic(struct ixgbe_hw *hw, u8 byte_offset,
				 u8 dev_addr, u8 data)
{
	return ixgbe_write_i2c_byte_generic_int(hw, byte_offset, dev_addr,
						data, true);
}

/**
 *  ixgbe_write_i2c_byte_generic_unlocked - Writes 8 bit word over I2C
 *  @hw: pointer to hardware structure
 *  @byte_offset: byte offset to write
 *  @dev_addr: device address
 *  @data: value to write
 *
 *  Performs byte write operation to SFP module's EEPROM over I2C interface at
 *  a specified device address.
 */
int ixgbe_write_i2c_byte_generic_unlocked(struct ixgbe_hw *hw, u8 byte_offset,
					  u8 dev_addr, u8 data)
{
	return ixgbe_write_i2c_byte_generic_int(hw, byte_offset, dev_addr,
						data, false);
}

/**
 *  ixgbe_i2c_start - Sets I2C start condition
 *  @hw: pointer to hardware structure
 *
 *  Sets I2C start condition (High -> Low on SDA while SCL is High)
 *  Set bit-bang mode on X550 hardware.
 **/
static void ixgbe_i2c_start(struct ixgbe_hw *hw)
{
	u32 i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));

	i2cctl |= IXGBE_I2C_BB_EN(hw);

	/* Start condition must begin with data and clock high */
	ixgbe_set_i2c_data(hw, &i2cctl, 1);
	ixgbe_raise_i2c_clk(hw, &i2cctl);

	/* Setup time for start condition (4.7us) */
	udelay(IXGBE_I2C_T_SU_STA);

	ixgbe_set_i2c_data(hw, &i2cctl, 0);

	/* Hold time for start condition (4us) */
	udelay(IXGBE_I2C_T_HD_STA);

	ixgbe_lower_i2c_clk(hw, &i2cctl);

	/* Minimum low period of clock is 4.7 us */
	udelay(IXGBE_I2C_T_LOW);

}

/**
 *  ixgbe_i2c_stop - Sets I2C stop condition
 *  @hw: pointer to hardware structure
 *
 *  Sets I2C stop condition (Low -> High on SDA while SCL is High)
 *  Disables bit-bang mode and negates data output enable on X550
 *  hardware.
 **/
static void ixgbe_i2c_stop(struct ixgbe_hw *hw)
{
	u32 i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	u32 data_oe_bit = IXGBE_I2C_DATA_OE_N_EN(hw);
	u32 clk_oe_bit = IXGBE_I2C_CLK_OE_N_EN(hw);
	u32 bb_en_bit = IXGBE_I2C_BB_EN(hw);

	/* Stop condition must begin with data low and clock high */
	ixgbe_set_i2c_data(hw, &i2cctl, 0);
	ixgbe_raise_i2c_clk(hw, &i2cctl);

	/* Setup time for stop condition (4us) */
	udelay(IXGBE_I2C_T_SU_STO);

	ixgbe_set_i2c_data(hw, &i2cctl, 1);

	/* bus free time between stop and start (4.7us)*/
	udelay(IXGBE_I2C_T_BUF);

	if (bb_en_bit || data_oe_bit || clk_oe_bit) {
		i2cctl &= ~bb_en_bit;
		i2cctl |= data_oe_bit | clk_oe_bit;
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), i2cctl);
		IXGBE_WRITE_FLUSH(hw);
	}
}

/**
 *  ixgbe_clock_in_i2c_byte - Clocks in one byte via I2C
 *  @hw: pointer to hardware structure
 *  @data: data byte to clock in
 *
 *  Clocks in one byte data via I2C data/clock
 **/
static int ixgbe_clock_in_i2c_byte(struct ixgbe_hw *hw, u8 *data)
{
	bool bit = false;
	int i;

	*data = 0;
	for (i = 7; i >= 0; i--) {
		ixgbe_clock_in_i2c_bit(hw, &bit);
		*data |= bit << i;
	}

	return 0;
}

/**
 *  ixgbe_clock_out_i2c_byte - Clocks out one byte via I2C
 *  @hw: pointer to hardware structure
 *  @data: data byte clocked out
 *
 *  Clocks out one byte data via I2C data/clock
 **/
static int ixgbe_clock_out_i2c_byte(struct ixgbe_hw *hw, u8 data)
{
	bool bit = false;
	int status;
	u32 i2cctl;
	int i;

	for (i = 7; i >= 0; i--) {
		bit = (data >> i) & 0x1;
		status = ixgbe_clock_out_i2c_bit(hw, bit);

		if (status != 0)
			break;
	}

	/* Release SDA line (set high) */
	i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	i2cctl |= IXGBE_I2C_DATA_OUT(hw);
	i2cctl |= IXGBE_I2C_DATA_OE_N_EN(hw);
	IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), i2cctl);
	IXGBE_WRITE_FLUSH(hw);

	return status;
}

/**
 *  ixgbe_get_i2c_ack - Polls for I2C ACK
 *  @hw: pointer to hardware structure
 *
 *  Clocks in/out one bit via I2C data/clock
 **/
static int ixgbe_get_i2c_ack(struct ixgbe_hw *hw)
{
	u32 i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	u32 data_oe_bit = IXGBE_I2C_DATA_OE_N_EN(hw);
	u32 timeout = 10;
	bool ack = true;
	int status = 0;
	u32 i = 0;

	if (data_oe_bit) {
		i2cctl |= IXGBE_I2C_DATA_OUT(hw);
		i2cctl |= data_oe_bit;
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), i2cctl);
		IXGBE_WRITE_FLUSH(hw);
	}
	ixgbe_raise_i2c_clk(hw, &i2cctl);

	/* Minimum high period of clock is 4us */
	udelay(IXGBE_I2C_T_HIGH);

	/* Poll for ACK.  Note that ACK in I2C spec is
	 * transition from 1 to 0 */
	for (i = 0; i < timeout; i++) {
		i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
		ack = ixgbe_get_i2c_data(hw, &i2cctl);

		udelay(1);
		if (ack == 0)
			break;
	}

	if (ack == 1) {
		hw_dbg(hw, "I2C ack was not received.\n");
		status = -EIO;
	}

	ixgbe_lower_i2c_clk(hw, &i2cctl);

	/* Minimum low period of clock is 4.7 us */
	udelay(IXGBE_I2C_T_LOW);

	return status;
}

/**
 *  ixgbe_clock_in_i2c_bit - Clocks in one bit via I2C data/clock
 *  @hw: pointer to hardware structure
 *  @data: read data value
 *
 *  Clocks in one bit via I2C data/clock
 **/
static int ixgbe_clock_in_i2c_bit(struct ixgbe_hw *hw, bool *data)
{
	u32 i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	u32 data_oe_bit = IXGBE_I2C_DATA_OE_N_EN(hw);

	if (data_oe_bit) {
		i2cctl |= IXGBE_I2C_DATA_OUT(hw);
		i2cctl |= data_oe_bit;
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), i2cctl);
		IXGBE_WRITE_FLUSH(hw);
	}
	ixgbe_raise_i2c_clk(hw, &i2cctl);

	/* Minimum high period of clock is 4us */
	udelay(IXGBE_I2C_T_HIGH);

	i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	*data = ixgbe_get_i2c_data(hw, &i2cctl);

	ixgbe_lower_i2c_clk(hw, &i2cctl);

	/* Minimum low period of clock is 4.7 us */
	udelay(IXGBE_I2C_T_LOW);

	return 0;
}

/**
 *  ixgbe_clock_out_i2c_bit - Clocks in/out one bit via I2C data/clock
 *  @hw: pointer to hardware structure
 *  @data: data value to write
 *
 *  Clocks out one bit via I2C data/clock
 **/
static int ixgbe_clock_out_i2c_bit(struct ixgbe_hw *hw, bool data)
{
	u32 i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	int status;

	status = ixgbe_set_i2c_data(hw, &i2cctl, data);
	if (status == 0) {
		ixgbe_raise_i2c_clk(hw, &i2cctl);

		/* Minimum high period of clock is 4us */
		udelay(IXGBE_I2C_T_HIGH);

		ixgbe_lower_i2c_clk(hw, &i2cctl);

		/* Minimum low period of clock is 4.7 us.
		 * This also takes care of the data hold time.
		 */
		udelay(IXGBE_I2C_T_LOW);
	} else {
		hw_dbg(hw, "I2C data was not set to %X\n", data);
		return -EIO;
	}

	return 0;
}
/**
 *  ixgbe_raise_i2c_clk - Raises the I2C SCL clock
 *  @hw: pointer to hardware structure
 *  @i2cctl: Current value of I2CCTL register
 *
 *  Raises the I2C clock line '0'->'1'
 *  Negates the I2C clock output enable on X550 hardware.
 **/
static void ixgbe_raise_i2c_clk(struct ixgbe_hw *hw, u32 *i2cctl)
{
	u32 clk_oe_bit = IXGBE_I2C_CLK_OE_N_EN(hw);
	u32 i = 0;
	u32 timeout = IXGBE_I2C_CLOCK_STRETCHING_TIMEOUT;
	u32 i2cctl_r = 0;

	if (clk_oe_bit) {
		*i2cctl |= clk_oe_bit;
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), *i2cctl);
	}

	for (i = 0; i < timeout; i++) {
		*i2cctl |= IXGBE_I2C_CLK_OUT(hw);
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), *i2cctl);
		IXGBE_WRITE_FLUSH(hw);
		/* SCL rise time (1000ns) */
		udelay(IXGBE_I2C_T_RISE);

		i2cctl_r = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
		if (i2cctl_r & IXGBE_I2C_CLK_IN(hw))
			break;
	}
}

/**
 *  ixgbe_lower_i2c_clk - Lowers the I2C SCL clock
 *  @hw: pointer to hardware structure
 *  @i2cctl: Current value of I2CCTL register
 *
 *  Lowers the I2C clock line '1'->'0'
 *  Asserts the I2C clock output enable on X550 hardware.
 **/
static void ixgbe_lower_i2c_clk(struct ixgbe_hw *hw, u32 *i2cctl)
{

	*i2cctl &= ~IXGBE_I2C_CLK_OUT(hw);
	*i2cctl &= ~IXGBE_I2C_CLK_OE_N_EN(hw);

	IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), *i2cctl);
	IXGBE_WRITE_FLUSH(hw);

	/* SCL fall time (300ns) */
	udelay(IXGBE_I2C_T_FALL);
}

/**
 *  ixgbe_set_i2c_data - Sets the I2C data bit
 *  @hw: pointer to hardware structure
 *  @i2cctl: Current value of I2CCTL register
 *  @data: I2C data value (0 or 1) to set
 *
 *  Sets the I2C data bit
 *  Asserts the I2C data output enable on X550 hardware.
 **/
static int ixgbe_set_i2c_data(struct ixgbe_hw *hw, u32 *i2cctl, bool data)
{
	u32 data_oe_bit = IXGBE_I2C_DATA_OE_N_EN(hw);

	if (data)
		*i2cctl |= IXGBE_I2C_DATA_OUT(hw);
	else
		*i2cctl &= ~IXGBE_I2C_DATA_OUT(hw);
	*i2cctl &= ~data_oe_bit;

	IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), *i2cctl);
	IXGBE_WRITE_FLUSH(hw);

	/* Data rise/fall (1000ns/300ns) and set-up time (250ns) */
	udelay(IXGBE_I2C_T_RISE + IXGBE_I2C_T_FALL + IXGBE_I2C_T_SU_DATA);

	if (!data)	/* Can't verify data in this case */
		return 0;
	if (data_oe_bit) {
		*i2cctl |= data_oe_bit;
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), *i2cctl);
		IXGBE_WRITE_FLUSH(hw);
	}

	/* Verify data was set correctly */
	*i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));
	if (data != ixgbe_get_i2c_data(hw, i2cctl)) {
		hw_dbg(hw, "Error - I2C data was not set to %X.\n", data);
		return -EIO;
	}

	return 0;
}

/**
 *  ixgbe_get_i2c_data - Reads the I2C SDA data bit
 *  @hw: pointer to hardware structure
 *  @i2cctl: Current value of I2CCTL register
 *
 *  Returns the I2C data bit value
 *  Negates the I2C data output enable on X550 hardware.
 **/
static bool ixgbe_get_i2c_data(struct ixgbe_hw *hw, u32 *i2cctl)
{
	u32 data_oe_bit = IXGBE_I2C_DATA_OE_N_EN(hw);

	if (data_oe_bit) {
		*i2cctl |= data_oe_bit;
		IXGBE_WRITE_REG(hw, IXGBE_I2CCTL(hw), *i2cctl);
		IXGBE_WRITE_FLUSH(hw);
		udelay(IXGBE_I2C_T_FALL);
	}

	if (*i2cctl & IXGBE_I2C_DATA_IN(hw))
		return true;
	return false;
}

/**
 *  ixgbe_i2c_bus_clear - Clears the I2C bus
 *  @hw: pointer to hardware structure
 *
 *  Clears the I2C bus by sending nine clock pulses.
 *  Used when data line is stuck low.
 **/
static void ixgbe_i2c_bus_clear(struct ixgbe_hw *hw)
{
	u32 i2cctl;
	u32 i;

	ixgbe_i2c_start(hw);
	i2cctl = IXGBE_READ_REG(hw, IXGBE_I2CCTL(hw));

	ixgbe_set_i2c_data(hw, &i2cctl, 1);

	for (i = 0; i < 9; i++) {
		ixgbe_raise_i2c_clk(hw, &i2cctl);

		/* Min high period of clock is 4us */
		udelay(IXGBE_I2C_T_HIGH);

		ixgbe_lower_i2c_clk(hw, &i2cctl);

		/* Min low period of clock is 4.7us*/
		udelay(IXGBE_I2C_T_LOW);
	}

	ixgbe_i2c_start(hw);

	/* Put the i2c bus back to default state */
	ixgbe_i2c_stop(hw);
}

/**
 *  ixgbe_tn_check_overtemp - Checks if an overtemp occurred.
 *  @hw: pointer to hardware structure
 *
 *  Checks if the LASI temp alarm status was triggered due to overtemp
 *
 *  Return true when an overtemp event detected, otherwise false.
 **/
bool ixgbe_tn_check_overtemp(struct ixgbe_hw *hw)
{
	u16 phy_data = 0;
	u32 status;

	if (hw->device_id != IXGBE_DEV_ID_82599_T3_LOM)
		return false;

	/* Check that the LASI temp alarm status was triggered */
	status = hw->phy.ops.read_reg(hw, IXGBE_TN_LASI_STATUS_REG,
				      MDIO_MMD_PMAPMD, &phy_data);
	if (status)
		return false;

	return !!(phy_data & IXGBE_TN_LASI_STATUS_TEMP_ALARM);
}

/** ixgbe_set_copper_phy_power - Control power for copper phy
 *  @hw: pointer to hardware structure
 *  @on: true for on, false for off
 **/
int ixgbe_set_copper_phy_power(struct ixgbe_hw *hw, bool on)
{
	u32 status;
	u16 reg;

	/* Bail if we don't have copper phy */
	if (hw->mac.ops.get_media_type(hw) != ixgbe_media_type_copper)
		return 0;

	if (!on && ixgbe_mng_present(hw))
		return 0;

	status = hw->phy.ops.read_reg(hw, MDIO_CTRL1, MDIO_MMD_VEND1, &reg);
	if (status)
		return status;

	if (on) {
		reg &= ~IXGBE_MDIO_PHY_SET_LOW_POWER_MODE;
	} else {
		if (ixgbe_check_reset_blocked(hw))
			return 0;
		reg |= IXGBE_MDIO_PHY_SET_LOW_POWER_MODE;
	}

	status = hw->phy.ops.write_reg(hw, MDIO_CTRL1, MDIO_MMD_VEND1, reg);
	return status;
}
