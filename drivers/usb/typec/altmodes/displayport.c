// SPDX-License-Identifier: GPL-2.0
/*
 * USB Typec-C DisplayPort Alternate Mode driver
 *
 * Copyright (C) 2018 Intel Corporation
 * Author: Heikki Krogerus <heikki.krogerus@linux.intel.com>
 *
 * DisplayPort is trademark of VESA (www.vesa.org)
 */

#include <linux/delay.h>
#include <linux/mutex.h>
#include <linux/module.h>
#include <linux/property.h>
#include <linux/usb/pd_vdo.h>
#include <linux/usb/typec_dp.h>
#include <drm/drm_connector.h>
#include "displayport.h"

#define DP_HEADER(_dp, ver, cmd)	(VDO((_dp)->alt->svid, 1, ver, cmd)	\
					 | VDO_OPOS(USB_TYPEC_DP_MODE))

enum {
	DP_CONF_USB,
	DP_CONF_DFP_D,
	DP_CONF_UFP_D,
	DP_CONF_DUAL_D,
};

/* Pin assignments that use USB3.1 Gen2 signaling to carry DP protocol */
#define DP_PIN_ASSIGN_GEN2_BR_MASK	(BIT(DP_PIN_ASSIGN_A) | \
					 BIT(DP_PIN_ASSIGN_B))

/* Pin assignments that use DP v1.3 signaling to carry DP protocol */
#define DP_PIN_ASSIGN_DP_BR_MASK	(BIT(DP_PIN_ASSIGN_C) | \
					 BIT(DP_PIN_ASSIGN_D) | \
					 BIT(DP_PIN_ASSIGN_E) | \
					 BIT(DP_PIN_ASSIGN_F))

/* DP only pin assignments */
#define DP_PIN_ASSIGN_DP_ONLY_MASK	(BIT(DP_PIN_ASSIGN_A) | \
					 BIT(DP_PIN_ASSIGN_C) | \
					 BIT(DP_PIN_ASSIGN_E))

/* Pin assignments where one channel is for USB */
#define DP_PIN_ASSIGN_MULTI_FUNC_MASK	(BIT(DP_PIN_ASSIGN_B) | \
					 BIT(DP_PIN_ASSIGN_D) | \
					 BIT(DP_PIN_ASSIGN_F))

enum dp_state {
	DP_STATE_IDLE,
	DP_STATE_ENTER,
	DP_STATE_ENTER_PRIME,
	DP_STATE_UPDATE,
	DP_STATE_CONFIGURE,
	DP_STATE_CONFIGURE_PRIME,
	DP_STATE_EXIT,
	DP_STATE_EXIT_PRIME,
};

struct dp_altmode {
	struct typec_displayport_data data;
	struct typec_displayport_data data_prime;

	enum dp_state state;
	bool hpd;
	bool pending_hpd;
	u32 irq_hpd_count;
	/*
	 * hpd is mandatory for irq_hpd assertion, so irq_hpd also needs its own pending flag if
	 * both hpd and irq_hpd are asserted in the first Status Update before the pin assignment
	 * is configured.
	 */
	bool pending_irq_hpd;

	struct mutex lock; /* device lock */
	struct work_struct work;
	struct typec_altmode *alt;
	const struct typec_altmode *port;
	struct fwnode_handle *connector_fwnode;
	struct typec_altmode *plug_prime;
};

static int dp_altmode_notify(struct dp_altmode *dp)
{
	unsigned long conf;
	u8 state;

	if (dp->data.conf) {
		state = get_count_order(DP_CONF_GET_PIN_ASSIGN(dp->data.conf));
		conf = TYPEC_MODAL_STATE(state);
	} else {
		conf = TYPEC_STATE_USB;
	}

	return typec_altmode_notify(dp->alt, conf, &dp->data);
}

static int dp_altmode_configure(struct dp_altmode *dp, u8 con)
{
	u8 pin_assign = 0;
	u32 conf;

	/* DP Signalling */
	conf = (dp->data.conf & DP_CONF_SIGNALLING_MASK) >> DP_CONF_SIGNALLING_SHIFT;

	switch (con) {
	case DP_STATUS_CON_DISABLED:
		return 0;
	case DP_STATUS_CON_DFP_D:
		conf |= DP_CONF_UFP_U_AS_DFP_D;
		pin_assign = DP_CAP_UFP_D_PIN_ASSIGN(dp->alt->vdo) &
			     DP_CAP_DFP_D_PIN_ASSIGN(dp->port->vdo);
		/* Account for active cable capabilities */
		if (dp->plug_prime)
			pin_assign &= DP_CAP_DFP_D_PIN_ASSIGN(dp->plug_prime->vdo);
		break;
	case DP_STATUS_CON_UFP_D:
	case DP_STATUS_CON_BOTH: /* NOTE: First acting as DP source */
		conf |= DP_CONF_UFP_U_AS_UFP_D;
		pin_assign = DP_CAP_PIN_ASSIGN_UFP_D(dp->alt->vdo) &
				 DP_CAP_PIN_ASSIGN_DFP_D(dp->port->vdo);
		/* Account for active cable capabilities */
		if (dp->plug_prime)
			pin_assign &= DP_CAP_UFP_D_PIN_ASSIGN(dp->plug_prime->vdo);
		break;
	default:
		break;
	}

	/* Determining the initial pin assignment. */
	if (!DP_CONF_GET_PIN_ASSIGN(dp->data.conf)) {
		/* Is USB together with DP preferred */
		if (dp->data.status & DP_STATUS_PREFER_MULTI_FUNC &&
		    pin_assign & DP_PIN_ASSIGN_MULTI_FUNC_MASK)
			pin_assign &= DP_PIN_ASSIGN_MULTI_FUNC_MASK;
		else if (pin_assign & DP_PIN_ASSIGN_DP_ONLY_MASK) {
			pin_assign &= DP_PIN_ASSIGN_DP_ONLY_MASK;
			/* Default to pin assign C if available */
			if (pin_assign & BIT(DP_PIN_ASSIGN_C))
				pin_assign = BIT(DP_PIN_ASSIGN_C);
		}

		if (!pin_assign)
			return -EINVAL;

		conf |= DP_CONF_SET_PIN_ASSIGN(pin_assign);
	}

	dp->data.conf = conf;
	if (dp->plug_prime)
		dp->data_prime.conf = conf;

	return 0;
}

static int dp_altmode_status_update(struct dp_altmode *dp)
{
	bool configured = !!DP_CONF_GET_PIN_ASSIGN(dp->data.conf);
	bool hpd = !!(dp->data.status & DP_STATUS_HPD_STATE);
	bool irq_hpd = !!(dp->data.status & DP_STATUS_IRQ_HPD);
	u8 con = DP_STATUS_CONNECTION(dp->data.status);
	int ret = 0;

	if (configured && (dp->data.status & DP_STATUS_SWITCH_TO_USB)) {
		dp->data.conf = 0;
		dp->data_prime.conf = 0;
		dp->state = dp->plug_prime ? DP_STATE_CONFIGURE_PRIME :
					     DP_STATE_CONFIGURE;
	} else if (dp->data.status & DP_STATUS_EXIT_DP_MODE) {
		dp->state = DP_STATE_EXIT;
	} else if (!(con & DP_CONF_CURRENTLY(dp->data.conf))) {
		ret = dp_altmode_configure(dp, con);
		if (!ret) {
			dp->state = dp->plug_prime ? DP_STATE_CONFIGURE_PRIME :
						     DP_STATE_CONFIGURE;
			if (dp->hpd != hpd) {
				dp->hpd = hpd;
				dp->pending_hpd = true;
			}
			if (dp->hpd && dp->pending_hpd && irq_hpd)
				dp->pending_irq_hpd = true;
		}
	} else {
		drm_connector_oob_hotplug_event(dp->connector_fwnode,
						hpd ? connector_status_connected :
						      connector_status_disconnected);
		dp->hpd = hpd;
		sysfs_notify(&dp->alt->dev.kobj, "displayport", "hpd");
		if (hpd && irq_hpd) {
			dp->irq_hpd_count++;
			sysfs_notify(&dp->alt->dev.kobj, "displayport", "irq_hpd");
		}
	}

	return ret;
}

static int dp_altmode_configured(struct dp_altmode *dp)
{
	sysfs_notify(&dp->alt->dev.kobj, "displayport", "configuration");
	sysfs_notify(&dp->alt->dev.kobj, "displayport", "pin_assignment");
	/*
	 * If the DFP_D/UFP_D sends a change in HPD when first notifying the
	 * DisplayPort driver that it is connected, then we wait until
	 * configuration is complete to signal HPD.
	 */
	if (dp->pending_hpd) {
		drm_connector_oob_hotplug_event(dp->connector_fwnode,
						connector_status_connected);
		sysfs_notify(&dp->alt->dev.kobj, "displayport", "hpd");
		dp->pending_hpd = false;
		if (dp->pending_irq_hpd) {
			dp->irq_hpd_count++;
			sysfs_notify(&dp->alt->dev.kobj, "displayport", "irq_hpd");
			dp->pending_irq_hpd = false;
		}
	}

	return dp_altmode_notify(dp);
}

static int dp_altmode_configure_vdm(struct dp_altmode *dp, u32 conf)
{
	int svdm_version = typec_altmode_get_svdm_version(dp->alt);
	u32 header;
	int ret;

	if (svdm_version < 0)
		return svdm_version;

	header = DP_HEADER(dp, svdm_version, DP_CMD_CONFIGURE);
	ret = typec_altmode_notify(dp->alt, TYPEC_STATE_SAFE, &dp->data);
	if (ret) {
		dev_err(&dp->alt->dev,
			"unable to put to connector to safe mode\n");
		return ret;
	}

	ret = typec_altmode_vdm(dp->alt, header, &conf, 2);
	if (ret)
		dp_altmode_notify(dp);

	return ret;
}

static int dp_altmode_configure_vdm_cable(struct dp_altmode *dp, u32 conf)
{
	int svdm_version = typec_altmode_get_cable_svdm_version(dp->plug_prime);
	u32 header;

	if (svdm_version < 0)
		return svdm_version;

	header = DP_HEADER(dp, svdm_version, DP_CMD_CONFIGURE);

	return typec_cable_altmode_vdm(dp->plug_prime, TYPEC_PLUG_SOP_P, header, &conf, 2);
}

static void dp_altmode_work(struct work_struct *work)
{
	struct dp_altmode *dp = container_of(work, struct dp_altmode, work);
	int svdm_version;
	u32 header;
	u32 vdo;
	int ret;

	mutex_lock(&dp->lock);

	switch (dp->state) {
	case DP_STATE_ENTER:
		ret = typec_altmode_enter(dp->alt, NULL);
		if (ret && ret != -EBUSY)
			dev_err(&dp->alt->dev, "failed to enter mode: %d\n", ret);
		break;
	case DP_STATE_ENTER_PRIME:
		ret = typec_cable_altmode_enter(dp->alt, TYPEC_PLUG_SOP_P, NULL);
		/*
		 * If we fail to enter Alt Mode on SOP', then we should drop the
		 * plug from the driver and attempt to run the driver without
		 * it.
		 */
		if (ret && ret != -EBUSY) {
			dev_err(&dp->alt->dev, "plug failed to enter mode\n");
			dp->state = DP_STATE_ENTER;
			goto disable_prime;
		}
		break;
	case DP_STATE_UPDATE:
		svdm_version = typec_altmode_get_svdm_version(dp->alt);
		if (svdm_version < 0)
			break;
		header = DP_HEADER(dp, svdm_version, DP_CMD_STATUS_UPDATE);
		vdo = 1;
		ret = typec_altmode_vdm(dp->alt, header, &vdo, 2);
		if (ret)
			dev_err(&dp->alt->dev,
				"unable to send Status Update command (%d)\n",
				ret);
		break;
	case DP_STATE_CONFIGURE:
		ret = dp_altmode_configure_vdm(dp, dp->data.conf);
		if (ret)
			dev_err(&dp->alt->dev,
				"unable to send Configure command (%d)\n", ret);
		break;
	case DP_STATE_CONFIGURE_PRIME:
		ret = dp_altmode_configure_vdm_cable(dp, dp->data_prime.conf);
		if (ret) {
			dev_err(&dp->plug_prime->dev,
				"unable to send Configure command (%d)\n",
				ret);
			dp->state = DP_STATE_CONFIGURE;
			goto disable_prime;
		}
		break;
	case DP_STATE_EXIT:
		if (typec_altmode_exit(dp->alt))
			dev_err(&dp->alt->dev, "Exit Mode Failed!\n");
		break;
	case DP_STATE_EXIT_PRIME:
		if (typec_cable_altmode_exit(dp->plug_prime, TYPEC_PLUG_SOP_P))
			dev_err(&dp->plug_prime->dev, "Exit Mode Failed!\n");
		break;
	default:
		break;
	}

	dp->state = DP_STATE_IDLE;

	mutex_unlock(&dp->lock);
	return;

disable_prime:
	typec_altmode_put_plug(dp->plug_prime);
	dp->plug_prime = NULL;
	schedule_work(&dp->work);
	mutex_unlock(&dp->lock);
}

static void dp_altmode_attention(struct typec_altmode *alt, const u32 vdo)
{
	struct dp_altmode *dp = typec_altmode_get_drvdata(alt);
	u8 old_state;

	mutex_lock(&dp->lock);

	old_state = dp->state;
	dp->data.status = vdo;

	if (old_state != DP_STATE_IDLE)
		dev_warn(&alt->dev, "ATTENTION while processing state %d\n",
			 old_state);

	if (dp_altmode_status_update(dp))
		dev_warn(&alt->dev, "%s: status update failed\n", __func__);

	if (dp_altmode_notify(dp))
		dev_err(&alt->dev, "%s: notification failed\n", __func__);

	if (old_state == DP_STATE_IDLE && dp->state != DP_STATE_IDLE)
		schedule_work(&dp->work);

	mutex_unlock(&dp->lock);
}

static int dp_altmode_vdm(struct typec_altmode *alt,
			  const u32 hdr, const u32 *vdo, int count)
{
	struct dp_altmode *dp = typec_altmode_get_drvdata(alt);
	int cmd_type = PD_VDO_CMDT(hdr);
	int cmd = PD_VDO_CMD(hdr);
	int ret = 0;

	mutex_lock(&dp->lock);

	if (dp->state != DP_STATE_IDLE) {
		ret = -EBUSY;
		goto err_unlock;
	}

	switch (cmd_type) {
	case CMDT_RSP_ACK:
		switch (cmd) {
		case CMD_ENTER_MODE:
			typec_altmode_update_active(alt, true);
			dp->state = DP_STATE_UPDATE;
			break;
		case CMD_EXIT_MODE:
			typec_altmode_update_active(alt, false);
			dp->data.status = 0;
			dp->data.conf = 0;
			if (dp->hpd) {
				drm_connector_oob_hotplug_event(dp->connector_fwnode,
								connector_status_disconnected);
				dp->hpd = false;
				sysfs_notify(&dp->alt->dev.kobj, "displayport", "hpd");
			}
			if (dp->plug_prime)
				dp->state = DP_STATE_EXIT_PRIME;
			break;
		case DP_CMD_STATUS_UPDATE:
			dp->data.status = *vdo;
			ret = dp_altmode_status_update(dp);
			break;
		case DP_CMD_CONFIGURE:
			ret = dp_altmode_configured(dp);
			break;
		default:
			break;
		}
		break;
	case CMDT_RSP_NAK:
		switch (cmd) {
		case DP_CMD_STATUS_UPDATE:
			dp->state = DP_STATE_EXIT;
			break;
		case DP_CMD_CONFIGURE:
			dp->data.conf = 0;
			ret = dp_altmode_configured(dp);
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}

	if (dp->state != DP_STATE_IDLE)
		schedule_work(&dp->work);

err_unlock:
	mutex_unlock(&dp->lock);
	return ret;
}

static int dp_cable_altmode_vdm(struct typec_altmode *alt, enum typec_plug_index sop,
				const u32 hdr, const u32 *vdo, int count)
{
	struct dp_altmode *dp = typec_altmode_get_drvdata(alt);
	int cmd_type = PD_VDO_CMDT(hdr);
	int cmd = PD_VDO_CMD(hdr);
	int ret = 0;

	mutex_lock(&dp->lock);

	if (dp->state != DP_STATE_IDLE) {
		ret = -EBUSY;
		goto err_unlock;
	}

	switch (cmd_type) {
	case CMDT_RSP_ACK:
		switch (cmd) {
		case CMD_ENTER_MODE:
			typec_altmode_update_active(dp->plug_prime, true);
			dp->state = DP_STATE_ENTER;
			break;
		case CMD_EXIT_MODE:
			dp->data_prime.status = 0;
			dp->data_prime.conf = 0;
			typec_altmode_update_active(dp->plug_prime, false);
			break;
		case DP_CMD_CONFIGURE:
			dp->state = DP_STATE_CONFIGURE;
			break;
		default:
			break;
		}
		break;
	case CMDT_RSP_NAK:
		switch (cmd) {
		case DP_CMD_CONFIGURE:
			dp->data_prime.conf = 0;
			/* Attempt to configure on SOP, drop plug */
			typec_altmode_put_plug(dp->plug_prime);
			dp->plug_prime = NULL;
			dp->state = DP_STATE_CONFIGURE;
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}

	if (dp->state != DP_STATE_IDLE)
		schedule_work(&dp->work);

err_unlock:
	mutex_unlock(&dp->lock);
	return ret;
}

static int dp_altmode_activate(struct typec_altmode *alt, int activate)
{
	struct dp_altmode *dp = typec_altmode_get_drvdata(alt);
	int ret;

	if (activate) {
		if (dp->plug_prime) {
			ret = typec_cable_altmode_enter(alt, TYPEC_PLUG_SOP_P, NULL);
			if (ret < 0) {
				typec_altmode_put_plug(dp->plug_prime);
				dp->plug_prime = NULL;
			} else {
				return ret;
			}
		}
		return typec_altmode_enter(alt, NULL);
	} else {
		return typec_altmode_exit(alt);
	}
}

static const struct typec_altmode_ops dp_altmode_ops = {
	.attention = dp_altmode_attention,
	.vdm = dp_altmode_vdm,
	.activate = dp_altmode_activate,
};

static const struct typec_cable_ops dp_cable_ops = {
	.vdm = dp_cable_altmode_vdm,
};

static const char * const configurations[] = {
	[DP_CONF_USB]	= "USB",
	[DP_CONF_DFP_D]	= "source",
	[DP_CONF_UFP_D]	= "sink",
};

static ssize_t
configuration_store(struct device *dev, struct device_attribute *attr,
		    const char *buf, size_t size)
{
	struct dp_altmode *dp = dev_get_drvdata(dev);
	u32 conf;
	u32 cap;
	int con;
	int ret = 0;

	con = sysfs_match_string(configurations, buf);
	if (con < 0)
		return con;

	mutex_lock(&dp->lock);

	if (dp->state != DP_STATE_IDLE) {
		ret = -EBUSY;
		goto err_unlock;
	}

	cap = DP_CAP_CAPABILITY(dp->alt->vdo);

	if ((con == DP_CONF_DFP_D && !(cap & DP_CAP_DFP_D)) ||
	    (con == DP_CONF_UFP_D && !(cap & DP_CAP_UFP_D))) {
		ret = -EINVAL;
		goto err_unlock;
	}

	conf = dp->data.conf & ~DP_CONF_DUAL_D;
	conf |= con;

	if (dp->alt->active) {
		ret = dp_altmode_configure_vdm(dp, conf);
		if (ret)
			goto err_unlock;
	}

	dp->data.conf = conf;

err_unlock:
	mutex_unlock(&dp->lock);

	return ret ? ret : size;
}

static ssize_t configuration_show(struct device *dev,
				  struct device_attribute *attr, char *buf)
{
	struct dp_altmode *dp = dev_get_drvdata(dev);
	int len;
	u8 cap;
	u8 cur;
	int i;

	mutex_lock(&dp->lock);

	cap = DP_CAP_CAPABILITY(dp->alt->vdo);
	cur = DP_CONF_CURRENTLY(dp->data.conf);

	len = sprintf(buf, "%s ", cur ? "USB" : "[USB]");

	for (i = 1; i < ARRAY_SIZE(configurations); i++) {
		if (i == cur)
			len += sprintf(buf + len, "[%s] ", configurations[i]);
		else if ((i == DP_CONF_DFP_D && cap & DP_CAP_DFP_D) ||
			 (i == DP_CONF_UFP_D && cap & DP_CAP_UFP_D))
			len += sprintf(buf + len, "%s ", configurations[i]);
	}

	mutex_unlock(&dp->lock);

	buf[len - 1] = '\n';
	return len;
}
static DEVICE_ATTR_RW(configuration);

static const char * const pin_assignments[] = {
	[DP_PIN_ASSIGN_A] = "A",
	[DP_PIN_ASSIGN_B] = "B",
	[DP_PIN_ASSIGN_C] = "C",
	[DP_PIN_ASSIGN_D] = "D",
	[DP_PIN_ASSIGN_E] = "E",
	[DP_PIN_ASSIGN_F] = "F",
};

/*
 * Helper function to extract a peripheral's currently supported
 * Pin Assignments from its DisplayPort alternate mode state.
 */
static u8 get_current_pin_assignments(struct dp_altmode *dp)
{
	if (DP_CONF_CURRENTLY(dp->data.conf) == DP_CONF_UFP_U_AS_DFP_D)
		return DP_CAP_PIN_ASSIGN_DFP_D(dp->alt->vdo);
	else
		return DP_CAP_PIN_ASSIGN_UFP_D(dp->alt->vdo);
}

static ssize_t
pin_assignment_store(struct device *dev, struct device_attribute *attr,
		     const char *buf, size_t size)
{
	struct dp_altmode *dp = dev_get_drvdata(dev);
	u8 assignments;
	u32 conf;
	int ret;

	ret = sysfs_match_string(pin_assignments, buf);
	if (ret < 0)
		return ret;

	conf = DP_CONF_SET_PIN_ASSIGN(BIT(ret));
	ret = 0;

	mutex_lock(&dp->lock);

	if (conf & dp->data.conf)
		goto out_unlock;

	if (dp->state != DP_STATE_IDLE) {
		ret = -EBUSY;
		goto out_unlock;
	}

	assignments = get_current_pin_assignments(dp);

	if (!(DP_CONF_GET_PIN_ASSIGN(conf) & assignments)) {
		ret = -EINVAL;
		goto out_unlock;
	}

	conf |= dp->data.conf & ~DP_CONF_PIN_ASSIGNEMENT_MASK;

	/* Only send Configure command if a configuration has been set */
	if (dp->alt->active && DP_CONF_CURRENTLY(dp->data.conf)) {
		/* todo: send manual configure over SOP'*/
		ret = dp_altmode_configure_vdm(dp, conf);
		if (ret)
			goto out_unlock;
	}

	dp->data.conf = conf;

out_unlock:
	mutex_unlock(&dp->lock);

	return ret ? ret : size;
}

static ssize_t pin_assignment_show(struct device *dev,
				   struct device_attribute *attr, char *buf)
{
	struct dp_altmode *dp = dev_get_drvdata(dev);
	u8 assignments;
	int len = 0;
	u8 cur;
	int i;

	mutex_lock(&dp->lock);

	cur = get_count_order(DP_CONF_GET_PIN_ASSIGN(dp->data.conf));

	assignments = get_current_pin_assignments(dp);

	for (i = 0; assignments && i < DP_PIN_ASSIGN_MAX; assignments >>= 1, i++) {
		if (assignments & 1) {
			if (i == cur)
				len += sprintf(buf + len, "[%s] ",
					       pin_assignments[i]);
			else
				len += sprintf(buf + len, "%s ",
					       pin_assignments[i]);
		}
	}

	mutex_unlock(&dp->lock);

	/* get_current_pin_assignments can return 0 when no matching pin assignments are found */
	if (len == 0)
		len++;

	buf[len - 1] = '\n';
	return len;
}
static DEVICE_ATTR_RW(pin_assignment);

static ssize_t hpd_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct dp_altmode *dp = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%d\n", dp->hpd);
}
static DEVICE_ATTR_RO(hpd);

static ssize_t irq_hpd_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct dp_altmode *dp = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%d\n", dp->irq_hpd_count);
}
static DEVICE_ATTR_RO(irq_hpd);

static struct attribute *displayport_attrs[] = {
	&dev_attr_configuration.attr,
	&dev_attr_pin_assignment.attr,
	&dev_attr_hpd.attr,
	&dev_attr_irq_hpd.attr,
	NULL
};

static const struct attribute_group displayport_group = {
	.name = "displayport",
	.attrs = displayport_attrs,
};

static const struct attribute_group *displayport_groups[] = {
	&displayport_group,
	NULL,
};

int dp_altmode_probe(struct typec_altmode *alt)
{
	const struct typec_altmode *port = typec_altmode_get_partner(alt);
	struct typec_altmode *plug = typec_altmode_get_plug(alt, TYPEC_PLUG_SOP_P);
	struct fwnode_handle *fwnode;
	struct dp_altmode *dp;

	/* FIXME: Port can only be DFP_U. */

	/* Make sure we have compatible pin configurations */
	if (!(DP_CAP_PIN_ASSIGN_DFP_D(port->vdo) &
	      DP_CAP_PIN_ASSIGN_UFP_D(alt->vdo)) &&
	    !(DP_CAP_PIN_ASSIGN_UFP_D(port->vdo) &
	      DP_CAP_PIN_ASSIGN_DFP_D(alt->vdo)))
		return -ENODEV;

	dp = devm_kzalloc(&alt->dev, sizeof(*dp), GFP_KERNEL);
	if (!dp)
		return -ENOMEM;

	INIT_WORK(&dp->work, dp_altmode_work);
	mutex_init(&dp->lock);
	dp->port = port;
	dp->alt = alt;

	alt->desc = "DisplayPort";
	typec_altmode_set_ops(alt, &dp_altmode_ops);

	if (plug) {
		plug->desc = "Displayport";
		plug->cable_ops = &dp_cable_ops;
	}

	dp->plug_prime = plug;

	fwnode = dev_fwnode(alt->dev.parent->parent); /* typec_port fwnode */
	if (fwnode_property_present(fwnode, "displayport"))
		dp->connector_fwnode = fwnode_find_reference(fwnode, "displayport", 0);
	else
		dp->connector_fwnode = fwnode_handle_get(fwnode); /* embedded DP */
	if (IS_ERR(dp->connector_fwnode))
		dp->connector_fwnode = NULL;

	typec_altmode_set_drvdata(alt, dp);
	if (plug)
		typec_altmode_set_drvdata(plug, dp);

	dp->state = plug ? DP_STATE_ENTER_PRIME : DP_STATE_ENTER;
	schedule_work(&dp->work);

	return 0;
}
EXPORT_SYMBOL_GPL(dp_altmode_probe);

void dp_altmode_remove(struct typec_altmode *alt)
{
	struct dp_altmode *dp = typec_altmode_get_drvdata(alt);

	cancel_work_sync(&dp->work);
	typec_altmode_put_plug(dp->plug_prime);

	if (dp->connector_fwnode) {
		drm_connector_oob_hotplug_event(dp->connector_fwnode,
						connector_status_disconnected);

		fwnode_handle_put(dp->connector_fwnode);
	}
}
EXPORT_SYMBOL_GPL(dp_altmode_remove);

static const struct typec_device_id dp_typec_id[] = {
	{ USB_TYPEC_DP_SID },
	{ },
};
MODULE_DEVICE_TABLE(typec, dp_typec_id);

static struct typec_altmode_driver dp_altmode_driver = {
	.id_table = dp_typec_id,
	.probe = dp_altmode_probe,
	.remove = dp_altmode_remove,
	.driver = {
		.name = "typec_displayport",
		.dev_groups = displayport_groups,
	},
};
module_typec_altmode_driver(dp_altmode_driver);

MODULE_AUTHOR("Heikki Krogerus <heikki.krogerus@linux.intel.com>");
MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("DisplayPort Alternate Mode");
