// SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0-only)
/* Copyright(c) 2015 - 2021 Intel Corporation */
#include <linux/workqueue.h>
#include <linux/pci.h>
#include <linux/device.h>
#include "adf_common_drv.h"
#include "adf_cfg.h"
#include "adf_pfvf_pf_msg.h"

#define ADF_VF2PF_RATELIMIT_INTERVAL	8
#define ADF_VF2PF_RATELIMIT_BURST	130

static struct workqueue_struct *pf2vf_resp_wq;

struct adf_pf2vf_resp {
	struct work_struct pf2vf_resp_work;
	struct adf_accel_vf_info *vf_info;
};

static void adf_iov_send_resp(struct work_struct *work)
{
	struct adf_pf2vf_resp *pf2vf_resp =
		container_of(work, struct adf_pf2vf_resp, pf2vf_resp_work);
	struct adf_accel_vf_info *vf_info = pf2vf_resp->vf_info;
	struct adf_accel_dev *accel_dev = vf_info->accel_dev;
	u32 vf_nr = vf_info->vf_nr;
	bool ret;

	mutex_lock(&vf_info->pfvf_mig_lock);
	ret = adf_recv_and_handle_vf2pf_msg(accel_dev, vf_nr);
	if (ret)
		/* re-enable interrupt on PF from this VF */
		adf_enable_vf2pf_interrupts(accel_dev, 1 << vf_nr);
	mutex_unlock(&vf_info->pfvf_mig_lock);

	kfree(pf2vf_resp);
}

void adf_schedule_vf2pf_handler(struct adf_accel_vf_info *vf_info)
{
	struct adf_pf2vf_resp *pf2vf_resp;

	pf2vf_resp = kzalloc(sizeof(*pf2vf_resp), GFP_ATOMIC);
	if (!pf2vf_resp)
		return;

	pf2vf_resp->vf_info = vf_info;
	INIT_WORK(&pf2vf_resp->pf2vf_resp_work, adf_iov_send_resp);
	queue_work(pf2vf_resp_wq, &pf2vf_resp->pf2vf_resp_work);
}

static int adf_enable_sriov(struct adf_accel_dev *accel_dev)
{
	struct pci_dev *pdev = accel_to_pci_dev(accel_dev);
	int totalvfs = pci_sriov_get_totalvfs(pdev);
	struct adf_hw_device_data *hw_data = accel_dev->hw_device;
	struct adf_accel_vf_info *vf_info;
	int i;

	for (i = 0, vf_info = accel_dev->pf.vf_info; i < totalvfs;
	     i++, vf_info++) {
		/* This ptr will be populated when VFs will be created */
		vf_info->accel_dev = accel_dev;
		vf_info->vf_nr = i;

		mutex_init(&vf_info->pf2vf_lock);
		mutex_init(&vf_info->pfvf_mig_lock);
		ratelimit_state_init(&vf_info->vf2pf_ratelimit,
				     ADF_VF2PF_RATELIMIT_INTERVAL,
				     ADF_VF2PF_RATELIMIT_BURST);
	}

	/* Set Valid bits in AE Thread to PCIe Function Mapping */
	if (hw_data->configure_iov_threads)
		hw_data->configure_iov_threads(accel_dev, true);

	/* Enable VF to PF interrupts for all VFs */
	adf_enable_vf2pf_interrupts(accel_dev, BIT_ULL(totalvfs) - 1);

	/*
	 * Due to the hardware design, when SR-IOV and the ring arbiter
	 * are enabled all the VFs supported in hardware must be enabled in
	 * order for all the hardware resources (i.e. bundles) to be usable.
	 * When SR-IOV is enabled, each of the VFs will own one bundle.
	 */
	return pci_enable_sriov(pdev, totalvfs);
}

static int adf_add_sriov_configuration(struct adf_accel_dev *accel_dev)
{
	unsigned long val = 0;
	int ret;

	ret = adf_cfg_section_add(accel_dev, ADF_KERNEL_SEC);
	if (ret)
		return ret;

	ret = adf_cfg_add_key_value_param(accel_dev, ADF_KERNEL_SEC, ADF_NUM_CY,
					  &val, ADF_DEC);
	if (ret)
		return ret;

	ret = adf_cfg_add_key_value_param(accel_dev, ADF_KERNEL_SEC, ADF_NUM_DC,
					  &val, ADF_DEC);
	if (ret)
		return ret;

	set_bit(ADF_STATUS_CONFIGURED, &accel_dev->status);

	return ret;
}

static int adf_do_disable_sriov(struct adf_accel_dev *accel_dev)
{
	int ret;

	if (adf_dev_in_use(accel_dev)) {
		dev_err(&GET_DEV(accel_dev),
			"Cannot disable SR-IOV, device in use\n");
		return -EBUSY;
	}

	if (adf_dev_started(accel_dev)) {
		if (adf_devmgr_in_reset(accel_dev)) {
			dev_err(&GET_DEV(accel_dev),
				"Cannot disable SR-IOV, device in reset\n");
			return -EBUSY;
		}

		ret = adf_dev_down(accel_dev);
		if (ret)
			goto err_del_cfg;
	}

	adf_disable_sriov(accel_dev);

	ret = adf_dev_up(accel_dev, true);
	if (ret)
		goto err_del_cfg;

	return 0;

err_del_cfg:
	adf_cfg_del_all_except(accel_dev, ADF_GENERAL_SEC);
	return ret;
}

static int adf_do_enable_sriov(struct adf_accel_dev *accel_dev)
{
	struct pci_dev *pdev = accel_to_pci_dev(accel_dev);
	int totalvfs = pci_sriov_get_totalvfs(pdev);
	unsigned long val;
	int ret;

	if (!device_iommu_mapped(&GET_DEV(accel_dev))) {
		dev_warn(&GET_DEV(accel_dev),
			 "IOMMU should be enabled for SR-IOV to work correctly\n");
	}

	if (adf_dev_started(accel_dev)) {
		if (adf_devmgr_in_reset(accel_dev) || adf_dev_in_use(accel_dev)) {
			dev_err(&GET_DEV(accel_dev), "Device busy\n");
			return -EBUSY;
		}

		ret = adf_dev_down(accel_dev);
		if (ret)
			return ret;
	}

	ret = adf_add_sriov_configuration(accel_dev);
	if (ret)
		goto err_del_cfg;

	/* Allocate memory for VF info structs */
	accel_dev->pf.vf_info = kcalloc(totalvfs, sizeof(struct adf_accel_vf_info),
					GFP_KERNEL);
	ret = -ENOMEM;
	if (!accel_dev->pf.vf_info)
		goto err_del_cfg;

	ret = adf_dev_up(accel_dev, false);
	if (ret) {
		dev_err(&GET_DEV(accel_dev), "Failed to start qat_dev%d\n",
			accel_dev->accel_id);
		goto err_free_vf_info;
	}

	ret = adf_enable_sriov(accel_dev);
	if (ret)
		goto err_free_vf_info;

	val = 1;
	ret = adf_cfg_add_key_value_param(accel_dev, ADF_GENERAL_SEC, ADF_SRIOV_ENABLED,
					  &val, ADF_DEC);
	if (ret)
		goto err_free_vf_info;

	return totalvfs;

err_free_vf_info:
	adf_dev_down(accel_dev);
	kfree(accel_dev->pf.vf_info);
	accel_dev->pf.vf_info = NULL;
	return ret;
err_del_cfg:
	adf_cfg_del_all_except(accel_dev, ADF_GENERAL_SEC);
	return ret;
}

void adf_reenable_sriov(struct adf_accel_dev *accel_dev)
{
	struct pci_dev *pdev = accel_to_pci_dev(accel_dev);
	char cfg[ADF_CFG_MAX_VAL_LEN_IN_BYTES] = {0};

	if (adf_cfg_get_param_value(accel_dev, ADF_GENERAL_SEC,
				    ADF_SRIOV_ENABLED, cfg))
		return;

	if (!accel_dev->pf.vf_info)
		return;

	if (adf_add_sriov_configuration(accel_dev))
		return;

	dev_dbg(&pdev->dev, "Re-enabling SRIOV\n");
	adf_enable_sriov(accel_dev);
}

/**
 * adf_disable_sriov() - Disable SRIOV for the device
 * @accel_dev:  Pointer to accel device.
 *
 * Function disables SRIOV for the accel device.
 *
 * Return: 0 on success, error code otherwise.
 */
void adf_disable_sriov(struct adf_accel_dev *accel_dev)
{
	struct adf_hw_device_data *hw_data = accel_dev->hw_device;
	int totalvfs = pci_sriov_get_totalvfs(accel_to_pci_dev(accel_dev));
	struct adf_accel_vf_info *vf;
	int i;

	if (!accel_dev->pf.vf_info)
		return;

	adf_pf2vf_notify_restarting(accel_dev);
	adf_pf2vf_wait_for_restarting_complete(accel_dev);
	pci_disable_sriov(accel_to_pci_dev(accel_dev));

	/* Disable VF to PF interrupts */
	adf_disable_all_vf2pf_interrupts(accel_dev);

	/* Clear Valid bits in AE Thread to PCIe Function Mapping */
	if (hw_data->configure_iov_threads)
		hw_data->configure_iov_threads(accel_dev, false);

	for (i = 0, vf = accel_dev->pf.vf_info; i < totalvfs; i++, vf++) {
		mutex_destroy(&vf->pf2vf_lock);
		mutex_destroy(&vf->pfvf_mig_lock);
	}

	if (!test_bit(ADF_STATUS_RESTARTING, &accel_dev->status)) {
		kfree(accel_dev->pf.vf_info);
		accel_dev->pf.vf_info = NULL;
	}
}
EXPORT_SYMBOL_GPL(adf_disable_sriov);

/**
 * adf_sriov_configure() - Enable SRIOV for the device
 * @pdev:  Pointer to PCI device.
 * @numvfs: Number of virtual functions (VFs) to enable.
 *
 * Note that the @numvfs parameter is ignored and all VFs supported by the
 * device are enabled due to the design of the hardware.
 *
 * Function enables SRIOV for the PCI device.
 *
 * Return: number of VFs enabled on success, error code otherwise.
 */
int adf_sriov_configure(struct pci_dev *pdev, int numvfs)
{
	struct adf_accel_dev *accel_dev = adf_devmgr_pci_to_accel_dev(pdev);

	if (!accel_dev) {
		dev_err(&pdev->dev, "Failed to find accel_dev\n");
		return -EFAULT;
	}

	if (numvfs)
		return adf_do_enable_sriov(accel_dev);
	else
		return adf_do_disable_sriov(accel_dev);
}
EXPORT_SYMBOL_GPL(adf_sriov_configure);

int __init adf_init_pf_wq(void)
{
	/* Workqueue for PF2VF responses */
	pf2vf_resp_wq = alloc_workqueue("qat_pf2vf_resp_wq", WQ_MEM_RECLAIM, 0);

	return !pf2vf_resp_wq ? -ENOMEM : 0;
}

void adf_exit_pf_wq(void)
{
	if (pf2vf_resp_wq) {
		destroy_workqueue(pf2vf_resp_wq);
		pf2vf_resp_wq = NULL;
	}
}
