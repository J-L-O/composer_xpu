# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The GPU device used for training."""

from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar

import torch
import intel_extension_for_pytorch as ipex
import torch.utils.data

from composer.devices.device import Device
from composer.utils import dist

__all__ = ['DeviceXPU']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceXPU(Device):
    """An extension of :class:`~composer.devices.device.Device` for Intel GPUs.

    Args:
        device_id (int, optional): Integer ID of a GPU device to train with. If not specified, the local rank
            of the current process is used. Default: None.
        allow_tf32 (bool, optional): Whether to allow TF32 matrix multiplications. Defaults to True.
            For more information, see :ref:`torch:tf32_on_ampere`.
    """
    dist_backend = 'ccl'
    name = 'xpu'

    def __init__(
        self,
        device_id: Optional[int] = None,
        *,
        allow_tf32: bool = True,
    ):
        if not torch.xpu.is_available():
            raise ValueError('DeviceXPU cannot be created as torch.xpu is not available.')
        if device_id is None:
            device_id = dist.get_local_rank()
        self._device = torch.device(f'xpu:{device_id}')
        torch.xpu.set_device(self._device)
        assert torch.xpu.current_device() == device_id

        if allow_tf32:
            ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.TF32)
        else:
            ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'rng': torch.xpu.get_rng_state(self._device),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        torch.xpu.set_rng_state(state['rng'], self._device)
