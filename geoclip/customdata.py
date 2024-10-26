import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class CustomDataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None):
        super(CustomDataParallel, self).__init__(module, device_ids)
        self.src_device = torch.device(f"cuda:{self.device_ids[0]}" if self.device_ids else "cpu")

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.src_device)

    def gather(self, outputs, output_device):
        # Move all tensors to the same device before concatenating
        outputs = [output.to(output_device) for output in outputs]
        return torch.cat(outputs, dim=0)