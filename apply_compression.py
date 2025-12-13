
# apply_compression.py
import torch.nn as nn
from compression import QuantizedConv2d

def apply_quantization(model, num_bits):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, QuantizedConv2d(module, num_bits))
        else:
            apply_quantization(module, num_bits)
