import torch
import torch.nn as nn
import torch.nn.functional as F
class UniformQuantizer:
    """
    Symmetric uniform quantizer for weights and activations
    """

    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

    def compute_scale(self, tensor):
        max_val = tensor.abs().max()
        if max_val == 0:
            return 1.0
        return max_val / self.qmax

    def quantize(self, tensor):
        scale = self.compute_scale(tensor)
        q_tensor = torch.clamp(
            (tensor / scale).round(),
            self.qmin,
            self.qmax
        )
        return q_tensor, scale

    def dequantize(self, q_tensor, scale):
        return q_tensor * scale

class QuantizedConv2d(nn.Module):
    """
    Conv2d layer with quantized weights and activations
    """

    def __init__(self, conv, num_bits):
        super().__init__()
        self.conv = conv
        self.weight_quant = UniformQuantizer(num_bits)
        self.act_quant = UniformQuantizer(num_bits)

    def forward(self, x):
        # -------- Activation Quantization --------
        q_x, x_scale = self.act_quant.quantize(x)
        x_deq = self.act_quant.dequantize(q_x, x_scale)

        # -------- Weight Quantization --------
        w = self.conv.weight
        q_w, w_scale = self.weight_quant.quantize(w)
        w_deq = self.weight_quant.dequantize(q_w, w_scale)

        return F.conv2d(
            x_deq,
            w_deq,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
