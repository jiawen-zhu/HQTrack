from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import spatial_correlation_sampler_backend as correlation


def spatial_correlation_sample(input1,
                               input2,
                               kernel_size=1,
                               patch_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               dilation_patch=1):
    """Apply spatial correlation sampling on from input1 to input2,

    Every parameter except input1 and input2 can be either single int
    or a pair of int. For more information about Spatial Correlation
    Sampling, see this page.
    https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/

    Args:
        input1 : The first parameter.
        input2 : The second parameter.
        kernel_size : total size of your correlation kernel, in pixels
        patch_size : total size of your patch, determining how many
            different shifts will be applied
        stride : stride of the spatial sampler, will modify output
            height and width
        padding : padding applied to input1 and input2 before applying
            the correlation sampling, will modify output height and width
        dilation_patch : step for every shift in patch

    Returns:
        Tensor: Result of correlation sampling

    """
    return SpatialCorrelationSamplerFunction.apply(input1, input2,
                                                   kernel_size, patch_size,
                                                   stride, padding, dilation, dilation_patch)


class SpatialCorrelationSamplerFunction(Function):

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                kernel_size=1,
                patch_size=1,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=1):

        ctx.save_for_backward(input1, input2)
        kH, kW = ctx.kernel_size = _pair(kernel_size)
        patchH, patchW = ctx.patch_size = _pair(patch_size)
        padH, padW = ctx.padding = _pair(padding)
        dilationH, dilationW = ctx.dilation = _pair(dilation)
        dilation_patchH, dilation_patchW = ctx.dilation_patch = _pair(dilation_patch)
        dH, dW = ctx.stride = _pair(stride)

        output = correlation.forward(input1, input2,
                                     kH, kW, patchH, patchW,
                                     padH, padW, dilationH, dilationW,
                                     dilation_patchH, dilation_patchW,
                                     dH, dW)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_variables

        kH, kW = ctx.kernel_size
        patchH, patchW = ctx.patch_size
        padH, padW = ctx.padding
        dilationH, dilationW = ctx.dilation
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride

        grad_input1, grad_input2 = correlation.backward(input1, input2, grad_output,
                                                        kH, kW, patchH, patchW,
                                                        padH, padW, dilationH, dilationW,
                                                        dilation_patchH, dilation_patchW,
                                                        dH, dW)
        return grad_input1, grad_input2, None, None, None, None, None, None


class SpatialCorrelationSampler(nn.Module):
    def __init__(self, kernel_size=1, patch_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(SpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return SpatialCorrelationSamplerFunction.apply(input1, input2, self.kernel_size,
                                                       self.patch_size, self.stride,
                                                       self.padding, self.dilation, self.dilation_patch)
