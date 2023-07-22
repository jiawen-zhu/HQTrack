from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

from spatial_correlation_sampler import SpatialCorrelationSampler


def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i))


def zero_grad(variables):
    for variable in variables:
        if variable.grad is not None: variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(input1, input2, correlation_sampler, verbose, gpu_index=0):
    device = torch.device(f"cuda:{gpu_index}")

    cpu_values = correlation_sampler(input1, input2)
    cuda_values = correlation_sampler(input1.to(device), input2.to(device))

    print(f"Forward: CPU vs. CUDA device:{gpu_index} ... ", end='')
    check_equal(cpu_values, cuda_values, verbose)
    print('Ok')


def check_backward(input1, input2, correlation_sampler, verbose, gpu_index=0):
    device = torch.device(f"cuda:{gpu_index}")

    zero_grad([input1, input2])

    cpu_values = correlation_sampler(input1, input2)
    cpu_values.sum().backward()
    grad_cpu = get_grads([input1, input2])

    zero_grad([input1, input2])

    cuda_values = correlation_sampler(input1.to(device), input2.to(device))
    cuda_values.sum().backward()
    grad_cuda = get_grads([input1, input2])

    print(f"Backward: CPU vs. CUDA device:{gpu_index} ... ", end='')
    check_equal(grad_cpu, grad_cuda, verbose)
    print('Ok')


def check_multi_gpu_forward(correlation_sampler, verbose):
    print("Multi-GPU forward")
    total_gpus = torch.cuda.device_count()
    for gpu in range(total_gpus):
        check_forward(input1, input2, correlation_sampler, verbose, gpu_index=gpu)

def check_multi_gpu_backward(correlation_sampler, verbose):
    print("Multi-GPU backward")
    total_gpus = torch.cuda.device_count()
    for gpu in range(total_gpus):
        check_backward(input1, input2, correlation_sampler, verbose, gpu_index=gpu)


parser = argparse.ArgumentParser()
parser.add_argument('direction', choices=['forward', 'backward'], nargs='+')
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-k', '--kernel-size', type=int, default=3)
parser.add_argument('--patch', type=int, default=3)
parser.add_argument('--patch_dilation', type=int, default=2)
parser.add_argument('-c', '--channel', type=int, default=10)
parser.add_argument('--height', type=int, default=10)
parser.add_argument('-w', '--width', type=int, default=10)
parser.add_argument('-s', '--stride', type=int, default=2)
parser.add_argument('-p', '--pad', type=int, default=5)
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('-d', '--dilation', type=int, default=2)
args = parser.parse_args()
print(args)

assert(torch.cuda.is_available()), "no comparison to make"
input1 = torch.randn(args.batch_size,
                     args.channel,
                     args.height,
                     args.width).double()
input2 = torch.randn(args.batch_size,
                     args.channel,
                     args.height,
                     args.width).double()
input1.requires_grad = True
input2.requires_grad = True

correlation_sampler = SpatialCorrelationSampler(
    args.kernel_size,
    args.patch,
    args.stride,
    args.pad,
    args.dilation,
    args.patch_dilation)

if 'forward' in args.direction:
    check_forward(input1, input2, correlation_sampler, args.verbose)
    if torch.cuda.device_count() > 1: check_multi_gpu_forward(correlation_sampler, args.verbose)

if 'backward' in args.direction:
    check_backward(input1, input2, correlation_sampler, args.verbose)
    if torch.cuda.device_count() > 1: check_multi_gpu_backward(correlation_sampler, args.verbose)
