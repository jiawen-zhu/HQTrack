import argparse
import torch
# torch.set_printoptions(precision=1, threshold=10000)
from torch.autograd import gradcheck
from spatial_correlation_sampler import SpatialCorrelationSampler

parser = argparse.ArgumentParser()
parser.add_argument('backend', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('-b', '--batch-size', type=int, default=2)
parser.add_argument('-k', '--kernel-size', type=int, default=3)
parser.add_argument('--patch', type=int, default=3)
parser.add_argument('--patch_dilation', type=int, default=2)
parser.add_argument('-c', '--channel', type=int, default=2)
parser.add_argument('--height', type=int, default=10)
parser.add_argument('-w', '--width', type=int, default=10)
parser.add_argument('-s', '--stride', type=int, default=2)
parser.add_argument('-p', '--pad', type=int, default=1)
parser.add_argument('-d', '--dilation', type=int, default=2)

args = parser.parse_args()

input1 = torch.randn(args.batch_size,
                     args.channel,
                     args.height,
                     args.width,
                     dtype=torch.float64,
                     device=torch.device(args.backend))
input2 = torch.randn(args.batch_size,
                     args.channel,
                     args.height,
                     args.width,
                     dtype=torch.float64,
                     device=torch.device(args.backend))

input1.requires_grad = True
input2.requires_grad = True

correlation_sampler = SpatialCorrelationSampler(args.kernel_size,
                                                args.patch,
                                                args.stride,
                                                args.pad,
                                                args.dilation,
                                                args.patch_dilation)


if gradcheck(correlation_sampler, [input1, input2]):
    print('Ok')
