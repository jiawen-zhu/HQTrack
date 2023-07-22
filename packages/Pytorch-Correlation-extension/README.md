
[![PyPI](https://img.shields.io/pypi/v/spatial-correlation-sampler.svg)](https://pypi.org/project/spatial-correlation-sampler/)


# Pytorch Correlation module

this is a custom C++/Cuda implementation of Correlation module, used e.g. in [FlowNetC](https://arxiv.org/abs/1504.06852)

This [tutorial](http://pytorch.org/tutorials/advanced/cpp_extension.html) was used as a basis for implementation, as well as
[NVIDIA's cuda code](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)

- Build and Install C++ and CUDA extensions by executing `python setup.py install`,
- Benchmark C++ vs. CUDA by running `python benchmark.py {cpu, cuda}`,
- Run gradient checks on the code by running `python grad_check.py --backend {cpu, cuda}`.

# Requirements

This module is expected to compile for Pytorch `1.6`.

# Installation

this module is available on pip

`pip install spatial-correlation-sampler`

For a cpu-only version, you can install from source with

`python setup_cpu.py install`

# Known Problems

This module needs compatible gcc version and CUDA to be compiled.
Namely, CUDA 9.1 and below will need gcc5, while CUDA 9.2 and 10.0 will need gcc7
See [this issue](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues/1) for more information

# Usage

API has a few difference with NVIDIA's module
 * output is now a 5D tensor, which reflects the shifts horizontal and vertical.
 ```
input (B x C x H x W) -> output (B x PatchH x PatchW x oH x oW)
 ```
 * Output sizes `oH` and `oW` are no longer dependant of patch size, but only of kernel size and padding
 * Patch size `patch_size` is now the whole patch, and not only the radii.
 * `stride1` is now `stride` and`stride2` is `dilation_patch`, which behave like dilated convolutions
 * equivalent `max_displacement` is then `dilation_patch * (patch_size - 1) / 2`.
 * `dilation` is a new parameter, it acts the same way as dilated convolution regarding the correlation kernel
 * to get the right parameters for FlowNetC, you would have
 ```
kernel_size=1
patch_size=21,
stride=1,
padding=0,
dilation=1
dilation_patch=2
 ```


## Example
```python
import torch
from spatial_correlation_sampler import SpatialCorrelationSampler, 

device = "cuda"
batch_size = 1
channel = 1
H = 10
W = 10
dtype = torch.float32

input1 = torch.randint(1, 4, (batch_size, channel, H, W), dtype=dtype, device=device, requires_grad=True)
input2 = torch.randint_like(input1, 1, 4).requires_grad_(True)

#You can either use the function or the module. Note that the module doesn't contain any parameter tensor.

#function

out = spatial_correlation_sample(input1,
	                         input2,
                                 kernel_size=3,
                                 patch_size=1,
                                 stride=2,
                                 padding=0,
                                 dilation=2,
                                 dilation_patch=1)

#module

correlation_sampler = SpatialCorrelationSampler(
    kernel_size=3,
    patch_size=1,
    stride=2,
    padding=0,
    dilation=2,
    dilation_patch=1)
out = correlation_sampler(input1, input2)

```

# Benchmark

 * default parameters are from `benchmark.py`, FlowNetC parameters are same as use in `FlowNetC` with a batch size of 4, described in [this paper](https://arxiv.org/abs/1504.06852), implemented [here](https://github.com/lmb-freiburg/flownet2) and [here](https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetC.py).
 * Feel free to file an issue to add entries to this with your hardware !

## CUDA Benchmark

 * See [here](https://gist.github.com/ClementPinard/270e910147119831014932f67fb1b5ea) for a benchmark script working with [NVIDIA](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)'s code, and Pytorch.
 * Benchmark are launched with environment variable `CUDA_LAUNCH_BLOCKING` set to `1`.
 * Only `float32` is benchmarked.
 * FlowNetC correlation parameters where launched with the following command:
 
 ```bash
 CUDA_LAUNCH_BLOCKING=1 python benchmark.py --scale ms -k1 --patch 21 -s1 -p0 --patch_dilation 2 -b4 --height 48 --width 64 -c256 cuda -d float
 
 CUDA_LAUNCH_BLOCKING=1 python NV_correlation_benchmark.py --scale ms -k1 --patch 21 -s1 -p0 --patch_dilation 2 -b4 --height 48 --width 64 -c256
 ```

 | implementation | Correlation parameters |  device |     pass |      min time |      avg time |
 | -------------- | ---------------------- | ------- | -------- | ------------: | ------------: |
 |           ours |                default | 980 GTX |  forward |  **5.745 ms** |  **5.851 ms** |
 |           ours |                default | 980 GTX | backward |     77.694 ms |     77.957 ms |
 |         NVIDIA |                default | 980 GTX |  forward |     13.779 ms |     13.853 ms |
 |         NVIDIA |                default | 980 GTX | backward | **73.383 ms** | **73.708 ms** |
 |                |                        |         |          |               |               |
 |           ours |               FlowNetC | 980 GTX |  forward |  **26.102 ms** |  **26.179 ms** |
 |           ours |               FlowNetC | 980 GTX | backward | **208.091 ms** | **208.510 ms** |
 |         NVIDIA |               FlowNetC | 980 GTX |  forward |      35.363 ms |      35.550 ms |
 |         NVIDIA |               FlowNetC | 980 GTX | backward |     283.748 ms |     284.346 ms |
 
### Notes
 * The overhead of our implementation regarding `kernel_size` > 1 during backward needs some investigation, feel free to
 dive in the code to improve it !
 * The backward pass of NVIDIA is not entirely correct when stride1 > 1 and kernel_size > 1, because not everything
 is computed, see [here](https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/correlation_package/src/correlation_cuda_kernel.cu#L120).

## CPU Benchmark

  * No other implementation is avalaible on CPU.
  * It is obviously not recommended to run it on CPU if you have a GPU.

 | Correlation parameters |               device |     pass |    min time |    avg time |
 | ---------------------- | -------------------- | -------- | ----------: | ----------: |
 |                default | E5-2630 v3 @ 2.40GHz |  forward |  159.616 ms |  188.727 ms |
 |                default | E5-2630 v3 @ 2.40GHz | backward |  282.641 ms |  294.194 ms |
 |               FlowNetC | E5-2630 v3 @ 2.40GHz |  forward |  2.138 s |  2.144 s |
 |               FlowNetC | E5-2630 v3 @ 2.40GHz | backward | 7.006 s | 7.075 s |
