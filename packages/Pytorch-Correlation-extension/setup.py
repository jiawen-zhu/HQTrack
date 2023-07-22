from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from os.path import join

CPU_ONLY = False
project_root = 'Correlation_Module'

source_files = ['correlation.cpp', 'correlation_sampler.cpp']

with open("README.md", "r") as fh:
    long_description = fh.read()


def launch_setup():
    if CPU_ONLY:
        Extension = CppExtension
        macro = []
    else:
        Extension = CUDAExtension
        source_files.append('correlation_cuda_kernel.cu')
        macro = [("USE_CUDA", None)]

    sources = [join(project_root, file) for file in source_files]

    setup(
        name='spatial_correlation_sampler',
        version="0.4.0",
        author="Clément Pinard",
        author_email="clement.pinard@ensta-paristech.fr",
        description="Correlation module for pytorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
        install_requires=['torch>=1.1', 'numpy'],
        ext_modules=[
            Extension('spatial_correlation_sampler_backend',
                      sources,
                      define_macros=macro,
                      extra_compile_args={'cxx': ['-fopenmp'], 'nvcc':[]},
                      extra_link_args=['-lgomp'])
        ],
        package_dir={'': project_root},
        packages=['spatial_correlation_sampler'],
        cmdclass={
            'build_ext': BuildExtension
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ])


if __name__ == '__main__':
    launch_setup()
