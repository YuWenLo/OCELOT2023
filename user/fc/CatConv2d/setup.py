from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='catconv2d_cuda',
    ext_modules=[
        CUDAExtension('catconv2d_cuda', [
            'catconv2d_cuda.cpp',
            'catconv2d_cuda_kernel.cu',
            ],extra_compile_args={'cxx': [],
                                  'nvcc': ['-O2', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__', '-arch=sm_75']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
