import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Our example needs CUTLASS. Luckily it is header-only library, so all we need to do is include
cutlass_dir = os.environ.get("CUTLASS_DIR", "")
if not os.path.isdir(cutlass_dir):
  raise Exception("Environment variable CUTLASS_DIR must point to the CUTLASS installation. Got {}".format(os.path.abspath(cutlass_dir))) 
_cutlass_include_dirs = ["tools/util/include","include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]

# Transpose include dir
cute_transpose_dir = [os.environ.get("CUTE_TRANSPOSE_DIR", "")]
if not os.path.isdir(cute_transpose_dir[0]):
  raise Exception("Environment variable CUTE_TRANSPOSE should point to the cute_transpose dir. Got {}".format(os.path.abspath(cute_transpose_dir))) 

# Set additional flags needed for compilation here
nvcc_flags=["-O3","-DNDEBUG","-std=c++17","--generate-code=arch=compute_90a,code=[sm_90a]"]
ld_flags=["cuda"]


setup(
    name='transpose_cute',
    ext_modules=[
        CUDAExtension(
                name="transpose_cute",  
                sources=["transpose_cute.cu"],
                include_dirs=cutlass_include_dirs+cute_transpose_dir,
                extra_compile_args={'nvcc': nvcc_flags},
                libraries=ld_flags),
        CUDAExtension(
                name="copy_cute",  
                sources=["copy_cute.cu"],
                include_dirs=cutlass_include_dirs+cute_transpose_dir,
                extra_compile_args={'nvcc': nvcc_flags},
                libraries=ld_flags)
   ],
    cmdclass={
        'build_ext': BuildExtension
    })
