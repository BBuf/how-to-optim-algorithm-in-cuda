# ROOTDIR should point to the projects directory where the NVIDIA CUDA SDK was installed.
# Since we don't depend on the CUDA SDK, this shouldn't matter.
ROOTDIR=.

# ROOTBINDIR set to override the binary output directory so they get placed here
# rather than into the CUDA SDK projects directory.
ROOTBINDIR=./bin

CUDACCFLAGS=-Xopencc -OPT:Olimit=0
