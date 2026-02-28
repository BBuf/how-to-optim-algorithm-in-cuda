CUTLASS_DIR=/home/ryo/cutlass
REPO_DIR=..
CXX=nvcc
APPS=benchmark
#APPS=collective_builder_original
#APPS=48_test

CXXFLAGS=-std=c++17 -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${REPO_DIR}/include/utils --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -DNDEBUG

LDFLAGS=

LDLIBS=-lcuda

.SUFFIXES: .cu

default: $(APPS)

$(APPS): % : %.cu
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean: 
	rm -f $(APPS)
