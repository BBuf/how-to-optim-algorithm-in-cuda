CUTLASS_DIR=../external/cutlass
REPO_DIR=..
CXX=nvcc
APPS=evt_gemm_cute 

CXXFLAGS=-std=c++17 -Xptxas=--verbose -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${REPO_DIR}/include/utils --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a

LDFLAGS= 

LDLIBS=-lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl
.SUFFIXES: .cu

default: clean $(APPS)

$(APPS): % : %.cu
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean: 
	rm -f $(APPS)
