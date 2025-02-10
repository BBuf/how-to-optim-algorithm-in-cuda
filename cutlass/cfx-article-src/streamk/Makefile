CUTLASS_DIR=../external/cutlass
CXX=nvcc
APP=streamk

CXXFLAGS=-std=c++17 -Xptxas=--verbose -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DNDEBUG -lineinfo

LDFLAGS=

LDLIBS=-lcuda

OBJECTS = main.o

.SUFFIXES: .o .cu

default: clean $(APP)

$(APP): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

main.o:
	$(CXX) -c $(CXXFLAGS) -o "$@" $(APP).cu

clean: 
	rm -f $(OBJECTS) $(APP)
