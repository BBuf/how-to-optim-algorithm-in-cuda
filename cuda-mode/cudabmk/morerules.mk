# Build the program
program: $(TARGET)

# Build other stuff, like PTX dumps of .cu files
otherstuff: $(PTX_OUTPUTS)

export CUDA_INSTALL_PATH



obj/release/%.cu_sm_13_o : %.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $@ -c $< -keep $(NVCCFLAGS) -arch sm_13
	$(VERBOSE)sh build_cubin $<


# Rule for making .ptx file from .cu file.
%.ptx : %.cu
	$(VERBOSE)$(NVCC) --ptx $(NVCCFLAGS) $< -o $@
# Run ptxas -v to print out stats on the PTX
#	$(VERBOSE)ptxas -v $@ -o /dev/null
