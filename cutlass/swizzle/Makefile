STATS := l1tex__data_bank_conflicts_pipe_lsu.max
# STATS := $(STATS)l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld.max,l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st.max
STATS := $(STATS),sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max,sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.max,gpu__time_active.max 
TARGET = main
all:
	@cd build && make
run: 
	@./build/$(TARGET)
ncu:
	ncu --metrics $(STATS) build/$(TARGET)