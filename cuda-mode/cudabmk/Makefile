# $LastChangedDate: 2010-07-14 11:06:31 -0400 (Wed, 14 Jul 2010) $
# $Revision: 398 $

# Demystifying GPU Microarchitecture through Microbenchmarking
# http://www.stuffedcow.net/research/cudabmk

include defines.mk

CUFILES=empty.cu clock.cu
CCFILES=main.cpp
CUFILES_sm_10=
CUFILES_sm_11=
CUFILES_sm_12=
CUFILES_sm_13=regfile.cu ksync_uint_dep128.cu pipeline.cu
CU_DEPS=instructions.h
# Hack: Because of the obj/release/%.cu_sm_13_o rule in morerules.mk, a side effect of 
# listing a .cu file under CUFILES_sm_13 is that it will do a modified compile 
# flow where it leaves all the intermediate files intact, and try to replace
# the compiled filename.cubin with filename.real_cubin and merge the custom
# cubin into the fat binary. If filename.real_cubin doesn't exist, then the only
# side effect is leaving a mess in the compile directory.

EXECUTABLE=main

all: program diverge sync icache1 icache2 icache3 icache4 cmem global shared texture2 texture4 otherstuff

# List of .ptx files to build from .cu files.
PTX_OUTPUTS= # clock.ptx pipeline.ptx


include common.mk
include morerules.mk

diverge:
	$(MAKE) -f Makefile-diverge

sync:
	$(MAKE) -f Makefile-sync

icache1:
	$(MAKE) -f Makefile-icache1

icache2:
	$(MAKE) -f Makefile-icache2

icache3:
	$(MAKE) -f Makefile-icache3
	
icache4:
	$(MAKE) -f Makefile-icache4

cmem:
	$(MAKE) -f Makefile-cmem

global:
	$(MAKE) -f Makefile-global

shared:
	$(MAKE) -f Makefile-shared

texture2:
	$(MAKE) -f Makefile-texture2
texture4:
	$(MAKE) -f Makefile-texture4



