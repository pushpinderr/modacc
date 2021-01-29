INCLUDE_DIR =./include
NVCC=nvcc
CFLAGS=-I$(INCLUDE_DIR)
OBJECT_DIR=./obj
SOURCE_DIR =./src
LIBS=-lcublas

profile_wallclock_gemm: ./include/common.h ./include/gemm.h ./src/fine_grained_gemm.cu
	$(NVCC) -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_gemm.cu -o $(OBJECT_DIR)/profile_wallclock_gemm $(LIBS)

profile_stream_gemm: ./include/common.h ./include/gemm.h ./src/fine_grained_gemm.cu
	$(NVCC) -DEVENT_PROFILE=1 -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_gemm.cu -o $(OBJECT_DIR)/profile_reorder $(LIBS)

profile_reorder: ./include/common.h ./include/gemm.h ./src/reorder_memory_layout.cu
	$(NVCC) -DEVENT_PROFILE=1 -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/reorder_memory_layout.cu -o $(OBJECT_DIR)/profile_reorder $(LIBS)

profile_perf_gemm_coarse: ./include/common.h ./include/gemm.h ./src/fine_grained_perf.cu
	$(NVCC) -DSEQ=1 -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_perf.cu -o $(OBJECT_DIR)/profile_perf_gemm_coarse $(LIBS)

profile_perf_gemm_fine: ./include/common.h ./include/gemm.h ./src/fine_grained_perf.cu
	$(NVCC) -DASYNC=1 -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_perf.cu -o $(OBJECT_DIR)/profile_perf_gemm_fine $(LIBS)

profile_perf_gemm_fine_all: ./include/common.h ./include/gemm.h ./src/fine_grained_perf.cu
	$(NVCC) -DASYNC=1 -DAYSNC_ALL=1 -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_perf.cu -o $(OBJECT_DIR)/profile_perf_gemm_fine_all $(LIBS)

