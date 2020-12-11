INCLUDE_DIR =./include
NVCC=nvcc
CFLAGS=-I$(INCLUDE_DIR)
OBJECT_DIR=./obj
SOURCE_DIR =./src
LIBS=-lcublas

profile_wallclock_gemm: ./include/common.h ./include/gemm.h ./src/fine_grained_gemm.cu
	$(NVCC) -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_gemm.cu -o $(OBJECT_DIR)/profile_wallclock_gemm $(LIBS)

profile_stream_gemm: ./include/common.h ./include/gemm.h ./src/fine_grained_gemm.cu
	$(NVCC) -DEVENT_PROFILE=1 -I $(INCLUDE_DIR)/ $(SOURCE_DIR)/fine_grained_gemm.cu -o $(OBJECT_DIR)/profile_stream_gemm $(LIBS)
