# Makefile for GNU Make

default: run

all: run

run: matmul
	./obj/./matmul

DPCPP_OPTS = -I${MKLROOT}/include -mkl -fsycl-device-code-split=per_kernel

matmul: src/matmul.cpp
	dpcpp $< -o ./obj/$@ $(DPCPP_OPTS)

clean:
	-rm -f matmul

.PHONY: clean run all
