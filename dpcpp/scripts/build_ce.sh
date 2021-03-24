#!/bin/bash
source $ONEAPI_INSTALL/setvars.sh --force> /dev/null 2>&1
dpcpp -I${MKLROOT}/include -mkl -fsycl-device-code-split=per_kernel tests/copy_engines.cpp -o tests/test_ce
