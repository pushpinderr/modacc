## Running Instructions
### fine_grained_*.cu (fine_grained_gemm.cu, fine_grained_normalize.cu ...)
compile with <br>
```nvcc -w -std=c++11 fine_grained_gemm.cu -lcublas -o fine_grained_gemm``` <br>
or <br>
```nvcc -w -std=c++11 fine_grained_normalize.cu -lcublas -o fine_grained_gemm``` <br>
run using run.py  (for various queue sizes, fixing the batch size) <br>
```python run.py ./fine_grained_gemm 192 1024 1024``` <br>
192: batch size <br>
1024: sequence length <br>
1024: hidden size

### encoder.cu
compile with <br>
```nvcc -Xcompiler -rdynamic -lineinfo -w -std=c++11 encoder.cu -lcublas -o encoder``` <br>
run using run.py (for various queue sizes, fixing the batch size) <br>
```./encoder 32 1024 1024 4096 12 4``` <br>

profiling results <br>
![alt text](https://github.com/atharva-naik/modacc/blob/cuda/cuda/test_encoder/profiling.png?raw=true)

on the hardware: <br>
![alt text](https://github.com/atharva-naik/modacc/blob/cuda/cuda/test_encoder/hardware.png?raw=true)
