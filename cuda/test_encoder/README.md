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

<!-- ### encoder.cu
compile with <br>
```nvcc -w -std=c++11 encoder.cu -lcublas -o encoder``` <br>
run using run.py (for various queue sizes, fixing the batch size) <br>
```python run.py ./fine_grained_gemm 192 1024 1024``` <br> -->

