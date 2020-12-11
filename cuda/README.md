Dependencies
------------------

+ Python 2.7
+ nvcc >=7.5
+ gcc >=4.8.2


Project Hierarchy
-----------------

<pre>
<code>

└── <b>cuda</b> (Base Folder for CUDA Experiments)
    ├── <b>include</b> (Folder containing header files)
    ├── <b>src</b> (Folder containing source files)
    ├── <b>obj</b> (Folder containing binaries)
    ├── <b>obj</b> (Folder containing binaries)
    ├── <b>scripts</b> (Folder containing bash and python scripts for experimentation and plot generation)
    ├── <b>plots</b> (Folder containing plots generated for different experiments)
    ├── <b>logs</b> (Folder containing timing logs and dumps for various data parallel applications)
    
  </code>
</pre>

Overview
-----------

The repository currently comprises the required source code for a streaming implementation of cuBLAS GEMM for evaluating the benefits of fine-grained scheduling over that of coarse-grained scheduling for the intial Linear Transformation involved in a Transformer Encoder Layer.
The main source file under question is fine_grained_gemm.cu inside src directory which takes as arguments batch_size, sequence_length and hidden_size.

Running
----------
The makefile has two options for generating executables -i) profile_wallclock_gemm for obtaining wallclock time of coarse-grained and fine-grained scheduling. and ii) profile_stream_gemm for obtaining the compute times involved for executing a gemm subtask on each stream. 

Bash scripts for running both versions of the executable are there in the scripts folder. 

A python script is also there in the scripts folder which generates the required plots.
For running and generating all the plots for i) evaluating speedups of fine-grained vs coarse-grained wallclock times and ii)comparative evaluation of stream-level compute times, execute the following command from the cuda directory.


<code>
bash scripts/generate_plots.sh batch_size sequence_length hidden_size
</code>
