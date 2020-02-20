////// INFO- READ ME /////////

The code is taken from git Repository "https://github.com/LaurabelleKay/Parallel-kMeans" and than its manipulated and ready to run for FPGA. 
some header files are editted for this implementation, in the common files provided by intel hello_world example.
So merged the Idea or Flow of Intel Example and this repository code to built my version of FPGA.. 
"LaurabelleKay" code was seemed to be optimized for GPU, so still there is a lot of things to optimized it for FPGAs.


kmeans_naive: using Euclidean distance without any vectorization attribute.
kmeans_naive_m: using Manhattan distance without any vectorization attribute.

kmeans_simd4: using Euclidean distance with SIMD vectorization attribute = 4.
kmeans_simd4_m: using Manhattan distance with SIMD vectorization attribute = 4

P.S any imporvement and help in the current version will be very nice.
