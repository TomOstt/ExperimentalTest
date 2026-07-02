# sgemm climb

build and run:
```
nvcc -allow-unsupported-compiler -std=c++17 -O3 sgemm_climb.cu -o sgemm_climb -lcublas
./sgemm_climb
```

rresults (N=4096, FP32):

| kernel   | GFLOPS | idea                              |
|----------|--------|-----------------------------------|
| naive    |  3,166 | every thread reads A,B from VRAM  |
| shared   |  4,150 | tile into on-chip shared memory   |
| **register** | **24,373** | **each thread does an 8×8 tile in registers** |
| cublas   | 37,314 | NVIDIA's hand-tuned reference     |

50-line `reg` kernel hit **24,373 GFLOPS — 7.7× the naive version and ~65% of cuBLAS.**
one lesson the whole way: speed isn't doing more work, it's not waiting on memory. ill keep improving it. 

