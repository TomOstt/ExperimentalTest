# sgemm climb

writing matrix multiply (C = A·B) from scratch on an RTX 4080 and making it fast
by climbing the memory hierarchy: get the data close to the cores and reuse it.
one idea the whole way: speed isn't doing more work, it's not waiting on memory.

build and run:
```
nvcc -allow-unsupported-compiler -std=c++17 -O3 sgemm_climb.cu -o sgemm_climb -lcublas
./sgemm_climb
```

results (N=4096, FP32):

| kernel       | GFLOPS | idea                                                      |
|--------------|--------|-----------------------------------------------------------|
| naive        |  3,052 | every thread reads A,B straight from VRAM (slow)          |
| shared       |  4,165 | block loads a tile into on-chip shared memory, reuses it  |
| **register** | **24,464** | **each thread does an 8×8 tile, operands kept in registers** |
| reg_vec      | 28,586 | + float4 vectorized loads (4 floats per instruction)      |
| reg_db       | 29,029 | + double buffering (prefetch next tile while computing)   |
| cublas       | 35,300 | NVIDIA's hand-tuned reference (the ceiling)               |

the register kernel (~50 lines) hit **24k, 8× naive**. vectorized loads added the
next real jump; double buffering added a sliver on top (the warp scheduler already
hides most of that latency for free). we land at ~62% of cublas.

the remaining gap to cublas is warptiling + bank-conflict-free layouts, a bigger
rewrite. ill keep improving it later on. 
