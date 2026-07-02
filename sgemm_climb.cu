// sgemm_climb.cu — climb from naive to peak. target: 47,000 GFLOPS on a 4080.
// C = A*B, row-major, square N. we measure each optimization rung.
//
// the whole story is ONE idea repeated: stop fetching from slow memory.
//   naive   : every thread reads A,B straight from VRAM (global). memory-bound.
//   shared  : block cooperatively loads a tile into SHARED memory (on-chip L1),
//             reuses it TS times. ~cache blocking, GPU edition.
//   register: each thread computes an 8x8 microtile, keeping operands in
//             REGISTERS (fastest memory there is). this is where it flies.
//   reg_vec : + float4 vectorized loads (4 floats per instruction).
//   reg_db  : + double buffering (prefetch next tile while computing current),
//             so the load units and CUDA cores work at the same time.
//   cublas  : NVIDIA's hand-tuned SGEMM = the real ceiling.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef N
#define N 4096
#endif

// ---------- rung 1: naive ----------
__global__ void naive(const float*A,const float*B,float*C){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(row<N&&col<N){ float s=0;
        for(int k=0;k<N;k++) s+=A[row*N+k]*B[k*N+col];
        C[row*N+col]=s; }
}

// ---------- rung 2: shared-memory tiling ----------
#define TS 32
__global__ void shared(const float*A,const float*B,float*C){
    __shared__ float As[TS][TS],Bs[TS][TS];
    int row=blockIdx.y*TS+threadIdx.y, col=blockIdx.x*TS+threadIdx.x;
    float acc=0;
    for(int t=0;t<N;t+=TS){
        As[threadIdx.y][threadIdx.x]=A[row*N+t+threadIdx.x];
        Bs[threadIdx.y][threadIdx.x]=B[(t+threadIdx.y)*N+col];
        __syncthreads();
        for(int k=0;k<TS;k++) acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row*N+col]=acc;
}

// ---------- rung 3: 2D register blocking (each thread does an 8x8 tile) ----------
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__global__ void reg(const float*A,const float*B,float*C){
    const int cRow=blockIdx.y, cCol=blockIdx.x;
    const int tCol=threadIdx.x%(BN/TN), tRow=threadIdx.x/(BN/TN);
    __shared__ float As[BM*BK], Bs[BK*BN];
    A+=cRow*BM*N; B+=cCol*BN; C+=cRow*BM*N+cCol*BN;
    const int iRowA=threadIdx.x/BK, iColA=threadIdx.x%BK;
    const int iRowB=threadIdx.x/BN, iColB=threadIdx.x%BN;
    float acc[TM*TN]={0.0f}, rM[TM], rN[TN];
    for(int bk=0;bk<N;bk+=BK){
        for(int o=0;o<BM;o+=32) As[(iRowA+o)*BK+iColA]=A[(iRowA+o)*N+iColA];
        for(int o=0;o<BK;o+=2)  Bs[(iRowB+o)*BN+iColB]=B[(iRowB+o)*N+iColB];
        __syncthreads();
        A+=BK; B+=BK*N;
        for(int d=0;d<BK;d++){
            for(int i=0;i<TM;i++) rM[i]=As[(tRow*TM+i)*BK+d];
            for(int j=0;j<TN;j++) rN[j]=Bs[d*BN+tCol*TN+j];
            for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i*TN+j]+=rM[i]*rN[j];
        }
        __syncthreads();
    }
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j++)
        C[(tRow*TM+i)*N+tCol*TN+j]=acc[i*TN+j];
}

// ---------- rung 4: + float4 vectorized loads (As stored transposed) ----------
__global__ void reg_vec(const float*A,const float*B,float*C){
    const int cRow=blockIdx.y, cCol=blockIdx.x;
    const int tCol=threadIdx.x%(BN/TN), tRow=threadIdx.x/(BN/TN);
    __shared__ float As[BK*BM], Bs[BK*BN];          // As transposed: [k][row]
    A+=cRow*BM*N; B+=cCol*BN; C+=cRow*BM*N+cCol*BN;
    const int iRowA=threadIdx.x/(BK/4), iColA=threadIdx.x%(BK/4);
    const int iRowB=threadIdx.x/(BN/4), iColB=threadIdx.x%(BN/4);
    float acc[TM*TN]={0.0f}, rM[TM], rN[TN];
    for(int bk=0;bk<N;bk+=BK){
        float4 a=*reinterpret_cast<const float4*>(&A[iRowA*N+iColA*4]);
        As[(iColA*4+0)*BM+iRowA]=a.x; As[(iColA*4+1)*BM+iRowA]=a.y;
        As[(iColA*4+2)*BM+iRowA]=a.z; As[(iColA*4+3)*BM+iRowA]=a.w;
        *reinterpret_cast<float4*>(&Bs[iRowB*BN+iColB*4])=
            *reinterpret_cast<const float4*>(&B[iRowB*N+iColB*4]);
        __syncthreads();
        A+=BK; B+=BK*N;
        for(int d=0;d<BK;d++){
            for(int i=0;i<TM;i++) rM[i]=As[d*BM+tRow*TM+i];
            for(int j=0;j<TN;j++) rN[j]=Bs[d*BN+tCol*TN+j];
            for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i*TN+j]+=rM[i]*rN[j];
        }
        __syncthreads();
    }
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j+=4)
        *reinterpret_cast<float4*>(&C[(tRow*TM+i)*N+tCol*TN+j])=
            *reinterpret_cast<float4*>(&acc[i*TN+j]);
}

// ---------- rung 5: + double buffering (prefetch next tile while computing) ----------
__global__ void reg_db(const float*A,const float*B,float*C){
    const int cRow=blockIdx.y, cCol=blockIdx.x;
    const int tCol=threadIdx.x%(BN/TN), tRow=threadIdx.x/(BN/TN);
    __shared__ float As[2][BK*BM], Bs[2][BK*BN];    // TWO shelves, ping-pong
    A+=cRow*BM*N; B+=cCol*BN; C+=cRow*BM*N+cCol*BN;
    const int iRowA=threadIdx.x/(BK/4), iColA=threadIdx.x%(BK/4);
    const int iRowB=threadIdx.x/(BN/4), iColB=threadIdx.x%(BN/4);
    float acc[TM*TN]={0.0f}, rM[TM], rN[TN];

    #define LOAD(buf) {                                                        \
        float4 a=*reinterpret_cast<const float4*>(&A[iRowA*N+iColA*4]);        \
        As[buf][(iColA*4+0)*BM+iRowA]=a.x; As[buf][(iColA*4+1)*BM+iRowA]=a.y;  \
        As[buf][(iColA*4+2)*BM+iRowA]=a.z; As[buf][(iColA*4+3)*BM+iRowA]=a.w;  \
        *reinterpret_cast<float4*>(&Bs[buf][iRowB*BN+iColB*4])=               \
            *reinterpret_cast<const float4*>(&B[iRowB*N+iColB*4]); }
    #define COMPUTE(buf) for(int d=0;d<BK;d++){                               \
        for(int i=0;i<TM;i++) rM[i]=As[buf][d*BM+tRow*TM+i];                  \
        for(int j=0;j<TN;j++) rN[j]=Bs[buf][d*BN+tCol*TN+j];                  \
        for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i*TN+j]+=rM[i]*rN[j]; }

    LOAD(0); A+=BK; B+=BK*N; __syncthreads();        // prologue: fill shelf 0
    int buf=0;
    for(int bk=BK;bk<N;bk+=BK){
        LOAD(buf^1); A+=BK; B+=BK*N;                  // prefetch next -> other shelf
        COMPUTE(buf);                                 // compute current, in parallel
        __syncthreads();
        buf^=1;
    }
    COMPUTE(buf);                                     // epilogue: last shelf
    #undef LOAD
    #undef COMPUTE
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j+=4)
        *reinterpret_cast<float4*>(&C[(tRow*TM+i)*N+tCol*TN+j])=
            *reinterpret_cast<float4*>(&acc[i*TN+j]);
}

static float* devrand(size_t n){
    float*h=(float*)malloc(n*4); for(size_t i=0;i<n;i++) h[i]=(i%13)*0.1f;
    float*d; cudaMalloc(&d,n*4); cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice);
    free(h); return d;
}
// time a launch (avg of 5, after warmup); returns GFLOPS
static double bench(const char*name,void(*run)(void)){
    run(); cudaDeviceSynchronize();               // warmup
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i=0;i<5;i++) run();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b); ms/=5;
    double g=2.0*N*N*N/(ms/1e3)/1e9;
    printf("%-9s %8.2f ms   %9.1f GFLOPS   (%.1f%% of 47000)\n",name,ms,g,100*g/47000);
    return g;
}

static float *dA,*dB,*dC; static cublasHandle_t H;
static void r_naive(void){ dim3 b(16,16),g((N+15)/16,(N+15)/16); naive<<<g,b>>>(dA,dB,dC);}
static void r_shared(void){ dim3 b(TS,TS),g(N/TS,N/TS); shared<<<g,b>>>(dA,dB,dC);}
static void r_reg(void){ dim3 g(N/BN,N/BM); reg<<<g,(BM*BN)/(TM*TN)>>>(dA,dB,dC);}
static void r_vec(void){ dim3 g(N/BN,N/BM); reg_vec<<<g,(BM*BN)/(TM*TN)>>>(dA,dB,dC);}
static void r_db(void){ dim3 g(N/BN,N/BM); reg_db<<<g,(BM*BN)/(TM*TN)>>>(dA,dB,dC);}
static void r_cublas(void){ float al=1,be=0;
    // row-major C=A*B  ==  column-major C = B*A with args swapped
    cublasSgemm(H,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&al,dB,N,dA,N,&be,dC,N);}

int main(void){
    size_t n=(size_t)N*N;
    dA=devrand(n); dB=devrand(n); cudaMalloc(&dC,n*4);
    cublasCreate(&H);
    printf("N=%d   peak FP32 (RTX 4080) ~= 48,700 GFLOPS\n\n",N);
    bench("naive",  r_naive);
    bench("shared", r_shared);
    bench("register",r_reg);
    bench("reg_vec",r_vec);
    bench("reg_db", r_db);
    bench("cublas", r_cublas);
    return 0;
}
