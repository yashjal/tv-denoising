#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

struct RGBImage {
  long Xsize;
  long Ysize;
  float* A;
};
void read_image(const char* fname, RGBImage* I) {
  I->Xsize = 0;
  I->Ysize = 0;
  I->A = NULL;

  FILE* f = fopen(fname, "rb");
  if (f == NULL) return;
  fscanf(f, "P6\n%d %d\n255\n", &I->Ysize, &I->Xsize);
  long N = I->Xsize * I->Ysize;
  if (N) {
    I->A = (float*) malloc(3*N * sizeof(float));
    unsigned char* I0 = (unsigned char*) malloc(3*N * sizeof(unsigned char));
    fread(I0, sizeof(unsigned char), 3*N, f);
    for (long i0 = 0; i0 < N; i0++) {
      for (long i1 = 0; i1 < 3; i1++) {
        I->A[i1*N+i0] = I0[i0*3+i1];
      }
    }
    free(I0);
  }
  fclose(f);
}
void write_image(const char* fname, const RGBImage I) {
  long N = I.Xsize * I.Ysize;
  if (!N) return;

  FILE* f = fopen(fname, "wb");
  if (f == NULL) return;
  fprintf(f, "P6\n%d %d\n255\n", I.Ysize, I.Xsize);
  unsigned char* I0 = (unsigned char*) malloc(3*N * sizeof(unsigned char));
  for (long i0 = 0; i0 < 3; i0++) {
    for (long i1 = 0; i1 < N; i1++) {
      I0[i1*3+i0] = I.A[i0*N+i1];
    }
  }
  fwrite(I0, sizeof(unsigned char), 3*N, f);
  free(I0);
  fclose(f);
}
void free_image(RGBImage* I) {
  long N = I->Xsize * I->Ysize;
  if (N) free(I->A);
  I->A = NULL;
}

#define BLOCK_DIM 32

__global__ void norm_upd(float* du, float* hf, float* u, float* f, float eps, float del, float h, long Xsize, long Ysize) {
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;

  if (idx > 0 && idy > 0 && idx < Xsize && idy < Ysize) {
    float ux = (u[idx*Ysize+idy] - u[(idx-1)*Ysize+idy])/h;
    float uy = (u[idx*Ysize+idy] - u[idx*Ysize+(idy-1)])/h;
    du[idx*Ysize+idy] = sqrt(ux*ux + uy*uy + eps);
    float uf = u[idx*Ysize+idy]-f[idx*Ysize+idy];
    hf[idx*Ysize+idy] = sqrt(uf*uf + del);
  } 
 
}

__global__ void GPU_jacobi(float* u0, float* u1, float *f, long Xsize, long Ysize, float h, float* du, float* hf, float lambda) {
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  
  if (idx > 0 && idx < Xsize-1 && idy > 0 && idy < Ysize-1) {
    u1[idx*Ysize+idy] = ((u0[(idx+1)*Ysize+idy] + u0[(idx)*Ysize+(idy+1)] + u0[(idx-1)*Ysize+idy] + u0[idx*Ysize+(idy-1)])/(du[idx*Ysize+idy])+ h*h*lambda*f[idx*Ysize+idy]/hf[idx*Ysize+idy]) / (4/du[idx*Ysize+idy] + h*h*lambda/hf[idx*Ysize+idy]);
  }
  
  __syncthreads();
  
  u0[idx*Ysize+idy] = u1[idx*Ysize+ysize];
}

int main() {
  long repeat = 500;
  long T = 100; // total variation 
  long N = 100; // jacobi
  float eps = 1e-2;
  float del = 1e-2;
  float lambda = 1; 

  const char fname[] = "bike.ppm";

  // Load image from file
  RGBImage u0, f; //I1_ref;
  read_image(fname, &u0);
  read_image(fname, &f);
  //read_image(fname, &I1_ref);
  long Xsize = u0.Xsize;
  long Ysize = u0.Ysize;
  float h = 1.0/Xsize;
  // denoise on CPU
  Timer t;
  //t.tic();
  //for (long i = 0; i < repeat; i++) CPU_convolution(I1_ref.A, I0.A, Xsize, Ysize);
  //double tt = t.toc();
  //printf("CPU time = %fs\n", tt);
  //printf("CPU flops = %fGFlop/s\n", repeat * 2*(Xsize-FWIDTH)*(Ysize-FWIDTH)*FWIDTH*FWIDTH/tt*1e-9);

  // Allocate GPU memory
  float *u0gpu, *fgpu, *u1gpu, *dugpu, *hfgpu;
  cudaMalloc(&u0gpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&fgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&u1gpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&dugpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&hfgpu, 3*Xsize*Ysize*sizeof(float));
 
  cudaMemcpy(u0gpu, u0.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(fgpu, f.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);

  // Create streams
  cudaStream_t streams[3];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);
  cudaStreamCreate(&streams[2]);

  // Dry run
  /*
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim(Xsize/(BLOCK_DIM-FWIDTH)+1, Ysize/(BLOCK_DIM-FWIDTH)+1);
  GPU_convolution<<<gridDim,blockDim, 0, streams[0]>>>(I1gpu+0*Xsize*Ysize, I0gpu+0*Xsize*Ysize, Xsize, Ysize);
  GPU_convolution<<<gridDim,blockDim, 0, streams[1]>>>(I1gpu+1*Xsize*Ysize, I0gpu+1*Xsize*Ysize, Xsize, Ysize);
  GPU_convolution<<<gridDim,blockDim, 0, streams[2]>>>(I1gpu+2*Xsize*Ysize, I0gpu+2*Xsize*Ysize, Xsize, Ysize);
  */

  // denoise on GPU
  cudaDeviceSynchronize();
  t.tic();
  for (long n = 0; n < T; n++) {
    norm_upd<<<gridDim,blockDim, 0, streams[0]>>>(dugpu+0*Xsize*Ysize, hfgpu+0*Xsize*Ysize, u0gpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, eps, del, h, Xsize, Ysize);
    norm_upd<<<gridDim,blockDim, 1, streams[1]>>>(dugpu+1*Xsize*Ysize, hfgpu+1*Xsize*Ysize, u0gpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, eps, del, h, Xsize, Ysize);
    norm_upd<<<gridDim,blockDim, 2, streams[2]>>>(dugpu+2*Xsize*Ysize, hfgpu+2*Xsize*Ysize, u0gpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, eps, del, h, Xsize, Ysize);

    for (long k = 0; k < N; k++) {
      GPU_jacobi<<<gridDim,blockDim, 0, streams[0]>>>(u0gpu+0*Xsize*Ysize, u1gpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, Xsize, Ysize, h, dugpu+0*Xsize*Ysize, hf+0*Xsize*Ysize, lambda);
      GPU_jacobi<<<gridDim,blockDim, 1, streams[1]>>>(u0gpu+1*Xsize*Ysize, u1gpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, Xsize, Ysize, h, dugpu+1*Xsize*Ysize, hf+1*Xsize*Ysize, lambda);
      GPU_jacobi<<<gridDim,blockDim, 2, streams[2]>>>(u0gpu+2*Xsize*Ysize, u1gpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, Xsize, Ysize, h, dugpu+2*Xsize*Ysize, hf+2*Xsize*Ysize, lambda);
    }
  }
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU time = %fs\n", tt);

  // Print error
  /*
  float err = 0;
  cudaMemcpy(I1.A, I1gpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
  for (long i = 0; i < 3*Xsize*Ysize; i++) err = std::max(err, fabs(I1.A[i] - I1_ref.A[i]));
  printf("Error = %e\n", err);
  */

  // Write output
  // write_image("CPU.ppm", I1_ref);
  write_image("GPU.ppm", I1);

  // Free memory
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  cudaStreamDestroy(streams[2]);
  cudaFree(I0gpu);
  cudaFree(I1gpu);
  free_image(&I0);
  free_image(&I1);
  free_image(&I1_ref);
  return 0;
}

