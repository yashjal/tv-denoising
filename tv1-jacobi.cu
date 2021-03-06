#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include <stdlib.h>

struct RGBImage {
  long Xsize;
  long Ysize;
  float* A;
};

/*
sample from normal dist 
*/
float randn (float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (float) X2);
    }
 
  do
    {
      U1 = -1 + ((float) rand () / RAND_MAX) * 2;
      U2 = -1 + ((float) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (float) X1);
}

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

double norm(float *err, long Xsize, long Ysize) {
  float sum = 0;
  //#pragma omp parallel for reduction (+:sum)
  for (long i = 0; i < Xsize*Ysize; i+=1) {
     sum += err[i]*err[i];
  }
  return sqrt(sum);
}

/*
kernel for computing du = |\nabla u| and hf = |u-f|
*/
__global__ void norm_upd(float* du, float* hf, float* u, float* f, float* err, float eps, float del, float h, long Xsize, long Ysize) {
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  
  int idx2 = (idx-1)*Ysize+idy;
  int idx1 = (idx+1)*Ysize+idy;
  int idy2 = idx*Ysize+(idy-1);
  int idy1 = idx*Ysize+(idy+1);

  // Neumann BCs handled symmetrically
  if (idx == 0) {
    idx2 = (idx+1)*Ysize+idy;
  }
  if (idy == 0) {
    idy2 = idx*Ysize+(idy+1);
  }
  if (idx == Xsize-1) {
    idx1 = (idx-1)*Ysize+idy;
  }
  if (idy == Ysize-1) {
    idy1 = idx*Ysize+(idy-1);
  }

  if (idx < Xsize && idy < Ysize) {
    float ux = (u[idx1] - u[idx2])/(2*h);
    float uy = (u[idy1] - u[idy2])/(2*h);
    du[idx*Ysize+idy] = sqrt(ux*ux + uy*uy + eps);
  }
 
  float uf = u[idx*Ysize+idy]-f[idx*Ysize+idy];
  hf[idx*Ysize+idy] = sqrt(uf*uf + del);
  
 //__syncthreads();
 //err[idx*Ysize+idy] = du[idx*Ysize+idy] + hf[idx*Ysize+idy];
}

/*
2D Jacobi using four point stencil for centered diff. discretized TV-L1 PDE
with homo Neumann BCs 
*/
__global__ void GPU_jacobi(float* u0, float* u1, float *f, float* err, long Xsize, long Ysize, float h, float* du, float* hf, float lambda) {
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
 
  int idx2 = (idx-1)*Ysize+idy;
  int idx1 = (idx+1)*Ysize+idy;
  int idy2 = idx*Ysize+(idy-1);
  int idy1 = idx*Ysize+(idy+1);
  
  // indices for the boundary values
  if (idx == 0) {
    idx2 = (idx+1)*Ysize+idy;
  }
  if (idy == 0) {
    idy2 = idx*Ysize+(idy+1);
  }
  if (idx == Xsize-1) {
    idx1 = (idx-1)*Ysize+idy;
  }
  if (idy == Ysize-1) {
    idy1 = idx*Ysize+(idy-1);
  }
  
  // jacobi step
  if (idx < Xsize && idy < Ysize) {
    u1[idx*Ysize+idy] = ((u0[idx1] + u0[idy1] + u0[idx2] + u0[idy2])/(du[idx*Ysize+idy])+ h*h*lambda*f[idx*Ysize+idy]/hf[idx*Ysize+idy]) / (4/du[idx*Ysize+idy] + h*h*lambda/hf[idx*Ysize+idy]);
  }
 
  __syncthreads();
  if (idx < Xsize && idy < Ysize) {
    u0[idx*Ysize+idy] = u1[idx*Ysize+idy];
  }
  
  /*
  __syncthreads();
  if (idx > 0 && idx < Xsize-1 && idy > 0 && idy < Ysize-1) {
    err[idx*Ysize+idy] = (-u0[(idx-1)*Ysize+idy] - u0[idx*Ysize+(idy-1)] + 4*u0[idx*Ysize+idy] - u0[(idx+1)*Ysize+idy] - u0[idx*Ysize+(idy+1)])/(h*h)/du[idx*Ysize+idy] + lambda*(u0[idx*Ysize+idy]-f[idx*Ysize+idy])/hf[idx*Ysize+idy];
  }
  */
}

/*
kernel for computing du = |\nabla u| and hf = |u-f|
using shared memory
*/
__global__ void norm_upd_smem(float* du, float* hf, float* u, float* f, float* err, float eps, float del, float h, long Xsize, long Ysize) {
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  __shared__ float su[BLOCK_DIM+2][BLOCK_DIM+2];

  int idx2 = (idx-1)*Ysize+idy;
  int idx1 = (idx+1)*Ysize+idy;
  int idy2 = idx*Ysize+(idy-1);
  int idy1 = idx*Ysize+(idy+1);

  // Neumann BCs handled symmetrically
  if (idx == 0) {
    idx2 = (idx+1)*Ysize+idy;
  }
  if (idy == 0) {
    idy2 = idx*Ysize+(idy+1);
  }
  if (idx == Xsize-1) {
    idx1 = (idx-1)*Ysize+idy;
  }
  if (idy == Ysize-1) {
    idy1 = idx*Ysize+(idy-1);
  }
  
  // ghost cells for inner shared mem blocks 
  if (blockIdx.x > 0 && threadIdx.x == 0){
    su[0][threadIdx.y+1] = u[idx2]; //u[((blockIdx.x - 1) *BLOCK_DIM + BLOCK_DIM - 1)*Ysize + idy];
  }
  if (blockIdx.y > 0 && threadIdx.y == 0){
    su[threadIdx.x + 1][0] = u[idy2]; //u[idx*Ysize + (blockIdx.y-1)*BLOCK_DIM + BLOCK_DIM - 1 ];
  }
  if(blockIdx.x < Xsize/BLOCK_DIM && threadIdx.x == BLOCK_DIM -1){
    su[BLOCK_DIM + 1][threadIdx.y+1] = u[idx1]; //u0[(blockIdx.x+1)*BLOCK_DIM + idy];
  }
  if(blockIdx.y < Ysize/BLOCK_DIM && threadIdx.y == BLOCK_DIM -1){
    su[threadIdx.x+1][BLOCK_DIM+1] = u[idy1]; //u0[idx*Ysize + (blockIdx.y + 1)*BLOCK_DIM];
  }

  su[threadIdx.x+1][threadIdx.y+1] = u[idx*Ysize+idy];

  __syncthreads();

  if (idx < Xsize && idy < Ysize) {
    float ux = (su[threadIdx.x+2][threadIdx.y+1] -su[threadIdx.x][threadIdx.y+1])/(2*h);
    float uy = (su[threadIdx.x+1][threadIdx.y+2] -su[threadIdx.x+1][threadIdx.y])/(2*h);
    du[idx*Ysize+idy] = sqrt(ux*ux + uy*uy + eps);
    float uf = su[threadIdx.x + 1][threadIdx.y+1]- f[idx*Ysize+idy];
    hf[idx*Ysize+idy] = sqrt(uf*uf + del);
  }
  
  //__syncthreads();
  //err[idx*Ysize+idy] = du[idx*Ysize+idy] + hf[idx*Ysize+idy];
}

/*
2D Jacobi using four point stencil for centered diff. discretized TV-L1 PDE, using shared memory 
with homo Neumann BCs
*/
__global__ void GPU_jacobi_smem(float* u0, float *f, float* err, long Xsize, long Ysize, float h, float* du, float* hf, float lambda) {
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  __shared__ float su0[BLOCK_DIM+2][BLOCK_DIM+2];
  __shared__ float su1[BLOCK_DIM+2][BLOCK_DIM+2];

  int idx2 = (idx-1)*Ysize+idy;
  int idx1 = (idx+1)*Ysize+idy;
  int idy2 = idx*Ysize+(idy-1);
  int idy1 = idx*Ysize+(idy+1);
  
  // indices for boundary values
  if (idx == 0) {
    idx2 = (idx+1)*Ysize+idy;
  }
  if (idy == 0) {
    idy2 = idx*Ysize+(idy+1);
  }
  if (idx == Xsize-1) {
    idx1 = (idx-1)*Ysize+idy;
  } 
  if (idy == Ysize-1) {
    idy1 = idx*Ysize+(idy-1);
  }

  // ghost cells for inner shared mem blocks
  if(blockIdx.x > 0 && threadIdx.x == 0 ){
    su0[0][threadIdx.y+1] = u0[idx2];
  }
  if(blockIdx.y > 0 && threadIdx.y == 0){
    su0[threadIdx.x+1][0] = u0[idy2];
  }
  if(blockIdx.x < Xsize/BLOCK_DIM && threadIdx.x == BLOCK_DIM -1){
    su0[BLOCK_DIM + 1][threadIdx.y+1] = u0[idx1];//u0[(blockIdx.x+1)*BLOCK_DIM + idy];
  }
  if(blockIdx.y < Ysize/BLOCK_DIM && threadIdx.y == BLOCK_DIM -1){
    su0[threadIdx.x+1][BLOCK_DIM+1] = u0[idy1];//u0[idx*Ysize + (blockIdx.y + 1)*BLOCK_DIM];
  }

  su0[threadIdx.x+1][threadIdx.y+1] = u0[idx*Ysize + idy];

  __syncthreads();

  // jacobi step
  if (idx < Xsize && idy < Ysize) {
    float ldu = du[idx*Ysize + idy];
    float lhf = hf[idx*Ysize + idy]; 
    su1[threadIdx.x+1][threadIdx.y+1] = ((su0[threadIdx.x+2][threadIdx.y+1] + su0[threadIdx.x+1][threadIdx.y+2] + su0[threadIdx.x][threadIdx.y+1] + su0[threadIdx.x+1][threadIdx.y])/ldu + h*h*lambda*f[idx*Ysize+idy]/lhf) / (4/ldu + h*h*lambda/lhf);
  }
  
  __syncthreads();
   if (idx < Xsize && idy < Ysize){ 
     u0[idx*Ysize+idy] = su1[threadIdx.x+1][threadIdx.y+1];
   }

  /*
  __syncthreads();
  if (idx > 0 && idx < Xsize-1 && idy > 0 && idy < Ysize-1) {
    err[idx*Ysize+idy] = (-u0[(idx-1)*Ysize+idy] - u0[idx*Ysize+(idy-1)] + 4*u0[idx*Ysize+idy] - u0[(idx+1)*Ysize+idy] - u0[idx*Ysize+(idy+1)])/(h*h)/du[idx*Ysize+idy] + lambda*(u0[idx*Ysize+idy]-f[idx*Ysize+idy])/hf[idx*Ysize+idy];
  }
  */
}


int main(int argc, char * argv[] ) {
  long T = 250; // total variation iters 
  long N = 20; // jacobi iters
  float eps = 1e-4;
  float del = 1e-4;
  float lambda = 2.0; 
  float mu = 0;
  float sigma = 50;
  const char fname[] = "car.ppm";
  
  //sscanf(argv[1],"%d",&T);
  //sscanf(argv[2],"%d",&N);
  //sscanf(argv[3],"%f",&lambda);
  //sscanf(argv[4],"%f",&sigma);
   
  // Load image from file
  RGBImage u0, unoise;
  read_image(fname, &u0);
 
  long Xsize = u0.Xsize;
  long Ysize = u0.Ysize;
  unoise.Xsize = Xsize;
  unoise.Ysize = Ysize;
  float h = 1.0;
  unoise.A = (float*) malloc(3*Xsize*Ysize*sizeof(float));  
 
  // add noise to clean image 
  for(int c = 0; c < 3; c++){
    for(int i = 0; i < Xsize; i++){
      for(int j = 0; j < Ysize; j++) {
        unoise.A[c*Xsize*Ysize+ i*Ysize + j] = u0.A[c*Xsize*Ysize + i*Ysize + j] + randn(mu,sigma);
      }
    }
  }

  write_image("tvl1-noise.ppm",unoise);
 
  Timer t;
  // Allocate GPU memory
  float *u0gpu, *u0smem, *fgpu, *u1gpu, *dugpu, *hfgpu, *errgpu, *err;
  cudaMalloc(&u0smem, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&u0gpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&fgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&u1gpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&dugpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&hfgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&errgpu, 3*Xsize*Ysize*sizeof(float));
  err = (float*)malloc(3*Xsize*Ysize*sizeof(float));
 
  cudaMemcpy(u0gpu, unoise.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(fgpu, unoise.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(u0smem, unoise.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);

  // Create streams
  cudaStream_t streams[3];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);
  cudaStreamCreate(&streams[2]);

  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim(Xsize/BLOCK_DIM+1, Ysize/BLOCK_DIM+1);

  cudaDeviceSynchronize();
  t.tic();
  
  for (long n = 0; n < T; n++) {
    
    norm_upd<<<gridDim,blockDim, 0, streams[0]>>>(dugpu+0*Xsize*Ysize, hfgpu+0*Xsize*Ysize, u0gpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, errgpu+0*Xsize*Ysize, eps, del, h, Xsize, Ysize);
    norm_upd<<<gridDim,blockDim, 1, streams[1]>>>(dugpu+1*Xsize*Ysize, hfgpu+1*Xsize*Ysize, u0gpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, errgpu+1*Xsize*Ysize, eps, del, h, Xsize, Ysize);
    norm_upd<<<gridDim,blockDim, 2, streams[2]>>>(dugpu+2*Xsize*Ysize, hfgpu+2*Xsize*Ysize, u0gpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, errgpu+2*Xsize*Ysize, eps, del, h, Xsize, Ysize);

    /*
    cudaMemcpy(err, errgpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
    float norm_err = norm(err,Xsize,Ysize);
    printf("TV iters: %d, err: %f\n", n, norm_err); 
    */
    
    // approx. solve linear system arising out of discretized pde
    for (long k = 0; k < N; k++) {

      GPU_jacobi<<<gridDim,blockDim, 0, streams[0]>>>(u0gpu+0*Xsize*Ysize, u1gpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, errgpu+0*Xsize*Ysize, Xsize, Ysize, h, dugpu+0*Xsize*Ysize, hfgpu+0*Xsize*Ysize, lambda);
      GPU_jacobi<<<gridDim,blockDim, 1, streams[1]>>>(u0gpu+1*Xsize*Ysize, u1gpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, errgpu+1*Xsize*Ysize, Xsize, Ysize, h, dugpu+1*Xsize*Ysize, hfgpu+1*Xsize*Ysize, lambda);
      GPU_jacobi<<<gridDim,blockDim, 2, streams[2]>>>(u0gpu+2*Xsize*Ysize, u1gpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, errgpu+2*Xsize*Ysize, Xsize, Ysize, h, dugpu+2*Xsize*Ysize, hfgpu+2*Xsize*Ysize, lambda);
      
      /* 
      if (k%2 == 0) {
        cudaMemcpy(err, errgpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
        float norm_err = norm(err,Xsize,Ysize);
        printf("Jacobi iters: %d, err: %f\n", k, norm_err);
      }
      */
    }
  }

  cudaDeviceSynchronize();
  double tt = t.toc();
  printf("non-smem GPU time = %fs\n", tt);
  printf("GPU Bandwidth %f GB/s \n", (12*N + 8)*Xsize*Ysize*T*sizeof(float)/tt/1e9);
  cudaMemcpy(u0.A, u0gpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
 
  // Write output
  write_image("tvl1-nsmem.ppm", u0);

  cudaDeviceSynchronize();
  t.tic();

  for (long n = 0; n < T; n++) {

    norm_upd_smem<<<gridDim,blockDim, 0, streams[0]>>>(dugpu+0*Xsize*Ysize, hfgpu+0*Xsize*Ysize, u0smem+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, errgpu+0*Xsize*Ysize, eps, del, h, Xsize, Ysize);
    norm_upd_smem<<<gridDim,blockDim, 1, streams[1]>>>(dugpu+1*Xsize*Ysize, hfgpu+1*Xsize*Ysize, u0smem+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, errgpu+1*Xsize*Ysize, eps, del, h, Xsize, Ysize);
    norm_upd_smem<<<gridDim,blockDim, 2, streams[2]>>>(dugpu+2*Xsize*Ysize, hfgpu+2*Xsize*Ysize, u0smem+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, errgpu+2*Xsize*Ysize, eps, del, h, Xsize, Ysize);

    /*
    cudaMemcpy(err, errgpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
    float norm_err = norm(err,Xsize,Ysize);
    printf("TV iters: %d, err: %f\n", n, norm_err);
    */

    // approx. solve linear system arising out of discretized pde
    for (long k = 0; k < N; k++) {

      GPU_jacobi_smem<<<gridDim,blockDim, 0, streams[0]>>>(u0smem+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, errgpu+0*Xsize*Ysize, Xsize, Ysize, h, dugpu+0*Xsize*Ysize, hfgpu+0*Xsize*Ysize, lambda);
      GPU_jacobi_smem<<<gridDim,blockDim, 1, streams[1]>>>(u0smem+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, errgpu+1*Xsize*Ysize, Xsize, Ysize, h, dugpu+1*Xsize*Ysize, hfgpu+1*Xsize*Ysize, lambda);
      GPU_jacobi_smem<<<gridDim,blockDim, 2, streams[2]>>>(u0smem+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, errgpu+2*Xsize*Ysize, Xsize, Ysize, h, dugpu+2*Xsize*Ysize, hfgpu+2*Xsize*Ysize, lambda);

     /*
     if (k%2 == 0) {
        cudaMemcpy(err, errgpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
        float norm_err = norm(err,Xsize,Ysize);
        printf("Jacobi iters: %d, err: %f\n", k, norm_err);
      }
      */

    }
  }
  
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("smem GPU time = %fs\n", tt);
  printf("GPU Bandwidth %f GB/s \n", (4/BLOCK_DIM*N + 5*N + 3)*Xsize*Ysize*T*sizeof(float)/tt/1e9);
  
  cudaMemcpy(u0.A, u0smem, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
  
  // Write output
  write_image("tvl1-smem.ppm", u0);

  // Free memory
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  cudaStreamDestroy(streams[2]);
  cudaFree(u0gpu);
  cudaFree(u0smem);
  cudaFree(u1gpu);
  cudaFree(fgpu);
  cudaFree(dugpu);
  cudaFree(hfgpu);
  cudaFree(errgpu);
  free_image(&u0);
  free_image(&unoise);
  free(err);
  
  return 0;
}

