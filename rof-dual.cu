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

/*
ROF-dual kernel without using shared memory
*/
__global__ void rof(float* u, float* p0x, float* p1x, float* p0y, float* p1y, float* f, float* gradx, float* grady,  float lambda, float tau, long Xsize, long Ysize, float h, float* div){
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  // u = f + lambda div(p)
  if (idx < Xsize && idy < Ysize) {
    u[idx*Ysize + idy] = f[idx*Ysize + idy] + lambda*div[idx*Ysize+idy];
  }
  __syncthreads();
  // gradient of u
  if (idx < Xsize-1 && idy < Ysize-1) {
    gradx[idx*Ysize+idy] = (u[(idx+1)*Ysize + idy] - u[idx*Ysize + idy]);
    grady[idx*Ysize+idy] = (u[idx*Ysize + idy+1] - u[idx*Ysize + idy]);  
  }
  // gradient descent step
  if (idx < Xsize && idy < Ysize) {
    float numx = p0x[idx*Ysize+idy] + (tau/lambda)*gradx[idx*Ysize+idy];
    float numy = p0y[idx*Ysize+idy] + (tau/lambda)*grady[idx*Ysize+idy];
    float norm = sqrt( numx*numx + numy*numy);
    p1x[idx*Ysize + idy] = numx/max(1.0,norm); 
    p1y[idx*Ysize+idy] = numy/max(1.0,norm);
  }
  __syncthreads();
  float ux;
  float uy;
  // derivatives of p for boundaries
  if ( idx == 0) {
    ux = p1x[idx*Ysize + idy];
  }  
  if (idx == Xsize -1 ) {
    ux = -p1x[(idx-1)*Ysize + idy];
  }
  if (idy == 0){
    uy = p1y[idx*Ysize + idy];
  }
  if (idy == Ysize-1){
    uy = -p1y[idx*Ysize + idy-1];
  }
  // derivatives of p for non-boundaries
  if (idx > 0 && idx < Xsize -1) {
    ux = p1x[idx*Ysize + idy] - p1x[(idx-1)*Ysize + idy];
  }
  if (idy > 0 && idy < Ysize -1) {
    uy = p1y[idx*Ysize + idy] - p1y[idx*Ysize + idy-1];
  }
  // div p
  if (idx < Xsize && idy < Ysize) {
    div[idx*Ysize + idy] = ux + uy;
    p0x[idx*Ysize + idy] = p1x[idx*Ysize + idy];
    p0y[idx*Ysize + idy] = p1y[idx*Ysize + idy];
  }
}

/*
ROF-dual with shared mem for gradx, grady
*/
__global__ void rof_gsmem(float* px, float* py, float* f, float lambda, float tau, long Xsize, long Ysize, float* div){
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  __shared__ float gradx[BLOCK_DIM+1][BLOCK_DIM+1]; // gradient of u in x direction
  __shared__ float grady[BLOCK_DIM+1][BLOCK_DIM+1]; // gradient of u in y direction
  __shared__ float u[BLOCK_DIM+2][BLOCK_DIM+2];
  __shared__ float pxsh[BLOCK_DIM+1][BLOCK_DIM+1]; // x coordinate of vector p
  __shared__ float pysh[BLOCK_DIM+1][BLOCK_DIM+1]; // y coordinate of vector p
  float numx, numy, norm;

  // u = f + lambda div p for ghost cells of shared mem block 
  if(blockIdx.x < Xsize/BLOCK_DIM && threadIdx.x == BLOCK_DIM -1){
    u[BLOCK_DIM+1][threadIdx.y+1] = f[(idx+1)*Ysize + idy] + lambda*div[(idx+1)*Ysize+idy];
  }
  if(blockIdx.y < Ysize/BLOCK_DIM && threadIdx.y == BLOCK_DIM -1){
    u[threadIdx.x+1][BLOCK_DIM+1] = f[idx*Ysize + idy + 1] + lambda*div[idx*Ysize+idy+1];;
  }
  if(blockIdx.x > 0 && threadIdx.x == 0) {
    u[0][threadIdx.y+1] = f[ (idx-1)*Ysize + idy] + lambda*div[(idx-1)*Ysize + idy];
  }
  if(blockIdx.y > 0 && threadIdx.y == 0){
    u[threadIdx.x+1][0] = f[idx*Ysize + idy-1] + lambda*div[idx*Ysize + idy-1];
  }

  // u = f + lambda div p
  if (idx < Xsize && idy < Ysize) {
    u[threadIdx.x+1][threadIdx.y+1] = f[idx*Ysize + idy] + lambda*div[idx*Ysize+idy];
  }
  __syncthreads();
  
  // one-sided difference gradient of u
  if (idx < Xsize-1 && idy < Ysize-1) {
    gradx[threadIdx.x+1][threadIdx.y+1] = u[threadIdx.x+2][threadIdx.y+1]-u[threadIdx.x+1][threadIdx.y+1];
    grady[threadIdx.x+1][threadIdx.y+1] = u[threadIdx.x+1][threadIdx.y+2]-u[threadIdx.x+1][threadIdx.y+1];  
  }

  // gradient values for ghost cells
  if (blockIdx.x>0 && threadIdx.x == 0){ 
    gradx[0][threadIdx.y+1] = u[1][threadIdx.y+1]-u[0][threadIdx.y+1];
    grady[0][threadIdx.y+1] = u[0][threadIdx.y+2]-u[0][threadIdx.y+1];
  }
  if (blockIdx.y > 0 && threadIdx.y == 0){
    gradx[threadIdx.x+1][0] = u[threadIdx.x+2][0] - u[threadIdx.x+1][0];
    grady[threadIdx.x+1][0] = u[threadIdx.x+1][1] - u[threadIdx.x+1][0];
  } 
  __syncthreads();

  // gradient descent step
  if (idx < Xsize && idy < Ysize) {
    numx = px[idx*Ysize+idy] + (tau/lambda)*gradx[threadIdx.x+1][threadIdx.y+1];
    numy = py[idx*Ysize+idy] + (tau/lambda)*grady[threadIdx.x+1][threadIdx.y+1];
    norm = sqrt( numx*numx + numy*numy);
    pxsh[threadIdx.x+1][threadIdx.y+1] = numx/max(1.0,norm);
    pysh[threadIdx.x+1][threadIdx.y+1] = numy/max(1.0,norm);
  }
  // gradient descent step for ghost cells 
  if(blockIdx.x > 0 && threadIdx.x == 0 ){
    numx = px[(idx-1)*Ysize+idy] + (tau/lambda)*gradx[0][threadIdx.y+1];
    numy = py[(idx-1)*Ysize+idy] + (tau/lambda)*grady[0][threadIdx.y+1];
    norm = sqrt( numx*numx + numy*numy);
    pxsh[0][threadIdx.y+1] = numx/max(1.0,norm);;
    pysh[0][threadIdx.y+1] = numy/max(1.0,norm);;
  }
  if(blockIdx.y > 0 && threadIdx.y == 0){
    numx = px[idx*Ysize+idy-1] + (tau/lambda)*gradx[threadIdx.x+1][0];
    numy = py[idx*Ysize+idy-1] + (tau/lambda)*grady[threadIdx.x+1][0];
    norm = sqrt( numx*numx + numy*numy);
    pxsh[threadIdx.x+1][0] = numx/max(1.0,norm);
    pysh[threadIdx.x+1][0] = numy/max(1.0,norm);
  }
  __syncthreads();
  float ux;
  float uy;
  // divergence values ux, uy (derivatives of p) for boundaries
  if ( idx == 0) {
    ux = pxsh[threadIdx.x+1][threadIdx.y+1]; //p1x[idx*Ysize + idy];
  }  
  if (idx == Xsize -1 ) {
    ux = -pxsh[threadIdx.x][threadIdx.y+1]; //-p1x[(idx-1)*Ysize + idy];
  }
  if (idy == 0){
    uy = pysh[threadIdx.x+1][threadIdx.y+1]; //p1y[idx*Ysize + idy];
  }
  if (idy == Ysize-1){
    uy = -pysh[threadIdx.x+1][threadIdx.y]; //-p1y[idx*Ysize + idy-1];
  }
  // div p at inner grid points
  if (idx > 0 && idx < Xsize -1) {
    ux = pxsh[threadIdx.x+1][threadIdx.y+1]-pxsh[threadIdx.x][threadIdx.y+1];// p1x[idx*Ysize + idy] - p1x[(idx-1)*Ysize + idy];
  }
  if (idy > 0 && idy < Ysize -1) {
    uy = pysh[threadIdx.x+1][threadIdx.y+1]-pysh[threadIdx.x+1][threadIdx.y]; //p1y[idx*Ysize + idy] - p1y[idx*Ysize + idy-1];
  }
  // divergence of p
  if (idx < Xsize && idy < Ysize) {
    div[idx*Ysize + idy] = ux + uy;
    px[idx*Ysize + idy] = pxsh[threadIdx.x+1][threadIdx.y+1]; //p1x[idx*Ysize + idy];
    py[idx*Ysize + idy] = pysh[threadIdx.x+1][threadIdx.y+1]; //p1y[idx*Ysize + idy];
  }
}

/*
u = f + \lambda div p
*/
__global__ void compute_u(float* u, float *f, float lambda, long Xsize, long Ysize, float* div){
  int idx = (blockIdx.x)*BLOCK_DIM + threadIdx.x;
  int idy = (blockIdx.y)*BLOCK_DIM + threadIdx.y;
  if (idx < Xsize && idy < Ysize) {
    u[idx*Ysize+idy] = f[idx*Ysize+idy] + lambda*div[idx*Ysize+idy];
  }
}

int main(int argc, char * argv[] ) {
  long T = 250; 
  float lambda = 100; 
  float mu = 0;
  float sigma = 50;
  float tau = 0.245;
  const char fname[] = "car.ppm";
  
  //sscanf(argv[1],"%d",&T);
  //sscanf(argv[2],"%d",&N);
  //sscanf(argv[2],"%f",&lambda);
  //sscanf(argv[2],"%f",&sigma);
   
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

  // write noisy image
  write_image("rof-noise.ppm",unoise);
 
  Timer t;
  // Allocate GPU memory
  float *ugpu, *fgpu, *p1xgpu, *p1ygpu, *p0xgpu, *p0ygpu, *gradx, *grady, *div;
  cudaMalloc(&ugpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&fgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&p1xgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&p1ygpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&p0xgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&p0ygpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&gradx, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&grady, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&div, 3*Xsize*Ysize*sizeof(float));

 
  cudaMemcpy(ugpu, unoise.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(fgpu, unoise.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);

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
    rof<<<gridDim,blockDim, 0, streams[0]>>>(ugpu+0*Xsize*Ysize, p0xgpu+0*Xsize*Ysize, p1xgpu+0*Xsize*Ysize, p0ygpu+0*Xsize*Ysize, p1ygpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, gradx+0*Xsize*Ysize, grady+0*Xsize*Ysize, lambda, tau, Xsize, Ysize, h, div+0*Xsize*Ysize);
    rof<<<gridDim,blockDim, 1, streams[1]>>>(ugpu+1*Xsize*Ysize, p0xgpu+1*Xsize*Ysize, p1xgpu+1*Xsize*Ysize, p0ygpu+1*Xsize*Ysize, p1ygpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, gradx+1*Xsize*Ysize, grady+1*Xsize*Ysize, lambda, tau, Xsize, Ysize, h, div+1*Xsize*Ysize);
    rof<<<gridDim,blockDim, 2, streams[2]>>>(ugpu+2*Xsize*Ysize, p0xgpu+2*Xsize*Ysize, p1xgpu+2*Xsize*Ysize, p0ygpu+2*Xsize*Ysize, p1ygpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, gradx+2*Xsize*Ysize, grady+2*Xsize*Ysize, lambda, tau, Xsize, Ysize, h, div+2*Xsize*Ysize);

  }

  cudaDeviceSynchronize();
  double tt = t.toc();
  printf("nsmem GPU time = %fs\n", tt);
  printf("GPU Bandwidth = %f GB/s\n", (24*Xsize*Ysize*T*sizeof(float)+3*Xsize*Ysize*sizeof(float))/tt/1e9);
  cudaMemcpy(u0.A, ugpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
 
  // Write output
  write_image("rof-nsmem.ppm", u0);
 
  cudaFree(gradx);
  cudaFree(grady);
  cudaFree(p1xgpu);
  cudaFree(p1ygpu);
  cudaFree(p0xgpu);
  cudaFree(p0ygpu);
  cudaFree(div);
  
  cudaMalloc(&p0xgpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&p0ygpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&div, 3*Xsize*Ysize*sizeof(float));

  cudaDeviceSynchronize();
  t.tic();

  for (long n = 0; n < T; n++) {
    rof_gsmem<<<gridDim,blockDim, 0, streams[0]>>>(p0xgpu+0*Xsize*Ysize, p0ygpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, lambda, tau, Xsize, Ysize, div+0*Xsize*Ysize);
    rof_gsmem<<<gridDim,blockDim, 1, streams[1]>>>(p0xgpu+1*Xsize*Ysize, p0ygpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, lambda, tau, Xsize, Ysize, div+1*Xsize*Ysize);
    rof_gsmem<<<gridDim,blockDim, 2, streams[2]>>>(p0xgpu+2*Xsize*Ysize, p0ygpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, lambda, tau, Xsize, Ysize, div+2*Xsize*Ysize);
  }

  compute_u<<<gridDim,blockDim, 0, streams[0]>>>(ugpu+0*Xsize*Ysize, fgpu+0*Xsize*Ysize, lambda, Xsize, Ysize, div+0*Xsize*Ysize);
  compute_u<<<gridDim,blockDim, 1, streams[1]>>>(ugpu+1*Xsize*Ysize, fgpu+1*Xsize*Ysize, lambda, Xsize, Ysize, div+1*Xsize*Ysize);
  compute_u<<<gridDim,blockDim, 2, streams[2]>>>(ugpu+2*Xsize*Ysize, fgpu+2*Xsize*Ysize, lambda, Xsize, Ysize, div+2*Xsize*Ysize);

  cudaDeviceSynchronize();
  tt = t.toc();
  printf("gsmem GPU time = %fs\n", tt);
  printf("GPU Bandwidth = %f GB/s\n", (7*Xsize*Ysize*T + 6*Xsize*Ysize*T/BLOCK_DIM+3*Xsize*Ysize)*sizeof(float)/tt/1e9);
  cudaMemcpy(u0.A, ugpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);

  // Write output
  write_image("rof-gsmem.ppm", u0);


  // Free memory
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  cudaStreamDestroy(streams[2]);
  cudaFree(ugpu);
  cudaFree(fgpu);
  cudaFree(div);
  cudaFree(p0xgpu);
  cudaFree(p0ygpu);
  free_image(&u0);
  free_image(&unoise);
  return 0;
}

