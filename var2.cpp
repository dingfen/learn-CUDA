#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h> 

#define N 8 

void vfill(float* v, int n)
{
    int i;
    for(i = 0; i < n; i++){
        v[i] = (float) rand() / RAND_MAX;
    }
}

void vprint(float* v, int n)
{
    int i;
    printf("v = \n");
    for(i = 0; i < n; i++){
        printf("%7.3f\n", v[i]);
    }
    printf("\n");
}
 
__global__ void psum(float* v)
{ 
    // Thread index.
    int t = threadIdx.x; 
    // Should be half the length of v.
    int n = blockDim.x; 

    while (n != 0) {
        if(t < n)
            v[t] += v[t + n];  
        __syncthreads();    
        n /= 2; 
    }
}

int main (void)
{ 
    float *v_h, *v_d;
    v_h = (float*) malloc(N * sizeof(*v_h)); 
    cudaMalloc ((float**) &v_d, N *sizeof(*v_d)); 
    vfill(v_h, N);
    vprint(v_h, N);
    cudaMemcpy( v_d, v_h, N * sizeof(float), cudaMemcpyHostToDevice );
    psum<<< 1, N/2 >>>(v_d);
    cudaMemcpy(v_h, v_d, sizeof(float), cudaMemcpyDeviceToHost );
    printf("Pairwise sum = %7.3f\n", v_h[0]);
    free(v_h);
    cudaFree(v_d);
}